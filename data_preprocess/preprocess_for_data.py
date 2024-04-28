# -*- coding : utf-8 -*-
import re

from torch.utils import data
import torch

from transformers import BertTokenizer, BertPreTrainedModel, BertModel
from transformers import RobertaTokenizer, RobertaModel

from constant import hacred_ner_labels_constant, hacred_rel_labels_constant, hacred_rel_to_question, get_labelmap, \
    scierc_ner_labels_constant, scierc_rel_labels_constant, scierc_rel_to_question, \
    _2017t10_rel_labels_constant, _2017t10_ner_labels_constant, _2017t10_rel_to_question, \
    ace05_rel_labels_constant, ace05_ner_labels_constant, ace05_rel_to_question, \
    conll04_ner_labels_constant, conll04_rel_labels_constant, conll04_rel_to_question, \
    ADE_ner_labels_constant, ADE_rel_labels_constant, ADE_rel_to_question
# from data_config import ArgumentParser

import numpy as np
import random
import json
import logging
from tqdm import tqdm
import pickle
import os
import sys

# data_config = ArgumentParser()

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end+1] = 1
    return mask


class BFT_data(data.DataLoader):
    def __init__(self, data_config, data_json, data_type, save_path, step):
        self.step = step
        self.config = data_config
        self.data_type = data_type
        self.data_json = data_json

        self.pretrained_model = self.config["pretrained_model"]
        if self.pretrained_model != "roberta-base":
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model)
        else:
            self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model)

        self.task = self.config["task"]
        self.star = self.config["star"]
        if self.task == "hacred":
            ner_label2id, ner_id2label = get_labelmap(hacred_ner_labels_constant)
        elif self.task == "scierc":
            ner_label2id, ner_id2label = get_labelmap(scierc_ner_labels_constant)
        else:
            ner_label2id, ner_id2label = eval("get_labelmap("+self.task+"_ner_labels_constant)")
        if os.path.exists(save_path + self.task + "_preprocessed_" + self.data_type + "_data.pkl"):
            preprocessed_data = pickle.load(open(save_path + self.task + "_preprocessed_" + self.data_type +
                                                 "_data.pkl", "rb"))
            print("Load preprocessed data finish!")
        else:
            preprocessed_data = self.prepare_data(ner_label2id)
            pickle.dump(preprocessed_data, open(save_path + self.task + "_preprocessed_" + self.data_type
                                                + "_data.pkl", "wb"))
            print("Save preprocessed data finish!")

        if os.path.exists(save_path + self.task + "_all_" + self.data_type + "_samples.pkl"):
            self.all_samples = self.convert_all_data_to_input_form(preprocessed_data)
            self.all_samples = pickle.load(open(save_path + self.task + "_all_" + self.data_type +
                                                "_samples.pkl", "rb"))
            print("Load all data finish!")
        else:
            self.all_samples = self.convert_all_data_to_input_form(preprocessed_data)
            pickle.dump(self.all_samples, open(save_path + self.task + "_all_" + self.data_type +
                                               "_samples.pkl", "wb"))
            print("Save all data finish!")
        print("Prepare data finish!")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, item):
        return self.all_samples[item]

    def generate_question_with_template(self, r):
        question_tokens = []
        question_tokens.append(self.tokenizer.mask_token)

        if self.task == "hacred":
            rel_to_question = hacred_rel_to_question
        else:
            # rel_to_question = scierc_rel_to_question  # 将类别映射成固定问题的模板
            rel_to_question = eval(self.task + '_rel_to_question')

        question_tokens += self.tokenizer.tokenize(rel_to_question[r])
        question_tokens.append(self.tokenizer.mask_token)
        question_tokens.append(self.tokenizer.sep_token)
        question_token_ids = self.tokenizer.convert_tokens_to_ids(question_tokens)
        return question_token_ids

    def get_decoder_input(self, sample):
        each_relation = {}
        for items in sample["relations"].values():
            for item in items:
                relation = item[2]
                answer = [item[0], item[1]]
                if relation not in each_relation.keys():
                    each_relation[relation] = []
                each_relation[relation].append(answer)

        if self.task == "hacred":
            rel_labels_constant = hacred_rel_labels_constant
        else:
            # rel_labels_constant = scierc_rel_labels_constant
            rel_labels_constant = eval(self.task + '_rel_labels_constant')

        rel_label2id, rel_id2label = get_labelmap(rel_labels_constant)

        for r in rel_labels_constant:
            if r not in each_relation.keys():
                each_relation[r] = []

        if self.config["sorted_entity"]:
            for i in each_relation.items():
                each_relation[i[0]] = sorted(i[1], key=lambda x: 1000000 * x[0] + x[1])
        else:
            for i in each_relation.items():
                random.shuffle(i[1])

        if self.config["sorted_relation"]:
            each_relation = sorted(each_relation.items(), key=lambda x: x[0])
        else:
            each_relation = sorted(each_relation.items(), key=lambda x: x[0])
            random.shuffle(each_relation)

        relations_and_answers = each_relation

        relation_id_for_each_template = []
        questions_input = []
        answers_input = []
        mask_pos_input = []
        question_token_type_ids = []
        for relation, answers in relations_and_answers:
            answer_input = []
            question_input = self.generate_question_with_template(relation) * self.config["duplicate_questions"]
            question_input[-1] = self.tokenizer.convert_tokens_to_ids(".")

            if self.config["token_type_ids"]:
                question_token_type_id = self.get_token_type_ids(question_input)
            else:
                question_token_type_id = [0] * len(question_input)

            mask_pos = np.argwhere(np.array(question_input) ==
                                   self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token))
            mask_pos = mask_pos.squeeze().tolist()
            answer_input.extend(answers)
            if len(answer_input) > self.config["duplicate_questions"]:
                answer_input = answer_input[:self.config["duplicate_questions"]]
            for i in range(self.config["duplicate_questions"] - len(answers)):
                answer_input.append([-1, -1])

            relation_id_for_each_template.append([rel_label2id[relation]])
            questions_input.append(question_input)
            answers_input.extend([answer_input])
            mask_pos_input.append(mask_pos)
            question_token_type_ids.append(question_token_type_id)

        answers_input = np.array(answers_input)
        answers_input = answers_input.reshape(answers_input.shape[0], -1).tolist()
        # questions_input.shape = (relation_num, *)
        # answers_input.shape = (relation_num, 20)
        # mask_pos_input.shape = (relation_num, 20)
        return relation_id_for_each_template, questions_input, answers_input, mask_pos_input, question_token_type_ids

    def get_SPN_input(self, sample):
        if self.task == "hacred":
            rel_labels_constant = hacred_rel_labels_constant
        else:
            # rel_labels_constant = scierc_rel_labels_constant
            rel_labels_constant = eval(self.task + '_rel_labels_constant')

        rel_label2id, rel_id2label = get_labelmap(rel_labels_constant)
        relation_id = []
        head_entity_pos = []
        tail_entity_pos = []
        for items in sample["relations"].values():
            for item in items:
                relation_id.append(rel_label2id[item[2]])
                head_entity_pos.append(item[0])
                tail_entity_pos.append(item[1])
        return relation_id, head_entity_pos, tail_entity_pos

    def get_token_type_ids(self, question_input):
        token_type_ids = []
        flag = True
        for q_id in question_input:
            if flag:
                token_type_ids.append(0)
            else:
                token_type_ids.append(1)
            if q_id == self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token):
                flag = not flag
        return token_type_ids

    def convert_each_data_to_input_form(self, sample):
        id2start = []
        id2end = []
        tokenizer_tokens = []
        input_sample = []
        span_entity_masks = []

        tokenizer_tokens.append(self.tokenizer.cls_token)  # cls
        for word in sample["sentences"]:
            id2start.append(len(tokenizer_tokens))
            sub_tokens = self.tokenizer.tokenize(word)
            tokenizer_tokens += sub_tokens
            id2end.append(len(tokenizer_tokens)-1)
        # tokenizer_tokens.append(self.tokenizer.sep_token)

        input_id = self.tokenizer.convert_tokens_to_ids(tokenizer_tokens)
        input_spans = [[id2start[span[0]], id2end[span[1]], span[2]] for span in sample["spans"]]
        # for item in input_spans:
        #     span_entity_masks.append(create_entity_mask(item[0], item[1], len(input_id)))
        # span_entity_masks = torch.stack(span_entity_masks)
        relation_id, head_entity_pos, tail_entity_pos = self.get_SPN_input(sample)
        relation_id_for_each_template, input_decoder_questions, input_decoder_answers, input_decoder_mask_pos, \
        input_question_token_type_ids = self.get_decoder_input(sample)

        negative_probability = None
        if self.step == "1":
            negative_probability = self.config["negative_probability"]
        if self.step == "2":
            negative_probability = 0

        for r_id, q, a, m, t in zip(relation_id_for_each_template, input_decoder_questions, input_decoder_answers,
                                    input_decoder_mask_pos, input_question_token_type_ids):
            p = random.random()
            if sum(a) == -len(a) and p < negative_probability:
                input_each_sample = {}
                input_each_sample["question"] = torch.tensor(q)     # list
                input_each_sample["template_relation_id"] = torch.tensor(r_id)
                input_each_sample["answer"] = torch.tensor(a)
                input_each_sample["decoder_mask_pos"] = torch.tensor(m)
                input_each_sample["question_token_type_id"] = torch.tensor(t)

                input_each_sample["id"] = torch.tensor(input_id)
                input_each_sample["spans"] = torch.tensor(input_spans)
                input_each_sample["span_labels"] = torch.tensor(sample["spans_label"])
                input_each_sample["doc_key"] = sample["doc_key"]

                input_each_sample["relation_id"] = torch.tensor(relation_id)
                input_each_sample["head_entity_pos"] = torch.tensor(head_entity_pos)
                input_each_sample["tail_entity_pos"] = torch.tensor(tail_entity_pos)
                input_each_sample["span_entity_masks"] = span_entity_masks
                input_sample.append(input_each_sample)
            elif sum(a) != -len(a):  # 有内容的部分
                input_each_sample = {}
                input_each_sample["question"] = torch.tensor(q)  # list  问题
                input_each_sample["template_relation_id"] = torch.tensor(r_id)  #
                input_each_sample["answer"] = torch.tensor(a)  # 头尾实体对
                input_each_sample["decoder_mask_pos"] = torch.tensor(m)  # 掩码位置
                input_each_sample["question_token_type_id"] = torch.tensor(t)  # 3组问题区分

                input_each_sample["id"] = torch.tensor(input_id)  # 原文输入内容
                input_each_sample["spans"] = torch.tensor(input_spans)  # span信息
                input_each_sample["span_labels"] = torch.tensor(sample["spans_label"])
                input_each_sample["doc_key"] = sample["doc_key"]

                input_each_sample["relation_id"] = torch.tensor(relation_id)  # 关系代码
                input_each_sample["head_entity_pos"] = torch.tensor(head_entity_pos)  # 头实体位置
                input_each_sample["tail_entity_pos"] = torch.tensor(tail_entity_pos)  # 尾实体位置
                input_each_sample["span_entity_masks"] = span_entity_masks
                input_sample.append(input_each_sample)
            else:
                pass
        if len(input_sample) == 0:
            rand_id = int(random.random() * len(input_decoder_questions))
            input_each_sample = {}
            input_each_sample["question"] = torch.tensor(input_decoder_questions[rand_id])  # list
            input_each_sample["template_relation_id"] = torch.tensor(relation_id_for_each_template[rand_id])
            input_each_sample["answer"] = torch.tensor(input_decoder_answers[rand_id])
            input_each_sample["decoder_mask_pos"] = torch.tensor(input_decoder_mask_pos[rand_id])
            input_each_sample["question_token_type_id"] = torch.tensor(input_question_token_type_ids[rand_id])

            input_each_sample["id"] = torch.tensor(input_id)
            input_each_sample["spans"] = torch.tensor(input_spans)
            input_each_sample["span_labels"] = torch.tensor(sample["spans_label"])
            input_each_sample["doc_key"] = sample["doc_key"]

            input_each_sample["relation_id"] = torch.tensor(relation_id)
            input_each_sample["head_entity_pos"] = torch.tensor(head_entity_pos)
            input_each_sample["tail_entity_pos"] = torch.tensor(tail_entity_pos)
            input_each_sample["span_entity_masks"] = span_entity_masks
            input_sample.append(input_each_sample)
        return input_sample

    def convert_all_data_to_input_form(self, preprocessed_data):
        all_samples = []
        for sample in tqdm(preprocessed_data):
            input_sample = self.convert_each_data_to_input_form(sample)
            all_samples.extend(input_sample)
        return all_samples

    def generate_samples(self, doc):
        keys_to_ignore = ["doc_key", "clusters"]
        keys = [key for key in doc.keys() if key not in keys_to_ignore]
        lengths = [len(doc[k]) for k in keys]
        assert len(set(lengths)) == 1
        length = lengths[0]
        res = [{k: doc[k][i] for k in keys} for i in range(length)]
        return res

    def prepare_data(self, ner_label2id):
        with open(self.data_json, 'r', encoding='utf8') as f:
            # lines = f.readlines()
            # documents = [eval(ele) for ele in lines]
            documents = json.load(f)
        max_span_num = 0
        span_num_count = 0
        final_samples = []
        for sentence_id, sample in tqdm(enumerate(documents)):
            final_sample = {}
            tokens = self.remove_accents(sample["sentText"]).split(" ")
            final_sample["doc_key"] = "sentence" + str(sentence_id)
            final_sample["sentences"] = tokens

            triples = sample["relationMentions"]
            final_sample["ner"] = {}
            for triple in triples:
                head_entity = self.remove_accents(triple["em1Text"]).split(" ")
                tail_entity = self.remove_accents(triple["em2Text"]).split(" ")
                if not self.config["is_chinese"]:
                    head_start, head_end = self.list_index(head_entity, tokens)
                    tail_start, tail_end = self.list_index(tail_entity, tokens)
                else:
                    head_start, head_end = triple['ent1_loc'][0], triple['ent1_loc'][1] - 1
                    tail_start, tail_end = triple['ent2_loc'][0], triple['ent2_loc'][1] - 1

                this_head_start = head_start
                this_head_end = head_end
                final_sample["ner"][(this_head_start, this_head_end)] = (tokens[this_head_start: this_head_end + 1],
                                                                         "entity")

                this_tail_start = tail_start
                this_tail_end = tail_end
                final_sample["ner"][(this_tail_start, this_tail_end)] = (tokens[this_tail_start: this_tail_end + 1],
                                                                         "entity")
                if this_tail_end - this_tail_start >= self.config["max_span_length"]:
                    print("\n" + "Note: there is one entity longer than the max_span_length: "
                          + str(self.config["max_span_length"]), ". The entity is ( "
                          + " ".join(tokens[this_tail_start: this_tail_end + 1]) + " ). Its length is "
                          + str(this_tail_end - this_tail_start) + " .")

            span2id = {}
            final_sample['spans'] = []
            final_sample['spans_label'] = []
            for i in range(len(tokens)):
                for j in range(i, min(len(tokens), i + self.config["max_span_length"])):  #
                    span_start = i
                    span_end = j
                    final_sample['spans'].append((span_start, span_end, j - i + 1))
                    span2id[(span_start, span_end)] = len(final_sample['spans']) - 1
                    if (span_start, span_end) not in final_sample["ner"].keys():
                        final_sample['spans_label'].append(0)
                    else:
                        label_name = final_sample["ner"][(span_start, span_end)][1]
                        final_sample['spans_label'].append(ner_label2id[label_name])

            max_span_num = max(len(final_sample['spans_label']), max_span_num)
            span_num_count = span_num_count + len(final_sample['spans_label'])
            final_sample["relations"] = {}

            for rel in triples:
                head_entity = self.remove_accents(rel["em1Text"]).split(" ")
                tail_entity = self.remove_accents(rel["em2Text"]).split(" ")
                if not self.config["is_chinese"]:
                    e1_start, e1_end = self.list_index(head_entity, tokens)
                    e2_start, e2_end = self.list_index(tail_entity, tokens)
                else:
                    e1_start, e1_end = rel['ent1_loc'][0], rel['ent1_loc'][1] - 1
                    e2_start, e2_end = rel['ent2_loc'][0], rel['ent2_loc'][1] - 1

                if e1_end - e1_start >= self.config["rel_span"] or e2_end - e2_start >= self.config["rel_span"]:
                    if e1_end - e1_start >= self.config["rel_span"]:
                        print("e1_end - e1_start is %d" % (e1_end - e1_start))
                    if e2_end - e2_start >= self.config["rel_span"]:
                        print("e2_end - e2_start is %d" % (e2_end - e2_start))
                else:
                    e1_span_id = span2id[(e1_start, e1_end)]
                    e2_span_id = span2id[(e2_start, e2_end)]
                    if ((e1_start, e1_end), (e2_start, e2_end)) not in final_sample["relations"].keys():
                        final_sample["relations"][((e1_start, e1_end), (e2_start, e2_end))] = []
                    final_sample["relations"][((e1_start, e1_end), (e2_start, e2_end))].append((e1_span_id,
                                                                                                e2_span_id,
                                                                                                rel["label"]))
            final_samples.append(final_sample)
        print("The max length of spans is ", max_span_num, ". The avg length of spans is ",
              int(span_num_count/len(documents)))
        return final_samples

    def remove_accents(self, text: str) -> str:
        accents_translation_table = str.maketrans(
            "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
            "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
        )
        return text.translate(accents_translation_table)

    def list_index(self, list1: list, list2: list) -> list:

        def find_all_locs(l1, l2, end_tag=False):
            records = []
            for i, x in enumerate(l2):
                if end_tag:
                    if x == l1[-1] or x.lower() == l1[-1].lower():
                        records.append(i)
                else:
                    if x == l1[0] or x.lower() == l1[0].lower():
                        records.append(i)
            return records

        index = (0, 0)
        start = find_all_locs(list1, list2, end_tag=False)
        end = find_all_locs(list1, list2, end_tag=True)
        if len(start) == 1 and len(end) == 1:
            return start[0], end[0]
        else:
            for i in start:
                for j in end:
                    if i <= j:
                        if list2[i:j + 1] == list1:
                            index = (i, j)
                            break
            if list1[0] != "Sudan":
                return index[0], index[1]
            else:
                return index[0], index[1]

    def list_index_for_chinese(self, list1: list, list2: list) -> list:
        index = (0, 0)
        if len(list1) > 1:
            temp = ''.join(list1)
            length_ = len(list1)
        else:
            temp = list1[0]
            length_ = len(list1[0]) - self.find_gap(list1[0])
        start = [i for i,x in enumerate(list2) if ''.join(list2[i:i+length_]) == temp]
        # end = [i for i,x in enumerate(list2) if ''.join(list2[i:i+len(list1[0])]) == list1[-1]]
        # if len(start) == 1 and len(end) == 1:
        return start[0], start[0] + length_
        # else:
        #     for i in start:
        #         for j in end:
        #             if i <= j:
        #                 if list2[i:j + 1] == list1:
        #                     index = (i, j)
        #                     break
    def find_gap(self, src):
        i = 0
        gap = 0
        while i < len(src):
            if re.match(r'[a-zA-Z0-9]', src[i]):
                j = i + 1
                while j < len(src) and re.match(r'[a-zA-Z0-9]', src[j]):
                    j += 1
                gap += j - i - 1
                i = j - 1
            i += 1
        return gap


if __name__ == '__main__':
    data_config = {
        "step": "1",
        "pretrained_model": 'bert_base_cased',  # scibert_cased for scierc; bert_base_cased for others
        "task": "conll04",  # nyt webnlg scierc _2017t10
        "is_chinese": 0,  # 0=No，1=yes
        "max_span_length": 8,  # 8 is normal(for others), 12 for scierc, 24 for 2017t10
        "rel_span": 8,  # 8 is normal(for others), 12 for scierc, 24 for 2017t10
        "star": 0,  # nyt* and WebNLG* need to change to 1
        "save_path": "./processed_data/",
        "log_path": "./processed_data/",
        "negative_probability": 0,
        "sorted_entity": 1,
        "sorted_relation": 0,
        "duplicate_questions": 3,
        "token_type_ids": 1,
    }
    logging.basicConfig(filename=os.path.join(data_config["log_path"], "info.log"),
                        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(os.path.join(data_config["log_path"], "data.log"), 'w'))
    logger.info(sys.argv)
    logger.info(data_config)
    if data_config["task"] == "hacred":
        if data_config["step"] == "1":
            train_data_json = "./data/hacred/train.json"
            test_data_json = "./data/hacred/test.json"
            dev_data_json = "./data/hacred/dev.json"
            print("-" * 10, "train", "-" * 10)
            save_path = data_config["save_path"]
            train_data_type = "train"
            train_data_BFT = BFT_data(data_config, train_data_json, train_data_type, save_path, step="1")

            print("-" * 10, "test", "-" * 10)
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="1")

            print("-" * 10, "dev", "-" * 10)
            dev_data_type = "dev"
            dev_data_BFT = BFT_data(data_config, dev_data_json, dev_data_type, save_path, step="1")
        if data_config["step"] == "2":
            save_path = "../pred_result/pred_data/"
            test_data_json = "../pred_result/generate_result/pred_json_for_BF.json"
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="2")
    elif data_config["task"] == "scierc":
        if data_config["step"] == "1":
            train_data_json = "./data/scierc/train.json"
            test_data_json = "./data/scierc/test.json"
            dev_data_json = "./data/scierc/dev.json"
            print("-" * 10, "train", "-" * 10)
            save_path = data_config["save_path"]
            train_data_type = "train"
            train_data_BFT = BFT_data(data_config, train_data_json, train_data_type, save_path, step="1")

            print("-" * 10, "test", "-" * 10)
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="1")

            print("-" * 10, "dev", "-" * 10)
            dev_data_type = "dev"
            dev_data_BFT = BFT_data(data_config, dev_data_json, dev_data_type, save_path, step="1")
        if data_config["step"] == "2":
            save_path = "../pred_result/pred_data/"
            test_data_json = "../pred_result/generate_result/pred_json_for_BF.json"
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="2")
    elif data_config["task"] == "_2017t10":
        if data_config["step"] == "1":
            train_data_json = "./data/_2017t10/train.json"
            test_data_json = "./data/_2017t10/test.json"
            dev_data_json = "./data/_2017t10/dev.json"
            print("-" * 10, "train", "-" * 10)
            save_path = data_config["save_path"]
            train_data_type = "train"
            train_data_BFT = BFT_data(data_config, train_data_json, train_data_type, save_path, step="1")

            print("-" * 10, "test", "-" * 10)
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="1")

            print("-" * 10, "dev", "-" * 10)
            dev_data_type = "dev"
            dev_data_BFT = BFT_data(data_config, dev_data_json, dev_data_type, save_path, step="1")
    else:
        if data_config["step"] == "1":
            train_data_json = "./data/"+data_config["task"]+"/train.json"
            test_data_json = "./data/"+data_config["task"]+"/test.json"
            dev_data_json = "./data/"+data_config["task"]+"/test.json"  # dev - test
            print("-" * 10, "train", "-" * 10)
            save_path = data_config["save_path"]
            train_data_type = "train"
            train_data_BFT = BFT_data(data_config, train_data_json, train_data_type, save_path, step="1")

            print("-" * 10, "test", "-" * 10)
            test_data_type = "test"
            test_data_BFT = BFT_data(data_config, test_data_json, test_data_type, save_path, step="1")

            print("-" * 10, "dev", "-" * 10)
            dev_data_type = "dev"
            dev_data_BFT = BFT_data(data_config, dev_data_json, dev_data_type, save_path, step="1")


