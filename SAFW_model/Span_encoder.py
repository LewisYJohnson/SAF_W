# -*- coding : utf-8 -*-
import torch
from torch import nn

from allennlp.nn.util import batched_index_select
from allennlp.modules import FeedForward

from transformers import BertTokenizer, BertPreTrainedModel, BertModel, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder

from copy import deepcopy

from mtvr_models.multi_turn_vector_resampling_module import MTVR_for_Entity


class Span_Encoder(BertPreTrainedModel):
    def __init__(self, config, config_for_model, num_ner_label):
        super().__init__(config)
        self.config_for_model = config_for_model

        self.bert = BertModel(config)
        if config_for_model.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.hidden_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.width_embedding = nn.Embedding(config_for_model.max_span_length + 1, config_for_model.width_embedding_dim)
        self.ner_classifier = nn.Sequential(
            FeedForward(input_dim=config.hidden_size * 2 + config_for_model.width_embedding_dim,
                        num_layers=2,
                        hidden_dims=config_for_model.hidden_dim_for_NER,
                        activations=torch.nn.ReLU(),
                        dropout=0.2),
            nn.Linear(config_for_model.hidden_dim_for_NER, num_ner_label)
        )

        self.linear1_for_decoder = nn.Linear(config.hidden_size * 2 + config_for_model.width_embedding_dim,
                                             config.hidden_size)
        self.linear2_for_decoder = nn.Linear(config.hidden_size, config.hidden_size)

        # 补充内容
        self.linear_change_for_concat = nn.Linear(2 * config.hidden_size, config.hidden_size)
        decoder_config = deepcopy(config)
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        decoder_config.num_hidden_layers = 2  # 减半，正常是12
        self.decoder_bert = BertEncoder(decoder_config)
        self.mtvr_entity = MTVR_for_Entity(config.hidden_size,
                                           config.hidden_size,
                                           self.decoder_bert,
                                           length_info=5)

        # self.linear1_for_CLS = nn.Linear(config.hidden_size, config.hidden_size)
        # self.linear2_for_CLS = nn.Linear(config.hidden_size, config.hidden_size)

        self.ner_linear = nn.Linear(config.hidden_size, num_ner_label)

        self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)  # 相当于带着总长度的mask的均值化
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(self, batch_tokens, batch_tokens_attention_mask, batch_spans):
        batch_spans_embedding, sequence_output_of_encoder, pooled_output = \
            self.get_spans_embedding(batch_tokens, batch_tokens_attention_mask, batch_spans)

        FFNN_hidden = []
        hidden1 = batch_spans_embedding
        if self.config_for_model.NER_mode == "2":
            for layer_each in self.ner_classifier:
                hidden1 = layer_each(hidden1)
                FFNN_hidden.append(hidden1)
            logits_for_NER = FFNN_hidden[-1]

        hidden2 = batch_spans_embedding
        logits_for_RE_origin = self.linear2_for_decoder(torch.relu(self.linear1_for_decoder(hidden2)))

        # pooled_output_hidden = self.linear2_for_CLS(torch.relu(self.linear1_for_CLS(pooled_output)))
        # pooled_output.shape = [bz, config.hidden_size(768)]
        pooled_output = pooled_output.unsqueeze(1)
        logits_for_BF = torch.cat((pooled_output, logits_for_RE_origin), dim=1)  # 原文和所有实体信息（头、尾、宽度，合并后的映射）横向绑定在一起
        logits_for_RD = logits_for_RE_origin

        if self.config_for_model.NER_mode == "1":
            logits_for_NER = self.ner_linear(torch.relu(logits_for_RE_origin))

        if self.config_for_model.cross_attention_mode_for_BF == "2":
            sequence_output_of_encoder_for_BF = logits_for_BF
        else:
            sequence_output_of_encoder_for_BF = sequence_output_of_encoder

        if self.config_for_model.cross_attention_mode_for_RD == "2":
            sequence_output_of_encoder_for_RD = logits_for_RD
        else:
            sequence_output_of_encoder_for_RD = sequence_output_of_encoder
        # logits_for_NER.shape = [bz, span_len, num_ner_label]
        # logits_for_RE.shape = [bz, span_len + 1 ( [CLS] ), config.hidden_size(768)]
        # sequence_output_of_encoder.shape = [bz, seq_len, config.hidden_size] or [bz, span_len, config.hidden_size]
        return logits_for_NER, logits_for_BF, logits_for_RD, \
               sequence_output_of_encoder_for_BF, sequence_output_of_encoder_for_RD

    def get_spans_embedding(self, batch_tokens, batch_tokens_attention_mask, batch_spans):
        sequence_output, pooled_output = self.bert(input_ids=batch_tokens, attention_mask=batch_tokens_attention_mask,
                                                   return_dict=False)
        sequence_output_drop = self.hidden_dropout(sequence_output)

        # 增加指针过程
        selected_hidden_states, selected_one_hot = self.mtvr_entity(
            # x=h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1),
            x=sequence_output_drop,
            pooled=pooled_output,
            context_masks=batch_tokens_attention_mask,
            # entity_masks=entity_masks,
        )
        # selected_final_hidden_states = selected_hidden_states.max(dim=1)[0]
        selected_final_hidden_states = self.masked_avgpool(selected_hidden_states, batch_tokens_attention_mask)
        # final hidden state的整合
        pooled_output = self.linear_change_for_concat(torch.cat([pooled_output, selected_final_hidden_states], -1))
        sequence_output_drop = self.linear_change_for_concat(
            torch.cat([sequence_output_drop, selected_hidden_states], -1))  # all hidden state的整合

        batch_spans_start = batch_spans[:, :, 0].view(batch_spans.size(0), -1)
        batch_spans_start_embedding = batched_index_select(sequence_output_drop, batch_spans_start)

        batch_spans_end = batch_spans[:, :, 1].view(batch_spans.size(0), -1)
        batch_spans_end_embedding = batched_index_select(sequence_output_drop, batch_spans_end)

        batch_width = batch_spans[:, :, 2].view(batch_spans.size(0), -1)
        batch_width_embedding = self.width_embedding(batch_width)

        batch_spans_embedding = torch.cat((batch_spans_start_embedding, batch_spans_end_embedding,
                                           batch_width_embedding), dim=-1)
        return batch_spans_embedding, sequence_output, pooled_output




