# SAFW
# Title

This is the program for the paper "Explore the Role of Self-Adaptive Feature Words in Relation Quintuple Extraction for Scientific Literature"

## Install

```
torch >= 1.6.0
tqdm
transformers >= 4.26.0
allennlp
```

## Usage

The data is in the "data_preprocess" category

First, change the data to binary data by run the following. Note that all the parameter can be changed in lines 471~485.
```
python data_preprocess/preprocess_for_data.py
```
Second, move the generated pkl files from the "processed_data" category into the sub path with corresponding names.

Third, start training with running the following.

```
python SAWF_main.py
--RE_loss_for_RD_parameter
10
--learning_rate_in_RD
3e-5
--pretrained_model
bert_base_cased
--learning_rate_for_RE_decoder_in_BF
2e-4
--task
scierc
--batch_size
32
--star
0
--epoch
250
--max_span_length
8
--model_save_path
save_model_new
```
The result will be printed into the training_result.txt in the model_save_path

## Reference

We reference to the previous work of RFBFN (https://github.com/lizhe2016/RFBFN) and make deep improvements in feature word selection. Thus, So we re-used some of their code and project structure, and modified and added to it as appropriate. 
The codes in multi_turn_vector_resampling_module.py is our original innovations.
