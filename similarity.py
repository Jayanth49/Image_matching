# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:35:37 2021

@author: Jayanth
"""
import os
from pytorch_transformers import BertTokenizer
from pytorch_transformers.modeling_utils import CONFIG_NAME
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel

import torch
from torch import nn


class BertSimilarityRegressor(BertPreTrainedModel):
    def __init__(self, bert_model_config: BertConfig):
        super(BertSimilarityRegressor, self).__init__(bert_model_config)
        self.bert = BertModel(bert_model_config)
        linear_size = bert_model_config.hidden_size

        self.regression = nn.Sequential(nn.Dropout(p=bert_model_config.hidden_dropout_prob), nn.Linear(linear_size, 1))

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_masks):
        """
        Feed forward network with one hidden layer.
        :param input_ids:
        :param token_type_ids:
        :param attention_masks:
        :return:
        """
        _, pooled_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_masks)

        return self.regression(pooled_output)
    
    
def bert_similarity(bert_model_path):
    bert_tokenizer = BertTokenizer.from_pretrained('D:\GitHub\Image_matching')
    config =  BertConfig.from_json_file('D:\GitHub\Image_matching\config.json')
    # if os.path.exists(bert_model_path):
    #     if os.path.exists(os.path.join(bert_model_path, CONFIG_NAME)):
    #         config = BertConfig.from_json_file(os.path.join(bert_model_path, CONFIG_NAME))
    #     elif os.path.exists('D:\GitHub\Image_matching\config.json'):
    #         config = BertConfig.from_json_file('config.json')
    #     else:
    #         raise ValueError("Cannot find a configuration for the BERT model you are attempting to load.")
    # print(bert_tokenizer)
    regressor_net = BertSimilarityRegressor.from_pretrained('D:\GitHub\Image_matching/', config=config)
    return bert_tokenizer, regressor_net


