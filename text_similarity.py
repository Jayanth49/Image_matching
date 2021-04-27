# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:53:38 2021

@author: Jayanth
"""
import torch

from dataset import train_test_val
from learner import Train
from similarity import bert_similarity


print("Getting Data....")
bert_tokenizer,model =  bert_similarity('D:\GitHub\Image_matching/')
# train_dataset, _,_ = train_test_val(bert_tokenizer, path='D:\GitHub\Image_matching\custom_data.csv')

# print("Training.....")
# Train(model,train_dataset,batch_size = 5)
print(bert_tokenizer)