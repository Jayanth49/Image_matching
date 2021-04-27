# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:30:49 2021

@author: Jayanth
"""


import torch
import pandas as pd
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self,df,bert_tokenizer):
        has_similarity = True
        
        # if df['similarity'] 
        
        
        data = []
        max_bert_input_length = 0
        
        for _,row in df.iterrows():
            item = {}
            sentence_1_tokenized, sentence_2_tokenized = bert_tokenizer.tokenize(row['sentence_1']),bert_tokenizer.tokenize(row['sentence_2'])
            
            item['sentence_1_tokenized'] = sentence_1_tokenized
            item['sentence_2_tokenized'] = sentence_2_tokenized
            
            max_bert_input_length = max(max_bert_input_length,len(sentence_1_tokenized)+len(sentence_2_tokenized)+3)
            
            if has_similarity:
                item['similarity'] = float(row['similarity'])
                
            data.append(item)
        
        self.data = data
        self.max_bert_input_length = max_bert_input_length
        self.bert_tokenizer = bert_tokenizer
        
    
    def __getitem__(self,index):
        
        data = self.data[index]
        
        tokens = []
        input_type_ids = []
        
        tokens.append("[CLS]")
        input_type_ids.append(0)
        
        for token in data['sentence_1_tokenized']:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(1)
        
        for token in data['sentence_2_tokenized']:
            tokens.append(token)
            input_type_ids.append(1)
        tokens.append("[SEP]")
        input_type_ids.append(1)
        
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(tokens)
        
        attention_masks = [1]*len(input_ids)
        
        while len(input_ids) < self.max_bert_input_length:
            input_ids.append(0)
            attention_masks.append(0)
            input_type_ids.append(0)
            
        dataset_input_ids = torch.tensor(input_ids, dtype=torch.long)
        dataset_token_type_ids = torch.tensor(input_type_ids, dtype=torch.long)
        dataset_attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        dataset_scores = torch.tensor(data['similarity'], dtype=torch.float)
        
        return dataset_input_ids, dataset_token_type_ids, dataset_attention_masks, dataset_scores

    def __len__(self):
        return len(self.data)
    
        
def train_test_val(bert_tokenizer,path='custom_data.csv'):
    df = pd.read_csv(path)
    train_data,val_data = train_test_split(df,shuffle=True,test_size = 0.2)
    train_data,test_data = train_test_split(train_data,shuffle=True,test_size = 0.3)
    
    return CustomDataset(train_data,bert_tokenizer),CustomDataset(val_data,bert_tokenizer),CustomDataset(test_data,bert_tokenizer)


