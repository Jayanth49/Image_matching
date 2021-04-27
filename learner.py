# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 11:42:22 2021

@author: Jayanth
"""
import os

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from processing import sentence_pair_processing
from similarity import bert_similarity
# from download import get_model_path



# class TextSimilarityLearner():
#     def __init__(self,
#                  batch_size=10,
#                  model_name='web-bert-similarity',
#                  loss_func=torch.nn.MSELoss(),
#                  learning_rate=5e-5,
#                  weight_decay=0,
#                  device=torch.device('cpu')):
#         self.batch_size = batch_size
#         self.loss_func = loss_func
#         self.from_pretrain(model_name)
#         self.to(device)
#         self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=weight_decay, lr=learning_rate)

#     def load_train_data(self, train_data):
#         self.train_data = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)

#     def to(self, device):
#         self.device = device
#         self.model = self.model.to(device)
#         return self

#     def from_pretrain(self, model_name):
#         # bert_model_path = get_model_path(model_name)
#         bert_model_path = ''
#         self.bert_tokenizer, self.model = bert_similarity(bert_model_path)
#         return self
    
#     # def get(self,model=True):
#     #     if model:
#     #         return self.bert_tokenizer
#     #     else:
#     #         return self.model

#     def save_model(self, path='web-bert-similarity.bin'):
#         torch.save(self.model.state_dict(), path)
#         return self

#     def load_model(self, path='web-bert-similarity.bin'):
#         parameters = torch.load(path)
#         self.model.load_state_dict(parameters)
#         return self

#     def train_one_epoch(self):
#         device = self.device
#         train_data = self.train_data

#         total = len(train_data)
#         epoch_loss = .0

#         pbar = tqdm(total=total)

#         self.model = self.model.to(device)
#         self.model.train()
#         for i, (dataset_input_ids, dataset_token_type_ids, dataset_attention_masks,
#                 dataset_scores) in enumerate(train_data):
#             dataset_input_ids = dataset_input_ids.to(device)
#             dataset_token_type_ids = dataset_token_type_ids.to(device)
#             dataset_attention_masks = dataset_attention_masks.to(device)
#             dataset_scores = dataset_scores.to(device)

#             self.optimizer.zero_grad()
#             outputs = self.model(dataset_input_ids, dataset_token_type_ids, dataset_attention_masks).squeeze(-1)
#             step_loss = self.loss_func(outputs, dataset_scores)
#             step_loss.backward()
#             self.optimizer.step()
#             epoch_loss += step_loss.item()

                
#             pbar.set_postfix(step_loss=step_loss)

#         pbar.close()

#         return {'loss': epoch_loss / total}

#     def train(self, epoch=10):
#         for i in range(epoch):
#             matric = self.train_one_epoch()
#             log_info = 'Epoch {}: '.format(i + 1)
#             for key, value in matric.items():
#                 log_info += key + '=' + str(value) + ' '
#             print(log_info)

#     def predict(self, data: list):
#         self.model.to(self.device)
#         self.model.eval()

#         with torch.no_grad():

#             if data is not None and isinstance(data, list):
#                 if isinstance(data[0], dict):
#                     input_ids_eval, token_type_ids_eval, attention_masks_eval, correct_scores_eval = sentence_pair_processing(
#                         data, self.bert_tokenizer)
#                 elif isinstance(data[0], tuple):
#                     input_ids_eval, token_type_ids_eval, attention_masks_eval, correct_scores_eval = sentence_pair_processing(
#                         [{
#                             'sentence_1': s1,
#                             'sentence_2': s2
#                         } for s1, s2 in data], self.bert_tokenizer)
#                 else:
#                     raise ValueError('Data must be a list of sentence pair tuples')

#             predictions = torch.empty_like(correct_scores_eval)
#             for i in range(0, input_ids_eval.shape[0], self.batch_size):
#                 input_id_eval = input_ids_eval[i:i + self.batch_size].to(device=self.device)
#                 token_type_id_eval = token_type_ids_eval[i:i + self.batch_size].to(device=self.device)
#                 attention_mask_eval = attention_masks_eval[i:i + self.batch_size].to(device=self.device)

#                 predicted_score = self.model(input_id_eval,
#                                              token_type_id_eval,
#                                              attention_mask_eval)
#                 predictions[i:i + self.batch_size] = predicted_score

#         return predictions.cpu().view(-1).numpy()
    
# print(learner.bert_tokenizer)
        
# from torch.utils.data import DataLoader

def Train(model,train_data,batch_size = 10):
    
    loss_fn = torch.nn.MSELoss()
    learning_rate = 5e-5
    weight_decay = 0
    epochs = 1
    device = torch.device('cuda:0')
    
    train_data = DataLoader(train_data,batch_size=batch_size,shuffle=False)
    
    total = len(train_data)
    print("size:",total)
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=weight_decay, lr=learning_rate)

    model.train()

    for i in range(epochs):
        epoch_loss = .0
        
        pbar = tqdm(total=total)
        
        for i, (dataset_input_ids, dataset_token_type_ids, dataset_attention_masks,
                dataset_scores) in enumerate(train_data):
            dataset_input_ids = dataset_input_ids.to(device)
            dataset_token_type_ids = dataset_token_type_ids.to(device)
            dataset_attention_masks = dataset_attention_masks.to(device)
            dataset_scores = dataset_scores.to(device)
            
            optimizer.zero_grad()
        
            outputs = model(dataset_input_ids, dataset_token_type_ids, dataset_attention_masks).squeeze(-1)
            step_loss = loss_fn(outputs, dataset_scores)
            step_loss.backward()
            optimizer.step()
            epoch_loss += step_loss.item()
            
            pbar.update(1)
            pbar.set_postfix(step_loss=step_loss)
            
        pbar.close()
        
        matric =  {'loss': epoch_loss / total}
        log_info = 'Epoch {}: '.format(i + 1)
        for key, value in matric.items():
                log_info += key + '=' + str(value) + ' '
        print(log_info)
    