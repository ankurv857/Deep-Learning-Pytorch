import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os


class Embedding_Layer(nn.Module):
    def __init__(self,in_size , out_size):
        super().__init__()
        self.emb =  nn.ModuleList([])
        for in_s , out_s in zip(in_size , out_size):
            self.emb.append(nn.Embedding(in_s , out_s))
            print('embedding layer' , self.emb)

    def forward(self , x):
        emb_list = []
        print('x.size()[-1]', x.size())
        for i in range(x.size()[-1]):
            print('check the embeddings' , i , self.emb[i] , x[:,i])
            emb_list.append(self.emb[i](x[:,i]))
            #print('emb_list' , emb_list , len(x.size()) - 1 ,'concatenated', torch.cat(emb_list, dim=len(x.size()) - 1))
        return torch.cat(emb_list, dim=len(x.size()) - 1)

class Neural_Network(nn.Module):
    def __init__(self,in_emb_list , out_emb_list ,in_text_emb_list , out_text_emb_list, inp_mlp_dim , out_mlp_dim):
        super().__init__()
        #for i ,in_text_emb in enumerate(in_text_emb_list):
            #print('YOLO',i , in_text_emb_list[i] , out_text_emb_list[i])
        self.text_embedding = nn.Embedding(in_text_emb_list[0] , out_text_emb_list[0]) ; print('self.text_embedding' , self.text_embedding )
        self.embedding = Embedding_Layer(in_emb_list , out_emb_list)
        print('check the output layer of embeddings' ,out_text_emb_list[0] , out_emb_list , inp_mlp_dim)
        # self.perceptron= torch.nn.Linear(inp_mlp_dim + sum(out_emb_list) + out_text_emb_list[0] , out_mlp_dim)
        self.perceptron= torch.nn.Linear(491 , out_mlp_dim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, embed , mlp , emb_text_data):
        #for i , data in enumerate(emb_text_data):
        print('emb_text_data' , emb_text_data[0])
        # text_emb = self.text_embedding(emb_text_data[0].size(0),-1)
        text_emb = self.text_embedding(emb_text_data[0])
        text_emb = text_emb.view(text_emb.size()[0], text_emb.size()[1]*text_emb.size()[2] )
        print('text_emb', text_emb, 'emb_text_data[0]', emb_text_data[0].shape ,'text_emb.shape',text_emb.shape)
        emb = self.embedding(embed)
        print('text_emb' , text_emb,'mlp & emb' , mlp , emb , text_emb.size(),mlp.size() , emb.size())
        # mlp_emb_concat = torch.cat((mlp.float(),emb.float()),text_emb.float(), dim=1)
        emb_concat = torch.cat((text_emb, emb), dim=1) 
        print('mlp',mlp, mlp.shape,'emb_concat', emb_concat, emb_concat.shape)
        mlp_emb_concat = torch.cat((mlp, emb_concat), dim=1)
        mlp_emb_concat = mlp_emb_concat.view(mlp_emb_concat.size(0), -1)
        #print('mlp_emb_concat' , mlp_emb_concat , mlp_emb_concat.size()) ; print('sum(out_emb_list)', sum(self.out_emb_list))
        mlp_net = self.sigmoid(self.perceptron(mlp_emb_concat))
        preds = self.sigmoid(mlp_net)
        return preds
