import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset , DataLoader
from config import config
from reader import data_read
from prepare import data_prep, ad_hoc
from loader import data_loader , data_loader_text , data_loader_pytorch
from model import Embedding_Layer , Neural_Network

if __name__ == '__main__':
    print('The journey begins! ML automation with Pytorch')

    #Call the class data_read()
    data = data_read(config["directory"], config['dataframe_list'] ,config['date_list'] , config['target_list'] ,
    config['idx'], config['multiclass_discontinuous'],config['text'] , config['remove_list'])

    # #call the class data_prep()
    dataframe_list = data.df_list
    base_join = data.df_list[0]
    join_list = [[data.df_list[1] , ['campaign_id'] , ['left']]]

    prep = data_prep(dataframe_list, base_join , join_list)
    print('lets check the data prepared' , prep.dataframe.head(2) , prep.dataframe.shape)

    # #call the classes from loader data_loader() , data_loader_pytorch()
    load = data_loader(prep.dataframe , data.dtype_list)
    load_text = data_loader_text(load.dataframe , config['text'] , load.in_emb_list , load.out_emb_list , load.mlp_features , load.emb_features)
    load_torch = data_loader_pytorch(load_text.dataframe , load_text.mlp_features , load_text.emb_features,load_text.emb_text_data , load.target)

    # #call the class Neural_Network()
    model = Neural_Network(load.in_emb_list , load.out_emb_list ,load_text.in_text_emb_list,load_text.out_text_emb_list ,inp_mlp_dim = 38, out_mlp_dim = 1)
    criterion = torch.nn.BCELoss(size_average = True)
    optimizer = torch.optim.SGD(model.parameters() , lr = 0.1)


    for epoch in range(10):
        target , mlp , embed , emb_text_data = load_torch.target ,load_torch.mlp , load_torch.emb , load_torch.emb_text_data
        pred = model(embed , mlp , emb_text_data)
        print('pred & target' , pred)
        loss = criterion(pred.float() , target.float()) ; print(epoch , loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
