import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import spacy
import re
import torch
import os

class data_loader():
    def __init__(self,dataframe ,dtype_list):
        self.dataframe = dataframe ; self.dtype_list = dtype_list
        self._init_emb_dict_([self.dataframe]) ; self._init_emb_insert_([self.dataframe]) ; print('shape' , self.dataframe.columns)
        self.dataframe = self._init_one_hot_([self.dataframe]) ; print('shape' , self.dataframe.columns)
        self._init_inout_emb_(self.emb_dict) ; print('Embeddings' ,'Input', self.in_emb_list ,'Output', self.out_emb_list)
        self.mlp_features += self.one_hot_vars
        self.emb_features = self.emb_int
        self.target = self.dtype_list[8]

    #Strings + Binary + Multiclass(1,3,4) less than equal to 12 are one hot ; Multiclass discontinuous(6) is embedding
    def _init_emb_dict_(self,dataframe):
        onehot_dict = {} ; onehot_len = {} ; emb_dict = {} ; emb_len = {} ; self.mlp_features = []
        for df in dataframe:
            for key in df.keys():
                for i in [5]:
                    if key in self.dtype_list[i]:
                        self.mlp_features += [key]
                for i in [1,3,4]:
                    if key in self.dtype_list[i]:
                        df[key + '_onehot'] = df[key]
                        onehot_dict[key + '_onehot'] = np.unique(df[key + '_onehot'])
                        onehot_len[key + '_onehot'] = len(np.unique(df[key + '_onehot']))
                        self.onehot_dict = onehot_dict ; self.onehot_len = onehot_len
                for i in [6]:
                    if key in self.dtype_list[i]:
                        df[key + '_emb'] = df[key]
                        emb_dict[key + '_emb'] = np.unique(df[key + '_emb'])
                        emb_len[key + '_emb'] = len(np.unique(df[key + '_emb']))
                        self.emb_dict = emb_dict ; self.emb_len = emb_len

    def _init_emb_insert_(self, dataframe):
        self.onehot = [] ; self.emb_int = []
        for df in dataframe:
            for key in self.onehot_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.onehot_dict.get(key)) , list(range(len(self.onehot_dict.get(key)))))
                    df['str_key'] = key
                    df[key] = df['str_key'].astype(str).str.cat(df[key].astype(str))
                    self.onehot += [key]
            for key in self.emb_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.emb_dict.get(key)) , list(range(len(self.emb_dict.get(key)))))
                    self.emb_int += [key]

    def _init_one_hot_(self,dataframes):
        one_hot_vars = []
        for df in dataframes:
            for key in self.onehot:
                print(key)
                one_hot = pd.get_dummies(df[key])
                df = df.join(one_hot)
                print('check the shape' , df.shape , df.head(2))
                one_hot_vars  += list(one_hot.columns.values) ; self.one_hot_vars = one_hot_vars ;
        return df

    def _init_inout_emb_(self,emb_dict):
        in_emb_list1 = [] ; in_emb_list2 = [] ; out_emb_list1 = [] ; out_emb_list2 = []; in_emb_cnt1 = 0 ; in_emb_cnt2 = 0
        self.in_emb_list = [] ; self.out_emb_list = []
        for key in emb_dict.keys():
            if len(emb_dict[key]) > 30:
                in_emb_list1.append(len(emb_dict[key]))
                in_emb_cnt1 += 1
                out_emb_list1 = [30] * in_emb_cnt1
            else:
                in_emb_list2.append(len(emb_dict[key]))
                in_emb_cnt2 += 1
                out_emb_list2 = [12] * in_emb_cnt2
        self.in_emb_list = in_emb_list1 + in_emb_list2 ; self.out_emb_list = out_emb_list1 + out_emb_list2

class data_loader_text():
    def __init__(self,dataframe , text_features , in_emb_list , out_emb_list , mlp_features , emb_features):
        self.dataframe = dataframe ; self.text_features = text_features
        self.stop_words = set(stopwords.words('english'))
        self.nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
        self.in_emb_list = in_emb_list ; self.out_emb_list = out_emb_list
        self.mlp_features = mlp_features ; self.emb_features = emb_features
        self.dataframe  = self._init_text_dict_([self.dataframe])

    def _init_indexer_(self,word_dict,s):
      return [word_dict[w.text.lower()] for w in self.nlp(s)]

    def _init_clean_text_(self,text):
        text = text.lower()
        text = re.sub(r'[\s]+', ' ', text) ; text = re.sub(r'https?:/\/\S+', ' ', text) ;text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
        text = text.split() ; text = " ".join([x for x in text if x not in self.stop_words])
        return text.strip()

    def _init_pad_data_(self, s):
        padded = np.zeros((self.maxlen,), dtype=np.int64)
        if len(s) > self.maxlen: padded[:] = s[:self.maxlen]
        else: padded[:len(s)] = s
        return padded

    def _init_text_dict_(self,dataframe):
        self.emb_text_data = [] ; self.in_text_emb_list = [] ; in_text_emb_cnt = 0 ; self.out_text_emb_list = []
        for df in dataframe:
            for key in df.keys():
                if key in self.text_features:
                    print(key)
                    df[key] = df[key].apply(lambda x: self._init_clean_text_(x))
                    print(df[[key]].head(2))
                    words = Counter()
                    for i in tqdm(df[key].values):
                        words.update(w.text.lower() for w in self.nlp(i))
                    if np.max(list(words.values())) < 200:
                        self.maxlen = np.max(list(words.values())) ; print('self.maxlen' , self.maxlen)
                    else:
                        self.maxlen = 200 ; print('self.maxlen' , self.maxlen)
                    print('counter' , words.most_common(50))
                    word_dict = {o:i for i,o in enumerate(words)}
                    print('word_dict' , word_dict , len(word_dict))
                    df[key + '_idx'] = df[key].apply(lambda x: self._init_indexer_(word_dict,x))
                    df[key + '_len'] = df[key + '_idx'].apply(len)
                    df[key + '_padded'] = df[key + '_idx'].apply(lambda x: self._init_pad_data_(x))
                    print(df[[key + '_padded' , key + '_len']].head(2))
                    emb_data = df[key + '_padded']
                    self.mlp_features += [key + '_len'] ; self.emb_features += []; self.emb_text_data.append(emb_data)
                    print('self.emb_text_data' , self.emb_text_data)
                    self.in_text_emb_list.append(len(word_dict)) ;print(self.in_text_emb_list); in_text_emb_cnt += 1 
        self.out_text_emb_list = [2] * in_text_emb_cnt
        self.mlp_features  = np.unique(self.mlp_features) ; self.emb_features = np.unique(self.emb_features)
        print(self.in_text_emb_list , self.out_text_emb_list)
        return df

class data_loader_pytorch():
    def __init__(self, dataframe , mlp_features , emb_features ,emb_text_data ,target):
        self.dataframe = dataframe ; self.mlp_features = mlp_features ; self.emb_features = emb_features ; self.target = target 
        self.emb_text_data = emb_text_data
        print('mlp',self.mlp_features, 'emb',self.emb_features ,'target', self.target)
        index = self._init_batch_index_([self.dataframe])
        self.target ,self.mlp , self.emb , self.emb_text_data = self._init_torch_data_(self.train)

    def _init_batch_index_(self,dataframe):
        for df in dataframe:
            self.train = df.sample(frac = 1) ; self.validation = df.loc[~df.index.isin(self.train.index)]
            index = self.train.index
        return index

    def _init_torch_data_(self , data):
        emb_text_data = []
        mlp_data = np.zeros((len(data) , len(self.mlp_features)) , float)
        emb_data = np.zeros((len(data) , len(self.emb_features)) , int)
        target_data = np.zeros(len(data) , int)
        self.mlp_id = 0 ; self.emb_id = 0
        for key in data.keys():
            if key == self.target:
                target_data[:] = data[key]
            if key in self.mlp_features:
                mlp_data[ :, self.mlp_id] = data[key]
                self.mlp_id += 1
            if key in self.emb_features:
                print('key' , key , data[[key]].head(2))
                emb_data[: , self.emb_id] = data[key]
                self.emb_id += 1
        for i , df in enumerate(self.emb_text_data):
            df = torch.LongTensor(df) ; print('print(type(xs))', print(type(df)))
            emb_text_data.append(df)
        print('tensors' , 'target', target_data ,'mlp_data' ,mlp_data ,'emb_data', emb_data, 'emb_text_data' , emb_text_data)
        print('tensor shape' , target_data.shape, mlp_data.shape, emb_data.shape , emb_text_data[1].shape)
        return torch.tensor(target_data) , torch.tensor(mlp_data, dtype=torch.float32) , torch.tensor(emb_data) , emb_text_data
