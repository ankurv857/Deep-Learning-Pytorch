import pandas as pd
import numpy as np
import os
import datetime 

class data_read():
    def __init__(self,dir ,dataframe_list , test_list ,date_list , target_list ,idx, remove_list):
        self.dir = dir ; self.date_list = date_list ; self.target_list = target_list ; self.target = target_list[0] ;self.idx = idx; self.remove_list = remove_list
        self.df_list = self._init_read_(dataframe_list)
        self._init_dtype_(self.df_list)
        self._init_number_dtype_split_(self.df_list)
        self._init_impute_vars_(self.df_list)
        self._init_number_dtype_split_(self.df_list)
        self.dtype_list = [self.date_list,self.strings,self.number,self.number_binary, self.number_multi ,self.number_continuous]

    def _init_read_(self,dataframes):
        df_list = []
        for df in dataframes:
            data = pd.read_csv(os.path.join(self.dir ,df) , na_values = ' ', low_memory=False)
            df_list.append(data)
        return df_list

    def _init_dtype_(self,dataframes):
    	strings = [] ; number = []
    	for df in dataframes:
    		strs = list(df.select_dtypes(include = [np.object])) ; strings += strs 
    		nbr = list(df.select_dtypes(exclude = [np.object])) ; number += nbr
    		for key in df.keys():
    			if key in self.date_list:
    				df[key] = pd.to_datetime(df[key])
    				df[key + '_year'] = df[key].dt.year
    				df[key + '_month'] = df[key].dt.month
    				df[key + '_week'] = df[key].dt.week
    				df[key + '_day'] = df[key].dt.day
    				df[key + '_weekday'] = df[key].dt.weekday
    				print('check the df' , df.head(2))
    				number += [key + '_year' , key + '_month', key + '_week', key + '_day', key + '_weekday' ]
    		self.strings = list(set(np.unique(strings)) - set((self.date_list + self.idx + self.remove_list))) 
    		self.number = list(set(np.unique(number)) - set((self.idx + self.remove_list)))

    def _init_number_dtype_split_(self,dataframes):
    	number_binary = [] ; number_multi = [] ; number_continuous = []
    	for df in dataframes:
    		for key in df.keys():
    			if (key in self.number and len(np.unique(df[key])) <=2):
    				number_binary += [key]
    			if (key in self.number and (len(np.unique(df[key])) > 2 and len(np.unique(df[key])) < 15)):
    				number_multi += [key]
    			if (key in self.number and len(np.unique(df[key])) >=15 ):
    				number_continuous += [key]
    		self.number_binary = np.unique(number_binary) ; self.number_multi = np.unique(number_multi) ;self.number_continuous = np.unique(number_continuous)

    def _init_impute_vars_(self,dataframes):
    	for df in dataframes:
    		for key in df.keys():
    			if key in self.strings:
    				df[key].fillna('Impute',inplace = True)
    			if key in self.number_binary:
    				df[key].fillna(0,inplace = True)
    			if key in self.number_multi:
    				df[key].fillna(0,inplace = True)    				
    			if key in self.number_continuous:
    				df[key].fillna(0 , inplace = True)
