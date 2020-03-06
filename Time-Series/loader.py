import pandas as pd
import numpy as np
import os
import datetime
from datetime import time , date

class data_load():
	def __init__(self,dataframe , target , dtype_list , granularities , temporal_vars , time , to_embed , int_str ,multivar_encodes , univar_encodes , interactions ):
		self.dataframe = dataframe ; self.target = target ; self.dtype_list = dtype_list ; self.granularities = granularities ; self.temporal_vars = temporal_vars ;
		self.time = time ; self.to_embed = to_embed ; self.int_str = int_str ; self.multivar_encodes = multivar_encodes ; self.univar_encodes = univar_encodes
		self.interactions = interactions

		self._init_str_emb_dict_([self.dataframe ])  ; print('print the embed dict' , self.str_emb_dict , self.str_emb_len)
		self._init_str_replace_num_([self.dataframe ]) ; print(self.str_int)
		self.dataframe = self._init_temporal_daysince_([self.dataframe ]) ; print(self.dataframe.head(2) , self.temporal_daysince)
		self.dataframe = self._init_temporal_vars_lag_([self.dataframe ]) ; print(self.dataframe.head(2) , self.temporal_lags) 
		self._init_count_dim_([self.dataframe]) ; print('embedding size' , self.emb_num_int , self.nonemb_num_int)
		self.dataframe = self._init_change_vars_([self.dataframe]) ; self.dataframe = self._init_one_hot_([self.dataframe])
		self._init_multivar_([self.dataframe]) ; self._init_univar_([self.dataframe]) ; self._init_encode_dict_([self.dataframe])  
		self._init_enc_insert_([self.dataframe]) ; self._init_float_interactions_([self.dataframe])

    #Create embedding dictionary for all string features and insert them into data

	def _init_str_emb_dict_(self,dataframe):
		str_emb_dict = {} ; str_emb_len = {}
		for df in dataframe:
			for key in df.keys():
				if key in self.dtype_list[1]:
					df[key + '_emb'] = df[key]
					str_emb_dict[key + '_emb'] = np.unique(df[key + '_emb'])
					str_emb_len[key + '_emb'] = len(np.unique(df[key + '_emb']))
		self.str_emb_dict = str_emb_dict ; self.str_emb_len = str_emb_len

	def _init_str_replace_num_(self, dataframe):
		self.str_int = []
		for df in dataframe:
			for key in self.str_emb_dict.keys():
				if key in df.keys():
					df[key] = df[key].replace(list(self.str_emb_dict.get(key)) , list(range(len(self.str_emb_dict.get(key)))))
					self.str_int += [key]

	#Create all the temporal features

	def _init_temporal_daysince_(self,dataframe):
		self.temporal_daysince = []
		for df in dataframe:
			for gran_list in self.granularities:
				print('granlist' , gran_list , gran_list[0] , gran_list[1])
				df = df.sort_values(gran_list[1] , ascending = True) ; print('sorted' , df.head(10))
				df[gran_list[0][0] + '_seq'] = df.groupby(gran_list[0])[self.time].cumcount() + 1
				print('datatype of date' , df['Date'].dtype)
				df[gran_list[0][0] + '_gapsince'] = df.groupby(gran_list[0])[self.time].diff()
				print('check the data' , df.head(2))
				for key in self.temporal_vars:
					print('temporal_daysince key' , key)
					df[gran_list[0][0] + key + '_daysince'] = df[gran_list[0]].groupby((df[key] != df[key].shift()).cumsum()).cumcount() + 1
					df[gran_list[0][0] + key + '_daysince'] = np.where(df[key]  != 0, 0, df[gran_list[0][0] + key + '_daysince'])
					self.temporal_daysince += [gran_list[0][0] + '_seq' , gran_list[0][0] + key + '_daysince']
		return df

	def _init_temporal_vars_lag_(self,dataframe):
		self.temporal_lags = []
		for df in dataframe:
			print(self.granularities)
			for gran_list in self.granularities:
				print('granlist' , gran_list)
				df = df.sort_values(gran_list[1], ascending = True)
				for key in self.temporal_vars:
					print('key' , key)
					df[gran_list[0][0] + key + '_roll4days'] = df.groupby(gran_list[0])[key].shift(1).rolling(4).mean()
					df[gran_list[0][0] + key + '_roll28days'] = df.groupby(gran_list[0])[key].shift(1).rolling(28).mean()
					df[gran_list[0][0] + key + '_roll90days'] = df.groupby(gran_list[0])[key].shift(1).rolling(90).mean()
					self.temporal_lags += [gran_list[0][0] + key + '_roll4' , gran_list[0][0] + key + '_roll13' , gran_list[0][0] + key + '_roll52']
		return df

	#One hot encoding for the required features
		
	def _init_change_vars_(self,dataframes):
		self.one_hot = []
		for df in dataframes:
			for key in df.keys():
				if key in self.int_str:
					df['str_key'] = key
					df[key+'_string'] = df[key]
					df[key+'_string'] = df['str_key'].astype(str).str.cat(df[key+'_string'].astype(str))
					self.one_hot += [key+'_string']
		return df

	def _init_one_hot_(self,dataframes):
		one_hot_vars = []
		for df in dataframes:
			for key in self.one_hot:
				print(key)
				one_hot = pd.get_dummies(df[key])
				df = df.join(one_hot)
				print('check the shape' , df.shape , df.head(2))
				one_hot_vars  += list(one_hot.columns.values) ; self.one_hot_vars = one_hot_vars ;
		return df

	#multivariate as well as univariate target encoding 
	def _init_multivar_(self,dataframes):
		multivar = []
		for df in dataframes:
			for i , key1 in enumerate(self.multivar_encodes):
				if (i+1 < len(self.multivar_encodes)):					
					for key2 in self.multivar_encodes[i+1:]:
						df[key1 +'_' + key2] = df[key1].astype(str).str.cat(df[key2].astype(str))
						print('df[key1key2]' , df[key1 +'_' + key2].head(10))
						multivar += [key1 +'_' + key2]
		self.multivar = multivar ; print( 'multivar' ,self.multivar)

	def _init_univar_(self,dataframes):
		encoding =[]
		for df in dataframes:
			for key in df.keys():
				if key in self.univar_encodes:
					df[key+'_enc'] = df[key]
					encoding += [key+'_enc']
		self.encoding = encoding ; self.encoding += self.multivar ; 
		print( 'encoding' ,self.encoding)

	def _init_encode_dict_(self,dataframes):
		encoded_feature_list = []
		emb_dict1 = {}
		encode_dict = {}
		for df in dataframes:
			for key in df.keys():
				if key in self.encoding:
					emb_dict1[key] = np.unique(df[key]) ; print('emb_dict1' ,emb_dict1[key].shape)
					encode = df.groupby([key])[self.target].mean().reset_index()
					encode_dict[key] = encode[self.target].values
					encoded_feature_list += [key]
		self.emb_dict1 = emb_dict1 ; self.encode_dict = encode_dict
		self.encoded_feature_list = encoded_feature_list 
		print( 'emb_dict1' ,self.emb_dict1) ; print( 'encode_dict' ,self.encode_dict)

	def _init_enc_insert_(self,dataframes):
		self.interactions = []
		for df in dataframes:
			for key in self.emb_dict1.keys():
				if key in df.keys():
					print( 'lets see where ?',key , df[key].shape , df[key].head(10))
					df[key] = df[key].replace(list(self.emb_dict1.get(key)) , list(self.encode_dict.get(key)))
					self.interactions += [key]

	#Interactions and Normalization of the features
	def _init_float_interactions_(self,dataframes):
		interaction_feature_list = []
		for df in dataframes:
			for i, key1 in enumerate(self.interactions):
				if(i+1 < len(self.interactions)):
					for key2 in self.interactions[i+1:]:
						df[key1 + '_' + key2 + '_interaction'] = df[key1]/df[key2]
						df[key1 + '_' + key2 + '_interaction'].replace(np.inf, 0, inplace=True)
						df[key1 + '_normalized'] = df[key1]/np.mean(df[key1])
						df[key1 + '_' + key2 + '_norm_mult'] = (df[key1]/np.mean(df[key1]))*(df[key2]/np.mean(df[key2]))
						interaction_feature_list += [key1 + '' + key2 + '_interaction', key1 + '_normalized', key1 + '' + key2 + '_norm_mult']
		self.interaction_feature_list = interaction_feature_list


	# def _init_sample_(self,data):
	# 	train = data[(data['wm_yr_wk_id'] < 11848)]
	# 	validation = data.loc[~data.index.isin(train.index)]
	# 	return train , validation
    
	# def _init_rolling_target_(self,train, validation):
	# 	data = train[(train['sales'] >= 0)]
	# 	data['rank_wk_desc'] = data.groupby(['acctg_dept_nbr'])['visit_date'].rank(ascending=False).astype(int)
	# 	data = data.sort_values(['acctg_dept_nbr','wm_yr_wk_id'],ascending=True)
	# 	data['roll_target1'] = data.groupby(['acctg_dept_nbr'])['sales'].shift(1).rolling(365).mean()
	# 	data['roll_target2'] = data.groupby(['acctg_dept_nbr'])['sales'].shift(1).rolling(180).mean()
	# 	data['roll_target3'] = data.groupby(['acctg_dept_nbr'])['sales'].shift(1).rolling(90).mean()
	# 	data['roll_target4'] = data.groupby(['acctg_dept_nbr'])['sales'].mean()
	# 	data['offset'] = np.nan
	# 	data['offset'] = data.offset.fillna(data.roll_target1).fillna(data.roll_target2).fillna(data.roll_target3).fillna(data.roll_target4)
	# 	val_offset = data[(data['rank_wk_desc'] == 1)] ; val_offset = val_offset[['acctg_dept_nbr' , 'offset']]
	# 	validation = pd.merge(validation , val_offset , on = 'acctg_dept_nbr' , how = 'left')
	# 	data = data[(data['rank_wk_desc'] > 90)]
	# 	return data , validation

	# def _init_remove_features_(self,dataframes):
	# 	for df in dataframes:
	# 		for key in self.xgb_features_exceptions:
	# 			df[key] = 0

	#data preparation for Embedding input layer for both int and real

	def _init_count_dim_(self, dataframe):
		self.emb_num_int , self.nonemb_num_int = 0,0
		for df in dataframe:
			for key in df.keys():
				if key in self.to_embed:
					self.emb_num_int += 1
				else:
					self.nonemb_num_int += 1
