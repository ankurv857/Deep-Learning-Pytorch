import pandas as pd
import numpy as np
import os

class data_prep():
	def __init__(self,dataframe_list, base_join, join_list):
		self.df_list = dataframe_list ; self.base_join = base_join ; self.join_list = join_list
		self.dataframe =  self._init_joins_()

	def _init_joins_(self):
		for df_sub_list in self.join_list:
			print('df_sub_list',df_sub_list[1:])
			self.base_join = pd.merge(self.base_join , df_sub_list[0] , on = df_sub_list[1] , how = df_sub_list[2][0])
			print('look at the join' , self.base_join.head(2) , self.base_join.shape)
		return self.base_join
