import pandas as pd
import numpy as np
import os
from datetime import date
from datetime import timedelta

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

class ad_hoc():
	def __init__(self,dir ,dataframe_list):
		self.dir = dir 
		df_list = self._init_read_(dataframe_list)
		self.prep_data(df_list[0], df_list[1])


	def _init_read_(self,dataframes):
		df_list = []
		for df in dataframes:
			data = pd.read_csv(os.path.join(self.dir ,df) , na_values = ' ', low_memory=False)
			df_list.append(data)
		return df_list

	def prep_data(self,order_prod, orders):
		orders = orders[(orders['user_id'] <= 100) & (orders['eval_set'] == 'prior')]
		orders['dt'] = date(2018, 1, 1)
		orders['dt'] =  pd.to_datetime(orders['dt'])
		orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0).astype(int)
		orders['days_since_prior_order'] =  orders.groupby('user_id')['days_since_prior_order'].cumsum()
		orders['date'] = orders['dt'] + pd.to_timedelta(orders['days_since_prior_order'], unit='d')
		orders['date'] = np.where(orders['date'] == np.nan, orders['dt'], orders['date'])
		orders = orders[['order_id', 'user_id', 'order_number','date']]
		unique_orders = np.unique(orders['order_id']).tolist()
		order_prod = order_prod[order_prod.order_id.isin(unique_orders)]
		order_prod = order_prod[['order_id', 'product_id', 'add_to_cart_order']]
		print(orders , order_prod)
		orders.to_csv(self.dir + '/orders1.csv', index = False)
		order_prod.to_csv(self.dir + '/orders_products.csv', index = False)