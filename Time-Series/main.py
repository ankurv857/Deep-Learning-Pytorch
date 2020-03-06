import pandas as pd
import numpy as np
import os
from config import config
from reader import data_read
from prepare import data_prep
from loader import data_load

if __name__ == '__main__':
	print('The journey begins! ML automation - TIME SERIES goes first')

	#Call the class data_read()

	data = data_read( config["directory"], config['dataframe_list'] ,config['test_list'] ,config['date_list'] , config['target_list'] ,
	 config['idx'], config['remove_list'])

	for i  in range(0,len(data.df_list)):
		print('data downloaded' , len(data.df_list) , data.df_list[i].shape)
	for i in range(0,len(data.dtype_list)):
		print('Data Types' , len(data.dtype_list) , data.dtype_list[i])

	#call the class data_prep()

	dataframe_list = data.df_list
	base_join = data.df_list[0] 
	join_list = [[data.df_list[1] , ['Store'] , ['left']]]
	
	prep = data_prep(dataframe_list, base_join , join_list)
	print('lets check the data prepared' , prep.dataframe.head(2) , prep.dataframe.shape)

	#call the class data_load()

	dataframe = prep.dataframe 
	target = config['target_list']  
	dtype_list = data.dtype_list  
	granularities = [[['Store'] , ['Store' , 'Date']]]  
	temporal_vars = ['Open','Promo',	'SchoolHoliday', 'Sales']  
	time = ['Date']
	to_embed = ['Store' , 'Date_week' , 'Date_day']
	int_str = ['Date_week' ,'Date_day' ]
	multivar_encodes = ['Date_week' , 'Date_day']
	univar_encodes = ['Store' , 'Date_week' , 'Date_day']
	interactions =  []


	load = data_load(dataframe , target , dtype_list , granularities , temporal_vars , time, to_embed , int_str , multivar_encodes ,univar_encodes , interactions )
	
	df = load.dataframe[(load.dataframe['Store'] == 1)]
	df.to_csv('df.csv')


