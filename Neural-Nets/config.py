import os

config = {}

#For the class data_read()
config['directory'] = '/Users/ankur/Documents/Auto ML/Pytorch ML/Logistic_Reg/Data/AV_Click'
config['dataframe_list'] = ['train.csv' , 'campaign_data.csv']
config['date_list'] = ['send_date']
config['target_list'] = ['is_click']
config['idx'] = ['id']
config['multiclass_discontinuous'] = ['campaign_id' , 'user_id' , 'email_url']
config['text'] = ['subject' , 'email_body']
config['remove_list'] = []
