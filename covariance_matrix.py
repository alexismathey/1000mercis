import numpy as np
import pandas as pd
from utils.kfolds import KFolds, read_csv

# loading the dataframe
path = './data/dataset_augmented.csv'
delimiter = ';'

data_frame, all_headers = read_csv(path, delimiter, verbose=True)

# selecting headers of interest
headers = [#'id',
	#'hash_email', 
	#'hash_email_conversion', 
	#'hash_userid', 
	#'rank', 
	'occurrences', 
	'lifetime', # trop biasé ?
	'nb_days', 
	'nb_idtags', 
	'nb_idtags_site', 
	'nb_idtags_media', 
	#'click_rate', 
	'nb_purchases', 
	'last_time', 
	'nb_ips']

# computing the co-variance matrix
values = data_frame[headers].values
values = (values - np.mean(values, axis=0)) / np.std(values, axis=0)
n = values.shape[0]
var = 1/n * np.dot(values.T, values)

print('\nheaders = ', headers)
print('var = \n', np.round(var, 2))

# printing highly correlated features

print('\nhighly correlated features are:')
threshold = 0.6
for i in range(len(headers)):
	for j in range(i+1, len(headers)):
		if abs(var[i, j]) > threshold:
			print('cor(' + str(headers[i]) + ', ' + str(headers[j]) + ') = ', var[i, j])