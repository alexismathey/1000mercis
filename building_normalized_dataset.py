import numpy as np
import pandas as pd
from utils.kfolds import KFolds, read_csv
from utils.feature_scaling import scaling, scaling_by_id
import time

t1 = time.time()

# loading dataframe
path = './data/dataset_augmented.csv'
delimiter = ';'
data_frame, all_headers = read_csv(path, delimiter, verbose=True)#, nrows=5000)

# selecting headers of interest for scaling
headers_to_scale = ['occurrences', 
	'lifetime', 
	'nb_days', 
	'nb_idtags', 
	'nb_idtags_site', 
	'nb_idtags_media', 
	#'click_rate', 
	'nb_purchases', 
	'last_time', 
	'nb_ips']

# normalizing
normalized_dataframe = scaling_by_id(data_frame, headers_to_scale, verbose=True)

# saving normalized dataset
print('saving dataframe ...', end=' ')
path_to_save = './data/dataset_augmented_scaled_by_id.csv'
normalized_dataframe.to_csv(path_to_save, sep=';', index=False)
print('csv saved.')

t2 = time.time()


print('it took ' + str(t2-t1))


