import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.kfolds import KFolds, read_csv
from utils.prediction import compute_prediction, compute_error
from utils.feature_scaling import scaling, scaling_by_id
from sklearn.ensemble import GradientBoostingClassifier
import argparse



#	path = '/Users/Alexis/Documents/Centrale/3A/PROJET_MILLE_MERCIS/MAIN/data/dataset_augmented.csv'


def main(train_path, test_path):

	delimiter = ';'
	verbose = True
	data_frame, all_headers = read_csv(train_path, delimiter, verbose)


	delimiter_test = ','
	verbose = True
	data_frame_test, all_headers_test = read_csv(test_path, delimiter_test, verbose)


	headers = ['id','rank',	'occurrences', 'lifetime', 'nb_days', 'nb_idtags', 'nb_idtags_site', 'nb_idtags_media', 'nb_purchases', 'last_time', 'nb_ips']

	headers_test = ['id', 'similarity', 'occurrences', 'lifetime', 'nb_days', 'nb_idtags', 'nb_idtags_site', 'nb_idtags_media', 'nb_purchases', 'last_time', 'nb_ips']

	headers_to_drop = list(set(all_headers) - set(headers))
	headers_to_drop_test = list(set(all_headers_test) - set(headers_test))


	headers_to_scale = headers[:]
	headers_to_scale.remove('id')
	headers_to_scale.remove('rank')

	headers_to_scale_test = headers_test[:]
	headers_to_scale_test.remove('id')
	headers_to_scale_test.remove('similarity')

	train = data_frame.copy()
	train = train.sort_values(by='id')

	test = data_frame_test.copy()
	test = test.sort_values(by='id')


	for drop in headers_to_drop:
	    train = train.drop(drop, 1)

	for drop in headers_to_drop_test:
	    test = test.drop(drop, 1)


	train = train.reset_index(drop=True)
	X = train[headers_to_scale].values
	Y = train['rank']==1





	similarity =  np.array(test.loc[:,'similarity'])
	baseline = np.zeros(similarity.shape)

	old_id = test.loc[0,'id']
	rows = []

	for row in range(test.shape[0]):
	    current_id = test.loc[row,'id']
	    if current_id == old_id:
	        rows.append(row)
	    else:
	        sample = np.random.randint(min(rows), max(rows)+1)
	        baseline[sample]=1
	        old_id = current_id
	        m = test.loc[rows, 'similarity'].max()
	        test.loc[rows, 'similarity'] = 2 - (test.loc[rows, 'similarity'] == m)
	        rows = []
	        rows.append(row)
	    
	    if row == test.shape[0]:
	        sample = np.random.randint(min(rows), max(rows)+1)
	        baseline[sample]=1
	        old_id = current_id
	        m = test.loc[rows, 'similarity'].max()
	        test.loc[rows, 'similarity'] = 2 - (test.loc[rows, 'similarity'] == m)
	        rows = []
	        rows.append(row)




	test = test.reset_index(drop=True)
	X_test = test[headers_to_scale].values
	Y_test = test['similarity']






	# training model
	regularization = 1e10
	xgb_reg = GradientBoostingClassifier(loss='exponential', n_estimators=50, criterion='friedman_mse', max_depth=5, verbose=verbose)
	xgb_reg.fit(X, Y)

	# computing score on train set
	Y_score_train = xgb_reg.predict_proba(X)[:,1]
	Y_predicted_train = compute_prediction(train, Y_score_train)#, verbose)
	missranked_train, wellranked_train, total_train = compute_error(Y, Y_predicted_train)


	# printing intermediate results
	print('train set:')
	print('   missranked =', round(missranked_train/total_train, 3))
	print('   wellranked =', round(wellranked_train/total_train, 3))





	# 
	# computing score on test set
	Y_score_test = xgb_reg.predict_proba(X_test)[:,1]
	Y_predicted_test = compute_prediction(test, Y_score_test)#, verbose)
	missranked_test, wellranked_test, total_test = compute_error(Y_test, Y_predicted_test)
	missranked_test_baseline, wellranked_test_baseline, total_test_baseline = compute_error(Y_test, baseline)


	# printing intermediate results
	print('test set:')
	print('   missranked =', round(missranked_test/total_test, 3))
	print('   wellranked =', round(wellranked_test/total_test, 3))
	print('baseline prediction:')
	print('   missranked =', round(missranked_test_baseline/total_test_baseline, 3))
	print('   wellranked =', round(wellranked_test_baseline/total_test_baseline, 3))






	# Les deux métriques du pdf


	df_2 = data_frame_test.copy()

	old_id = df_2.loc[0, 'id']
	rows = []
	score_1 = 0
	score_2 = 0
	score_1_baseline = 0
	score_2_baseline = 0
	N = 0

	for row in range(len(Y_predicted_test)):
	    current_id = df_2.loc[row,'id']

	    if current_id == old_id:
	    	rows.append(row)
	    	if Y_predicted_test[row] == 1:
	    		prediction = row
	    		N += 1
	    	if baseline[row] == 1:
	    		prediction_baseline = row
	    else:
	        old_id = current_id
	        m = df_2.loc[rows, 'similarity'].max()    
	        score_1 += m - df_2.loc[prediction, 'similarity']
	        score_2 += m*df_2.loc[prediction, 'similarity']
	        score_1_baseline += m - df_2.loc[prediction_baseline, 'similarity']
	        score_2_baseline += m*df_2.loc[prediction_baseline, 'similarity']        
	        rows = []
	        rows.append(row)
	        if Y_predicted_test[row] == 1:
	        	prediction = row
	        	N += 1
	        if baseline[row] == 1:
	        	prediction_baseline = row

	    if row == len(Y_predicted_test):
	        old_id = current_id
	        m = df_2.loc[rows, 'similarity'].max()    
	        score_1 += m - df_2.loc[prediction, 'similarity']
	        score_2 += m*df_2.loc[prediction, 'similarity']
	        score_1_baseline += m - df_2.loc[prediction_baseline, 'similarity']
	        score_2_baseline += m*df_2.loc[prediction_baseline, 'similarity']        
	        rows = []
	        rows.append(row)
	        


	score_1 /= N
	score_2 /= N
	score_1_baseline /= N
	score_2_baseline /= N





	print('score_1 = ' + str(score_1))
	print('score_2 = ' + str(score_2))
	print('score_1_baseline = ' + str(score_1_baseline))
	print('score_2_baseline = ' + str(score_2_baseline))




if __name__ == '__main__':


	parser = argparse.ArgumentParser()
	parser.add_argument('--trainset', help='path to the dataset', required=True)
	parser.add_argument('--testset', help='path to the dataset', required=True)
	
	opts = parser.parse_args()

	main(opts.trainset, opts.testset)







