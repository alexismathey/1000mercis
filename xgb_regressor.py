import math
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.kfolds import KFolds, read_csv
from utils.prediction import compute_prediction, compute_error
from utils.feature_scaling import scaling, scaling_by_id
from sklearn.ensemble import GradientBoostingRegressor
import argparse


def main(path, delimiter, verbose):
	# loading the dataframe
	data_frame, all_headers = read_csv(path, delimiter, verbose)

	# selecting headers of interest
	headers = ['id',
		#'hash_email', 
		#'hash_email_conversion', 
		#'hash_userid', 
		'rank', 
		'occurrences', 
		'lifetime', 
		'nb_days', 
		'nb_idtags', 
		'nb_idtags_site', 
		'nb_idtags_media', 
		#'click_rate', 
		'nb_purchases', 
		'last_time', 
		'nb_ips']
	headers_to_drop = list(set(all_headers) - set(headers))
	headers_to_scale = headers[:]
	headers_to_scale.remove('id')
	headers_to_scale.remove('rank')

	# K-Fold cross-validation
	nb_folds = 10
	fold = KFolds(data_frame, nb_folds)
	missranked_scores_train = []
	missranked_scores_test = []

	for k in range(nb_folds):
		train, test = fold.get_fold(k)
		train = train.sort_values(by='id')
		test = test.sort_values(by='id')

		# dropping not usefull columns
		for drop in headers_to_drop:
		    train = train.drop(drop, 1)
		    test = test.drop(drop, 1)

		# train set
		#train, mean, std = scaling(train, headers_to_scale)
		train = train.reset_index(drop=True)
		X = train[headers_to_scale].values
		Y = train['rank']==1

		# training model
		xgb_reg = xgb_reg = GradientBoostingRegressor(criterion='mse', max_depth=8, n_estimators=40, verbose=verbose)
		xgb_reg.fit(X, Y)

		# computing score on train set
		Y_score_train = xgb_reg.predict(X)
		Y_predicted_train = compute_prediction(train, Y_score_train, verbose)
		missranked_train, wellranked_train, total_train = compute_error(Y, Y_predicted_train)
		missranked_scores_train.append(missranked_train/total_train)

		# test set
		#test = scaling(test, headers_to_scale, mean, std)
		test = test.reset_index(drop=True)
		X_test = test[headers_to_scale].values
		Y_test = test['rank'].values==1

		# computing score on test set
		Y_score_test = xgb_reg.predict(X_test)
		Y_predicted_test = compute_prediction(test, Y_score_test, verbose)
		missranked_test, wellranked_test, total_test = compute_error(Y_test, Y_predicted_test)
		missranked_scores_test.append(missranked_test/total_test)

		# printing intermediate results
		if verbose:
			print('\n**** fold ', k, '****')
			print('train set:')
			print('   missranked =', round(missranked_train/total_train, 3))
			print('   wellranked =', round(wellranked_train/total_train, 3))
			print('test set:')
			print('   missranked =', round(missranked_test/total_test, 3))
			print('   wellranked =', round(wellranked_test/total_test, 3))

	# printing final result
	if verbose:
		print('\n******** MEAN over all folds ********')
		print('Train missranked = ', np.mean(missranked_scores_train))
		print(' Test missranked = ', np.mean(missranked_scores_test))
		print('  Train accuracy = ', 1 - np.mean(missranked_scores_train))
		print('   Test accuracy = ', 1 - np.mean(missranked_scores_test))


if __name__ == '__main__':
    # Parse the different arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to the dataset', required=True)
    parser.add_argument('--delimiter', help='delimiter used in the dataset', required=True)
    parser.add_argument('--verbose', help='increase output verbosity', action='store_true', required=False)
	
    # Recover the arguments
    opts = parser.parse_args()

    # Execute the main function
    main(opts.dataset, opts.delimiter, opts.verbose)