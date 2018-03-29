import numpy as np
import pandas as pd


def compute_prediction(dataset, Y_score, verbose=False):
	"""
		Computes the rank prediction from a vector of scores. For a given `id` value, assigns 1 to the line achieving the highest score
	"""
	n = len(Y_score)
	Y_predicted = np.zeros((n,), dtype=np.int)
	current_id = dataset.get_value(0, 'id')
	highest_proba = Y_score[0]
	best_index = 0
	for row in range(1, n):
		if verbose:
			print('(compute prediction) ', round(float(row)/n, 2), end='   \r')
		if current_id != dataset.get_value(row, 'id'):
			Y_predicted[best_index] = 1
			current_id = dataset.get_value(row, 'id')
			best_index = row
			highest_proba = Y_score[row]
		else:
			if Y_score[row] > highest_proba:
				best_index = row
				highest_proba = Y_score[row]
		if row == n-1:
			Y_predicted[best_index] = 1
	if verbose:
		print(30*' ', end='\r')
	return Y_predicted


def compute_error(Y_test, Y_predicted):
	"""
		counts the number of missranked predictions over all id values
	"""
	missranked = 0
	wellranked = 0
	total = 0
	for i in range(len(Y_test)):
		if Y_predicted[i] == 1:
		    total += 1
		    if Y_test[i] == 1:
		        wellranked += 1
		    else:
		        missranked += 1
	return missranked, wellranked, total