def compute_prediction_old(dataset, Y_score):
	"""
		Wrong way of computing prediction (bad running time)
	"""
	id_values = sorted(list(set(dataset['id'].values)))
	Y_predicted = np.zeros(len(Y_score))
	for id in id_values:
		id_indexes = dataset.index[dataset['id'] == id].tolist()
		id_index_min = min(id_indexes)
		id_index_max = max(id_indexes)
		# indice où le max est atteint dans le sous-vecteur considéré
		index_temp = np.where(Y_score[id_index_min:id_index_max+1] == max(Y_score[id_index_min:id_index_max+1]))[0][0]
		# indice où ce max est atteint dans le test set
		index_max = id_indexes[index_temp] 
		Y_predicted[index_max] = 1
	return Y_predicted


def compute_prediction(dataset, Y_score, verbose=False):
	n = len(Y_score)
	Y_predicted = np.zeros((n,), dtype=np.int)
	current_id = dataset.get_value(0, 'id')
	highest_proba = Y_score[0]
	best_index = 0
	for row in range(1, n):
		if verbose:
			print(round(float(row)/n, 2), end='   \r')
		if current_id != dataset.get_value(row, 'id'):
			Y_predicted[best_index] = 1
			current_id = dataset.get_value(row, 'id')
			best_index = row
			highest_proba = Y_score[row]
		else:
			if Y_score[row] > highest_proba:
				best_index = row
				highest_proba = Y_score[row]
	return Y_predicted


def compute_error(Y_test, Y_predicted):
	"""
		counts the number of missranked predictions over all user ids
	"""
	missranked = 0
	wellranked = 0
	total = 0
	for i in range(len(Y_test)):
		if Y_test[i] == 1:
		    total += 1
		    if Y_predicted[i] == 1:
		        wellranked += 1
		    else:
		        missranked += 1
	return missranked, wellranked, total