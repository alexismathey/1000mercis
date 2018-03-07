import numpy as np
import pandas as pd


class KFolds:
	"""
	Class to manage KFolds cross-validation.
	Separates the dataset putting together lines linked to the same `id` value
	"""
	def __init__(self, dataframe, nb_folds, seed=12345):
		self.dataframe = dataframe
		self.nb_folds = nb_folds
		self.id_per_fold = []
		for k in range(self.nb_folds):
			self.id_per_fold.append([])
		self.assign_folds(seed)

	def assign_folds(self, seed):
		id_values = set(self.dataframe['id'])
		nb_id_values = len(id_values)
	
		# assigning a fold to each user_id
		np.random.seed(seed)
		max_id_value = max(id_values)
		fold_per_id = np.random.randint(self.nb_folds, size = max_id_value)

		# filling in the list of all user_id for a given fold
		for id_val in id_values:
			self.id_per_fold[fold_per_id[id_val-1]].append(id_val)

	def get_fold(self, fold):
		assert fold >= 0 and fold < self.nb_folds
		id_values = self.id_per_fold[fold]
		train_set = self.dataframe[~self.dataframe['id'].isin(id_values)]
		test_set = self.dataframe[self.dataframe['id'].isin(id_values)]
		return train_set, test_set


def read_csv(filepath, delimiter, verbose=True, nrows=None):
	"""
		load the csv file using pandas library
		returns: the dataframe, a list of the headers names
	"""
	if verbose:
		print('reading csv...', end=' ')
	data_frame = pd.read_csv(filepath, delimiter=delimiter, nrows=nrows)
	if verbose:
		print('Done.')
	all_headers = data_frame.columns.tolist()
	return data_frame, all_headers