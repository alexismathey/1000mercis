import pandas as pd
import time

def scaling(dataset, headers, mean=None, std=None):
	"""
		Performs feature scaling:
			- either computing mean and std, and returning it for later use ('None' case)
			- or using values given as argument ('not None' case)
	"""
	trainset_scaling = (mean is None) and (std is None)
	if trainset_scaling:
		mean = {}
		std = {}
	for header in headers:
		if trainset_scaling:
			mean[header] = dataset[header].mean()
			std[header] = dataset[header].std()
		dataset[header] = (dataset[header] - mean[header]) / std[header]
	if trainset_scaling:
		return dataset, mean, std
	else:
		return dataset


def scaling_by_id(dataset, headers, verbose=False):
	"""
		Performs feature scaling id by id
	"""

	if verbose:
		print('starting scaling by id ...', end=' ')

	dataset = dataset.sort_values(by='id').reset_index(drop=True)
	mean = dataset.groupby('id')[headers].mean()
	std = dataset.groupby('id')[headers].std()

	# to avoid division by 0
	std = std.replace(0, 1)	
	dataset.loc[:,headers] = dataset\
		.apply(lambda row: (row[headers] - mean.loc[row['id'], headers]) / std.loc[row['id'], headers], axis=1)

	if verbose:
		print('done.')

	return dataset


"""
def scaling_by_id(dataset, headers):
	id_values = sorted(list(set(dataset['id'].values)))

	length = len(id_values)
	current = 0
	print(length)

	for id_val in id_values:
		
		current += 1
		print(current, end='   \r')

		for header in headers:
			mean = dataset.loc['id'==id][header].mean()
			std = dataset.loc['id'==id][header].std()
			dataset.loc['id'==id][header] = (dataset.loc['id'==id][header] - mean) / std

	return dataset
"""