import numpy as np
import argparse
from utils.kfolds import KFolds, read_csv
from utils.pairwise_approach import RankSVM

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
        
        X_train = train[headers_to_scale].values
        Y_train = train[['rank', 'id']]
        
        X_test = test[headers_to_scale].values
        Y_test = test[['rank', 'id']]
        
        rank_svm = RankSVM()
        
        rank_svm = rank_svm.fit(X_train, Y_train)
        
        missranked_score_train = 1 - rank_svm.scoreInversion(X_train, Y_train)
        missranked_scores_train.append(missranked_score_train)
        
        missranked_score_test = 1 - rank_svm.scoreInversion(X_test, Y_test)
        missranked_scores_test.append(missranked_score_test)
        
        # printing intermediate results
        if verbose:
            print('\n**** fold ', k, '****')
            print('train set:')
            print('   missranked =', round(missranked_score_train, 3))
            print('   wellranked =', round(1 - missranked_score_train, 3))
            print('test set:')
            print('   missranked =', round(missranked_score_test, 3))
            print('   wellranked =', round(1 - missranked_score_test, 3))
            
    if verbose:
        print('\n******** MEAN over all folds ********')
        print('Train missranked = ', np.mean(missranked_scores_train))
        print(' Test missranked = ', np.mean(missranked_scores_test))
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='path to the dataset', required=True)
    parser.add_argument('--delimiter', help='delimiter used in the dataset', required=True)
    parser.add_argument('--verbose', help='increase output verbosity', action='store_true', required=False)
	
    opts = parser.parse_args()

    main(opts.dataset, opts.delimiter, opts.verbose)