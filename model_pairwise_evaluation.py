import numpy as np
from utils.kfolds import read_csv
from utils.prediction import compute_error
from utils.pairwise_approach import RankSVM
import argparse

def main(train_path, test_path):
    delimiter = ';'
    verbose = True
    data_frame, all_headers = read_csv(train_path, delimiter, verbose)

    delimiter_test = ','
    verbose = True
    data_frame_test, all_headers_test = read_csv(test_path, delimiter_test, verbose)

    headers = [
        'id',
        'rank',
        'occurrences',
        'lifetime',
        'nb_days',
        'nb_idtags',
        'nb_idtags_site',
        'nb_idtags_media',
        'nb_purchases',
        'last_time',
        'nb_ips'
    ]

    headers_test = [
        'id',
        'similarity',
        'occurrences',
        'lifetime',
        'nb_days',
        'nb_idtags',
        'nb_idtags_site',
        'nb_idtags_media',
        'nb_purchases',
        'last_time',
        'nb_ips'
    ]

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
    X_train = train[headers_to_scale].values
    Y_train = train[['rank', 'id']].values

    similarity =  np.array(test.loc[:,'similarity'])
    baseline = np.zeros(similarity.shape)

    test = test.reset_index(drop=True)
    test['arg_max']=0

    old_id = test.loc[0,'id']
    rows = []

    for row in range(test.shape[0]):
        current_id = test.loc[row,'id']
        if current_id == old_id:
            rows.append(row)
        else:
            sample = np.random.randint(min(rows), max(rows)+1)
            baseline[sample] = 1
            old_id = current_id
            m = test.loc[rows, 'similarity'].max()
            test.loc[rows, 'arg_max'] = 2 - (test.loc[rows, 'similarity'] == m)
            rows = []
            rows.append(row)
        
        if row == test.shape[0]-1:
            sample = np.random.randint(min(rows), max(rows)+1)
            baseline[sample] = 1
            old_id = current_id
            m = test.loc[rows, 'similarity'].max()
            test.loc[rows, 'arg_max'] = 2 - (test.loc[rows, 'similarity'] == m)
            rows = []
            rows.append(row)

    X_test = test[headers_to_scale].values
    Y_test = test[['arg_max','id']].values

    # training model
    rank_svm = RankSVM()
    rank_svm = rank_svm.fit(X_train, Y_train)

    # computing score on train set
    missranked_score_train = 1 - rank_svm.scoreId(X_train, Y_train)
    
    # printing intermediate results
    print('train set:')
    print('   missranked =', round(missranked_score_train, 3))
    print('   wellranked =', round(1 - missranked_score_train, 3))

    # computing score on test set
    missranked_score_test = 1 - rank_svm.scoreId(X_test, Y_test)
    missranked_test_baseline, wellranked_test_baseline, total_test_baseline = compute_error(Y_test[:,0], baseline)

    # printing intermediate results
    print('test set:')
    print('   missranked =', round(missranked_score_test, 3))
    print('   wellranked =', round(1 - missranked_score_test, 3))
    print('baseline prediction:')
    print('   missranked =', round(missranked_test_baseline/total_test_baseline, 3))
    print('   wellranked =', round(wellranked_test_baseline/total_test_baseline, 3))

# =============================================================================
#     # Les deux métriques du pdf
# 
#     #df_2 = data_frame_test.copy()
# 
#     old_id = test.loc[0, 'id']
#     rows = []
#     score_1 = 0
#     score_2 = 0
#     score_1_baseline = 0
#     score_2_baseline = 0
#     N = 0
# 
#     for row in range(len(Y_predicted_test)):
#         current_id = test.loc[row,'id']
# 
#         if current_id == old_id:
#             rows.append(row)
#             if Y_predicted_test[row] == 1:
#                 prediction = row
#                 N += 1
#             if baseline[row] == 1:
#                 prediction_baseline = row
#         else:        
#             m = test.loc[rows, 'similarity'].max()    
#             score_1 += m - test.loc[prediction, 'similarity']
#             score_2 += m*test.loc[prediction, 'similarity']
#             score_1_baseline += m - test.loc[prediction_baseline, 'similarity']
#             score_2_baseline += m*test.loc[prediction_baseline, 'similarity']       
#             #print(str(N)+' : commande '+str(old_id)+' nb_lignes = '+str(len(rows))+' ; sim_max = '+str(m)+' ; sim_predicted = '+str(test.loc[prediction, 'similarity'])+' ; sim_baseline = '+str(test.loc[prediction_baseline, 'similarity'])) 
#             old_id = current_id
#             rows = []
#             rows.append(row)
#             if Y_predicted_test[row] == 1:
#                 prediction = row
#                 N += 1
#             if baseline[row] == 1:
#                 prediction_baseline = row
# 
#         if row == len(Y_predicted_test)-1:        
#             m = test.loc[rows, 'similarity'].max()    
#             score_1 += m - test.loc[prediction, 'similarity']
#             score_2 += m*test.loc[prediction, 'similarity']
#             score_1_baseline += m - test.loc[prediction_baseline, 'similarity']
#             score_2_baseline += m*test.loc[prediction_baseline, 'similarity']        
#             #print(str(N)+' : commande '+str(old_id)+' nb_lignes = '+str(len(rows))+' ; sim_max = '+str(m)+' ; sim_predicted = '+str(test.loc[prediction, 'similarity'])+' ; sim_baseline = '+str(test.loc[prediction_baseline, 'similarity'])) 
# 
# 
#     score_1 /= N
#     score_2 /= N
#     score_1_baseline /= N
#     score_2_baseline /= N
# 
# 
# 
# 
# 
#     print('score_1 = ' + str(score_1))
#     print('score_2 = ' + str(score_2))
#     print('score_1_baseline = ' + str(score_1_baseline))
#     print('score_2_baseline = ' + str(score_2_baseline))
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainset', help='path to the dataset', required=True)
    parser.add_argument('--testset', help='path to the dataset', required=True)
    
    opts = parser.parse_args()

    main(opts.trainset, opts.testset)