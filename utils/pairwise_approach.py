import numpy as np
from sklearn import svm
import copy

def transform_pairwise(X, Y):
    """
    Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem.
    In this method, all pairs are chosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    Y : array, shape (n_samples, 2)
        Target labels. The second column represents the grouping of samples,
        i.e., samples with different groups will not be considered.
        
    Returns
    -------
    X_trans : array, shape (k, n_features)
        Data as pairs
    Y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    Z_trans : array, shape (k, 3)
        [sample1, sample2, current_id]
    T_trans : dict
        {id : number of conversions}
    """

    #Initiate the output
    X_new = []
    Y_new = []
    Z_new = []
    T_new = {}
    
    current_id = -1
    i = 0
    k = 0
    rank1 = []
    id1 = -1
    rank2 = []
    id2 = []
    nConv = 0
    #sumId = []
    
    #For each sample, we check its id
    while i < len(X):
        # If its id is matching the id we are currently investigating, 
        # we append the sample according to its rank and move to the next one
        if Y[i,1] == current_id:
            if Y[i,0] == 2:
                rank2.append(copy.deepcopy(X[i]))
                id2.append(copy.deepcopy(i))
            elif Y[i,0] ==1:
                rank1 = copy.deepcopy(X[i])
                id1 = copy.deepcopy(i)
            
            #if len(sumId) == 0:
            #    sumId = copy.deepcopy(X[i])
            #else:
            #    sumId += copy.deepcopy(X[i])
            
            nConv +=1
            i += 1
        # If the id doesn't match, then we finished all the samples with the
        # current id, we append the differences (rank1 <-> rank2) in our
        # output and we change the id
        else:
            for l in range(0, len(rank2)):
                X_new.append((-1)**k * (rank2[l] - rank1)/(rank2[l] + rank1+0.0001))
                Y_new.append((-1)**k)
                
                if k % 2 == 0:
                    Z_new.append([id2[l], id1, current_id])
                else:
                    Z_new.append([id1, id2[l], current_id])
                    
                k += 1
            
            T_new[copy.deepcopy(current_id)] = copy.deepcopy(nConv)
            
            rank1 = []
            id1 = -1
            rank2 = []
            id2 = []
            nConv = 0
            #sumId = []
            current_id = Y[i,1]
    
    # If there is one id which is not in the output        
    if len(rank1) > 0 and len(rank2) > 0:
        for l in range(0, len(rank2)):
            X_new.append((-1)**k * (rank2[l] - rank1)/(rank2[l] + rank1+0.0001))
            Y_new.append((-1)**k)
            
            if k % 2 == 0:
                Z_new.append([id2[l], id1, current_id])
            else:
                Z_new.append([id1, id2[l], current_id])
            k += 1
            
        T_new[current_id] = nConv

    return np.asarray(X_new), np.asarray(Y_new).ravel(), np.asarray(Z_new), T_new

class RankSVM(svm.LinearSVC):
    """
    Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem.
    """    
    def __init__(self):
        """
        Initiate the parameters of the model
        """
        self.penalty = 'l2'
        self.loss = 'squared_hinge'
        self.dual = False
        self.tol = 0.0001
        self.C = 1.0
        self.multi_class = 'ovr'
        self.fit_intercept = True
        self.intercept_scaling = 1
        self.class_weight = None
        self.verbose = 0
        self.random_state = np.random.RandomState()
        self.max_iter = 1000
        self.coef_= []
        self.intercept_ = []
    
    def fit(self, X, Y):
        """
        Fit a pairwise ranking model.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples, 2)
            The rank and the id
        
        Returns
        -------
        self
        """
        
        # Convert the dataset for the pairwise approach
        X_trans, Y_trans, _, _ = transform_pairwise(X, Y)
        
        return super().fit(X_trans, Y_trans)

    def predictInversion(self, X):
        """
        Predict the order of each pair within X
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data already shaped for the pairwise approach
        
        Returns
        -------
        Y : array, shape (n_samples,)
            Output predicted class
        """
        
        return super().predict(X)
    
    def predictId(self, X, Y):
        """
        Predict the rank of each X
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        
        Returns
        -------
        Y : array, shape (n_samples,)
            Output predicted class
        """
        
        # Convert the dataset for the pairwise approach
        X_trans, _, Z_trans, _ = transform_pairwise(X, Y)
        
        # Store for each id the ids of each conversion and the number of pairs 
        # where this conversion won
        scoreDict = {}
        
        # Predict the output class for each pair
        Y_trans_pred = self.predictInversion(X_trans)
        
        # For each pair, we recover its id, then the winning part and we add the
        # information in the dict scoreDict[id]
        for i in range(0,len(X_trans)):
            current_id = Z_trans[i][2]
            
            if not (current_id in scoreDict):
                if Y_trans_pred[i] == 1:
                    scoreDict[current_id] = {Z_trans[i][1] : 1}
                elif Y_trans_pred[i] == -1:
                    scoreDict[current_id] = {Z_trans[i][0] : 1}
            else:
                if Y_trans_pred[i] == 1:
                    if not (Z_trans[i][1] in scoreDict[current_id]):
                        scoreDict[current_id][Z_trans[i][1]] = 1
                    else:
                        scoreDict[current_id][Z_trans[i][1]] += 1
                elif Y_trans_pred[i] == -1:
                    if not (Z_trans[i][0] in scoreDict[current_id]):
                        scoreDict[current_id][Z_trans[i][0]] = 1
                    else:
                        scoreDict[current_id][Z_trans[i][0]] += 1
        
        # Compute Predictions
        Y_pred = 2 + np.zeros(len(X), dtype=int)
        for scoreId in scoreDict.values():
            bestIdList = []
            bestScore = -1
            
            # For each conversion, we check its score, e.g. the number of won 
            # pairs, if it's strictly greater than the current best score 
            # then we reinitialize the list of best conversion and we update 
            # the bestScore. But if we have the same score, we add the 
            # conversionId to the list of best conversion.
            for idConversion, scoreConversion in scoreId.items():
                if scoreConversion > bestScore:
                    bestIdList = [idConversion]
                    bestScore = copy.deepcopy(scoreConversion)
                elif scoreConversion == bestScore:
                    bestIdList.append(idConversion)
                    
            Y_pred[np.random.choice(bestIdList)] = 1
            
        return Y_pred

    def scoreInversion(self, X, Y):
        """
        Compute the number of good matches between a pair and its rank difference
        over the total number of pairs.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples, 2)
            The rank and the id
        
        Returns
        -------
        score : float
            The output score as described previously
        """
        
        # Convert the dataset for the pairwise approach
        X_trans, Y_trans, _, _ = transform_pairwise(X, Y)
        
        # Number of pairs
        nb_tot = len(X_trans)  
        
        # Predict the output class for each pair
        pred = (self.predictInversion(X_trans) != Y_trans)
        
        # Count the number of misclassified pairs
        misclassified = sum(pred)
        
        # Compute the score 
        score = 1 - misclassified/nb_tot
        return score
    
    def scoreThresholdId(self, X, Y, p):
        """
        Compute the number of id where at least a fraction p of all the pairs 
        with the same id are correctly classified
        over the number of id.
        At the moment, it is too restrictive
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples, 2)
            The rank and the id
        p : float in [0,1]
            The threshold we consider to fetch the rank 1 conversion from the pairs
        
        Returns
        -------
        score : float
            The output score as described previously
        """
        
        # Convert the dataset for the pairwise approach
        X_trans, Y_trans, Z_trans, T_trans = transform_pairwise(X, Y)
        
        # Store the proportion of correctly clasified pair for each id
        scoreDict = {}
        
        # Predict the output class for each pair
        pred = (self.predictInversion(X_trans) == Y_trans)
        
        # For each pair, we recover its id and we add the result of the match
        # in scoreDict[id]
        for i in range(0,len(X_trans)):
            current_id = Z_trans[i][2]
            
            if not (current_id in scoreDict):
                scoreDict[current_id] = pred[i]/T_trans[current_id]
            else:
                scoreDict[current_id] += pred[i]/T_trans[current_id]
        
        # Compute the score
        score = 0
        for scoreId in scoreDict.values():
            if scoreId > p:
                score += 1
                
        score = score/len(scoreDict)
        
        return score
    
    def scoreId(self, X, Y):
        """
        Compute the number of id where the rank 1 conversion is detected over 
        the number of id
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            The data
        y : array, shape (n_samples, 2)
            The rank and the id
        
        Returns
        -------
        score : float
            The output score as described previously
        """
        
        Y_pred = self.predictId(X, Y)
        
        # Compute the score
        score = 0
        total = 0
        for i in range(0, len(X)):
            if Y_pred[i] == 1:
                total +=1
                
                if Y[i, 0] == 1:
                    score += 1
                
        score = score/total
        
        return score