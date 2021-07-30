
from numpy.core.defchararray import count
from numpy.lib.function_base import append
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,  weight_type='inverse_distance',normalize=False, K=3): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.weight_type = weight_type
        self.K = K
        self.hot_encode = {}
        self.normalize_ = normalize

        self.yHat = []

    def fit(self, data: pd.DataFrame , labels: pd.DataFrame, columntype=[]):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        self.X = data
        self.y = labels
        self.columntype = {col_name: col_type for col_name, col_type in zip(data.columns.values, columntype)} #Note This won't be needed until part 5
        # self.column_names = list(data.columns.values)

        # if len(list(data.columns))==len(columntype):
        for key, value in self.columntype.items():
            if value == 'catagorical':                    
                # self.hot_encode(key)
                unique_values = self.X[key].unique()
                self.X[key]=self.X[key].replace(unique_values, [x for x in range(len(unique_values))])
                self.hot_encode[key] = {value: i for i,value in enumerate(unique_values)}
        self.X = self.X.to_numpy()
        self.y = self.y.to_numpy()
        if self.normalize_==True:
            self.X = self._normalize_(self.X)
        return self

    def hot_encode(self, column_name):
        unique_values = self.X[column_name].unique()
        self.X[column_name]=self.X[column_name].replace(unique_values, [x for x in range(len(unique_values))])
        self.hot_encode[column_name] = {value: i for i,value in enumerate(unique_values)}
    
    def predict(self, data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        weghtList = []
        distanceList = []
        if self.weight_type == 'inverse_distance':
            for x in self.X:
                distance = np.linalg.norm(data-x)
                distanceList.append(distance)
                if distance == 0:
                    weghtList.append(np.inf)
                else:
                    weghtList.append(1/distance**2)
        elif self.weight_type == 'None':
            for x in self.X:
                distance = np.linalg.norm(data-x)
                distanceList.append(distance)
        
        nearestDistances = sorted(distanceList)[:self.K] 
        nearestIndexes = [distanceList.index(distance) for distance in nearestDistances]
        closestLabels = [self.y[i] for i in nearestIndexes]
        if self.weight_type=='None':
            prediction = max(closestLabels, key=closestLabels.count)
            self.yHat.append(prediction)
            return prediction
        elif self.weight_type=='inverse_distance':
            closestWeights = [weghtList[i] for i in nearestIndexes]
            label_values = []
            for label in list(set(closestLabels)):
                indexes = [i for i in range(len(closestLabels)) if closestLabels[i]==label]
                weights = [closestWeights[i] for i in indexes]
                distances = [nearestDistances[i] for i in indexes]
                value_for_label = np.array(weights) @ np.array(distances)
                label_values.append(value_for_label)
            index = label_values.index(max(label_values))
            prediction = list(set(closestLabels))[index]
            self.yHat.append(prediction)
            return prediction            

    #Returns the Mean score given input data and labels
    def score(self, X:pd.DataFrame, y:pd.DataFrame):
        """ Return accuracy of model on a given dataset. Must implement own score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        for key, value in self.columntype.items():
             if value == 'catagorical':
                 encode_dict = self.hot_encode[key]
                 X[key] = X[key].replace(list(encode_dict.keys()), list(encode_dict.values()))
        X = X.to_numpy()
        if type(y) != np.ndarray: 
            y = y.to_numpy()
        if self._normalize_==True:
            X = self.normalize(X)
        count_ = 0
        for x_,y_ in zip(X,y):
            prediction = self.predict(x_)
            if prediction==y_:
                count_+=1
        return count_/X.shape[0]

    def _normalize_(self, X:np.ndarray):
        if X.dtype != 'float64':
            X=X.astype('float64')
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - min(X[:,i]))/(max(X[:,i])-min(X[:,i]))
        return X