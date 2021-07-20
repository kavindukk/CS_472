
from numpy.core.defchararray import count
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KNNClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,  weight_type='inverse_distance', K=3): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.weight_type = weight_type
        self.K = K

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
        self.columntype = columntype #Note This won't be needed until part 5
        if len(data.columms)==len(columntype):
            for i in range(len(columntype)):
                if columntype[i] == 'catagorical':                    
                    self.hot_encode(i)
        self.X = self.X.to_numpy()
        self.y = self.y.to_numpy()
        return self

    def hot_encode(self, index):
        unique_values = self.X.iloc[:,1].unique()
        self.X.iloc[:,index]=self.X.iloc[:,index].replace(unique_values, [x for x in range(len(unique_values))])
    
    def predict(self, data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        for i in range(len(data)):
            if self.columntype[i] == 'catagorical':
                data[i] = list(self.X.iloc[:,i].unique()).index(data[i])

        weghtList = []
        for x in self.X.shape:
            distance = np.linalg.norm(data-x)
            if self.weight_type == 'inverse_distance':
                weghtList.append(1/distance**2)
            elif self.weight_type == 'None':
                weghtList.append(distance)
        
        nearestLabels = sorted(weghtList)[:self.K]
        nearestIndexes = weghtList.index(nearestLabels)
        closestLabels = [self.y[i] for i in nearestIndexes]
        prediction = max(closestLabels, key=closestLabels.count)
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
        count_ = 0
        for x_,y_ in zip(X,y):
            prediction = self.predict(x_)
            if prediction==y_:
                count_+=1
        return count_/X.shpe[0]