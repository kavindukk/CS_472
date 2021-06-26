from numpy.core.defchararray import count
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron

import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd
import numpy as np

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, epochs=None):
        self.lr = lr
        self.shuffle = shuffle
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray, initial_weights=None):
        self.weights = self.initialize_weights(X.shape[1]) if not initial_weights else initial_weights
        self.X = X
        self.y = y

        bias = np.ones(X.shape[0]).reshape(X.shape[0],1)
        X = np.append(X, bias, axis=1)
   
        if self.epochs == None:
            i=0

            while self.accuracy <= .95 and i <=50:
                self.epoch(X)
                self.accuracy = self.score(X, self.y)
                i = i + 1
                print(self.score(X, self.y))  
        else:
            for i in range(self.epochs):
                self.epoch(X)
                print(self.score(X, self.y))         
        return self

    def predict(self, X:np.ndarray):
        net  = X @ self.weights
        net = net.tolist()
        yHat = [1 if item>0 else 0 for item in net]
        return yHat

    def initialize_weights(self, n):
        weights = np.zeros((n+1))
        return weights
        
    def score(self, X, y):
        yHat = self.predict(X)
        counter = 0
        for yy, yh in zip(y, yHat):
            if yy == yh:
                counter = counter + 1
        return counter/len(y)

    def _shuffle_data(self, X, y):
        concat = np.append(X,y.reshape(len(y),1), axis=1)
        np.random.shuffle(concat)
        return concat[:, :-1], concat[:,-1]

    def epoch(self,X):
        if self.shuffle == True:
            X, y = self._shuffle_data(X,self.y)
        else:
            y = self.y

        net = None
        output = None

        for i in range(X.shape[0]):        
            net = np.dot(X[i],self.weights)
            output = 1 if net > 0 else 0
            dWeight = self.lr*(y[i] - output)*X[i]
            self.weights = self.weights + dWeight
        print(self.weights)        

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights

data = arff.loadarff("b.arff")
df = pd.DataFrame(data[0])
X = df.iloc[:,:-1].to_numpy()
Y = df.iloc[:,-1].to_numpy().astype(np.int)

# print(y)

# X = np.array([
# [-0.4, 0.3],
# [-0.3, 0.8],
# [-0.2, 0.3],
# [-0.1, 0.9],
# [-0.1, 0.1],
# [0.0, -0.2],
# [0.1, 0.2,],
# [0.2, -0.2],
# ])
# Y = np.array([1,1,1,1,0,0,0,0])

A = PerceptronClassifier(lr=0.1, shuffle=False, epochs=None)
A.fit(X=X, y=Y)