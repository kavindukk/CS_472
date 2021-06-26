from numpy.core.defchararray import count
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, epochs=None):
        self.lr = lr
        self.shuffle = shuffle
        self.epochs = epochs
        self.accuracy = 0

    def fit(self, X: np.ndarray, y: np.ndarray, initial_weights=None):
        self.weights = self.initialize_weights(X.shape[1]) if not initial_weights else initial_weights
        self.X = X
        self.y = y

        bias = np.ones(X.shape[0]).reshape(X.shape[0],1)
        X = np.append(X, bias, axis=1)
   
        if self.epoch == None:
            i=0
            accuracy = 2
            while np.mod(accuracy-self.accuracy) > 1 or i <=50:
                accuracy = self.accuracy
                self.epoch(X)
                self.accuracy = self.score(self.X, self.y)
                i = i + 1
        else:
            for i in range(self.epochs):
                self.epoch(X)
                # print(self.score(self.X, self.y))         
        return self

    def predict(self, X:np.ndarray):
        net  = X @ self.weights
        net = net.to_list()
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
        return counter*100/len(y)

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
        pass

X = np.array([
[-0.4, 0.3],
[-0.3, 0.8],
[-0.2, 0.3],
[-0.1, 0.9],
[-0.1, 0.1],
[0.0, -0.2],
[0.1, 0.2,],
[0.2, -0.2],
])
Y = np.array([1,1,1,1,0,0,0,0])

A = PerceptronClassifier(lr=0.1, shuffle=False, epochs=10)
A.fit(X=X, y=Y)