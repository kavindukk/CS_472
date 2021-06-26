from numpy.core.defchararray import count
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, epochs=None):
        """ 
            Initialize class with chosen hyperparameters.
        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT 
            SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray, initial_weights=None):
        """ 
            Fit the data; run the algorithm and adjust the weights to find a 
            good solution
        Args:
            X (array-like): A 2D numpy array with the training data, excluding
            targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial 
            weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.weights = self.initialize_weights(X.shape[1]) if not initial_weights else initial_weights
        self.X = X
        self.y = y

        # bias = np.ones(X.shape[0]).reshape(X.shape[0],1)
        # X = np.append(X, bias, axis=1)

        # net = None
        # output = None

        # for i in range(X.shape[0]):        
        #     net = np.dot(X[i],self.weights)
        #     output = 1 if net > 0 else 0
        #     dWeight = self.lr*(y[i] - output)*X[i]
        #     self.weights = self.weights + dWeight    
        if self.epoch == None:
            pass
        else:
            for i in range(self.epochs):
                self.epoch()         
        return self

    def predict(self, X:np.ndarray):
        """ 
            Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding 
            targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        net  = X @ self.weights
        net = net.to_list()
        yHat = [1 if item>0 else 0 for item in net]
        return yHat
 

    def initialize_weights(self, n):
        """ Initialize weights for perceptron. Don't forget the bias!
        Returns:
        """
        weights = np.zeros((n+1))
        return weights

    def score(self, X, y):
        """ 
            Return accuracy of model on a given dataset. Must implement own 
            score function.
        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets
        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """
        yHat = self.predict(X)
        counter = 0
        for yy, yh in zip(y, yHat):
            if yy == yh:
                counter = counter + 1
        return counter*100/len(y)

    def _shuffle_data(self, X, y):
        """ 
            Shuffle the data! This _ prefix suggests that this method should 
            only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D 
            array, rather than shuffling X and y exactly the same way, 
            independently.
        """
        concat = np.append(X,y, axis=1)
        np.random.shuffle(concat)
        return concat[:, :-1], concat[:,-1]

    def epoch(self):
        bias = np.ones(self.X.shape[0]).reshape(self.X.shape[0],1)
        X = np.append(self.X, bias, axis=1)
        net = None
        output = None

        for i in range(X.shape[0]):        
            net = np.dot(X[i],self.weights)
            output = 1 if net > 0 else 0
            dWeight = self.lr*(self.y[i] - output)*X[i]
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