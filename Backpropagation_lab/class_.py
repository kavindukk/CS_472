from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

class MLP(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,hidden_layer_widths=None, bias=1):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent 
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLP(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.bias = bias
        self.DeltaW = None


    def fit(self, X, y, initial_weights=None, epochs=10):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        # self.weights = self.initialize_weights(X.shape[1],y.shape[1], initial_weights) if not initial_weights else initial_weights
        Xdim = self.get_dim(X)
        Ydim = self.get_dim(y)
        self.weights = self.initialize_weights(Xdim,Ydim, initial_weights) 
        self.initialize_delta_weights()
        # self.X = X
        # self.y = y

        for i in range(epochs):
            self.epoch(X,y)
        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

    def epoch(self, X, y):        
        if self.shuffle == True:
            X, y = self._shuffle_data(X,y)
        
        for i in range(X.shape[0]): 
            self.process_a_data_instance_of_an_epoch(X[i], y[i])

    def process_a_data_instance_of_an_epoch(self, X, y):
        op, opList = self.forward_pass(X)
        deltaLast = self.delta_last_layer(y,op)
        deltaList = self.delta_full_list(opList, deltaLast)
        
        dwList = list()
        for delta, op in zip(deltaList, opList[:-1]):
            dw = self.calculate_dW(op, delta)
            dwList.append(dw)
            # print(dw)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + dwList[i] + self.momentum*self.DeltaW[i]
        self.DeltaW = dwList


    def initialize_weights(self, noOfInputs, noOfOutputs, weights):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        if self.hidden_layer_widths == None:
            self.network = [noOfInputs, 2*noOfInputs, noOfOutputs]
            weights = self.assign_weight_matrix(weights)
        else:
            
            self.network = [noOfInputs] + self.hidden_layer_widths + [noOfOutputs]
            weights = self.assign_weight_matrix(weights)
        return weights

    def assign_weight_matrix(self, weight_):
        weights = list()
        for i in range(1,len(self.network)):
            m = self.network[i]
            n = self.network[i-1]+1
            if not weight_:
                weight = weight_*np.ones((m,n))
            else:
                weight = np.random.normal(loc=0, scale=np.sqrt(1), size=(m,n))
            weights.append(weight)
        return weights
    
    def initialize_delta_weights(self):
        dWList = list()
        for item in self.weights:
            dw = np.zeros_like(item)
            dWList.append(dw)
        self.DeltaW = dWList

    def forward_pass(self,input_):
        x_ = np.append(input_, self.bias)
        output = None
        outputList = list()
        for i in range(len(self.network)):
            if i==0:
                output = self.weights[i] @ x_ 
                outputList.append(np.array(input_))                
            elif i != len(self.network) -1:
                output = self.sigmoid(output)
                outputList.append(output)
                output = self.weights[i] @ np.append(output,self.bias)                 
            else:
                output = self.sigmoid(output)  
                outputList.append(output)
        return output, outputList

    def sigmoid(self, net):
        return 1/(1+np.exp(-net))

    def calculate_dW(self, output:np.array, delta:list):
        dW = list()
        for d in delta:
            dW.append(np.array(np.append(output,1))*d)
        dW = np.array(dW)*self.lr
        # print(dW)
        return dW

    def delta_last_layer(self, target, result:list):
        if type(target) == np.int32:
            target = np.array([target])
        delta = list()
        for t,o in zip(target, result):
            dt = (t-o)*o*(1-o)
            delta.append(dt)
        return delta

    def delta_full_list(self, outputL, delta):
        deltaL = [delta]
        Routput = outputL[::-1]
        Rweights = self.weights[::-1]
        for i in range(1,len(Routput)-1):
            output = Routput[i]
            dList = list()
            for j, item in enumerate(output):
                d = item*(1-item)*np.array(deltaL[i-1]) @ Rweights[i-1][:,j]
                dList.append(d)
            deltaL.append(dList)
        # print(dList)
        return deltaL[::-1]

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        concat = np.append(X,y.reshape(len(y),1), axis=1)
        np.random.shuffle(concat)
        return concat[:, :-1], concat[:,-1]

    def get_dim(self, X:np.ndarray):
        if X.ndim == 1 :
            dimX = X.ndim
        else:
            dimX = X.shape[1]
        return dimX 

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights