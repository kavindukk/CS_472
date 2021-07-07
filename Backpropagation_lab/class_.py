from numpy.core.defchararray import count
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt

class MLP(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,hidden_layer_widths=None, bias=1):
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle
        self.bias = bias
        self.DeltaW = None
        self.mse = []
        self.accuracyList = []


    def fit(self, X, y, initial_weights=None, epochs=10, validation_percentage=.15):
        # self.weights = self.initialize_weights(X.shape[1],y.shape[1], initial_weights) if not initial_weights else initial_weights
        Xdim = self.get_dim(X)
        Ydim = self.get_dim(y)
        # Ydim = 2 if self.get_dim(y)==1 else self.get_dim(y)
        self.weights = self.initialize_weights(Xdim,Ydim, initial_weights) 
        self.initialize_delta_weights()

        if not epochs:
            xShuffled, yShuffled = self._shuffle_data(X,y)            
            xTrain, xValid, yTrain, yValid = self.split_data(xShuffled, yShuffled, percentage=validation_percentage)
            # xTrain, xValid, yTrain, yValid = self.split_data(X, y, percentage=validation_percentage)
            accuracy = 0
            accuracyRepeatCount = 0
            while accuracy< 0.95 and accuracyRepeatCount <10 :
                mseOfEpoch = self.epoch(xTrain,yTrain)
                self.mse.append(mseOfEpoch)
                currentAccuracy = self.score(xValid,yValid)
                self.accuracyList.append(currentAccuracy)
                accuracyRepeatCount = accuracyRepeatCount + 1 if np.abs(accuracy-currentAccuracy) <= 0.02 else 0 
                accuracy = currentAccuracy
        else:
            for i in range(epochs):
                mseOfEpoch = self.epoch(X,y)
                self.mse.append(mseOfEpoch)
        return self

    def score(self, X, y):
        outputsOrIndexes = self.predict(X, y.ndim)
        accuracyCount = 0
        if y.ndim == 1:
            for output, target in zip(outputsOrIndexes, y):
                if output == target: accuracyCount = accuracyCount + 1
        else:
            for index_, target in zip(outputsOrIndexes, y):
                if target[index_] == 1 : accuracyCount = accuracyCount + 1
        
        return accuracyCount/X.shape[0]

    def predict(self, X, outputDimension):
        if outputDimension==1: #perceptron Logic
            opVector = []
            for x_ in X:
                net, _ = self.forward_pass(x_)
                op = 1 if net>0 else 0
                opVector.append(op)
            return opVector
        else:
            indexes = []
            for x_ in X: #Multi-output classification
                op, _ = self.forward_pass(x_)
                op = op.tolist()
                # index = op.index(max(op))
                index = op.index(max(op))
                indexes.append(index)
            return indexes

    def epoch(self, X, y): 
        mse = 0       
        if self.shuffle == True:
            X, y = self._shuffle_data(X,y)
        
        for i in range(X.shape[0]): 
            mseDataPoint =  self.process_a_data_instance_of_an_epoch(X[i], y[i])
            mse = mse + mseDataPoint
        return mse/X.shape[0]

    def process_a_data_instance_of_an_epoch(self, X, y):
        op, opList = self.forward_pass(X)
        deltaLast = self.delta_last_layer(y,op)
        deltaList = self.delta_full_list(opList, deltaLast)

        mseDataPoint = np.square(op-y).mean()
        
        dwList = list()
        for delta, op in zip(deltaList, opList[:-1]):
            dw = self.calculate_dW(op, delta)
            dwList.append(dw)
            # print(dw)
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] + dwList[i] + self.momentum*self.DeltaW[i]
        self.DeltaW = dwList
        return mseDataPoint


    def initialize_weights(self, noOfInputs, noOfOutputs, weights):
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
            if weight_==0 or weight_:
                weight = weight_*np.ones((m,n))
            elif weight_==None:
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

    def _shuffle_data(self, X, y):
        if y.ndim == 1:
            concat = np.append(X,y.reshape(len(y),1), axis=1)
            np.random.shuffle(concat)   
            return concat[:, :-1], concat[:,-1]
        else:
            concat = np.append(X,y, axis=1)
            np.random.shuffle(concat)
            return concat[:, :-y.shape[1]], concat[:,-y.shape[1]:]
        

    def get_dim(self, X:np.ndarray):
        if X.ndim == 1 :
            dimX = X.ndim
        else:
            dimX = X.shape[1]
        return dimX 

    def split_data(self, X, y, percentage):
        xTrain, xEval, yTrain, yEval = train_test_split(X,y,test_size=percentage)
        return xTrain, xEval, yTrain, yEval

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights