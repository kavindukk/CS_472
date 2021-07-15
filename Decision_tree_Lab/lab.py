from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import arff
import math as m

class Node:
    def __init__(self) -> None:
        self.name = None
        self.childs = {}

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts=None):
        self.node = Node()  
        self.infoGain = []      

    def fit(self, X, y, feature_names=None):
        self.X = X
        self.y = y
        fType = type(feature_names)
        self.feature_names = ['feature_'+str(i) for i in range(X.shape[1])] if not (fType==list or fType==np.ndarray) else list(feature_names)
        self.id3()

        return self

    def find_information_gain(self, X_ids):
        y = [self.y[y] for y in X_ids]
        labelCategories = list(set(y))
        labelCategoriesCount = [list(y).count(x) for x in labelCategories]
        labelsCount = len(list(self.y))
        informationGain = 0
        for value in labelCategoriesCount:
            informationGain -= (value/labelsCount)*m.log(value/labelsCount,2)
        return informationGain

    def find_entropy_of_a_feature(self, X_ids, feature_id):
        X = [self.X[x][feature_id] for x in X_ids ]
        y = [self.y[y] for y in X_ids]
        xLabelList = list(set(X))
        xLabelsCount = [list(X).count(x) for x in xLabelList]
        instanceCount = len(list(X)) 
        labelEntropyList = []
        for label in xLabelList:
            labelEntropy = 0
            labelIndexes = [i for i in range(len(X)) if X[i]==label ]        
            yNew = [list(y)[i] for i in labelIndexes]
            yNewLabels = list(set(yNew))
            yNewLabelsCount = [list(yNew).count(x) for x in yNewLabels]
            yNewCount = len(yNew)
            for value in yNewLabelsCount:
                labelEntropy -= (value/yNewCount)*m.log(value/yNewCount,2)
            labelEntropyList.append(labelEntropy)

        featureEntropy = sum([xCount*entropy/instanceCount for xCount, entropy in zip(xLabelsCount, labelEntropyList)])
        return featureEntropy

    def find_max_information_gain_feature(self, X_ids, feature_ids):
        infoGain = self.find_information_gain(X_ids)
        maxInfoGain = -1e10
        maxInfoGainFeature = -1
        for id_ in feature_ids:
            entropy = self.find_entropy_of_a_feature(X_ids,id_)
            featureInfoGain = infoGain - entropy
            # print("id"+str(id_)+" "+str(featureInfoGain))
            if featureInfoGain > maxInfoGain:
                maxInfoGain = featureInfoGain
                maxInfoGainFeature = id_
        self.infoGain.append(maxInfoGain)
        return maxInfoGainFeature, self.feature_names[maxInfoGainFeature]

    def id3(self):
        xIds = [x for x in range(self.X.shape[0])]
        featureIds = [x for x in range(self.X.shape[1])]
        self.node = self.id3_recursive(xIds, featureIds, self.node)

    def id3_recursive(self, x_ids, feature_ids, node):
        if not node:
            node = Node()
        labels_in_features = [self.y[x] for x in x_ids]
        if len(set(labels_in_features)) == 1:           
            return  self.y[x_ids[0]]

        if len(feature_ids) == 0:
            return max(set(labels_in_features), key=labels_in_features.count)             
        
        best_feature_id, best_feature_name  = self.find_max_information_gain_feature(x_ids, feature_ids)
        node.name = best_feature_name
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
        for value in feature_values:
            x_value_ids = [x for x in x_ids if self.X[x][best_feature_id] == value ]
            value_feature_ids = list(feature_ids)
            to_remove = value_feature_ids.index(best_feature_id)
            value_feature_ids.pop(to_remove)
            node.childs[value] = self.id3_recursive(x_value_ids,value_feature_ids, node=None)
        return node

    def predict(self, x:np.array):
        currentNode = self.node
        while isinstance(currentNode, Node):
            nodeName = currentNode.name
            featureIndex = self.feature_names.index(nodeName)
            featureValue = x[featureIndex]
            currentNode = currentNode.childs[featureValue]
        return currentNode


    def score(self, X, y):
        count = 0
        for x,y_ in zip(X,y):
            yHat = self.predict(x)
            print('yHat: '+str(yHat)+' y: '+str(y_))
            if yHat == y_:
                count += 1
        return count/len(y)

# data_train = arff.loadarff('debug_train.arff')
# df_train = pd.DataFrame(data_train[0])
# X_train = df_train.iloc[:,:-1].to_numpy().astype(str)
# y_train = df_train.iloc[:,-1].to_numpy().astype(str)

# data_test = arff.loadarff('lenses_test.arff')
# df_test = pd.DataFrame(data_test[0])
# X_test = df_test.iloc[:,:-1].to_numpy().astype(str)
# y_test = df_test.iloc[:,-1].to_numpy().astype(str)


data_train = arff.loadarff('zoo.arff')
df_train = pd.DataFrame(data_train[0])
X_train = df_train.iloc[:,:-1].to_numpy().astype(str)
y_train = df_train.iloc[:,-1].to_numpy().astype(str)

data_test = arff.loadarff('zoo_test.arff')
df_test = pd.DataFrame(data_test[0])
X_test = df_test.iloc[:,:-1].to_numpy().astype(str)
y_test = df_test.iloc[:,-1].to_numpy().astype(str)

id3 = DTClassifier()
self_ = id3.fit(X_train, y_train)
# print(self_.infoGain)
# score = self_.score(X_test,y_test)
# print(score)

score = self_.score(X_test,y_test)
print('####################')
print('Accuracy: '+str(score))                                         
                                              

# Print the information gain of every split you make.
print('####################')
print("info gain: ")
print(self_.infoGain)