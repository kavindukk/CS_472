import numpy as np
import pandas as pd
import math as m

df = pd.read_csv("play_tennis.csv")
y = np.array(['Gr', 'B', 'G', 'Gr', 'G', 'Gr', 'G', 'G', 'B'])
X = np.array([['Y','Thin','N'],
                ['N','Deep','N'],
                ['N','Stuffed','Y'],
                ['Y','Stuffed','Y'],
                ['Y','Deep','N'],
                ['Y','Deep','Y'],
                ['N','Thin','Y'],
                ['Y','Deep','N'],
                ['N','Thin','N']])
featureLabels = np.array(['Meat', 'Crust', 'Veg'])

class Node:
    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None

    def __repr__(self) -> str:
        str_ = "branch: " + self.value +" "
        str_ += "next node: " + self.next.value + " "
        # str_ += "childs: "
        # for child in self.childs:
        #     str_ += child.value + " "
        return str_

class decision_tree_ID3_classifier:
    def __init__(self, X, y, labels) -> None:
        self.X = X
        self.y = y
        self.feature_names = labels
        self.node = Node()

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
        maxInfoGain = 0
        maxInfoGainFeature = -1
        for id_ in feature_ids:
            entropy = self.find_entropy_of_a_feature(X_ids,id_)
            featureInfoGain = infoGain - entropy
            if featureInfoGain > maxInfoGain:
                maxInfoGain = featureInfoGain
                maxInfoGainFeature = id_
        return maxInfoGainFeature, self.feature_names[maxInfoGainFeature]
    
    def id3(self):
        xIds = [x for x in range(self.X.shape[0])]
        featureIds = [x for x in range(self.X.shape[1])]
        self.node = self._id3_recv(xIds, featureIds, self.node)

    def _id3_recv(self, x_ids, feature_ids, node):
        if not node:
            node = Node()
        labels_in_features = [self.y[x] for x in x_ids]
        if len(set(labels_in_features)) == 1:
            node.value = self.y[x_ids[0]]
            return node

        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)  
            return node
       
        best_feature_id, best_feature_name  = self.find_max_information_gain_feature(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []   
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
        for value in feature_values:
            child = Node()
            child.value = value  
            node.childs.append(child)  
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print('')
            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove) 
                child.next = self._id3_recv(child_x_ids, feature_ids, child.next)
        return node





id3 = decision_tree_ID3_classifier(X,y,featureLabels)
# xIDs = [x for x in range(X.shape[0])]
# yIDs = [y for y in range(X.shape[1])]
# infoGain = id3.find_information_gain(xIDs)
# entropyMeat = id3.find_entropy_of_a_feature(xIDs,0)
# maxFeature, maxFeatureID = id3.find_max_information_gain_feature(xIDs,yIDs)

id3.id3()
node = id3.node
print("dsldjsldj")
