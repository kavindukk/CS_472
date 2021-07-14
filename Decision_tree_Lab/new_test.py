import numpy as np
import pandas as pd
import math as m

df = pd.read_csv("play_tennis.csv")

# print(list(df))

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


def find_information_gain(labels:np.ndarray):
    labelCategories = list(set(labels))
    labelCategoriesCount = [list(labels).count(x) for x in labelCategories]
    labelsCount = len(list(labels))
    informationGain = 0
    for value in labelCategoriesCount:
        informationGain -= (value/labelsCount)*m.log(value/labelsCount,2)
    return informationGain

def find_entropy_of_a_feature(X,y):
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







k = find_entropy_of_a_feature(X[:,0], y)
print(k)





