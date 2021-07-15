import numpy as np
import pandas as pd
from scipy.io import arff
import arff as arf

# from my_class import Node, decision_tree_ID3_classifier

from lab import DTClassifier

# data_train = arff.loadarff('debug_train.arff')
# df_train = pd.DataFrame(data_train[0])
# X_train = df_train.iloc[:,:-1].to_numpy().astype(str)
# y_train = df_train.iloc[:,-1].to_numpy().astype(str)

# id3  = decision_tree_ID3_classifier(X_train,y_train)
# id3.id3()

# data_test = arff.loadarff('lenses_test.arff')
# df_test = pd.DataFrame(data_test[0])
# X_test = df_test.iloc[:,:-1].to_numpy().astype(str)
# y_test = df_test.iloc[:,-1].to_numpy().astype(str)


# score = id3.score(X_test, y_test)
# print(score)

# df  = pd.read_csv('play_tennis.csv')
# X = df.iloc[:,1:-1].to_numpy().astype(str)
# y = df.iloc[:,-1].to_numpy().astype(str)
# fatures = list(df.iloc[:,1:-1].columns.values)

# id3  = decision_tree_ID3_classifier(X,y,fatures)
# id3.id3()

# print(X)
# print(y)
# print(fatures)
data_ = list(arf.load('cars.arff'))
data_ = np.asarray(data_, dtype=str)
df = pd.DataFrame(data_)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
data = df.to_numpy().astype(str)

# data_ = arff.loadarff('voting_with_missing.arff')
# df = pd.DataFrame(data_[0])
# features = df.columns.values
# data = df.to_numpy().astype(str)
# df = pd.DataFrame(data, columns=features)
# df = df.replace('?', np.nan)
# df.fillna(method='ffill', inplace=True)
# df.fillna(method='bfill', inplace=True)

# A = df.isnull().any(axis=1)
# B = [x for x in A if x == True]

# print(B)

# print(data)
# df = df.apply(str)
# df.replace('?',np.nan)
# print(df.head(2))
# df.fillna(df.mean())
# data = df.to_numpy().astype(str)

from random import randrange

def cross_validation_split(dataset, folds=10):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split


# print(data_splits[0])
def calculate_cross_validation_accuracy(data_splits):
    accuracyList = []
    for i in range(len(data_splits)):
        test_data = np.array(data_splits[i])    
        train_data = []
        for j in range(len(data_splits)):
            if not j==i:
                for data_point in data_splits[j]:
                    train_data.append(data_point)           
        train_data = np.array(train_data)
        id3 = DTClassifier()
        accuracy = id3.fit(train_data[:,:-1],train_data[:,-1]).score(test_data[:,:-1], test_data[:,-1])
        print("Accuracy_"+str(i)+" : "+str(accuracy))
        accuracyList.append(accuracy)
    return sum(accuracyList)/len(accuracyList)

data_splits = cross_validation_split(data)
accuracy = calculate_cross_validation_accuracy(data_splits)
print('###############')
print("Average Accuracy: "+str(accuracy))