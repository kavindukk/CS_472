import numpy as np
import pandas as pd
from scipy.io import arff

from my_class import Node, decision_tree_ID3_classifier

data_train = arff.loadarff('debug_train.arff')
df_train = pd.DataFrame(data_train[0])
X_train = df_train.iloc[:,:-1].to_numpy().astype(str)
y_train = df_train.iloc[:,-1].to_numpy().astype(str)

id3  = decision_tree_ID3_classifier(X_train,y_train)
id3.id3()

data_test = arff.loadarff('lenses_test.arff')
df_test = pd.DataFrame(data_test[0])
X_test = df_test.iloc[:,:-1].to_numpy().astype(str)
y_test = df_test.iloc[:,-1].to_numpy().astype(str)


score = id3.score(X_test, y_test)
print(score)

# df  = pd.read_csv('play_tennis.csv')
# X = df.iloc[:,1:-1].to_numpy().astype(str)
# y = df.iloc[:,-1].to_numpy().astype(str)
# fatures = list(df.iloc[:,1:-1].columns.values)

# id3  = decision_tree_ID3_classifier(X,y,fatures)
# id3.id3()

# print(X)
# print(y)
# print(fatures)
