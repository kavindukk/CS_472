import pandas as pd
import numpy as np

from KNN import KNNClassifier
from scipy.io import arff

# df = pd.read_csv('data.csv', encoding='utf-8')
# labels = df.columns.values
# unique_values = df.iloc[:,1].unique()
# print(list(unique_values).index('Sunny'))
# print(unique_values)
# df.iloc[:,1] = df.iloc[:,1].replace(unique_values, [x for x in range(len(unique_values))])
# print(df.iloc[:,1])
data = arff.loadarff('glass_train.arff')
df_train = pd.DataFrame(data[0])
# print(df_train.columns.values[0])
X_train = df_train.iloc[:,:-1]
y_train = df_train.iloc[:,-1].str.decode('UTF-8')
columntype = ['real'if x !=object else 'catogorical' for x in X_train.dtypes ]
# print(X.head(2))
# print(y.head(2))
# print(columntype)

data = arff.loadarff('glass_test.arff')
df_test = pd.DataFrame(data[0])
# print(df_test.columns.values[0])
X_test = df_test.iloc[:,:-1]
y_test = df_test.iloc[:,-1].str.decode('UTF-8')
KNNobj = KNNClassifier()
KNNobj.fit(X_train,y_train, columntype).score(X_test, y_test)