from scipy.io import arff
import pandas as pd
import numpy as np

# data = arff.loadarff("a.arff")
# df = pd.DataFrame(data[0])


# a = df['class'].to_numpy()
# a = a.tolist()
# a = [int(s.decode()) for s in a ]
# # print(type(a[0]))

# b = df['class'].to_numpy()
# # b = b.astype('U13')
# b = b.astype(np.int)
# print(b)

# print(df.iloc[:,:-1])
# X = df.iloc[:,:-1].to_numpy()
# y = df.iloc[:,-1].to_numpy().astype(np.int)
# print(np.dot(X[1,:], X[1,:]))
# print(y)
# print(X.shape[1])

X = np.array([
[-0.4, 0.3, 1],
[-0.3, 0.8, 1],
[-0.2, 0.3, 1],
[-0.1, 0.9, 1],
[-0.1, 0.1, 1],
[0.0, -0.2, 1],
[0.1, 0.2,  1],
[0.2, -0.2, 1],
])
Y = np.array([1,1,1,1,0,0,0,0])
weights = np.array([0,0,0])
lr = 0.1
epoch=10


def epoch(X, y, lr=1, weights=np.array([0,0,0,0])):
    for i in range(X.shape[0]):        
        net = np.dot(X[i],weights)
        output = 1 if net > 0 else 0
        dWeight = lr*(y[i] - output)*X[i]
        weights = weights + dWeight  
    print(weights)

# X = np.array([[0,0,1,1],[1,1,1,1],[1,0,1,1],[0,1,1,1]])
# Y = np.array([0,1,1,0])
# weights = np.array([0,0,0,0])

def iterations(X:np.ndarray, y:np.ndarray, lr, weights, epochs):    
    for i in range(epochs):
        for i in range(X.shape[0]):        
            net = np.dot(X[i],weights)
            output = 1 if net > 0 else 0
            dWeight = lr*(y[i] - output)*X[i]
            weights = weights + dWeight  
        print(weights)

iterations(X=X, y=Y, lr=lr, weights=weights, epochs=10)