from class_ import MLP

from scipy.io import arff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# mlp = MLP(lr=1,  hidden_layer_widths=[2])
# mlp.fit(X=None, y=None).process_a_data_instance_of_an_epoch([0,0],np.array([0,1]))

######debug dataset
# data  = arff.loadarff("debug.arff")
# df = pd.DataFrame(data[0])
# X = df.iloc[:,:-1].to_numpy()
# Y = df.iloc[:,-1].to_numpy().astype(np.int)

# debugMLP = MLP(lr=0.1,shuffle=False, momentum=0.5, hidden_layer_widths=[4])
# debugMLP.fit(X,Y, initial_weights=0, epochs=10)
# W = debugMLP.get_weights()[1]
# print(W)

######evaluation dataset
# data  = arff.loadarff("evaluation.arff")
# df = pd.DataFrame(data[0])
# X = df.iloc[:,:-1].to_numpy()
# Y = df.iloc[:,-1].to_numpy().astype(np.int)

# print(Y)

# debugMLP = MLP(lr=0.1,shuffle=False, momentum=0.5, hidden_layer_widths=[4])
# debugMLP.fit(X,Y, initial_weights=0, epochs=10)
# W = debugMLP.get_weights()[1]
# print(W)

#iris data set###########################################
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data  = arff.loadarff("iris.arff")
df = pd.DataFrame(data[0])
LE = LabelEncoder()
OHE= OneHotEncoder(sparse=False)
y = OHE.fit_transform(LE.fit_transform(df['class']).reshape(-1,1))
X = df.iloc[:,:-1].to_numpy()

xTrain, xEval, yTrain, yEval = train_test_split(X,y, train_size=.8)
irisMLP = MLP(lr=0.1,shuffle=False, momentum=0, hidden_layer_widths=None)
accuracy = irisMLP.fit(xTrain, yTrain, epochs=None, validation_percentage=.15).score(xEval,yEval)
print(accuracy)

#Part B
# LR = [0.01, 0.115, 0.168, 0.536, .852, 1.01]
# mseData = []
# for lr_ in LR:
#     irisMLP = MLP(lr=lr_,shuffle=False, momentum=0, hidden_layer_widths=None)
#     accuracy = irisMLP.fit(xTrain, yTrain, epochs=None, validation_percentage=.15).score(xEval,yEval)
#     mseData.append(irisMLP.mse)

# def draw_the_plot_MSE(mseData, LR):
#     for mse, lr_ in zip(mseData, LR):
#         x = [i for i in range(1,len(mse)+1)]
#         y = mse
#         label = 'lr = ' + str(lr_)
#         plt.plot(x,y, label=label)
#     plt.ylabel("MSE change over the epochs")
#     plt.title("MSE vs Epochs")
#     plt.legend()
#     plt.show()
#     plt.grid(True)

#Part C

irisMLP = MLP(lr=0.1,shuffle=False, momentum=0, hidden_layer_widths=None)
accuracy = irisMLP.fit(xTrain, yTrain, epochs=None, validation_percentage=.15).score(xEval,yEval)
accuracyList = irisMLP.accuracyList

def draw_the_plot_accuracy(accuracyList):
    x = [i for i in range(1,len(accuracyList)+1)]
    y = accuracyList
    plt.plot(x,y)
    plt.ylabel("Accuracy change over the epochs")
    plt.title("Accuracy vs Epochs")
    plt.show()
# draw_the_plot_MSE(mseData, LR)
draw_the_plot_accuracy(accuracyList)
