# from test import Rweights
import numpy as np  

def crate_weights(network):
    weights = list()
    for i in range(1,len(network)):
        m = network[i]
        n = network[i-1]+1
        weight = np.ones((m,n))
        # weight[:,-1]=np.ones(weight.shape[0])
        weights.append(weight)
    return weights

network = np.array([2,2,2])
weights = crate_weights(network=network)
##############################################

def sigmoid(net):
    return 1/(1+np.exp(-net))
    
def forward_pass(input_, network,  weights, bias=1):
    x_ = np.append(input_, bias)
    output = None
    outputList = list()
    for i in range(len(network)):
        if i==0:
            output = weights[i] @ x_  
            outputList.append(input_)               
        elif i != len(network) -1:
            output = sigmoid(output)
            outputList.append(output)
            output = weights[i] @ np.append(output,bias)                 
        else:
            output = sigmoid(output)  
            outputList.append(output)
    return output, outputList

op, opList = forward_pass([0,0],network, weights)
# print(opList)
################################################
def calculate_dW(lr:int, output:np.array, delta:list):
    dW = list()
    for d in delta:
        dW.append(np.array(np.append(output,1))*d)
    dW = np.array(dW)*lr
    # print(dW)
    return dW
###############################################

def delta_last_layer(target:np.array, result:list):
    delta = list()
    for t,o in zip(target, result):
        dt = (t-o)*o*(1-o)
        delta.append(dt)
    return delta

delta1 = delta_last_layer(np.array([0,1]), op)
# print(delta1)
dW = calculate_dW(1,opList[-2], delta1)
#######################################

def delta_list(outputList, deltaList, weights):
    # deltaList = np.array(deltaList)
    Routput = outputList[::-1]
    Rweights = weights[::-1]
    for i in range(1,len(Routput)-1):
        output = Routput[i]
        dList = list()
        for j,item in enumerate(output):
            d = Routput[i][j]*(1-Routput[i][j])*(np.array(deltaList[i-1])@Rweights[i-1][:,j])
            dList.append(d)
        print(dList)
    return dList

def delta_full_list(outputL, delta, weights):
    deltaL = [delta]
    Routput = outputL[::-1]
    Rweights = weights[::-1]
    for i in range(1,len(Routput)-1):
        output = Routput[i]
        dList = list()
        for j, item in enumerate(output):
            d = item*(1-item)*np.array(deltaL[i-1]) @ Rweights[i-1][:,j]
            dList.append(d)
        deltaL.append(dList)
    # print(dList)
    return deltaL[::-1]


dList = delta_full_list(opList, delta1, weights)
dw2 = calculate_dW(1,opList[0], dList[0])
# print(dw2)



