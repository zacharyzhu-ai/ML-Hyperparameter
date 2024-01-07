# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:41:45 2023

@author: 1055842
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:26:19 2023

@author: 898976
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
# =============================================================================
# Complete the code for linear regression
# 1a, 1b, 2, 3, 4
# Code should be vectorized when complete
# =============================================================================


def getDataSets():
    fileName = 'CricketChirpData_Train.csv'

    print("fileName:", fileName)
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to floats
    data = np.loadtxt(raw_data, usecols = (0, 1), skiprows = 1, delimiter=",")
   
    x_train = data[:, 0:2]
    y_train = data[:, -1]
    ###TEST DATA SET
    fileName = 'CricketChirpData_Test.csv'
    print("fileName:", fileName)
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to floats
    data = np.loadtxt(raw_data, usecols = (0, 1), skiprows = 1, delimiter=",")
   
    xtest = data[:, 0:2]
    ytest = data[:, -1]
    print("Xtrain example data: ", x_train[0:1, :])
    print("Ytrain example data:", y_train[0])
    print("Xtest example data: ", xtest[0:1, :])
    print("Ytest example data:", ytest[0])
    return x_train, y_train, xtest, ytest
                 

def createData():
   
    #print (data)
#1a
    x_train, y_train, x_test, y_test = getDataSets()
    x_train, X_mean, X_std = standardize(x_train)
    x_train = np.hstack((np.ones((len(x_train), 1)),x_train))
    return x_train,y_train, x_test, y_test, X_mean, X_std

def standardize(X):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X = (X-X_mean)/X_std
    return X, X_mean, X_std



#2
def calcCost(X,W,Y):
  pred = np.dot(X, W)
  cost_arr = (pred-Y)**2
  return np.mean(cost_arr)


#4
def calcGradient(X,Y,W):
  pred = np.dot(X, W)
  val = pred-Y
  grad = np.dot(X.T, val)
  return grad/len(Y)

def predictTestData(x_test, y_test, w, mean, std):
    print("Xtest Data sample: ", x_test[0:3, :])
    x_scale = (x_test-mean)/ std
    print("X scaled Data sample (for example scaled)", x_scale[0:3, :])
    x_scale = np.hstack((np.ones((len(y_test),1)),x_scale))
    pred = np.dot(x_scale,w)
   
    print("Prediction sample data:", pred[0:3])
    print("Ytest Sample data:", y_test[0:3])
    print("Pred-Test (Actual) sample data ", pred[0:3]-y_test[0:3])
   
    return pred

############################################################
# Create figure objects

fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]

#1b
# =============================================================================
#  X,Y use createData method to create the X,Y matrices
# Weights - Create initial weight matrix to have the same weights as features
# Weights - should be set to 0
# =============================================================================
#
np.set_printoptions(precision=4, suppress=True)

X, Y, x_test, y_test, X_mean, X_std = createData()
numRows, numCols = X.shape
W=np.zeros(numCols)

# set learning rate - the list is if we want to try multiple LR's
# We are only doing one of them today
lrList = [0.649,.01]
lr = lrList[0]

#set up the cost array for graphing
costArray = []
costArray.append(calcCost(X, W, Y))

#initalize while loop flags
finished = False
count =0

while (not finished and count <100000):
    gradient = calcGradient(X,Y,W)
    #print (gradient)
    #5 update weights

    W = W - (lr * gradient)


    costArray.append(calcCost(X, W, Y))
    lengthOfGradientVector = np.linalg.norm(gradient)
    if (lengthOfGradientVector < .00001):
        finished=True
    count+=1

print("weights: ", W, "Count: ", count, "Cost: ", costArray[-1])        
ax.plot(np.arange(len(costArray)), costArray, "ro", label = "cost")
predictTestData(x_test, y_test, W, X_mean, X_std)

ax.set_title("Cost as weights are changed")
ax.set_xlabel("iteration")
ax.set_ylabel("Costs")
ax.legend()  
