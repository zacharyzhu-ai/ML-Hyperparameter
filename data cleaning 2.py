# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:37:15 2023

@author: 1055842
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from sklearn.preprocessing import OneHotEncoder
# =============================================================================
# Complete the code for linear regression
# 1a, 1b, 2, 3, 4
# Code should be vectorized when complete
# =============================================================================


def getDataSets():
    fileName = '1B_Train.csv'
    print("fileName: ", fileName)
    raw_data = open(fileName, 'rt')
    #loadtxt defaults to floats
    #class gender age
    data = np.loadtxt(raw_data, usecols = (1,2,3,4,5,6,7,8, 9, 10, 11), skiprows = 1, delimiter=",", dtype=str)
    #gender age class
    xtrain = data[:,0:-1]
    xtrain[:,0:8] = xtrain[:,0:8].astype(float)
    ohe = OneHotEncoder(categories = 'auto')
    xtrain = np.hstack((
        xtrain[:,0:8],
                        ohe.fit_transform(xtrain[:,8:10]).toarray()))
    #survived
    xtrain = xtrain.astype(float)
    ytrain = data[:,-1]
    ytrain = (ytrain == "1").astype(float).reshape((-1,1))
    actual = ytrain.copy()
    return xtrain, ytrain, actual
   
def createData():
    xtrain, ytrain, actual = getDataSets()
    xtrain[:,0:4] = standardize(xtrain[:,0:4])
   
    xones = np.ones((len(xtrain),1))
    xtrain = np.hstack((xones,xtrain))
       
    return xtrain,ytrain, actual

def standardize(X):
    Xmean =X.mean(axis=0)
    Xstd = X.std(axis=0)
    x = (X-Xmean)/Xstd
    return x

def activation(X,W):
    z = np.dot(X,W)
    return (1/(1+np.exp(-z)))
#2
def calcCost(X,W,Y):
  pred = activation(X,W)
  cost = -1*(Y*np.log(pred)+(1-Y)*np.log(1-pred))
  return np.mean(cost)
#4
def calcGradient(X,Y,W):
  pred = activation(X,W)
  XYmat = pred-Y
  grad = np.dot(X.T,XYmat)
  grad /= len(XYmat)
  return grad

def comparePredActual(pred,actual):
  #Given a Probablility Matrix and Actual values matrix
  #1 Find the max colum of the predicted - use argmax
  predColumn = np.argmax(pred, axis=1)
  actual = actual.astype(float)
  #2 Add 1 to the values in the array created by argmax
    #actuals are one column over - because of the bias
  predColumn = predColumn + 1
  predColumn = predColumn.reshape((-1,1))
  #3 Create a boolean array comparing pred and actual
  correct = (predColumn == actual)
  #4 Create a count of the correct values using the correct boolean array
  correctCount = np.sum(correct)
  #5 return both the boolean array and the count of correct values
  #Given a Probablility Matrix
  #Find the max colum of the predicted - use argmax
  return correct, correctCount

def calcConfusionMatrix(P, Y):
    # choose the indices of pred where Y==1 and see if it is pred is  equal to 1

    ##################  Code goes below #######
  TP = ((P[Y==1])==1)
  TPC = np.sum(TP)
  TN = ((P[Y==0])==0)
  TNC = np.sum(TN)
  FP = ((P[Y==0])==1)
  FPC = np.sum(FP)
  FN = ((P[Y==1])==0)
  FNC = np.sum(FN)

  precision = TPC / (TPC + FPC)
  recall = TPC / (TPC + FNC)
  accuracy = (TPC + TNC) / (TPC + TNC + FPC + FNC)
  totalP = TPC + FPC
  totalN = TNC + FNC



  f1 = (2*precision*recall)/(precision+recall)
  print("\nTrue Positives: \t", TPC,
          "\nPercent Positives correct:", f"{(TPC/totalP):.0%}")
  print("False Positives: \t", FPC,
          "\nPercent Positives Incorrect:", f"{(FPC/totalP):.0%}")
  print("True Negatives: \t", TNC,
          "\nPercent Negatives correct:", f"{(TNC/totalN):.0%}")
  print("False Negatives: \t", FNC,
          "\nPercent Negatives incorrect:", f"{(FNC/totalN):.0%}")

  print("\nPrecision: \t\t", f"{precision:.0%}")
  print("Recall:\t\t\t", f"{recall:.0%}")
  print("\nF1: \t\t\t", f"{f1:.0%}")

  print("Accuracy: \t\t", f"{accuracy:.0%}")

  return TPC, TNC, FPC, FNC, precision, recall, f1

############################################################
# Create figure objects
np.set_printoptions(precision=4, suppress=True)
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])# [left, bottom, width, height]

#1b
# =============================================================================
#  X,Y use createData method to create the X,Y matrices
# Weights - Create initial weight matrix to have the same weights as features
# Weights - should be set to 0
# =============================================================================
#
xtrain,ytrain,actual = createData()
numRows, numCols = xtrain.shape
W=np.zeros(numCols).reshape((numCols, 1))


# set learning rate - the list is if we want to try multiple LR's
# We are only doing one of them today
lrList = [0.1,.0001]
lr = lrList[0]

#set up the cost array for graphing
costArray = []
costArray.append(calcCost(xtrain, W, ytrain))

#initalize while loop flags
finished = False
count =0

while (not finished and count <10000):
    gradient = calcGradient(xtrain,ytrain,W)
    #print (gradient)
    #5 update weights
    W = W-(lr*gradient)

    costArray.append(calcCost(xtrain, W, ytrain))
    lengthOfGradientVector = np.linalg.norm(gradient)
    if (lengthOfGradientVector <= .00001):
        finished=True
    count+=1
   

print("weights: ", W, "count: ", count, "cost: ", costArray[-1])        

pred = activation(xtrain, W)
pred = (pred > 0.5).astype(int)
TPC, TNC, FPC, FNC, precision, recall, f1 = calcConfusionMatrix(pred, actual)

correctArray,correctCount = comparePredActual(pred,actual)


