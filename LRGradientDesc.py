# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 09:00:52 2020

@author: Santhanakrishnan
"""
import pandas as pd
import numpy as np
import math as mt

#Choose these as per requirements
lrAlpha= 0.05
threshold=0.01
max_steps=10000

#Create X and Y which are input and target variables respectively. Encode and scale them. Without Scaling, unexpected results are possible. A sample shown in comments

"""
dataset = pd.read_csv('inFile.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

from sklearn import preprocessing
X=preprocessing.scale(X)
"""


def costFunction(y,y_pred):
    res=(y-y_pred)
    restranspose=res.transpose()
    return mt.sqrt(np.dot(restranspose,res))

def residuals(y,y_pred):
    res=y-y_pred
    return res

def delta(y, y_pred, feature, residuals):
    featuretranspose=feature.transpose()
    #residualtranspose=residuals.transpose()
    dP=np.dot(featuretranspose,residuals)
    #dP=1
    dlt=2*lrAlpha*dP
    
    
    return dlt

def getFeatures(X):
    return X.shape[1]

def addConstant(X):
    return np.insert(X,0,1.0,axis=1)

def pred(coeff,X):
    return np.dot(X,coeff)

def gradDesc(X,Y):
    X=addConstant(X)
    nFeatures=getFeatures(X)
    dltCheck=np.full(nFeatures, True)
    coeffC=np.ones(nFeatures)
    
    for j in range(max_steps):
        if(not any(dltCheck)):
            break
        else:
            y_pred=pred(coeffC,X)
            residuals=y-y_pred
            for i in range(nFeatures):
                if(dltCheck[i]):
                    dlt = delta(y,y_pred, X[:,i],residuals)
                    dltn=dlt
                    if(dlt<0):
                        dltn=dlt*-1
                    #print(dlt)
                    if(dltn<threshold):
                        dltCheck[i]=False
                    else:
                        coeffC[i]=coeffC[i]+dlt
    print(j,i)
            
    return y_pred, coeffC
    
    
y_pred,coeddC=gradDesc(X,y)

print(costFunction(y,y_pred))
