# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:34:57 2018

@author: Facenomore
"""

import numpy as np

"""classify the data set in 3 differents classes"""

def classif(X):
    X0=X[1][(X[1][:,22]==0),:]
    X1=X[1][(X[1][:,22]==1),:]
    X2=X[1][(X[1][:,22]>1),:]
    Y0=X[0][(X[1][:,22]==0)]
    Y1=X[0][(X[1][:,22]==1)]
    Y2=X[0][(X[1][:,22]>1)]
     
    X0=delcoll(X0)
    X1=delcoll(X1)
    X2=delcoll(X2)
    return X0, Y0, X1, Y1, X2, Y2



def delcoll(X):
    delcol=[None]*len(X[1,:])
    for i in range(len(X[1,:])):
        delcol[i]= (-999 in X[:,i]) 
    delcol=np.array(delcol)
    type(delcol)
    X = X[:,~delcol]
    return X

