# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 14:34:57 2018

@author: Facenomore
"""

import numpy as np

"""classify the data set in 3 differents classes"""

def classif(X):
    X0=X[(X[:,22]==0),:]
    X1=X[(X[:,22]==1),:]
    X2=X[(X[:,22]>1),:]
    
    X0=delcoll(X0)
    X1=delcoll(X1)
    X2=delcoll(X2)
    return X0, X1, X2



def delcoll(X):
    delcol=[None]*len(X[1,:])
    for i in range(len(X[1,:])):
        delcol[i]= (-999 in X[:,i]) 
    delcol=np.array(delcol)
    type(delcol)
    X = X[:,~delcol]
    return X

