# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 16:34:59 2018

@author: Facenomore
"""

from implementations.py import *

def pred(x,y,w):
    predict=1. * (x.dot(w) >= 0.5)
    res=sum(1. * (predict!=y))/len(y)
    return res
""" least squares GD"""
train0_tr, train0_te, trainy0_tr, trainy0_te=split_data(train0, trainy0, 0.2)
train1_tr, train1_te, trainy1_tr, trainy1_te=split_data(train1, trainy1, 0.2)
train2_tr, train2_te, trainy2_tr, trainy2_te=split_data(train2, trainy2, 0.2)


lossGD0, weightGD0=least_squares_GD(trainy0_tr, train0_tr, np.zeros(len(train0_tr[1,:])), 500, 0.03)

lossGD1, weightGD1=least_squares_GD(trainy1_tr, train1_tr, np.zeros(len(train1_tr[1,:])), 500, 0.03)

lossGD2, weightGD2=least_squares_GD(trainy2_tr, train2_tr, np.zeros(len(train2_tr[1,:])), 500, 0.03)



resultGD0=pred(train0_te, trainy0_te,weightGD0[-1])
resultGD1=pred(train1_te, trainy1_te,weightGD1[-1])
resultGD2=pred(train2_te, trainy2_te,weightGD2[-1])

"""Least squares SGD"""

lossSGD0, weightSGD0=least_squares_SGD(trainy0_tr, train0_tr, np.zeros(len(train0_tr[1,:])),1500 ,500, 0.03)

lossSGD1, weightSGD1=least_squares_SGD(trainy1_tr, train1_tr, np.zeros(len(train1_tr[1,:])),1500 ,500, 0.03)

lossSGD2, weightSGD2=least_squares_SGD(trainy2_tr, train2_tr, np.zeros(len(train2_tr[1,:])),1500 ,500, 0.03)


resultSGD0=pred(train0_te, trainy0_te,weightSGD0[-1])
resultSGD1=pred(train1_te, trainy1_te,weightSGD1[-1])
resultSGD2=pred(train2_te, trainy2_te,weightSGD2[-1])

"""least squares"""

weightLS0=least_squares(trainy0_tr,train0_tr)
weightLS1=least_squares(trainy1_tr,train1_tr)
weightLS2=least_squares(trainy2_tr,train1_tr)

resultLS0=pred(train0_te, trainy0_te,weightLS0[-1])
resultLS1=pred(train1_te, trainy1_te,weightLS1[-1])
resultLS2=pred(train2_te, trainy2_te,weightLS2[-1])




"""Ridge reg"""

weightRig0=ridge_regression(trainy0_tr, train0_tr, 1)

weightRig1=ridge_regression(trainy1_tr, train1_tr, 1)

weightRig2=ridge_regression(trainy2_tr, train2_tr, 1)

resultRig0=pred(train0_te, trainy0_te,weightRig0)
resultRig1=pred(train1_te, trainy1_te,weightRig1)
resultRig2=pred(train2_te, trainy2_te,weightRig2)


"""Logistic regression"""


lossLR0,weightLR0=logistic_regression(trainy0_tr, train0_tr, np.zeros(len(train0_tr[1,:])), 500, 0.03)

lossLR1,weightLR1=logistic_regression(trainy1_tr, train1_tr, np.zeros(len(train1_tr[1,:])), 500, 0.03)

lossLR2,weightLR2=logistic_regression(trainy2_tr, train2_tr, np.zeros(len(train2_tr[1,:])), 500, 0.03)

resultLR0=pred(train0_te, trainy0_te,weightLR0)
resultLR1=pred(train1_te, trainy1_te,weightLR1)
resultLR2=pred(train2_te, trainy2_te,weightLR2)


""" Regu Logistic Regression"""

lossRLR0,weightRLR0=reg_logistic_regression(trainy0_tr, train0_tr,0.1, np.zeros(len(train0_tr[1,:])), 500, 0.03)

lossRLR1,weightRLR1=reg_logistic_regression(trainy1_tr, train1_tr,0.1, np.zeros(len(train1_tr[1,:])), 500, 0.03)

lossRLR2,weightRLR2=reg_logistic_regression(trainy2_tr, train2_tr,0.1, np.zeros(len(train2_tr[1,:])), 500, 0.03)

resultRLR0=pred(train0_te, trainy0_te,weightRLR0)
resultRLR1=pred(train1_te, trainy1_te,weightRLR1)
resultRLR2=pred(train2_te, trainy2_te,weightRLR2)

"""test du model de R"""

trainlm=extend_and_standardize(trainlm)

lossSGDboss, weightSGDboss=least_squares_SGD(train[0], trainlm, np.zeros(len(trainlm[1,:])),1500 ,500, 0.03)

resultSGDboss=pred(trainlm, train[0],weightSGDboss[-1])

