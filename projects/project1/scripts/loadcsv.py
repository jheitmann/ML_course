# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:01:31 2018

@author: Facenomore
"""
import numpy as np
from proj1_helpers import load_csv_data

def extend_and_standardize(input_data):
      x, mean_x, std_x = standardize(input_data)
      tx = build_model_data(x)
      return tx

train=load_csv_data(r"C:\Users\Facenomore\Documents\Maths\Master_3eme_sem\ML\project\train.csv")

"""Création des variable de travail et de nom en enlevant les colonnes avec des valeurs -999"""


trainvar=train[1][:,~delcol]
trainname=train[3]
trainvartes=extend_and_standardize(trainvar)

"""construction of linear regression variables"""
itemindex1=[0,1,2,3,4,5,6,7,8,9,11,12,14,16,17,18]
itemindex2=[[0,1],[0,2],[0,3],[0,4],[0,6],[0,8],[0,9],[0,11],[0,14],[0,16],
            [1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,14],[1,16],[1,17],[1,18],
            [2,3],[2,5],[2,6],[2,14],[2,17],[2,18],
            [3,5],[3,6], [3,7], [3,14], [3,16], [3,17], [3,18],
            [4,7],[4,8],[4,17],[4,18],  
            [5,8],[5,11],[5,14],[5,17],
            [6,14],[6,16],
            [7,8],[7,14],[7,17],[7,18],
            [8,18],[9,11],[9,12],
            [11,16],[11,17],[11,18],
            [14,18],[16,17],[17,18]   ]

long=len(itemindex1)+len(itemindex2)
long2=len(itemindex1)


trainlm=np.zeros((250000,long))

for i in range(long): 
    trainlm[:,i]=trainvar[:,itemindex1[i]]
    
for i in range(len(itemindex2)):
    trainlm[:,i+long2]=trainvar[:,itemindex2[i][0]]*trainvar[:,itemindex2[i][1]]

