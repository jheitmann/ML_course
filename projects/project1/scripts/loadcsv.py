# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 11:01:31 2018

@author: Facenomore
"""
import numpy as np
from proj1_helpers import load_csv_data

train=load_csv_data(r"C:\Users\Facenomore\Documents\Maths\Master_3eme_sem\ML\project\train.csv")

"Création des variable de travail et de nom en enlevant les colonnes avec des valeurs -999"
delcol=[None]*len(train[1][1,:])
for i in range(len(train[1][1,:])):
    delcol[i]= (-999 in train[1][:,i]) 
delcol=np.array(delcol)


trainvar=train[1][:,~delcol]
trainname=train[3][~delcol]

