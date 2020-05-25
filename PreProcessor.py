#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 08:19:07 2020

@author: niket
"""

'''
PreProcessor Version2.0
This function can:
        1 Data Pre Processing
    
    Inputs:
        x->Input Data(as numpr arrey)
        y->Output Data(as numpy arrey)
        value->One Value or a list of values whose output should be predicted(as 2D nympy arrey)
    
    Output:
        A tuple contaning pre processed (x,y,value)
    
    Loop Holes:
        1. No missing string should be there in data.
        2. y must have no missing data.
'''

def DataPreProcessor(x,y,value):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder,LabelEncoder
    from sklearn.impute import SimpleImputer
    #Data Pre-Processing
    x_dimension=x.shape #Stores the dimension of the array, (rows, column)
    value_dimension=value.shape #Stores the dimension of the array, (rows, column)
    x_str=list() #Stores index of column contaning strings.
    x_int=list() #Stroes index of column containg integers,floats.
    for _ in range(x_dimension[1]):
        if type(x[0,_]) is str:
            x_str.append(_)
        else:
            x_int.append(_)
    #Appling One Hot Encoding
    temp=np.concatenate((x, value),0)
    ohe=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),x_str)],remainder='passthrough')
    temp=ohe.fit_transform(temp)
    #Erasing missing values
    si=SimpleImputer(missing_values=np.nan ,strategy='mean')
    temp_dimension=temp.shape
    num=temp_dimension[1]-len(x_int)
    si.fit(temp[:,num:])
    temp[:,num:]=si.transform(temp[:,num:])
    x_new=temp[:x_dimension[0],:] #X gained after pre processing
    value_new=temp[:value_dimension[0]:-1,:]
    value_new=value_new[-1] #value gained after pre-processing
    #Appling Lable Encoding on y
    y_dimension=y.shape
    y_str=list()
    for _ in range(y_dimension[1]):
        if type(y[0][_]) is str:
            y_str.append(_)
    y_new=y[:,:]
    le=LabelEncoder()
    for _ in y_str:
        y_new[:,_]=le.fit_transform(y_new[:,_])
    return x_new,y_new,value_new;