#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 11:51:22 2020

@author: niket
"""

def LRegression(x,y,value,task='value'):
    '''
    LinearRegressor Version 2.0
    
    This Function can:
        1.Use Linear Regression to predict the value of given data.
        2.Pre Process y and return the invrese transformed value.
        
    Custom Library used:
        1. PreProcessor
    
    Input:
        x->Input Data(as numpr arrey)
        y->Output Data(as numpy arrey)
        value->One Value or a list of values whose output should be predicted(as 2D numpy arrey)
        task->takes either 'value' or 'precision'.
    Output:
        (task='value')A number/list contaning the predicted value.
        (task='precision')The percentage approximation applied(in range 0 to 1).
    '''
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    from PreProcessor import DataPreProcessor
    #Pre-Processing y:
    y_dimension=y.shape
    value_dimension=value.shape
    y_str=list()
    for _ in range(y_dimension[1]):
        if type(y[0][_]) is str:
            y_str.append(_)
    y_new=y[:,:]
    le=LabelEncoder()
    for _ in y_str:
        y_new[:,_]=le.fit_transform(y_new[:,_])
    #Data Pre Processing
    x_pro,y_pro,value_pro=DataPreProcessor(x, y,value) #Pre-Processed values of x,y,and value.
    #Data Processing
    lr=LinearRegression()
    lr.fit(x_pro, y_pro)
    value_pro=value_pro.reshape(1,len(value_pro))
    value_predict=lr.predict(value_pro) #This is the predicted value gained.
    if task=='precision':
        app=0 #Stores the approximation applied
        for _ in y_str:
            for __ in range(value_dimension[0]):
                app+=abs((round(value_predict[__][_])-value_predict[__][_]))
        return app          
    elif task=='value':
        value_predict=value_predict.astype(object)
        for _ in y_str:
            for __ in range(value_dimension[0]):
                value_predict[__][_]=int(round(value_predict[__][_],0))
                value_predict[__][_]=le.inverse_transform([[value_predict[__,_]]])
        temp=list(value_predict)
        value_predict=temp[0][0]
        return value_predict