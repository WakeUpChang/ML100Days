# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:03:15 2020

@author: sandra_chang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

def model(testData, tragetData):
    x_train, x_test, y_train, y_test = train_test_split(testData, tragetData, test_size=0.1, random_state=4)

    # 建立模型
    logreg = linear_model.LogisticRegression()
    
    # 訓練模型
    logreg.fit(x_train, y_train)
    
    # 預測測試集
    y_pred = logreg.predict(x_test)
    
    acc = accuracy_score(y_test, y_pred)
    return acc


wine = datasets.load_wine()
boston = datasets.load_boston()
breast_cancer = datasets.load_breast_cancer()

acc_wine = model(wine.data, wine.target)
print("Wine Accuracy: ", acc_wine)

#acc_boston = model(boston.data, boston.target)
#print("boston Accuracy: ", acc_boston)

acc_breast_cancer = model(breast_cancer.data, breast_cancer.target)
print("breast_cancer Accuracy: ", acc_breast_cancer)