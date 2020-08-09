# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 11:02:34 2020

@author: sandra_chang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

wineData = datasets.load_wine()

x_train, x_test, y_train, y_test = train_test_split(wineData.data, wineData.target, test_size=0.2, random_state=4)

LRModel = linear_model.LogisticRegression()

LRModel.fit(x_train, y_train)

y_pred = LRModel.predict(x_test)

print(LRModel.coef_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

LassoModel = linear_model.Lasso(alpha = 1)

LassoModel.fit(x_train, y_train)

y_pred_Lasso = LassoModel.predict(x_test)

print(("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_Lasso)))

RidgeModel = linear_model.Ridge(alpha = 1)

RidgeModel.fit(x_train, y_train)

y_pred_Ridge = RidgeModel.predict(x_test)

print(("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred_Ridge)))


