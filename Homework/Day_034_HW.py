# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:06:17 2020

@author: sandra_chang
"""
from sklearn.model_selection import train_test_split, KFold
import numpy as np
X = np.arange(1000).reshape(200, 5)
y = np.zeros(200)
y[:40] = 1

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[:40], y[:40], test_size=0.25, random_state=10)
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[40:200], y[40:200], test_size=1/16, random_state=10)

X_train = np.append( X_train_1, X_train_0).reshape(180, 5)
y_train = np.append( y_train_1, y_train_0)
X_test = np.append( X_test_1, X_test_0).reshape(20, 5)
y_test = np.append( y_test_1, y_test_0)