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

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X[:40], y[:40], test_size=10, random_state=10)
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X[40:200], y[40:200], test_size=10, random_state=10)

x_train, y_train = np.concatenate([X_train_1, X_train_0]), np.concatenate([y_train_1, y_train_0])
x_test, y_test = np.concatenate([X_test_1, X_test_0]), np.concatenate([y_test_1, y_test_0])