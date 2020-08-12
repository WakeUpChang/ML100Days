# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 22:15:49 2020

@author: admin
"""

# 做完特徵工程前的所有準備
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
# 因為擬合(fit)與編碼(transform)需要分開, 因此不使用.get_dummy, 而採用 sklearn 的 OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics 


data_path = 'data/data-science-london-scikit-learn/'
test = pd.read_csv(data_path + 'test.csv',header=None)
train = pd.read_csv(data_path + 'train.csv',header=None)
trainLabels = pd.read_csv(data_path + 'trainLabels.csv',header=None)

train_X, test_X, train_Y, test_Y = train_test_split(train, trainLabels, test_size=0.5)
train_X, val_X, train_Y, val_Y = train_test_split(train, trainLabels, test_size=0.5)

gdbt = GradientBoostingClassifier(subsample=0.93, n_estimators=320, min_samples_split=0.1, min_samples_leaf=0.3, 
                                  max_features=4, max_depth=4, learning_rate=0.16)

onehot = OneHotEncoder()
lr = LogisticRegression(solver='lbfgs', max_iter=1000)

gdbt.fit(train_X, train_Y)
pred_gdbt = gdbt.predict(test_X)


acc = metrics.accuracy_score(test_Y, pred_gdbt)
print("Accuracy gdbt: ", acc)


# pred_gdbt_test = gdbt.predict(test)


# df = pd.DataFrame({'Solution':pred_gdbt_test})
# df.to_csv('trytry.csv', index = True)