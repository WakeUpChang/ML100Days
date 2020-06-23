# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 21:00:14 2020

@author: sandra_chang
"""

import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data_path = data_path = r'C:\Users\sandra_chang\Documents\GitHub\ML100Days\Homework\data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

df = df[num_features]
train_num = train_Y.shape[0]
df.head()

df_m1 = df.fillna(-1)
train_X = df_m1[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_mn = df.fillna(df.mean())
train_X = df_mn[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())


df_0 = df.fillna(0)
train_X = df_0[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())


df_m1 = df.fillna(-1)
df_temp = MinMaxScaler().fit_transform(df_m1)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_mn = df.fillna(df.mean())
df_temp = MinMaxScaler().fit_transform(df_mn)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_0 = df.fillna(0)
df_temp = MinMaxScaler().fit_transform(df_0)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())



df_m1 = df.fillna(-1)
df_temp = StandardScaler().fit_transform(df_m1)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_mn = df.fillna(df.mean())
df_temp = StandardScaler().fit_transform(df_mn)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

df_0 = df.fillna(0)
df_temp = StandardScaler().fit_transform(df_0)
train_X = df_temp[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())