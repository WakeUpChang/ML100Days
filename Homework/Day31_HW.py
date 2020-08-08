# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:15:42 2020

@author: sandra_chang
"""

# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

data_path = 'data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()

# 因為需要把類別型與數值型特徵都加入, 故使用最簡版的特徵工程
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))

# 隨機森林擬合後, 將結果依照重要性由高到低排序
estimator = RandomForestClassifier()
estimator.fit(df.values, train_Y)
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)

# 高重要性特徵 + 隨機森林 (39大約是79的一半)
high_feature = list(feats[:int(feats.size/2)].index)
train_X = MMEncoder.fit_transform(df[high_feature])
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())


# 製作新特徵看效果
df['Add_char'] = (df[high_feature[0]] + df[high_feature[1]]) / 2
df['Multi_char'] = df[high_feature[0]] * df[high_feature[1]]
df['GO_div1p'] = df[high_feature[0]] / (df[high_feature[1]]+1) * 2
df['OG_div1p'] = df[high_feature[0]] / (df[high_feature[1]]+1) * 2

train_X = MMEncoder.fit_transform(df)
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())