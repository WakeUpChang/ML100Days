# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 21:32:02 2020

@author: sandra_chang
"""

# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

data_path = 'data/'
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()

df['Sex'] = df['Sex'].fillna('None')
mean_df = df.groupby(['Sex'])['Age'].mean().reset_index()
mode_df = df.groupby(['Sex'])['Age'].apply(lambda x: x.mode()[0]).reset_index()
median_df = df.groupby(['Sex'])['Age'].median().reset_index()
max_df = df.groupby(['Sex'])['Age'].max().reset_index()
temp = pd.merge(mean_df, mode_df, how='left', on=['Sex'])
temp = pd.merge(temp, median_df, how='left', on=['Sex'])
temp = pd.merge(temp, max_df, how='left', on=['Sex'])
temp.columns = ['Sex', 'Sex_Age_Mean', 'Sex_Age_Mode', 'Sex_Age_Median', 'Sex_Age_Max']

df = pd.merge(df, temp, how='left', on=['Sex'])
df = df.drop(['Sex'] , axis=1)
df.head()

#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(-1)
MMEncoder = MinMaxScaler()
df.head()


df_minus = df.drop([ 'Sex_Age_Mean', 'Sex_Age_Mode', 'Sex_Age_Median', 'Sex_Age_Max'] , axis=1)

# 原始特徵 + 線性迴歸
train_X = MMEncoder.fit_transform(df_minus)
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

# 新特徵 + 線性迴歸 : 有些微改善
train_X = MMEncoder.fit_transform(df)
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())
