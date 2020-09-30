# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 16:52:58 2020

@author: sandra_chang
"""

# 載入基本套件
import pandas as pd
import numpy as np

# 讀取訓練與測試資料
data_path = r'C:\Users\sandra_chang\Documents\GitHub\ML100Days\homework\data\\'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
df_train.shape

# 重組資料成為訓練 / 預測用格式
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

# 秀出資料欄位的類型與數量
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df

#確定只有 int64, float64, object 三種類型後, 分別將欄位名稱存於三個 list 中
int_features = []
float_features = []
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64':
        float_features.append(feature)
    elif dtype == 'int64':
        int_features.append(feature)
    else:
        object_features.append(feature)
print(f'{len(int_features)} Integer Features : {int_features}\n')
print(f'{len(float_features)} Float Features : {float_features}\n')
print(f'{len(object_features)} Object Features : {object_features}')

# 例 : 整數 (int) 特徵取平均 (mean)
print(df[int_features].mean())
print(df[int_features].max())
print(df[int_features].nunique())

print(df[float_features].mean())
print(df[float_features].max())
print(df[float_features].nunique())

print(df[object_features].mean())
print(df[object_features].max())
print(df[object_features].nunique())