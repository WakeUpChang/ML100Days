# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 18:45:08 2020

@author: sandra_chang
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dir_data =  r'C:\Users\sandra_chang\Documents\GitHub\ML100Days\Homework\data/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
print(app_train.head())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder, 以加入相關係數檢查
            app_train[col] = le.fit_transform(app_train[col])            
print(app_train.shape)
print(app_train.head())

# 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

# 出生日數 (DAYS_BIRTH) 取絕對值 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])

print(app_train.corr()['TARGET'])

sort_corr = app_train.corr()['TARGET'].sort_values()

plt.scatter(app_train['EXT_SOURCE_3'], app_train['TARGET'])

app_train.boxplot(by='TARGET', column='EXT_SOURCE_3')

