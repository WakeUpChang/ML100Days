# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 21:29:11 2020

@author: sandra_chang
"""

# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

plt.style.use('ggplot')

# 忽略警告訊息
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data = r'C:\Users\sandra_chang\Documents\GitHub\ML100Days\Homework\data/'

f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

discreteData = pd.cut(app_train["AMT_INCOME_TOTAL"],bins= np.linspace(0,app_train["AMT_INCOME_TOTAL"].max()+1,100))

print(discreteData.value_counts())

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#
## 檢查每一個 column
#for col in app_train:
#    if app_train[col].dtype == 'object':
#        # 如果只有兩種值的類別型欄位
#        if len(list(app_train[col].unique())) <= 2:
#            # 就做 Label Encoder, 以加入相關係數檢查
#            app_train[col] = le.fit_transform(app_train[col])            
#print(app_train.shape)
#app_train.head()
#
## 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
#app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
#app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
#
## 出生日數 (DAYS_BIRTH) 取絕對值 
#app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
#

