# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:51:39 2020

@author: sandra_chang
"""

# Import 需要的套件
import os
import numpy as np
import pandas as pd

# 設定 data_path

dir_data = r'C:\Users\sandra_chang\Documents\GitHub\ML100Days\homework\data\\'

f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)

#dtype_df = app_train.dtypes.reset_index()
#dtype_df.columns = ["Count", "Column Type"]
#dtype_df = dtype_df.groupby("Column Type").aggregate('count').reset_index()
#
#
#int_features = []
#float_features = []
#object_features = []
#for dtype, feature in zip(app_train.dtypes, app_train.columns):
#    if dtype == 'float64':
#        float_features.append(feature)
#    elif dtype == 'int64':
#        int_features.append(feature)
#    else:
#        object_features.append(feature)
#        
#print(f'{len(int_features)} Integer Features : {int_features}\n')
#print(f'{len(float_features)} Float Features : {float_features}\n')
#print(f'{len(object_features)} Object Features : {object_features}')

import matplotlib.pyplot as plt

AMT_Income = app_train["DAYS_BIRTH"]
std = AMT_Income.std()
mean = AMT_Income.mean()

#s = pd.Series(AMT_Income)
#s.hist(bins = int(AMT_Income.max()/1000000)) 
n, bins, patches = plt.hist(AMT_Income, 100)
plt.show()