# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:50:51 2020

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

cut_rule = [-1,0,2,5,np.inf]

app_train['CNT_CHILDREN_GROUP'] = pd.cut(app_train['CNT_CHILDREN'].values, cut_rule, include_lowest=False)
print(app_train['CNT_CHILDREN_GROUP'].value_counts())


grp = 'CNT_CHILDREN_GROUP'

grouped_df = app_train.groupby(grp)['AMT_INCOME_TOTAL']
print(grouped_df.mean())

plt_column = 'AMT_INCOME_TOTAL'
plt_by =[ 'CNT_CHILDREN_GROUP','TARGET']

app_train.boxplot(column=plt_column, by = plt_by, showfliers = False, figsize=(12,12))
plt.show()

app_train['AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET'] = grouped_df.apply(lambda x: (x-x.mean())/x.std())

app_train[['AMT_INCOME_TOTAL','AMT_INCOME_TOTAL_Z_BY_CHILDREN_GRP-TARGET']].head()