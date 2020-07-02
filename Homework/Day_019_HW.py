# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:29:38 2020

@author: sandra_chang
"""

# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

# 忽略警告訊息
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data =  r'C:\Users\sandra_chang\Documents\GitHub\ML100Days\Homework\data/'

# 讀取檔案
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()

unique_house_type = np.sort(app_train['HOUSETYPE_MODE'].fillna('nan').unique())

nrows = len(unique_house_type)
ncols = nrows // 2
#
plt.figure(figsize=(10,30))
for i in range(len(unique_house_type)):
    plt.subplot(nrows, ncols, i+1)

    app_train.loc[(app_train['HOUSETYPE_MODE'] == unique_house_type[i])  , 'HOUSETYPE_MODE'].hist()
    
    plt.title(str(unique_house_type[i]))
plt.show()    