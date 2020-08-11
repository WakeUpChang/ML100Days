# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 09:55:09 2020

@author: sandra_chang
"""

from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import uniform
import numpy as np

wine = datasets.load_wine()
# 切分訓練集/測試集
x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.25, random_state=4)

# 建立模型
clf = GradientBoostingRegressor(random_state=7)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(metrics.mean_squared_error(y_test, y_pred))

# 設定要訓練的超參數組合
n_estimators =  np.arange(20,200,20)
max_depth = np.arange(1,7)

#param_grid = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])
param_grid = dict(n_estimators=n_estimators, max_depth=max_depth)

## 建立搜尋物件，放入模型及參數組合字典 (n_jobs=-1 會使用全部 cpu 平行運算)
random_search = RandomizedSearchCV(clf, param_grid, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1, cv=3)

# 開始搜尋最佳參數
random_result = random_search.fit(x_train, y_train)

# 預設會跑 3-fold cross-validadtion，總共 9 種參數組合，總共要 train 27 次模型

clf_bestparam = GradientBoostingRegressor(max_depth=random_result.best_params_['max_depth'],
                                           n_estimators=random_result.best_params_['n_estimators'])

# 訓練模型
clf_bestparam.fit(x_train, y_train)

# 預測測試集
y_pred = clf_bestparam.predict(x_test)

print(metrics.mean_squared_error(y_test, y_pred))