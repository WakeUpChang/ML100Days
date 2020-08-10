# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:24:37 2020

@author: sandra_chang
"""

from sklearn import datasets, metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

x_train, x_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.25,random_state=4)

clf = GradientBoostingClassifier()

# 訓練模型
clf.fit(x_train, y_train)

# 預測測試集
y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Acuuracy: ", acc)