# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:56:14 2020

@author: sandra_chang
"""

from sklearn import datasets, metrics

# 如果是分類問題，請使用 DecisionTreeClassifier，若為回歸問題，請使用 DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model

import warnings
warnings.filterwarnings('ignore')

wineData = datasets.load_wine()

x_train, x_test, y_train, y_test = train_test_split(wineData.data, wineData.target, test_size=0.2, random_state=4)

DTC = DecisionTreeClassifier()

DTC.fit(x_train,y_train)

y_pred = DTC.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Decision Tree Acuuracy: ", acc)


DTC = DecisionTreeClassifier(max_depth = 3,criterion = "entropy")

DTC.fit(x_train,y_train)

y_pred = DTC.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Decision Tree 2 Acuuracy: ", acc)


#print(wineData.feature_names)
#
#print("Feature importance: ", DTC.feature_importances_)


x_train, x_test, y_train, y_test = train_test_split(wineData.data, wineData.target, test_size=0.2, random_state=4)

LRModel = linear_model.LogisticRegression()

LRModel.fit(x_train, y_train)

y_pred = LRModel.predict(x_test)

#print(LRModel.coef_)

print("Logistic Regression Acuuracy: " , metrics.accuracy_score(y_test, y_pred))
