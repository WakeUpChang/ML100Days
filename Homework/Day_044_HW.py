# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 17:33:29 2020

@author: sandra_chang
"""

from sklearn import datasets, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

wineData = datasets.load_wine()

x_train, x_test, y_train, y_test = train_test_split(wineData.data, wineData.target, test_size=0.2, random_state=4)

RFC = RandomForestClassifier(n_estimators = 5, max_depth = 2)

RFC.fit(x_train, y_train)

y_pred = RFC.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)

print(wineData.feature_names)

print("Feature importance: ", RFC.feature_importances_)