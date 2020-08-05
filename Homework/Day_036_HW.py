# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:33:39 2020

@author: sandra_chang
"""
from sklearn import metrics
import numpy as np
y_pred = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 prediction
y_true = np.random.randint(2, size=100)  # 生成 100 個隨機的 0 / 1 ground truth

precision = metrics.precision_score(y_pred, y_true) # 使用 Precision 評估
recall  = metrics.recall_score(y_pred, y_true) # 使用 recall 評估

Fb = (26)*(precision*recall)/(25*precision+recall)
