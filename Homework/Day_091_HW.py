#!/usr/bin/env python
# coding: utf-8

# ## 作業目標
# * 嘗試比較用 color histogram 和 HOG 特徵來訓練的 SVM 分類器在 cifar10 training 和 testing data 上準確度的差別

# In[ ]:


import os
import keras
os.environ["CUDA_VISIBLE_DEVICES"] = "" # 使用 CPU

import numpy as np
import cv2 # 載入 cv2 套件
import matplotlib.pyplot as plt

train, test = keras.datasets.cifar10.load_data()


# In[ ]:


x_train, y_train = train
x_test, y_test = test
y_train = y_train.astype(int)
y_test = y_test.astype(int)


# #### 產生直方圖特徵的訓練資料

# In[ ]:


x_train_histogram = []
x_test_histogram = []

# 對於所有訓練資料
for i in range(len(x_train)):
    chans = cv2.split(x_train[i]) # 把圖像的 3 個 channel 切分出來
    # 對於所有 channel
    hist_feature = []
    for chan in chans:
        # 計算該 channel 的直方圖
        hist = cv2.calcHist([chan], [0], None, [16], [0, 256]) # 切成 16 個 bin
        hist_feature.extend(hist.flatten())
    # 把計算的直方圖特徵收集起來
    x_train_histogram.append(hist_feature)

# 對於所有測試資料也做一樣的處理
for i in range(len(x_test)):
    chans = cv2.split(x_test[i]) # 把圖像的 3 個 channel 切分出來
    # 對於所有 channel
    hist_feature = []
    for chan in chans:
        # 計算該 channel 的直方圖
        hist = cv2.calcHist([chan], [0], None, [16], [0, 256]) # 切成 16 個 bin
        hist_feature.extend(hist.flatten())
    x_test_histogram.append(hist_feature)

x_train_histogram = np.array(x_train_histogram)
x_test_histogram = np.array(x_test_histogram)


# #### 產生 HOG 特徵的訓練資料
# * HOG 特徵通過計算和統計圖像局部區域的梯度方向直方圖來構建特徵，具體細節不在我們涵蓋的範圍裡面，有興趣的同學請參考[補充資料](https://www.cnblogs.com/zyly/p/9651261.html)哦

# In[ ]:


# SZ=20
bin_n = 16 # Number of bins

def hog(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)     # hist is a 64 bit vector
    return hist.astype(np.float32)

x_train_hog = np.array([hog(x) for x in x_train])
x_test_hog = np.array([hog(x) for x in x_test])


# #### SVM model
# * SVM 是機器學習中一個經典的分類算法，具體細節有興趣可以參考 [該知乎上的解釋](https://www.zhihu.com/question/21094489)，我們這裡直接調用 opencv 中實現好的函數

# #### 用 histogram 特徵訓練 SVM 模型
# * 訓練過程可能會花點時間，請等他一下

# In[ ]:


SVM_hist = cv2.ml.SVM_create()
SVM_hist.setKernel(cv2.ml.SVM_LINEAR)
SVM_hist.setGamma(5.383)
SVM_hist.setType(cv2.ml.SVM_C_SVC)
SVM_hist.setC(2.67)

#training
SVM_hist.train(x_train_histogram, cv2.ml.ROW_SAMPLE, y_train)

# prediction
_, y_hist_train = SVM_hist.predict(x_train_histogram)
_, y_hist_test = SVM_hist.predict(x_test_histogram)


# #### 用 HOG 特徵訓練 SVM 模型
# * 訓練過程可能會花點時間，請等他一下

# In[ ]:


SVM_hog = cv2.ml.SVM_create()
SVM_hog.setKernel(cv2.ml.SVM_LINEAR)
SVM_hog.setGamma(5.383)
SVM_hog.setType(cv2.ml.SVM_C_SVC)
SVM_hog.setC(2.67)

#training
SVM_hog.train(x_train_hog, cv2.ml.ROW_SAMPLE, y_train)

# prediction
_, y_hog_train = SVM_hog.predict(x_train_hog)
_, y_hog_test = SVM_hog.predict(x_test_hog)


# In[ ]:
from sklearn import metrics
hog_accuracy = metrics.accuracy_score(y_test, y_hog_test)
hist_accuracy = metrics.accuracy_score(y_test, y_hist_test)