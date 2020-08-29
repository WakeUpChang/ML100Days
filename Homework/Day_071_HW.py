# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 14:40:18 2020

@author: sandra_chang
"""

import keras
from keras.datasets import cifar10
import numpy as np
np.random.seed(10)
#np.random.seed(10)的作用：使得隨機數據可預測

#取得Keras CIFAR10 Dataset, 並分成Training 與 Test set
(x_img_train,y_label_train),(x_img_test,y_label_test)=cifar10.load_data()

#確認 CIFAR10 Dataset 資料維度
print("train data:",'images:',x_img_train.shape, " labels:",y_label_train.shape) 
print("test  data:",'images:',x_img_test.shape , " labels:",y_label_test.shape) 

#資料正規化, 並設定 data array 為浮點數
x_img_train_normalize = x_img_train.astype('float32') / 255.0
x_img_test_normalize = x_img_test.astype('float32') / 255.0

#針對Label 做 ONE HOT ENCODE, 並查看維度資訊
from keras.utils import np_utils
y_label_train_OneHot = np_utils.to_categorical(y_label_train)
y_label_test_OneHot = np_utils.to_categorical(y_label_test)
y_label_test_OneHot.shape

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

import matplotlib.pyplot as plt

def show_train_history(train_history,lossFcn,train_acc,test_acc):
    plt.Figure()
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title(lossFcn)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train_acc', 'val_acc'], loc='upper left')
    plt.show()

def trainModel(lossFcn):

    # 宣告採用序列模型
    model = Sequential()
    
    model.add(Conv2D(filters=32,kernel_size=(3,3),
                     input_shape=(32, 32,3), 
                     activation='relu', 
                     padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=8, kernel_size=(3, 3), 
                     activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    #建立全網路連接層
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    
    
    #建立輸出層
    model.add(Dense(10, activation='softmax'))
    #檢查model 的STACK
    print(model.summary())

    #模型編譯
    model.compile(loss=lossFcn, optimizer='Adam', metrics=['accuracy'])

    #模型訓練, "Train_History" 把訓練過程所得到的數值存起來
    train_history=model.fit(x_img_train_normalize, y_label_train_OneHot,
                            validation_split=0.25,
                            epochs=3, batch_size=10, verbose=1)
    
    show_train_history(train_history,lossFcn,'acc','val_acc')
    
    return train_history
    
LossFunctions= {"MSE","binary_crossentropy","categorical_crossentropy"}

train_history = []
for LF in LossFunctions:
    train_history.append(trainModel(LF))
    
count = int(0)
for LF in LossFunctions:
    print("%s acc: %f" %(LF ,train_history[count].history['val_acc'][2]))
    count+=1




    