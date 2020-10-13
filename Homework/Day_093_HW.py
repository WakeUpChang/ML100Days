#!/usr/bin/env python
# coding: utf-8

# ## 範例重點
# * 學習如何在 keras 中撰寫自定義的 loss function
# * 知道如何在訓練時使用自定義的 loss function

# In[1]:


import os
import keras

# 本範例不需使用 GPU, 將 GPU 設定為 "無"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# In[2]:


train, test = keras.datasets.cifar10.load_data()


# In[3]:


## 資料前處理
def preproc_x(x, flatten=True):
    x = x / 255.
    if flatten:
        x = x.reshape((len(x), -1))
    return x

def preproc_y(y, num_classes=10):
    if y.shape[-1] == 1:
        y = keras.utils.to_categorical(y, num_classes)
    return y    


# In[4]:


x_train, y_train = train
x_test, y_test = test

# 資料前處理 - X 標準化
x_train = preproc_x(x_train)
x_test = preproc_x(x_test)

# 資料前處理 -Y 轉成 onehot
y_train = preproc_y(y_train)
y_test = preproc_y(y_test)


# In[5]:


from keras.layers import BatchNormalization

"""
建立神經網路，並加入 BN layer
"""
def build_mlp(input_shape, output_units=10, num_neurons=[512, 256, 128]):
    input_layer = keras.layers.Input(input_shape)
    
    for i, n_units in enumerate(num_neurons):
        if i == 0:
            x = keras.layers.Dense(units=n_units, 
                                   activation="relu", 
                                   name="hidden_layer"+str(i+1))(input_layer)
            x = BatchNormalization()(x)
        else:
            x = keras.layers.Dense(units=n_units, 
                                   activation="relu", 
                                   name="hidden_layer"+str(i+1))(x)
            x = BatchNormalization()(x)
    
    out = keras.layers.Dense(units=output_units, activation="softmax", name="output")(x)
    
    model = keras.models.Model(inputs=[input_layer], outputs=[out])
    return model


# In[6]:


## 超參數設定
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 1024
MOMENTUM = 0.95


keras.backend.clear_session()
model = build_mlp(input_shape=x_train.shape[1:])
model.summary()


# In[]:

from keras import layers
from keras import models
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

#確認keras 版本
print(keras.__version__)

keras.backend.clear_session()


# In[3]:


#建立一個序列模型
model = models.Sequential()
#建立一個卷績層, 32 個內核, 內核大小 3x3, 
#輸入影像大小 28x28x1
model.add(layers.Conv2D(32, (3, 3), input_shape=(28, 28, 3)))

#新增一池化層, 採用maxpooling
model.add(MaxPooling2D(2,2))

#建立第二個卷績層, 池化層, 
#請注意, 不需要再輸入 input_shape
model.add(layers.Conv2D(25, (3, 3)))
model.add(MaxPooling2D(2,2))

#新增平坦層
model.add(Flatten())

#建立一個全連接層
model.add(Dense(units=100))
model.add(Activation('relu'))

#建立一個輸出層, 並採用softmax
model.add(Dense(units=10))
model.add(Activation('softmax'))

# 網路模型
# ![Layers.png](attachment:Layers.png)

#輸出模型的堆疊
model.summary()

