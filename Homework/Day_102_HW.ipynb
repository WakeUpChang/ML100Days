# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:58:31 2020

@author: admin
"""
import os
import glob
import shutil
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import numpy as np
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.layers import Activation,BatchNormalization

# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import LearningRateScheduler
# from tensorflow.keras.callbacks import ReduceLROnPlateau

base_dir =  r"C:\Users\admin\Documents\GitHub\ML100Days\Homework\data\image_data\train"
num_classes = 5
classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


for cl in classes:
  img_path = os.path.join(base_dir, cl)                          # 取得單一類別資料夾路徑
  images = glob.glob(img_path + '/*.jpg')                        # 載入所有 jpg 檔成為一個 list
  print("{}: {} Images".format(cl, len(images)))                 # 印出單一類別有幾張圖片
  num_train = int(round(len(images)*0.8))                        # 切割 80% 資料作為訓練集
  train, val = images[:num_train], images[num_train:]            # 訓練 > 0~80%，驗證 > 80%~100%

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):  # 如果資料夾不存在
      os.makedirs(os.path.join(base_dir, 'train', cl))           # 建立新資料夾
    shutil.move(t, os.path.join(base_dir, 'train', cl))          # 搬運圖片資料到新的資料夾

  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):    # 如果資料夾不存在
      os.makedirs(os.path.join(base_dir, 'val', cl))             # 建立新資料夾
    shutil.move(v, os.path.join(base_dir, 'val', cl))            # 搬運圖片資料到新的資料夾
    
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')

batch_size = 4
IMG_SHAPE =150

image_gen = ImageDataGenerator(
    rescale=1./255,                    # 壓縮資料到 0~1 之間
    horizontal_flip=True               # 隨機反轉圖片資料
)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,             # batch_size=100
    directory=train_dir,               # 指定訓練集的資料夾路徑
    shuffle=True,                      # 洗牌，打亂圖片排序
    target_size=(IMG_SHAPE,IMG_SHAPE)  # 圖片大小統一為 150 * 150 pixel
)

image_gen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=45  # 隨機旋轉圖片 ±45°
)

image_gen = ImageDataGenerator(
    rescale=1./255, 
    zoom_range=0.5  # 縮小或放大 50%，
)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE)
)

train_data_gen = image_gen.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE, IMG_SHAPE)
)

image_gen_train = ImageDataGenerator(
    rescale=1./255,               # 從0~255整數，壓縮為0~1浮點數
    rotation_range=45,            # 隨機旋轉 ±45°
    width_shift_range=.15,        # 隨機水平移動 ±15%
    height_shift_range=.15,       # 隨機垂直移動 ±15%
    horizontal_flip=True,         # 隨機水平翻轉
    zoom_range=0.5                # 隨機縮放 50%
)


train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_SHAPE,IMG_SHAPE),
    class_mode='sparse'           # 分類標籤定義為 0, 1, 2, 3, 4
)

image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    target_size=(IMG_SHAPE, IMG_SHAPE),
    class_mode='sparse'
)

from keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from keras import applications
from keras.preprocessing.image import ImageDataGenerator 
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Concatenate, GlobalMaxPooling2D
from keras.callbacks import ReduceLROnPlateau, LambdaCallback, ModelCheckpoint, LearningRateScheduler

# 訓練用的超參數

input_shape=(IMG_SHAPE, IMG_SHAPE, 3)

# 架構主要 model

main_model = Sequential()

model = applications.resnet_v2.ResNet50V2(include_top=False, pooling='avg', weights='imagenet', input_shape=input_shape)
main_model.add(model)

# 想辦法讓訓練不要這麼飄
# main_model.add(BatchNormalization())
# main_model.add(Dense(2048, activation='relu'))
# main_model.add(BatchNormalization())
# main_model.add(Dense(1024, activation='relu'))
# main_model.add(Dropout(0.5))
# main_model.add(Flatten())
main_model.add(Dense(num_classes, activation='softmax'))
main_model.summary()

# 因為是遷移學習，本來就有訓練了，縮小 learning rate，才不會讓訓練好的成果跑掉
opt = optimizers.Adam(lr=1e-4)
main_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# 檢查是否凍結 ALL
print("--檢查 ALL 是否凍結-----------------")
for layer in main_model.layers:
    print(layer.name, ' is trainable? ', layer.trainable)

main_model.summary()


# 訓練模型囉！
batch_size=1

# checkpoint 儲存最好的一個

# shutil.rmtree('best_loss', ignore_errors=True)
if not os.path.exists('best_loss'):
  os.makedirs('best_loss')

# tensorflow v2 val_acc -> val_accuracy
weight_path="./best_loss/epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}_val_loss_{val_loss:.4f}.h5"
best_weight_path = "./best_weight.h5"

ck_epoch_weight = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                                          save_weights_only=True,
                                          save_best_only=True,
                                          mode='min')

ck_best_weight = ModelCheckpoint(best_weight_path, monitor='val_loss', verbose=1,
                                        save_weights_only=True,
                                        save_best_only=True,
                                        mode='min')

# 使用自動降低學習率 (當 validation loss 連續 5 次沒有下降時，自動降低學習率)
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.8,
    patience=5,
    verbose=1,
    mode='auto',
    min_delta=0.0001,
    cooldown=5,
    min_lr=1e-14)

# early = EarlyStopping(monitor="val_loss", 
#                       mode="min", 
#                       patience=10) # probably needs to be more patient, but kaggle time is limited

# 學習率動態調整。當跑到第幾個 epcoh 時，根據設定修改學習率。這邊的數值都是參考原 paper
def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 500:
        lr = 1e-7
    elif epoch > 200:
        lr = 1e-6
    elif epoch > 100:
        lr = 1e-5
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

# 設定 callbacks
callbacks = [ck_epoch_weight, ck_best_weight, lr_reducer]

epochs = 80

history = main_model.fit(
    train_data_gen,
    epochs=epochs,
    validation_data=val_data_gen,
    callbacks=callbacks,
    workers=2
)

#%%

import os

folderPath = r"C:\Users\admin\Documents\GitHub\ML100Days\Homework\data\image_data\test"
testFolder = os.listdir(folderPath)
fileName = []
prediction_class = []

import tensorflow as tf

for file in testFolder:
    filePath = os.path.join(folderPath,file)
    image = tf.keras.preprocessing.image.load_img(filePath,target_size=(IMG_SHAPE,IMG_SHAPE))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    
    prediction_class.append(np.argmax(predictions))
    fileName.append(file.replace(".jpg", ""))
    
    print("image: %s, class: %d" %(file, np.argmax(predictions)))


import pandas  as pd


sub = pd.DataFrame({'id': fileName, 'flower_class': prediction_class})

sub.to_csv('final_exam_tesh.csv', index=False)
