# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:53:35 2020

@author: sandra_chang
"""
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, MaxPooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Sequential

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

BATCH_SIZE = 16 # batch 的大小，如果出現 OOM error，請降低這個值
num_classes = 5 # 類別的數量，Cifar 10 共有 10 個類別
NUM_EPOCHS = 200 # 訓練的 epochs 數量
FREEZE_LAYERS = 2 # 凍結網路層數
IMAGE_SIZE = (128, 128)


# Image preprocess
train_datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.125, height_shift_range=0.125, zoom_range=0.125, horizontal_flip=True,
                                   validation_split=0.2, rescale=1. / 255)
train_batches = train_datagen.flow_from_directory( r"C:\Users\admin\Documents\GitHub\ML100Days\Homework\data\image_data\train", 
                                                  subset = 'training',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE)
valid_batches = train_datagen.flow_from_directory( r"C:\Users\admin\Documents\GitHub\ML100Days\Homework\data\image_data\train", 
                                                  subset = 'validation',
                                                  target_size=IMAGE_SIZE,
                                                  interpolation='bicubic',
                                                  class_mode='categorical',
                                                  shuffle=False,
                                                  batch_size=BATCH_SIZE)
for cls, idx in train_batches.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))


# #%%
# net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
#                 input_shape=(512,512,3))
# x = net.output
# x = Flatten()(x)

# # 增加 DropOut layer
# x = Dropout(0.5)(x)

# # 增加 Dense layer，以 softmax 產生個類別的機率值
# output_layer = Dense(num_classes, activation='softmax', name='softmax')(x)

# # 設定凍結與要進行訓練的網路層
# net_final = Model(inputs=net.input, outputs=output_layer)
# for layer in net_final.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in net_final.layers[FREEZE_LAYERS:]:
#     layer.trainable = True
# # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
# net_final.compile(optimizer=Adam(lr=1e-5),
#                   loss='categorical_crossentropy', metrics=['accuracy'])


# 學習率動態調整。當跑到第幾個 epcoh 時，根據設定修改學習率。這邊的數值都是參考原 paper
def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

# 使用 resnet_layer 來建立我們的 ResNet 模型
def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    # 建立卷積層
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    # 對輸入進行卷機，根據 conv_first 來決定 conv. bn, activation 的順序
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

# Resnet v1 共有三個 stage，每經過一次 stage，影像就會變小一半，但 channels 數量增加一倍。ResNet-20 代表共有 20 層 layers，疊越深參數越多
def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # 模型的初始設置，要用多少 filters，共有幾個 residual block （組成 ResNet 的單元）
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    
    # 建立 Input layer
    inputs = Input(shape=input_shape)
    
    # 先對影像做第一次卷機
    x = resnet_layer(inputs=inputs)
    
    # 總共建立 3 個 stage
    for stack in range(3):
        # 每個 stage 建立數個 residual blocks (數量視你的層數而訂，越多層越多 block)
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = tf.keras.layers.add([x, y]) # 此處把 featuremaps 與 上一層的輸入加起來 (欲更了解結構需閱讀原論文)
            x = Activation('relu')(x)
        num_filters *= 2

    # 建立分類
    # 使用 average pooling，且 size 跟 featuremaps 的 size 一樣 （相等於做 GlobalAveragePooling）
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    
    # 接上 Dense layer 來做分類
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 建立模型
    model = Model(inputs=inputs, outputs=outputs)
    return model

# # 建立 ResNet v1 模型
# model = resnet_v1(input_shape=input_shape, depth=depth)

# net = ResNet50(include_top=False, weights='imagenet', input_tensor=None,
#                 input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
# x = net.output
# x = Flatten()(x)

# # 增加 DropOut layer
# x = Dropout(0.5)(x)

# # 增加 Dense layer，以 softmax 產生個類別的機率值
# output_layer = Dense(num_classes, activation='softmax', name='softmax')(x)

# # 設定凍結與要進行訓練的網路層
# net_final = Model(inputs=net.input, outputs=output_layer)
# for layer in net_final.layers[:FREEZE_LAYERS]:
#     layer.trainable = False
# for layer in net_final.layers[FREEZE_LAYERS:]:
#     layer.trainable = True


# # 編譯模型，使用 Adam 優化器並使用學習率動態調整的函數，０代表在第一個 epochs
# # net_final.compile(loss='categorical_crossentropy',
# #               optimizer=Adam(lr=lr_schedule(0)),
# #               metrics=['accuracy'])

# net_final.compile(optimizer=Adam(lr=1e-5),
#                   loss='categorical_crossentropy', metrics=['accuracy'])
# net_final.summary()


#%%
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

# 使用動態調整學習率
lr_scheduler = LearningRateScheduler(lr_schedule)

# 使用自動降低學習率 (當 validation loss 連續 5 次沒有下降時，自動降低學習率)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
# 設定 callbacks
callbacks = [lr_reducer, lr_scheduler]


#%%
# saved model
WEIGHTS_FINAL = 'model-test.h5'
# Train
model.fit_generator(train_batches, validation_data = valid_batches, epochs = NUM_EPOCHS,callbacks=callbacks)
# Store Model
model.save(WEIGHTS_FINAL)


#%%
# model = tf.keras.models.load_model('model-InceptionResNetV2.h5')

score = model.evaluate(valid_batches, verbose=0)

# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])


#%%


folderPath = r"C:\Users\admin\Documents\GitHub\ML100Days\Homework\data\image_data\test"
testFolder = os.listdir(folderPath)
fileName = []
prediction_class = []

for file in testFolder:
    filePath = os.path.join(folderPath,file)
    image = tf.keras.preprocessing.image.load_img(filePath,target_size=(IMAGE_SIZE[0],IMAGE_SIZE[1]))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(input_arr)
    
    prediction_class.append(np.argmax(predictions))
    fileName.append(file.replace(".jpg", ""))
    
    print("image: %s, class: %d",file, np.argmax(predictions))


import pandas  as pd


sub = pd.DataFrame({'id': fileName, 'flower_class': prediction_class})

sub.to_csv('final_exam_tesh.csv', index=False)