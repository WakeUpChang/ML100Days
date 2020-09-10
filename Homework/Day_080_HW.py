# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 07:38:36 2020

@author: admin
"""

import os
import keras

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


train, test = keras.datasets.cifar10.load_data()

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

x_train, y_train = train
x_test, y_test = test

# Preproc the inputs
x_train = preproc_x(x_train)
x_test = preproc_x(x_test)

# Preprc the outputs
y_train = preproc_y(y_train)
y_test = preproc_y(y_test)

def build_mlp(input_shape, output_units=10):
    input_layer = keras.layers.Input(input_shape)
    
    x = keras.layers.Dense(units=512, activation="relu", name="hidden_layer1")(input_layer)
    x = keras.layers.Dense(units=256, activation="relu", name="hidden_layer2")(x)
    x = keras.layers.Dense(units=64, activation="relu", name="hidden_layer3")(x)
    
    out = keras.layers.Dense(units=output_units, activation="softmax", name="output")(x)
    
    model = keras.models.Model(inputs=[input_layer], outputs=[out])
    
    return model

def getOptimizer(optimizer):
    if(optimizer=="RMSprop"):
        return keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    if(optimizer=="Adam"):
        return keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    if(optimizer=="AdaGrad"):
        return keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    if(optimizer=="SGD"):
        return keras.optimizers.SGD(learning_rate = LEARNING_RATE,momentum=MOMENTUM)


## 超參數設定

LEARNING_RATE = 0.01
EPOCHS = 50
BATCH_SIZE = 256
MOMENTUM = 0.95

optimizers = {"RMSprop","Adam","AdaGrad","SGD"}

results = {}
"""
使用迴圈，建立不同 Learning rate 的模型並訓練
"""
for optStr in optimizers:
    keras.backend.clear_session() # 把舊的 Graph 清掉
    print("Experiment with LR = %.6f" % (LEARNING_RATE))
    model = build_mlp(input_shape=x_train.shape[1:])
    optimizer = getOptimizer(optStr)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

    model.fit(x_train, y_train, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              validation_data=(x_test, y_test), 
              shuffle=True)
    
        # Collect results
    train_loss = model.history.history["loss"]
    valid_loss = model.history.history["val_loss"]
    train_acc = model.history.history["accuracy"]
    valid_acc = model.history.history["val_accuracy"]
    
    exp_name_tag = "optimizer-%s" % optStr
    results[exp_name_tag] = {'train-loss': train_loss,
                             'valid-loss': valid_loss,
                             'train-acc': train_acc,
                             'valid-acc': valid_acc}
    
import matplotlib.pyplot as plt

color_bar = ["r", "g", "b","k"]

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-loss'])),results[cond]['train-loss'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-loss'])),results[cond]['valid-loss'], '--', label=cond, color=color_bar[i])
plt.title("Loss")
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-acc'])),results[cond]['train-acc'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-acc'])),results[cond]['valid-acc'], '--', label=cond, color=color_bar[i])
plt.title("Accuracy")
plt.legend()
plt.show()
