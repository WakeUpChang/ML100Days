import os
import keras
import itertools
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
from keras.regularizers import l1, l2, l1_l2

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

def build_mlp(input_shape,Lx, output_units=10,num_neurons=[512, 256, 128], ratio=1e-4, drp_ratio = 0.1,bBatchNormalization=False):
    input_layer = keras.layers.Input(input_shape)
    for i, n_units in enumerate(num_neurons):
      if i == 0:
          if bBatchNormalization:
              x = BatchNormalization()(input_layer)
              x = keras.layers.Dense(units=n_units, activation="relu",kernel_regularizer=Lx(ratio),  name="hidden_layer"+str(i+1))(x)
          else:
              x = keras.layers.Dense(units=n_units, activation="relu",kernel_regularizer=Lx(ratio),   name="hidden_layer"+str(i+1))(input_layer)
          x = Dropout(drp_ratio)(x)
        
      else:
          if bBatchNormalization:
              x = BatchNormalization()(x)
          x = keras.layers.Dense(units=n_units, 
                                 activation="relu", 
                                 kernel_regularizer=Lx(ratio), 
                                 name="hidden_layer"+str(i+1))(x)
          x = Dropout(drp_ratio)(x)
          
    out = keras.layers.Dense(units=output_units, activation="softmax",kernel_regularizer=Lx(ratio),  name="output")(x)
    model = keras.models.Model(inputs=[input_layer], outputs=[out])
    return model

LEARNING_RATE = 1e-3
BATCH_SIZE =512
MOMENTUM = 0.95

EPOCHS = 30
bBatchNormalization = [True,False]
Dropout_EXP = [0.1, 0.5]
LxRegularization = [l1, l2]


results = {}
"""
使用迴圈建立不同的帶不同 L1/L2 的模型並訓練
"""
for i,(dropRatio,ln, bBN) in enumerate(itertools.product(Dropout_EXP,LxRegularization,bBatchNormalization)):
    keras.backend.clear_session() # 把舊的 Graph 清掉
    # print("Experiment with Regulizer = %.6f" % (regulizer_ratio))
    model = build_mlp(input_shape=x_train.shape[1:],Lx=ln,drp_ratio = dropRatio,bBatchNormalization=bBN)
    model.summary()
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    
    model.fit(x_train, y_train, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              validation_data=(x_test, y_test), 
              shuffle=True)

    # Collect results
    train_loss = model.history.history["loss"]
    valid_loss = model.history.history["val_loss"]
    train_acc = model.history.history["acc"]
    valid_acc = model.history.history["val_acc"]
    
    exp_name_tag = "dropRatio-%s, ln-%s, bBN-%s" %(dropRatio ,ln ,bBN)
    results[exp_name_tag] = {'train-loss': train_loss,
                             'valid-loss': valid_loss,
                             'train-acc': train_acc,
                             'valid-acc': valid_acc}
    

plt.figure(figsize=(20,10))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-loss'])),results[cond]['train-loss'], '-', label=cond)
    plt.plot(range(len(results[cond]['valid-loss'])),results[cond]['valid-loss'], '--', label=cond)
plt.title("Loss")
plt.ylim([0, 5])
plt.legend()
plt.show()

plt.figure(figsize=(20,10))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-acc'])),results[cond]['train-acc'], '-', label=cond)
    plt.plot(range(len(results[cond]['valid-acc'])),results[cond]['valid-acc'], '--', label=cond)
plt.title("Accuracy")
plt.legend()
plt.show()