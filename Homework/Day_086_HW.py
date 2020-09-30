
import os
import keras
import itertools
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization

# Disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

def build_mlp(input_shape, output_units=10,num_neurons=[512, 256, 128], ratio=1e-4, drp_ratio = 0.1):
    input_layer = keras.layers.Input(input_shape)
    for i, n_units in enumerate(num_neurons):
      if i == 0:
          x = BatchNormalization()(input_layer)
          x = keras.layers.Dense(units=n_units, 
                                 activation="relu", 
                                 name="hidden_layer"+str(i+1))(x)
          # x = Dropout(drp_ratio)(x)
          # x = BatchNormalization()(x)
      else:
          x = BatchNormalization()(x)
          x = keras.layers.Dense(units=n_units, 
                                 activation="relu", 
                                 name="hidden_layer"+str(i+1))(x)
          # x = Dropout(drp_ratio)(x)
          
    out = keras.layers.Dense(units=output_units, activation="softmax", name="output")(x)
    model = keras.models.Model(inputs=[input_layer], outputs=[out])
    return model

LEARNING_RATE = 1e-3
# BATCH_SIZE =[ 2, 16, 32, 128, 256]
BATCH_SIZE =256
MOMENTUM = 0.95
drop_ratio=0.1
EPOCHS = 50
EARLY_STOP = [10,25]
# Dropout_EXP = [0.1, 0.5, 0.9]
# LAYER_NEURONS = [[128, 128, 128], [128, 256, 256], [128, 256, 512]]
results = {}

BOOL =[True,False]


for i in BOOL:
    keras.backend.clear_session() # 把舊的 Graph 清掉
    # print("Experiment with Regulizer = %.6f" % (regulizer_ratio))
    # earlystop = EarlyStopping(monitor="val_loss", 
    #                       patience=i, 
    #                       verbose=1
    #                       )
    model_ckpt = ModelCheckpoint(filepath="./ModelCheckpoint.h5", 
                             monitor="val_loss", 
                             save_best_only=i)
    
    model = build_mlp(input_shape=x_train.shape[1:])
    # model.summary()
    optimizer = keras.optimizers.Adam(lr=LEARNING_RATE)
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
    
    model.fit(x_train, y_train, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              validation_data=(x_test, y_test), 
              callbacks=[model_ckpt],
              shuffle=True)

    # Collect results
    train_loss = model.history.history["loss"]
    valid_loss = model.history.history["val_loss"]
    train_acc = model.history.history["accuracy"]
    valid_acc = model.history.history["val_accuracy"]
    
    exp_name_tag = "exp-%s" % i
    results[exp_name_tag] = {'train-loss': train_loss,
                             'valid-loss': valid_loss,
                             'train-acc': train_acc,
                             'valid-acc': valid_acc}
    
color_bar = ["b", "y", "m", "k", "r", "g"]

# Load back
model = keras.models.load_model("./ModelCheckpoint.h5")
loss_loadback, acc_loadback = model.evaluate(x_test, y_test)

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-loss'])),results[cond]['train-loss'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-loss'])),results[cond]['valid-loss'], '--', label=cond, color=color_bar[i])
plt.hlines(y=loss_loadback, xmin=0, xmax=len(train_loss), colors='r', linestyles='--')
plt.title("Loss")
plt.ylim([0, 5])
plt.legend()
plt.show()

plt.figure(figsize=(8,6))
for i, cond in enumerate(results.keys()):
    plt.plot(range(len(results[cond]['train-acc'])),results[cond]['train-acc'], '-', label=cond, color=color_bar[i])
    plt.plot(range(len(results[cond]['valid-acc'])),results[cond]['valid-acc'], '--', label=cond, color=color_bar[i])
plt.hlines(y=acc_loadback, xmin=0, xmax=len(train_loss), colors='r', linestyles='--')
plt.title("Accuracy")
plt.legend()
plt.show()