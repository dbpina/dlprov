import data_utils as du
import os
import sys

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten,\
 Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split

import time
import math

import numpy as np

from datetime import datetime

np.random.seed(1000)

x, y = du.load_data()

y_tmp = np.zeros((x.shape[0], 17))

for i in range(0, x.shape[0]):
  y_tmp[i][y[i]] = 1
y = y_tmp

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

dropout = 0.4
epochs = 10
lerate = 0.002
actv = 'relu'

initial_lrate = lerate
drop = 0.5
epochs_drop = 10.0

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        temp_lr = lr
        try:
            lr = self.schedule(epoch, lr)
        except TypeError:
            lr = self.schedule(epoch)
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if round(temp_lr, 5) != lr:
            K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

def step_decay(epoch):
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay, verbose=1)

model = Sequential()

model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(3,3), padding='valid'))
model.add(Activation(actv))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation(actv))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation(actv))
model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation(actv))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation(actv))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation(actv))
model.add(Dropout(dropout))
model.add(BatchNormalization())

model.add(Dense(4096))
model.add(Activation(actv))
model.add(Dropout(dropout))
model.add(BatchNormalization())

model.add(Dense(1000))
model.add(Activation(actv))
model.add(Dropout(dropout))
model.add(BatchNormalization())

model.add(Dense(17))
model.add(Activation(actv))

model.summary()

opt = Adam(learning_rate=initial_lrate)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=epochs, callbacks=[lrate], verbose=1, validation_split=0.2, shuffle=True)

a = model.evaluate(x_test, y_test)
