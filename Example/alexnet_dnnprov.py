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

from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus


np.random.seed(1000)


dataflow_tag = "dnnp-1"
exec_tag = dataflow_tag + "-" + str(datetime.now())

df = Dataflow(dataflow_tag, True)
df.save()

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
adaptation_id = 0

t1 = Task(1, dataflow_tag, exec_tag, "TrainingModel")

class LearningRateScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """
    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.epoch_num = 0
        self.start_time = None       
        self.adaptation_id = 0


    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.epoch_num +=1

        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = float(K.get_value(self.model.optimizer.lr))
        temp_lr = lr
        try:  # new API
            lr = self.schedule(epoch, lr)
        except TypeError:  # old API for backward compatibility
            lr = self.schedule(epoch)        
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        if (round(temp_lr, 5) != lr):
            self.adaptation_id += 1
            K.set_value(self.model.optimizer.lr, lr)
            t2_output = DataSet("oAdaptation", [Element([lr, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch, str(self.adaptation_id)])])
            t2.add_dataset(t2_output)
            if (epoch == epochs-1):
                t2.end()     
            else:
                t2.save()
        else:
            if (epoch == epochs-1):
                t2.end() 
            #adicionar ao dataset
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time()-self.start_time        
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

        tf1_output = DataSet("oTrainingModel", [Element([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), elapsed_time, -9999999 if np.isnan(logs['loss']) else logs['loss'], -9999999 if np.isnan(logs['accuracy']) else logs['accuracy'], -9999999 if np.isnan(logs['val_loss']) else logs['val_loss'], -9999999 if np.isnan(logs['val_accuracy']) else logs['val_accuracy'] , epoch])])
        t1.add_dataset(tf1_output)
        if(epoch==epochs-1):
          t1.end()
        else:
          t1.save()    

# learning rate schedule
def step_decay(epoch):
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lrate = LearningRateScheduler(step_decay, verbose=1)  

# (3) Create a sequential model
model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(filters=96, input_shape=(224,224,3), kernel_size=(11,11), strides=(3,3), padding='valid'))
model.add(Activation(actv))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
model.add(Activation(actv))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation(actv))
# Batch Normalisation
model.add(BatchNormalization())


# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation(actv))
# Batch Normalisation
model.add(BatchNormalization())

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation(actv))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())


# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(4096, input_shape=(224*224*3,)))
model.add(Activation(actv))
# Add Dropout to prevent overfitting
model.add(Dropout(dropout))
# Batch Normalisation
model.add(BatchNormalization())

# 2nd Dense Layer
model.add(Dense(4096))
model.add(Activation(actv))
# Add Dropout
model.add(Dropout(dropout))
# Batch Normalisation
model.add(BatchNormalization())

# 3rd Dense Layer
model.add(Dense(1000))
model.add(Activation(actv))
# Add Dropout
model.add(Dropout(dropout))
# Batch Normalisation
model.add(BatchNormalization())

# Output Layer
model.add(Dense(17))
model.add(Activation(actv))


model.summary()


# (4) Compile
opt = Adam(learning_rate=initial_lrate)
#opt = SGD(lr=lerate)

tf1_input = DataSet("iTrainingModel", [Element([opt.get_config()['name'], opt.get_config()['learning_rate'], epochs, len(model.layers)])])
t1.add_dataset(tf1_input)
t1.begin() 

t2 = Task(2, dataflow_tag, exec_tag, "Adaptation", dependency=t1)
  # learning rate schedule
t2_input = DataSet("iAdaptation", [Element([epochs_drop, drop, initial_lrate])])
t2.add_dataset(t2_input)
t2.begin()  

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


# (5) Train
model.fit(x_train, y_train, batch_size=32, epochs=epochs, callbacks=[lrate], verbose=1, validation_split=0.2, shuffle=True)

t3 = Task(3, dataflow_tag, exec_tag, "TestingModel", dependency=t1)     
t3.begin()                

a = model.evaluate(x_test, y_test)

testing_output = DataSet("oTestingModel", [Element([-9999999 if np.isnan(a[0]) else a[0], -9999999 if np.isnan(a[1]) else a[1]])])
t3.add_dataset(testing_output)
t3.end()
