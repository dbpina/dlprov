import tensorflow as tf
import numpy as np
import data_utils as du
from pathlib import Path
import time
from datetime import datetime
import gc
import os
from tensorflow.keras import layers, models, optimizers, callbacks

IMG_SIZE = 227
BATCH_SIZE = 32
NUM_CLASSES = 17
EPOCHS = 20
LEARNING_RATE = 0.002

def load_dataset():
    x, y = du.load_data()
    return x, y

def preprocess_dataset(x, y):
    x_gray = tf.image.rgb_to_grayscale(x)
    x_gray = tf.image.grayscale_to_rgb(x_gray)
    x_gray = tf.image.resize(x_gray, [IMG_SIZE, IMG_SIZE])
    x_gray = x_gray.numpy()
    return x_gray, y

def split_data(x_data, y_data):
    total = len(x_data)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    x_train, y_train = x_data[:train_end], y_data[:train_end]
    x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
    x_test, y_test = x_data[val_end:], y_data[val_end:]

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test

def alexnet(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                      activation='relu', input_shape=input_shape, padding='valid'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                      activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'),

        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'),

        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Flatten(),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_alexnet():
    x_raw, y_raw = load_dataset()
    x_processed, y_processed = preprocess_dataset(x_raw, y_raw)
    ds_train, ds_val, ds_test = split_data(x_processed, y_processed)
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    del x_raw, y_raw
    gc.collect()
    tf.keras.backend.clear_session()

    model = alexnet(input_shape, NUM_CLASSES)
    model.summary()

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        verbose=1
    )

    trained_path = os.path.join(os.getcwd(), "models/trained.keras")
    os.makedirs(os.path.dirname(trained_path), exist_ok=True)
    model.save(trained_path)

    weights_path = os.path.join(os.getcwd(), "weights/model.weights.h5")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_weights(weights_path)

    test_loss, test_acc = model.evaluate(ds_test, verbose=0)

    return history, model

if __name__ == '__main__':
    history, trained_model = train_alexnet()
    print("Script finished.")
