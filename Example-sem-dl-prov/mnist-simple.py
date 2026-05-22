import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def train_model(data, epochs):
    (x_train, y_train), (x_test, y_test) = data

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    learning_rate = 0.001
    opt = Adam(learning_rate=learning_rate)

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, validation_split=0.2)

    trained_path = os.path.join(os.getcwd(), "mnist-trained.keras")
    model.save(trained_path)

    return trained_path

def evaluate_model(trained_path, data):
    (x_train, y_train), (x_test, y_test) = data
    model = load_model(trained_path)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    return test_acc

def main():
    epochs = 2
    data = load_data()
    trained_model_path = train_model(data, epochs)
    test_accuracy = evaluate_model(trained_model_path, data)
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()
