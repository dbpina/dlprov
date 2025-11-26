import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_utils as du  
from pathlib import Path
import time
from datetime import datetime
import gc
import os
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.callbacks import Callback

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

# --- Configuration ---
IMG_SIZE = 227 # AlexNet input size
BATCH_SIZE = 32
NUM_CLASSES = 17 # Oxford Flowers 17 has 17 categories
EPOCHS = 20 
LEARNING_RATE = 0.002


dataflow_tag = "alexnet"    
df = Dataflow(dataflow_tag, predefined=True, email='bob@fictional.com')
df.save()

exec_tag = dataflow_tag + "-" + str(datetime.now())        
safe_tag = exec_tag.replace(":", "").replace("-", "").replace(" ", "_").replace(".", "")
base_dir = Path(safe_tag)
base_dir.mkdir(exist_ok=True)

class DLProvCallback(Callback):
    def __init__(self, transformation):
        super(DLProvCallback, self).__init__()
        self.transformation = transformation

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()           

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        epoch_end_time = time.time()
        elapsed_time = epoch_end_time - self.epoch_start_time
        logs['elapsed_time'] = float(elapsed_time)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed_time = logs.get('elapsed_time', 'N/A')
        loss = -9999999 if np.isnan(logs.get('loss', np.nan)) else logs['loss']
        accuracy = -9999999 if np.isnan(logs.get('accuracy', np.nan)) else logs['accuracy']
        val_loss = -9999999 if np.isnan(logs.get('val_loss', np.nan)) else logs['val_loss']
        val_accuracy = -9999999 if np.isnan(logs.get('val_accuracy', np.nan)) else logs['val_accuracy']

        tf4_output = DataSet("oTrain", [
            Element([
                timestamp,
                elapsed_time,
                loss,
                accuracy,
                val_loss,
                val_accuracy,
                epoch + 1
            ])
        ])

        self.transformation.add_dataset(tf4_output)
        self.transformation.save()  

def load_dataset(t1):
    """
    Loads the Oxford Flowers 17 dataset using `du.load_data()`,
    splits into training and validation sets, and then creates TensorFlow Datasets.
    """

    DATASET_NAME = "OxfordFlowers17"
    DATASET_SOURCE = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
    tf1_input = DataSet("iInputDataset", [Element([DATASET_NAME, DATASET_SOURCE])])
    t1.add_dataset(tf1_input)
    t1.begin()

    x, y = du.load_data()

    dataset_dir = base_dir / "data"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    input_data_path_x = dataset_dir / "raw_images.npy"
    input_data_path_y = dataset_dir / "raw_labels.npy"
    np.save(input_data_path_x, x)
    np.save(input_data_path_y, y)

    tf1_output = DataSet("oLoadData", [Element([input_data_path_x])])
    t1.add_dataset(tf1_output)
    t1.end()        

    return x, y

def preprocess_dataset(x, y, t2):
    tf2_input = DataSet("iFilter", [Element(["Grayscale"])])
    t2.add_dataset(tf2_input)
    t2.begin()

    x_gray = tf.image.rgb_to_grayscale(x)
    x_gray = tf.image.grayscale_to_rgb(x_gray)  # to keep 3 channels for AlexNet
    x_gray = tf.image.resize(x_gray, [IMG_SIZE, IMG_SIZE])
    x_gray = x_gray.numpy()  # Convert back to NumPy

    # plt.figure(figsize=(10,4))
    # plt.subplot(1,2,1)
    # plt.imshow(x[0])
    # plt.title("Original")
    # plt.axis('off')
    # plt.subplot(1,2,2)
    # plt.imshow(x_gray[0])
    # plt.title("Grayscale")
    # plt.axis('off')
    # plt.show()	    

    pp_dir = base_dir / "data"
    pp_dir.mkdir(parents=True, exist_ok=True)
    pp_path_x = pp_dir / "x_gray.npy"
    pp_path_y = pp_dir / "y.npy"
    np.save(pp_path_x, x_gray)
    np.save(pp_path_y, y)

    tf2_output = DataSet("oFilter", [Element([str(pp_path_x)])])
    t2.add_dataset(tf2_output)
    t2.end()

    return x_gray, y 


def split_data(x_data, y_data, t3):

	total = len(x_data)
	n_train = int(0.8 * total)
	n_val = int(0.1 * total)
	n_test = total - n_train - n_val  

	tf3_input = DataSet("iSplitConfig", [Element([n_train, n_val, n_test])])
	t3.add_dataset(tf3_input)
	t3.begin()

	train_end = int(0.8 * total)
	val_end = int(0.9 * total)

	x_train, y_train = x_data[:train_end], y_data[:train_end]
	x_val, y_val = x_data[train_end:val_end], y_data[train_end:val_end]
	x_test, y_test = x_data[val_end:], y_data[val_end:]

	split_dir = base_dir / "data"
	split_dir.mkdir(parents=True, exist_ok=True)

	train_path_x = split_dir / "x_train.npy"
	train_path_y = split_dir / "y_train.npy"
	val_path_x = split_dir / "x_val.npy"
	val_path_y = split_dir / "y_val.npy"
	test_path_x = split_dir / "x_test.npy"
	test_path_y = split_dir / "y_test.npy"

	np.save(train_path_x, x_train)
	np.save(train_path_y, y_train)
	np.save(val_path_x, x_val)
	np.save(val_path_y, y_val)
	np.save(test_path_x, x_test)
	np.save(test_path_y, y_test)

	ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
	ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

	tf3_train_output = DataSet("oTrainSet", [Element([str(train_path_x), str(train_path_y)])])
	t3.add_dataset(tf3_train_output)
	t3.save()
	tf3_val_output = DataSet("oValSet", [Element([str(val_path_x),str(val_path_y)])])
	t3.add_dataset(tf3_val_output)
	t3.save()
	tf3_test_output = DataSet("oTestSet", [Element([str(test_path_x),str(test_path_y)])])
	t3.add_dataset(tf3_test_output)
	t3.end()

	return ds_train, ds_val, ds_test

def alexnet(input_shape, num_classes):
    """
    Defines the AlexNet model architecture.
    Original AlexNet uses Local Response Normalization (LRN), which is less common now
    and often replaced by Batch Normalization. For simplicity and modern practices,
    this implementation uses Batch Normalization where applicable for better training stability.
    However, if strict adherence to the original AlexNet is required, LRN layers could be added.
    """
    model = models.Sequential([
        layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                      activation='relu', input_shape=input_shape, padding='valid'),
        layers.BatchNormalization(), # Added for modern practice, LRN in original
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                      activation='relu', padding='same'),
        layers.BatchNormalization(), # Added for modern practice, LRN in original
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
        layers.Dropout(0.5), # Dropout for regularization

        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization

        layers.Dense(num_classes, activation='softmax') 
    ])
    return model

def train_alexnet():
    """
    Executes the AlexNet training process.
    """

    t1 = Task(1, df, exec_tag, "LoadData")
    x_raw, y_raw = load_dataset(t1)

    t2 = Task(2, df, exec_tag, "ApplyFilter", dependency=t1)    
    x_processed, y_processed = preprocess_dataset(x_raw, y_raw, t2)

    t3 = Task(3, df, exec_tag, "SplitData", dependency=t2)
    ds_train, ds_val, ds_test = split_data(x_processed, y_processed, t3)
    input_shape = (IMG_SIZE, IMG_SIZE, 3) 

    del x_raw, y_raw
    gc.collect()
    tf.keras.backend.clear_session()

    model = alexnet(input_shape, NUM_CLASSES)
    model.summary()

    t4 = Task(4, df, exec_tag, "Train", dependency = t3)

    callbacks = [DLProvCallback(t4)]

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)

    tf4_input = DataSet("iTrain", [Element([opt.get_config()['name'], round(opt.get_config()['learning_rate'], 3), EPOCHS, BATCH_SIZE, len(model.layers)])])
    t4.add_dataset(tf4_input)
    t4.begin()     

    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        verbose=1,
        callbacks=callbacks
    )

    trained_path = os.path.join(os.getcwd(), base_dir / "models/trained.keras")
    os.makedirs(os.path.dirname(trained_path), exist_ok=True)
    model.save(trained_path)

    tf4_output_model = DataSet("oTrainedModel", [Element(["trained_model", str(trained_path)])])
    t4.add_dataset(tf4_output_model)
    t4.save()        

    weights_path = os.path.join(os.getcwd(), base_dir / "weights/model.weights.h5")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_weights(weights_path)

    tf4_output_weights = DataSet("oWeights", [Element([str(weights_path)])])
    t4.add_dataset(tf4_output_weights)
    t4.end()      

    t5 = Task(5, df, exec_tag, "Test", dependency = [t3,t4])  
    t5.begin()                 
    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    testing_output = DataSet("oTest", [Element([-9999999 if np.isnan(test_loss) else test_loss, -9999999 if np.isnan(test_acc) else test_acc])])
    t5.add_dataset(testing_output)
    t5.end()     

    return history, model  

if __name__ == '__main__':

	history, trained_model = train_alexnet() 

	print("Script finished.")      