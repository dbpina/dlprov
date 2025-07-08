import tensorflow as tf
# import tensorflow_datasets as tfds # No longer needed if loading locally
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import Callback
import numpy as np
from pathlib import Path   
from datetime import datetime 
import time
import os
import json
import sys
from sklearn.model_selection import train_test_split # Import for train_test_split
import matplotlib.pyplot as plt # For plotting training history
import re
import gc

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

from dfa_lib_python import dataverse_uploader

print(f"TensorFlow version: {tf.__version__}")
print(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

# --- Configuration ---
IMG_SIZE = 227 # AlexNet input size
BATCH_SIZE = 32
NUM_CLASSES = 17 # Oxford Flowers 17 has 17 categories
EPOCHS = 2 # You might want to increase this for better performance
LEARNING_RATE = 0.001


dataflow_tag = "alexnet-dverse"    
df = Dataflow(dataflow_tag, predefined=True)
df.save()

exec_tag = dataflow_tag + "-" + str(datetime.now())        


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

        tf3_output = DataSet("oTrain", [
            Element([
                timestamp,
                elapsed_time,
                loss,
                accuracy,
                val_loss,
                val_accuracy,
                epoch
            ])
        ])

        self.transformation.add_dataset(tf3_output)
        self.transformation.save()  

# --- 1. Load and Preprocess the Oxford Flowers 17 Dataset ---

def preprocess_image(image, label):
    """
    Resizes the image to IMG_SIZE x IMG_SIZE and normalizes pixel values.
    Adjusts label to be 0-indexed.
    NOTE: Assumes labels from du.load_data() are 1-indexed, similar to
    how tfds.load('oxford_flowers17') behaves. If your du.load_data()
    already provides 0-indexed labels, remove `label = label - 1`.
    """
    # Resize image to AlexNet input size (227x227)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    # Normalize pixel values to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # Oxford Flowers 17 labels are 1-indexed, convert to 0-indexed
    #label = label - 1
    return image, label

def load_dataset(t1):
    """
    Loads the Oxford Flowers 17 dataset using your local `du.load_data()`,
    splits into training and validation sets, and then creates TensorFlow Datasets.
    """
    print("Loading Oxford Flowers 17 dataset using du.load_data()...")

    DATASET_NAME = "OxfordFlowers17"
    DATASET_SOURCE = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz"
    tf1_input = DataSet("iInputDataset", [Element([DATASET_NAME, DATASET_SOURCE])])
    t1.add_dataset(tf1_input)
    t1.begin()

    # IMPORTANT: You need to ensure 'du' module is available and
    # 'du.load_data()' correctly loads your X (images) and Y (labels).
    try:
        import data_utils as du # Assuming 'du' is your data utility module
        x, y = du.load_data()
        print(y)
    except ModuleNotFoundError:
        print("Error: 'data_utils' module not found. Please ensure 'data_utils.py' is in your path.")
        print("Using simulated data for demonstration. Replace this with your actual data loading.")
        # --- Simulate du.load_data() for demonstration if 'du' is not found ---
        total_images = 1360
        x = np.random.randint(0, 256, size=(total_images, 300, 300, 3), dtype=np.uint8)
        y = np.array([(i // 80) + 1 for i in range(total_images)], dtype=np.int64) # Simulate 1-indexed labels
        # ---------------------------------------------------------------------

    # Save original loaded data for logging its path
    dataset_dir = Path("data")
    dataset_dir.mkdir(exist_ok=True)
    input_data_path_x = dataset_dir / "raw_images.npy"
    input_data_path_y = dataset_dir / "raw_labels.npy"
    np.save(input_data_path_x, x)
    np.save(input_data_path_y, y)

    tf1_output = DataSet("oLoadData", [Element([input_data_path_x])])
    t1.add_dataset(tf1_output)
    t1.end()        
    
    print("Dataset loaded successfully.")

    return x, y

def split_data(x_data, y_data, t2):
    """
    Splits the preprocessed data into train, validation, and test sets (80/10/10)
    and logs the process using Task/DataSet.
    """

    # The split config from your snippet "80","10","10"
    tf2_input = DataSet("iSplitConfig", [Element(["80","10","10"])])
    t2.add_dataset(tf2_input)
    t2.begin()

    print("Splitting data into 80% train, 10% validation, 10% test...")
        
    # First, split into training+validation (90%) and test (10%)
    x_train_val_raw, x_test_raw, y_train_val_raw, y_test_raw = train_test_split(
        x_data, y_data, test_size=0.1, random_state=42, shuffle=True, stratify=y_data
    )

    # Then, split training+validation (90%) into training (80%) and validation (10%)
    # This means train is 8/9 of x_train_val, and val is 1/9 of x_train_val
    x_train_raw, x_val_raw, y_train_raw, y_val_raw = train_test_split(
        x_train_val_raw, y_train_val_raw, test_size=(1/9), random_state=42, shuffle=True, stratify=y_train_val_raw
    )

    PREPROCESSED_DATA_DIR = Path("preprocessed_data")
    PREPROCESSED_DATA_DIR.mkdir(exist_ok=True)
    
    print("Converting raw split data to TensorFlow Datasets for preprocessing...")
    # Create temporary tf.data.Dataset from raw NumPy arrays to apply preprocessing efficiently
    ds_train_temp = tf.data.Dataset.from_tensor_slices((x_train_raw, y_train_raw))
    ds_val_temp = tf.data.Dataset.from_tensor_slices((x_val_raw, y_val_raw))
    ds_test_temp = tf.data.Dataset.from_tensor_slices((x_test_raw, y_test_raw))

    print("Applying preprocessing to split datasets and converting back to NumPy arrays...")
    # Apply preprocessing (resize, normalize, and 1-indexed to 0-indexed label conversion)
    # Then convert back to NumPy arrays to save them.
    x_train_preprocessed = np.array([img for img, _ in ds_train_temp.map(preprocess_image).as_numpy_iterator()])
    y_train_preprocessed = np.array([lbl for _, lbl in ds_train_temp.map(preprocess_image).as_numpy_iterator()])

    x_val_preprocessed = np.array([img for img, _ in ds_val_temp.map(preprocess_image).as_numpy_iterator()])
    y_val_preprocessed = np.array([lbl for _, lbl in ds_val_temp.map(preprocess_image).as_numpy_iterator()])

    x_test_preprocessed = np.array([img for img, _ in ds_test_temp.map(preprocess_image).as_numpy_iterator()])
    y_test_preprocessed = np.array([lbl for _, lbl in ds_test_temp.map(preprocess_image).as_numpy_iterator()])


    # --- Saving preprocessed data to disk and logging paths ---
    print(f"Saving preprocessed data to '{PREPROCESSED_DATA_DIR}'...")
    train_path_x = PREPROCESSED_DATA_DIR / "x_train.npy"
    train_path_y = PREPROCESSED_DATA_DIR / "y_train.npy"
    val_path_x = PREPROCESSED_DATA_DIR / "x_val.npy"
    val_path_y = PREPROCESSED_DATA_DIR / "y_val.npy"
    test_path_x = PREPROCESSED_DATA_DIR / "x_test.npy"
    test_path_y = PREPROCESSED_DATA_DIR / "y_test.npy"

    np.save(train_path_x, x_train_preprocessed)
    np.save(train_path_y, y_train_preprocessed)
    np.save(val_path_x, x_val_preprocessed)
    np.save(val_path_y, y_val_preprocessed)
    np.save(test_path_x, x_test_preprocessed)
    np.save(test_path_y, y_test_preprocessed)

    print(f"Preprocessed x_train shape: {x_train_preprocessed.shape}, y_train shape: {y_train_preprocessed.shape}")
    print(f"Preprocessed x_val shape: {x_val_preprocessed.shape}, y_val shape: {y_val_preprocessed.shape}")
    print(f"Preprocessed x_test shape: {x_test_preprocessed.shape}, y_test shape: {y_test_preprocessed.shape}")
    
    tf2_train_output = DataSet("oTrainSet", [Element([str(train_path_x)])])
    t2.add_dataset(tf2_train_output)
    t2.save() # As per your snippet, save after adding each dataset

    tf2_val_output = DataSet("oValSet", [Element([str(val_path_x)])])
    t2.add_dataset(tf2_val_output)
    t2.save() # As per your snippet

    tf2_test_output = DataSet("oTestSet", [Element([str(test_path_x)])])
    t2.add_dataset(tf2_test_output)
    t2.end()

    # Now, create the actual batched and prefetched TF Datasets for training
    # from the already preprocessed NumPy arrays. No further mapping of preprocess_image needed here.
    ds_train = tf.data.Dataset.from_tensor_slices((x_train_preprocessed, y_train_preprocessed))
    ds_train = ds_train.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_val = tf.data.Dataset.from_tensor_slices((x_val_preprocessed, y_val_preprocessed))
    ds_val = ds_val.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    ds_test = tf.data.Dataset.from_tensor_slices((x_test_preprocessed, y_test_preprocessed))
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test    

# --- 2. Define the AlexNet Architecture ---

def alexnet(input_shape, num_classes):
    """
    Defines the AlexNet model architecture.
    Original AlexNet uses Local Response Normalization (LRN), which is less common now
    and often replaced by Batch Normalization. For simplicity and modern practices,
    this implementation uses Batch Normalization where applicable for better training stability.
    However, if strict adherence to the original AlexNet is required, LRN layers would be added.
    """
    model = models.Sequential([
        # First Convolutional Layer
        layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),
                      activation='relu', input_shape=input_shape, padding='valid'),
        layers.BatchNormalization(), # Added for modern practice, LRN in original
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Second Convolutional Layer
        layers.Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1),
                      activation='relu', padding='same'),
        layers.BatchNormalization(), # Added for modern practice, LRN in original
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Third Convolutional Layer
        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'),

        # Fourth Convolutional Layer
        layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'),

        # Fifth Convolutional Layer
        layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),

        # Flatten Layer
        layers.Flatten(),

        # First Fully Connected Layer
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization

        # Second Fully Connected Layer
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5), # Dropout for regularization

        # Output Layer
        layers.Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])
    return model

# --- 3. Main Training Function ---

def train_alexnet():
    """
    Executes the AlexNet training process.
    """
    # Load and preprocess dataset
    # load_dataset now returns tf.data.Dataset objects directly
    t1 = Task(1, dataflow_tag, exec_tag, "LoadData")
    x_raw, y_raw = load_dataset(t1)

    # Split data into train/val/test and log the splitting process,
    # then prepare TensorFlow Datasets
    t2 = Task(2, dataflow_tag, exec_tag, "SplitData", dependency=t1)
    ds_train, ds_val, ds_test = split_data(x_raw, y_raw, t2)

    # Define input shape
    input_shape = (IMG_SIZE, IMG_SIZE, 3) # 3 for RGB channels

    del x_raw, y_raw
    gc.collect()
    tf.keras.backend.clear_session()

    # Create the AlexNet model
    model = alexnet(input_shape, NUM_CLASSES)
    model.summary()

    t3 = Task(3, dataflow_tag, exec_tag, "Train", dependency = t2)

    callbacks = [DLProvCallback(t3)]

    opt = optimizers.Adam(learning_rate=LEARNING_RATE)

    tf3_input = DataSet("iTrain", [Element([opt.get_config()['name'], opt.get_config()['learning_rate'], EPOCHS, BATCH_SIZE, len(model.layers)])])
    t3.add_dataset(tf3_input)
    t3.begin()     

    # Compile the model
    # Using Adam optimizer, SparseCategoricalCrossentropy for 0-indexed integer labels
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    print("\nStarting model training...")
    # Train the model
    history = model.fit(
        ds_train,
        epochs=EPOCHS,
        validation_data=ds_val,
        verbose=1,
        callbacks=callbacks
    )
    print("Model training completed.")

    os.makedirs("models", exist_ok=True)
    trained_path = os.path.join(os.getcwd(), "models", "trained.keras")
    model.save(trained_path)

    tf3_output_model = DataSet("oTrainedModel", [Element(["trained_model", str(trained_path)])])
    t3.add_dataset(tf3_output_model)
    t3.save()        

    weights_path = "weights/model.weights.h5"
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    model.save_weights(weights_path)

    tf3_output_weights = DataSet("oWeights", [Element([str(weights_path)])])
    t3.add_dataset(tf3_output_weights)
    t3.end()      

    # Evaluate the model on the test set
    print("\nEvaluating model on test set...")
    t4 = Task(4, dataflow_tag, exec_tag, "Test", dependency = [t2,t3])  
    t4.begin()                 
    test_loss, test_acc = model.evaluate(ds_test, verbose=0)
    testing_output = DataSet("oTest", [Element([-9999999 if np.isnan(test_loss) else test_loss, -9999999 if np.isnan(test_acc) else test_acc])])
    t4.add_dataset(testing_output)
    t4.end()     
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # You can save the trained model if needed
    # model.save('alexnet_flowers17.h5')
    # print("Model saved as 'alexnet_flowers17.h5'")

    return history, model

if __name__ == '__main__':
    # Run the training process
    history, trained_model = train_alexnet()

    # Optional: Plot training history
    # plt.figure(figsize=(12, 4))

    # plt.subplot(1, 2, 1)
    # plt.plot(history.history['accuracy'], label='Training Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.grid(True)

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history['loss'], label='Training Loss')
    # plt.plot(history.history['val_loss'], label='Validation Loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, root_dir)

    from generate_prov import generate_prov

    generate_prov.run(["--df_exec", exec_tag])

    directory_label = "provenance"
    base_path = "/Users/debora/Documents/Doutorado/dlprov/Example/output"

    exec_tag = exec_tag.replace(" ", "_")
    exec_tag = re.sub(r"[/:]", "-", exec_tag)  # Replace slashes and colons
    exec_tag = re.sub(r"[^\w\-.]", "", exec_tag)   

    pdf_path   = os.path.join(base_path, f"{exec_tag}.pdf")
    json_path  = os.path.join(base_path, f"{exec_tag}.json")
    provn_path = os.path.join(base_path, f"{exec_tag}.provn")


    with open("dataset_pid.json", "r") as f:
        pid = json.load(f)["dataset_pid"]

    print(f"Uploading: {pdf_path} to {directory_label}")
    persistent_id = dataverse_uploader.upload_file(pid, pdf_path, directory_label)    

    print(f"Uploading: {json_path} to {directory_label}")
    persistent_id = dataverse_uploader.upload_file(pid, json_path, directory_label)    

    print(f"Uploading: {provn_path} to {directory_label}")
    persistent_id = dataverse_uploader.upload_file(pid, provn_path, directory_label)        

    print("Script finished.")
