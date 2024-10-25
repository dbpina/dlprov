import os
import time
import numpy as np
from pathlib import Path   
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
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

from datetime import datetime      

def load_data(t1, t2, dataflow_tag, exec_tag, dataset):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
        
    tf1_input = DataSet("iInputDataset", [Element(["mnist", "tf.keras.datasets.mnist"])])
    t1.add_dataset(tf1_input)
    t1.begin()

    dataset_dir = Path("./temp_mnist")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    tf1_output = DataSet("oLoadData", [Element([os.path.join(os.getcwd()) + "./temp_mnist"])])
    t1.add_dataset(tf1_output)
    t1.end()  

    tf2_input = DataSet("iSplitConfig", [Element(["80-20"])])
    t2.add_dataset(tf2_input)
    t2.begin() 

    np.savez_compressed(dataset_dir / "train.npz", x_train=x_train, y_train=y_train)
    np.savez_compressed(dataset_dir / "test.npz", x_test=x_test, y_test=y_test)     

    tf2_train_output = DataSet("oTrainSet", [Element([str(dataset_dir / "train.npz")])])
    t2.add_dataset(tf2_train_output)
    t2.save()

    tf2_test_output = DataSet("oTestSet", [Element([str(dataset_dir / "test.npz")])])
    t2.add_dataset(tf2_test_output)
    t2.end()   

    return (x_train, y_train), (x_test, y_test)

def train_model(t3, dataflow_tag, exec_tag, y, epochs):
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
            logs['elapsed_time'] = elapsed_time            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            elapsed_time = logs.get('elapsed_time', 'N/A')
            loss = -9999999 if np.isnan(logs.get('loss', np.nan)) else logs['loss']
            accuracy = -9999999 if np.isnan(logs.get('accuracy', np.nan)) else logs['accuracy']
            val_loss = -9999999 if np.isnan(logs.get('val_loss', np.nan)) else logs['val_loss']
            val_accuracy = -9999999 if np.isnan(logs.get('val_accuracy', np.nan)) else logs['val_accuracy']

            tf3_output = DataSet("oTrainModel", [
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

    (x_train, y_train), (x_test, y_test) = y
    
    callbacks = [DLProvCallback(t3)]

    model = models.Sequential([
            layers.Flatten(input_shape=(28, 28)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(10, activation='softmax')
        ])

    learning_rate = 0.001  
    opt = Adam(learning_rate=learning_rate)    

    tf3_input = DataSet("iTrainModel", [Element([opt.get_config()['name'], opt.get_config()['learning_rate'], epochs, 
        len(model.layers)])])
    t3.add_dataset(tf3_input)
    t3.begin()     

    model.compile(optimizer=opt, 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']) 

    model.fit(x_train, y_train, epochs=epochs, validation_split=0.2, callbacks=callbacks)   

    trained_path = os.path.join(os.getcwd(), "mnist-trained.keras")
    model.save(trained_path)

    tf3_output_model = DataSet("oTrainedModel", [Element(["trained_model", str(trained_path)])])
    t3.add_dataset(tf3_output_model)
    t3.end()        

    return trained_path

def evaluate_model(t4, dataflow_tag, exec_tag, y, x):
    (x_train, y_train), (x_test, y_test) = y
    model = load_model(x)
    t4.begin()                
    test_loss, test_acc = model.evaluate(x_test, y_test)
    testing_output = DataSet("oTestModel", [Element([-9999999 if np.isnan(test_loss) else test_loss, -9999999 if np.isnan(test_acc) else test_acc])])
    t4.add_dataset(testing_output)
    t4.end()    

    return test_acc    

def main():
    epochs = 10
    
    dataflow_tag = "mnist"    

    df = Dataflow(dataflow_tag, predefined=True)
    df.save()

    exec_tag = dataflow_tag + "-" + str(datetime.now())        

    t1 = Task(1, dataflow_tag, exec_tag, "LoadData")
    t2 = Task(2, dataflow_tag, exec_tag, "SplitData", dependency = t1)
    t3 = Task(3, dataflow_tag, exec_tag, "TrainModel", dependency = t2)
    t4 = Task(4, dataflow_tag, exec_tag, "TestModel", dependency = [t2,t3])         
    
    data = load_data(t1, t2, dataflow_tag, exec_tag, "mnist")
    model_path = os.path.join(os.getcwd(), "mnist-model.keras")
    trained_model_path = train_model(t3, dataflow_tag, exec_tag, data, epochs)
    test_accuracy = evaluate_model(t4, dataflow_tag, exec_tag, data, trained_model_path)
    print(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    main()    