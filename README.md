# DLProv

DLProv is a service that evolved from [DNNProv](https://github.com/dbpina/dnnprov). Originally rooted in DNNProv, DLProv has expanded its scope and capabilities to accommodate the broader domain of Deep Learning (DL).

DNNProv began as a provenance service designed to support online hyperparameter analysis in DL, integrating retrospective provenance data (r-prov) with typical DNN software data, e.g. hyperparameters, DNN architecture attributes, etc.

A DL life cycle involves several data transformations, such as performing data pre-processing, defining datasets to train and test a deep neural network (DNN), and training and evaluating the DL model. Choosing a final model requires DL model selection, which involves analyzing data from several training configurations (e.g. hyperparameters and DNN architectures). We have understood that tracing training data back to pre-processing operations can provide insights into the model selection step. However, there are challenges in providing an integration of the provenance of these different steps. Therefore, we decided to integrate these steps. DLProv is a prototype for provenance data integration using different capture solutions while maintaining DNNProv capabilities.

## Overview

DLProv is .......

DLProv is developed on top of [DfAnalyzer](https://gitlab.com/ssvitor/dataflow_analyzer) provenance services. It uses the columnar DBMS MonetDB to support online provenance data analysis and to generate W3C PROV-compliant documents. In addition, these provenance documents can be analyzed through graph DBMS such as Neo4j.

## Software requirements

The following list of software has to be configured/installed for running a DL model training that collects provenance with DLProv.

* [Java](https://java.com/pt-BR/)
* [MonetDB](http://www.monetdb.org/Documentation/UserGuide/Tutorial)
* [DfAnalyzer](https://github.com/dbpina/keras-prov/tree/main/DfAnalyzer)
* [dfa-lib-python](https://github.com/dbpina/keras-prov/tree/main/dfa-lib-python/) 

## Installation

<!---### RESTful services -->


###  Python library: dfa-lib-python

The DfAnalyzer library for the programming language Python can be built with the following command lines:

```

cd dfa-lib-python
python setup.py install

```

## RESTful services initialization

DLProv depends on the initialization of DfAnalyzer and the DBMS MonetDB.

Instructions for this step can also be found at [GitLab](https://gitlab.com/ssvitor/dataflow_analyzer). The project DfAnalyzer contains web applications and RESTful services provided by the tool. 

The following components are present in this project: Dataflow Viewer (DfViewer), Query Interface (QI), and Query Dashboard (QP). We provide a compressed file of our MonetDB database (to DfAnalyzer) for local execution of the project DfAnalyzer. Therefore, users only need to run the script start-dfanalyzer.sh at the path DfAnalyzer. We assume the execution of these steps with a Unix-based operating system, as follows:

```

cd DfAnalyzer
./start-dfanalyzer.sh

```

## How to run DNN applications

The DLProv has a few predefined hyperparameters (e.g. optimizer, learning rate, number of epochs, number of layers, etc.) and metrics (e.g. loss, accuracy, elapsed time) to be captured. In the case that these hyperparameters and metrics are enough, the user has to set the attribute “hyperparameters” as True, and the library will take care of it. It's important to set a tag to identify the workflow and associate it with the provenance data, e.g. hyperparameters. This method captures provenance data as the deep learning workflow executes and sends them to the provenance database managed by MonetDB. As the data reaches the database, it can be analyzed through the Dataflow Viewer (DfViewer), Query Interface (QI), and Query Dashboard (QP). The data received by the provenance method are defined by the user in the source code of the DNN application, as follows:

```
df = Dataflow(dataflow_tag, hyperparameters=True)
df.save()
```

To capture the retrospective provenance, the user should add the following code:

```
tf1_input = DataSet("iTrainingModel", [Element([opt.get_config()['name'], opt.get_config()['learning_rate'], epochs, len(model.layers)])])
t1.add_dataset(tf1_input)
t1.begin() 

## Data manipulation

tf1_output = DataSet("oTrainingModel", [Element([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), elapsed_time, loss, accuracy, val_loss, val_accuracy, epoch])])
t1.add_dataset(tf1_output)
if(epoch==final_epoch):
	t1.end()
else:
	t1.save()    
```

In case there is an adaptation of the hyperparameters during training (e.g., an update of the learning rate), that is, the use of methods such as LearningRateScheduler offered by Keras, the hyperparameter’s values are updated, therefore, the adaptation should be registered for further analysis. To capture these data, the user should add code for this specific transformation.

## Example

The path `Example` shows how to use Keras-Prov. To run it, the user just needs to run the Python command, as follows: 

```
python alexnet_DLProv.py
```

To add new parameters, hyperparameters or metrics to be captured and stored, the user needs to specify the new transformation. For example, if they want to capture data related to the DNN architecture like a dense block (growth rate and number of layers in the dense block), the specification has to be added before the model.fit command on user's training code and should be like:

```
df = model.get_dataflow()

tf_denseb = Transformation("DenseBlock")
tf_denseb_input = Set("iDenseBlock", SetType.INPUT, 
    [Attribute("growth_rate", AttributeType.NUMERIC), 
    Attribute("layers_db", AttributeType.NUMERIC)])
tf_denseb_output = Set("oDenseBlock", SetType.OUTPUT, 
    [Attribute("output", AttributeType.TEXT)])
tf_denseb.set_sets([tf_denseb_input, tf_denseb_output])
df.add_transformation(tf_denseb) 
```

The second step is the moment when the user must instrument the code to capture the parameter value. For example:

```
t_denseb = Task(identifier=4, dataflow_tag, "DenseBlock")
##Data manipulation, example:
growth_rate = 1
layers_db = 33
t_denseb_input = DataSet("iExtrairNumeros", [Element([growth_rate, layers_db])])
t_denseb.add_dataset(t_denseb_input)
t_denseb.begin()
##Data manipulation, example:
t_denseb_output= DataSet("oExtrairNumeros", [Element([output])])
t_denseb.add_dataset(t_denseb_output)
t_denseb.end()
```

Both steps, the specification of the transformation and the activity definition follow the definitions of [dfa-lib-python](http://monografias.poli.ufrj.br/monografias/monopoli10026387.pdf) for DfAnalyzer.


## Presentation Video

To watch the video, please, click [here](https://www.youtube.com/watch?v=QOZY2CQfXJ8).

## Publications

* [Pina, D., Chapman, A., De Oliveira, D., & Mattoso, M. (2023, April). Deep learning provenance data integration: a practical approach. In Companion Proceedings of the ACM Web Conference 2023 (pp. 1542-1550).](https://dl.acm.org/doi/abs/10.1145/3543873.3587561)
