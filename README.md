# DLProv

This project, called DLProv, is part of my DSc research in the Program of Systems and Computer Science (PESC), COPPE, at the Federal University of Rio de Janeiro (UFRJ).

DLProv is a service that evolved from [DNNProv](https://github.com/dbpina/dnnprov). Originally rooted in DNNProv, DLProv has expanded its scope and capabilities to accommodate the broader domain of Deep Learning (DL).

DNNProv began as a provenance service designed to support online hyperparameter analysis in DL, integrating retrospective provenance data (r-prov) with typical DNN software data, e.g. hyperparameters, DNN architecture attributes, etc.

A DL life cycle involves several data transformations, such as performing data pre-processing, defining datasets to train and test a deep neural network (DNN), and training and evaluating the DL model. Choosing a final model requires DL model selection, which involves analyzing data from several training configurations (e.g. hyperparameters and DNN architectures). We have understood that tracing training data back to pre-processing operations can provide insights into the model selection step. However, there are challenges in providing an integration of the provenance of these different steps. Therefore, we decided to integrate these steps. DLProv is a prototype for provenance data integration using different capture solutions while maintaining DNNProv capabilities.

## Overview

DLProv is developed on top of [DfAnalyzer](https://gitlab.com/ssvitor/dataflow_analyzer) provenance services. It uses the columnar DBMS MonetDB to support online provenance data analysis and to generate W3C PROV-compliant documents. In addition, these provenance documents can be analyzed through graph DBMS such as Neo4j.

**This repository provides a Docker container for DLProv, which can be found in step [Running an Example in a Docker Environment](#running-an-example-in-a-docker-environment).**

## Software requirements

The following list of software has to be configured/installed for running a DL model training that collects provenance with DLProv.

* [Java](https://java.com/pt-BR/)
* [MonetDB](http://www.monetdb.org/Documentation/UserGuide/Tutorial) and [pymonetdb](https://pypi.org/project/pymonetdb/)
* [neo4j](https://neo4j.com) and [neo4j python](https://pypi.org/project/neo4j/)
* [prov](https://pypi.org/project/prov/), [pydot](https://pypi.org/project/pydot/), and [provdbconnector](https://github.com/DLR-SC/prov-db-connector/tree/master/provdbconnector)
* [DfAnalyzer](https://github.com/dbpina/keras-prov/tree/main/DfAnalyzer)
* [dfa-lib-python](https://github.com/dbpina/keras-prov/tree/main/dfa-lib-python/) 


<!-- ## Installation -->

<!---### RESTful services -->


<!-- ###  Python library: dfa-lib-python

The DLProv library for the programming language Python can be built with the following command lines:

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

``` -->

## Downloading Large File Due to Git LFS Restrictions

Due to Git LFS (Large File Storage) restrictions, this repository includes a file that cannot be tracked by Git. Please follow these steps to download and add the file manually:

1. Download the file from [this Google Drive link](https://drive.google.com/file/d/12ICfutGJ0pULDpUXcdBVT0iCz03w0dlv/view?usp=share_link).


2. After downloading the file, move it to the `dlprov/DfAnalyzer/target` folder in the repository.

3. Once the file is in the correct folder, you can continue with the setup or use the repository as intended.

**Note**: If you are using the provided Docker container, this step is not required, as the necessary files will be automatically handled within the container.

## How to instrument a DL model

The DLProv has a few predefined hyperparameters (e.g. optimizer, learning rate, number of epochs, number of layers, etc.) and metrics (e.g. loss, accuracy, elapsed time) to be captured. In the case that these hyperparameters and metrics are enough, the user has to set the attribute “hyperparameters” as True, and the library will take care of it. It's important to set a tag to identify the workflow and associate it with the provenance data, e.g. hyperparameters. This method captures provenance data as the deep learning workflow executes and sends them to the provenance database managed by MonetDB. As the data reaches the database, it can be analyzed through the Dataflow Viewer (DfViewer), Query Interface (QI), and Query Dashboard (QP). The data received by the provenance method are defined by the user in the source code of the DNN application, as follows:

```
df = Dataflow(dataflow_tag, predefined=True)
df.save()
```

To capture the retrospective provenance, the user should add the following code:

```
tf1_input = DataSet("iTrainModel", [Element([opt.get_config()['name'], opt.get_config()['learning_rate'], epochs, len(model.layers)])])
t1.add_dataset(tf1_input)
t1.begin() 

## Data manipulation

tf1_output = DataSet("oTrainModel", [Element([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), elapsed_time, loss, accuracy, val_loss, val_accuracy, epoch])])
t1.add_dataset(tf1_output)
if(epoch==final_epoch):
	t1.end()
else:
	t1.save()    
```

In case there is an adaptation of the hyperparameters during training (e.g., an update of the learning rate), that is, the use of methods such as LearningRateScheduler offered by Keras, the hyperparameter’s values are updated, therefore, the adaptation should be registered for further analysis. To capture these data, the user should add code for this specific transformation.


## Running an Example in a Docker Environment

We provide a pre-built Docker container image that includes all necessary dependencies and data from this repository, ensuring a consistent and reproducible environment for running the example.

### Steps to Run the Pre-Built Container

1. **Pull the Docker Image**

To get started, pull the pre-built Docker image from the container registry:

```
docker pull dbpina/dlprov-ex1
```

2. **Run the Container**

Once the image is downloaded, run the container with:    

```
docker run -p 7474:7474 -p 7687:7687 -p 22000:22000 -d \
  -e NEO4J_dbms_default__listen__address=0.0.0.0 \
  -e NEO4J_dbms_connector_http_listen__address=0.0.0.0 \
  --name dlprov-container dbpina/dlprov-ex1
```    

Important: If you are using a machine with ARM architecture (such as MacBooks with Apple M1 or M2 chips), use the option `--platform=linux/amd64` to ensure compatibility with the Docker image.

```
docker run --platform=linux/amd64 -p 7474:7474 -p 7687:7687 -p 22000:22000 -d \
  -e NEO4J_dbms_default__listen__address=0.0.0.0 \
  -e NEO4J_dbms_connector_http_listen__address=0.0.0.0 \
  --name dlprov-container dbpina/dlprov-ex1
```

```
docker exec -it dlprov-container /bin/bash
```

3. **Run the example**    

Once you are in the container shell, the first step is to initialize the MonetDB database. This initialization is only required the first time before the experiments. (When running the experiments script, it will only stop and start MonetDB as needed. Note that restore-database.sh deletes all existing data, so use it with caution.)

To start the database, run the following commands:

```
cd /opt/dlprov/DfAnalyzer
./restore-database.sh
```

Now you can start the server:
```
java -jar target/DfAnalyzer-2.0.jar &
```

The & at the end allows the server to run in the background, freeing up the terminal for other commands.

Now you can run an example:

```
cd ..
cd Example
python filter-prov.py
```

<!-- After that, you navigate to the folder `/opt/dlprov/`, where you will find a script named `run_experiment.sh`. This script:

- Starts the database and the server.
- Runs an experiment that trains a DL model on the MNIST dataset (with only a few epochs; you can adjust the epoch count as needed).
- Generates the provenance document.
- Inserts the provenance data into Neo4j for analysis.

To execute the script, use:    

```
cd /opt/dlprov/
./run_experiment.sh
``` -->

4. **Submit a query**

#### Submitting Queries to MonetDB

To submit queries to MonetDB, connect to the database using the following command:

```
mclient -u monetdb -d dataflow_analyzer
```


The default password is `monetdb`. Once connected, you can submit queries such as:

```
SELECT * FROM dataflow;
```

```
SELECT * FROM dataflow_execution; (This will show the execution identifier.)
```

To analyze data related to the training process, switch to the schema with:

```
SET SCHEMA "alexnet";
```

Then, to view available tables, use:

```
\d
```

For specific data, you can submit queries like:

```
SELECT * FROM itrainmodel; to see the hyperparameters.
```

```
SELECT * FROM otrainmodel; to view training metrics.
```

```
SELECT * FROM otestmodel; to see test metrics.
```

#### Generating the **provenance document** for your chosen deep learning model involves identifying the specific execution and running a generation script based on the **W3C PROV standard**.

Here is the improved Markdown guide:


## Generating the DL Model Provenance Document

After exploring the provenance data in MonetDB, the next step is to select one of the analyzed runs as if it were the chosen model for deployment in a production environment. This choice should be based on the results observed in the queries performed, such as hyperparameter values, performance metrics (e.g., loss and accuracy), use of preprocessing, scientist responsible for the run, and the computational environment used.

Based on the selected execution, you can generate a provenance document that follows the **W3C PROV standard**. This document allows you to visualize, in a structured way, the relationships between **entities**, **activities**, and **agents** involved in the execution, facilitating the understanding of the data flow.

### 1\. Identify the Execution Identifier (`df_exec`)

First, you need to find the specific **execution identifier** (`df_exec`) corresponding to your chosen model. You can obtain this value by querying the `dataflow_execution` table in **MonetDB**:

```sql
SELECT * FROM dataflow_execution;
```

### 2\. Run the Provenance Generation Script

After locating the desired `df_exec` identifier, navigate to the generation directory and run the following command. **Replace `<include exec_tag>` with the value of the selected execution identifier.**

```bash
cd /opt/dlprov/generate-prov
python generate_prov.py --df_exec "<include exec_tag>"
```


### 3\. Review the Output Files

This command will create the provenance files related to the chosen execution in the `/opt/dlprov/generate-prov/output` directory. The generated files will have the following formats:

  * **`.pdf`**: A **graphical visualization** based on the **PROV-DM** model.
  * **`.json`**: The structured representation in **PROV-JSON** format, which is useful for programmatic analysis.
  * **`.provn`**: The **textual representation** in **PROV-N** format.

### 4\. Output Comparison

Now, you can check the folder `/opt/dlprov/generate-prov/output` where you will find the provenance document for your experiment, named something like `alexnet-<timestamp>`. You can compare it with the example file, `mnist-example`, provided in the directory `/opt/dlprov/output/`. There are `.json`, `.provn`, and `.png` files for review and analysis.

**To visualize the PNG file, follow these steps:**

a. Use the following command to copy the file from the Docker container to your host system:
```
docker cp dlprov-container:/opt/dlprov/output/<insert_file_name.png> </host/path/target>
```

Replace <insert_file_name.png> with the actual name of your PNG file. Replace </host/path/target> with the desired destination path on your host system where you want to save the file.

b. After executing the command, navigate to the specified target directory on your host to view the PNG file.

## Exploring and Analyzing Provenance Data

The files generated in the previous step (e.g., `.provn`, `.json`, `.pdf`) all contain the same provenance information but are represented in different formats. An important step for verification and detailed analysis is to leverage the **W3C PROV Validator** and the **Neo4j graph database**.

### 1\. Validating the PROV-N Structure

The **`.provn`** file provides a clean, textual representation of the provenance. For validation and a direct view using an official tool, you can:

1.  Copy the contents of the generated **`.provn`** file.
2.  Paste it into the **W3C PROV Validator** (e.g., `https://openprovenance.org/service/validator.html`).

This allows you to validate the structure against the W3C PROV standard and inspect the data directly.


### 2\. Ingesting Provenance Graphs into Neo4j

After generating and inspecting the documents, the next step is to ingest the provenance data into the **Neo4j graph database** using the DLProv ingestor. This allows for powerful graph-based visualization and querying.

#### Ingestion Command

Execute the following command from the `generate-prov` directory. **Ensure you replace `<include tag>` with the execution identifier (`df_exec`) used to generate the files.**

```bash
python ingest_prov.py --file_name "<include tag>"
```

This command uses the provided **PROV Database Connector** to insert the graph into Neo4j.

#### Accessing Neo4j

To access the Neo4j web interface and verify that the graph was correctly inserted, open your browser and navigate to:

```
http://localhost:7474
```

> **Note:** The `docker run` command used in the setup process should have already exposed the necessary ports to allow this access.

Use the default credentials if requested:

  * **Username:** `neo4j`
  * **Password:** `neo4jneo4j`

#### Initial Verification Query (Cypher)

Once inside the Neo4j interface, you can run the following **Cypher query** to display a sample of the ingested graph, confirming the presence of the entities, activities, and agents:

```cypher
MATCH (n) RETURN n LIMIT 25;
```

> **Difference between MonetDB and Neo4j:** **MonetDB** stores provenance in a **tabular format**, ideal for quantitative analysis. **Neo4j** allows you to visualize and explore provenance as a **graph**, which simplifies understanding the complex structure and derivation paths.

### 3\. Analyzing Derivation Paths with Cypher Queries

The central idea of this step is to develop and execute **Cypher queries** that explore the derivation paths and relationships within the provenance graph.

#### Querying Principles

Connect to the Neo4j browser and use the Cypher query language:

  * **Select an entity** of interest (e.g., a file, a hyperparameter configuration, or a metric).
  * **Traverse the graph** using relationships (like `:wasGeneratedBy`, `:used`, `:wasInformedBy`) to reconstruct the derivation path.
  * Include the **`:wasAssociatedWith`** relationship to check which agents (users/scripts) participated in the activities.

#### Example Cypher Queries for Analysis

1. **Average Training Loss Query**  
This query calculates the average loss for the training activity, providing insights into model performance over training iterations. Other metrics, such as elapsed time, can also be used in place of loss to analyze different aspects of the training process.

```
MATCH (b:Entity)-[:wasGeneratedBy]->(c:Activity)
RETURN avg(b.`dlprov:loss`)
```

```
MATCH (b:Entity)-[:wasGeneratedBy]->(c:Activity)
RETURN avg(toFloat(b.`dlprov:elapsed_time`)) AS avg_elapsed_time
```

2. **Shortest Path Queries**  
These queries find the shortest paths from the resulting test metrics to key components in the workflow:  
- (i) to the data used for model input, tracking data lineage,  
- (ii) to the activity responsible for generating these metrics, helping trace back to the source of the results.

```
MATCH p = shortestPath(
    (a:Entity {`dlprov:ds_tag`: 'otest'})-[*]-
    (b:Entity {`dlprov:ds_tag`: 'oloaddata'})
)
RETURN p
```

```
MATCH p = shortestPath(
    (a:Entity {`dlprov:ds_tag`: 'otest'})-[:wasGeneratedBy]-
    (b:Activity {`dlprov:dt_tag`: 'test'})
)
RETURN p
```

3. **Complete Path Query**  
This query presents the full path from the resulting test metrics to the original input dataset, detailing each step in the data processing pipeline. Information about the dataset source and intermediate transformations is included to support data traceability.

```
MATCH p = (a:Entity {`dlprov:ds_tag`: 'otest'})-[*]-(b:Entity {`dlprov:ds_tag`: 'iinputdataset'})
RETURN p
```

### Note

This project is a work in progress. If you encounter any issues, errors, or have suggestions for improvements, please feel free to contact us. We appreciate your feedback as we continue to refine and expand this project. 



<!-- 5. **Generate provenance graph for several executions**

If you would like to generate a W3C PROV document for multiple executions of the same DL model (for example, after running two training executions), you can do so by running the following script:

```
./run_df_experiment.sh
```

This script first restores the Neo4j database, as the current Neo4j version only supports one active database. After the restoration, it generates the provenance document and inserts it into Neo4j, allowing you to analyze the provenance data using the commands previously provided. -->
 


<!-- ## Example

The path `Example` shows how to use DLProv. To run it, the user just needs to run the Python command, as follows: 

```
python mnist-simple.py
```

To add new parameters, hyperparameters or metrics to be captured and stored, the user needs to specify the new transformation. For example, if they want to capture data related to the DNN architecture like a dense block (growth rate and number of layers in the dense block), the specification has to be added before the model.fit command on user's training code and should be like:

```
df = Dataflow.get_dataflow(dataflow_tag)

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

Both steps, the specification of the transformation and the activity definition follow the definitions of [dfa-lib-python](http://monografias.poli.ufrj.br/monografias/monopoli10026387.pdf) for DfAnalyzer. -->


<!-- ## Presentation Video

To watch the video, please, click [here](https://www.youtube.com/watch?v=QOZY2CQfXJ8). -->


## Publications

* [Pina, D., Chapman, A., Kunstmann, L., de Oliveira, D., & Mattoso, M. (2024, June). DLProv: A Data-Centric Support for Deep Learning Workflow Analyses. In Proceedings of the Eighth Workshop on Data Management for End-to-End Machine Learning (pp. 77-85)](https://dl.acm.org/doi/abs/10.1145/3650203.3663337)

* [Pina, D., Chapman, A., De Oliveira, D., & Mattoso, M. (2023, April). Deep learning provenance data integration: a practical approach. In Companion Proceedings of the ACM Web Conference 2023 (pp. 1542-1550).](https://dl.acm.org/doi/abs/10.1145/3543873.3587561)

* [de Oliveira, L. S., Kunstmann, L., Pina, D., de Oliveira, D., & Mattoso, M. (2023, October). PINNProv: Provenance for Physics-Informed Neural Networks. In 2023 International Symposium on Computer Architecture and High Performance Computing Workshops (SBAC-PADW) (pp. 16-23). IEEE.](https://ieeexplore.ieee.org/abstract/document/10306106?casa_token=iv1zibycPjMAAAAA:cRwbSq1IoyZSTInaxtVql98KYyDyHgM9vJBiEQuWIr7x_USngIQXBur07mMGeypm0KHKgVaPg0eF)
