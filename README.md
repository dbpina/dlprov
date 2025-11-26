# DLProv and Dataverse Integration

This repository provides an example of how to run DLProv with Dataverse, capturing provenance during Deep Learning workflows and ingesting all related artifacts into Dataverse.

## 1. Clone the Repository

Start by cloning this repository to your local machine.

## 2. Add Configuration Files

Next, place your configuration files inside:

```

lib_python/dfa_lib_python/

````

You **must** add two files:

### Dataverse Connection Settings (`dataverse_config.json`)

This file provides the necessary credentials for the DLProv script to authenticate and connect to your Dataverse instance.

⚠️ Warning: This file contains your API_TOKEN. DO NOT commit this file to a public repository like GitHub.

```json
{
  "BASE_URL": "YOUR_DATAVERSE_INSTANCE", 
  "API_TOKEN": "YOUR_PERSONAL_API_TOKEN_HERE", 
  "NOME_DATAVERSE": "DLProv",
  "SUB_DATAVERSE": "AlexNet"
}
```

### Dataset Metadata (`dataset3.json`)

This file must contain the metadata expected by the Dataverse API (e.g., title, author, description, subject). This metadata is crucial for creating and publishing the new dataset. Please ensure the structure is complete and valid according to the Dataverse native API schema. A simple example structure is shown below:

```json
{
  "datasetVersion": {
    "metadataBlocks": {
      "citation": {
        "fields": [
          {
            "typeName": "title",
            "value": "My Deep Learning Experiment Results"
          },
          {
            "typeName": "author",
            "value": [
              {
                "authorName": {
                  "value": "Smith, Jane"
                },
                "authorAffiliation": {
                  "value": "Your Institution"
                }
              }
            ]
          }
        ]
      }
    }
  }
}
```

Fill in both files with your own Dataverse and dataset information. For more details, contact us.


## 3. Install `dfa_lib_python`

From inside the library folder, install the Python package:

```bash
cd lib_python
python setup.py install
```

## 4. Download DLProv Server (Large File Notice)

### Downloading Required Large File

Due to Git LFS restrictions, this repository requires a file that cannot be tracked directly by Git.

Please complete the following steps:

1. Download the file from **[this Google Drive link](https://drive.google.com/file/d/12ICfutGJ0pULDpUXcdBVT0iCz03w0dlv/view?usp=share_link)**.

2. Move the downloaded file into:

```
dlprov_dataverse/DfAnalyzer/target
```

## 5. Running the Example

### Initialize the MonetDB Database (First Time Only)

Before running experiments for the first time, initialize the database.
Be aware that `restore-database.sh` **deletes all existing data**, so use it with caution.

```bash
cd dlprov_dataverse/DfAnalyzer
./restore-database.sh
```

### Run the Experiment Script

Navigate to the main folder:

```bash
cd ..
```

The script `run_experiment.sh` will:

* Start MonetDB and the DLProv server
* Train a DL model (default: a few epochs; feel free to adjust)
* Generate the provenance document
* Upload all data, preprocessed data, model, weights, and provenance files to Dataverse

To execute:

```bash
./run_experiment.sh
```

## 6. Explore the Stored Data in Dataverse

Once the experiment completes, you can inspect the ingested datasets directly in your Dataverse collection.

## 7. Submitting Queries to MonetDB

While the experiment runs, and data is being uploaded, provenance is continuously captured and stored in the provenance database. You can query it at any time.

### Connect to MonetDB:

```bash
mclient -u monetdb -d dataflow_analyzer
```

(Default password: `monetdb`)

### Example Queries

List the main provenance table:

```sql
SELECT * FROM dataflow;
```

```sql
SELECT * FROM dataflow_execution; (This will show the execution identifier.)
```

To analyze data related to the training process, switch to the schema with:

```
SET SCHEMA "alexnet-dverse";
```

Then, to view available tables, use:

```sql
\d
```

For specific data, you can submit queries like:

```sql
SELECT * FROM itrainmodel; to see the hyperparameters.
```

```sql
SELECT * FROM otrainmodel; to view training metrics.
```

```sql
SELECT * FROM otestmodel; to see test metrics.
```

For more details, see the **[DLProv repository](https://github.com/dbpina/dlprov)**.