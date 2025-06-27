import os
import json
from pyDataverse.api import NativeApi
from pyDataverse.models import Dataverse, Dataset, Datafile
from pyDataverse.utils import read_file

CONFIG_PATH = os.environ.get(
    "DATAVERSE_CONFIG",
    "/Users/debora/Documents/Doutorado/dlprov/lib-python/dfa_lib_python/dataverse_config.json"
)
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

BASE_URL = config["BASE_URL"]
API_TOKEN = config["API_TOKEN"]
NOME_DATAVERSE = config["NOME_DATAVERSE"]
SUB_DATAVERSE = config["SUB_DATAVERSE"]

API = NativeApi(BASE_URL, API_TOKEN)

response = API.get_dataverse(SUB_DATAVERSE)

if response.status_code == 200:
    print(f"Successfully accessed the Dataverse: {SUB_DATAVERSE}")
else:
    print(f"Failed to access Dataverse: {SUB_DATAVERSE}")
    print(response.json())

def define_dataset(ds_filename):
    # Criação do Dataset dentro do Dataverse
    print("criando dataset")
    dataset = Dataset()
    dataset.from_json(read_file(ds_filename))
    dataset.validate_json() # Sempre valida, para ver se está com todos os campos. Qualquer campo faltante, ele NÃO CRIA NADA

    # Enviar o Dataset
    response = API.create_dataset(SUB_DATAVERSE, dataset.json())
    print(response.json())
    print(f"Dataset criado: {response.status_code}")

    # Obter o ID do Dataset criado (DOI)
    dataset_pid = response.json()['data']['persistentId']
    return dataset_pid


def upload_file(dataset_pid, file_path, directory_label):
    """Uploads a file to Dataverse and returns the file metadata."""
    print("uploading file")
    print(dataset_pid)
    print(file_path)
    print(directory_label)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File does not exist: {file_path}")

    df = Datafile()
    df.set({
        "pid": dataset_pid,
        "filename": os.path.basename(file_path),
        "directoryLabel": directory_label
    })

    response = API.upload_datafile(dataset_pid, file_path, df.json())
    if response.status_code != 200:
        raise Exception(f"Upload failed: {response.status_code} - {response.text}")

    return response.json()
