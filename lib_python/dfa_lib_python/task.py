import os
import json
import sys
import requests
from datetime import datetime

# Local imports
from .ProvenanceObject import ProvenanceObject
from .dependency import Dependency
from .task_status import TaskStatus
from .dataset import DataSet
from .performance import Performance
from .system_info import get_system_info
from .dataverse_uploader import define_dataset, upload_file


# # -------------------------------------------------------------------
# # 1.  Helper: resolve JSON files inside the package directory
# # -------------------------------------------------------------------
# MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

# def load_json_resource(filename):
#     """Load a JSON file stored in the dfa_lib_python package directory."""
#     file_path = os.path.join(MODULE_DIR, filename)
#     print(MODULE_DIR)
#     print(file_path)
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"JSON resource not found: {file_path}")
#     with open(file_path, "r") as f:
#         return json.load(f)

# def save_json_resource(filename, obj):
#     """Save a JSON file in the same directory as this module."""
#     file_path = os.path.join(MODULE_DIR, filename)
#     with open(file_path, "w") as f:
#         json.dump(obj, f, indent=4)
#     return file_path

# Folder where example.py is located
CALLER_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
# Alternative if you always run from inside Example/
# CALLER_DIR = os.getcwd()

def load_json_resource(filename):
    """Load a JSON file stored next to example.py."""
    file_path = os.path.join(CALLER_DIR, filename)
    print("[DEBUG] Looking for:", file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON resource not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

def save_json_resource(filename, obj):
    """Save a JSON file next to example.py."""
    file_path = os.path.join(CALLER_DIR, filename)
    print("[DEBUG] Saving:", file_path)
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4)
    return file_path

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))    

def load_json_ds3(filename):
    """Load a JSON file stored in the dfa_lib_python package directory."""
    file_path = os.path.join(MODULE_DIR, filename)
    print(MODULE_DIR)
    print(file_path)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON resource not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

# -------------------------------------------------------------------
# 2.  DFA URL configuration
# -------------------------------------------------------------------
dfa_url = os.environ.get("DFA_URL", "http://localhost:22000/")


# -------------------------------------------------------------------
# 3.  Task class
# -------------------------------------------------------------------
class Task(ProvenanceObject):

    def __init__(self, id, dataflow, exec_tag, transformation_tag,
                 sub_id="", dependency=None, workspace="", resource="",
                 output="", error=""):

        super().__init__(transformation_tag)

        self._first = "0"
        self._dataflow_tag = dataflow._tag
        self._email = dataflow.email
        self._exec = exec_tag

        self._workspace = workspace
        self._resource = resource
        self._dependency = ""
        self._output = output
        self._error = error

        self._sets = []
        self._status = TaskStatus.READY.value
        self._dataflow = dataflow._tag.lower()
        self._transformation = transformation_tag.lower()
        self._id = str(id)
        self._sub_id = sub_id
        self._performances = []
        self.dfa_url = dfa_url

        self.start_time = None
        self.end_time = None
        self._pid = ""

        # Handle dependency object
        if isinstance(dependency, list):
            dependency = Dependency([d._tag for d in dependency],
                                    [d._id for d in dependency])
            self._dependency = dependency.get_specification()

        elif isinstance(dependency, Task):
            dependency = Dependency([dependency._tag], [dependency._id])
            self._dependency = dependency.get_specification()


    # -------------------------------------------------------------------
    # Dependency Handling
    # -------------------------------------------------------------------
    def add_dependency(self, dependency):
        assert isinstance(dependency, Dependency), "Invalid dependency."
        self._dependency = dependency.get_specification()


    # -------------------------------------------------------------------
    # Dataset Handling
    # -------------------------------------------------------------------
    def set_datasets(self, datasets):
        assert isinstance(datasets, list), "Parameter must be a list."
        for dataset in datasets:
            self.add_dataset(dataset)

    def add_dataset(self, dataset):
        assert isinstance(dataset, DataSet), "Invalid dataset."
        self._sets = [dataset.get_specification()]


    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------
    def set_status(self, status):
        assert isinstance(status, TaskStatus), "Invalid status."
        self._status = status.value


    # -------------------------------------------------------------------
    # BEGIN
    # -------------------------------------------------------------------
    def begin(self):

        self.set_status(TaskStatus.RUNNING)
        self.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if self.get_specification()['id'] == "1":
            self._first = "1"

            # System info
            system_info = get_system_info()
            self._hardware = json.dumps(system_info)

            # Load dataset template
            dataset_json = load_json_ds3("dataset3.json")

            # Update metadata
            title_value = self._exec
            description_value = (
                "Dataset usado nos experimentos de treinamento do modelo - "
                + str(self._exec)
            )

            dataset_json["datasetVersion"]["metadataBlocks"]["citation"]["fields"][0]["value"] = title_value

            for field in dataset_json["datasetVersion"]["metadataBlocks"]["citation"]["fields"]:
                if field["typeName"] == "dsDescription":
                    field["value"][0]["dsDescriptionValue"]["value"] = description_value
                    break

            # Save updated dataset JSON
            dataset_final_path = save_json_resource("dataset_final.json", dataset_json)

            # Create dataset in Dataverse
            self._pid = define_dataset(dataset_final_path)

            # Save PID for later use
            save_json_resource("dataset_pid.json", {"dataset_pid": self._pid})

        # Save task execution
        self.save()
        self._sets = []
        self._first = "0"


    # -------------------------------------------------------------------
    # END
    # -------------------------------------------------------------------
    def end(self):

        self.set_status(TaskStatus.FINISHED)
        self.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        performance = Performance(self.start_time, self.end_time)
        self._performances.append(performance.get_specification())

        self.save()
        self._sets = []


    # -------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------
    def save(self):

        # Save Task to DFA
        url = dfa_url + "/pde/task/json"
        message = self.get_specification()

        r = requests.post(url, json=message)
        print(r.status_code)

        # Load Dataset PID
        self._pid = load_json_resource("dataset_pid.json")["dataset_pid"]

        # Attribute -> directory mapping
        ATTRIBUTE_PATH_HINTS = {
            "MODEL_DIR": "models",
            "DATASET_DIR": "data",
            "TrainSet": "preprocessed_data",
            "TestSet": "preprocessed_data",
            "ValSet": "preprocessed_data",
            "WEIGHTS_PATH": "weights",
            "PROV_FILE": "provenance"
        }

        # Load file attributes table
        file_attributes = load_json_resource("file_attrs.json")

        # Upload files for each dataset
        for s in message.get("sets", []):
            tag = s["tag"]
            elements = s.get("elements", [])

            if tag not in file_attributes:
                continue

            for index, attr_name in file_attributes[tag]:
                for element in elements:
                    if len(element) > index:
                        path = element[index]
                        directory_label = ATTRIBUTE_PATH_HINTS.get(attr_name, "misc")
                        print(f"Uploading: {path} → {directory_label}")
                        upload_file(self._pid, path, directory_label)
