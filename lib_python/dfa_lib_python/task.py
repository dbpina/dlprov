import requests
import os
from .ProvenanceObject import ProvenanceObject
from .dependency import Dependency
from .task_status import TaskStatus
from .dataset import DataSet
from .performance import Performance
from datetime import datetime

import json
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataverse_uploader import define_dataset
from dataverse_uploader import upload_file

dfa_url = os.environ.get('DFA_URL',"http://localhost:22000/")


class Task(ProvenanceObject):
    """
    This class defines a dataflow task.

    Attributes:
        - id (:obj:`str`): Task Id.
        - dataflow_tag (:obj:`str`): Dataflow tag.
        - transformation_tag (:obj:`str`): Transformation tag.
        - sub_id (:obj:`str`, optional): Task Sub Id.
        - dependency (:obj:`Task`): Task which the object has a dependency.
        - workspace (:obj:`str`, optional): Task workspace.
        - resource (:obj:`str`, optional): Task resource.
        - output (:obj:`str`, optional): Task output.
        - error (:obj:`str`, optional): Task error.
    """
    def __init__(self, id, dataflow_tag, exec_tag, transformation_tag,
                 sub_id="", dependency=None, workspace="", resource="",
                 output="", error=""):
        ProvenanceObject.__init__(self, transformation_tag)
        self._first = str(0)     
        self._exec = exec_tag
        self._workspace = workspace
        self._resource = resource
        self._dependency = ""
        self._output = output
        self._error = error
        self._sets = []
        self._status = TaskStatus.READY.value
        self._dataflow = dataflow_tag.lower()
        self._transformation = transformation_tag.lower()
        self._id = str(id)
        self._sub_id = sub_id
        self._performances = []
        self.dfa_url = dfa_url
        self.start_time = None
        self.end_time = None
        self._pid = ""
        # if isinstance(dependency, Task):
        #     dependency = Dependency([dependency._tag], [dependency._id])
        #     self._dependency = dependency.get_specification()
        if isinstance(dependency, list):
            dependency = Dependency([d._tag for d in dependency], [d._id for d in dependency])
            self._dependency = dependency.get_specification()
        elif isinstance(dependency, Task):
            dependency = Dependency([dependency._tag], [dependency._id])
            self._dependency = dependency.get_specification()


    def add_dependency(self, dependency):
        """ Add a dependency to the Task.

        Args:
            - dependency (:obj:`Dependency`): A :obj:`Dependency` object.
        """

        assert isinstance(dependency, Dependency), \
            "The dependency must be valid."
        self._dependency = dependency.get_specification()

    def set_datasets(self, datasets):
        """ Set the Task DataSets.

        Args:
            - dataset (:obj:`list`): A :obj:`list` containing :obj:`DataSet` objects.
        """
        assert isinstance(datasets, list), \
            "The parameter must be a list."
        for dataset in datasets:
            self.add_dataset(dataset)

    def add_dataset(self, dataset):
        """ Add a dataset to the Task.

        Args:
            - dataset (:obj:`DataSet`): A :obj:`DataSet` object.
        """
        assert isinstance(dataset, DataSet), "The dataset must be valid."
        self._sets = [dataset.get_specification()]

    def set_status(self, status):
        """ Change the Task Status.

        Args:
            - status (:obj:`TaskStatus`): A :obj:`TaskStatus` object.
        """
        assert isinstance(status, TaskStatus), \
            "The task status must be valid."
        self._status = status.value

    def begin(self):
        """ Send a post request to the Dataflow Analyzer API to store the Task.
        """        
        self.set_status(TaskStatus.RUNNING)
        self.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if(self.get_specification()['id'] == str(1)):
            self._first = str(1)
            with open("dataset3.json", "r") as f:
                dataset_json = json.load(f)

            title_value = self._exec
            description_value = "Dataset usado nos experimentos de treinamento do modelo - " + str(self._exec)

            dataset_json["datasetVersion"]["metadataBlocks"]["citation"]["fields"][0]["value"] = title_value

            for field in dataset_json["datasetVersion"]["metadataBlocks"]["citation"]["fields"]:
                if field["typeName"] == "dsDescription":
                    field["value"][0]["dsDescriptionValue"]["value"] = description_value
                    break

            with open("dataset_final.json", "w") as f:
                json.dump(dataset_json, f, indent=4)

            self._pid = define_dataset("dataset_final.json")
            with open("dataset_pid.json", "w") as f:
                json.dump({"dataset_pid": self._pid}, f)
        self.save()
        self._sets = []
        self._first = str(0)    

    def end(self):
        """ Send a post request to the Dataflow Analyzer API to store the Task.
        """
        self.set_status(TaskStatus.FINISHED)
        self.end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        performance = Performance(self.start_time, self.end_time)
        self._performances.append(performance.get_specification())
        print(self._performances)
        self.save()
        self._sets = []   


    def save(self):
        """ Send a post request to the Dataflow Analyzer API to store the Task.
        """
        url = dfa_url + '/pde/task/json'
        message = self.get_specification()
        r = requests.post(url, json=message)
        print(r.status_code)  

        with open("dataset_pid.json", "r") as f:
            self._pid = json.load(f)["dataset_pid"]

        ATTRIBUTE_PATH_HINTS = {
            "MODEL_DIR": "models",
            "DATASET_DIR": "data",
            "TrainSet": "preprocessed_data",
            "TestSet": "preprocessed_data",
            "ValSet": "preprocessed_data",
            "WEIGHTS_PATH": "weights",
            "PROV_FILE": "provenance"
        }

        with open("file_attrs.json", "r") as f:
            file_attributes = json.load(f)            

        for s in message.get("sets", []):
            tag = s["tag"]
            elements = s.get("elements", [])
            
            if tag in file_attributes:
                for index, attr_name in file_attributes[tag]:
                    for element in elements:
                        if len(element) > index:
                            path = element[index]
                            directory_label = ATTRIBUTE_PATH_HINTS.get(attr_name, "misc")
                            print(f"Uploading: {path} to {directory_label}")
                            persistent_id = upload_file(self._pid, path, directory_label)

      
