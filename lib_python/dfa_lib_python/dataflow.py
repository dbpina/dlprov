import requests
import os
import json
from .ProvenanceObject import ProvenanceObject
from .transformation import Transformation

from .attribute import Attribute
from .attribute_type import AttributeType
from .set import Set
from .set_type import SetType

dfa_url = os.environ.get('DFA_URL', "http://localhost:22000/")


class Dataflow(ProvenanceObject):
    """
    This class defines a dataflow.
    
    Attributes:
        - tag (str): Dataflow tag.
        - transformations (list, optional): Dataflow transformations.
    """
    def __init__(self, tag, predefined=False, transformations=[]):
        ProvenanceObject.__init__(self, tag)
        self.transformations = transformations
        self.predefined = predefined

    @property
    def transformations(self):
        """Get or set the dataflow transformations."""
        return self._transformations

    @transformations.setter
    def transformations(self, transformations):
        assert isinstance(transformations, list), \
            "The Transformations must be in a list."
        result = []
        for transformation in transformations:
            assert isinstance(transformation, Transformation), \
                "The Transformation must be valid."
            result.append(transformation.get_specification())
        self._transformations = result

    def add_transformation(self, transformation):
        """ Add a transformation to the dataflow.

        Args:
            transformation (:obj:`Transformation`): A dataflow transformation.
        """
        assert isinstance(transformation, Transformation), \
            "The parameter must must be a transformation."
        self._transformations.append(transformation.get_specification())

    @property
    def predefined(self):
        return self._predefined

    @predefined.setter
    def predefined(self, predefined):
        if(predefined == True):
            assert isinstance(predefined, bool), \
                "The parameter must must be a user."   
            tf1 = Transformation("LoadData")
            tf1_input = Set("iInputDataset", SetType.INPUT, 
                [Attribute("DATASET_NAME", AttributeType.TEXT), 
                Attribute("DATASET_SOURCE", AttributeType.TEXT)])
            tf1_output = Set("oLoadData", SetType.OUTPUT, 
                [Attribute("DATASET_DIR", AttributeType.FILE)])
            tf1.set_sets([tf1_input, tf1_output])
            self.add_transformation(tf1)

            # tf1_1 = Transformation("RandomHorizontal")
            # tf1_1_output = Set("oRandomHorizontal", SetType.OUTPUT, 
            #     [Attribute("DATASET_DIR", AttributeType.FILE)])
            # tf1_output.set_type(SetType.INPUT)
            # tf1_output.dependency=tf1._tag
            # tf1_1.set_sets([tf1_output, tf1_1_output])
            # self.add_transformation(tf1_1)

            # tf1_2 = Transformation("Normalize")
            # tf1_2_output = Set("oNormalize", SetType.OUTPUT, 
            #     [Attribute("DATASET_DIR", AttributeType.FILE)])
            # tf1_1_output.set_type(SetType.INPUT)
            # tf1_1_output.dependency=tf1_1._tag
            # tf1_2.set_sets([tf1_1_output, tf1_2_output])
            # self.add_transformation(tf1_2)            

            tf2 = Transformation("SplitData")
            tf2_input = Set("iSplitConfig", SetType.INPUT, 
                [Attribute("TRAIN_RATIO", AttributeType.NUMERIC),
                Attribute("VAL_RATIO", AttributeType.NUMERIC),
                Attribute("TEST_RATIO", AttributeType.NUMERIC)])
            tf2_train_output = Set("oTrainSet", SetType.OUTPUT, 
                [Attribute("TrainSet", AttributeType.FILE)])
            tf2_val_output = Set("oValSet", SetType.OUTPUT, 
                [Attribute("ValSet", AttributeType.FILE)])            
            tf2_test_output = Set("oTestSet", SetType.OUTPUT, 
                [Attribute("TestSet", AttributeType.FILE)])
            tf1_output.set_type(SetType.INPUT)
            tf1_output.dependency=tf1._tag
            tf2.set_sets([tf1_output, tf2_input, tf2_train_output, tf2_val_output, tf2_test_output])            
            self.add_transformation(tf2)

            tf3 = Transformation("Train")
            tf3_input = Set("iTrain", SetType.INPUT, 
                [Attribute("OPTIMIZER_NAME", AttributeType.TEXT), 
                Attribute("LEARNING_RATE", AttributeType.NUMERIC),
                Attribute("NUM_EPOCHS", AttributeType.NUMERIC),
                Attribute("BATCH_SIZE", AttributeType.NUMERIC),
                Attribute("NUM_LAYERS", AttributeType.NUMERIC)])
            tf3_output = Set("oTrain", SetType.OUTPUT, 
                [Attribute("TIMESTAMP", AttributeType.TEXT), 
                Attribute("ELAPSED_TIME", AttributeType.TEXT),
                Attribute("LOSS", AttributeType.NUMERIC),
                Attribute("ACCURACY", AttributeType.NUMERIC),
                Attribute("VAL_LOSS", AttributeType.NUMERIC),
                Attribute("VAL_ACCURACY", AttributeType.NUMERIC),                
                Attribute("EPOCH", AttributeType.NUMERIC)])
            tf3_output_model = Set("oTrainedModel", SetType.OUTPUT, 
                [Attribute("MODEL_NAME", AttributeType.TEXT),
                Attribute("MODEL_DIR", AttributeType.FILE)])
            tf3_output_weights = Set("oWeights", SetType.OUTPUT, 
                [Attribute("WEIGHTS_PATH", AttributeType.FILE)])            
            tf2_train_output.set_type(SetType.INPUT)
            tf2_train_output.dependency=tf2._tag
            tf2_val_output.set_type(SetType.INPUT)
            tf2_val_output.dependency=tf2._tag            
            tf3.set_sets([tf2_train_output, tf2_val_output, tf3_input, tf3_output, tf3_output_model])
            self.add_transformation(tf3)

            tf4 = Transformation("Test")
            tf4_output = Set("oTest", SetType.OUTPUT, 
                [Attribute("LOSS", AttributeType.NUMERIC),
                Attribute("ACCURACY", AttributeType.NUMERIC)])
            tf2_test_output.set_type(SetType.INPUT)
            tf2_test_output.dependency=tf2._tag
            tf3_output_model.set_type(SetType.INPUT)
            tf3_output_model.dependency=tf3._tag
            tf4.set_sets([tf2_test_output, tf3_output_model, tf4_output])
            self.add_transformation(tf4)    

    def save(self):
        """ Send a post request to the Dataflow Analyzer API to store
            the dataflow.
        """
        url = dfa_url + '/pde/dataflow/json'
        r = requests.post(url, json=self.get_specification())  
        print(r.status_code)

        self.file_attributes = {}  

        for tf in self.transformations:
            for s in tf.get("sets", []):
                tag = s["tag"]
                file_attrs = [
                    (idx, attr["name"]) for idx, attr in enumerate(s["attributes"])
                    if attr["type"] == "FILE"
                ]
                if file_attrs:
                    self.file_attributes[tag] = file_attrs            

        with open("file_attrs.json", "w") as f:
            json.dump(self.file_attributes, f)


