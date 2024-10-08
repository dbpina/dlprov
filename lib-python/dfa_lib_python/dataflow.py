import requests
import os
from .ProvenanceObject import ProvenanceObject
from .transformation import Transformation

from .attribute import Attribute
from .attribute_type import AttributeType
from .set import Set
from .set_type import SetType

from .build_w3c import config  # Import the config object

dfa_url = os.environ.get('DFA_URL', "http://localhost:22000/")


class Dataflow(ProvenanceObject):
    """
    This class defines a dataflow.
    
    Attributes:
        - tag (str): Dataflow tag.
        - transformations (list, optional): Dataflow transformations.
    """
    def __init__(self, tag, predefined=False, w3c=None, transformations=[]):
        ProvenanceObject.__init__(self, tag)
        self.transformations = transformations
        self.predefined = predefined
        config.w3c = w3c
        config.transformations_number = len(self.transformations)

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
        config.transformations_number = len(self._transformations)

    @property
    def predefined(self):
        return self._predefined

    @predefined.setter
    def predefined(self, predefined):
        if(predefined == True):
            assert isinstance(predefined, bool), \
                "The parameter must must be a user."   
            tf1 = Transformation("NormalizeData")
            tf1_input = Set("iInputDataset", SetType.INPUT, 
                [Attribute("DATASET_NAME", AttributeType.TEXT), 
                Attribute("DATASET_SOURCE", AttributeType.TEXT)])
            tf1_output = Set("oNormalizeData", SetType.OUTPUT, 
                [Attribute("DATASET_DIR", AttributeType.TEXT)])
            tf1.set_sets([tf1_input, tf1_output])
            self.add_transformation(tf1)

            tf2 = Transformation("SplitData")
            tf2_input = Set("iSplitConfig", SetType.INPUT, 
                [Attribute("RATIO", AttributeType.NUMERIC)])
            tf2_train_output = Set("oTrainSet", SetType.OUTPUT, 
                [Attribute("TrainSet", AttributeType.TEXT)])
            tf2_test_output = Set("oTestSet", SetType.OUTPUT, 
                [Attribute("TestSet", AttributeType.TEXT)])
            tf1_output.set_type(SetType.INPUT)
            tf1_output.dependency=tf1._tag
            tf2.set_sets([tf2_input, tf1_output, tf2_train_output, tf2_test_output])
            self.add_transformation(tf2)

            tf3 = Transformation("TrainModel")
            tf3_input = Set("iTrainModel", SetType.INPUT, 
                [Attribute("OPTIMIZER_NAME", AttributeType.TEXT), 
                Attribute("LEARNING_RATE", AttributeType.NUMERIC),
                Attribute("NUM_EPOCHS", AttributeType.NUMERIC),
                Attribute("NUM_LAYERS", AttributeType.NUMERIC)])
            tf3_output = Set("oTrainModel", SetType.OUTPUT, 
                [Attribute("TIMESTAMP", AttributeType.TEXT), 
                Attribute("ELAPSED_TIME", AttributeType.TEXT),
                Attribute("LOSS", AttributeType.NUMERIC),
                Attribute("ACCURACY", AttributeType.NUMERIC),
                Attribute("VAL_LOSS", AttributeType.NUMERIC),
                Attribute("VAL_ACCURACY", AttributeType.NUMERIC),                
                Attribute("EPOCH", AttributeType.NUMERIC)])
            tf3_output_model = Set("oTrainedModel", SetType.OUTPUT, 
                [Attribute("MODEL_NAME", AttributeType.TEXT),
                Attribute("MODEL_DIR", AttributeType.TEXT)])
            tf2_train_output.set_type(SetType.INPUT)
            tf2_train_output.dependency=tf2._tag
            tf3.set_sets([tf2_train_output, tf3_input, tf3_output, tf3_output_model])
            self.add_transformation(tf3)

            tf4 = Transformation("TestModel")
            tf4_output = Set("oTestModel", SetType.OUTPUT, 
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
