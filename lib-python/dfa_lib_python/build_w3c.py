import pymonetdb
import uuid
from prov.model import ProvDocument
from prov.dot import prov_to_dot
import os

import sys
import json

class Config:
    """
    Configuration class to store global settings like w3c.
    The w3c parameter can be set and accessed globally across different modules.
    """
    def __init__(self):
        self._w3c = None
        self._transformations_number = 0

    @property
    def w3c(self):
        return self._w3c 

    @w3c.setter
    def w3c(self, value):
        """Get the current w3c value."""
        self._w3c = value    

    @property
    def transformations_number(self):
        return self._transformations_number 

    @transformations_number.setter
    def transformations_number (self, value):
        """Get the current w3c value."""
        self._transformations_number = value    

config = Config()        


def build_provenance_document(df_tag, df_exec, mode):
    URL = "localhost"
    PORT = 50000
    DATABASE = "dataflow_analyzer"
    USERNAME = "monetdb"
    PASSWORD = "monetdb"

    # Connect to MonetDB
    conn = pymonetdb.connect(hostname=URL, port=PORT, database=DATABASE, username=USERNAME, password=PASSWORD)

    # Create a cursor
    cursor = conn.cursor()

    # Create a new PROV document
    w3c_name = ""
    prov_document = ProvDocument()
    prov_document.add_namespace('dlprov', '')  

    dataflow_entity_created = False  # Flag to check if the entity was already created

    # If w3c is "current", proceed normally
    if mode == "current":
        # Create the dataflow entity for the current execution
        dataflow_entity_id = 'dlprov:' + str(uuid.uuid4())
        dataflow_attributes = {
            "prov:type:": "prov:Plan",
            "dlprov:tag:": df_tag
        }
        prov_dataflow_entity = prov_document.entity(dataflow_entity_id, other_attributes=dataflow_attributes)

        # Create the execution activity for the current execution
        df_exec_activity_id = 'dlprov:' + str(uuid.uuid4())
        df_exec_attributes = {"dlprov:exec_tag:": df_exec}
        prov_df_exec_activity = prov_document.activity(df_exec_activity_id, other_attributes=df_exec_attributes)

        # Relate the execution activity to the dataflow entity
        prov_document.used(prov_df_exec_activity, prov_dataflow_entity)

        generate_each_execution(df_tag, df_exec, prov_df_exec_activity, cursor, prov_document)        

        # Save the file with desired name
        w3c_name = f'w3c-graph-{df_exec}'


    elif mode == "all":
        # Create the dataflow entity only once
        if not dataflow_entity_created:
            dataflow_entity_id = 'dlprov:' + str(uuid.uuid4())
            dataflow_attributes = {
                "prov:type:": "prov:Plan",
                "dlprov:tag:": df_tag
            }
            prov_dataflow_entity = prov_document.entity(dataflow_entity_id, other_attributes=dataflow_attributes)
            dataflow_entity_created = True

        query_0 = f"SELECT tag FROM \"public\".dataflow_execution WHERE df_id = (SELECT id FROM dataflow WHERE tag = '{df_tag}');"
        cursor.execute(query_0)
        df_execs = [row[0] for row in cursor.fetchall()]

        for exec_tag in df_execs:  
            df_exec_activity_id = 'dlprov:' + str(uuid.uuid4())
            df_exec_attributes = {"dlprov:exec_tag:": exec_tag}
            prov_df_exec_activity = prov_document.activity(df_exec_activity_id, other_attributes=df_exec_attributes)

            # Relate each execution activity to the dataflow entity
            prov_document.used(prov_df_exec_activity, prov_dataflow_entity)

            generate_each_execution(df_tag, exec_tag, prov_df_exec_activity, cursor, prov_document)

            w3c_name = f'w3c-graph-{df_tag}'

    cursor.close()
    conn.close()            

    current_directory = os.getcwd()
    full_filename = os.path.join(current_directory, f'{w3c_name}.png')
    full_filename_provn = os.path.join(current_directory, f'{w3c_name}.provn')             

    dot = prov_to_dot(prov_document)
    dot.write_png(full_filename) 

    # Serialize the ProvDocument to PROV-N format
    prov_n_content = prov_document.serialize(format='provn')

    # Save the PROV-N content to a file
    with open(full_filename_provn, 'w') as f:
        f.write(prov_n_content)

    image_path = "w3c-graph.png"
    provn_path = "w3c-prov.provn"

    return image_path, provn_path            

def generate_each_execution(df_tag, df_exec, prov_df_exec_activity, cursor, prov_document):            
    # Get all transformations, i.e. activities
    query_1 = f"SELECT id, tag FROM \"public\".data_transformation WHERE df_id = (SELECT id FROM \"public\".dataflow WHERE tag = '{df_tag}');"
    cursor.execute(query_1)
    items = cursor.fetchall()
    dt_ids, dt_tags = [row[0] for row in items], [row[1] for row in items]
    dts = {dt_ids[i]: dt_tags[i] for i in range(len(dt_ids))}

    # Get task_id for each transformation
    tasks = {}
    the_activities = []
    the_activities_dict = {}
    other_attributes = {}

    for dt_id in dt_ids:
        other_attributes["dlprov:dt_tag:"] = dts[dt_id]
        activity_id = 'dlprov:' + str(uuid.uuid4())
        prov_activity = prov_document.activity(activity_id, other_attributes=other_attributes)
        the_activities.append(prov_activity)
        the_activities_dict[dts[dt_id]] = prov_activity
        prov_document.wasInformedBy(prov_activity, prov_df_exec_activity)

        query_2 = f"SELECT id FROM \"public\".task WHERE df_exec = '{df_exec}' AND dt_id = {dt_id};"
        cursor.execute(query_2)
        tasks[dt_id] = cursor.fetchone()[0]

    # Get all data sets, i.e. entities
    query_3 = f"SELECT id, tag FROM \"public\".data_set WHERE df_id = (SELECT id FROM \"public\".dataflow WHERE tag = '{df_tag}');"
    cursor.execute(query_3)
    items = cursor.fetchall()
    ds_ids, ds_tags = [row[0] for row in items], [row[1] for row in items]

    res = {ds_tags[i]: ds_ids[i] for i in range(len(ds_tags))}

    the_entities = []
    for ds_tag in ds_tags:
        query = f"SELECT name FROM \"public\".attribute WHERE ds_id = {res[ds_tag]};"
        cursor.execute(query)
        attribute_list = [row[0] for row in cursor.fetchall()]

        query = f"SELECT previous_dt_id, next_dt_id FROM \"public\".data_dependency WHERE ds_id = {res[ds_tag]};"
        cursor.execute(query)
        row = cursor.fetchone()
        previous_dt_id, next_dt_id = row[0], row[1]


        # if row:
        #     previous_dt_id, next_dt_id = row[0], row[1]
        # else:
        #     # Set default values if no rows are found
        #     previous_dt_id, next_dt_id = None, None

        # Handle None values as needed
        if previous_dt_id is None:
            # Handle when previous_dt_id is None
            pass

        if next_dt_id is None:
            pass


        # Construct the SELECT clause by joining the attribute names
        select_clause = ', '.join(attribute_list)

        if previous_dt_id is None and next_dt_id is not None:
            query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE {dts[next_dt_id]}_task_id = {tasks[next_dt_id]};"
            my_type = "used"
            #used

        elif previous_dt_id is not None and next_dt_id is None:
            query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE {dts[previous_dt_id]}_task_id = {tasks[previous_dt_id]};"
            # generated
            my_type = "generated"

        elif previous_dt_id is not None and next_dt_id is not None:
            query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE {dts[previous_dt_id]}_task_id = {tasks[previous_dt_id]} AND {dts[next_dt_id]}_task_id = {tasks[next_dt_id]};"
            my_type = "both"

        cursor.execute(query)
        # Fetch all rows from the cursor
        rows = cursor.fetchall()
        other_attributes = {}
        
        #for i in range(len(rows[row])):
        #        other_attributes["dlprov:" + attribute_list[i]] = rows[row][i]
        for row in rows:
            entity_id = 'dlprov:' + str(uuid.uuid4())
            for i in range(len(row)):
                other_attributes["dlprov:" + attribute_list[i] + ":"] = row[i]
            prov_entity = prov_document.entity(entity_id, other_attributes=other_attributes)  
            the_entities.append(prov_entity)

            if (my_type == "used"):
                prov_document.used(the_activities_dict[dts[next_dt_id]], prov_entity)
            elif (my_type == "generated"):
                prov_document.wasGeneratedBy(prov_entity, the_activities_dict[dts[previous_dt_id]])
            elif (my_type == "both"):
                prov_document.used(the_activities_dict[dts[next_dt_id]], prov_entity)
                prov_document.wasGeneratedBy(prov_entity, the_activities_dict[dts[previous_dt_id]])