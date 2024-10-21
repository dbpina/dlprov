import pymonetdb
import uuid
from prov.model import ProvDocument
from prov.dot import prov_to_dot
import os
import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate W3C document")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--df_tag', type=str, help='Dataflow tag for generating W3C document for all executions.')
    group.add_argument('--df_exec', type=str, help='Dataflow execution tag for generating W3C document for a specific execution.')

    args = parser.parse_args()
    
    prov_document = ProvDocument()
    prov_document.add_namespace('dlprov', 'DLProv')  

    URL = "localhost"
    PORT = 50000
    DATABASE = "dataflow_analyzer"
    USERNAME = "monetdb"
    PASSWORD = "monetdb"

    w3c_name = ""

    conn = pymonetdb.connect(hostname=URL, port=PORT, database=DATABASE, username=USERNAME, password=PASSWORD)

    cursor = conn.cursor()

    if args.df_tag:
        print(f"Generating W3C document for all execution of a dataflow: {args.df_tag}")
        generate_w3c_for_all_runs(args.df_tag, cursor, prov_document)
        w3c_name = f'{args.df_tag}' 
    elif args.df_exec:
        print(f"Generating W3C document for specific execution: {args.df_exec}")
        generate_w3c_for_specific_execution(args.df_exec, cursor, prov_document)
        w3c_name = f'{args.df_exec}' 

    cursor.close()
    conn.close()                

    output_dir = os.path.join(os.getcwd(), "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    full_filename = os.path.join(output_dir, f'{w3c_name}.png')
    full_filename_provn = os.path.join(output_dir, f'{w3c_name}.provn')  
    full_filename_json = os.path.join(output_dir, f'{w3c_name}.json')  

    dot_file = prov_to_dot(prov_document)
    dot_file.write_png(full_filename) 

    prov_n_content = prov_document.serialize(format='provn')
    with open(full_filename_provn, 'w') as f:
        f.write(prov_n_content)        

    prov_n_content = prov_document.serialize(format='json')
    with open(full_filename_json, 'w') as f:
        f.write(prov_n_content)                

def generate_each_execution(df_tag, df_exec, prov_df_exec_activity, cursor, prov_document):            
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
    entities_dict = {}

    with open("output_log.txt", "w") as file:
        for ds_tag in ds_tags:
            file.write(f"Processing ds_tag: {ds_tag}\n")
            query = f"SELECT name FROM \"public\".attribute WHERE ds_id = {res[ds_tag]};"
            cursor.execute(query)
            attribute_list = [row[0] for row in cursor.fetchall()]

            query = f"SELECT previous_dt_id, next_dt_id FROM \"public\".data_dependency WHERE ds_id = {res[ds_tag]};"
            cursor.execute(query)
            rows = cursor.fetchall()  # Fetch all rows

            for row in rows:  # Iterate through each row
                file.write(f"Row for ds_tag {ds_tag}: {row}\n")
                previous_dt_id, next_dt_id = row[0], row[1]

                # Handle None values as needed
                if previous_dt_id is None and next_dt_id is None:
                    continue  # Skip if both IDs are None

                # Construct the SELECT clause by joining the attribute names
                select_clause = ', '.join(attribute_list)

                # If no entity exists for this ds_tag, create a new one
                query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE "

                if previous_dt_id is None and next_dt_id is not None:
                    query += f"{dts[next_dt_id]}_task_id = {tasks[next_dt_id]};"
                    my_type = "used"
                elif previous_dt_id is not None and next_dt_id is None:
                    query += f"{dts[previous_dt_id]}_task_id = {tasks[previous_dt_id]};"
                    my_type = "generated"
                elif previous_dt_id is not None and next_dt_id is not None:
                    query += f"{dts[previous_dt_id]}_task_id = {tasks[previous_dt_id]} AND {dts[next_dt_id]}_task_id = {tasks[next_dt_id]};"
                    my_type = "both"

                cursor.execute(query)
                rows_data = cursor.fetchall()

                # Create a new prov_entity for the ds_tag
                other_attributes = {}
                for row_data in rows_data:
                    file.write(f"Row data for ds_tag {ds_tag}: {row_data}\n")
                    entity_id = 'dlprov:' + str(uuid.uuid4())
                    other_attributes["dlprov:ds_tag"] = ds_tag
                    for i in range(len(row_data)):
                        other_attributes["dlprov:" + attribute_list[i] + ":"] = row_data[i]

                    # Create the prov_entity and store it in the dictionary
                    prov_entity = prov_document.entity(entity_id, other_attributes=other_attributes)

                    if ds_tag not in entities_dict:
                        entities_dict[ds_tag] = []
                    entities_dict[ds_tag].append(prov_entity)
                    the_entities.append(prov_entity)

                    # Add the usage or generation relations
                    if my_type == "used":
                        prov_document.used(the_activities_dict[dts[next_dt_id]], prov_entity)
                    elif my_type == "generated":
                        prov_document.wasGeneratedBy(prov_entity, the_activities_dict[dts[previous_dt_id]])
                    elif my_type == "both":
                        prov_document.used(the_activities_dict[dts[next_dt_id]], prov_entity)
                        prov_document.wasGeneratedBy(prov_entity, the_activities_dict[dts[previous_dt_id]])  

def generate_w3c_for_all_runs(df_tag, cursor, prov_document):
    dataflow_entity_id = 'dlprov:' + str(uuid.uuid4())
    dataflow_attributes = {
        "prov:type:": "prov:Plan",
        "dlprov:tag:": df_tag
    }
    prov_dataflow_entity = prov_document.entity(dataflow_entity_id, other_attributes=dataflow_attributes)

    query_0 = f"SELECT tag FROM \"public\".dataflow_execution WHERE df_id = (SELECT id FROM dataflow WHERE tag = '{df_tag}');"
    cursor.execute(query_0)
    df_execs = [row[0] for row in cursor.fetchall()]

    for exec_tag in df_execs:  
        df_exec_activity_id = 'dlprov:' + str(uuid.uuid4())
        df_exec_attributes = {"dlprov:exec_tag:": exec_tag}
        prov_df_exec_activity = prov_document.activity(df_exec_activity_id, other_attributes=df_exec_attributes)

        prov_document.used(prov_df_exec_activity, prov_dataflow_entity)

        generate_each_execution(df_tag, exec_tag, prov_df_exec_activity, cursor, prov_document)

def generate_w3c_for_specific_execution(df_exec, cursor, prov_document):
    query_0 = f"SELECT tag FROM \"public\".dataflow WHERE id = (SELECT df_id FROM \"public\".dataflow_execution WHERE tag = '{df_exec}');"
    cursor.execute(query_0)
    df_tag = cursor.fetchone()[0]

    dataflow_entity_id = 'dlprov:' + str(uuid.uuid4())
    dataflow_attributes = {
        "prov:type:": "prov:Plan",
        "dlprov:tag:": df_tag
    }
    prov_dataflow_entity = prov_document.entity(dataflow_entity_id, other_attributes=dataflow_attributes)

    df_exec_activity_id = 'dlprov:' + str(uuid.uuid4())
    df_exec_attributes = {"dlprov:exec_tag:": df_exec}
    prov_df_exec_activity = prov_document.activity(df_exec_activity_id, other_attributes=df_exec_attributes)

    prov_document.used(prov_df_exec_activity, prov_dataflow_entity)

    generate_each_execution(df_tag, df_exec, prov_df_exec_activity, cursor, prov_document)        

if __name__ == "__main__":
    main()

