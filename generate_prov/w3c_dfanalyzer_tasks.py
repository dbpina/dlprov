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
    group.add_argument('--df_tag', type=str, help='Dataflow tag for generating W3C document.')

    args = parser.parse_args()
    
    prov_document = ProvDocument()
    prov_document.add_namespace('dfanalyzer', 'DfAnalyzer')  

    URL = "localhost"
    PORT = 50000
    DATABASE = "dataflow_analyzer"
    USERNAME = "monetdb"
    PASSWORD = "monetdb"

    w3c_name = ""

    conn = pymonetdb.connect(hostname=URL, port=PORT, database=DATABASE, username=USERNAME, password=PASSWORD)

    cursor = conn.cursor()

    if args.df_tag:
        print(f"Generating W3C document for the dataflow: {args.df_tag}")
        generate_w3cprov(args.df_tag, cursor, prov_document)
        w3c_name = f'{args.df_tag}' 
    else:
        print(f"Please provide the dataflow tag.")

    cursor.close()
    conn.close()                

    output_dir = os.path.join(os.getcwd(), "output")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    full_filename = os.path.join(output_dir, f'{w3c_name}.png')
    full_filename_pdf = os.path.join(output_dir, f'{w3c_name}.pdf')
    full_filename_provn = os.path.join(output_dir, f'{w3c_name}.provn')  
    full_filename_json = os.path.join(output_dir, f'{w3c_name}.json')  

    dot_file = prov_to_dot(prov_document)
    #dot_file.write_png(full_filename) 
    dot_file.write_pdf(full_filename_pdf) 

    prov_n_content = prov_document.serialize(format='provn')
    with open(full_filename_provn, 'w') as f:
       f.write(prov_n_content)       

    prov_n_content = prov_document.serialize(format='json')
    with open(full_filename_json, 'w') as f:
        f.write(prov_n_content)                

def generate_w3cprov(df_tag, cursor, prov_document):
    with open("output_log.txt", "w") as file:
        query_0 = f"SELECT id FROM \"public\".dataflow WHERE tag = '{df_tag}';"
        cursor.execute(query_0)
        df_id = cursor.fetchone()[0]

        # dataflow_entity_id = 'dfanalyzer:' + str(uuid.uuid4())
        # dataflow_attributes = {
        #     "prov:type": "prov:Plan",
        #     "dfanalyzer:tag": df_tag
        # }
        # prov_dataflow_entity = prov_document.entity(dataflow_entity_id, other_attributes=dataflow_attributes)

        file.write(f"Processing df_tag: {df_tag}\n")
     
        query_0_1 = f"SELECT id, tag FROM \"public\".data_transformation WHERE df_id = (SELECT id FROM \"public\".dataflow WHERE tag = '{df_tag}');"
        cursor.execute(query_0_1)
        items = cursor.fetchall()
        dt_ids, dt_tags = [row[0] for row in items], [row[1] for row in items]
        dts = {dt_ids[i]: dt_tags[i] for i in range(len(dt_ids))}

        file.write(f"Processing transformations \n")

        query_1 = f"SELECT identifier, dt_id FROM \"public\".task WHERE df_version = (SELECT version FROM \"public\".dataflow_version WHERE df_id = '{df_id}');"
        cursor.execute(query_1)
        items = cursor.fetchall()
        task_ids, task_dt_ids = [row[0] for row in items], [row[1] for row in items]
        tasks_dt = {task_ids[i]: task_dt_ids[i] for i in range(len(task_ids))}
        file.write(f"Processing tasks\n")


        tasks = {}
        the_activities = []
        the_activities_dict = {}
        other_attributes = {}  

        # df_exec_activity_id = 'dfanalyzer:' + str(uuid.uuid4())
        # df_exec_attributes = {"dfanalyzer:exec_tag": "execution"}
        # prov_df_exec_activity = prov_document.activity(df_exec_activity_id, other_attributes=df_exec_attributes)     

        for task_id in task_ids:
            other_attributes["dfanalyzer:dt_tag"] = dts[tasks_dt[task_id]]
            other_attributes["dfanalyzer:task_id"] = task_id
            activity_id = 'dfanalyzer:' + str(uuid.uuid4())
            prov_activity = prov_document.activity(activity_id, other_attributes=other_attributes)
            the_activities.append(prov_activity)
            the_activities_dict[task_id] = prov_activity
            # prov_document.wasInformedBy(prov_activity, prov_df_exec_activity)

            query_2 = f"SELECT id FROM \"public\".task WHERE dt_id = {tasks_dt[task_id]};"
            cursor.execute(query_2)
            tasks[tasks_dt[task_id]] = [row[0] for row in cursor.fetchall()]

        # Get task_id for each transformation
        # tasks = {}
        # the_activities = []
        # the_activities_dict = {}
        # other_attributes = {}  

        # for dt_id in dt_ids:
        #     other_attributes["dfanalyzer:dt_tag"] = dts[dt_id]
        #     activity_id = 'dfanalyzer:' + str(uuid.uuid4())
        #     prov_activity = prov_document.activity(activity_id, other_attributes=other_attributes)
        #     the_activities.append(prov_activity)
        #     the_activities_dict[dts[dt_id]] = prov_activity
        #     prov_document.wasInformedBy(prov_activity, prov_df_exec_activity)

        #     query_2 = f"SELECT id FROM \"public\".task WHERE dt_id = {dt_id};"
        #     cursor.execute(query_2)
        #     tasks[dt_id] = [row[0] for row in cursor.fetchall()]
        #     print("These are the tasks")
        #     print(tasks)

        # Get all data sets, i.e. entities
        query_3 = f"SELECT id, tag FROM \"public\".data_set WHERE df_id = '{df_id}';"
        cursor.execute(query_3)
        items = cursor.fetchall()
        ds_ids, ds_tags = [row[0] for row in items], [row[1] for row in items]

        res = {ds_tags[i]: ds_ids[i] for i in range(len(ds_tags))}    

        the_entities = []
        entities_dict = {}

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
                # query = f"SELECT {select_clause} FROM {ds_tag} WHERE "

                if previous_dt_id is None and next_dt_id is not None:
                    select_clause = ', '.join([select_clause, f"{dts[next_dt_id]}_task_id"])
                    query = f"SELECT {select_clause} FROM {ds_tag} WHERE "
                    task_ids = ', '.join(map(str, tasks[next_dt_id]))  # Convert list to string
                    query += f"{dts[next_dt_id]}_task_id IN ({task_ids});"
                    my_type = "used"

                elif previous_dt_id is not None and next_dt_id is None:
                    select_clause = ', '.join([select_clause, f"{dts[previous_dt_id]}_task_id"])
                    query = f"SELECT {select_clause} FROM {ds_tag} WHERE "
                    task_ids = ', '.join(map(str, tasks[previous_dt_id]))
                    query += f"{dts[previous_dt_id]}_task_id IN ({task_ids});"
                    my_type = "generated"

                elif previous_dt_id is not None and next_dt_id is not None:
                    select_clause = ', '.join([select_clause, f"{dts[previous_dt_id]}_task_id", f"{dts[next_dt_id]}_task_id"])
                    query = f"SELECT {select_clause} FROM {ds_tag} WHERE "
                    previous_task_ids = ', '.join(map(str, tasks[previous_dt_id]))
                    next_task_ids = ', '.join(map(str, tasks[next_dt_id]))
                    query += (
                        f"{dts[previous_dt_id]}_task_id IN ({previous_task_ids}) "
                        f"AND {dts[next_dt_id]}_task_id IN ({next_task_ids});"
                    )
                    my_type = "both"

                cursor.execute(query)
                rows_data = cursor.fetchall()


                # Create a new prov_entity for the ds_tag
                other_attributes = {}
                for row_data in rows_data:
                    file.write(f"Row data for ds_tag {ds_tag}: {row_data}\n")
                    entity_id = 'dfanalyzer:' + str(uuid.uuid4())
                    other_attributes["dfanalyzer:ds_tag"] = ds_tag
                    if my_type == "both":
                        used_task = row_data[-1]
                        generated_task = row_data[-2]
                        row_data = row_data[:-2]
                    elif my_type == "generated":
                        generated_task = row_data[-1]
                        row_data = row_data[:-1]
                    elif my_type == "used":
                        used_task = row_data[-1]
                        row_data = row_data[:-1]

                    for i in range(len(row_data)):
                        other_attributes["dfanalyzer:" + attribute_list[i]] = row_data[i]

                    # Create the prov_entity and store it in the dictionary
                    prov_entity = prov_document.entity(entity_id, other_attributes=other_attributes)

                    if ds_tag not in entities_dict:
                        entities_dict[ds_tag] = []
                    entities_dict[ds_tag].append(prov_entity)
                    the_entities.append(prov_entity)


                    # Add the usage or generation relations
                    if my_type == "used":
                        prov_document.used(the_activities_dict[used_task], prov_entity)
                    elif my_type == "generated":
                        prov_document.wasGeneratedBy(prov_entity, the_activities_dict[generated_task])
                    elif my_type == "both":
                        prov_document.used(the_activities_dict[used_task], prov_entity)
                        prov_document.wasGeneratedBy(prov_entity, the_activities_dict[generated_task])  
      

if __name__ == "__main__":
    main()

