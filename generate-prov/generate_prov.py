import pymonetdb
import uuid
from prov.model import ProvDocument
from prov.dot import prov_to_dot
import os
import sys
import json
import argparse
import re


def run(args_list=None):
    parser = argparse.ArgumentParser(description="Generate W3C document")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--df_tag', type=str, help='Dataflow tag for generating W3C document for all executions.')
    group.add_argument('--df_exec', type=str, help='Dataflow execution tag for generating W3C document for a specific execution.')

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    if args.df_exec:
        print(f"Generating W3C document for execution: {args.df_exec}")
    elif args.df_tag:
        print(f"Generating W3C document for tag: {args.df_tag}")
    
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
        generate_w3c_for_all_runs(args.df_tag, cursor, conn, prov_document)
        w3c_name = f'{args.df_tag}' 
    elif args.df_exec:
        generate_w3c_for_specific_execution(args.df_exec, cursor, conn, prov_document) 
        w3c_name = f'{args.df_exec}'.replace(" ", "_")
        w3c_name = re.sub(r"[/:]", "-", w3c_name)  # Replace slashes and colons
        w3c_name = re.sub(r"[^\w\-.]", "", w3c_name)

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
    dot_file.write_png(full_filename) 
    dot_file.write_pdf(full_filename_pdf) 

    prov_n_content = prov_document.serialize(format='provn')
    with open(full_filename_provn, 'w') as f:
       f.write(prov_n_content)        

    prov_n_content = prov_document.serialize(format='json')
    with open(full_filename_json, 'w') as f:
        f.write(prov_n_content)                

def ensure_table_uuid(cursor, conn, schema, table, id_column, id_value, extra_where_clause=None, extra_params=None):
    """
    Ensures a UUID exists for a record in any table. Returns the existing UUID or sets a new one.

    Parameters:
        cursor: Database cursor
        conn: Database connection
        schema (str): Schema name (e.g., 'public')
        table (str): Table name (e.g., 'dataflow')
        id_column (str): Name of the primary key column (e.g., 'id', 'tag')
        id_value (Any): Value of the primary key
        extra_where_clause (str): Optional additional WHERE clause (e.g., "AND df_exec = %s")
        extra_params (tuple): Parameters to match the extra_where_clause, if provided

    Returns:
        str: UUID (either existing or newly inserted)
    """

    where_clause = f"{id_column} = %s"
    params = [id_value]

    if extra_where_clause and extra_params:
        where_clause += f" {extra_where_clause}"
        params.extend(extra_params)

    select_query = f'SELECT uuid FROM "{schema}"."{table}" WHERE {where_clause};'
    cursor.execute(select_query, tuple(params))
    result = cursor.fetchone()

    if result and result[0] is not None:
        return str(result[0])

    new_uuid = str(uuid.uuid4())
    update_query = f'UPDATE "{schema}"."{table}" SET uuid = %s WHERE {where_clause};'
    cursor.execute(update_query, tuple([new_uuid] + params))
    conn.commit()
    return new_uuid

def generate_each_execution(df_tag, df_exec, prov_df_exec_activity, cursor, conn, prov_document): 
    query_0 = f"SELECT id FROM \"public\".dataflow WHERE tag = '{df_tag}';"
    cursor.execute(query_0)
    df_id = cursor.fetchone()[0]

    query_0_1 = f"SELECT id, tag FROM \"public\".data_transformation WHERE df_id = '{df_id}';"
    cursor.execute(query_0_1)
    items = cursor.fetchall()
    dt_ids, dt_tags = [row[0] for row in items], [row[1] for row in items]
    dts = {dt_ids[i]: dt_tags[i] for i in range(len(dt_ids))}

    query_1 = f"SELECT id, dt_id FROM \"public\".task WHERE df_exec = '{df_exec}' AND df_version = (SELECT version FROM \"public\".dataflow_version WHERE df_id = '{df_id}');"
    cursor.execute(query_1)
    items = cursor.fetchall()
    task_ids, task_dt_ids = [row[0] for row in items], [row[1] for row in items]
    tasks_dt = {task_ids[i]: task_dt_ids[i] for i in range(len(task_ids))}

    # Get task_id for each transformation
    tasks = {}
    the_activities = []
    the_activities_dict = {}
    other_attributes = {}    

    for task_id in task_ids:
        other_attributes["dlprov:dt_tag"] = dts[tasks_dt[task_id]]
        other_attributes["dlprov:task_id"] = task_id
        uuid_task = ensure_table_uuid(cursor, conn, "public", "task", "id", task_id, extra_where_clause="AND df_exec = %s", extra_params=(df_exec,))
        activity_id = "dlprov:" + str(uuid_task)
        prov_activity = prov_document.activity(activity_id, other_attributes=other_attributes)
        the_activities.append(prov_activity)
        the_activities_dict[task_id] = prov_activity
        prov_document.wasInformedBy(prov_activity, prov_df_exec_activity)

        query_2 = f"SELECT id FROM \"public\".task WHERE dt_id = {tasks_dt[task_id]} AND df_exec = '{df_exec}';"
        cursor.execute(query_2)
        tasks[tasks_dt[task_id]] = [row[0] for row in cursor.fetchall()]

    # Get all data sets, i.e. entities
    query_3 = f"SELECT id, tag FROM \"public\".data_set WHERE df_id = '{df_id}';"
    cursor.execute(query_3)
    items = cursor.fetchall()
    ds_ids, ds_tags = [row[0] for row in items], [row[1] for row in items]

    res = {ds_tags[i]: ds_ids[i] for i in range(len(ds_tags))}         

    the_entities = []
    entities_dict = {}

    #with open("output_log.txt", "w") as file:
    for ds_tag in ds_tags:
        #file.write(f"Processing ds_tag: {ds_tag}\n")

        query = f"SELECT name FROM \"public\".attribute WHERE ds_id = {res[ds_tag]};"
        cursor.execute(query)
        #attribute_list = [row[0] for row in cursor.fetchall()]
        attribute_list = ['id'] + [row[0] for row in cursor.fetchall()]

        query = f"SELECT previous_dt_id, next_dt_id FROM \"public\".data_dependency WHERE ds_id = {res[ds_tag]};"
        cursor.execute(query)
        rows = cursor.fetchall()  # Fetch all rows

        for row in rows:  # Iterate through each row
            #file.write(f"Row for ds_tag {ds_tag}: {row}\n")
            previous_dt_id, next_dt_id = row[0], row[1]

            # Handle None values as needed
            if previous_dt_id is None and next_dt_id is None:
                continue  # Skip if both IDs are None

            # Construct the SELECT clause by joining the attribute names
            select_clause = ', '.join(attribute_list)

    
            # If no entity exists for this ds_tag, create a new one
            #query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE "

            if previous_dt_id is None and next_dt_id is not None:
                select_clause = ', '.join([select_clause, f"{dts[next_dt_id]}_task_id"])
                query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE "
                task_ids = ', '.join(map(str, tasks[next_dt_id]))  # Convert list to string
                query += f"{dts[next_dt_id]}_task_id IN ({task_ids});"
                my_type = "used"
            elif previous_dt_id is not None and next_dt_id is None:
                select_clause = ', '.join([select_clause, f"{dts[previous_dt_id]}_task_id"])
                query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE "
                task_ids = ', '.join(map(str, tasks[previous_dt_id]))
                query += f"{dts[previous_dt_id]}_task_id IN ({task_ids});"
                my_type = "generated"
            elif previous_dt_id is not None and next_dt_id is not None:
                select_clause = ', '.join([select_clause, f"{dts[previous_dt_id]}_task_id", f"{dts[next_dt_id]}_task_id"])
                query = f"SELECT {select_clause} FROM \"{df_tag}\".{ds_tag} WHERE "
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
                #file.write(f"Row data for ds_tag {ds_tag}: {row_data}\n")
                uuid_ds = ensure_table_uuid(cursor, conn, df_tag, "ds_" + ds_tag, "id", row_data[0])
                entity_id = 'dlprov:' + str(uuid_ds)
                other_attributes["dlprov:ds_tag"] = ds_tag

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
                    other_attributes["dlprov:" + attribute_list[i]] = row_data[i]

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

                if ds_tag == "ofilter":
                    prov_document.wasDerivedFrom(prov_entity, entities_dict["oloaddata"][0])  
                    
                if ds_tag == "otrainset" or ds_tag == "ovalset" or ds_tag == "otestset":
                    if "applyfilter" in dt_tags:
                        prov_document.wasDerivedFrom(prov_entity, entities_dict["ofilter"][0])  



def generate_w3c_for_all_runs(df_tag, cursor, conn, prov_document):
    created_user_agents = set()
    created_hw_agents = set()

    query_dfid = f"SELECT id FROM \"public\".dataflow WHERE tag = '{df_tag}';"
    cursor.execute(query_dfid)
    result = cursor.fetchone()

    if result:
        df_id = result[0] 
    else:
        df_id = None

    uuid_df = ensure_table_uuid(cursor, conn, "public", "dataflow", "id", df_id)
    dataflow_entity_id = 'dlprov:' + str(uuid_df)
    dataflow_attributes = {
        "prov:type": "prov:Plan",
        "dlprov:df_tag": df_tag,
        "dlprov:df_id": df_id
    }
    prov_dataflow_entity = prov_document.entity(dataflow_entity_id, other_attributes=dataflow_attributes)

    query_0 = f"SELECT tag FROM \"public\".dataflow_execution WHERE df_id = {df_id};"
    cursor.execute(query_0)
    df_execs = [row[0] for row in cursor.fetchall()]   

    for df_exec in df_execs:  
        uuid_exec = ensure_table_uuid(cursor, conn, "public", "dataflow_execution", "tag", df_exec)
        df_exec_activity_id = 'dlprov:' + str(uuid_exec)
        df_exec_attributes = {"dlprov:exec_tag": df_exec}
        prov_df_exec_activity = prov_document.activity(df_exec_activity_id, other_attributes=df_exec_attributes)

        prov_document.used(prov_df_exec_activity, prov_dataflow_entity)  

        user_attributes = {}
        hw_attributes = {}

        query_user = f"SELECT * FROM \"public\".data_scientist WHERE id = (SELECT scientist_id FROM \"public\".dataflow_execution WHERE tag = '{df_exec}');"
        cursor.execute(query_user)
        row_user = cursor.fetchone()

        if row_user:
            columns = [desc[0] for desc in cursor.description]
            for col, val in zip(columns, row_user):
                user_attributes["dlprov:" + col] = val
                if col == 'id':
                    uuid_user = ensure_table_uuid(cursor, conn, "public", "data_scientist", "id", val)

            entity_user_id = 'dlprov:' + str(uuid_user)
            if entity_user_id not in created_user_agents:
                user_agent = prov_document.agent(entity_user_id, other_attributes=user_attributes)
                created_user_agents.add(entity_user_id)

            prov_document.association(df_exec_activity_id, user_agent)


        query_hw = f"SELECT * FROM \"public\".hardware_info WHERE id = (SELECT hardware_id FROM \"public\".dataflow_execution WHERE tag = '{df_exec}');"
        cursor.execute(query_hw)
        row_hw = cursor.fetchone()

        if row_hw:
            columns = [desc[0] for desc in cursor.description]
            for col, val in zip(columns, row_hw):
                hw_attributes["dlprov:" + col] = val
                if col == 'id':
                    uuid_hw = ensure_table_uuid(cursor, conn, "public", "hardware_info", "id", val)

            entity_hw_id = 'dlprov:' + str(uuid_hw)
            if entity_hw_id not in created_hw_agents:
                hw_agent = prov_document.agent(entity_hw_id, other_attributes=hw_attributes)
                created_hw_agents.add(entity_hw_id)

            prov_document.association(df_exec_activity_id, hw_agent)    

        generate_each_execution(df_tag, df_exec, prov_df_exec_activity, cursor, conn, prov_document)

def generate_w3c_for_specific_execution(df_exec, cursor, conn, prov_document):
    query_0 = f"SELECT id, tag FROM \"public\".dataflow WHERE id = (SELECT df_id FROM \"public\".dataflow_execution WHERE tag = '{df_exec}');"
    cursor.execute(query_0)
    result = cursor.fetchone()

    if result:
        df_id = result[0] 
        df_tag = result[1]
    else:
        df_id = None
        df_tag = None

    uuid_df = ensure_table_uuid(cursor, conn, "public", "dataflow", "id", df_id)
    dataflow_entity_id = 'dlprov:' + str(uuid_df)
    dataflow_attributes = {
        "prov:type": "prov:Plan",
        "dlprov:df_tag": df_tag,
        "dlprov:df_id": df_id
    }
    prov_dataflow_entity = prov_document.entity(dataflow_entity_id, other_attributes=dataflow_attributes)

    uuid_exec = ensure_table_uuid(cursor, conn, "public", "dataflow_execution", "tag", df_exec)
    df_exec_activity_id = 'dlprov:' + str(uuid_exec)
    df_exec_attributes = {"dlprov:exec_tag": df_exec}
    prov_df_exec_activity = prov_document.activity(df_exec_activity_id, other_attributes=df_exec_attributes)

    prov_document.used(prov_df_exec_activity, prov_dataflow_entity)

    user_attributes = {}
    hw_attributes = {}

    query_user = f"SELECT * FROM \"public\".data_scientist WHERE id = (SELECT scientist_id FROM \"public\".dataflow_execution WHERE tag = '{df_exec}');"
    cursor.execute(query_user)
    row_user = cursor.fetchone()

    if row_user:
        columns = [desc[0] for desc in cursor.description]
        for col, val in zip(columns, row_user):
            user_attributes["dlprov:" + col] = val
            if col == 'id':
                uuid_user = ensure_table_uuid(cursor, conn, "public", "data_scientist", "id", val)            
        entity_user_id = 'dlprov:' + str(uuid_user)
        user_agent = prov_document.agent(entity_user_id, other_attributes=user_attributes)  
        prov_document.association(df_exec_activity_id, user_agent)

    query_hw = f"SELECT * FROM \"public\".hardware_info WHERE id = (SELECT hardware_id FROM \"public\".dataflow_execution WHERE tag = '{df_exec}');"
    cursor.execute(query_hw)
    row_hw = cursor.fetchone()

    if row_hw:
        columns = [desc[0] for desc in cursor.description]
        for col, val in zip(columns, row_hw):
            hw_attributes["dlprov:" + col] = val
            if col == 'id':
                uuid_hw = ensure_table_uuid(cursor, conn, "public", "hardware_info", "id", val)
        entity_hw_id = 'dlprov:' + str(uuid_hw)
        hw_agent = prov_document.agent(entity_hw_id, other_attributes=hw_attributes) 
        prov_document.association(df_exec_activity_id, hw_agent)      


    generate_each_execution(df_tag, df_exec, prov_df_exec_activity, cursor, conn, prov_document)        

if __name__ == "__main__":
    run()
