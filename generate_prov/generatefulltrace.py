import pymonetdb
import pandas as pd
from prov.model import ProvDocument
from prov.dot import prov_to_dot
import os
import uuid
import json

# Connect to MonetDB
def connect_to_db():
    connection_params = {
        'username': 'monetdb',
        'password': 'monetdb',
        'hostname': 'localhost',
        'port': 50000,  
        'database': 'dataflow_analyzer'
    }

    conn = pymonetdb.connect(**connection_params)
    return conn

def get_ids(conn):
    query = "SELECT id, tag FROM dataflow;"
    ids = pd.read_sql(query, conn)
    return ids

def get_related_ids(conn, df_id):
    query = f"SELECT tag FROM data_set WHERE df_id = {df_id}"
    related_ids = pd.read_sql(query, conn)
    return related_ids

def save_dependencies(dependencies, file_name="dependencies.json"):
    with open(file_name, 'w') as file:
        json.dump(dependencies, file, indent=4)

def define_dependencies_interactively(conn, df_id_list, related_ids_dict):
    dependencies = {}

    for df_id in df_id_list[::-1]:
        define_dep = input(f"\nHey, do you want to define a dependency for df_id {df_id}? (yes/no): ").strip().lower()
        
        if define_dep == 'yes':
            related_ids = related_ids_dict.get(df_id, pd.DataFrame())
            dependencies[df_id] = {}
            
            if not related_ids.empty:
                for related_df_id in df_id_list:
                    if related_df_id != df_id:
                        print(f"\nIs there a dependency from df_id {related_df_id}?")
                        has_dependency = input("Type 'yes' or 'no': ").strip().lower()
                        
                        if has_dependency == 'yes':
                            tags = input(f"Type the tag or tags for the dependency from df_id {related_df_id}: ")
                            dependencies[df_id][related_df_id] = tags
                            print(f"Dependency defined: df_id {df_id} depends on df_id {related_df_id} with tags: {tags}.")
            else:
                print(f"No related ids found for df_id {df_id}")
        else:
            print(f"No dependency defined for df_id {df_id}. Moving to the next df_id.")
    
    save_dependencies(dependencies)

def generate_trace(conn, df_id_list, dependencies_file):
    print("Time to generate provenance trace.")     

def main():
    conn = connect_to_db()
    
    ids = get_ids(conn)
    if ids.empty:
        print("No IDs available.")
        return
    
    related_ids_dict = {}

    for _, row in ids.iloc[::-1].iterrows():
        df_id = row['id']
        related_ids = get_related_ids(conn, df_id)
        if not related_ids.empty:
            print(f"\ndf_id = {df_id}: These are the related ids: {related_ids['tag'].tolist()}")
            related_ids_dict[df_id] = related_ids
        else:
            print(f"No related ids found for df_id {df_id}")
        
    id_list = ids['id'].tolist()
    define_dependencies_interactively(conn, id_list, related_ids_dict)


    image_path = generate_trace(conn, ids, "dependencies.json")

    # for df_id in df_id_list:
    #     image = generate_trace(conn, df_id)
    #     print(f"Trace generated: {image}")    

    conn.close()


if __name__ == "__main__":
    main()
