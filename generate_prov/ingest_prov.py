import argparse
import os
import re
from prov.model import ProvDocument
from provdbconnector import ProvDb
from provdbconnector import Neo4jAdapter
from neo4j import GraphDatabase

NEO4J_USER = os.environ.get('NEO4J_USERNAME', 'neo4j')
NEO4J_PASS = os.environ.get('NEO4J_PASSWORD', 'neo4jneo4j')
NEO4J_HOST = os.environ.get('NEO4J_HOST', 'localhost')
NEO4J_BOLT_PORT = os.environ.get('NEO4J_BOLT_PORT', '7687')

auth_info = {
    "user_name": NEO4J_USER,
    "user_password": NEO4J_PASS,
    "host": f"{NEO4J_HOST}:{NEO4J_BOLT_PORT}"
}

driver = GraphDatabase.driver(f"bolt://{NEO4J_HOST}:{NEO4J_BOLT_PORT}", auth=(NEO4J_USER, NEO4J_PASS))

def create_database(db_name):
    with driver.session() as session:
        session.run(f"CREATE DATABASE {db_name}")

prov_api = ProvDb(adapter=Neo4jAdapter, auth_info=auth_info)

def read_from_file(file_path):
    with open(file_path, 'r') as prov_json_file:
        prov_json_content = prov_json_file.read()
        return ProvDocument.deserialize(content=prov_json_content, format='json')

def parse_args():
    parser = argparse.ArgumentParser(description="Insert PROVfile into Neo4j")
    parser.add_argument('--file_name', type=str, help="The path to the PROV file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    output_dir = os.path.join(os.getcwd(), "output")

    create_database(re.sub(r'[^a-zA-Z0-9_.-]', '', args.file_name.replace('-', '').replace('.','')))

    if 'json' not in args.file_name:
        args.file_name = args.file_name + '.json'

    provn_file_path = os.path.join(output_dir, args.file_name)

    if not os.path.exists(provn_file_path):
        print(f"Error: The file '{provn_file_path}' does not exist.")
        exit(1)

    prov_document = read_from_file(provn_file_path)

    document_id = prov_api.save_document(prov_document)

    driver.close()
