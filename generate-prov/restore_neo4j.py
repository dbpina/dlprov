from neo4j import GraphDatabase
import os


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

# Function to test the connection to Neo4j
def test_connection():
    try:
        with driver.session() as session:
            session.run("RETURN 1")  # Simple query to test the connection
        print("Connection to Neo4j was successful.")
        return True
    except Exception as e:
        print(f"Error: Unable to connect to Neo4j - {e}")
        return False

# Function to clear the Neo4j database
def clear_database():
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # Deletes all nodes and relationships
        print("Database cleared successfully.")
    except Exception as e:
        print(f"Error clearing database: {e}")

if test_connection():
    clear_database()
