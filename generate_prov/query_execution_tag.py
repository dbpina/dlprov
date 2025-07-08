import sys
import pymonetdb

def main():
    URL = "localhost"
    PORT = 50000
    DATABASE = "dataflow_analyzer"
    USERNAME = "monetdb"
    PASSWORD = "monetdb"

    conn = pymonetdb.connect(hostname=URL, port=PORT, database=DATABASE, username=USERNAME, password=PASSWORD)

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT tag FROM dataflow_execution;")
        result = cursor.fetchone()
        
        if result and result[0] is not None:
            print(result[0])  
        else:
            print("No tag found.")
            sys.exit(1)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
