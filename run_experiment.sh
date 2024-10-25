#!/bin/bash

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/

# Change to the DfAnalyzer directory
cd /opt/dlprov/DfAnalyzer

# Start and restore the MonetDB database
echo "Restoring the database..."
#./restore-database.sh  # Uncomment if you need to restore the database
monetdbd stop data || { echo "Failed to stop MonetDB"; exit 1; }
monetdbd start data || { echo "Failed to start MonetDB"; exit 1; }
monetdb start dataflow_analyzer || { echo "Failed to start the dataflow_analyzer database"; exit 1; }

# Start the DfAnalyzer server
echo "Starting the .jar file (server)..."
/opt/jdk1.8.0_66/bin/java -jar target/DfAnalyzer-1.0.jar &

echo "Waiting for the server to start..."
sleep 15  # Adjust the sleep duration if necessary
echo ".jar server started successfully."

# Run mnist example with DLProv
cd /opt/dlprov/Example
echo "Running mnist-simple.py..."

# Clean up any previous temporary data
rm -rf temp_mnist
rm -f mnist-trained.keras

python mnist-simple.py

if [ $? -ne 0 ]; then
    echo "Error: mnist-simple.py script failed to execute."
    exit 1
fi
echo "mnist-simple.py completed successfully."

# Generate provenance document
cd /opt/dlprov/generate-prov

# Restore Neo4j database
python restore_neo4j.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to restore neo4j database."
    exit 1
fi
echo "Neo4j was restored."

echo "Running provenance generation..."

# Run the Python script and capture the output
output=$(python query_execution_tag.py)
echo "output was '$output'"

if [[ $output ]]; then
    python generate_prov.py --df_exec "$output"
    if [ $? -ne 0 ]; then
        echo "Provenance generation not completed."
        exit 1
    fi
    echo "Provenance generation completed successfully."
else
    echo "Error executing the script."
    exit 1
fi

# Safely stop the Java process
pkill -f '/opt/jdk1.8.0_66/bin/java -jar target/DfAnalyzer-1.0.jar'

# Final message
echo "Experiment completed!"