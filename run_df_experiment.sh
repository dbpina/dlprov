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

# Restore Neo4j database
python restore_neo4j.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to restore neo4j database."
    exit 1
fi
echo "Neo4j was restored."

# Generate provenance document
cd /opt/dlprov/generate-prov

echo "Running provenance generation..."

# Run the Python script and capture the output
output=$(python query_dftag.py)
echo "output was '$output'"

if [[ $output ]]; then
    python generate_prov.py --df_tag "$output"
    if [ $? -ne 0 ]; then
        echo "Provenance generation not completed."
        exit 1
    fi
    echo "Provenance generation completed successfully."
else
    echo "Error executing the script."
    exit 1
fi

killall monetdbd

echo "Experiment completed!"