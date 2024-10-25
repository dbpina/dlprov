#!/bin/bash

# Clean up any previous temporary data
rm -rf temp_cifar100
rm resnet50-model.keras
rm resnet50-trained.keras

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64/
neo4j stop
neo4j-admin set-initial-password neo4jneo4j
neo4j start


# Change to the DfAnalyzer directory
cd /opt/dlprov/DfAnalyzer

# Start and restore the MonetDB database
echo "Restoring the database..."
./restore-database.sh  # Uncomment if you need to restore the database
#monetdbd stop data || { echo "Failed to stop MonetDB"; exit 1; }
#monetdbd start data || { echo "Failed to start MonetDB"; exit 1; }
#monetdb start dataflow_analyzer || { echo "Failed to start the dataflow_analyzer database"; exit 1; }

# Start the DfAnalyzer server
echo "Starting the .jar file (server)..."
/opt/jdk1.8.0_66/bin/java -jar target/DfAnalyzer-1.0.jar &

echo "Waiting for the server to start..."
sleep 10  # Adjust the sleep duration if necessary
echo ".jar server started successfully."

# Run mnist example with DLProv
cd /opt/dlprov/Example
echo "Running mnist-simple.py..."
python mnist-simple.py

if [ $? -ne 0 ]; then
    echo "Error: mnist-simple.py script failed to execute."
    exit 1
fi
echo "mnist-simple.py completed successfully."

# Safely stop the Java process
pkill -f '/opt/jdk1.8.0_66/bin/java -jar target/DfAnalyzer-1.0.jar'

# Generate provenance document
cd /opt/dlprov/
rm output_log.txt
echo "Running provenance generation..."
python generate_w3c.py

if [ $? -ne 0 ]; then
    echo "Error: provenance generation failed to execute."
    exit 1
fi
echo "Provenance generation completed successfully."

# Final message
echo "Experiment completed!"