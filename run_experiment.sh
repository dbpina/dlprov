#!/bin/bash

# Change to the DfAnalyzer directory
cd DfAnalyzer

# Start and restore the MonetDB database
echo "Restoring the database..."
#./restore-database.sh  # Uncomment if you need to restore the database
monetdbd stop data || { echo "Failed to stop MonetDB"; exit 1; }
monetdbd start data || { echo "Failed to start MonetDB"; exit 1; }
monetdb start dataflow_analyzer || { echo "Failed to start the dataflow_analyzer database"; exit 1; }

# Start the DfAnalyzer server
echo "Starting the .jar file (server)..."
java -jar target/DfAnalyzer-2.0.jar &

echo "Waiting for the server to start..."
sleep 15  # Adjust the sleep duration if necessary
echo ".jar server started successfully."

# Run alexnet example with DLProv and Dataverse
cd ..
cd Example
echo "Running alexnet.py..."

python alexnet.py

if [ $? -ne 0 ]; then
    echo "Error: alexnet.py script failed to execute."
    exit 1
fi
echo "alexnet.py completed successfully."

# Safely stop the Java process
pkill -f 'java -jar target/DfAnalyzer-2.0.jar'

# Final message
echo "Experiment completed!"
