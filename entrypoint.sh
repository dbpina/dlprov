#!/bin/bash
set -e

echo "=== Starting DLProv stack ==="

# Start MonetDB (required for DL Prov data capture)
echo "Starting MonetDB..."
mkdir -p /opt/dlprov/DfAnalyzer/data
monetdbd create /opt/dlprov/DfAnalyzer/data 2>/dev/null || true
monetdbd start /opt/dlprov/DfAnalyzer/data
monetdb create dataflow_analyzer 2>/dev/null || true
monetdb release dataflow_analyzer 2>/dev/null || true
monetdb start dataflow_analyzer 2>/dev/null || true

# Optional: Neo4j (for post-hoc graph analysis, not needed during training)
# echo "Starting Neo4j..."
# neo4j start &

# Optional: DfAnalyzer (for web UI, not needed during training)
# echo "Starting DfAnalyzer..."
# cd /opt/dlprov/DfAnalyzer
# if [ -f "target/DfAnalyzer-1.0.jar" ]; then
#     java -jar target/DfAnalyzer-1.0.jar &
# fi
# cd /opt/dlprov

echo "=== DLProv stack ready ==="
exec "$@"
