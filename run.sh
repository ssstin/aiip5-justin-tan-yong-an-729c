#!/bin/bash

# Ensure the data directory exists
mkdir -p data

# Run the ML pipeline
python src/main.py

# Print completion message
echo "ML pipeline execution completed."