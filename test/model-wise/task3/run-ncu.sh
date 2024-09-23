#!/bin/bash

# Default configuration file
CONFIG_FILE="config.json"

# Parse input arguments to override the default config file if provided
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Extract world_size and model_name from the configuration file using a Python one-liner
WORLD_SIZE=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['world_size'])")
MODEL_NAME=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['model_name'])")

# Define variables for output files, appending world_size, model_name, and config file name to the file name
OUTPUT_NAME="model_training_$(basename ${CONFIG_FILE} .json)_${MODEL_NAME}_ws${WORLD_SIZE}"

ncu -o ${OUTPUT_NAME} \
    -f --replay-mode kernel \
    --device 0 \
    --nvtx --nvtx-include "backward/" --nvtx-include "forward/" \
    --target-processes all \
    python train-task3.py --config_file=${CONFIG_FILE}