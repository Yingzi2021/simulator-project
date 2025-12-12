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
STATS_OUTPUT="output_file__$(basename ${CONFIG_FILE} .json)_${MODEL_NAME}_ws${WORLD_SIZE}"

# Step 1: Profile the model training process using the configuration file
nsys profile --trace=cuda,nvtx,osrt,python-gil --sample=cpu --python-sampling=true --python-backtrace=cuda --gpuctxsw=true \
    --output=${OUTPUT_NAME} --export=none --force-overwrite true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    python train.py --config_file=${CONFIG_FILE}

# Step 2: Generate statistics and export to sqlite
nsys export -t sqlite --force-overwrite true -o ${STATS_OUTPUT}.sqlite ${OUTPUT_NAME}.nsys-rep
echo "Profiling and analysis complete. Output saved to ${STATS_OUTPUT}.sqlite"