#!/bin/bash

# Default path to the configuration file
CONFIG_FILE="config.json"

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --config_file) CONFIG_FILE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Extract specific values from config.json using Python
WORLD_SIZE=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['world_size'])")
OVERLAP=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['overlap'])")
TENSOR_SIZE=$(python -c "import json; print(json.load(open('${CONFIG_FILE}'))['tensor_size'])")

# Define variables for output files, appending config_file name and specific fields to the file name
OUTPUT_NAME="model_training_$(basename ${CONFIG_FILE} .json)_ws${WORLD_SIZE}_overlap_${OVERLAP}_tensor_size_${TENSOR_SIZE}"
STATS_OUTPUT="output_file_$(basename ${CONFIG_FILE} .json)_ws${WORLD_SIZE}_overlap_${OVERLAP}_tensor_size_${TENSOR_SIZE}"

# Step 1: Profile the model training process using the configuration file
nsys profile --trace=cuda,nvtx,osrt,python-gil --sample=cpu --python-sampling=true --python-backtrace=cuda --gpuctxsw=true \
    --output=${OUTPUT_NAME} --export=none --force-overwrite true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    python combined_operations.py --config_file ${CONFIG_FILE}

# Step 2: Generate statistics and export to sqlite
nsys export -t sqlite --force-overwrite true -o ${STATS_OUTPUT}.sqlite ${OUTPUT_NAME}.nsys-rep
echo "Profiling and analysis complete. Output saved to ${STATS_OUTPUT}.sqlite"
