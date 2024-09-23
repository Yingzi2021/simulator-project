#!/bin/bash
# Default values for arguments
CAP_SIZE=2
DATASET="imdb"
WORLD_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)  # Default to the number of GPUs

# Parse input arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cap_size) CAP_SIZE="$2"; shift ;;
        --dataset) DATASET="$2"; shift ;;
        --world_size) WORLD_SIZE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Define variables for output files, appending dataset, cap_size, and world_size to the file name
OUTPUT_NAME="model_training_${DATASET}_cap${CAP_SIZE}_ws${WORLD_SIZE}"
STATS_OUTPUT="output_file_${DATASET}_cap${CAP_SIZE}_ws${WORLD_SIZE}" 

# Step 1: Profile the model training process with the specified cap_size, dataset, and world_size
nsys profile --trace=cuda,nvtx,osrt,python-gil --sample=cpu --python-sampling=true --python-backtrace=cuda --gpuctxsw=true \
    --output=${OUTPUT_NAME} --export=none --force-overwrite true --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    python train.py --cap_size=${CAP_SIZE} --dataset=${DATASET} --world_size=${WORLD_SIZE}

# Step 2: Generate statistics and export to sqlite
nsys export -t sqlite --force-overwrite true -o ${STATS_OUTPUT}.sqlite ${OUTPUT_NAME}.nsys-rep
echo "Profiling and analysis complete. Output saved to ${STATS_OUTPUT}.sqlite"
