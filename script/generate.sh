#!/bin/bash
# Generation script for federated learning models

# Get GPU devices from command line argument
cuda_devices=$1

# Run generation
CUDA_VISIBLE_DEVICES="${cuda_devices}" python generate.py \
    --exp_name 'homo-1B' \
    --base_model 'meta-llama/Llama-3.2-1B' \
    --model_dir './output_models' \
    --communication_rounds 20 \
    --test_file_path './data/dataset1/dolly_test_200.jsonl' \
    --prediction_dir './predictions' \
    --batch_size 4 \
    --is_global_model
