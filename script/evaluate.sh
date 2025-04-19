#!/bin/bash
# Evaluation script for federated learning models

# Run evaluation
python metric.py \
    --exp_name 'homo-1B' \
    --target_file './data/dataset1/flan_test_200_selected_nstrict_1.jsonl' \
    --target_key 'output' \
    --prediction_dir './predictions' \
    --prediction_key 'answer' \
    --evaluation_dir './evaluations_new_metric' \
    --communication_rounds 20
