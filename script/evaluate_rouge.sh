#!/bin/bash
# Evaluation script for federated learning models

# Run evaluation
python evaluate_rouge.py \
    --exp_name 'homo-1B' \
    --target_file './data/dataset1/dolly_test_200.jsonl' \
    --target_key 'output' \
    --prediction_dir './predictions' \
    --prediction_key 'answer' \
    --evaluation_dir './evaluations_rouge' \
    --communication_rounds 20 \