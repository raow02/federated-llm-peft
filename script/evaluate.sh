#!/bin/bash
# Evaluation script for federated learning models

# Run evaluation
python evaluate.py \
  --api_key="YOUR_OPENAI_KEY" \
  --exp_name="homo-1B" \
  --prediction_dir="./predictions" \
  --baseline_file="baseline_output.jsonl" \
  --evaluation_dir './evaluations' \
  --communication_rounds=20 \
  --client_id=0
  
