#!/bin/bash
# Evaluation script for federated learning models

# ------------------- Configurable Variables ------------------- #
is_global_model=true         # true or false
exp_name="homo-1B"
communication_rounds=20
target_file="./data/dataset1/dolly_test_200.jsonl"
target_key="output"
prediction_key="answer"
prediction_dir="./predictions"
evaluation_dir="./evaluations_rouge"

# ------------------- Evaluation Logic ------------------- #
if [ "$is_global_model" = true ]; then
  echo "Evaluating global model..."
  python evaluate_rouge.py \
    --exp_name "$exp_name" \
    --target_file "$target_file" \
    --target_key "$target_key" \
    --prediction_dir "$prediction_dir" \
    --prediction_key "$prediction_key" \
    --evaluation_dir "$evaluation_dir" \
    --communication_rounds "$communication_rounds"
else
  echo "Evaluating client models..."
  for client_id in {0..7}; do
    echo "Evaluating client $client_id..."
    python evaluate_rouge.py \
      --exp_name "$exp_name" \
      --target_file "$target_file" \
      --target_key "$target_key" \
      --prediction_dir "$prediction_dir" \
      --prediction_key "$prediction_key" \
      --evaluation_dir "$evaluation_dir" \
      --communication_rounds "$communication_rounds" \
      --client_id "$client_id"
  done
fi
