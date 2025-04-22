#!/bin/bash
# Evaluation script for federated learning models

# ------------------- Configurable Variables ------------------- #
api_key=""
gpt_model="gpt-4.1-mini"
is_global_model=false         # true or false
exp_name="none-1B"
communication_rounds=1
proposed_file="homo-1B/20/global_output.jsonl"
prediction_dir="./predictions"
evaluation_dir="./evaluations_llm"
batch_size=20


# ------------------- Evaluation Logic ------------------- #
if [ "$is_global_model" = true ]; then
  echo "Evaluating global model..."
  python evaluate_llm.py \
    --api_key="$api_key" \
    --gpt_model="$gpt_model" \
    --exp_name="$exp_name" \
    --prediction_dir="$prediction_dir" \
    --proposed_file="$proposed_file" \
    --evaluation_dir="$evaluation_dir" \
    --communication_rounds="$communication_rounds" \
    --batch_size="$batch_size"
else
  echo "Evaluating client models..."
  for client_id in {0..7}; do
    echo "Evaluating client $client_id..."
    python evaluate_llm.py \
      --api_key="$api_key" \
      --gpt_model="$gpt_model" \
      --exp_name="$exp_name" \
      --prediction_dir="$prediction_dir" \
      --proposed_file="$proposed_file" \
      --evaluation_dir="$evaluation_dir" \
      --communication_rounds="$communication_rounds" \
      --batch_size="$batch_size" \
      --client_id="$client_id" 
  done
fi
