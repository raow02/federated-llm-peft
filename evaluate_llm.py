"""
Batched pairwise GPT evaluation: compare one prediction file (global or client)
against our proposed model prediction file, with batch evaluation
of samples per category at a time, ensuring n_eval successful evaluations per batch.
"""

import json
import os
import sys
import time
from typing import Dict, Tuple, Optional, List, Any
from collections import defaultdict

import fire
import openai
from tqdm import tqdm

# ---------------- GPT prompt helpers ---------------- #
# Batch prompt for evaluating multiple samples at once
def get_batch_prompt(batch_size: int) -> str:
    return f"""
You are an expert language‑model evaluator.
You will be given {batch_size} examples, each containing an instruction (and optional input) plus two candidate responses (Model A and Model B).
Evaluate all examples collectively and assign one overall score for each model on a 1–100 scale, where:
 1 = completely wrong or irrelevant
100 = perfect in relevance, completeness, fluency, and style

Evaluate according to:
• Relevance & correctness
• Completeness
• Fluency & coherence

Consider all {batch_size} examples when determining the overall quality of each model.

**Output exactly one JSON object** with these keys:
{{
  "model_a_score": int,    // 1–100
  "model_b_score": int,    // 1–100
  "justification": str     // Brief explanation of your scoring decision
}}

Do not output any additional text.
""".strip()

# Original PAIR_PROMPT as fallback for individual evaluation
PAIR_PROMPT = """
You are an expert language‑model evaluator.
You will be given an instruction (and optional input) plus two candidate responses (Model A and Model B).
Score each response on a 1–10 scale, where:
 1 = completely wrong or irrelevant
10 = perfect in relevance, completeness, fluency, and style

Evaluate according to:
• Relevance & correctness
• Completeness
• Fluency & coherence

Then choose which model is better. If they're essentially equal, choose "Tie."

**Output exactly one JSON object** with these keys:
{
  "model_a_score": int,    // 1–10
  "model_b_score": int,    // 1–10
  "better_model": "A" | "B" | "Tie",
  "justification": str     // 1–2 sentences explaining your choice
}
Do not output any additional text.
""".strip()

def test_openai_api(client: openai.OpenAI, model: str) -> bool:
    """Test if the OpenAI API is working properly with a simple query."""
    try:
        test_prompt = "Return only the number 42 as a plain integer, nothing else."
        response = client.responses.create(
            model=model,
            instructions="You are a helpful AI assistant.",
            input=test_prompt
        )
        content = response.output_text
        if "42" in content:
            print("✓ API test successful! Connection works.")
            return True
        else:
            print(f"✗ API test response unexpected. Got: '{content}', expected '42'")
            return False
    except Exception as e:
        print(f"✗ API test failed with error: {e}")
        return False

def gpt_batch_eval(
    client: openai.OpenAI,
    samples: List[Dict[str, Any]],
    model: str = "o4-mini-2025-04-16",
    batch_size: int = 20
) -> Tuple[int, int]:
    """
    Evaluate a batch of samples with GPT and return overall scores for model A and B.
    
    Args:
        client: OpenAI client
        samples: List of dictionaries with instruction, input, resp_a, resp_b
        model: OpenAI model to use
        batch_size: Maximum number of samples to process in a single batch
    
    Returns:
        Tuple of (model_a_score, model_b_score) or (0, 0) if parsing fails
    """
    # Ensure we don't exceed batch size
    if len(samples) > batch_size:
        samples = samples[:batch_size]
    
    # Create the batch prompt
    formatted_samples = []
    for i, sample in enumerate(samples):
        block = f'Sample {i}:\nInstruction:\n"{sample["instruction"]}"'
        if sample["input"]:
            block += f'\n\nInput:\n"{sample["input"]}"'
        block += f'\n\nResponse **Model A**:\n\"\"\"{sample["resp_a"]}\"\"\"\n\nResponse **Model B**:\n\"\"\"{sample["resp_b"]}\"\"\"\n'
        formatted_samples.append(block)
    
    prompt = get_batch_prompt(len(samples)) + "\n\n" + "\n".join(formatted_samples)
    
    try:
        reply = client.responses.create(
            model=model,
            instructions="",
            input=prompt,
        )
        content = reply.output_text
        
        # Parse JSON response
        import re
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                if "model_a_score" in result and "model_b_score" in result:
                    a_score = result["model_a_score"]
                    b_score = result["model_b_score"]
                    
                    # Ensure scores are within valid range
                    a_score = max(1, min(100, a_score))
                    b_score = max(1, min(100, b_score))
                    
                    return (a_score, b_score)
                else:
                    print(f"Warning: Missing score keys in response")
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON from response: {e}")
        else:
            print(f"Warning: Could not find JSON object in response")
    except Exception as e:
        print(f"Error during batch evaluation: {e}")
    
    # Return zeros to indicate failure
    return (0, 0)

def gpt_pair(
    client: openai.OpenAI,
    instruction: str,
    _input: str,
    resp_a: str,
    resp_b: str,
    model: str = "o4-mini-2025-04-16"
) -> Tuple[int, int]:
    """Call GPT and parse the two scores using regex, warning & raising if parse fails."""
    block = f'Instruction:\n"{instruction}"'
    if _input:
        block += f'\n\nInput:\n"{_input}"'
    prompt = (
        PAIR_PROMPT
        + f"\n\n{block}\n\nResponse **Model A**:\n\"\"\"{resp_a}\"\"\""
        + f"\n\nResponse **Model B**:\n\"\"\"{resp_b}\"\"\""
    )
    reply = client.responses.create(
        model=model,
        instructions="",
        input=prompt,
    )
    content = reply.output_text
    
    # Extract scores using regex
    import re
    
    # Look for model_a_score in various formats
    a_matches = re.search(r"\"model_a_score\"[:\s]+(\d+)", content)
    # Look for model_b_score in various formats
    b_matches = re.search(r"\"model_b_score\"[:\s]+(\d+)", content)
    
    if a_matches and b_matches:
        try:
            a_score = int(a_matches.group(1))
            b_score = int(b_matches.group(1))
            # Validate the scores are in valid range
            if 1 <= a_score <= 10 and 1 <= b_score <= 10:
                return a_score, b_score
            else:
                raise ValueError(f"Scores out of valid range (1-10): A={a_score}, B={b_score}")
        except Exception as e:
            print(f"Warning: error extracting or validating scores: {e}")
            raise
    else:
        print(f"Warning: Could not find scores in response:\n{content}")
        raise ValueError("Could not extract scores from response")

# ---------------- I/O helpers ---------------- #

def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file and return list of dictionaries."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file {path}: {e}")
        return []
    except Exception as e:
        print(f"Error: Unexpected error loading {path}: {e}")
        return []

def check_prediction_keys(records: List[dict], key: str) -> bool:
    """Check if all records have the specified prediction key."""
    missing = [i for i, r in enumerate(records) if key not in r]
    if missing:
        print(f"Warning: {len(missing)} records missing key '{key}'")
        print(f"First few missing indices: {missing[:5]}")
        return False
    return True

# ------------------------- main ------------------------ #

def main(
    exp_name: str = "none-1B",
    prediction_dir: str = "./predictions",
    communication_rounds: int = 50,
    client_id: Optional[int] = None,       # None → global predictions
    proposed_file: str = "homo-1B/20/global_output.jsonl",
    prediction_key: str = "answer",
    proposed_key: str = "answer",
    evaluation_dir: str = "./evaluations",
    api_key: str = "",
    gpt_model: str = "o4-mini-2025-04-16",
    n_eval: int = 3,
    sleep_between_calls: float = 1,
    skip_api_check: bool = False,
    batch_size: int = 20,  # Parameter for batch size
    use_batching: bool = True,  # Parameter to toggle batching
    max_retry_attempts: int = 5,  # Maximum number of retries per evaluation
):
    """Compare *prediction* vs *proposed* model with batched evaluation."""
    if n_eval < 1:
        raise ValueError("n_eval must be ≥ 1")

    # ---------- resolve paths ---------- #
    pred_fname = (
        "global_output.jsonl" if client_id is None else f"client_{client_id}_output.jsonl"
    )
    pred_file = os.path.join(
        prediction_dir, exp_name, str(communication_rounds), pred_fname
    )
    if not os.path.isabs(proposed_file):
        proposed_file = os.path.join(prediction_dir, proposed_file)

    print(f"Prediction file        : {pred_file}")
    print(f"Proposed model file    : {proposed_file}")
    print(f"Using batched evaluation: {use_batching} (batch size: {batch_size})")
    print(f"Evaluations per batch  : {n_eval} (max retry attempts: {max_retry_attempts})")
    
    # Check if files exist
    print(f"Prediction file exists : {os.path.exists(pred_file)}")
    print(f"Proposed file exists   : {os.path.exists(proposed_file)}")
    
    if not os.path.exists(pred_file) or not os.path.exists(proposed_file):
        print("Error: One or both files do not exist. Exiting.")
        return

    # ---------- load data ---------- #
    print("Loading prediction file...")
    preds = load_jsonl(pred_file)
    if not preds:
        print("Error: No valid records found in prediction file. Exiting.")
        return
        
    print("Loading proposed model file...")
    proposed = load_jsonl(proposed_file)
    if not proposed:
        print("Error: No valid records found in proposed model file. Exiting.")
        return
    
    # Check keys exist in both files
    print(f"Checking for '{prediction_key}' in prediction file...")
    if not check_prediction_keys(preds, prediction_key):
        print(f"Warning: Some records missing '{prediction_key}' key in prediction file")
    
    print(f"Checking for '{proposed_key}' in proposed file...")
    if not check_prediction_keys(proposed, proposed_key):
        print(f"Warning: Some records missing '{proposed_key}' key in proposed file")
    
    # Verify file lengths
    print(f"Prediction records: {len(preds)}, Proposed records: {len(proposed)}")
    
    # Use the smaller length of the two files to ensure we don't go out of bounds
    min_length = min(len(preds), len(proposed))
    if min_length < len(preds) or min_length < len(proposed):
        print(f"Warning: Files have different lengths. Using first {min_length} records from each file.")
    
    if min_length == 0:
        print("Error: No valid records to compare. Exiting.")
        return

    # ---------- initialize API client ---------- #
    client = openai.OpenAI(api_key=api_key)
    
    # Test API connection
    if not skip_api_check:
        print("Testing API connection...")
        if not test_openai_api(client, gpt_model):
            print("API test failed. Please check your API key and model. Exiting.")
            return
    else:
        print("API check skipped.")

    # ---------- organize samples by category ---------- #
    if use_batching:
        # First, organize samples by category for batched processing
        print("Organizing samples by category...")
        category_samples = defaultdict(list)
        valid_samples = []
        
        for idx in range(min_length):
            p_rec = preds[idx]
            b_rec = proposed[idx]
            
            # Verify prediction keys exist
            if prediction_key not in p_rec or proposed_key not in b_rec:
                continue
            
            instr = p_rec["instruction"]
            inp = p_rec.get("input", "")
            cat = p_rec.get("category", "unknown")
            
            sample = {
                "idx": idx,
                "instruction": instr,
                "input": inp,
                "resp_a": p_rec[prediction_key],
                "resp_b": b_rec[proposed_key],
                "category": cat
            }
            
            valid_samples.append(sample)
            category_samples[cat].append(sample)
        
        print(f"Valid samples: {len(valid_samples)}")
        print(f"Categories found: {len(category_samples)}")
        
        # ---------- process batches by category ---------- #
        cat_stats: Dict[str, Dict[str, float]] = {}
        total_samples = 0
        eval_failures = 0
        retry_count = 0
        
        # Process each category in batches
        for cat, samples in tqdm(category_samples.items(), desc="Processing categories"):
            entry = cat_stats.setdefault(
                cat, {"num_samples": 0, "pred_sum": 0.0, "proposed_sum": 0.0}
            )
            
            # Process this category in batches
            for batch_start in range(0, len(samples), batch_size):
                batch_end = min(batch_start + batch_size, len(samples))
                batch = samples[batch_start:batch_end]
                
                # We'll track successful evaluations and scores
                successful_evals = 0
                batch_pred_scores = []
                batch_proposed_scores = []
                
                # Try to get n_eval successful evaluations
                while successful_evals < n_eval and len(batch_pred_scores) < n_eval + max_retry_attempts:
                    p_score, b_score = gpt_batch_eval(
                        client,
                        batch,
                        model=gpt_model,
                        batch_size=batch_size
                    )
                    
                    if p_score > 0 and b_score > 0:  # Both scores are valid
                        batch_pred_scores.append(p_score)
                        batch_proposed_scores.append(b_score)
                        successful_evals += 1
                    else:
                        retry_count += 1
                    
                    # Always sleep between calls to avoid rate limiting
                    time.sleep(sleep_between_calls)
                
                # Check if we got enough successful evaluations
                if successful_evals == n_eval:
                    # Calculate averages for this batch
                    batch_size_actual = len(batch)
                    pred_avg = sum(batch_pred_scores) / n_eval
                    proposed_avg = sum(batch_proposed_scores) / n_eval
                    
                    # Add to category stats (multiply by batch size since we're applying
                    # the same scores to all samples in the batch)
                    entry["num_samples"] += batch_size_actual
                    entry["pred_sum"] += pred_avg * batch_size_actual
                    entry["proposed_sum"] += proposed_avg * batch_size_actual
                    total_samples += batch_size_actual
                else:
                    # If we didn't get enough evaluations, count as failures
                    eval_failures += len(batch)
                    print(f"Warning: Batch evaluation failed after {len(batch_pred_scores)} attempts. Got {successful_evals}/{n_eval} successful evaluations.")
    else:
        # ---------- use original sample-by-sample approach ---------- #
        cat_stats: Dict[str, Dict[str, float]] = {}
        total_samples = 0
        eval_failures = 0

        # Original logic for processing samples individually
        for idx in tqdm(range(min_length), desc="Evaluating"):
            p_rec = preds[idx]
            b_rec = proposed[idx]
            
            instr = p_rec["instruction"]
            inp = p_rec.get("input", "")
            cat = p_rec.get("category", "unknown")
            
            # Verify prediction keys exist
            if prediction_key not in p_rec:
                print(f"Warning: Sample {idx} missing prediction key '{prediction_key}', skipping.")
                continue
                
            if proposed_key not in b_rec:
                print(f"Warning: Sample {idx} missing proposed key '{proposed_key}', skipping.")
                continue

            pred_votes: List[int] = []
            proposed_votes: List[int] = []

            for _ in range(n_eval):
                try:
                    p_score, b_score = gpt_pair(
                        client,
                        instr,
                        inp,
                        p_rec[prediction_key],
                        b_rec[proposed_key],
                        model=gpt_model,
                    )
                except Exception:
                    # warning already printed by gpt_pair; skip this entire sample
                    pred_votes = []
                    break
                pred_votes.append(p_score)
                proposed_votes.append(b_score)
                time.sleep(sleep_between_calls)

            # if any vote failed, skip this sample entirely
            if len(pred_votes) != n_eval:
                eval_failures += 1
                continue

            total_samples += 1
            entry = cat_stats.setdefault(
                cat, {"num_samples": 0, "pred_sum": 0.0, "base_sum": 0.0}
            )
            entry["num_samples"] += 1
            entry["pred_sum"] += sum(pred_votes) / n_eval
            entry["proposed_sum"] += sum(proposed_votes) / n_eval

    # Print summary of processed samples
    print(f"\nTotal samples in files: {min_length}")
    print(f"Successfully processed: {total_samples}")
    print(f"Evaluation failures: {eval_failures}")
    if use_batching:
        print(f"Total retry attempts: {retry_count}")
    print(f"Total processed + failures: {total_samples + eval_failures}")

    # ---------- summarise ---------- #
    for stats in cat_stats.values():
        n = stats["num_samples"]
        stats["avg_prediction_score"] = stats.pop("pred_sum") / n if n else None
        stats["avg_proposed_score"] = stats.pop("proposed_sum") / n if n else None

    out_summary = {
        "votes_per_sample": n_eval,
        "num_samples": total_samples,
        "eval_failures": eval_failures,
        "total_in_files": min_length,
        "categories": cat_stats,
        "batched_evaluation": use_batching,
        "batch_size": batch_size if use_batching else None,
        "retry_attempts": retry_count if use_batching else None,
    }

    # ---------- save ---------- #
    out_path = os.path.join(
        evaluation_dir,
        exp_name,
        str(communication_rounds),
        f"{'global' if client_id is None else f'client_{client_id}'}_results.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)

    print("\nSaved results to", out_path)

if __name__ == "__main__":
    fire.Fire(main)