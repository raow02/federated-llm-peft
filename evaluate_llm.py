"""
Pairwise GPT evaluation: compare one prediction file (global or client)
against our proposed model prediction file, with n_eval
independent GPT votes per sample (default = 3).
"""

import json
import os
import sys
import time
from typing import Dict, Tuple, Optional, List

import fire
import openai
from tqdm import tqdm

# ---------------- GPT prompt helpers ---------------- #
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
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": test_prompt}],
            max_completion_tokens=10
        )
        content = response.choices[0].message.content.strip()
        if "42" in content:
            print("✓ API test successful! Connection works.")
            return True
        else:
            print(f"✗ API test response unexpected. Got: '{content}', expected '42'")
            return False
    except Exception as e:
        print(f"✗ API test failed with error: {e}")
        return False

def gpt_pair(
    client: openai.OpenAI,
    instruction: str,
    _input: str,
    resp_a: str,
    resp_b: str,
    model: str = "o4-mini-2025-04-16",
    max_tokens: int = 150,
) -> Tuple[int, int]:
    """Call GPT and parse the two scores from JSON, warning & raising if parse fails."""
    block = f'Instruction:\n"{instruction}"'
    if _input:
        block += f'\n\nInput:\n"{_input}"'
    prompt = (
        PAIR_PROMPT
        + f"\n\n{block}\n\nResponse **Model A**:\n\"\"\"{resp_a}\"\"\""
        + f"\n\nResponse **Model B**:\n\"\"\"{resp_b}\"\"\""
    )
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=max_tokens
    )
    content = reply.choices[0].message.content.strip()
    try:
        data = json.loads(content)
        return data["model_a_score"], data["model_b_score"]
    except Exception as e:
        print(f"Warning: failed to parse GPT JSON response: {e}\nResponse was:\n{content}")
        # raise so that the main loop will skip this sample entirely
        raise

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

def match_key(rec: dict) -> Tuple[str, str]:
    """Unique match key between prediction & proposed."""
    return rec["instruction"].strip(), rec.get("input", "")

# ------------------------- main ------------------------ #

def main(
    exp_name: str = "none-1B",
    prediction_dir: str = "./predictions",
    communication_rounds: int = 50,
    client_id: Optional[int] = None,       # None → global predictions
    proposed_file: str = "homo-1B/20/global_output.jsonl",
    prediction_key: str = "answer",
    baseline_key: str = "answer",
    evaluation_dir: str = "./evaluations",
    api_key: str = "",
    gpt_model: str = "o4-mini-2025-04-16",
    n_eval: int = 3,
    sleep_between_calls: float = 1.5,
    skip_api_check: bool = False,
):
    """Compare *prediction* vs *proposed* model with *n_eval* GPT votes per sample."""
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
    base = load_jsonl(proposed_file)
    if not base:
        print("Error: No valid records found in proposed model file. Exiting.")
        return
    
    # Check keys exist in both files
    print(f"Checking for '{prediction_key}' in prediction file...")
    if not check_prediction_keys(preds, prediction_key):
        print(f"Warning: Some records missing '{prediction_key}' key in prediction file")
    
    print(f"Checking for '{baseline_key}' in proposed file...")
    if not check_prediction_keys(base, baseline_key):
        print(f"Warning: Some records missing '{baseline_key}' key in proposed file")
    
    # Check for matching keys
    print("Creating key mappings...")
    pred_map = {match_key(r): r for r in preds}
    base_map = {match_key(r): r for r in base}
    
    pred_keys = set(pred_map.keys())
    base_keys = set(base_map.keys())
    common_keys = pred_keys.intersection(base_keys)
    
    print(f"Prediction keys: {len(pred_keys)}, Proposed keys: {len(base_keys)}, Common: {len(common_keys)}")
    
    if len(common_keys) == 0:
        print("Error: No common keys found between prediction and proposed files. Exiting.")
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

    # ---------- accumulate per‑category stats ---------- #
    cat_stats: Dict[str, Dict[str, float]] = {}
    total_samples = 0

    for k, p_rec in tqdm(pred_map.items(), desc="Evaluating"):
        b_rec = base_map.get(k)
        if b_rec is None:
            print("Proposed model missing for a sample, skip.")
            continue

        instr, inp = k
        cat = p_rec.get("category", "unknown")
        pred_votes: List[int] = []
        base_votes: List[int] = []

        for _ in range(n_eval):
            try:
                p_score, b_score = gpt_pair(
                    client,
                    instr,
                    inp,
                    p_rec[prediction_key],
                    b_rec[baseline_key],
                    model=gpt_model,
                )
            except Exception:
                # warning already printed by gpt_pair; skip this entire sample
                pred_votes = []
                break
            pred_votes.append(p_score)
            base_votes.append(b_score)
            time.sleep(sleep_between_calls)

        # if any vote failed, skip this sample entirely
        if len(pred_votes) != n_eval:
            continue

        total_samples += 1
        entry = cat_stats.setdefault(
            cat, {"num_samples": 0, "pred_sum": 0.0, "base_sum": 0.0}
        )
        entry["num_samples"] += 1
        entry["pred_sum"]  += sum(pred_votes) / n_eval
        entry["base_sum"]  += sum(base_votes) / n_eval

    # ---------- summarise ---------- #
    for stats in cat_stats.values():
        n = stats["num_samples"]
        stats["avg_prediction_score"] = stats.pop("pred_sum") / n if n else None
        stats["avg_baseline_score"]   = stats.pop("base_sum")   / n if n else None

    out_summary = {
        "votes_per_sample": n_eval,
        "num_samples": total_samples,
        "categories": cat_stats,
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