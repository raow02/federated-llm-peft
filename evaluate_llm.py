"""
Pairwise GPT‑4 evaluation: compare *one* prediction file (global or client)
against an explicit *baseline* prediction file, with *n_eval* independent
GPT‑4 votes per sample (default = 3).
"""

import json, os, re, time
from typing import Dict, Tuple, Optional, List

import fire, openai
from tqdm import tqdm

# ---------------- GPT prompt helpers ---------------- #
PAIR_PROMPT = """
You are an expert language‑model evaluator.
Given an instruction (and optional input), compare **two** model responses.
Score each response from 1‑10 (10 = perfect) considering:
• Relevance / correctness
• Completeness
• Fluency & coherence
Return *exactly*:
- Model A Score: X
- Model B Score: Y
- Better Model: [A / B / Tie]
""".strip()

_SCORE_RE = re.compile(
    r"Model\s+A\s+Score:\s*(\d+).*?Model\s+B\s+Score:\s*(\d+)", re.S | re.I
)

def gpt_pair(
    client: openai.OpenAI,
    instruction: str,
    _input: str,
    resp_a: str,
    resp_b: str,
    model: str = "gpt-4o",
    max_tokens: int = 150,
) -> Tuple[int, int]:
    """Call GPT‑4 and parse the two scores."""
    block = f'Instruction:\n"{instruction}"'
    if _input:
        block += f'\n\nInput:\n"{_input}"'
    prompt = (
        PAIR_PROMPT
        + f"\n\n{block}\n\nResponse **Model A**:\n\"\"\"{resp_a}\"\"\""
        + f"\n\nResponse **Model B**:\n\"\"\"{resp_b}\"\"\""
    )
    reply = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    m = _SCORE_RE.search(reply.choices[0].message.content)
    if not m:
        raise ValueError("Cannot parse GPT reply:\n" + reply.choices[0].message.content)
    return int(m.group(1)), int(m.group(2))

# ---------------- I/O helpers ---------------- #

def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def match_key(rec: dict) -> Tuple[str, str]:
    """Unique match key between prediction & baseline."""
    return (rec["instruction"].strip(), rec.get("input", ""))

# ------------------------- main ------------------------ #

def main(
    exp_name: str = "fedavg-1B",
    prediction_dir: str = "./predictions",
    communication_rounds: int = 50,
    client_id: Optional[int] = None,       # None → global predictions
    baseline_file: str = "base_model_output.jsonl",
    prediction_key: str = "answer",
    baseline_key: str = "answer",
    evaluation_dir: str = "./evaluations",
    api_key: str = "",
    gpt_model: str = "gpt-4o",
    n_eval: int = 3,
    sleep_between_calls: float = 1.5,
):
    """Compare *prediction* vs *baseline* with *n_eval* GPT‑4 votes per sample."""

    if n_eval < 1:
        raise ValueError("n_eval must be ≥ 1")

    # ---------- resolve paths ---------- #
    pred_fname = (
        "global_output.jsonl" if client_id is None else f"client_{client_id}_output.jsonl"
    )
    pred_file = os.path.join(
        prediction_dir, exp_name, str(communication_rounds), pred_fname
    )
    if not os.path.isabs(baseline_file):
        baseline_file = os.path.join(prediction_dir, baseline_file)

    print(f"Prediction file : {pred_file}")
    print(f"Baseline file   : {baseline_file}")

    # ---------- load data ---------- #
    preds = load_jsonl(pred_file)
    base  = load_jsonl(baseline_file)

    pred_map = {match_key(r): r for r in preds}
    base_map = {match_key(r): r for r in base}

    client = openai.OpenAI(api_key=api_key)

    # ---------- accumulate per‑category stats ---------- #
    cat_stats: Dict[str, Dict[str, float]] = {}

    total_samples = 0

    for k, p_rec in tqdm(pred_map.items(), desc="Evaluating"):
        b_rec = base_map.get(k)
        if b_rec is None:
            print("Baseline missing for a sample, skip.")
            continue

        cat = p_rec.get("category", "unknown")
        instr, inp = k

        pred_votes, base_votes = [], []
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
            except Exception as e:
                print("GPT error:", e)
                break
            pred_votes.append(p_score)
            base_votes.append(b_score)
            time.sleep(sleep_between_calls)

        if len(pred_votes) != n_eval:
            continue  # incomplete votes ⇒ skip sample

        total_samples += 1
        cat_entry = cat_stats.setdefault(
            cat,
            dict(num_samples=0, pred_sum=0.0, base_sum=0.0),
        )
        cat_entry["num_samples"] += 1
        cat_entry["pred_sum"]  += sum(pred_votes) / n_eval
        cat_entry["base_sum"]  += sum(base_votes) / n_eval

    # ---------- summarise ---------- #
    for c, d in cat_stats.items():
        n = d["num_samples"]
        d["avg_prediction_score"] = d.pop("pred_sum") / n if n else None
        d["avg_baseline_score"]   = d.pop("base_sum") / n if n else None

    out_summary = {
        "votes_per_sample": n_eval,
        "num_samples"     : total_samples,
        "categories"      : cat_stats,
    }

    # ---------- save ---------- #
    out_path = os.path.join(
        evaluation_dir,
        exp_name,
        str(communication_rounds),
        f"{('global' if client_id is None else f'client_{client_id}')}_results.json",
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)
    print("\n  Saved results to", out_path)

if __name__ == "__main__":
    fire.Fire(main)
