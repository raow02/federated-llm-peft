"""
Evaluation script for federated‑learning outputs.
For all task categories we compute ROUGE scores (rouge1, rouge2, rougeL, rougeLsum).
"""

from __future__ import annotations

import os
import json
from typing import Dict, List

import fire
import evaluate
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_data(file_path: str, key: str) -> Dict[str, List[Dict]]:
    """Load a JSONL file and group samples by their `category` field."""
    grouped: Dict[str, List[Dict]] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            category: str = record["category"]
            grouped.setdefault(category, [])

            # prediction / target text may carry a closing `</s>` — strip it off
            value = record[key].split("</s>")[0]
            inst_key = "instruction" if "instruction" in record else "text"

            grouped[category].append({
                "instruction": record[inst_key],
                "output": value
            })
    return grouped


def compute_rouge(targets: List[str], preds: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores for a list of predictions vs. references."""
    rouge = evaluate.load("rouge")
    return rouge.compute(predictions=preds, references=targets)  # rouge1, rouge2, rougeL, rougeLsum


# -----------------------------------------------------------------------------
# Core evaluation logic
# -----------------------------------------------------------------------------

def evaluate_results(
    targets: Dict[str, List[Dict]],
    predictions: Dict[str, List[Dict]],
    output_path: str,
):
    """Compute ROUGE for every category and save a JSON summary."""
    per_category: Dict[str, Dict[str, float]] = {}
    macro_avgs: Dict[str, float] = {k: 0.0 for k in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    counted_cats = 0

    for cat, tgt_samples in targets.items():
        if cat not in predictions:
            print(f"[WARN] no predictions for category '{cat}', skipping.")
            continue

        # Align targets and predictions by index; truncate to shortest length
        tgt_texts = [s["output"] for s in tgt_samples]
        pred_texts = [s["output"] for s in predictions[cat][: len(tgt_texts)]]

        scores = compute_rouge(tgt_texts, pred_texts)
        per_category[cat] = scores

        for k in macro_avgs:
            macro_avgs[k] += scores[k]
        counted_cats += 1

    if counted_cats:
        for k in macro_avgs:
            macro_avgs[k] /= counted_cats
    per_category["overall"] = {f"avg_{k}": v for k, v in macro_avgs.items()}

    # Pretty print
    print("\nEvaluation Results (ROUGE):")
    for cat, sc in per_category.items():
        if cat != "overall":
            print(f"  {cat}:")
            for m, v in sc.items():
                print(f"    {m}: {v:.4f}")
    print("\nOverall (macro average across categories):")
    for m, v in per_category["overall"].items():
        print(f"  {m}: {v:.4f}")

    # Persist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(per_category, fp, indent=2)
    print(f"\nSaved results → {output_path}")


# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------

def main(
    exp_name: str = "homo-1B",
    target_file: str = "./data/dataset1/dolly_test_200.jsonl",
    target_key: str = "output",
    prediction_dir: str = "./predictions",
    prediction_key: str = "answer",
    evaluation_dir: str = "./evaluations_rouge",
    communication_rounds: int = 20,
    client_id: int | None = None,
):
    """Evaluate predictions against references using only ROUGE metrics."""

    pred_fname = "global_output.jsonl" if client_id is None else f"client_{client_id}_output.jsonl"
    pred_path = os.path.join(prediction_dir, exp_name, str(communication_rounds), pred_fname)

    out_path = os.path.join(
        evaluation_dir,
        exp_name,
        str(communication_rounds),
        pred_fname.replace(".jsonl", "_rouge.json"),
    )

    print("Evaluating:")
    print(f"  predictions → {pred_path}")
    print(f"  targets     → {target_file}\n")

    targets = load_data(target_file, target_key)
    preds = load_data(pred_path, prediction_key)

    evaluate_results(targets, preds, out_path)


if __name__ == "__main__":
    fire.Fire(main)
