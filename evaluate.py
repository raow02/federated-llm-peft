"""
Pairwise GPT‑4 evaluation: compare *one* prediction file (global or client)
against an explicit *baseline* prediction file.

All files are JSON‑Lines with at least:
    instruction • input (optional) • <answer‑field>
"""

import json, os, re, time
from typing import Dict, Tuple, Optional, List

import fire, openai
from tqdm import tqdm

# ---------------- GPT prompt helpers ---------------- #
PAIR_PROMPT = """
You are an expert language‑model evaluator.

Given an instruction (and optional input), compare **two** model responses.

Score each response from 1‑10 (10 = perfect) considering:
• Relevance / correctness
• Completeness
• Fluency & coherence

Return *exactly*:

- Model A Score: X
- Model B Score: Y
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
    block = f'Instruction:\n"{instruction}"'
    if _input:
        block += f'\n\nInput:\n"{_input}"'
    prompt = (
        PAIR_PROMPT
        + f"\n\n{block}\n\nResponse **Model A**:\n\"\"\"{resp_a}\"\"\""
          f"\n\nResponse **Model B**:\n\"\"\"{resp_b}\"\"\""
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

def key(rec: dict) -> Tuple[str, str]:
    """unique match key: (instruction, input)"""
    return (rec["text"].strip(), "")


# ------------------------- main ------------------------ #
def main(
    exp_name: str = "fedavg-1B",
    prediction_dir: str = "./predictions",
    communication_rounds: int = 50,
    client_id: Optional[int] = None,       # None → global predictions
    baseline_file: str = "baseline_output.jsonl",
    prediction_key: str = "answer",
    baseline_key: str = "answer",
    evaluation_dir: str = "./evaluations",
    api_key: str = "",
    gpt_model: str = "gpt-4o",
    sleep_between_calls: float = 1.5,
):
    """
    Compare <prediction> vs <baseline> with GPT‑4 pairwise scoring.
    """

    # ---------- resolve paths ---------- #
    pred_fname = (
        "global_output.jsonl"
        if client_id is None
        else f"client_{client_id}_output.jsonl"
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

    pred_map = {key(r): r for r in preds}
    base_map = {key(r): r for r in base}

    client = openai.OpenAI(api_key=api_key)

    pair_results = []
    for k, p_rec in tqdm(pred_map.items(), desc="Evaluating"):
        b_rec = base_map.get(k)
        if b_rec is None:
            print("baseline missing for a sample, skip.")
            continue

        instr, inp = k
        try:
            a_s, b_s = gpt_pair(
                client,
                instr,
                inp,
                p_rec[prediction_key],
                b_rec[baseline_key],
                model=gpt_model,
            )
        except Exception as e:
            print("GPT error:", e)
            continue

        pair_results.append(
            dict(
                instruction=instr,
                input=inp,
                prediction_score=a_s,
                baseline_score=b_s,
            )
        )
        time.sleep(sleep_between_calls)

    # ---------- summarise ---------- #
    ps = [r["prediction_score"] for r in pair_results]
    bs = [r["baseline_score"]   for r in pair_results]
    summary = {
        "avg_prediction_score": sum(ps) / len(ps) if ps else None,
        "avg_baseline_score"  : sum(bs) / len(bs) if bs else None,
        "num_samples"         : len(pair_results),
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
        json.dump(dict(summary=summary, details=pair_results), f, indent=2)
    print("\n  Saved results to", out_path)


if __name__ == "__main__":
    fire.Fire(main)
