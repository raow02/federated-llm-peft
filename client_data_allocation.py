"""
Partition a Dolly‑style dataset into
  • local training sets (JSON array, one per category)
  • one cross‑category test set (JSON‑Lines)

Output layout:
./data/<dataset_name>/{n}/               # n = number of categories
    ├── local_training_0.json            # JSON array
    ├── local_training_1.json
    └── ...
./data/<dataset_name>/dolly_test_200.jsonl   # JSON‑Lines
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import fire


def load_dataframe(path: str, orient: str = "records") -> pd.DataFrame:
    """Load a JSON file as DataFrame. Supports JSON array or JSON‑Lines."""
    if orient == "records":
        return pd.read_json(path, orient="records")
    if orient == "lines":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported orient='{orient}'")


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Ensure the DataFrame contains the required columns."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")


def build_partitions(
    input_file: str = "./data/new-databricks-dolly-15k.json",
    output_root: str = "./data",
    dataset_name: str = "dataset1",
    train_per_cat: int = 300,
    test_per_cat: int = 200,
    seed: int = 42,
    read_orient: str = "records",
) -> None:
    """Split the original dataset into per‑category training sets and one global test set."""
    rng = np.random.RandomState(seed)

    df = load_dataframe(input_file, orient=read_orient)

    # Rename fields to match downstream expectations
    df = df.rename(columns={"context": "input", "response": "output"})
    ensure_columns(df, ["instruction", "input", "output", "category"])

    categories = sorted(df["category"].unique().tolist())
    n_cats = len(categories)

    # Directory for local training sets: ./data/<dataset_name>/<n>/
    local_dir = Path(output_root) / dataset_name / str(n_cats)
    local_dir.mkdir(parents=True, exist_ok=True)

    test_records_all: List[dict] = []

    print(f"Found {n_cats} categories – splitting "
          f"train={train_per_cat} / test={test_per_cat} per category ...")

    for idx, cat in enumerate(categories):
        cat_df = df[df["category"] == cat].sample(frac=1, random_state=rng)  # shuffle
        total = len(cat_df)

        # Test slice
        n_test = min(test_per_cat, total)
        test_df = cat_df.iloc[:n_test]
        test_records_all.extend(test_df.to_dict(orient="records"))

        # Training slice
        start_train = n_test
        end_train = min(n_test + train_per_cat, total)
        train_df = cat_df.iloc[start_train:end_train]

        # Save local training JSON array
        local_path = local_dir / f"local_training_{idx}.json"
        train_df.to_json(local_path, orient="records", force_ascii=False)

        print(f"[{idx}] '{cat}': total={total}, "
              f"train_saved={len(train_df)}, test_saved={len(test_df)}")

    # Save the global test set as JSON‑Lines
    test_path = Path(output_root) / dataset_name / "dolly_test_200.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for rec in test_records_all:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n   Dataset split complete")
    print(f"  • Local training dir : {local_dir}")
    print(f"  • Global test file   : {test_path}")
    print(f"  • Random seed        : {seed}")


if __name__ == "__main__":
    fire.Fire(build_partitions)
