"""
Partition a Dolly‑style dataset into
  • per‑category local training sets (JSON‑Lines, fixed sample count)
  • one cross‑category test set (JSON‑Lines)

Directory layout:
./data/<dataset_name>/{n}/
    ├── local_training_0.jsonl
    ├── local_training_1.jsonl
    └── ...
./data/<dataset_name>/dolly_test_200.jsonl
"""

import json
from pathlib import Path
from typing import List

import pandas as pd
import numpy as np
import fire


def load_dataframe(path: str, orient: str = "records") -> pd.DataFrame:
    """Load a JSON file as DataFrame. Supports array or JSON‑Lines."""
    if orient == "records":
        return pd.read_json(path, orient="records")
    if orient == "lines":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported orient='{orient}'")


def ensure_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Raise if any required column is missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing columns: {missing}")


def dataframe_to_jsonl(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to JSON‑Lines file."""
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def write_jsonl(records: List[dict], path: Path) -> None:
    """Write a list of dicts to JSON‑Lines file."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_partitions(
    input_file: str = "./data/new-databricks-dolly-15k.json",
    output_root: str = "./data",
    dataset_name: str = "dataset1",
    train_per_cat: int = 300,
    test_per_cat: int = 200,
    seed: int = 42,
    read_orient: str = "records",
) -> None:
    """
    Split the dataset into per‑category training JSONL files and a global test JSONL file.

    Parameters
    ----------
    input_file : str
        Path to the original JSON (array or JSON‑Lines).
    output_root : str
        Root directory for output.
    dataset_name : str
        Subdirectory name under output_root.
    train_per_cat : int
        Samples per category for local training sets.
    test_per_cat : int
        Samples per category for the global test set.
    seed : int
        Random seed.
    read_orient : str
        "records" for JSON array, "lines" for JSON‑Lines input.
    """
    rng = np.random.RandomState(seed)

    df = load_dataframe(input_file, orient=read_orient)

    # Ensure expected column names
    df = df.rename(columns={"context": "input", "response": "output"})
    ensure_columns(df, ["instruction", "input", "output", "category"])

    categories: List[str] = sorted(df["category"].unique())
    n_cats = len(categories)

    local_dir = Path(output_root) / dataset_name / str(n_cats)
    local_dir.mkdir(parents=True, exist_ok=True)

    test_records_all: List[dict] = []

    print(f"Found {n_cats} categories – splitting "
          f"train={train_per_cat} / test={test_per_cat} per category ...")

    for idx, cat in enumerate(categories):
        cat_df = df[df["category"] == cat].sample(frac=1, random_state=rng)
        total = len(cat_df)

        # Test split
        n_test = min(test_per_cat, total)
        test_df = cat_df.iloc[:n_test]
        test_records_all.extend(test_df.to_dict(orient="records"))

        # Training split
        start_train = n_test
        end_train = min(n_test + train_per_cat, total)
        train_df = cat_df.iloc[start_train:end_train]

        # Save local training JSONL
        local_path = local_dir / f"local_training_{idx}.jsonl"
        dataframe_to_jsonl(train_df, local_path)

        print(f"[{idx}] '{cat}': total={total}, "
              f"train_saved={len(train_df)}, test_saved={len(test_df)}")

    # Save global test JSONL
    test_path = Path(output_root) / dataset_name / "dolly_test_200.jsonl"
    write_jsonl(test_records_all, test_path)

    print("\n   Dataset split complete")
    print(f"  • Local training dir : {local_dir}")
    print(f"  • Global test file   : {test_path}")
    print(f"  • Random seed        : {seed}")


if __name__ == "__main__":
    fire.Fire(build_partitions)
