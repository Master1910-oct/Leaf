"""
scripts/prepare_m72_data.py
===========================
Prepares and verifies the cumulative dataset for M7.2:
  - Copies existing M7.1 20% subset as stage1_20pct.csv
  - Samples reproducible non-overlapping 30% subset as stage2_added_30pct.csv
  - Combines them into cumulative_50pct.csv
  - Runs all 8 data integrity checks
  - Prepares stage1_reference/stage1_summary.json
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path("D:/Leaf")
OUT_DIR = ROOT / "results/m7_2_progressive_sto"
DATA_DIR = OUT_DIR / "data"
REF_DIR = OUT_DIR / "stage1_reference"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REF_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load source datasets
df_train = pd.read_csv(ROOT / "data/train.csv")
df_val = pd.read_csv(ROOT / "data/val.csv")
m71_subset_path = ROOT / "results/m7_1_sto_refinement/rapid_sto/sto_train_subset_20pct.csv"
df_stage1 = pd.read_csv(m71_subset_path)

# Verify Stage 1 size
assert len(df_stage1) == 2542, f"Expected 2542 in Stage 1, got {len(df_stage1)}"

# Save stage1_20pct.csv
stage1_csv = DATA_DIR / "stage1_20pct.csv"
df_stage1.to_csv(stage1_csv, index=False)

# 2. Form candidate pool of remaining training samples
stage1_fps = set(df_stage1["filepath"])
df_pool = df_train[~df_train["filepath"].isin(stage1_fps)].copy().reset_index(drop=True)
assert len(df_pool) == 10170, f"Expected pool of 10170, got {len(df_pool)}"

# 3. Sample exactly 3,814 samples (30.0% of full train 12,712) using StratifiedShuffleSplit seed=42
sss = StratifiedShuffleSplit(n_splits=1, train_size=3814, random_state=42)
added_idx, _ = next(sss.split(df_pool, df_pool["class_idx"]))
df_added = df_pool.iloc[sorted(added_idx)].copy().reset_index(drop=True)
assert len(df_added) == 3814, f"Expected 3814 added samples, got {len(df_added)}"

# Save stage2_added_30pct.csv
added_csv = DATA_DIR / "stage2_added_30pct.csv"
df_added.to_csv(added_csv, index=False)

# 4. Construct cumulative 50% dataset (6,356 samples)
df_cum50 = pd.concat([df_stage1, df_added], ignore_index=True)
assert len(df_cum50) == 6356, f"Expected 6356 cumulative samples, got {len(df_cum50)}"

# Save cumulative_50pct.csv
cum_csv = DATA_DIR / "cumulative_50pct.csv"
df_cum50.to_csv(cum_csv, index=False)

# 5. Data Integrity Checks (Section 23)
print("=" * 60)
print("M7.2 DATA INTEGRITY CHECKS (Section 23)")
print("=" * 60)

train_fps = set(df_train["filepath"])
val_fps = set(df_val["filepath"])

# Check 1: Stage-1 contains only training samples
c1 = set(df_stage1["filepath"]).issubset(train_fps)
print(f"Check 1: Stage-1 subset contains only training samples? {c1} [OK]")
assert c1

# Check 2: Added 30% contains only training samples
c2 = set(df_added["filepath"]).issubset(train_fps)
print(f"Check 2: Added 30% contains only training samples? {c2} [OK]")
assert c2

# Check 3: Stage-1 and added-30% overlap = 0
overlap = len(set(df_stage1["filepath"]).intersection(set(df_added["filepath"])))
print(f"Check 3: Stage-1 and added-30% overlap: {overlap} [OK]")
assert overlap == 0

# Check 4: Cumulative 50% contains no duplicates
c4 = len(df_cum50["filepath"]) == len(df_cum50["filepath"].unique())
print(f"Check 4: Cumulative 50% contains no duplicates? {c4} [OK]")
assert c4

# Check 5: Cumulative 50% contains only training samples
c5 = set(df_cum50["filepath"]).issubset(train_fps)
print(f"Check 5: Cumulative 50% contains only training samples? {c5} [OK]")
assert c5

# Check 6: Validation samples do not overlap with training samples
c6 = len(set(df_cum50["filepath"]).intersection(val_fps)) == 0
print(f"Check 6: Validation samples do not overlap with cumulative 50%? {c6} [OK]")
assert c6

# Check 7: Test samples are never accessed (check filepath pattern or lack of test split)
c7 = all(s == "train" for s in df_cum50["split"])
print(f"Check 7: All split labels in cumulative 50% are 'train' (no test)? {c7} [OK]")
assert c7

# Check 8: Class distribution approximately preserved
full_prop = df_train["class_idx"].value_counts(normalize=True).sort_index()
cum50_prop = df_cum50["class_idx"].value_counts(normalize=True).sort_index()
max_diff = (cum50_prop - full_prop).abs().max()
print(f"Check 8: Max class proportion difference vs full train: {max_diff:.6f} [OK]")
assert max_diff < 0.001

print("=" * 60)
print("ALL 8 INTEGRITY CHECKS PASSED.")
print("=" * 60)

# Save Stage 1 Reference summary
stage1_results_path = ROOT / "results/m7_1_sto_refinement/rapid_sto/rapid_sto_results.json"
with open(stage1_results_path, encoding="utf-8") as f:
    stage1_data = json.load(f)

stage1_summary = {
    "milestone": "M7.1 Rapid STO Screening",
    "training_subset": "20% stratified",
    "sample_count": len(df_stage1),
    "total_candidates": stage1_data["total_candidates_evaluated"],
    "epochs_per_candidate": 2,
    "best_candidate_id": stage1_data["best_candidate"]["candidate_id"],
    "best_validation_macro_f1": stage1_data["best_candidate"]["validation_macro_f1"],
    "best_hyperparameters": stage1_data["best_candidate"]["hyperparameters"] if "hyperparameters" in stage1_data["best_candidate"] else {
        "learning_rate": stage1_data["best_candidate"]["learning_rate"],
        "weight_decay": stage1_data["best_candidate"]["weight_decay"],
        "dropout": stage1_data["best_candidate"]["dropout"],
        "batch_size": stage1_data["best_candidate"]["batch_size"],
    },
    "default_baseline_val_macro_f1": stage1_data["default_baseline"]["val_macro_f1"],
}

with open(REF_DIR / "stage1_summary.json", "w", encoding="utf-8") as f:
    json.dump(stage1_summary, f, indent=2)

print(f"Saved stage1_summary.json to {REF_DIR / 'stage1_summary.json'}")
