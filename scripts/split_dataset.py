"""
AgriAI — M1.4: Reproducible Stratified Split Implementation
Splits the PlantVillage tomato dataset into:
- Train: 70%
- Validation: 15%
- Test: 15%

Key Research Specification Guarantees:
1. Stratified by class label to strictly preserve class distribution across splits.
2. Fixed random seed: 42 (locked in M0).
3. Leakage prevention: Splits are created once and stored statically as CSV manifests.
4. Augmentation policy: Training augmentation is applied strictly at DataLoader runtime to Train only.
5. Saves CSV manifests to both data/ and data/splits/ for compatibility.
"""

import os
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = Path("d:/Leaf/data/raw/plantvillage_tomato")
DATA_DIR = Path("d:/Leaf/data")
SPLITS_DIR = Path("d:/Leaf/data/splits")
SEED = 42

def create_splits():
    print("=" * 70)
    print("AgriAI — M1.4: Reproducible Stratified Split")
    print(f"Random Seed: {SEED} (M0 Locked)")
    print(f"Target Ratios: Train=70%, Val=15%, Test=15%")
    print("=" * 70)

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in RAW_DATA_DIR.iterdir() if d.is_dir()], key=lambda d: d.name)
    class_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}

    records = []
    for cdir in class_dirs:
        class_name = cdir.name
        class_idx = class_to_idx[class_name]
        files = sorted([f for f in cdir.iterdir() if f.is_file()])
        for f in files:
            # Store path relative to d:/Leaf
            rel_path = f.relative_to(DATA_DIR.parent).as_posix()
            records.append({
                "filepath": rel_path,
                "filename": f.name,
                "class_name": class_name,
                "class_idx": class_idx
            })

    df = pd.DataFrame(records)
    print(f"Total verified samples: {len(df)}")
    print(f"Class count: {len(class_to_idx)}")

    # Step 1: Stratified train split (70%) vs temp (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=SEED,
        stratify=df["class_idx"],
        shuffle=True
    )

    # Step 2: Stratified temp split into Val (15% of total = 50% of temp) and Test (15% of total = 50% of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_df["class_idx"],
        shuffle=True
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    print("\nSplit Summary:")
    print(f"  Train samples:      {len(train_df):,} ({len(train_df)/len(df)*100:.2f}%)")
    print(f"  Validation samples: {len(val_df):,} ({len(val_df)/len(df)*100:.2f}%)")
    print(f"  Test samples:       {len(test_df):,} ({len(test_df)/len(df)*100:.2f}%)")
    print(f"  Total:              {len(df):,} (100.00%)")

    # Verify per-class proportions
    print("\nPer-Class Stratification Verification:")
    strat_records = []
    for cname, cidx in class_to_idx.items():
        total_c = len(df[df["class_idx"] == cidx])
        tr_c = len(train_df[train_df["class_idx"] == cidx])
        va_c = len(val_df[val_df["class_idx"] == cidx])
        te_c = len(test_df[test_df["class_idx"] == cidx])
        strat_records.append({
            "class_name": cname,
            "total": total_c,
            "train": tr_c,
            "train_pct": f"{tr_c/total_c*100:.1f}%",
            "val": va_c,
            "val_pct": f"{va_c/total_c*100:.1f}%",
            "test": te_c,
            "test_pct": f"{te_c/total_c*100:.1f}%"
        })
    strat_df = pd.DataFrame(strat_records)
    print(strat_df.to_string(index=False))

    # Save to data/ and data/splits/
    for base_dir in [DATA_DIR, SPLITS_DIR]:
        train_df.to_csv(base_dir / "train.csv", index=False)
        val_df.to_csv(base_dir / "val.csv", index=False)
        test_df.to_csv(base_dir / "test.csv", index=False)
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        full_df.to_csv(base_dir / "all_splits.csv", index=False)

    print(f"\nManifest CSVs saved to {DATA_DIR} and {SPLITS_DIR}")
    print("Stratified split completed successfully.")
    return train_df, val_df, test_df

if __name__ == "__main__":
    create_splits()
