"""
scripts/run_proxy_fidelity.py
==============================
AgriAI M6 — Proxy-Fidelity Verification Experiment.

Evaluates whether the short proxy training ranking (128x128, 3 epochs)
correlates with longer reference training ranking (128x128, 6 epochs)
across a sample of candidate configurations.

Calculates:
- Spearman rank correlation (rho) and p-value
- Outputs:
  results/m6_proxy_fidelity/proxy_fidelity.json
  results/m6_proxy_fidelity/proxy_vs_long_training.csv
  results/m6_proxy_fidelity/proxy_fidelity_plot.png

CRITICAL RULE:
Test set is NEVER touched. Only train and val splits are used.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.optimization.sto import SearchSpace
from src.optimization.fitness import FitnessEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("proxy_fidelity")


def get_candidate_configs(num_configs: int = 5, seed: int = 777) -> List[Dict[str, Any]]:
    """Sample diverse candidate configurations across the search space."""
    rng = np.random.default_rng(seed)
    ss = SearchSpace()
    configs = []
    for i in range(num_configs):
        theta = ss.random_position(rng)
        params = ss.decode(theta)
        configs.append({
            "config_id": i + 1,
            "theta": theta.tolist(),
            **params
        })
    return configs


def main():
    parser = argparse.ArgumentParser(description="AgriAI M6 — Proxy-Fidelity Verification")
    parser.add_argument("--num_configs", type=int, default=5,
                        help="Number of candidate configurations to test (default: 5).")
    parser.add_argument("--proxy_epochs", type=int, default=3,
                        help="Proxy training epochs (default: 3).")
    parser.add_argument("--long_epochs", type=int, default=5,
                        help="Longer reference training epochs (default: 5).")
    parser.add_argument("--image_size", type=int, default=128,
                        help="Image size (default: 128).")
    parser.add_argument("--train_csv", type=str, default="data/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/val.csv")
    parser.add_argument("--root_dir", type=str, default="d:/Leaf")
    parser.add_argument("--output_dir", type=str, default="results/m6_proxy_fidelity")
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train_subset_size", type=int, default=0,
                        help="If > 0, train on a stratified subset to respect CPU budget.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("AgriAI M6 — Proxy-Fidelity Experiment")
    logger.info("=" * 70)
    logger.info("Configurations    : %d", args.num_configs)
    logger.info("Proxy budget      : %d epochs, %dx%d", args.proxy_epochs, args.image_size, args.image_size)
    logger.info("Long budget       : %d epochs, %dx%d", args.long_epochs, args.image_size, args.image_size)
    logger.info("Seed              : %d", args.seed)
    logger.info("=" * 70)

    # Datasets (train + val only — NEVER test)
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform   = get_eval_transforms(image_size=args.image_size)
    train_ds = PlantVillageDataset(args.train_csv, args.root_dir, train_transform)
    val_ds   = PlantVillageDataset(args.val_csv,   args.root_dir, val_transform)

    if args.train_subset_size > 0 and args.train_subset_size < len(train_ds):
        logger.info("Using stratified training subset of %d images for CPU feasibility", args.train_subset_size)
        from sklearn.model_selection import StratifiedShuffleSplit
        labels = train_ds.df["class_idx"].tolist()
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.train_subset_size, random_state=args.seed)
        indices, _ = next(sss.split(range(len(train_ds)), labels))
        train_ds = torch.utils.data.Subset(train_ds, indices)

    logger.info("Train: %d | Val: %d | Test: UNTOUCHED", len(train_ds), len(val_ds))

    configs = get_candidate_configs(args.num_configs, seed=args.seed)
    ss = SearchSpace()

    results = []

    for cfg in configs:
        cid = cfg["config_id"]
        theta = np.array(cfg["theta"])
        params = {k: cfg[k] for k in ["learning_rate", "weight_decay", "dropout", "batch_size"]}

        logger.info("--- Configuration #%d/%d ---", cid, args.num_configs)
        logger.info("Params: %s", params)

        # 1. Proxy run (e.g. 3 epochs)
        logger.info("Running Proxy evaluation (%d epochs)...", args.proxy_epochs)
        ev_proxy = FitnessEvaluator(
            train_dataset=train_ds, val_dataset=val_ds,
            num_epochs=args.proxy_epochs, search_space=ss,
            method="proxy_eval", image_size=args.image_size,
            device=args.device, seed=args.seed + cid * 10,
            use_cache=False, save_dir=None,
        )
        proxy_fit = ev_proxy(theta)
        proxy_f1 = 1.0 - proxy_fit
        proxy_rec = ev_proxy.candidate_log[0]

        # 2. Long run (e.g. 5 epochs)
        logger.info("Running Long reference evaluation (%d epochs)...", args.long_epochs)
        ev_long = FitnessEvaluator(
            train_dataset=train_ds, val_dataset=val_ds,
            num_epochs=args.long_epochs, search_space=ss,
            method="long_eval", image_size=args.image_size,
            device=args.device, seed=args.seed + cid * 10,
            use_cache=False, save_dir=None,
        )
        long_fit = ev_long(theta)
        long_f1 = 1.0 - long_fit
        long_rec = ev_long.candidate_log[0]

        logger.info("Config #%d: Proxy F1=%.4f%% | Long F1=%.4f%%",
                    cid, proxy_f1 * 100, long_f1 * 100)

        results.append({
            "config_id": cid,
            "learning_rate": params["learning_rate"],
            "weight_decay": params["weight_decay"],
            "dropout": params["dropout"],
            "batch_size": params["batch_size"],
            "proxy_epochs": args.proxy_epochs,
            "proxy_val_macro_f1": proxy_f1,
            "proxy_val_loss": proxy_rec.validation_loss,
            "proxy_val_accuracy": proxy_rec.validation_accuracy,
            "proxy_time_s": proxy_rec.training_time_seconds,
            "long_epochs": args.long_epochs,
            "long_val_macro_f1": long_f1,
            "long_val_loss": long_rec.validation_loss,
            "long_val_accuracy": long_rec.validation_accuracy,
            "long_time_s": long_rec.training_time_seconds,
        })

    # Rankings
    proxy_scores = [r["proxy_val_macro_f1"] for r in results]
    long_scores  = [r["long_val_macro_f1"] for r in results]

    rho, pval = spearmanr(proxy_scores, long_scores)
    logger.info("=" * 70)
    logger.info("Spearman Rank Correlation: rho = %.4f (p = %.4f)", rho, pval)
    logger.info("=" * 70)

    # Save CSV
    csv_path = out_dir / "proxy_vs_long_training.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved CSV to %s", csv_path)

    # Save JSON summary
    summary = {
        "experiment": "m6_proxy_fidelity",
        "num_configs": args.num_configs,
        "proxy_protocol": {
            "epochs": args.proxy_epochs,
            "image_size": args.image_size,
        },
        "long_protocol": {
            "epochs": args.long_epochs,
            "image_size": args.image_size,
        },
        "spearman_rank_correlation": float(rho) if not np.isnan(rho) else 0.0,
        "p_value": float(pval) if not np.isnan(pval) else 1.0,
        "results": results,
        "interpretation": (
            f"Observed Spearman rho = {rho:.4f} (p = {pval:.4f}) between {args.proxy_epochs}-epoch proxy and "
            f"{args.long_epochs}-epoch reference training over {args.num_configs} configurations. "
            "Note that a small sample size provides empirical indication rather than definitive proof of proxy fidelity."
        ),
        "limitations": [
            "Evaluated on a sample of configurations due to CPU compute constraints.",
            "Proxy fidelity may vary between extreme configurations and near-optimal configurations.",
            "Validation Macro-F1 ranking is an empirical heuristic, not a guarantee of monotonic rank preservation.",
        ],
    }
    with open(out_dir / "proxy_fidelity.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved JSON to %s", out_dir / "proxy_fidelity.json")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([s * 100 for s in proxy_scores], [s * 100 for s in long_scores],
               color="#2563EB", s=80, zorder=5, label="Configurations")
    for r in results:
        ax.annotate(f"C{r['config_id']}",
                    (r["proxy_val_macro_f1"] * 100 + 0.3, r["long_val_macro_f1"] * 100 + 0.3),
                    fontsize=9)
    ax.set_xlabel(f"Proxy Validation Macro-F1 (%) [{args.proxy_epochs} Epochs, {args.image_size}x{args.image_size}]", fontsize=11)
    ax.set_ylabel(f"Reference Validation Macro-F1 (%) [{args.long_epochs} Epochs, {args.image_size}x{args.image_size}]", fontsize=11)
    ax.set_title(f"M6 Proxy Fidelity Analysis: Short vs. Long Training\nSpearman Rank Correlation: $\\rho = {rho:.4f}$ ($p = {pval:.4f}$)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "proxy_fidelity_plot.png", dpi=150)
    plt.close()
    logger.info("Saved plot to %s", out_dir / "proxy_fidelity_plot.png")


if __name__ == "__main__":
    main()
