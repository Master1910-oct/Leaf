"""
scripts/run_random_search.py
=============================
AgriAI M6 - Random Search Control Experiment.

Equal-budget comparison with STO.
Uses IDENTICAL training protocol, search space, and evaluation infrastructure.

Usage
-----
Development run (25 evaluations, same budget as STO dev):
  .venv\Scripts\python scripts/run_random_search.py `
      --num_evaluations 25 `
      --eval_epochs 3 `
      --image_size 128 `
      --output_dir results/m6_random_dev `
      --seed 12345

IMPORTANT:
  - Test set is NEVER accessed.  Fitness = 1 - validation_macro_F1.
  - Seed must differ from STO seed (42) -> use 12345 by default.
  - num_evaluations must equal STO's N*T for fair comparison.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.optimization.sto import SearchSpace
from src.optimization.fitness import FitnessEvaluator
from src.optimization.random_search import RandomSearch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_random_search")


def parse_args():
    p = argparse.ArgumentParser(description="AgriAI M6 - Random Search Control")
    p.add_argument("--num_evaluations", type=int, default=25,
                   help="Total candidate evaluations (must equal STO N*T).")
    p.add_argument("--eval_epochs",  type=int, default=3)
    p.add_argument("--image_size",   type=int, default=128)
    p.add_argument("--train_csv",    type=str, default="data/train.csv")
    p.add_argument("--val_csv",      type=str, default="data/val.csv")
    p.add_argument("--root_dir",     type=str, default="d:/Leaf")
    p.add_argument("--output_dir",   type=str, default="results/m6_random_dev")
    p.add_argument("--seed",         type=int, default=12345,
                   help="Random seed. MUST differ from STO seed (42).")
    p.add_argument("--base_channels",  type=int, default=64)
    p.add_argument("--num_stacks",     type=int, default=3)
    p.add_argument("--encoder_levels", type=int, default=4)
    p.add_argument("--train_subset_size", type=int, default=0,
                   help="If > 0, use a stratified subset of training data for proxy runtime feasibility.")
    p.add_argument("--device",         type=str, default="cpu")
    p.add_argument("--no_cache",       action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(out / "random_search_run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("="*70)
    logger.info("AgriAI M6 - Random Search Control")
    logger.info("="*70)
    logger.info("Evaluations : %d", args.num_evaluations)
    logger.info("Eval epochs : %d", args.eval_epochs)
    logger.info("Image size  : %d", args.image_size)
    logger.info("Seed        : %d (different from STO seed 42)", args.seed)
    logger.info("Output dir  : %s", out)
    logger.info("="*70)

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    # Load TRAIN and VAL only - never test
    logger.info("Loading datasets (train + val only, test is inaccessible)...")
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform   = get_eval_transforms(image_size=args.image_size)
    train_dataset = PlantVillageDataset(args.train_csv, args.root_dir, train_transform)
    val_dataset   = PlantVillageDataset(args.val_csv,   args.root_dir, val_transform)

    if args.train_subset_size > 0 and args.train_subset_size < len(train_dataset):
        logger.info(
            "Using stratified training subset of %d images (runtime proxy protocol)",
            args.train_subset_size
        )
        labels = train_dataset.df["class_idx"].tolist()
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, train_size=args.train_subset_size, random_state=args.seed)
        subset_idx, _ = next(sss.split(range(len(train_dataset)), labels))
        train_dataset = torch.utils.data.Subset(train_dataset, subset_idx)

    logger.info("Train: %d  Val: %d (FULL)  Test: UNTOUCHED", len(train_dataset), len(val_dataset))

    search_space = SearchSpace()
    evaluator = FitnessEvaluator(
        train_dataset=train_dataset, val_dataset=val_dataset,
        num_epochs=args.eval_epochs, search_space=search_space,
        method="random_search",
        base_channels=args.base_channels, num_stacks=args.num_stacks,
        encoder_levels=args.encoder_levels, num_classes=10,
        image_size=args.image_size, device=args.device,
        seed=args.seed, use_cache=not args.no_cache,
        save_dir=str(out / "evals"),
    )

    rs = RandomSearch(
        num_evaluations=args.num_evaluations,
        search_space=search_space,
        seed=args.seed,
        verbose=True,
    )

    logger.info("Starting Random Search...")
    t0 = time.perf_counter()
    result = rs.optimize(evaluator)
    total_runtime = time.perf_counter() - t0
    logger.info("Random Search complete. Runtime: %.1fs", total_runtime)

    # Save results
    summary = {
        "method": "random_search",
        "num_evaluations": result.num_evaluations,
        "eval_epochs": args.eval_epochs,
        "image_size": args.image_size,
        "seed": args.seed,
        "best_fitness": result.best_fitness,
        "best_val_macro_f1": float(1.0 - result.best_fitness),
        "best_params": result.best_params,
        "best_position": result.best_position.tolist(),
        "convergence_fitness": result.convergence,
        "convergence_f1": [1.0 - f for f in result.convergence],
        "runtime_s": result.runtime_s,
        "total_runtime_s": total_runtime,
    }
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Best config
    with open(out / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(result.best_params, f, indent=2)

    # All candidate records
    from src.optimization.fitness import record_to_dict
    candidates = [record_to_dict(r) for r in evaluator.candidate_log]
    with open(out / "candidate_results.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    # Fitness history
    fitness_history = {
        "best_so_far_fitness": evaluator.get_best_so_far(),
        "best_so_far_f1":      evaluator.get_best_so_far_f1(),
        "all_fitnesses":       result.all_fitnesses,
        "all_f1":              [1.0 - x for x in result.all_fitnesses],
    }
    with open(out / "fitness_history.json", "w", encoding="utf-8") as f:
        json.dump(fitness_history, f, indent=2)

    # History
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(fitness_history, f, indent=2)

    # config.yaml
    config_dict = {
        "experiment": "m6_random_dev",
        "method": "random_search",
        "num_evaluations": result.num_evaluations,
        "eval_epochs": args.eval_epochs,
        "image_size": args.image_size,
        "seed": args.seed,
        "search_space": {
            "learning_rate": [1e-4, 1e-2],
            "weight_decay": [1e-6, 1e-2],
            "dropout": [0.0, 0.5],
            "batch_size": [16, 32, 64],
        },
        "model": {
            "base_channels": args.base_channels,
            "num_stacks": args.num_stacks,
            "encoder_levels": args.encoder_levels,
        },
    }
    try:
        import yaml
        with open(out / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False)
    except Exception as e:
        logger.warning("Could not write config.yaml: %s", e)

    # convergence.png
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        evals_x = list(range(1, len(fitness_history["best_so_far_f1"]) + 1))
        plt.figure(figsize=(8, 5))
        plt.plot(evals_x, [f * 100 for f in fitness_history["best_so_far_f1"]],
                 marker="s", linewidth=2, color="#DC2626", label="Random Search")
        plt.xlabel("Objective Evaluation Number", fontsize=11)
        plt.ylabel("Best-so-far Validation Macro-F1 (%)", fontsize=11)
        plt.title("Random Search Convergence - Best Val Macro-F1", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out / "convergence.png", dpi=150)
        plt.close()
        logger.info("Saved convergence plot to %s", out / "convergence.png")
    except Exception as e:
        logger.warning("Could not save convergence plot: %s", e)

    logger.info("="*70)
    logger.info("RANDOM SEARCH COMPLETE")
    logger.info("  Evaluations      : %d", result.num_evaluations)
    logger.info("  Best fitness     : %.6f", result.best_fitness)
    logger.info("  Best val F1      : %.4f%%", (1.0-result.best_fitness)*100)
    logger.info("  Best params      : %s", result.best_params)
    logger.info("  Results saved to : %s", out)
    logger.info("  Test set         : UNTOUCHED AND INACCESSIBLE")
    logger.info("="*70)


if __name__ == "__main__":
    main()
