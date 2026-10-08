"""
scripts/run_sto.py
==================
AgriAI M5 - Sooty Tern Optimization (STO) Hyperparameter Search Script.

Usage
-----
Development run (fast validation, ~25 evaluations):
  .venv\Scripts\python scripts/run_sto.py `
      --variant stabilized `
      --num_agents 5 --num_iterations 5 `
      --eval_epochs 3 `
      --output_dir results/sto_dev_stabilized

Final run (100 evaluations, matches experimental protocol):
  .venv\Scripts\python scripts/run_sto.py `
      --variant stabilized `
      --num_agents 10 --num_iterations 10 `
      --eval_epochs 5 `
      --output_dir results/sto_final_stabilized

Original/strict ablation run:
  .venv\Scripts\python scripts/run_sto.py `
      --variant original_strict `
      --num_agents 10 --num_iterations 10 `
      --eval_epochs 5 `
      --output_dir results/sto_final_original

IMPORTANT:
  - Test set is NEVER accessed.  Fitness = 1 - validation_macro_F1.
  - Both variants use identical evaluation infrastructure.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.optimization.sto import SearchSpace, create_optimizer
from src.optimization.fitness import STOFitnessEvaluator

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("run_sto")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgriAI M5 - STO Hyperparameter Optimization"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="stabilized",
        choices=["stabilized", "original_strict"],
        help=(
            "STO variant: 'stabilized' (project-modified, default) or "
            "'original_strict' (faithful Dhiman & Kaur 2019 ablation)."
        ),
    )
    parser.add_argument(
        "--num_agents", type=int, default=10,
        help="Population size N (default: 10 for final, 5 for dev).",
    )
    parser.add_argument(
        "--num_iterations", type=int, default=10,
        help="Number of STO iterations T (default: 10 for final, 5 for dev).",
    )
    parser.add_argument(
        "--eval_epochs", type=int, default=5,
        help="Training epochs per fitness evaluation (default: 5).",
    )
    parser.add_argument(
        "--image_size", type=int, default=224,
        help="Input image size (default: 224).",
    )
    parser.add_argument(
        "--train_csv", type=str, default="data/train.csv",
        help="Path to training split CSV.",
    )
    parser.add_argument(
        "--val_csv", type=str, default="data/val.csv",
        help="Path to validation split CSV.",
    )
    parser.add_argument(
        "--root_dir", type=str, default="d:/Leaf",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/sto_dev_stabilized",
        help="Output directory for results and logs.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42).",
    )
    parser.add_argument(
        "--base_channels", type=int, default=64,
        help="LS-HGNet base_channels (must match M4 final config).",
    )
    parser.add_argument(
        "--num_stacks", type=int, default=3,
        help="LS-HGNet num_stacks (must match M4 final config).",
    )
    parser.add_argument(
        "--encoder_levels", type=int, default=4,
        help="LS-HGNet encoder_levels (must match M4 final config).",
    )
    parser.add_argument(
        "--train_subset_size", type=int, default=0,
        help="If > 0, use a stratified subset of training data for proxy runtime feasibility.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device (default: cpu).",
    )
    parser.add_argument(
        "--no_cache", action="store_true",
        help="Disable fitness caching.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # File logger
    fh = logging.FileHandler(output_dir / "sto_run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)

    logger.info("=" * 70)
    logger.info("AgriAI M5 - STO Hyperparameter Optimization")
    logger.info("=" * 70)
    logger.info("Variant          : %s", args.variant)
    logger.info("Agents (N)       : %d", args.num_agents)
    logger.info("Iterations (T)   : %d", args.num_iterations)
    logger.info("Total evaluations: %d", args.num_agents * args.num_iterations)
    logger.info("Eval epochs      : %d", args.eval_epochs)
    logger.info("Image size       : %d", args.image_size)
    logger.info("Seed             : %d", args.seed)
    logger.info("Output dir       : %s", output_dir)
    logger.info("=" * 70)

    if args.variant == "original_strict":
        logger.warning(
            "Running ORIGINAL/STRICT STOA (Dhiman & Kaur 2019 source-derived). "
            "Known numerical pathologies (exponential explosion, zero-bias) will "
            "appear. This is expected for ablation study EXP_003_original."
        )

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load datasets (NEVER load test split here)
    logger.info("Loading datasets...")
    train_transform = get_train_transforms(image_size=args.image_size)
    val_transform   = get_eval_transforms(image_size=args.image_size)

    train_dataset = PlantVillageDataset(
        manifest_csv=args.train_csv,
        root_dir=args.root_dir,
        transform=train_transform,
    )
    val_dataset = PlantVillageDataset(
        manifest_csv=args.val_csv,
        root_dir=args.root_dir,
        transform=val_transform,
    )

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

    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Val samples  : %d (FULL validation split)", len(val_dataset))

    # Build fitness evaluator
    search_space = SearchSpace()
    evaluator = STOFitnessEvaluator(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.eval_epochs,
        search_space=search_space,
        base_channels=args.base_channels,
        num_stacks=args.num_stacks,
        encoder_levels=args.encoder_levels,
        num_classes=10,
        image_size=args.image_size,
        device=args.device,
        seed=args.seed,
        use_cache=not args.no_cache,
        eval_id_prefix=args.variant[:4],
    )

    # Build STO optimizer
    optimizer = create_optimizer(
        variant=args.variant,
        num_agents=args.num_agents,
        num_iterations=args.num_iterations,
        search_space=search_space,
        seed=args.seed,
        verbose=True,
    )

    # Run optimization
    logger.info("Starting STO optimization...")
    t0 = time.perf_counter()
    result = optimizer.optimize(evaluator)
    total_runtime = time.perf_counter() - t0
    logger.info("STO optimization complete. Total runtime: %.1fs", total_runtime)

    # Save results
    summary = {
        "variant":             result.variant,
        "num_agents":          args.num_agents,
        "num_iterations":      args.num_iterations,
        "total_evaluations":   result.num_evaluations,
        "eval_epochs":         args.eval_epochs,
        "image_size":          args.image_size,
        "seed":                args.seed,
        "best_fitness":        result.best_fitness,
        "best_val_macro_f1":   float(1.0 - result.best_fitness),
        "best_params":         result.best_params,
        "best_position":       result.best_position.tolist(),
        "convergence":         result.convergence,
        "convergence_f1":      [1.0 - f for f in result.convergence],
        "all_fitnesses":       result.all_fitnesses,
        "all_f1":              [1.0 - f for f in result.all_fitnesses],
        "runtime_s":           result.runtime_s,
        "total_runtime_s":     total_runtime,
        "base_channels":       args.base_channels,
        "num_stacks":          args.num_stacks,
        "encoder_levels":      args.encoder_levels,
    }

    # summary.json & sto_results.json
    for fname in ["summary.json", "sto_results.json"]:
        with open(output_dir / fname, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # best_config.json
    with open(output_dir / "best_config.json", "w", encoding="utf-8") as f:
        json.dump(result.best_params, f, indent=2)

    # candidate_results.json
    from src.optimization.fitness import record_to_dict
    candidates = [record_to_dict(r) for r in evaluator.candidate_log]
    with open(output_dir / "candidate_results.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    # fitness_history.json
    fitness_history = {
        "best_so_far_fitness": evaluator.get_best_so_far(),
        "best_so_far_f1":      evaluator.get_best_so_far_f1(),
        "all_fitnesses":       result.all_fitnesses,
        "all_f1":              [1.0 - f for f in result.all_fitnesses],
    }
    with open(output_dir / "fitness_history.json", "w", encoding="utf-8") as f:
        json.dump(fitness_history, f, indent=2)

    # history.json
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(fitness_history, f, indent=2)

    # config.yaml
    config_dict = {
        "experiment": "m6_sto_dev",
        "variant": args.variant,
        "num_agents": args.num_agents,
        "num_iterations": args.num_iterations,
        "total_evaluations": result.num_evaluations,
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
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
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
                 marker="o", linewidth=2, color="#2563EB", label=f"STO ({args.variant})")
        plt.xlabel("Objective Evaluation Number", fontsize=11)
        plt.ylabel("Best-so-far Validation Macro-F1 (%)", fontsize=11)
        plt.title(f"STO Convergence ({args.variant}) - Best Val Macro-F1", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(output_dir / "convergence.png", dpi=150)
        plt.close()
        logger.info("Saved convergence plot to %s", output_dir / "convergence.png")
    except Exception as e:
        logger.warning("Could not save convergence plot: %s", e)

    logger.info("=" * 70)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Variant           : %s", result.variant)
    logger.info("Total evaluations : %d", result.num_evaluations)
    logger.info("Best fitness      : %.6f", result.best_fitness)
    logger.info("Best val macro-F1 : %.4f%%", (1.0 - result.best_fitness) * 100)
    logger.info("Best hyperparameters:")
    for k, v in result.best_params.items():
        logger.info("  %-20s : %s", k, v)
    logger.info("Results saved to  : %s", output_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
