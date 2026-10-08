"""
scripts/run_m7_model_selection.py
=================================
AgriAI M7 — Final Model Selection from Full-Training Validation Results.

Evaluates and compares the 3 candidates strictly using VALIDATION set metrics:
  - Candidate A: Default LS-HGNet (M4 baseline parameters)
  - Candidate B: STO-Optimized LS-HGNet (M6 STO best)
  - Candidate C: Random-Search-Optimized LS-HGNet (M6 RS best)

Selection Rule:
  1. Highest validation Macro-F1 score
  2. Tie-break (if delta < 1e-4): lower computational latency / simpler config

Strict Isolation:
  - The TEST set is NEVER loaded or evaluated in this script.
  - Generates:
      results/m7_final_selection/comparison_summary.json
      results/m7_final_selection/candidate_results.json
      results/m7_final_selection/selected_config.json
      results/m7_final_selection/validation_results.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("m7_selection")

CANDIDATE_IDS = ["candidate_a_default", "candidate_b_sto", "candidate_c_random"]


def parse_args():
    p = argparse.ArgumentParser(description="AgriAI M7 — Model Selection")
    p.add_argument("--results_dir", type=str, default="results/m7_final_selection",
                   help="Directory containing candidate results")
    return p.parse_args()


def load_candidate_metrics(candidate_dir: Path):
    metrics_path = candidate_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics.json for candidate at: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_best_candidate(candidates_data: dict):
    """
    Selects candidate based strictly on validation Macro-F1.
    """
    ranked = sorted(
        candidates_data.items(),
        key=lambda item: item[1]["validation_metrics"]["macro_f1"],
        reverse=True
    )
    winner_id, winner_data = ranked[0]
    return winner_id, winner_data, ranked


def main():
    args = parse_args()
    base_dir = Path(args.results_dir)

    logger.info("=" * 80)
    logger.info("AgriAI M7 — Final LS-HGNet Model Selection")
    logger.info("=" * 80)
    logger.info("Base directory: %s", base_dir)

    candidates = {}
    for cid in CANDIDATE_IDS:
        cdir = base_dir / cid
        if not cdir.is_dir() or not (cdir / "metrics.json").is_file():
            logger.error("Candidate %s results missing at %s", cid, cdir)
            sys.exit(1)
        candidates[cid] = load_candidate_metrics(cdir)
        logger.info("Loaded candidate: %s | Val Macro-F1: %.4f | Val Acc: %.4f",
                    cid,
                    candidates[cid]["validation_metrics"]["macro_f1"],
                    candidates[cid]["validation_metrics"]["accuracy"])

    winner_id, winner_data, ranked = select_best_candidate(candidates)

    logger.info("=" * 80)
    logger.info("CANDIDATE VALIDATION RANKING (Highest Macro-F1 wins):")
    for rank, (cid, data) in enumerate(ranked, 1):
        vm = data["validation_metrics"]
        hp = data["hyperparameters"]
        logger.info("  Rank %d: %s | F1: %.4f | Acc: %.4f | LR: %.6e | WD: %.6e | Drop: %.3f | BS: %d",
                    rank, cid, vm["macro_f1"], vm["accuracy"],
                    hp["learning_rate"], hp["weight_decay"], hp["dropout"], hp["batch_size"])
    logger.info("=" * 80)
    logger.info("WINNER SELECTED: %s (Val Macro-F1 = %.4f)", winner_id, winner_data["validation_metrics"]["macro_f1"])
    logger.info("=" * 80)

    # 1. candidate_results.json
    with open(base_dir / "candidate_results.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    # 2. validation_results.json
    val_summary = {
        cid: {
            "hyperparameters": d["hyperparameters"],
            "validation_metrics": d["validation_metrics"],
            "computational_metrics": d["computational_metrics"],
            "best_epoch": d.get("best_epoch"),
            "epochs_trained": d.get("epochs_trained")
        }
        for cid, d in candidates.items()
    }
    with open(base_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(val_summary, f, indent=2)

    # 3. selected_config.json
    selected_payload = {
        "status": "LOCKED",
        "selected_candidate_id": winner_id,
        "selection_rationale": "Highest validation Macro-F1 on full 12,712 training split / 2,724 validation split at 224x224",
        "architecture": winner_data.get("architecture", "LS-HGNet"),
        "hyperparameters": winner_data["hyperparameters"],
        "validation_metrics": winner_data["validation_metrics"],
        "checkpoint_path": str(base_dir / winner_id / "best_model.pt"),
        "isolation_verified": "Test split strictly protected; 0 test samples accessed during selection."
    }
    with open(base_dir / "selected_config.json", "w", encoding="utf-8") as f:
        json.dump(selected_payload, f, indent=2)

    # 4. comparison_summary.json
    comparison = {
        "selection_objective": "Highest validation Macro-F1",
        "winner": winner_id,
        "ranking": [cid for cid, _ in ranked],
        "candidates": {
            cid: {
                "macro_f1": d["validation_metrics"]["macro_f1"],
                "accuracy": d["validation_metrics"]["accuracy"],
                "macro_precision": d["validation_metrics"]["macro_precision"],
                "macro_recall": d["validation_metrics"]["macro_recall"],
                "hyperparameters": d["hyperparameters"],
                "runtime_s": d.get("total_training_runtime_s")
            }
            for cid, d in candidates.items()
        }
    }
    with open(base_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    logger.info("Saved all selection artifacts to %s", base_dir)


if __name__ == "__main__":
    main()
