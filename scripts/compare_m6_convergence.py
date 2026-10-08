"""
scripts/compare_m6_convergence.py
==================================
AgriAI M6 — Compare STO vs. Random Search Convergence.

Generates:
- results/m6_comparison/sto_vs_random_convergence.png
- results/m6_comparison/sto_convergence.json
- results/m6_comparison/random_convergence.json
- results/m6_comparison/comparison_summary.json
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    sto_dir = Path("results/m6_sto_dev")
    rs_dir = Path("results/m6_random_dev")
    out_dir = Path("results/m6_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load STO summary and fitness history
    with open(sto_dir / "summary.json", "r", encoding="utf-8") as f:
        sto_summary = json.load(f)
    with open(sto_dir / "fitness_history.json", "r", encoding="utf-8") as f:
        sto_history = json.load(f)

    # Load RS summary and fitness history
    with open(rs_dir / "summary.json", "r", encoding="utf-8") as f:
        rs_summary = json.load(f)
    with open(rs_dir / "fitness_history.json", "r", encoding="utf-8") as f:
        rs_history = json.load(f)

    sto_f1 = sto_history["best_so_far_f1"]
    rs_f1 = rs_history["best_so_far_f1"]

    sto_fit = sto_history["best_so_far_fitness"]
    rs_fit = rs_history["best_so_far_fitness"]

    # Save convergence JSONs
    with open(out_dir / "sto_convergence.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "sto_stabilized",
            "evaluations": len(sto_f1),
            "best_so_far_fitness": sto_fit,
            "best_so_far_f1": sto_f1,
            "best_fitness": sto_summary["best_fitness"],
            "best_val_macro_f1": sto_summary["best_val_macro_f1"],
            "best_params": sto_summary["best_params"],
        }, f, indent=2)

    with open(out_dir / "random_convergence.json", "w", encoding="utf-8") as f:
        json.dump({
            "method": "random_search",
            "evaluations": len(rs_f1),
            "best_so_far_fitness": rs_fit,
            "best_so_far_f1": rs_f1,
            "best_fitness": rs_summary["best_fitness"],
            "best_val_macro_f1": rs_summary["best_val_macro_f1"],
            "best_params": rs_summary["best_params"],
        }, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    evals_sto = list(range(1, len(sto_f1) + 1))
    evals_rs = list(range(1, len(rs_f1) + 1))

    ax.plot(evals_sto, [v * 100 for v in sto_f1],
            marker="o", markersize=6, linewidth=2.5, color="#1D4ED8",
            label=f"ProjectSpecificStabilizedSTO (Best F1: {sto_summary['best_val_macro_f1']*100:.2f}%)")
    ax.plot(evals_rs, [v * 100 for v in rs_f1],
            marker="s", markersize=6, linewidth=2.5, color="#DC2626", linestyle="--",
            label=f"Random Search Control (Best F1: {rs_summary['best_val_macro_f1']*100:.2f}%)")

    ax.set_xlabel("Objective Evaluation Number", fontsize=12, fontweight="medium")
    ax.set_ylabel("Best-so-far Validation Macro-F1 (%)", fontsize=12, fontweight="medium")
    ax.set_title("M6 Controlled Comparison: STO vs. Random Search Convergence\n(Equal Budget: 25 Evaluations, 128×128, 3 Epochs)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    plt.tight_layout()

    plot_path = out_dir / "sto_vs_random_convergence.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Convergence plot saved to: {plot_path}")

    # Summary
    diff = sto_summary["best_val_macro_f1"] - rs_summary["best_val_macro_f1"]
    summary = {
        "budget_evaluations": len(sto_f1),
        "sto": {
            "best_val_macro_f1": sto_summary["best_val_macro_f1"],
            "best_fitness": sto_summary["best_fitness"],
            "best_params": sto_summary["best_params"],
            "runtime_s": sto_summary.get("total_runtime_s", sto_summary.get("runtime_s")),
        },
        "random_search": {
            "best_val_macro_f1": rs_summary["best_val_macro_f1"],
            "best_fitness": rs_summary["best_fitness"],
            "best_params": rs_summary["best_params"],
            "runtime_s": rs_summary.get("total_runtime_s", rs_summary.get("runtime_s")),
        },
        "f1_difference_sto_minus_random": diff,
    }
    with open(out_dir / "comparison_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison summary saved to: {out_dir / 'comparison_summary.json'}")


if __name__ == "__main__":
    main()
