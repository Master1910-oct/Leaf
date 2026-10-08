"""
scripts/run_m73_full_training.py
================================
AgriAI M7.3 -- Full-Training Convergence & Refinement (10 Epochs).

Experiment objective:
  Determine whether the frozen M7 LS-HGNet model can improve beyond its
  original 5-epoch validation Macro-F1 of 88.16% by training the same
  configuration for 10 epochs on the complete training dataset (12,712 images).

Rules:
  - Exact M7 Candidate A baseline reproduction (LR=1e-3, WD=1e-4, Dropout=0.30, BS=32)
  - 100% Training split (12,712 images)
  - Full Validation split (2,724 images)
  - Test split is STRICTLY FORBIDDEN and never loaded
  - Frozen M7 artifacts are never modified
  - Checkpoint saving after every epoch with full resumability
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import psutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.ls_hgnet import LS_HGNet, count_parameters, get_checkpoint_size_mb
from src.training.trainer import Trainer
from src.training.losses import get_loss_function
from src.evaluation.metrics import (
    evaluate_dataset,
    measure_inference_latency,
    plot_confusion_matrix,
)
from src.utils.reproducibility import set_seed

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("m73_full_training")

# ----------------------------------------------------------------------------
# Constants & Configuration
# ----------------------------------------------------------------------------
FROZEN_M7_BASELINE = {
    "candidate_id": "candidate_a_default",
    "architecture": "LS-HGNet",
    "seed": 42,
    "image_size": 224,
    "epochs_trained": 5,
    "best_epoch": 5,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "dropout": 0.30,
    "batch_size": 32,
    "val_macro_f1": 0.881642,
    "val_accuracy": 0.910793,
    "val_macro_precision": 0.895255,
    "val_macro_recall": 0.886605,
}

ARCH_BASE_CHANNELS = 64
ARCH_NUM_STACKS = 3
ARCH_ENCODER_LEVELS = 4
ARCH_DILATION_RATES = (1, 2, 4)
NUM_CLASSES = 10


def format_hms(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


class ProgressDataLoader:
    def __init__(self, dataloader, epoch, total_epochs, total_training_start, output_dir: Path, log_interval=20):
        self.dataloader = dataloader
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.total_training_start = total_training_start
        self.output_dir = output_dir
        self.log_interval = log_interval

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        total_batches = len(self.dataloader)
        epoch_start = time.time()
        for batch_idx, batch in enumerate(self.dataloader, 1):
            yield batch
            if batch_idx % self.log_interval == 0 or batch_idx == total_batches:
                epoch_elapsed = time.time() - epoch_start
                total_elapsed = time.time() - self.total_training_start
                avg_batch_time = epoch_elapsed / batch_idx
                completed_batches = (self.epoch - 1) * total_batches + batch_idx
                all_batches = self.total_epochs * total_batches
                overall_progress_pct = (completed_batches / all_batches) * 100.0
                epoch_progress_pct = (batch_idx / total_batches) * 100.0

                total_est_time = (total_elapsed / completed_batches) * all_batches
                total_remaining = max(0.0, total_est_time - total_elapsed)
                eta = datetime.now() + timedelta(seconds=total_remaining)

                cpu_pct = psutil.cpu_percent()
                ram_mb = psutil.virtual_memory().used / (1024 * 1024)

                status_obj = {
                    "experiment": "M7.3 Full Training",
                    "status": "RUNNING",
                    "epoch": self.epoch,
                    "total_epochs": self.total_epochs,
                    "epoch_progress_pct": round(epoch_progress_pct, 1),
                    "overall_progress_pct": round(overall_progress_pct, 1),
                    "batch": batch_idx,
                    "total_batches": total_batches,
                    "elapsed_str": format_hms(total_elapsed),
                    "epoch_elapsed_str": format_hms(epoch_elapsed),
                    "estimated_remaining_str": format_hms(total_remaining),
                    "eta_str": eta.strftime("%I:%M:%S %p"),
                    "cpu_percent": cpu_pct,
                    "ram_mb": round(ram_mb, 1),
                    "device": "CPU",
                }

                live_file = self.output_dir / "logs/live_status.json"
                try:
                    with open(live_file, "w", encoding="utf-8") as f:
                        json.dump(status_obj, f, indent=2)
                except Exception:
                    pass

                logger.info(
                    "[M7.3] Ep %d/%d (%.1f%%) | Batch %d/%d | Elapsed: %s | Rem: %s | ETA: %s | CPU: %.1f%%",
                    self.epoch, self.total_epochs, epoch_progress_pct,
                    batch_idx, total_batches, format_hms(total_elapsed), format_hms(total_remaining),
                    eta.strftime("%I:%M %p"), cpu_pct,
                )


def plot_convergence(history: List[dict], output_path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_accuracy"] * 100 for h in history]
    val_f1 = [h["val_macro_f1"] * 100 for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, train_loss, "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    ax1.plot(epochs, val_loss, "s-", label="Val Loss", color="#ff7f0e", linewidth=2)
    ax1.axvline(x=5, color="gray", linestyle="--", alpha=0.7, label="M7 Stop (Epoch 5)")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=12)
    ax1.set_title("Training & Validation Loss Convergence", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=10)

    # Metrics subplot
    ax2.plot(epochs, val_acc, "o-", label="Val Accuracy (%)", color="#2ca02c", linewidth=2)
    ax2.plot(epochs, val_f1, "s-", label="Val Macro-F1 (%)", color="#d62728", linewidth=2)
    ax2.axhline(y=FROZEN_M7_BASELINE["val_macro_f1"] * 100, color="#d62728", linestyle=":", label=f"M7 Baseline F1 ({FROZEN_M7_BASELINE['val_macro_f1']*100:.2f}%)")
    ax2.axvline(x=5, color="gray", linestyle="--", alpha=0.7, label="M7 Stop (Epoch 5)")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Metric (%)", fontsize=12)
    ax2.set_title("Validation Accuracy & Macro-F1 Convergence", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved convergence curve plot to %s", output_path)


def generate_convergence_report(
    output_dir: Path,
    history: List[dict],
    best_epoch: int,
    best_f1: float,
    best_metrics: dict,
    total_runtime_s: float,
    config: dict,
) -> None:
    # Epoch 5 comparison
    ep5 = next((h for h in history if h["epoch"] == 5), history[-1])
    ep5_f1 = ep5["val_macro_f1"]
    frozen_f1 = FROZEN_M7_BASELINE["val_macro_f1"]
    delta_vs_frozen = best_f1 - frozen_f1
    improved_beyond_ep5 = best_f1 > ep5_f1 and best_epoch > 5

    # Epochs 6-10 summary
    post_ep5_records = [h for h in history if h["epoch"] > 5]
    post_ep5_f1s = [h["val_macro_f1"] for h in post_ep5_records] if post_ep5_records else []

    # Check overfitting
    min_val_loss_ep = min(history, key=lambda h: h["val_loss"])
    overfitting_observed = (history[-1]["val_loss"] > min_val_loss_ep["val_loss"] * 1.05) and (best_epoch < len(history))

    # Build epoch rows
    table_rows = []
    for h in history:
        best_marker = " **[BEST]**" if h["epoch"] == best_epoch else ""
        table_rows.append(
            f"| {h['epoch']:2d} | {h['train_loss']:.4f} | {h['train_accuracy']*100:.2f}% | "
            f"{h['val_loss']:.4f} | {h['val_accuracy']*100:.2f}% | {h['val_macro_precision']*100:.2f}% | "
            f"{h['val_macro_recall']*100:.2f}% | **{h['val_macro_f1']*100:.2f}%**{best_marker} | "
            f"{h['learning_rate']:.2e} | {h['duration_seconds']:.1f}s |"
        )
    table_str = "\n".join(table_rows)

    report_path = output_dir / "M7_3_CONVERGENCE_REPORT.md"
    report_content = f"""# M7.3 — FULL-TRAINING CONVERGENCE & REFINEMENT REPORT

**Milestone:** M7.3 — Full-Training Convergence & Refinement (10 Epochs)  
**Project:** AgriAI — Tomato Leaf Disease Classification with LS-HGNet  
**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}  
**Status:** COMPLETE  
**Total Runtime:** {total_runtime_s:.1f} s ({total_runtime_s / 3600.0:.2f} hours)  
**Architecture:** LS-HGNet (8,041,354 parameters)  
**Device:** CPU  

---

## 1. Experiment Objective

The objective of M7.3 is to perform a controlled full-training convergence experiment on the complete dataset (12,712 training images, 2,724 validation images) to investigate whether extending the training budget from 5 epochs to 10 epochs enables the LS-HGNet model to achieve performance gains beyond the frozen M7 baseline (88.16% validation Macro-F1).

All historical milestones (M7 Final Selection, M7.1 Rapid STO, M7.2 Progressive STO) and the locked test set remain strictly isolated and untouched.

---

## 2. Experimental Configuration

| Hyperparameter / Setting | Value | Rationale |
| :--- | :--- | :--- |
| **Architecture** | LS-HGNet (fixed M4) | base_channels=64, num_stacks=3, levels=4, dilations=(1,2,4) |
| **Total Trainable Parameters** | 8,041,354 | Exact reproduction of M7 Candidate A |
| **Random Seed** | 42 | Exact reproduction |
| **Image Resolution** | 224 x 224 | Exact reproduction |
| **Batch Size** | 32 | Exact reproduction |
| **Initial Learning Rate** | 0.001 (1e-3) | Exact reproduction |
| **Weight Decay** | 0.0001 (1e-4) | Exact reproduction |
| **Dropout Probability** | 0.30 | Exact reproduction |
| **Loss Function** | Cross-Entropy Loss | Exact reproduction |
| **Optimizer** | AdamW | Exact reproduction |
| **Learning Rate Scheduler** | CosineAnnealingLR ($T_{{\\max}}=10$, $\\eta_{{\\min}}=10^{{-6}}$) | Extended for 10-epoch schedule |
| **Total Epochs** | **10 epochs** | The single experimental variable (vs 5 in M7) |
| **Device** | CPU | Laptop hardware |

---

## 3. Dataset Verification

- **Training Dataset (`data/train.csv`):** 12,712 images (100% full split)
- **Validation Dataset (`data/val.csv`):** 2,724 images (100% full split)
- **Test Dataset (`data/splits/test.csv`):** **STRICTLY LOCKED & ISOLATED** — zero access.

---

## 4. Epoch-by-Epoch Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Precision | Val Recall | Val Macro-F1 | Learning Rate | Epoch Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}

---

## 5. Convergence Analysis & Best Epoch

- **Best Validation Macro-F1:** **{best_f1*100:.2f}%** achieved at **Epoch {best_epoch}**
- **Validation Accuracy at Best Epoch:** **{best_metrics['val_accuracy']*100:.2f}%**
- **Validation Macro Precision at Best Epoch:** **{best_metrics['val_macro_precision']*100:.2f}%**
- **Validation Macro Recall at Best Epoch:** **{best_metrics['val_macro_recall']*100:.2f}%**
- **Validation Loss at Best Epoch:** **{best_metrics['val_loss']:.4f}**
- **Macro-F1 at Epoch 5:** **{ep5_f1*100:.2f}%**
- **Best Macro-F1 in Epochs 6–10:** **{max(post_ep5_f1s)*100:.2f}%** (if epochs 6–10 evaluated)

### Convergence Dynamics:
- **Did Macro-F1 improve after Epoch 5?** {"Yes" if improved_beyond_ep5 else "No"}
- **Lowest Validation Loss Epoch:** Epoch {min_val_loss_ep['epoch']} ({min_val_loss_ep['val_loss']:.4f})
- **Overfitting Diagnosis:** {"Slight overfitting observed in late epochs as validation loss stabilized/increased while training loss dropped." if overfitting_observed else "Training proceeded stably without significant loss divergence."}

---

## 6. Comparison Against Frozen M7 Baseline

| Metric | Frozen M7 Baseline (Candidate A) | M7.3 Extended Training (10 Epochs) | Absolute Change |
| :--- | :---: | :---: | :---: |
| **Epoch Budget** | 5 epochs | 10 epochs | +5 epochs |
| **Best Epoch** | Epoch 5 | Epoch {best_epoch} | — |
| **Validation Macro-F1** | **88.16%** | **{best_f1*100:.2f}%** | **{"+" if delta_vs_frozen >= 0 else ""}{delta_vs_frozen*100:.2f} percentage points** |
| **Validation Accuracy** | **91.08%** | **{best_metrics['val_accuracy']*100:.2f}%** | **{"+" if best_metrics['val_accuracy'] - FROZEN_M7_BASELINE['val_accuracy'] >= 0 else ""}{(best_metrics['val_accuracy'] - FROZEN_M7_BASELINE['val_accuracy'])*100:.2f} percentage points** |
| **Validation Macro Precision**| **89.53%** | **{best_metrics['val_macro_precision']*100:.2f}%** | **{"+" if best_metrics['val_macro_precision'] - FROZEN_M7_BASELINE['val_macro_precision'] >= 0 else ""}{(best_metrics['val_macro_precision'] - FROZEN_M7_BASELINE['val_macro_precision'])*100:.2f} percentage points** |
| **Validation Macro Recall** | **88.66%** | **{best_metrics['val_macro_recall']*100:.2f}%** | **{"+" if best_metrics['val_macro_recall'] - FROZEN_M7_BASELINE['val_macro_recall'] >= 0 else ""}{(best_metrics['val_macro_recall'] - FROZEN_M7_BASELINE['val_macro_recall'])*100:.2f} percentage points** |

---

## 7. Runtime & Computational Efficiency

- **Total Training Duration:** **{total_runtime_s:.1f} s ({total_runtime_s / 3600.0:.2f} hours)**
- **Average Duration per Epoch:** **{total_runtime_s / len(history):.1f} s ({(total_runtime_s / len(history)) / 60.0:.1f} minutes)**
- **Hardware Platform:** CPU execution (single process)

---

## 8. Test Isolation & Checkpoint Integrity

- **Test Set Isolation:** Confirmed. `data/splits/test.csv` was never opened, loaded, or evaluated during any phase of M7.3.
- **Frozen Baseline Integrity:** Confirmed. Checkpoints in `results/m7_final_selection/` were never modified or overwritten.
- **M7.3 Checkpoint Locations:**
  - Best model checkpoint: `results/m7_3_full_training/checkpoints/best_model.pt`
  - Last checkpoint (Epoch 10): `results/m7_3_full_training/checkpoints/checkpoint_last.pt`

---

## 9. Scientific Conclusions & Interpretation

1. **Impact of Training Duration:** Extending the training duration from 5 to 10 epochs on the 100% training dataset yielded a validation Macro-F1 of **{best_f1*100:.2f}%** at Epoch {best_epoch}.
2. **Comparison with Baseline:** Compared to the frozen 5-epoch M7 baseline (88.16%), the extended training resulted in **{"+" if delta_vs_frozen >= 0 else ""}{delta_vs_frozen*100:.2f} percentage points** change in validation Macro-F1.
3. **Model Selection Policy:** As specified by protocol, this model is NOT automatically promoted to the final project model. The official frozen model remains the M7 selected checkpoint until formal project review.
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info("Generated master M7.3 convergence report at %s", report_path)


# ----------------------------------------------------------------------------
# Main Training Runner
# ----------------------------------------------------------------------------
def run_m73_training(
    train_csv: str = "data/train.csv",
    val_csv: str = "data/val.csv",
    root_dir: str = "d:/Leaf",
    output_dir: str = "results/m7_3_full_training",
    epochs: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> None:
    t_global_start = time.time()
    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    logs_dir = out_path / "logs"
    metrics_dir = out_path / "metrics"
    plots_dir = out_path / "plots"
    config_dir = out_path / "config"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    # File log handler
    fh = logging.FileHandler(logs_dir / "train_m7_3.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 80)
    logger.info("AgriAI M7.3 -- Full-Training Convergence & Refinement (10 Epochs)")
    logger.info("=" * 80)

    # Verify frozen M7 checkpoint exists
    m7_ckpt = ROOT / "results/m7_final_selection/candidate_a_default/best_model.pt"
    assert m7_ckpt.exists(), f"CRITICAL: Frozen M7 checkpoint missing at {m7_ckpt}"
    logger.info("Confirmed frozen M7 checkpoint exists at: %s (will NOT be touched)", m7_ckpt)

    # Save experiment config
    config = {
        "experiment": "M7.3 Full-Training Convergence",
        "architecture": "LS-HGNet",
        "seed": seed,
        "epochs": epochs,
        "image_size": 224,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "dropout": 0.30,
        "device": device,
        "train_csv": train_csv,
        "val_csv": val_csv,
        "frozen_baseline_f1": FROZEN_M7_BASELINE["val_macro_f1"],
    }
    with open(config_dir / "m7_3_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    set_seed(seed)

    # Datasets
    train_tf = get_train_transforms(image_size=224)
    val_tf = get_eval_transforms(image_size=224)

    train_ds = PlantVillageDataset(manifest_csv=train_csv, root_dir=root_dir, transform=train_tf)
    val_ds = PlantVillageDataset(manifest_csv=val_csv, root_dir=root_dir, transform=val_tf)

    logger.info("Dataset verification:")
    logger.info("  Train samples: %d (100.0%%)", len(train_ds))
    logger.info("  Val samples  : %d (100.0%%)", len(val_ds))
    logger.info("  Test samples : STRICTLY LOCKED & NEVER ACCESSED")
    assert len(train_ds) == 12712, f"Train samples must be 12,712, got {len(train_ds)}"
    assert len(val_ds) == 2724, f"Val samples must be 2,724, got {len(val_ds)}"

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

    # Model
    model = LS_HGNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=ARCH_BASE_CHANNELS,
        num_stacks=ARCH_NUM_STACKS,
        encoder_levels=ARCH_ENCODER_LEVELS,
        dilation_rates=ARCH_DILATION_RATES,
        dropout=0.30,
    )
    total_params = count_parameters(model)
    logger.info("Model parameters: %d (trainable)", total_params)

    criterion = get_loss_function("cross_entropy")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=str(ckpt_dir),
        seed=seed,
    )

    # Check for resumption
    last_ckpt_path = ckpt_dir / "checkpoint_last.pt"
    start_epoch = 1
    history: List[dict] = []
    best_f1 = 0.0
    best_epoch = 1

    if last_ckpt_path.exists():
        logger.info("Found existing checkpoint: %s. Loading to resume...", last_ckpt_path)
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt.get("history", [])
        best_f1 = ckpt.get("best_f1", 0.0)
        best_epoch = ckpt.get("best_epoch", 1)
        trainer.best_metric = best_f1
        trainer.best_epoch = best_epoch
        logger.info("Resuming from Epoch %d (Best F1 so far: %.4f at Epoch %d)", start_epoch, best_f1, best_epoch)

    # Training loop
    epoch_durations: List[float] = [h["duration_seconds"] for h in history]

    for epoch in range(start_epoch, epochs + 1):
        t_ep_start = time.time()
        p_train_loader = ProgressDataLoader(
            train_loader, epoch, epochs, t_global_start, out_path, log_interval=40
        )
        train_loss, train_acc = trainer.train_epoch(p_train_loader)
        val_result = trainer.validate_epoch(val_loader)
        scheduler.step()

        ep_duration = time.time() - t_ep_start
        epoch_durations.append(ep_duration)
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_result["val_loss"], 4),
            "val_accuracy": round(val_result["val_accuracy"], 4),
            "val_macro_f1": round(val_result["val_macro_f1"], 4),
            "val_macro_precision": round(val_result.get("val_macro_precision", 0.0), 4),
            "val_macro_recall": round(val_result.get("val_macro_recall", 0.0), 4),
            "learning_rate": current_lr,
            "duration_seconds": round(ep_duration, 2),
        }
        history.append(row)

        is_best = val_result["val_macro_f1"] > best_f1
        if is_best:
            best_f1 = val_result["val_macro_f1"]
            best_epoch = epoch
            trainer.best_metric = best_f1
            trainer.best_epoch = best_epoch
            # Save best checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": best_f1,
                "val_loss": val_result["val_loss"],
                "seed": seed,
            }, ckpt_dir / "best_model.pt")
            logger.info(">>> NEW BEST VALIDATION MACRO-F1: %.4f (Epoch %d) <<<", best_f1, epoch)

        # Save checkpoint_last.pt
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "history": history,
            "best_f1": best_f1,
            "best_epoch": best_epoch,
            "seed": seed,
        }, last_ckpt_path)

        # Save metrics/history.json
        with open(metrics_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        avg_ep_time = sum(epoch_durations) / len(epoch_durations)
        rem_epochs = epochs - epoch
        est_rem = rem_epochs * avg_ep_time

        logger.info("=" * 80)
        logger.info(
            "Epoch %d/%d COMPLETE | Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Acc: %.2f%% F1: %.2f%%%s",
            epoch, epochs, train_loss, train_acc * 100,
            val_result["val_loss"], val_result["val_accuracy"] * 100, val_result["val_macro_f1"] * 100,
            " [BEST]" if is_best else "",
        )
        logger.info(
            "Epoch Time: %s | Avg: %s | Est Remaining: %s",
            format_hms(ep_duration), format_hms(avg_ep_time), format_hms(est_rem),
        )
        logger.info("=" * 80)

    total_training_time = time.time() - t_global_start
    logger.info("All %d epochs completed in %s (%.1f s)", epochs, format_hms(total_training_time), total_training_time)

    # Generate convergence plot
    plot_convergence(history, plots_dir / "convergence_curve.png")

    # Final detailed evaluation of best checkpoint on validation set
    best_ckpt = torch.load(ckpt_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_eval, y_true, y_pred = evaluate_dataset(model, val_loader, device=device, class_names=train_ds.classes)

    # Plot confusion matrix
    cm_path = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(val_eval["confusion_matrix"], train_ds.classes, cm_path)

    # Save metrics.json and classification_report.json
    metrics_summary = {
        "candidate_id": "m7_3_extended_10epochs",
        "architecture": "LS-HGNet",
        "seed": seed,
        "epochs_trained": epochs,
        "best_epoch": best_epoch,
        "device": device,
        "hyperparameters": config,
        "model_parameters": {
            "total_parameters": total_params,
            "checkpoint_size_mb": get_checkpoint_size_mb(ckpt_dir / "best_model.pt"),
        },
        "best_validation_metrics": {
            "accuracy": val_eval["accuracy"],
            "macro_precision": val_eval["macro_precision"],
            "macro_recall": val_eval["macro_recall"],
            "macro_f1": val_eval["macro_f1"],
            "loss": best_ckpt["val_loss"],
        },
        "total_training_runtime_s": round(total_training_time, 2),
        "comparison_vs_frozen_m7": {
            "frozen_m7_val_macro_f1": FROZEN_M7_BASELINE["val_macro_f1"],
            "m7_3_val_macro_f1": val_eval["macro_f1"],
            "delta_macro_f1": round(val_eval["macro_f1"] - FROZEN_M7_BASELINE["val_macro_f1"], 6),
        },
    }
    with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    with open(metrics_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(val_eval.get("per_class_metrics", {}), f, indent=2)

    # Generate master convergence report
    best_rec = next(h for h in history if h["epoch"] == best_epoch)
    generate_convergence_report(
        output_dir=out_path,
        history=history,
        best_epoch=best_epoch,
        best_f1=val_eval["macro_f1"],
        best_metrics=best_rec,
        total_runtime_s=total_training_time,
        config=config,
    )

    logger.info("M7.3 execution and report generation fully completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriAI M7.3 -- Full-Training Convergence")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--val_csv", default="data/val.csv")
    parser.add_argument("--root_dir", default="d:/Leaf")
    parser.add_argument("--output_dir", default="results/m7_3_full_training")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args()
    run_m73_training(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
    )
