"""
scripts/train_m7_candidate.py
================================
AgriAI M7 — Candidate Training for Final Model Selection.

Trains a single named candidate configuration on the FULL training split
(12,712 images) at 224×224 for 5 epochs, evaluates on the validation split
(2,724 images), and saves all artifacts to results/m7_final_selection/<candidate_id>/.

STRICT RULES:
- Test set is NEVER loaded in this script.
- Model selection is performed via validation metrics ONLY.
- The test split CSV is read ONLY by scripts/run_m7_test_evaluation.py AFTER lock.

Usage:
    python -u scripts/train_m7_candidate.py \
        --candidate_id candidate_b_sto \
        --lr 4.977535e-3 \
        --weight_decay 1.8342e-6 \
        --dropout 0.50 \
        --batch_size 32 \
        --epochs 5 \
        --image_size 224 \
        --seed 42

Supported candidate_ids:
    candidate_a_default   (LR=1e-3, WD=1e-4, dropout=0.3, BS=32) — reuses M4
    candidate_b_sto       (LR=4.98e-3, WD=1.83e-6, dropout=0.5, BS=32) — STO-best
    candidate_c_random    (LR=4.75e-3, WD=5.25e-6, dropout=0.431, BS=16) — RS-best
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import psutil
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ls_hgnet import LS_HGNet, count_parameters, get_checkpoint_size_mb
from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.training.trainer import Trainer
from src.training.losses import get_loss_function
from src.evaluation.metrics import (
    evaluate_dataset,
    measure_inference_latency,
    plot_confusion_matrix,
)
from src.utils.reproducibility import set_seed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("m7_candidate")

# ──────────────────────────────────────────────────────────────────────────────
# Architecture constants — M4 values, NEVER changed in M7
# ──────────────────────────────────────────────────────────────────────────────
ARCH_BASE_CHANNELS  = 64
ARCH_NUM_STACKS     = 3
ARCH_ENCODER_LEVELS = 4
ARCH_DILATION_RATES = (1, 2, 4)
NUM_CLASSES         = 10


def format_hms(seconds: float) -> str:
    s = int(round(max(0.0, seconds)))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


class ProgressDataLoader:
    def __init__(self, dataloader, epoch, total_epochs, total_training_start, candidate_id, output_base, log_interval=20):
        self.dataloader = dataloader
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.total_training_start = total_training_start
        self.candidate_id = candidate_id
        self.output_base = Path(output_base)
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
                epoch_remaining = (total_batches - batch_idx) * avg_batch_time

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
                    "candidate_id": self.candidate_id,
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
                    "device": "CPU"
                }

                live_file = self.output_base / "live_status.json"
                try:
                    with open(live_file, "w", encoding="utf-8") as f:
                        json.dump(status_obj, f, indent=2)
                except Exception:
                    pass

                logger.info(
                    "[%s] Ep %d/%d (%.1f%%) | Batch %d/%d | Elapsed: %s | Rem: %s | ETA: %s | CPU: %.1f%%",
                    self.candidate_id, self.epoch, self.total_epochs, epoch_progress_pct,
                    batch_idx, total_batches, format_hms(total_elapsed), format_hms(total_remaining),
                    eta.strftime("%I:%M %p"), cpu_pct
                )


def parse_args():
    p = argparse.ArgumentParser(description="AgriAI M7 — Candidate Training")
    p.add_argument("--candidate_id",   type=str, required=True,
                   choices=["candidate_a_default", "candidate_b_sto", "candidate_c_random"],
                   help="Identifier for the candidate configuration")
    p.add_argument("--lr",             type=float, required=True,  help="Learning rate")
    p.add_argument("--weight_decay",   type=float, required=True,  help="Weight decay (L2)")
    p.add_argument("--dropout",        type=float, required=True,  help="Dropout probability [0, 0.5]")
    p.add_argument("--batch_size",     type=int,   required=True,  help="Batch size {16, 32, 64}")
    p.add_argument("--epochs",         type=int,   default=5,      help="Training epochs (default: 5)")
    p.add_argument("--image_size",     type=int,   default=224,    help="Image size (default: 224)")
    p.add_argument("--seed",           type=int,   default=42,     help="Random seed (default: 42)")
    p.add_argument("--device",         type=str,   default="cpu",  help="Device (default: cpu)")
    p.add_argument("--output_base",    type=str,
                   default="results/m7_final_selection",
                   help="Base directory for M7 results")
    p.add_argument("--train_csv",      type=str,   default="data/splits/train.csv")
    p.add_argument("--val_csv",        type=str,   default="data/splits/val.csv")
    p.add_argument("--root_dir",       type=str,   default="d:/Leaf")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_base) / args.candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("AgriAI M7 — Candidate Training: %s", args.candidate_id)
    logger.info("=" * 80)
    logger.info("Hyperparameters:")
    logger.info("  Learning Rate:   %.6e", args.lr)
    logger.info("  Weight Decay:    %.6e", args.weight_decay)
    logger.info("  Dropout:         %.4f", args.dropout)
    logger.info("  Batch Size:      %d", args.batch_size)
    logger.info("  Epochs:          %d", args.epochs)
    logger.info("  Image Size:      %dx%d", args.image_size, args.image_size)
    logger.info("  Seed:            %d", args.seed)
    logger.info("Architecture (FIXED M4):")
    logger.info("  base_channels=%d  num_stacks=%d  encoder_levels=%d  dilation=%s",
                ARCH_BASE_CHANNELS, ARCH_NUM_STACKS, ARCH_ENCODER_LEVELS, ARCH_DILATION_RATES)
    logger.info("  num_classes=%d", NUM_CLASSES)
    logger.info("=" * 80)

    # ── Data (TEST set never touched) ──
    train_tf = get_train_transforms(image_size=args.image_size)
    val_tf   = get_eval_transforms(image_size=args.image_size)

    train_csv_path = Path(args.root_dir) / args.train_csv if not Path(args.train_csv).is_absolute() else Path(args.train_csv)
    val_csv_path   = Path(args.root_dir) / args.val_csv   if not Path(args.val_csv).is_absolute() else Path(args.val_csv)
    # Fallback
    if not train_csv_path.is_file():
        train_csv_path = PROJECT_ROOT / args.train_csv
    if not val_csv_path.is_file():
        val_csv_path = PROJECT_ROOT / args.val_csv

    train_ds = PlantVillageDataset(train_csv_path, args.root_dir, train_tf)
    val_ds   = PlantVillageDataset(val_csv_path,   args.root_dir, val_tf)
    class_names = train_ds.classes

    logger.info("Train: %d images  |  Val: %d images  |  Test: STRICTLY ISOLATED",
                len(train_ds), len(val_ds))
    assert len(train_ds) == 12712, f"Train must have 12,712 images; got {len(train_ds)}"
    assert len(val_ds)   == 2724,  f"Val must have 2,724 images; got {len(val_ds)}"

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=False)

    # ── Model (fixed architecture) ──
    model = LS_HGNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=ARCH_BASE_CHANNELS,
        num_stacks=ARCH_NUM_STACKS,
        encoder_levels=ARCH_ENCODER_LEVELS,
        dilation_rates=ARCH_DILATION_RATES,
        dropout=args.dropout,
    )
    total_params = count_parameters(model)
    logger.info("Model parameters: %d (trainable)", total_params)

    # ── Optimizer, loss, scheduler ──
    criterion = get_loss_function("cross_entropy")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training ──
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        save_dir=str(out_dir),
        seed=args.seed,
    )

    t_total_start = time.time()
    epoch_durations = []
    history = []

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        p_train_loader = ProgressDataLoader(
            train_loader, epoch, args.epochs, t_total_start, args.candidate_id, args.output_base, log_interval=20
        )
        train_loss, train_acc = trainer.train_epoch(p_train_loader)
        val_result = trainer.validate_epoch(val_loader)
        scheduler.step()

        ep_time = time.time() - t_ep
        epoch_durations.append(ep_time)
        current_lr = optimizer.param_groups[0]["lr"]

        avg_ep_time = sum(epoch_durations) / len(epoch_durations)
        rem_epochs = args.epochs - epoch
        est_rem = rem_epochs * avg_ep_time
        eta = datetime.now() + timedelta(seconds=est_rem)

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
            "duration_seconds": round(ep_time, 2),
        }
        history.append(row)
        logger.info(
            "================================================================================"
        )
        logger.info(
            "Epoch %d/%d COMPLETE | Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f F1: %.4f | LR: %.6e",
            epoch, args.epochs, train_loss, train_acc,
            val_result["val_loss"], val_result["val_accuracy"], val_result["val_macro_f1"],
            current_lr
        )
        logger.info(
            "Epoch Time: %s | Avg Epoch: %s | Est Remaining: %s | ETA: %s",
            format_hms(ep_time), format_hms(avg_ep_time), format_hms(est_rem), eta.strftime("%I:%M:%S %p")
        )
        logger.info(
            "================================================================================"
        )

        # Track best
        if val_result["val_macro_f1"] > trainer.best_metric:
            trainer.best_metric = val_result["val_macro_f1"]
            trainer.best_epoch = epoch
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_result["val_loss"],
                "val_macro_f1": val_result["val_macro_f1"],
                "hyperparameters": {
                    "learning_rate": args.lr,
                    "weight_decay": args.weight_decay,
                    "dropout": args.dropout,
                    "batch_size": args.batch_size,
                },
                "architecture": {
                    "base_channels": ARCH_BASE_CHANNELS,
                    "num_stacks": ARCH_NUM_STACKS,
                    "encoder_levels": ARCH_ENCODER_LEVELS,
                    "dilation_rates": list(ARCH_DILATION_RATES),
                    "num_classes": NUM_CLASSES,
                },
            }
            torch.save(checkpoint, out_dir / "best_model.pt")
            logger.info("  [BEST] New best at epoch %d: val_macro_f1=%.4f", epoch, trainer.best_metric)

    total_runtime = time.time() - t_total_start
    logger.info("Training complete. Total runtime: %.1fs (%.2f h)", total_runtime, total_runtime / 3600)

    # ── Final evaluation on best checkpoint ──
    best_ckpt = torch.load(out_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    val_metrics, y_true, y_pred = evaluate_dataset(
        model=model,
        dataloader=val_loader,
        device=args.device,
        class_names=class_names
    )

    # Confusion matrix
    plot_confusion_matrix(
        cm=val_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=out_dir / "confusion_matrix.png",
        title=f"M7 {args.candidate_id} — Validation Confusion Matrix (224×224, Epoch {best_ckpt['epoch']})"
    )

    # Inference latency
    latency = measure_inference_latency(
        model,
        input_shape=(1, 3, args.image_size, args.image_size),
        device=args.device,
        num_warmup=10,
        num_runs=50,
    )

    ckpt_size_mb = get_checkpoint_size_mb(str(out_dir / "best_model.pt"))

    # ── Artifacts ──
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics["per_class_report"], f, indent=2)

    metrics_payload = {
        "candidate_id": args.candidate_id,
        "architecture": "LS-HGNet (fixed M4)",
        "seed": args.seed,
        "image_size": args.image_size,
        "epochs_trained": args.epochs,
        "best_epoch": best_ckpt["epoch"],
        "device": args.device,
        "hyperparameters": {
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
        },
        "model_parameters": {
            "base_channels": ARCH_BASE_CHANNELS,
            "num_stacks": ARCH_NUM_STACKS,
            "encoder_levels": ARCH_ENCODER_LEVELS,
            "dilation_rates": list(ARCH_DILATION_RATES),
            "total_parameters": total_params,
            "checkpoint_size_mb": round(ckpt_size_mb, 4),
        },
        "validation_metrics": {
            "accuracy": round(val_metrics["accuracy"], 6),
            "macro_precision": round(val_metrics["macro_precision"], 6),
            "macro_recall": round(val_metrics["macro_recall"], 6),
            "macro_f1": round(val_metrics["macro_f1"], 6),
        },
        "computational_metrics": {
            "total_parameters": total_params,
            "checkpoint_size_mb": round(ckpt_size_mb, 4),
            "median_inference_latency_ms": round(latency["median_latency_ms"], 4),
            "mean_inference_latency_ms": round(latency["mean_latency_ms"], 4),
        },
        "dataset_split": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": "2,724 (STRICTLY ISOLATED — not accessed in this script)",
        },
        "total_training_runtime_s": round(total_runtime, 2),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    logger.info("=" * 80)
    logger.info("M7 CANDIDATE TRAINING COMPLETE: %s", args.candidate_id)
    logger.info("  Val Accuracy  : %.4f%%", val_metrics["accuracy"] * 100)
    logger.info("  Val Macro-F1  : %.4f%%", val_metrics["macro_f1"] * 100)
    logger.info("  Val Precision : %.4f%%", val_metrics["macro_precision"] * 100)
    logger.info("  Val Recall    : %.4f%%", val_metrics["macro_recall"] * 100)
    logger.info("  Parameters    : %d", total_params)
    logger.info("  Checkpoint    : %.2f MB", ckpt_size_mb)
    logger.info("  Latency (med) : %.2f ms", latency["median_latency_ms"])
    logger.info("  Best Epoch    : %d", best_ckpt["epoch"])
    logger.info("  Output dir    : %s", out_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
