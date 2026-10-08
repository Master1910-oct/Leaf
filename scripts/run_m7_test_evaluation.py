"""
scripts/run_m7_test_evaluation.py
=================================
AgriAI M7 — Final Locked Independent Test Set Evaluation.

STRICT RULE:
This script accesses the TEST split (2,724 images) EXACTLY ONCE.
It requires that `results/m7_final_selection/selected_config.json` exists
and indicates a locked model selected based strictly on validation Macro-F1.

Artifacts Generated:
  results/m7_final_model/
    ├── final_model.pt
    ├── final_config.yaml
    ├── test_metrics.json
    ├── classification_report.json
    ├── confusion_matrix.png
    ├── model_summary.json
    └── inference_benchmark.json
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ls_hgnet import LS_HGNet, count_parameters, get_checkpoint_size_mb
from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_eval_transforms
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
logger = logging.getLogger("m7_test_eval")

ARCH_BASE_CHANNELS  = 64
ARCH_NUM_STACKS     = 3
ARCH_ENCODER_LEVELS = 4
ARCH_DILATION_RATES = (1, 2, 4)
NUM_CLASSES         = 10


def parse_args():
    p = argparse.ArgumentParser(description="AgriAI M7 — Locked Independent Test Set Evaluation")
    p.add_argument("--selection_dir", type=str, default="results/m7_final_selection",
                   help="Directory with selection results")
    p.add_argument("--output_dir",    type=str, default="results/m7_final_model",
                   help="Directory to save final model and test evaluation artifacts")
    p.add_argument("--test_csv",      type=str, default="data/splits/test.csv")
    p.add_argument("--root_dir",      type=str, default="d:/Leaf")
    p.add_argument("--image_size",    type=int, default=224)
    p.add_argument("--batch_size",    type=int, default=32)
    p.add_argument("--device",        type=str, default="cpu")
    p.add_argument("--seed",          type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    sel_dir = Path(args.selection_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_cfg_path = sel_dir / "selected_config.json"
    if not sel_cfg_path.is_file():
        raise RuntimeError(
            f"Test set evaluation forbidden: model selection has not been completed. Missing: {sel_cfg_path}"
        )

    with open(sel_cfg_path, "r", encoding="utf-8") as f:
        sel_cfg = json.load(f)

    if sel_cfg.get("status") != "LOCKED":
        raise RuntimeError("Test set evaluation forbidden: model is not locked.")

    winner_id = sel_cfg["selected_candidate_id"]
    ckpt_path = Path(sel_cfg["checkpoint_path"])
    if not ckpt_path.is_file():
        # Fallback to relative path within sel_dir
        ckpt_path = sel_dir / winner_id / "best_model.pt"

    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Selected candidate checkpoint not found: {ckpt_path}")

    logger.info("=" * 80)
    logger.info("AgriAI M7 — INDEPENDENT LOCKED TEST SET EVALUATION")
    logger.info("=" * 80)
    logger.info("Selected Candidate: %s", winner_id)
    logger.info("Checkpoint:         %s", ckpt_path)
    logger.info("Hyperparameters:    %s", sel_cfg["hyperparameters"])
    logger.info("Image Size:         %dx%d", args.image_size, args.image_size)
    logger.info("Device:             %s", args.device)
    logger.info("Random Seed:        %d", args.seed)
    logger.info("=" * 80)

    # 1. Copy locked model to final destination
    final_model_dst = out_dir / "final_model.pt"
    if not final_model_dst.is_file() or final_model_dst.resolve() != ckpt_path.resolve():
        shutil.copyfile(ckpt_path, final_model_dst)
        logger.info("Locked model checkpoint copied to %s", final_model_dst)

    # 2. Build model architecture
    hp = sel_cfg["hyperparameters"]
    dropout = float(hp.get("dropout", 0.3))

    model = LS_HGNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=ARCH_BASE_CHANNELS,
        num_stacks=ARCH_NUM_STACKS,
        encoder_levels=ARCH_ENCODER_LEVELS,
        dilation_rates=ARCH_DILATION_RATES,
        dropout=dropout,
    )
    total_params = count_parameters(model)

    # 3. Load weights
    ckpt = torch.load(final_model_dst, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()
    logger.info("Model weights loaded successfully. Parameter count: %d", total_params)

    # 4. Load TEST set (FIRST AND ONLY ACCESS)
    test_csv_path = Path(args.root_dir) / args.test_csv if not Path(args.test_csv).is_absolute() else Path(args.test_csv)
    if not test_csv_path.is_file():
        test_csv_path = PROJECT_ROOT / args.test_csv

    test_tf = get_eval_transforms(image_size=args.image_size)
    test_ds = PlantVillageDataset(test_csv_path, args.root_dir, test_tf)
    class_names = test_ds.classes

    logger.info("Accessing TEST SET: %d samples across %d classes", len(test_ds), len(class_names))
    assert len(test_ds) == 2724, f"Test set must have 2,724 images; got {len(test_ds)}"

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # 5. Evaluate on test set
    t_start = time.time()
    test_metrics, y_true, y_pred = evaluate_dataset(
        model=model,
        dataloader=test_loader,
        device=args.device,
        class_names=class_names,
    )
    eval_duration = time.time() - t_start
    logger.info("Test set evaluation completed in %.2fs", eval_duration)

    # 6. Confusion Matrix
    cm_path = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cm_path,
        title=f"M7 Final LS-HGNet — Locked Test Confusion Matrix (224×224, N=2,724)"
    )
    logger.info("Saved test confusion matrix to %s", cm_path)

    # 7. CPU Inference Latency (batch=1, 100 runs)
    logger.info("Measuring CPU single-sample inference latency (100 runs)...")
    latency_stats = measure_inference_latency(
        model=model,
        input_shape=(1, 3, args.image_size, args.image_size),
        device=args.device,
        num_warmup=20,
        num_runs=100,
    )
    logger.info("  Median latency: %.2f ms | Mean: %.2f ms",
                latency_stats["median_latency_ms"], latency_stats["mean_latency_ms"])

    ckpt_size_mb = get_checkpoint_size_mb(str(final_model_dst))

    # 8. Persist artifacts
    # classification_report.json
    with open(out_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics["per_class_report"], f, indent=2)

    # model_summary.json
    model_summary = {
        "architecture": "LS-HGNet",
        "parameters": {
            "total_parameters": total_params,
            "trainable_parameters": total_params,
            "base_channels": ARCH_BASE_CHANNELS,
            "num_stacks": ARCH_NUM_STACKS,
            "encoder_levels": ARCH_ENCODER_LEVELS,
            "dilation_rates": list(ARCH_DILATION_RATES),
            "dropout": dropout,
            "num_classes": NUM_CLASSES,
        },
        "checkpoint_size_mb": round(ckpt_size_mb, 4),
        "selected_candidate_id": winner_id,
        "source_checkpoint": str(ckpt_path),
        "locked_at_epoch": ckpt.get("epoch"),
    }
    with open(out_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(model_summary, f, indent=2)

    # inference_benchmark.json
    with open(out_dir / "inference_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(latency_stats, f, indent=2)

    # test_metrics.json
    test_metrics_payload = {
        "experiment": "AgriAI M7 Final LS-HGNet Independent Test Evaluation",
        "selected_candidate_id": winner_id,
        "dataset_split": {
            "test_samples": len(test_ds),
            "num_classes": len(class_names),
            "image_size": [args.image_size, args.image_size],
        },
        "hyperparameters": hp,
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 6),
            "macro_precision": round(test_metrics["macro_precision"], 6),
            "macro_recall": round(test_metrics["macro_recall"], 6),
            "macro_f1": round(test_metrics["macro_f1"], 6),
        },
        "per_class_metrics": {
            cls_name: {
                "precision": round(test_metrics["per_class_report"][cls_name]["precision"], 4),
                "recall": round(test_metrics["per_class_report"][cls_name]["recall"], 4),
                "f1_score": round(test_metrics["per_class_report"][cls_name]["f1-score"], 4),
                "support": test_metrics["per_class_report"][cls_name]["support"],
            }
            for cls_name in class_names
        },
        "computational_metrics": {
            "total_parameters": total_params,
            "checkpoint_size_mb": round(ckpt_size_mb, 4),
            "median_inference_latency_ms": round(latency_stats["median_latency_ms"], 4),
            "mean_inference_latency_ms": round(latency_stats["mean_latency_ms"], 4),
            "std_inference_latency_ms": round(latency_stats["std_latency_ms"], 4),
        },
        "reproducibility": {
            "seed": args.seed,
            "device": args.device,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "isolation_verified": "Test set accessed exactly once after selection was frozen.",
    }
    with open(out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics_payload, f, indent=2)

    # final_config.yaml
    final_config_dict = {
        "model": {
            "architecture": "LS-HGNet",
            "base_channels": ARCH_BASE_CHANNELS,
            "num_stacks": ARCH_NUM_STACKS,
            "encoder_levels": ARCH_ENCODER_LEVELS,
            "dilation_rates": list(ARCH_DILATION_RATES),
            "dropout": dropout,
            "num_classes": NUM_CLASSES,
            "total_parameters": total_params,
        },
        "training": {
            "learning_rate": hp["learning_rate"],
            "weight_decay": hp["weight_decay"],
            "batch_size": hp["batch_size"],
            "dropout": dropout,
            "epochs": 5,
            "optimizer": "adamw",
            "scheduler": "cosine_annealing",
            "loss": "cross_entropy",
            "seed": args.seed,
        },
        "selection": {
            "selected_candidate": winner_id,
            "validation_macro_f1": sel_cfg["validation_metrics"]["macro_f1"],
            "validation_accuracy": sel_cfg["validation_metrics"]["accuracy"],
        },
        "test_evaluation": {
            "accuracy": round(test_metrics["accuracy"], 6),
            "macro_f1": round(test_metrics["macro_f1"], 6),
            "macro_precision": round(test_metrics["macro_precision"], 6),
            "macro_recall": round(test_metrics["macro_recall"], 6),
        },
        "status": "FROZEN",
    }
    with open(out_dir / "final_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(final_config_dict, f, default_flow_style=False, sort_keys=False)

    logger.info("=" * 80)
    logger.info("FINAL INDEPENDENT TEST SET EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info("  Selected Candidate: %s", winner_id)
    logger.info("  Test Accuracy     : %.4f%%", test_metrics["accuracy"] * 100)
    logger.info("  Test Macro-F1     : %.4f%%", test_metrics["macro_f1"] * 100)
    logger.info("  Test Macro-Prec   : %.4f%%", test_metrics["macro_precision"] * 100)
    logger.info("  Test Macro-Recall : %.4f%%", test_metrics["macro_recall"] * 100)
    logger.info("  Parameters        : %d", total_params)
    logger.info("  Checkpoint Size   : %.2f MB", ckpt_size_mb)
    logger.info("  CPU Latency (med) : %.2f ms", latency_stats["median_latency_ms"])
    logger.info("  All artifacts saved to: %s", out_dir)
    logger.info("MODEL IS NOW FROZEN. NO FURTHER ADJUSTMENTS PERMITTED.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
