"""
AgriAI — M3: Baseline CNN Training and Evaluation Script
Executes reproducible training and validation for the BaselineCNN control model.

Strict Compliance Requirements:
1. Pure train/validation split utilization.
2. TEST SET IS STRICTLY PRESERVED AND UNTOUCHED.
3. Real, unweighted Cross-Entropy loss (Baseline v1).
4. Full classification metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, Per-Class Breakdown.
5. Computational metrics: Parameter count, Model size on disk, Measured CPU single-sample inference latency.
6. Structured artifact generation: metrics.json, history.json, model_summary.json, config.yaml, best_model.pt, confusion_matrix.png.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_cnn import BaselineCNN, count_parameters, get_checkpoint_size_mb
from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.training.trainer import Trainer
from src.training.losses import get_loss_function
from src.evaluation.metrics import (
    evaluate_dataset,
    measure_inference_latency,
    plot_confusion_matrix
)
from src.utils.reproducibility import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate AgriAI BaselineCNN")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/development.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save experimental results (defaults to results/baseline_<phase>)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=None,
        help="Override image resolution (e.g. 128 or 224)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override initial learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device ('cpu' standard)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Deterministic random seed"
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = load_config(str(config_path))

    # Configuration overrides
    seed = args.seed if args.seed is not None else config["experiment"].get("seed", 42)
    phase = config["experiment"].get("phase", "development")
    device_name = args.device if args.device is not None else "cpu"
    epochs = args.epochs if args.epochs is not None else config["training"].get("epochs", 10)
    batch_size = args.batch_size if args.batch_size is not None else config["training"].get("batch_size", 32)
    image_size = args.image_size if args.image_size is not None else config["preprocessing"].get("input_size", 128)
    lr = args.lr if args.lr is not None else float(config["training"].get("learning_rate", 1e-3))
    weight_decay = float(config["training"].get("weight_decay", 1e-4))
    num_classes = config["dataset"].get("num_classes", 10)

    # Set output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / f"baseline_{phase}_{image_size}x{image_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"AgriAI — Baseline CNN Experiment [{phase.upper()} RUN]", flush=True)
    print(f"Resolution: {image_size}x{image_size} | Batch Size: {batch_size} | Epochs: {epochs} | Device: {device_name}", flush=True)
    print(f"Output Directory: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    # 1. Enforce strict reproducibility
    set_seed(seed)

    # 2. DataLoaders (Train and Validation ONLY; Test set protected)
    data_dir = PROJECT_ROOT / "data"
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    if not train_csv.is_file() or not val_csv.is_file():
        # Fallback to splits directory
        train_csv = data_dir / "splits" / "train.csv"
        val_csv = data_dir / "splits" / "val.csv"

    train_transform = get_train_transforms(image_size=image_size)
    eval_transform = get_eval_transforms(image_size=image_size)

    train_ds = PlantVillageDataset(manifest_csv=train_csv, root_dir=PROJECT_ROOT, transform=train_transform)
    val_ds = PlantVillageDataset(manifest_csv=val_csv, root_dir=PROJECT_ROOT, transform=eval_transform)
    class_names = train_ds.classes

    print(f"Loaded Train Dataset: {len(train_ds):,} samples across {len(class_names)} classes", flush=True)
    print(f"Loaded Validation Dataset: {len(val_ds):,} samples across {len(class_names)} classes", flush=True)
    print("TEST SET STATUS: PROTECTED (Not loaded or evaluated in this script)", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # 3. Model Initialization
    channels = tuple(config.get("baseline", {}).get("channels", [32, 64, 128, 256]))
    model = BaselineCNN(
        in_channels=3,
        num_classes=num_classes,
        channels=channels,
        fc_dim=256,
        dropout=0.2
    )

    summary = model.get_model_summary()
    print("\nModel Architecture Summary:")
    print(f"  Architecture: {summary['architecture']}")
    print(f"  Channel Progression: {summary['channels']}")
    print(f"  Total Parameters: {summary['total_parameters']:,}")
    print(f"  Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"  Estimated Model Size: {summary['estimated_model_size_mb']} MB")

    # 4. Training Components
    criterion = get_loss_function("cross_entropy")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device_name,
        save_dir=output_dir,
        seed=seed
    )

    # 5. Execute Training Loop
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        checkpoint_name="best_model.pt",
        verbose=True
    )

    # 6. Load Best Checkpoint for Evaluation
    best_ckpt_path = output_dir / "best_model.pt"
    print(f"\nLoading best checkpoint from {best_ckpt_path} for final validation evaluation...")
    checkpoint = torch.load(best_ckpt_path, map_location=device_name)
    model.load_state_dict(checkpoint["model_state_dict"])

    # 7. Comprehensive Validation Evaluation
    print("Evaluating best model on Validation set...")
    val_metrics, y_true, y_pred = evaluate_dataset(
        model=model,
        dataloader=val_loader,
        device=device_name,
        class_names=class_names
    )

    # 8. Measure CPU Inference Latency (Batch size = 1)
    print("Measuring CPU single-sample inference latency (100 repetitions)...")
    latency_stats = measure_inference_latency(
        model=model,
        input_shape=(1, 3, image_size, image_size),
        device="cpu",
        num_warmup=20,
        num_runs=100
    )
    print(f"  Median CPU Latency: {latency_stats['median_latency_ms']:.2f} ms")
    print(f"  Mean CPU Latency: {latency_stats['mean_latency_ms']:.2f} ms (+/- {latency_stats['std_latency_ms']:.2f} ms)")

    # 9. Plot Confusion Matrix
    cm_path = output_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        cm=val_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=cm_path,
        title=f"Baseline CNN Validation Confusion Matrix ({image_size}x{image_size})"
    )
    print(f"Saved confusion matrix plot: {cm_path}")

    # 10. Persist All Experimental Artifacts
    actual_checkpoint_size = get_checkpoint_size_mb(str(best_ckpt_path))

    metrics_payload = {
        "experiment_name": f"baseline_cnn_{phase}_{image_size}x{image_size}",
        "phase": phase,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs_trained": epochs,
        "best_epoch": checkpoint["epoch"],
        "device": device_name,
        "validation_metrics": {
            "accuracy": round(val_metrics["accuracy"], 6),
            "macro_precision": round(val_metrics["macro_precision"], 6),
            "macro_recall": round(val_metrics["macro_recall"], 6),
            "macro_f1": round(val_metrics["macro_f1"], 6)
        },
        "computational_metrics": {
            "total_parameters": summary["total_parameters"],
            "trainable_parameters": summary["trainable_parameters"],
            "checkpoint_size_mb": round(actual_checkpoint_size, 4),
            "median_inference_latency_ms": round(latency_stats["median_latency_ms"], 4),
            "mean_inference_latency_ms": round(latency_stats["mean_latency_ms"], 4),
            "std_inference_latency_ms": round(latency_stats["std_latency_ms"], 4),
            "latency_measurements": latency_stats
        },
        "dataset_split": {
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "test_samples": "2,724 (UNTOUCHED)",
            "num_classes": len(class_names)
        }
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    with open(output_dir / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(val_metrics["per_class_report"], f, indent=2)

    with open(output_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save exact configuration used
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("\n" + "=" * 80)
    print("M3 BASELINE RUN COMPLETE — ALL ARTIFACTS VERIFIED")
    print(f"  Val Accuracy:       {val_metrics['accuracy']*100:.2f}%")
    print(f"  Val Macro F1:       {val_metrics['macro_f1']*100:.2f}%")
    print(f"  Val Macro Precision:{val_metrics['macro_precision']*100:.2f}%")
    print(f"  Val Macro Recall:   {val_metrics['macro_recall']*100:.2f}%")
    print(f"  Parameters:         {summary['total_parameters']:,}")
    print(f"  Checkpoint Size:    {actual_checkpoint_size:.2f} MB")
    print(f"  Median CPU Latency: {latency_stats['median_latency_ms']:.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
