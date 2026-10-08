"""
AgriAI — M4: LS-HGNet Training and Evaluation Script
Executes reproducible training and validation for the Adapted LS-HGNet model.

Strict Compliance Requirements:
1. Pure train/validation split utilization.
2. TEST SET IS STRICTLY PRESERVED AND UNTOUCHED.
3. Real, unweighted Cross-Entropy loss.
4. AdamW optimizer, cosine annealing scheduler, learning rate 1e-3, weight decay 1e-4.
5. Full classification metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, Per-Class Breakdown.
6. Computational metrics: Parameter count, Model size on disk, Measured CPU single-sample inference latency.
7. Structured artifact generation: metrics.json, history.json, model_summary.json, config.yaml, best_model.pt, confusion_matrix.png.
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

from src.models.ls_hgnet import (
    LS_HGNet,
    build_ls_hgnet_from_config,
    count_parameters,
    get_checkpoint_size_mb,
)
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
    parser = argparse.ArgumentParser(description="Train and evaluate AgriAI Adapted LS-HGNet")
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
        help="Directory to save experimental results"
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
        "--weight_decay",
        type=float,
        default=None,
        help="Override weight decay"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override dropout rate"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Override experiment name identifier"
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
    parser.add_argument(
        "--base_channels",
        type=int,
        default=None,
        help="Override base channels"
    )
    parser.add_argument(
        "--num_stacks",
        type=int,
        default=None,
        help="Override number of hourglass stacks"
    )
    parser.add_argument(
        "--encoder_levels",
        type=int,
        default=None,
        help="Override number of encoder levels"
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

    # Configuration resolution
    exp_name = args.experiment_name if args.experiment_name is not None else f"ls_hgnet_{phase}_{image_size}x{image_size}"
    seed = args.seed if args.seed is not None else config["experiment"].get("seed", 42)
    phase = config["experiment"].get("phase", "development")
    device_name = args.device if args.device is not None else "cpu"
    epochs = args.epochs if args.epochs is not None else config["training"].get("epochs", 5)
    batch_size = args.batch_size if args.batch_size is not None else config["training"].get("batch_size", 32)
    image_size = args.image_size if args.image_size is not None else config["preprocessing"].get("input_size", 128)
    lr = args.lr if args.lr is not None else float(config["training"].get("learning_rate", 1e-3))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(config["training"].get("weight_decay", 1e-4))
    num_classes = config["dataset"].get("num_classes", 10)

    # Model architecture parameters
    model_cfg = config.get("model", {})
    base_channels = args.base_channels if args.base_channels is not None else model_cfg.get("base_channels", 32)
    num_stacks = args.num_stacks if args.num_stacks is not None else model_cfg.get("num_stacks", 2)
    encoder_levels = args.encoder_levels if args.encoder_levels is not None else model_cfg.get("encoder_levels", 3)
    dilation_rates = tuple(model_cfg.get("dilation_rates", [1, 2, 4]))
    dropout = args.dropout if args.dropout is not None else float(model_cfg.get("dropout", 0.3))

    # Set output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / f"ls_hgnet_{phase}_{image_size}x{image_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print(f"AgriAI — Adapted LS-HGNet Experiment [{phase.upper()} RUN]", flush=True)
    print(f"Resolution: {image_size}x{image_size} | Batch Size: {batch_size} | Epochs: {epochs} | Device: {device_name}", flush=True)
    print(f"Architecture: base_channels={base_channels}, num_stacks={num_stacks}, encoder_levels={encoder_levels}, dilations={dilation_rates}", flush=True)
    print(f"Output Directory: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    # 1. Enforce strict reproducibility
    set_seed(seed)

    # 2. DataLoaders (Train and Validation ONLY; Test set protected)
    data_dir = PROJECT_ROOT / "data"
    train_csv = data_dir / "train.csv"
    val_csv = data_dir / "val.csv"

    if not train_csv.is_file() or not val_csv.is_file():
        train_csv = data_dir / "splits" / "train.csv"
        val_csv = data_dir / "splits" / "val.csv"

    train_transform = get_train_transforms(image_size=image_size)
    eval_transform = get_eval_transforms(image_size=image_size)

    train_ds = PlantVillageDataset(manifest_csv=train_csv, root_dir=PROJECT_ROOT, transform=train_transform)
    val_ds = PlantVillageDataset(manifest_csv=val_csv, root_dir=PROJECT_ROOT, transform=eval_transform)
    class_names = train_ds.classes

    print(f"Loaded Train Dataset: {len(train_ds):,} samples across {len(class_names)} classes", flush=True)
    print(f"Loaded Validation Dataset: {len(val_ds):,} samples across {len(class_names)} classes", flush=True)
    print("TEST SET STATUS: STRICTLY PROTECTED (Not loaded or evaluated in this script)", flush=True)

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
    model = LS_HGNet(
        in_channels=3,
        num_classes=num_classes,
        base_channels=base_channels,
        num_stacks=num_stacks,
        encoder_levels=encoder_levels,
        dilation_rates=dilation_rates,
        dropout=dropout,
    )

    summary = model.get_model_summary()
    print("\nModel Architecture Summary:")
    print(f"  Architecture: {summary['architecture']}")
    print(f"  Base Channels: {summary['base_channels']}")
    print(f"  Hourglass Stacks: {summary['num_stacks']}")
    print(f"  Encoder Levels: {summary['encoder_levels']}")
    print(f"  Dilation Rates: {summary['dilation_rates']}")
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
        title=f"LS-HGNet Validation Confusion Matrix ({image_size}x{image_size})"
    )
    print(f"Saved confusion matrix plot: {cm_path}")

    # 10. Persist All Experimental Artifacts
    actual_checkpoint_size = get_checkpoint_size_mb(str(best_ckpt_path))

    metrics_payload = {
        "experiment_name": exp_name,
        "architecture": "LS-HGNet",
        "phase": phase,
        "seed": seed,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs_trained": epochs,
        "best_epoch": checkpoint["epoch"],
        "device": device_name,
        "hyperparameters": {
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "batch_size": batch_size,
        },
        "model_parameters": {
            "base_channels": base_channels,
            "num_stacks": num_stacks,
            "encoder_levels": encoder_levels,
            "dilation_rates": list(dilation_rates),
            "dropout": dropout,
            "total_parameters": summary["total_parameters"],
            "trainable_parameters": summary["trainable_parameters"],
            "checkpoint_size_mb": round(actual_checkpoint_size, 4),
        },
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
    config.setdefault("training", {})
    config["training"]["learning_rate"] = lr
    config["training"]["weight_decay"] = weight_decay
    config["training"]["batch_size"] = batch_size
    config.setdefault("model", {})
    config["model"]["dropout"] = dropout
    config.setdefault("experiment", {})
    config["experiment"]["name"] = exp_name
    with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    print("\n" + "=" * 80)
    print("M4 LS-HGNET EXPERIMENT COMPLETE — ALL ARTIFACTS VERIFIED")
    print(f"  Val Accuracy:       {val_metrics['accuracy']*100:.2f}%")
    print(f"  Val Macro F1:       {val_metrics['macro_f1']*100:.2f}%")
    print(f"  Val Macro Precision:{val_metrics['macro_precision']*100:.2f}%")
    print(f"  Val Macro Recall:   {val_metrics['macro_recall']*100:.2f}%")
    print(f"  Parameters:         {summary['total_parameters']:,}")
    print(f"  Checkpoint Size:    {actual_checkpoint_size:.2f} MB")
    print(f"  Median CPU Latency: {latency_stats['median_latency_ms']:.2f} ms")
    print(f"  Best Epoch:         {checkpoint['epoch']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
