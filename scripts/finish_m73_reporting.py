"""
scripts/finish_m73_reporting.py
===============================
Finalizes post-training artifacts for M7.3:
- Loads completed 10-epoch best_model.pt and history.json
- Computes detailed validation metrics and classification report
- Generates confusion_matrix.png and convergence_curve.png
- Writes metrics.json and M7_3_CONVERGENCE_REPORT.md
"""

import sys
sys.path.insert(0, ".")
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_m73_full_training import (
    ROOT,
    FROZEN_M7_BASELINE,
    ARCH_BASE_CHANNELS,
    ARCH_NUM_STACKS,
    ARCH_ENCODER_LEVELS,
    ARCH_DILATION_RATES,
    NUM_CLASSES,
    plot_convergence,
    generate_convergence_report,
)
from src.data.dataset import PlantVillageDataset
from src.data.transforms import get_eval_transforms
from src.models.ls_hgnet import LS_HGNet, count_parameters, get_checkpoint_size_mb
from src.evaluation.metrics import evaluate_dataset, plot_confusion_matrix

out_path = ROOT / "results/m7_3_full_training"
ckpt_dir = out_path / "checkpoints"
metrics_dir = out_path / "metrics"
plots_dir = out_path / "plots"
config_dir = out_path / "config"

# 1. Load history
with open(metrics_dir / "history.json", encoding="utf-8") as f:
    history = json.load(f)

with open(config_dir / "m7_3_config.json", encoding="utf-8") as f:
    config = json.load(f)

# 2. Plot convergence curve
plot_convergence(history, plots_dir / "convergence_curve.png")

# 3. Load dataset
val_tf = get_eval_transforms(image_size=224)
val_ds = PlantVillageDataset(manifest_csv=ROOT / "data/val.csv", root_dir=ROOT, transform=val_tf)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=False)

# 4. Load model
model = LS_HGNet(
    in_channels=3,
    num_classes=NUM_CLASSES,
    base_channels=ARCH_BASE_CHANNELS,
    num_stacks=ARCH_NUM_STACKS,
    encoder_levels=ARCH_ENCODER_LEVELS,
    dilation_rates=ARCH_DILATION_RATES,
    dropout=0.30,
)
best_ckpt_path = ckpt_dir / "best_model.pt"
best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(best_ckpt["model_state_dict"])
best_epoch = best_ckpt["epoch"]
best_val_f1 = best_ckpt["val_macro_f1"]

total_params = count_parameters(model)

# 5. Evaluate
val_eval, y_true, y_pred = evaluate_dataset(model, val_loader, device="cpu", class_names=val_ds.classes)

# 6. Confusion matrix
cm_path = plots_dir / "confusion_matrix.png"
plot_confusion_matrix(val_eval["confusion_matrix"], val_ds.classes, cm_path)

total_runtime_s = sum(h["duration_seconds"] for h in history)

# 7. Metrics JSON
metrics_summary = {
    "candidate_id": "m7_3_extended_10epochs",
    "architecture": "LS-HGNet",
    "seed": 42,
    "epochs_trained": 10,
    "best_epoch": best_epoch,
    "device": "cpu",
    "hyperparameters": config,
    "model_parameters": {
        "total_parameters": total_params,
        "checkpoint_size_mb": get_checkpoint_size_mb(best_ckpt_path),
    },
    "best_validation_metrics": {
        "accuracy": val_eval["accuracy"],
        "macro_precision": val_eval["macro_precision"],
        "macro_recall": val_eval["macro_recall"],
        "macro_f1": val_eval["macro_f1"],
        "loss": best_ckpt["val_loss"],
    },
    "total_training_runtime_s": round(total_runtime_s, 2),
    "comparison_vs_frozen_m7": {
        "frozen_m7_val_macro_f1": FROZEN_M7_BASELINE["val_macro_f1"],
        "m7_3_val_macro_f1": val_eval["macro_f1"],
        "delta_macro_f1": round(val_eval["macro_f1"] - FROZEN_M7_BASELINE["val_macro_f1"], 6),
    },
}
with open(metrics_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_summary, f, indent=2)

with open(metrics_dir / "classification_report.json", "w", encoding="utf-8") as f:
    json.dump(val_eval.get("per_class_report", {}), f, indent=2)

# 8. Report
best_rec = next(h for h in history if h["epoch"] == best_epoch)
generate_convergence_report(
    output_dir=out_path,
    history=history,
    best_epoch=best_epoch,
    best_f1=val_eval["macro_f1"],
    best_metrics=best_rec,
    total_runtime_s=total_runtime_s,
    config=config,
)

print("=" * 70)
print("M7.3 REPORTING COMPLETED SUCCESSFULLY!")
print(f"Best Epoch: {best_epoch}")
print(f"Best Validation Macro-F1: {val_eval['macro_f1']*100:.2f}%")
print(f"Validation Accuracy: {val_eval['accuracy']*100:.2f}%")
print(f"Improvement over Frozen M7 (88.16%): +{(val_eval['macro_f1'] - FROZEN_M7_BASELINE['val_macro_f1'])*100:.2f} percentage points")
print("=" * 70)
