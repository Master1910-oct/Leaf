"""
M8.2 -- Calibration, Robustness and Deployment Readiness
=========================================================
Uses the frozen M7.3 Epoch-9 checkpoint (best_model.pt).

Sections:
  A. Calibration   -- Temperature Scaling on validation split
  B. Robustness    -- Controlled image perturbations on val holdout
  C. Deployment    -- CPU inference benchmark
  D. Model Integrity -- SHA-256 verification
  E. Data Isolation Audit
  F. Reproducibility config
  G. Final report

RULES:
  - DO NOT open/load/evaluate on test data.
  - DO NOT retrain or modify best_model.pt.
  - All outputs go to results/m8_2_calibration_robustness/
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import hashlib
import json
import time
import platform
import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from PIL import Image, ImageFilter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path("D:/Leaf")
CHECKPOINT_PATH = PROJECT_ROOT / "results/m7_3_full_training/checkpoints/best_model.pt"
VAL_CSV         = PROJECT_ROOT / "data/val.csv"
OUT_DIR         = PROJECT_ROOT / "results/m8_2_calibration_robustness"

EXPECTED_SHA256 = "4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8"

NUM_CLASSES    = 10
IMAGE_SIZE     = 224
BATCH_SIZE     = 32
SEED           = 42
CALIB_FRACTION = 0.50

METRICS_DIR   = OUT_DIR / "metrics"
PLOTS_DIR     = OUT_DIR / "plots"
CONFIG_DIR    = OUT_DIR / "config"
MANIFESTS_DIR = OUT_DIR / "manifests"

for d in [METRICS_DIR, PLOTS_DIR, CONFIG_DIR, MANIFESTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# M8.2-D  Model Integrity Check
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-D  MODEL INTEGRITY CHECK")
log("=" * 60)

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

log(f"Computing SHA-256 for: {CHECKPOINT_PATH}")
actual_sha256 = sha256_file(CHECKPOINT_PATH)
log(f"  Expected : {EXPECTED_SHA256}")
log(f"  Actual   : {actual_sha256}")

if actual_sha256 != EXPECTED_SHA256:
    raise RuntimeError(
        f"INTEGRITY FAILURE -- SHA-256 mismatch!\n"
        f"  Expected : {EXPECTED_SHA256}\n"
        f"  Actual   : {actual_sha256}\n"
        "STOPPING. Do not replace or repair the checkpoint."
    )
log("[PASS] Checkpoint integrity verified.")

# ---------------------------------------------------------------------------
# Load Model (frozen)
# ---------------------------------------------------------------------------

log("=" * 60)
log("Loading LS-HGNet from checkpoint ...")
log("=" * 60)

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.ls_hgnet import LS_HGNet


# M7.3 exact architecture (verified from run_m73_full_training.py lines 93-96)
ARCH_BASE_CHANNELS  = 64
ARCH_NUM_STACKS     = 3
ARCH_ENCODER_LEVELS = 4
ARCH_DILATION_RATES = (1, 2, 4)
ARCH_DROPOUT        = 0.30


def load_model(checkpoint_path: Path, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LS_HGNet(
        in_channels=3,
        num_classes=NUM_CLASSES,
        base_channels=ARCH_BASE_CHANNELS,
        num_stacks=ARCH_NUM_STACKS,
        encoder_levels=ARCH_ENCODER_LEVELS,
        dilation_rates=ARCH_DILATION_RATES,
        dropout=ARCH_DROPOUT,
    )
    raw_sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(raw_sd, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


DEVICE = "cpu"
model, ckpt_data = load_model(CHECKPOINT_PATH, DEVICE)
model_summary     = model.get_model_summary()
total_params      = model_summary["total_parameters"]
checkpoint_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
log(f"  Architecture   : LS-HGNet")
log(f"  Total params   : {total_params:,}")
log(f"  Checkpoint MB  : {checkpoint_size_mb:.2f}")

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

eval_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class ValSubsetDataset(torch.utils.data.Dataset):
    def __init__(self, df, root_dir, transform=None, perturbation=None):
        self.df          = df.reset_index(drop=True)
        self.root        = root_dir
        self.transform   = transform
        self.perturbation = perturbation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        img   = Image.open(self.root / row["filepath"]).convert("RGB")
        if self.perturbation is not None:
            img = self.perturbation(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["class_idx"])


# ---------------------------------------------------------------------------
# M8.2-A  Calibration / Holdout Split
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-A  CALIBRATION / HOLDOUT SPLIT")
log("=" * 60)

val_df = pd.read_csv(VAL_CSV)
log(f"  Val CSV rows : {len(val_df)}")

labels   = val_df["class_idx"].values
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)
calib_idx, holdout_idx = next(splitter.split(np.zeros(len(labels)), labels))

calib_df   = val_df.iloc[calib_idx].reset_index(drop=True)
holdout_df = val_df.iloc[holdout_idx].reset_index(drop=True)

calib_files   = set(calib_df["filepath"].tolist())
holdout_files = set(holdout_df["filepath"].tolist())
assert len(calib_files & holdout_files) == 0, "SAFETY: duplicate samples!"
assert len(calib_files) + len(holdout_files) == len(val_df), "SAFETY: count mismatch!"

log(f"  Calibration subset : {len(calib_df)}")
log(f"  Holdout subset     : {len(holdout_df)}")
log(f"  Overlap            : 0 (verified)")

calib_class_counts   = calib_df["class_idx"].value_counts().sort_index().to_dict()
holdout_class_counts = holdout_df["class_idx"].value_counts().sort_index().to_dict()

calib_df.to_csv(MANIFESTS_DIR / "calibration_subset.csv", index=False)
holdout_df.to_csv(MANIFESTS_DIR / "holdout_subset.csv", index=False)

split_info = {
    "seed": SEED,
    "split_strategy": "StratifiedShuffleSplit",
    "calib_fraction": CALIB_FRACTION,
    "total_val_samples": len(val_df),
    "calib_samples": len(calib_df),
    "holdout_samples": len(holdout_df),
    "calib_class_counts": {int(k): int(v) for k, v in calib_class_counts.items()},
    "holdout_class_counts": {int(k): int(v) for k, v in holdout_class_counts.items()},
    "overlap_count": 0,
    "source_csv": "data/val.csv",
    "test_csv_accessed": False,
}
with open(METRICS_DIR / "split_info.json", "w") as f:
    json.dump(split_info, f, indent=2)
log("  Split info saved.")

# ---------------------------------------------------------------------------
# Logit extraction
# ---------------------------------------------------------------------------

def extract_logits(model, df, root, transform, cache_name=None):
    if cache_name is not None:
        cache_path = METRICS_DIR / f"{cache_name}.pt"
        if cache_path.exists():
            log(f"  Loading cached logits from {cache_path.name} ...")
            data = torch.load(cache_path, map_location="cpu", weights_only=False)
            return data["logits"], data["labels"]

    ds     = ValSubsetDataset(df, root, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            all_logits.append(model(imgs.to(DEVICE)).cpu())
            all_labels.append(lbls)
    cat_logits = torch.cat(all_logits)
    cat_labels = torch.cat(all_labels)

    if cache_name is not None:
        torch.save({"logits": cat_logits, "labels": cat_labels}, cache_path)
        log(f"  Saved cached logits to {cache_path.name}")

    return cat_logits, cat_labels


# ---------------------------------------------------------------------------
# Calibration metrics helper
# ---------------------------------------------------------------------------

def calibration_metrics(logits, labels, temperature=1.0, n_bins=15):
    scaled = logits / temperature
    probs  = F.softmax(scaled, dim=1)
    preds  = probs.argmax(dim=1)

    acc    = (preds == labels).float().mean().item()
    nll    = F.cross_entropy(scaled, labels).item()
    oh     = F.one_hot(labels, num_classes=NUM_CLASSES).float()
    brier  = ((probs - oh) ** 2).sum(dim=1).mean().item()
    conf   = probs.max(dim=1).values.mean().item()

    confs  = probs.max(dim=1).values.numpy()
    pred_n = preds.numpy()
    true_n = labels.numpy()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (confs > lo) & (confs <= hi)
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append(float((lo + hi) / 2))
            bin_counts.append(0)
            continue
        ba = (pred_n[mask] == true_n[mask]).mean()
        bc = confs[mask].mean()
        ece += (mask.sum() / len(labels)) * abs(ba - bc)
        bin_accs.append(float(ba))
        bin_confs.append(float(bc))
        bin_counts.append(int(mask.sum()))

    metrics = {
        "accuracy": round(acc * 100, 4),
        "nll": round(nll, 6),
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "mean_confidence": round(conf * 100, 4),
        "temperature": temperature,
    }
    bins = {
        "bin_accs": bin_accs,
        "bin_confs": bin_confs,
        "bin_counts": bin_counts,
        "bin_edges": list(bin_edges),
    }
    return metrics, bins


# ---------------------------------------------------------------------------
# Extract logits
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-A  TEMPERATURE SCALING CALIBRATION")
log("=" * 60)
log("  Extracting logits from calibration subset ...")
calib_logits, calib_labels = extract_logits(model, calib_df, PROJECT_ROOT, eval_transform, cache_name="calib_logits")
log("  Extracting logits from calibration holdout ...")
holdout_logits, holdout_labels = extract_logits(model, holdout_df, PROJECT_ROOT, eval_transform, cache_name="holdout_logits")

# Before calibration
metrics_before, bins_before = calibration_metrics(holdout_logits, holdout_labels, temperature=1.0)
log(f"  BEFORE -- NLL={metrics_before['nll']:.4f}  Brier={metrics_before['brier']:.4f}  "
    f"ECE={metrics_before['ece']:.4f}  MeanConf={metrics_before['mean_confidence']:.2f}%  "
    f"Acc={metrics_before['accuracy']:.2f}%")

# Fit temperature on calibration subset
log("  Fitting temperature on calibration subset (LBFGS) ...")
temperature_param = nn.Parameter(torch.ones(1) * 1.5)
optimizer = torch.optim.LBFGS([temperature_param], lr=0.01, max_iter=100)


def closure():
    optimizer.zero_grad()
    loss = F.cross_entropy(calib_logits / temperature_param, calib_labels)
    loss.backward()
    return loss


optimizer.step(closure)
T_learned = float(temperature_param.item())
log(f"  Learned temperature T = {T_learned:.6f}")

# After calibration
metrics_after, bins_after = calibration_metrics(holdout_logits, holdout_labels, temperature=T_learned)
log(f"  AFTER  -- NLL={metrics_after['nll']:.4f}  Brier={metrics_after['brier']:.4f}  "
    f"ECE={metrics_after['ece']:.4f}  MeanConf={metrics_after['mean_confidence']:.2f}%  "
    f"Acc={metrics_after['accuracy']:.2f}%")

# Save
calib_out = {
    "method": "Temperature Scaling",
    "learned_temperature": T_learned,
    "calibration_subset_size": len(calib_df),
    "holdout_subset_size": len(holdout_df),
    "seed": SEED,
    "before": metrics_before,
    "after": metrics_after,
    "note": "Temperature fitted on calibration subset; evaluated on holdout. Model weights unchanged.",
}
with open(METRICS_DIR / "calibration_metrics.json", "w") as f:
    json.dump(calib_out, f, indent=2)

temp_out = {
    "learned_temperature": T_learned,
    "optimizer": "LBFGS",
    "lr": 0.01,
    "max_iter": 100,
    "initial_temperature": 1.5,
    "fitted_on": "calibration_subset",
    "evaluated_on": "calibration_holdout",
    "model_weights_modified": False,
}
with open(METRICS_DIR / "temperature_scaling.json", "w") as f:
    json.dump(temp_out, f, indent=2)
log("  Calibration metrics saved.")

# ---------------------------------------------------------------------------
# Reliability diagrams
# ---------------------------------------------------------------------------

def plot_reliability(bins_data, title, out_path, ece):
    fig, ax = plt.subplots(figsize=(6, 6))
    n_bins    = len(bins_data["bin_accs"])
    bin_edges = bins_data["bin_edges"]
    bin_w     = 1.0 / n_bins
    for i in range(n_bins):
        if bins_data["bin_counts"][i] == 0:
            continue
        lo  = bin_edges[i]
        acc = bins_data["bin_accs"][i]
        cof = bins_data["bin_confs"][i]
        ax.bar(lo, acc, width=bin_w, align="edge",
               color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.bar(lo, cof - acc, width=bin_w, align="edge", bottom=acc,
               color="#DD8452", alpha=0.5, edgecolor="white", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect calibration")
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    p1 = mpatches.Patch(color="#4C72B0", alpha=0.8, label="Accuracy")
    p2 = mpatches.Patch(color="#DD8452", alpha=0.5, label="Gap")
    ax.legend(handles=[p1, p2, ax.lines[0]], fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"  Saved: {out_path.name}")


plot_reliability(bins_before, "Reliability Diagram (Before Temperature Scaling)",
                 PLOTS_DIR / "reliability_before.png", metrics_before["ece"])
plot_reliability(bins_after,  "Reliability Diagram (After Temperature Scaling)",
                 PLOTS_DIR / "reliability_after.png",  metrics_after["ece"])

# ---------------------------------------------------------------------------
# M8.2-B  Robustness Evaluation
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-B  ROBUSTNESS EVALUATION")
log("=" * 60)

robustness_df = holdout_df.copy()
log(f"  Robustness subset : {len(robustness_df)} samples (calibration holdout, val only)")

import io as _io


def jpeg_fn(quality=60):
    def fn(img):
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()
    return fn


PERT_NOISE_STD = 0.05
_rng = torch.Generator()
_rng.manual_seed(SEED)


def add_noise(t):
    return (t + torch.randn(t.shape, generator=_rng, dtype=t.dtype, device=t.device) * PERT_NOISE_STD).clamp(0, 1)


noise_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    add_noise,
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

PERTURBATIONS = {
    "1_clean": {
        "description": "No perturbation (clean images)",
        "pil_fn": None,
        "tensor_transform": eval_transform,
        "params": {},
    },
    "2_brightness": {
        "description": "Mild brightness increase (factor=1.3)",
        "pil_fn": lambda img: T.functional.adjust_brightness(img, brightness_factor=1.3),
        "tensor_transform": eval_transform,
        "params": {"brightness_factor": 1.3},
    },
    "3_contrast": {
        "description": "Mild contrast reduction (factor=0.75)",
        "pil_fn": lambda img: T.functional.adjust_contrast(img, contrast_factor=0.75),
        "tensor_transform": eval_transform,
        "params": {"contrast_factor": 0.75},
    },
    "4_gaussian_noise": {
        "description": "Mild Gaussian noise (std=0.05 in [0,1] tensor space)",
        "pil_fn": None,
        "tensor_transform": noise_transform,
        "params": {"noise_std": PERT_NOISE_STD, "seed": SEED},
    },
    "5_gaussian_blur": {
        "description": "Mild Gaussian blur (radius=1.5)",
        "pil_fn": lambda img: img.filter(ImageFilter.GaussianBlur(radius=1.5)),
        "tensor_transform": eval_transform,
        "params": {"radius": 1.5},
    },
    "6_jpeg_compression": {
        "description": "Mild JPEG compression (quality=60)",
        "pil_fn": jpeg_fn(60),
        "tensor_transform": eval_transform,
        "params": {"quality": 60},
    },
}


def eval_perturbation(df, pil_fn, tensor_transform, cache_key=None):
    if cache_key is not None:
        cache_file = METRICS_DIR / f"perturbation_{cache_key}.json"
        if cache_file.exists():
            log(f"    Loaded cached metrics from {cache_file.name}")
            with open(cache_file, "r") as f:
                return json.load(f)

    ds     = ValSubsetDataset(df, PROJECT_ROOT, transform=tensor_transform, perturbation=pil_fn)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, lbls in loader:
            all_preds.append(model(imgs.to(DEVICE)).argmax(dim=1).cpu())
            all_labels.append(lbls)
    p = torch.cat(all_preds).numpy()
    l = torch.cat(all_labels).numpy()
    res = {
        "accuracy":        round(accuracy_score(l, p) * 100, 4),
        "macro_precision": round(precision_score(l, p, average="macro", zero_division=0) * 100, 4),
        "macro_recall":    round(recall_score(l, p, average="macro", zero_division=0) * 100, 4),
        "macro_f1":        round(f1_score(l, p, average="macro", zero_division=0) * 100, 4),
    }
    if cache_key is not None:
        cache_file = METRICS_DIR / f"perturbation_{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(res, f, indent=2)
    return res


robustness_results = {}
clean_f1 = None

for key, pert in PERTURBATIONS.items():
    log(f"  Evaluating: {key}")
    if key == "1_clean":
        p = holdout_logits.argmax(dim=1).numpy()
        l = holdout_labels.numpy()
        res = {
            "accuracy":        round(accuracy_score(l, p) * 100, 4),
            "macro_precision": round(precision_score(l, p, average="macro", zero_division=0) * 100, 4),
            "macro_recall":    round(recall_score(l, p, average="macro", zero_division=0) * 100, 4),
            "macro_f1":        round(f1_score(l, p, average="macro", zero_division=0) * 100, 4),
        }
    else:
        res = eval_perturbation(robustness_df, pert["pil_fn"], pert["tensor_transform"], cache_key=key)

    robustness_results[key] = {
        "description": pert["description"],
        "params": pert["params"],
        "metrics": res,
        "macro_f1_drop_vs_clean": 0.0,
    }
    if key == "1_clean":
        clean_f1 = res["macro_f1"]
    log(f"    Accuracy={res['accuracy']:.2f}%  Macro-F1={res['macro_f1']:.2f}%")

for key in robustness_results:
    if key != "1_clean":
        robustness_results[key]["macro_f1_drop_vs_clean"] = round(
            clean_f1 - robustness_results[key]["metrics"]["macro_f1"], 4)

robustness_out = {
    "subset": "calibration_holdout",
    "n_samples": len(robustness_df),
    "clean_macro_f1": clean_f1,
    "perturbation_seed": SEED,
    "image_size": IMAGE_SIZE,
    "perturbations": robustness_results,
}
with open(METRICS_DIR / "robustness_metrics.json", "w") as f:
    json.dump(robustness_out, f, indent=2)
log("  Robustness metrics saved.")


# Plots
def wrap_label(s, n=18):
    words = s.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= n:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur: lines.append(cur)
    return "\n".join(lines)


labels_plot = [wrap_label(v["description"].split("(")[0].strip()) for v in robustness_results.values()]
f1_vals     = [v["metrics"]["macro_f1"] for v in robustness_results.values()]
acc_vals    = [v["metrics"]["accuracy"]  for v in robustness_results.values()]
drops       = [v["macro_f1_drop_vs_clean"] for v in robustness_results.values()]
colors_f1   = ["#2196F3" if d == 0 else ("#4CAF50" if d < 1 else ("#FF9800" if d < 3 else "#F44336"))
               for d in drops]

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(labels_plot, f1_vals, color=colors_f1, edgecolor="white", linewidth=0.8)
ax.axhline(clean_f1, color="navy", linestyle="--", linewidth=1.2, label=f"Clean F1 = {clean_f1:.2f}%")
for bar, v in zip(bars, f1_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(max(0, min(f1_vals) - 5), 102)
ax.set_ylabel("Macro-F1 (%)", fontsize=12)
ax.set_title("Robustness -- Macro-F1 per Perturbation", fontsize=13)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "robustness_macro_f1.png", dpi=150, bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(labels_plot, acc_vals, color="#5C6BC0", edgecolor="white", linewidth=0.8, alpha=0.85)
clean_acc = robustness_results["1_clean"]["metrics"]["accuracy"]
ax.axhline(clean_acc, color="darkred", linestyle="--", linewidth=1.2, label=f"Clean Acc = {clean_acc:.2f}%")
for bar, v in zip(bars, acc_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
            f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylim(max(0, min(acc_vals) - 5), 102)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Robustness -- Accuracy per Perturbation", fontsize=13)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "robustness_accuracy.png", dpi=150, bbox_inches="tight")
plt.close()
log("  Robustness plots saved.")

# ---------------------------------------------------------------------------
# M8.2-C  Deployment Benchmark
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-C  DEPLOYMENT BENCHMARK")
log("=" * 60)

set_seed(SEED)
sample_row    = holdout_df.iloc[0]
sample_img    = Image.open(PROJECT_ROOT / sample_row["filepath"]).convert("RGB")
sample_tensor = eval_transform(sample_img).unsqueeze(0)

WARMUP_RUNS  = 10
MEASURE_RUNS = 200

model.eval()
log(f"  Warming up ({WARMUP_RUNS} runs) ...")
with torch.no_grad():
    for _ in range(WARMUP_RUNS):
        _ = model(sample_tensor)

log(f"  Measuring ({MEASURE_RUNS} runs) ...")
latencies_ms = []
with torch.no_grad():
    for _ in range(MEASURE_RUNS):
        t0 = time.perf_counter()
        _  = model(sample_tensor)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

arr        = np.array(latencies_ms)
mean_lat   = float(np.mean(arr))
median_lat = float(np.median(arr))
std_lat    = float(np.std(arr))
min_lat    = float(np.min(arr))
max_lat    = float(np.max(arr))
p95_lat    = float(np.percentile(arr, 95))
p99_lat    = float(np.percentile(arr, 99))
throughput = 1000.0 / mean_lat

log(f"  Mean={mean_lat:.2f}ms  Median={median_lat:.2f}ms  Std={std_lat:.2f}ms  "
    f"Min={min_lat:.2f}ms  Max={max_lat:.2f}ms  P95={p95_lat:.2f}ms  P99={p99_lat:.2f}ms")
log(f"  Throughput: {throughput:.2f} img/sec (batch=1)")

try:
    import psutil
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    log(f"  Process RSS: {mem_mb:.1f} MB")
except ImportError:
    mem_mb = None
    log("  psutil not available -- memory not measured")

cpu_info  = platform.processor()
cpu_count = os.cpu_count()

benchmark_out = {
    "checkpoint": str(CHECKPOINT_PATH),
    "checkpoint_size_mb": round(checkpoint_size_mb, 4),
    "total_parameters": total_params,
    "image_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
    "batch_size": 1,
    "device": "CPU",
    "warmup_runs": WARMUP_RUNS,
    "measured_runs": MEASURE_RUNS,
    "latency_ms": {
        "mean":   round(mean_lat,   3),
        "median": round(median_lat, 3),
        "std":    round(std_lat,    3),
        "min":    round(min_lat,    3),
        "max":    round(max_lat,    3),
        "p95":    round(p95_lat,    3),
        "p99":    round(p99_lat,    3),
    },
    "throughput_images_per_sec": round(throughput, 3),
    "process_memory_mb": round(mem_mb, 1) if mem_mb else "unavailable",
    "python_version":      sys.version,
    "pytorch_version":     torch.__version__,
    "torchvision_version": torchvision.__version__,
    "cpu_device": cpu_info,
    "cpu_count":  cpu_count,
    "platform":   platform.platform(),
    "seed":       SEED,
    "timestamp":  datetime.datetime.utcnow().isoformat() + "Z",
}
with open(METRICS_DIR / "deployment_benchmark.json", "w") as f:
    json.dump(benchmark_out, f, indent=2)
log("  Benchmark saved.")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(arr, bins=40, color="#1976D2", edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(mean_lat,   color="red",    linestyle="--", linewidth=1.5, label=f"Mean={mean_lat:.1f}ms")
ax.axvline(median_lat, color="orange", linestyle="-",  linewidth=1.5, label=f"Median={median_lat:.1f}ms")
ax.axvline(p95_lat,    color="green",  linestyle=":",  linewidth=1.5, label=f"P95={p95_lat:.1f}ms")
ax.axvline(p99_lat,    color="purple", linestyle=":",  linewidth=1.5, label=f"P99={p99_lat:.1f}ms")
ax.set_xlabel("Latency (ms)", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.set_title(f"CPU Inference Latency Distribution ({MEASURE_RUNS} runs, batch=1)", fontsize=12)
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "inference_latency_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log("  Latency plot saved.")

# ---------------------------------------------------------------------------
# M8.2-E  Data Isolation Audit
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-E  DATA ISOLATION AUDIT")
log("=" * 60)


def audit_subset(df, name, val_df):
    val_files = set(val_df["filepath"].tolist())
    sub_files = set(df["filepath"].tolist())
    return {
        "subset_name": name,
        "n_samples": len(df),
        "all_samples_from_val": sub_files.issubset(val_files),
        "test_samples_detected": any("test" in str(fp).lower() for fp in sub_files),
    }


audit_results = [
    audit_subset(calib_df,   "calibration_subset",  val_df),
    audit_subset(holdout_df, "calibration_holdout",  val_df),
    audit_subset(holdout_df, "robustness_subset",    val_df),
    {
        "subset_name": "deployment_benchmark",
        "n_samples": 1,
        "all_samples_from_val": True,
        "test_samples_detected": False,
        "note": "Uses first row of holdout_df (val sample only)",
    },
]
isolation_out = {
    "test_csv_opened": False,
    "test_images_loaded": False,
    "test_evaluated": False,
    "val_csv": "data/val.csv",
    "subsets": audit_results,
    "isolation_confirmed": all(
        a["all_samples_from_val"] and not a["test_samples_detected"]
        for a in audit_results
    ),
}
with open(MANIFESTS_DIR / "data_isolation_audit.json", "w") as f:
    json.dump(isolation_out, f, indent=2)
log(f"  Isolation confirmed: {isolation_out['isolation_confirmed']}")

# ---------------------------------------------------------------------------
# M8.2-F  Reproducibility Config
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-F  REPRODUCIBILITY CONFIG")
log("=" * 60)

repro_config = {
    "experiment": "M8.2 -- Calibration, Robustness and Deployment Readiness",
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "seed": SEED,
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_sha256": actual_sha256,
    "checkpoint_size_mb": round(checkpoint_size_mb, 4),
    "epoch": 9,
    "val_split": {
        "csv": "data/val.csv",
        "n_total": len(val_df),
        "n_calib": len(calib_df),
        "n_holdout": len(holdout_df),
        "strategy": "StratifiedShuffleSplit",
        "calib_fraction": CALIB_FRACTION,
        "seed": SEED,
    },
    "temperature_scaling": {
        "learned_temperature": T_learned,
        "optimizer": "LBFGS",
        "lr": 0.01,
        "max_iter": 100,
        "initial_temperature": 1.5,
    },
    "robustness": {
        "subset": "calibration_holdout",
        "n_samples": len(robustness_df),
        "perturbations": {k: v["params"] for k, v in robustness_results.items()},
        "noise_seed": SEED,
    },
    "deployment_benchmark": {
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURE_RUNS,
        "batch_size": 1,
        "device": "CPU",
        "image_size": IMAGE_SIZE,
    },
    "software": {
        "python_version":      sys.version,
        "pytorch_version":     torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cpu_device": cpu_info,
        "cpu_count":  cpu_count,
        "platform":   platform.platform(),
    },
    "test_set_accessed": False,
}
with open(CONFIG_DIR / "m8_2_config.json", "w") as f:
    json.dump(repro_config, f, indent=2)
log("  Reproducibility config saved.")

# ---------------------------------------------------------------------------
# M8.2-G  Final Report
# ---------------------------------------------------------------------------

log("=" * 60)
log("M8.2-G  GENERATING FINAL REPORT")
log("=" * 60)

max_drop = max(
    v["macro_f1_drop_vs_clean"]
    for k, v in robustness_results.items() if k != "1_clean"
)

lines = []
lines.append("# M8.2 -- Calibration, Robustness & Deployment Readiness Report")
lines.append(f"\n**Date**: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
lines.append(f"\n**Project**: D:/Leaf  |  **Experiment**: M8.2")
lines.append("\n---\n")

lines.append("## 1. Executive Summary\n")
lines.append(
    "This report documents **M8.2 -- Calibration, Robustness & Deployment Readiness**, "
    "a post-training analysis of the frozen M7.3 LS-HGNet model (Epoch 9). "
    "The checkpoint achieved **94.23% validation Macro-F1** and **95.52% validation accuracy** "
    "during M7.3. **The locked test set was not accessed during M8.2.**\n\n"
    f"Temperature scaling was applied (T = {T_learned:.4f}), reducing ECE from "
    f"{metrics_before['ece']:.4f} to {metrics_after['ece']:.4f}. "
    f"Maximum Macro-F1 degradation under mild perturbations: {max_drop:.2f}%. "
    f"Mean CPU latency (batch=1): {mean_lat:.1f} ms (P95 = {p95_lat:.1f} ms).\n"
)

lines.append("## 2. Input Model\n")
lines.append("| Field | Value |")
lines.append("|---|---|")
lines.append("| Architecture | LS-HGNet |")
lines.append("| Experiment | M7.3 Full-Training Convergence |")
lines.append("| Best Epoch | 9 |")
lines.append("| Checkpoint | `results/m7_3_full_training/checkpoints/best_model.pt` |")
lines.append(f"| SHA-256 | `{actual_sha256}` |")
lines.append(f"| Parameter Count | {total_params:,} |")
lines.append(f"| Checkpoint Size | {checkpoint_size_mb:.2f} MB |")
lines.append("| Val Accuracy (M7.3) | 95.52% |")
lines.append("| Val Macro-F1 (M7.3) | 94.23% |")
lines.append("| M7 Frozen Baseline Macro-F1 | 88.16% |")
lines.append("")

lines.append("## 3. Calibration Method\n")
lines.append(
    f"The validation set (2,724 samples) was split into a **calibration subset ({len(calib_df)} samples, 50%)** "
    f"and a **calibration holdout ({len(holdout_df)} samples, 50%)** using `StratifiedShuffleSplit` "
    f"(seed={SEED}). No test data was used.\n\n"
    "**Temperature scaling** (Guo et al., 2017): a single scalar temperature T was fitted on the "
    "calibration subset by minimising NLL via L-BFGS (100 iterations, lr=0.01). "
    "**Model weights were NOT modified.**\n"
)
lines.append(f"**Learned temperature**: T = {T_learned:.6f}\n")

lines.append("## 4. Reliability Analysis\n")
lines.append("### Before / After Temperature Scaling (Holdout Set)\n")
lines.append("| Metric | Before | After |")
lines.append("|---|---|---|")
lines.append(f"| NLL | {metrics_before['nll']:.6f} | {metrics_after['nll']:.6f} |")
lines.append(f"| Brier Score | {metrics_before['brier']:.6f} | {metrics_after['brier']:.6f} |")
lines.append(f"| ECE | {metrics_before['ece']:.6f} | {metrics_after['ece']:.6f} |")
lines.append(f"| Mean Confidence | {metrics_before['mean_confidence']:.2f}% | {metrics_after['mean_confidence']:.2f}% |")
lines.append(f"| Accuracy | {metrics_before['accuracy']:.2f}% | {metrics_after['accuracy']:.2f}% |")
lines.append("")
lines.append("**Plots**: `plots/reliability_before.png`, `plots/reliability_after.png`\n")

lines.append("## 5. Robustness Evaluation\n")
lines.append(
    f"Evaluated on {len(robustness_df)} holdout samples (validation only, no test data). "
    "All six conditions applied to the **same samples** for paired comparison.\n"
)
lines.append("| Condition | Description | Accuracy | Macro-F1 | F1 Drop |")
lines.append("|---|---|---|---|---|")
for key, val in robustness_results.items():
    m = val["metrics"]
    lines.append(
        f"| {key} | {val['description']} | {m['accuracy']:.2f}% | "
        f"{m['macro_f1']:.2f}% | {val['macro_f1_drop_vs_clean']:.2f}% |"
    )
lines.append("")
lines.append(f"**Maximum Macro-F1 degradation**: {max_drop:.2f}%\n")
lines.append("**Plots**: `plots/robustness_macro_f1.png`, `plots/robustness_accuracy.png`\n")

lines.append("## 6. Deployment Benchmark\n")
lines.append("| Field | Value |")
lines.append("|---|---|")
lines.append(f"| Checkpoint Size | {checkpoint_size_mb:.2f} MB |")
lines.append(f"| Total Parameters | {total_params:,} |")
lines.append("| Device | CPU |")
lines.append(f"| Image Size | {IMAGE_SIZE}x{IMAGE_SIZE} |")
lines.append("| Batch Size | 1 |")
lines.append(f"| Warmup Runs | {WARMUP_RUNS} |")
lines.append(f"| Measured Runs | {MEASURE_RUNS} |")
lines.append(f"| Mean Latency | {mean_lat:.2f} ms |")
lines.append(f"| Median Latency | {median_lat:.2f} ms |")
lines.append(f"| Std Dev | {std_lat:.2f} ms |")
lines.append(f"| Min | {min_lat:.2f} ms |")
lines.append(f"| Max | {max_lat:.2f} ms |")
lines.append(f"| P95 | {p95_lat:.2f} ms |")
lines.append(f"| P99 | {p99_lat:.2f} ms |")
lines.append(f"| Throughput | {throughput:.2f} images/sec |")
if mem_mb:
    lines.append(f"| Process RSS | {mem_mb:.1f} MB |")
lines.append(f"| PyTorch | {torch.__version__} |")
lines.append(f"| Python | {sys.version.split()[0]} |")
lines.append(f"| CPU | {cpu_info} |")
lines.append("")
lines.append("**Plot**: `plots/inference_latency_distribution.png`\n")

lines.append("## 7. Limitations\n")
lines.append(
    "- **CPU-only benchmark**: GPU latency not claimed or measured.\n"
    "- **Single-image benchmark**: batch_size=1; batch throughput will be higher.\n"
    "- **Calibration scope**: Temperature scaling fitted on 50% val. Calibration quality on test set unknown.\n"
    "- **Synthetic perturbations**: Mild and controlled; real deployment may exhibit different degradation patterns.\n"
    "- **Memory measurement**: Process RSS includes Python interpreter overhead.\n"
)

lines.append("## 8. Test Set Status\n")
lines.append("> **The locked test set was not accessed during M8.2.**\n")
lines.append(
    "- `data/test.csv` was NOT opened.\n"
    "- No test images were loaded.\n"
    "- No test predictions were generated.\n"
    "- No test metrics were computed.\n"
    "- No parameters were tuned using test data.\n"
)

lines.append("## 9. Scientific Interpretation\n")
lines.append(
    f"The M7.3 checkpoint achieves **94.23% validation Macro-F1** -- a +6.07 point improvement "
    "over the M7 frozen baseline of 88.16%. This is strictly **validation performance**; "
    "test set generalisation remains unknown until final evaluation.\n\n"
    f"Temperature scaling adjusts probability confidence (ECE: {metrics_before['ece']:.4f} -> "
    f"{metrics_after['ece']:.4f}) but does not change classification decisions -- "
    f"accuracy before ({metrics_before['accuracy']:.2f}%) and after ({metrics_after['accuracy']:.2f}%) "
    "scaling is identical. Temperature scaling is a probability-calibration method only.\n\n"
    f"Robustness analysis shows a maximum Macro-F1 drop of {max_drop:.2f}% under mild perturbations, "
    f"indicating {'reasonable stability' if max_drop < 5 else 'notable sensitivity to image quality changes'}.\n\n"
    f"CPU latency of {mean_lat:.1f} ms/image (P95={p95_lat:.1f} ms) is practical for offline "
    "or low-throughput workflows. High-throughput production deployment would require GPU or "
    "batched inference optimisation.\n"
)

lines.append("## 10. Readiness for Final Evaluation\n")
lines.append(
    "M8.2 has produced all required artefacts:\n\n"
    "- [x] SHA-256 verified checkpoint (`PASS`)\n"
    "- [x] Calibration split manifests\n"
    "- [x] Temperature scaling metrics (before/after)\n"
    "- [x] Reliability diagrams\n"
    "- [x] Robustness metrics (6 conditions)\n"
    "- [x] Robustness plots\n"
    "- [x] CPU deployment benchmark\n"
    "- [x] Latency distribution plot\n"
    "- [x] Data isolation audit\n"
    "- [x] Reproducibility config\n\n"
    "**The project is ready for the final locked-test evaluation (M9) when the team decides to proceed.**\n\n"
    "> IMPORTANT: The final test evaluation must be performed exactly once, deliberately, "
    "on the frozen M7.3 Epoch-9 checkpoint (`best_model.pt`). "
    "It was NOT performed in M8.2.\n"
)

report_text = "\n".join(lines)
report_path = OUT_DIR / "M8_2_REPORT.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
log(f"  Report saved: {report_path}")

# ---------------------------------------------------------------------------
# Final Summary to stdout
# ---------------------------------------------------------------------------

log("")
log("=" * 60)
log("M8.2 COMPLETE -- FINAL SUMMARY")
log("=" * 60)
log(f"  Checkpoint SHA-256     : {actual_sha256}")
log(f"  Checkpoint size        : {checkpoint_size_mb:.2f} MB")
log(f"  Total parameters       : {total_params:,}")
log(f"  Learned temperature T  : {T_learned:.6f}")
log(f"  ECE before             : {metrics_before['ece']:.6f}")
log(f"  ECE after              : {metrics_after['ece']:.6f}")
log(f"  NLL before             : {metrics_before['nll']:.6f}")
log(f"  NLL after              : {metrics_after['nll']:.6f}")
log(f"  Brier before           : {metrics_before['brier']:.6f}")
log(f"  Brier after            : {metrics_after['brier']:.6f}")
log(f"  Holdout Accuracy       : {metrics_before['accuracy']:.2f}%")
log(f"  Clean Macro-F1         : {clean_f1:.2f}%")
for key, val in robustness_results.items():
    m = val["metrics"]
    log(f"  {key:30s} Macro-F1={m['macro_f1']:.2f}%  drop={val['macro_f1_drop_vs_clean']:.2f}%")
log(f"  Max Macro-F1 drop      : {max_drop:.2f}%")
log(f"  Mean CPU latency       : {mean_lat:.2f} ms")
log(f"  Median CPU latency     : {median_lat:.2f} ms")
log(f"  P95 latency            : {p95_lat:.2f} ms")
log(f"  P99 latency            : {p99_lat:.2f} ms")
log(f"  Throughput             : {throughput:.2f} images/sec")
log(f"  Test set accessed      : NO")
log("")
log("Output: results/m8_2_calibration_robustness/")
log("M8.2 DONE. No further experiments triggered.")
