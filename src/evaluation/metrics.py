"""
AgriAI Evaluation Metrics Module
Computes classification and computational metrics for experimental benchmarking.

Strict adherence to research specification:
- Macro-averaged metrics (Accuracy, Macro Precision, Macro Recall, Macro F1)
- Per-class metrics
- Confusion matrix generation and visualization
- Measured CPU inference latency (batch size 1, excluded warmup, median latency)
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Computes standard classification metrics.
    
    Args:
        y_true: Ground truth class indices.
        y_pred: Predicted class indices.
        class_names: Optional list of class string names.
        
    Returns:
        Dictionary containing overall macro metrics and per-class reports.
    """
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    target_names = class_names if class_names is not None else [f"Class_{i}" for i in range(len(np.unique(y_true)))]
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_report": report_dict,
        "confusion_matrix": cm.tolist()
    }


def measure_inference_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 128, 128),
    device: str = "cpu",
    num_warmup: int = 20,
    num_runs: int = 100
) -> Dict[str, Any]:
    """
    Measures CPU single-sample inference latency.
    Excludes warm-up cycles, runs multiple repetitions, and reports median latency.
    
    Args:
        model: Evaluated PyTorch neural network.
        input_shape: Tensor shape with batch size = 1.
        device: Execution device ('cpu' standard).
        num_warmup: Number of unmeasured warm-up iterations.
        num_runs: Number of timed repetitions.
        
    Returns:
        Dictionary of latency statistics in milliseconds.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    dummy_input = torch.randn(*input_shape, device=dev)

    # Warm-up cycles
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)

    # Timed runs
    timings = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            timings.append((end - start) * 1000.0)  # Convert to ms

    timings_arr = np.array(timings)
    return {
        "batch_size": input_shape[0],
        "input_shape": list(input_shape),
        "device": device,
        "num_warmup_iterations": num_warmup,
        "num_measured_iterations": num_runs,
        "median_latency_ms": float(np.median(timings_arr)),
        "mean_latency_ms": float(np.mean(timings_arr)),
        "std_latency_ms": float(np.std(timings_arr)),
        "min_latency_ms": float(np.min(timings_arr)),
        "max_latency_ms": float(np.max(timings_arr))
    }


def plot_confusion_matrix(
    cm: np.ndarray or list,
    class_names: List[str],
    output_path: str or Path,
    title: str = "Confusion Matrix"
) -> None:
    """
    Renders and saves a confusion matrix heatmap.
    """
    if isinstance(cm, list):
        cm = np.array(cm)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def evaluate_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
    class_names: Optional[List[str]] = None
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Performs full evaluation pass over a DataLoader.
    
    Args:
        model: Evaluated PyTorch model.
        dataloader: PyTorch DataLoader.
        device: Device string.
        class_names: List of class names.
        
    Returns:
        metrics_dict, y_true, y_pred
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(dev)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    metrics = compute_classification_metrics(y_true, y_pred, class_names=class_names)
    return metrics, y_true, y_pred
