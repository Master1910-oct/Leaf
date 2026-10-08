# M9 — Final Locked-Test Evaluation Report

**Execution Date**: 2026-10-08 10:37 UTC
**Project**: D:/Leaf  |  **Milestone**: M9 (Final Evaluation)
**Status**: **COMPLETED & VERIFIED (PASS)**

---

## 1. Executive Summary

This document reports the **final generalization evaluation (M9)** of the frozen **M7.3 Epoch-9 LS-HGNet checkpoint** on the locked test dataset (`data/test.csv`, 2,724 images). Prior to this evaluation, the test set remained strictly locked throughout milestones M0 through M8.2. Evaluation was performed in a **single, deterministic pass** without fine-tuning, threshold tuning, or test-based calibration fitting.

- **Test Accuracy**: **95.56%** (2603 / 2724)
- **Test Macro-F1**: **94.06%**
- **Test Weighted-F1**: **95.55%**
- **Test Macro Precision**: **94.00%**
- **Test Macro Recall**: **94.43%**
- **Generalization Gap vs M7.3 Full Val**: **+0.17% Macro-F1** (Val: 94.23% → Test: 94.06%)
- **Generalization Gap vs M8.2 Holdout**: **+1.06% Macro-F1** (Holdout: 95.12% → Test: 94.06%)

## 2. Model Integrity & Architecture

| Specification | Detail |
|---|---|
| Model Architecture | LS-HGNet (Lightweight Stacked Hourglass Network) |
| Checkpoint Path | `results/m7_3_full_training/checkpoints/best_model.pt` |
| Expected SHA-256 | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` |
| Verified SHA-256 | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` (**PASS**) |
| Parameter Count | 8,041,354 (**Verified exact: 8,041,354**) |
| Architecture Hyperparameters | base_channels=64, stacks=3, levels=4, dilation=(1,2,4), dropout=0.30 |
| State Dict Loading | `strict=True` (**PASS — 0 missing, 0 unexpected keys**) |
| Gradient Tracking | `requires_grad=False`, `torch.no_grad()` (**Weights unmodified**) |

## 3. Test Dataset & Preprocessing

| Dataset Attribute | Verification |
|---|---|
| Test Manifest | `data/test.csv` |
| Total Samples | 2,724 images |
| Classes | 10 distinct tomato leaf conditions |
| Duplicate Filepaths | 0 (**Verified**) |
| Null Entries | 0 (**Verified**) |
| Missing Images on Disk | 0 (**Verified**) |
| Overlap with Train/Val | 0 samples (**Zero data leakage verified**) |
| Image Resolution | 224 × 224 pixels |
| Normalization | ImageNet Standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) |
| Data Augmentations | NONE (Strictly deterministic evaluation transform) |

### Class Distribution (Support)

| Class Index | Class Name | Test Support | % of Test Set |
|---|---|---|---|
| 0 | `Tomato___Bacterial_spot` | 319 | 11.71% |
| 1 | `Tomato___Early_blight` | 150 | 5.51% |
| 2 | `Tomato___Late_blight` | 286 | 10.5% |
| 3 | `Tomato___Leaf_Mold` | 143 | 5.25% |
| 4 | `Tomato___Septoria_leaf_spot` | 265 | 9.73% |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | 252 | 9.25% |
| 6 | `Tomato___Target_Spot` | 211 | 7.75% |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 804 | 29.52% |
| 8 | `Tomato___Tomato_mosaic_virus` | 56 | 2.06% |
| 9 | `Tomato___healthy` | 238 | 8.74% |

## 4. Final Test Performance

| Metric | Test Performance | M7.3 Val Reference | M7 Baseline Reference |
|---|---|---|---|
| **Overall Accuracy** | **95.56%** | 95.52% | 91.08% |
| **Macro-F1** | **94.06%** | 94.23% | 88.16% |
| **Macro Precision** | **94.00%** | 94.35% | — |
| **Macro Recall** | **94.43%** | 94.37% | — |
| **Weighted-F1** | **95.55%** | — | — |
| **Weighted Precision** | **95.72%** | — | — |
| **Weighted Recall** | **95.56%** | — | — |

## 5. Complete Per-Class Results

| Index | Class Name | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|---|
| 0 | `Tomato___Bacterial_spot` | 97.08% | 93.73% | 95.37% | 319 |
| 1 | `Tomato___Early_blight` | 95.52% | 85.33% | 90.14% | 150 |
| 2 | `Tomato___Late_blight` | 93.98% | 98.25% | 96.07% | 286 |
| 3 | `Tomato___Leaf_Mold` | 97.12% | 94.41% | 95.74% | 143 |
| 4 | `Tomato___Septoria_leaf_spot` | 94.55% | 98.11% | 96.30% | 265 |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | 85.96% | 97.22% | 91.25% | 252 |
| 6 | `Tomato___Target_Spot` | 91.58% | 82.46% | 86.78% | 211 |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 99.62% | 98.63% | 99.12% | 804 |
| 8 | `Tomato___Tomato_mosaic_virus` | 84.62% | 98.21% | 90.91% | 56 |
| 9 | `Tomato___healthy` | 100.00% | 97.90% | 98.94% | 238 |

## 6. Confusion Matrix Analysis

A total of **121 classification errors** occurred across the 2,724 test samples.

- **Raw Count Matrix**: [`plots/confusion_matrix.png`](plots/confusion_matrix.png)
- **Normalized Matrix**: [`plots/confusion_matrix_normalized.png`](plots/confusion_matrix_normalized.png)
- **Metrics CSV**: [`metrics/confusion_matrix.csv`](metrics/confusion_matrix.csv)

### Top 5 Most Frequent Confusion Pairs

| True Class | Predicted Class | Misclassified Count | % of True Class |
|---|---|---|---|

| `Tomato___Target_Spot` | `Tomato___Spider_mites Two-spotted_spider_mite` | 29 | 13.74% |
| `Tomato___Bacterial_spot` | `Tomato___Target_Spot` | 10 | 3.13% |
| `Tomato___Early_blight` | `Tomato___Late_blight` | 9 | 6.0% |
| `Tomato___Spider_mites Two-spotted_spider_mite` | `Tomato___Tomato_mosaic_virus` | 5 | 1.98% |
| `Tomato___Target_Spot` | `Tomato___Septoria_leaf_spot` | 5 | 2.37% |

## 7. Confidence Analysis

| Metric | Raw Softmax Probabilities | Temp-Scaled (T=1.1558) |
|---|---|---|

| Mean Overall Confidence | 95.80% | 94.83% |
| Median Overall Confidence | 99.93% | 99.80% |
| Mean Confidence (Correct Predictions) | 96.97% | 96.10% |
| Mean Confidence (Incorrect Predictions) | 70.65% | 67.64% |
| Min / Max Confidence | 32.17% / 100.00% | — |

- **Plot**: [`plots/confidence_distribution.png`](plots/confidence_distribution.png)

## 8. Generalization Gap & Comparison

| Milestone | Split | Sample Count | Accuracy | Macro-F1 |
|---|---|---|---|---|

| M7 Baseline | Full Val | 2,724 | 91.08% | 88.16% |
| M7.3 Convergence | Full Val | 2,724 | 95.52% | 94.23% |
| M8.2 Calibration | Holdout Val (50%) | 1,362 | 96.11% | 95.12% |
| **M9 Final Locked Test** | **Locked Test** | **2,724** | **95.56%** | **94.06%** |

### Gap Analysis
- **M7.3 Full Val vs M9 Test Gap**:
  - Accuracy: `95.52% - 95.56% = -0.04%`
  - Macro-F1: `94.23% - 94.06% = +0.17%`
- **M8.2 Val Holdout vs M9 Test Gap**:
  - Accuracy: `96.11% - 95.56% = +0.55%`
  - Macro-F1: `95.12% - 94.06% = +1.06%`

**Scientific Interpretation**:
The generalization drop from validation to held-out test data is **0.17% in Macro-F1** and **0.04% in Accuracy**. A drop of this magnitude is typical for deep convolutional networks evaluated on independent test partitions with natural visual variance. Crucially, the final test performance of **94.06% Macro-F1** represents a substantial **+5.90% absolute improvement** over the original frozen M7 baseline (88.16%).

## 9. Error Analysis & Class Dynamics

### Highest Performing Classes

- **`Tomato___Tomato_Yellow_Leaf_Curl_Virus`**: F1 = **99.12%** (Precision: 99.62%, Recall: 98.63%, Support: 804)
- **`Tomato___healthy`**: F1 = **98.94%** (Precision: 100.00%, Recall: 97.90%, Support: 238)
- **`Tomato___Septoria_leaf_spot`**: F1 = **96.30%** (Precision: 94.55%, Recall: 98.11%, Support: 265)

### Most Vulnerable Classes

- **`Tomato___Target_Spot`**: F1 = **86.78%** (Precision: 91.58%, Recall: 82.46%, Support: 211)
- **`Tomato___Early_blight`**: F1 = **90.14%** (Precision: 95.52%, Recall: 85.33%, Support: 150)
- **`Tomato___Tomato_mosaic_virus`**: F1 = **90.91%** (Precision: 84.62%, Recall: 98.21%, Support: 56)

### Primary Pathology Confusion Patterns

1. **Early Blight vs Septoria / Target Spot**: Foliar necrotic lesions with concentric rings exhibit visually overlapping chlorotic halos, leading to minor confusion between Early Blight, Late Blight, and Septoria leaf spot.
2. **Class Imbalance Support**: Certain classes such as `Tomato___Tomato_mosaic_virus` (56 samples) have lower support compared to `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (804 samples), yet the model maintains robust recall across rare classes.

## 10. Test Data Governance Certification

> **M9 GOVERNANCE CERTIFICATION**:
>
> Test data (`data/test.csv`) was used strictly and exclusively for final generalization evaluation.
>
> - No training or fine-tuning occurred on test samples.
> - No hyperparameter optimization or architecture searches touched test samples.
> - Classification decisions used standard `argmax`; no threshold tuning was performed using test labels.
> - Temperature Scaling calibration was NOT fitted on test data (frozen T = 1.1558 was applied from M8.2 validation strictly for reporting).
> - The test evaluation was performed in a single one-shot execution pass.

## 11. Limitations

- **Domain Shift**: While the model achieves 95.56% test accuracy on the curated dataset, performance on in-field phone camera images with diverse backgrounds, lighting extremes, and multi-disease co-infections may exhibit lower reliability (as documented in the M8.2 robustness study).
- **High-Frequency Degradation**: As observed in M8.2, severe blur and sensor noise significantly degrade discriminability.
- **Evaluation Modality**: Evaluated deterministically at 224×224 on CPU; edge deployment requires latency budgeting (mean CPU latency 132 ms).

## 12. Final Scientific Conclusion

The M9 Final Locked-Test Evaluation demonstrates that the **LS-HGNet architecture trained through M7.3** successfully generalizes to completely unseen test data with **95.56% Accuracy** and **94.06% Macro-F1**. The evaluation confirms that the improvements gained throughout the STO exploration and full-convergence phases translate directly into genuine out-of-sample generalization.

With M9 completed, all experimental and evaluation milestones for the Smart Farming project are **officially complete and locked**.