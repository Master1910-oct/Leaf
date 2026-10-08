# M10 Final Evidence Consolidation Report

**Date**: 2026-10-08 UTC  
**Project**: Tomato Leaf Disease Classification (AgriAI / Leaf)  
**Milestone**: M10 — Final Evidence Consolidation & Research Package  
**Status**: **FINAL RESEARCH EVIDENCE PACKAGE: PASS**

---

## 1. Executive Summary

This report completes milestone **M10**, synthesizing all verified experimental evidence from research inception through the final locked-test evaluation (M9). 
The core model, a custom **Lightweight Stacked Hourglass Network (LS-HGNet)** with 8,041,354 parameters, completed 10-epoch full convergence in M7.3, underwent calibration and robustness profiling in M8.2, and was evaluated on the locked test partition in M9.
The final locked-test evaluation achieved **95.56% Accuracy** and **94.06% Macro-F1** across 2,724 unseen images, demonstrating exceptional generalization stability (0.17 percentage points gap vs full validation).

---

## 2. Final Model

- **Architecture**: LS-HGNet (Lightweight Stacked Hourglass Network)
- **Parameters**: 8,041,354 trainable parameters
- **Structure**: 3 hourglass stacks, 4 encoder levels per stack, base channels $C_{base} = 64$.
- **Bottlenecks**: Multi-dilation residual bottleneck branches with dilation rates $(1, 2, 4)$ and channel reduction factor 2.
- **Classifier**: AdaptiveAvgPool2d $\to$ Dropout (0.30) $\to$ Linear(64, 10).
- **Frozen Checkpoint**: `results/m7_3_full_training/checkpoints/best_model.pt`
- **Verified SHA-256**: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`

---

## 3. Dataset

- **Dataset Source**: Curated PlantVillage tomato leaf pathology dataset.
- **Total Images**: 18,160 images across 10 distinct pathology and healthy conditions.
- **Partitions**:
  - Training: 12,712 images (70.0%)
  - Validation: 2,724 images (15.0%)
  - Locked Test: 2,724 images (15.0%)
- **Data Integrity**: 0 missing files, 0 duplicate paths, 0 null fields, 0 overlap between splits.

---

## 4. Experimental Timeline

- **M0**: Mathematical specification of LS-HGNet and Social Spider Optimization (STO).
- **M1–M5**: Data pipeline stratification, Baseline CNN, and LS-HGNet implementation.
- **M6**: STO vs Random Search benchmark on validation proxies.
- **M7**: Candidate screening; Candidate A locked as official baseline (88.16% Macro-F1).
- **M7.1–M7.2**: Rapid STO screening and data-scaling dynamics (20% $\to$ 50%).
- **M7.3**: Full 100% training convergence (10 epochs); Epoch 9 frozen as global best checkpoint (94.23% Macro-F1).
- **M8.2**: Validation Temperature Scaling ($T=1.1558$), 6-condition robustness evaluation, and CPU latency profiling.
- **M9**: One-shot evaluation on locked test partition (95.56% Accuracy, 94.06% Macro-F1).
- **M10**: Evidence synthesis and master packaging.

---

## 5. Validation Results (M7.3)

- **Accuracy**: 95.52% (2,602 / 2,724)
- **Macro-F1**: 94.23%
- **Macro Precision**: 94.35%
- **Macro Recall**: 94.37%
- **Weighted-F1**: 95.49%
- **Validation Loss**: 0.1262

---

## 6. Holdout Results (M8.2)

- **Holdout Partition**: 1,362 samples (50% stratified validation split, val only)
- **Holdout Accuracy**: 96.11%
- **Holdout Macro-F1**: 95.12%
- **Holdout Precision**: 95.31%
- **Holdout Recall**: 95.18%
- **Learned Temperature**: $T = 1.155831$ (fitted on separate 50% calibration split)

---

## 7. Locked Test Results (M9)

- **Test Partition**: 2,724 samples (`data/test.csv`, 100% test split)
- **Test Accuracy**: **95.56%** (2,603 / 2,724)
- **Test Macro-F1**: **94.06%**
- **Test Weighted-F1**: **95.55%**
- **Macro Precision**: **94.00%**
- **Macro Recall**: **94.43%**
- **Weighted Precision**: 95.72%
- **Weighted Recall**: 95.56%

---

## 8. Per-Class Analysis

- **Top Classes**:
  1. *Tomato Yellow Leaf Curl Virus*: 99.12% F1 (Support: 804)
  2. *Tomato Healthy*: 98.94% F1 (Precision: 100.0%, Support: 238)
  3. *Septoria Leaf Spot*: 96.30% F1 (Support: 265)
- **Challenging Classes**:
  1. *Target Spot*: 86.78% F1 (Recall: 82.46%, Support: 211)
  2. *Early Blight*: 90.14% F1 (Recall: 85.33%, Support: 150)
  3. *Tomato Mosaic Virus*: 90.91% F1 (Precision: 84.62%, Support: 56)

---

## 9. Confusion Matrix Analysis

- **Total Test Errors**: 121 errors out of 2,724 samples (4.44% error rate).
- **Major Confusion Pairs**:
  - *Target Spot $\to$ Spider Mites*: 29 errors (13.74% of Target Spot)
  - *Bacterial Spot $\to$ Target Spot*: 10 errors (3.13% of Bacterial Spot)
  - *Early Blight $\to$ Late Blight*: 9 errors (6.00% of Early Blight)
  - *Spider Mites $\to$ Mosaic Virus*: 5 errors (1.98% of Spider Mites)
  - *Target Spot $\to$ Septoria*: 5 errors (2.37% of Target Spot)

---

## 10. Generalization Analysis

- **M7.3 Full Val vs M9 Test Gap**:
  - Accuracy: `95.52% - 95.56% = -0.04 percentage points`
  - Macro-F1: `94.23% - 94.06% = +0.17 percentage points`
- **M8.2 Holdout vs M9 Test Gap**:
  - Accuracy: `96.11% - 95.56% = +0.55 percentage points`
  - Macro-F1: `95.12% - 94.06% = +1.06 percentage points`
- **Net Gain over M7 Baseline**:
  - Macro-F1: `94.06% - 88.16% = +5.90 percentage points`
- **Interpretation**: Out-of-sample generalization is exceptionally tight, with a performance delta of under 0.2 percentage points on Macro-F1.

---

## 11. Reproducibility

- Single deterministic evaluation transform: ImageNet normalization at $224 \times 224$.
- Checkpoint loading: verified `strict=True` with 0 missing and 0 unexpected keys.
- Environment: Python 3.12.10, PyTorch 2.2.2+cpu.
- Complete prediction logs with raw and scaled probabilities archived in `results/m9_final_locked_test/metrics/test_predictions.csv`.

---

## 12. Checkpoint Integrity

- Expected SHA-256: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`
- Actual SHA-256: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`
- Verdict: **PASS (100% Cryptographic Match)**

---

## 13. Artifact Inventory

All required milestone files exist and are verified:
- M7.3 convergence checkpoints, logs, and history curves.
- M8.2 calibration metrics, robustness results, and CPU latency benchmarks.
- M9 locked-test metrics, confusion matrix CSVs, and prediction records.
- M10 complete evidence and analysis documentation.

---

## 14. Limitations

1. **Environmental Diversity**: Laboratory/field dataset leaves against uniform or controlled backgrounds. Performance under complex natural canopy clutter may degrade.
2. **Frequency Sensitivity**: Severe blur (47.8% F1 drop) and sensor noise (23.7% F1 drop) degrade performance significantly.
3. **CPU Inference Latency**: Unbatched CPU latency is $132.2 \pm 16.7$ ms (7.57 img/s), requiring GPU or batch optimization for high-throughput edge deployment.
4. **Single-Plant Focus**: Evaluated specifically on tomato foliage; cross-crop transferability has not been tested.

---

## 15. Final Evidence Status

All required components of the scientific evaluation have been executed, cryptographically verified, audited for data leakage, and cross-referenced with zero discrepancies:

```text
FINAL RESEARCH EVIDENCE PACKAGE: PASS
```
