# M10 — Out-of-Sample Generalization Analysis

**Objective**: Rigorous quantitative comparison of model performance across the independent validation, holdout, and locked test partitions.

---

## 1. Quantitative Milestone Comparison

| Evaluation Stage | Evaluated Split | Sample Count | Accuracy (%) | Macro-F1 (%) |
| :--- | :--- | :---: | :---: | :---: |
| **M7 Frozen Selection** | Full Validation | 2,724 | 91.08% | 88.16% |
| **M7.3 Convergence** | Full Validation | 2,724 | 95.52% | 94.23% |
| **M8.2 Calibration** | Holdout Validation (50%) | 1,362 | 96.11% | 95.12% |
| **M9 Final Locked Test** | **Locked Test (100%)** | **2,724** | **95.56%** | **94.06%** |

---

## 2. Generalization Gap Calculations

### A. Full Validation (M7.3) vs Locked Test (M9)
- **Accuracy Difference**:
  $$\Delta_{\text{Acc}} = 95.52\% - 95.56\% = -0.04\text{ percentage points}$$
  *(Test Accuracy is 0.04 percentage points higher than Full Validation Accuracy)*
- **Macro-F1 Difference**:
  $$\Delta_{\text{F1}} = 94.23\% - 94.06\% = +0.17\text{ percentage points}$$
  *(Macro-F1 drops by a negligible 0.17 percentage points on the locked test set)*

### B. Holdout Validation (M8.2) vs Locked Test (M9)
- **Accuracy Difference**:
  $$\Delta_{\text{Acc}} = 96.11\% - 95.56\% = +0.55\text{ percentage points}$$
- **Macro-F1 Difference**:
  $$\Delta_{\text{F1}} = 95.12\% - 94.06\% = +1.06\text{ percentage points}$$

### C. Frozen Baseline (M7) vs Final Locked Test (M9)
- **Accuracy Improvement**:
  $$95.56\% - 91.08\% = +4.48\text{ percentage points}$$
- **Macro-F1 Improvement**:
  $$94.06\% - 88.16\% = +5.90\text{ percentage points}$$

---

## 3. Scientific Interpretation & Generalization Verdict

1. **Exceptionally Low Generalization Gap**:
   - The validation-to-test performance difference of **0.17 percentage points in Macro-F1** and **-0.04 percentage points in Accuracy** provides strong empirical evidence that the model has not overfit the validation partition.
2. **Stable Out-of-Sample Performance**:
   - A drop of less than 0.2 percentage points on an untouched 2,724-image test set indicates genuine feature learning rather than hyperparameter over-tuning on validation feedback.
3. **Conservative Research Claims**:
   - *What the evidence confirms*: The LS-HGNet model generalizes reliably across independently partitioned subsets of the curated tomato leaf dataset.
   - *What is NOT claimed*: Universal field deployment robustness, immunity to severe environmental degradation (as shown in M8.2 blur/noise vulnerabilities), or clinical multi-crop disease diagnostics.
