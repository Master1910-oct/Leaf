# M7.3 — FINAL READ-ONLY SCIENTIFIC AUDIT REPORT

**Audit Date:** 2026-10-08 05:50:00 UTC  
**Audited Milestone:** M7.3 — Full-Training Convergence & Refinement (10 Epochs)  
**Project:** AgriAI — Tomato Leaf Disease Classification with LS-HGNet  
**Audited Directory:** `results/m7_3_full_training/`  
**Verdict:** **PASS WITH LIMITATIONS**  

---

## 1. Audit Scope & Rules

This audit is an independent, read-only scientific evaluation of the completed M7.3 experimental run.
- **Strict Read-Only Enforcement:** No model training, retraining, inference, or re-evaluations were performed during this audit.
- **Test Set Isolation:** The locked test split (`data/splits/test.csv`) was never loaded, evaluated, or referenced.
- **Historical Milestone Protection:** `results/m7_final_selection/`, `results/m7_1_sto_refinement/`, and `results/m7_2_progressive_sto/` were verified as unmodified.
- **Promotion Rule:** No automatic promotion of M7.3 to the final model is permitted; the official project model remains the frozen M7 selected checkpoint.

---

## 2. Artifact Verification

All required M7.3 files exist, are non-empty, and maintain complete structural integrity:

| Artifact Path | File Size | Description & Verification |
| :--- | :---: | :--- |
| `checkpoints/best_model.pt` | 97,278,348 B (92.77 MB) | Best checkpoint; verified Epoch 9 state dict |
| `checkpoints/checkpoint_last.pt` | 97,288,135 B (92.78 MB) | Complete resume checkpoint at Epoch 10 |
| `metrics/metrics.json` | 1,100 B | Full validation metric summary |
| `metrics/history.json` | 3,141 B | Complete 10-epoch training and validation trajectory |
| `metrics/classification_report.json` | 2,048 B | Per-class precision, recall, and F1 metrics |
| `plots/convergence_curve.png` | 299,106 B | Training loss, validation loss, and Macro-F1 curves |
| `plots/confusion_matrix.png` | 338,559 B | Normalized validation confusion matrix heatmap |
| `config/m7_3_config.json` | 352 B | Complete experimental configuration parameters |
| `M7_3_CONVERGENCE_REPORT.md` | 6,290 B | Master convergence analysis report |

- **Model Parameter Count:** Verified as **8,041,354** trainable parameters.
- **Checkpoint Epoch:** `best_model.pt` metadata confirms `epoch: 9`, `val_macro_f1: 0.9423056`, and `val_loss: 0.126168`.

---

## 3. Metric Consistency & Verification

Cross-checks performed across `history.json`, `metrics.json`, `classification_report.json`, and `train_m7_3.log` confirm complete numerical consistency:

### Epoch 9 (Global Best Checkpoint):
- **Train Loss:** 0.1981
- **Train Accuracy:** 93.01%
- **Validation Loss:** 0.1262
- **Validation Accuracy:** 95.52% (95.5213%)
- **Validation Macro Precision:** 94.35% (94.3487%)
- **Validation Macro Recall:** 94.37% (94.3652%)
- **Validation Macro-F1:** **94.23%** (94.2306%)
- **Learning Rate:** $2.5447\times 10^{-5}$
- **Global Maximum Check:** Verified that Epoch 9 represents the global maximum validation Macro-F1 across all 10 epochs.

### Epoch 10 (Final Schedule Epoch):
- **Train Loss:** 0.1635
- **Train Accuracy:** 94.61%
- **Validation Loss:** 0.1274
- **Validation Accuracy:** 95.26%
- **Validation Macro Precision:** 93.67%
- **Validation Macro Recall:** 94.33%
- **Validation Macro-F1:** 93.90%
- **Learning Rate:** $1.0000\times 10^{-6}$

---

## 4. Convergence & Improvement Claims

- **Epoch 5 Macro-F1 (Within M7.3 Run):** 85.54%
- **Epoch 9 Macro-F1 (M7.3 Peak):** **94.23%** (+8.69 percentage points over Epoch 5)
- **Frozen M7 Baseline Macro-F1 (Candidate A, 5 Epochs):** **88.16%** (88.1642%)
- **Frozen M7 Baseline Accuracy (Candidate A, 5 Epochs):** **91.08%** (91.0793%)
- **Independent Improvement Calculation:**
  - Macro-F1 Delta: $94.2306\% - 88.1642\% = \mathbf{+6.07}$ **percentage points**
  - Accuracy Delta: $95.5213\% - 91.0793\% = \mathbf{+4.44}$ **percentage points**

---

## 5. Overfitting & Dynamics Audit

The convergence report wording was audited and refined to adhere strictly to scientific precision:
> *"No substantial overfitting was observed through the best checkpoint at Epoch 9. At Epoch 10, training loss continued to decrease while validation loss increased slightly and validation Macro-F1 decreased from 94.23% to 93.90%, indicating the onset of mild overfitting or convergence saturation. Therefore, Epoch 9 was selected as the best checkpoint using validation Macro-F1."*

---

## 6. Dataset & Test Set Isolation Audit

- **Training Split (`data/train.csv`):** Verified 12,712 images (100% full training split).
- **Validation Split (`data/val.csv`):** Verified 2,724 images (100% full validation split).
- **Test Set Isolation:** Confirmed. `data/splits/test.csv` was never imported, opened, or evaluated. Model selection was governed strictly by validation Macro-F1.
- **Terminology:** The metric 94.23% is strictly a **Validation Macro-F1 score**, NOT a test evaluation metric.

---

## 7. Historical Experiment Integrity

All preceding experimental milestones were audited for filesystem integrity:

| Experiment Directory | File Count | Latest Modification Timestamp (UTC) | Integrity Status |
| :--- | :---: | :---: | :--- |
| `results/m7_final_selection/` | 20 files | 2026-10-06 10:17:35 UTC | **UNTOUCHED / FROZEN** |
| `results/m7_1_sto_refinement/` | 30 files | 2026-10-06 20:03:33 UTC | **UNTOUCHED / FROZEN** |
| `results/m7_2_progressive_sto/` | 22 files | 2026-10-07 13:52:27 UTC | **UNTOUCHED / FROZEN** |

---

## 8. Checkpoint Cryptographic Hash

The exact SHA-256 cryptographic hash of the selected M7.3 checkpoint was computed:

- **File Path:** `results/m7_3_full_training/checkpoints/best_model.pt`
- **File Size:** **97,278,348 bytes** (92.7719 MB)
- **SHA-256 Checksum:**  
  `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`

---

## 9. Training Protocol Specification

The M7.3 protocol retained the underlying model architecture and hyperparameter settings while extending the learning rate schedule:
- **Architecture:** LS-HGNet (fixed M4 structure: `base_channels=64`, `num_stacks=3`, `encoder_levels=4`, `dilation_rates=(1, 2, 4)`)
- **Image Resolution:** 224 $\times$ 224
- **Batch Size:** 32
- **Initial Learning Rate:** 0.001
- **Weight Decay:** 0.0001
- **Dropout:** 0.30
- **Optimizer:** AdamW
- **Scheduler:** CosineAnnealingLR ($T_{\max}=10$, $\eta_{\min}=10^{-6}$)
- **Device:** CPU
- **Distinction from M7:** M7.3 must not be described as an identical protocol; it utilized an extended 10-epoch CosineAnnealingLR schedule rather than M7's 5-epoch schedule.

---

## 10. Environment State Audit

Post-training environment maintenance was performed for reporting compatibility; the M7.3 training run itself had already completed prior to package adjustments.
Currently verified library versions in the virtual environment:
- **Python:** 3.12.10
- **PyTorch:** 2.2.2+cpu
- **torchvision:** 0.17.2+cpu
- **NumPy:** 1.26.4
- **SciPy:** 1.13.1
- **scikit-learn:** 1.9.1

---

## 11. Limitations & Governance Status

1. **Proxy vs Final Status:** M7.3 is designated as the **M7.3 extended-training candidate** (or best validation checkpoint). It has **NOT** been automatically promoted to the final project model.
2. **Test Set Evaluation:** The 94.23% Macro-F1 metric was obtained strictly on the validation set. An independent locked-test evaluation has not been conducted for M7.3.
3. **Hardware Runtime:** Training required 25,547.2 seconds (7.10 hours) on CPU, reflecting full convergence dynamics.

---

## 12. Final Scientific Verdict

**Verdict:** **PASS WITH LIMITATIONS**

**Justification:**  
The M7.3 artifacts, metrics, and convergence logs are mathematically and structurally consistent. The model demonstrated genuine convergence gains (+6.07 percentage points Macro-F1) through Epoch 9 under the extended 10-epoch schedule without severe overfitting. The verdict is assigned *with limitations* solely because the model has not yet undergone independent evaluation on the locked test set, and official model promotion remains subject to future project governance.
