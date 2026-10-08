# M7.3 — FULL-TRAINING CONVERGENCE & REFINEMENT REPORT

**Milestone:** M7.3 — Full-Training Convergence & Refinement (10 Epochs)  
**Project:** AgriAI — Tomato Leaf Disease Classification with LS-HGNet  
**Date:** 2026-10-08 05:48:36 UTC  
**Status:** COMPLETE  
**Total Runtime:** 25547.2 s (7.10 hours)  
**Architecture:** LS-HGNet (8,041,354 parameters)  
**Device:** CPU  

---

## 1. Experiment Objective

The objective of M7.3 is to perform a controlled full-training convergence experiment on the complete dataset (12,712 training images, 2,724 validation images) to investigate whether extending the training budget from 5 epochs to 10 epochs enables the LS-HGNet model to achieve performance gains beyond the frozen M7 baseline (88.16% validation Macro-F1).

All historical milestones (M7 Final Selection, M7.1 Rapid STO, M7.2 Progressive STO) and the locked test set remain strictly isolated and untouched.

---

## 2. Experimental Configuration

| Hyperparameter / Setting | Value | Rationale |
| :--- | :--- | :--- |
| **Architecture** | LS-HGNet (fixed M4) | base_channels=64, num_stacks=3, levels=4, dilations=(1,2,4) |
| **Total Trainable Parameters** | 8,041,354 | Exact reproduction of M7 Candidate A |
| **Random Seed** | 42 | Exact reproduction |
| **Image Resolution** | 224 x 224 | Exact reproduction |
| **Batch Size** | 32 | Exact reproduction |
| **Initial Learning Rate** | 0.001 (1e-3) | Exact reproduction |
| **Weight Decay** | 0.0001 (1e-4) | Exact reproduction |
| **Dropout Probability** | 0.30 | Exact reproduction |
| **Loss Function** | Cross-Entropy Loss | Exact reproduction |
| **Optimizer** | AdamW | Exact reproduction |
| **Learning Rate Scheduler** | CosineAnnealingLR ($T_{\max}=10$, $\eta_{\min}=10^{-6}$) | Extended for 10-epoch schedule |
| **Total Epochs** | **10 epochs** | The single experimental variable (vs 5 in M7) |
| **Device** | CPU | Laptop hardware |

---

## 3. Dataset Verification

- **Training Dataset (`data/train.csv`):** 12,712 images (100% full split)
- **Validation Dataset (`data/val.csv`):** 2,724 images (100% full split)
- **Test Dataset (`data/splits/test.csv`):** **STRICTLY LOCKED & ISOLATED** — zero access.

---

## 4. Epoch-by-Epoch Results

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Precision | Val Recall | Val Macro-F1 | Learning Rate | Epoch Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  1 | 2.3661 | 50.26% | 1.0233 | 67.07% | 74.33% | 58.39% | **57.26%** | 9.76e-04 | 2826.7s |
|  2 | 0.9247 | 69.82% | 0.5735 | 79.26% | 77.58% | 72.97% | **70.47%** | 9.05e-04 | 2801.7s |
|  3 | 0.6514 | 77.59% | 0.3693 | 86.78% | 83.10% | 83.78% | **81.41%** | 7.94e-04 | 3753.1s |
|  4 | 0.5638 | 80.54% | 0.3431 | 88.18% | 85.62% | 85.07% | **84.56%** | 6.55e-04 | 2699.5s |
|  5 | 0.4420 | 84.94% | 0.2860 | 89.24% | 87.56% | 86.62% | **85.54%** | 5.01e-04 | 2016.7s |
|  6 | 0.3471 | 88.13% | 0.2438 | 91.30% | 87.70% | 89.67% | **87.64%** | 3.46e-04 | 3058.8s |
|  7 | 0.2923 | 90.21% | 0.1663 | 94.09% | 92.42% | 92.38% | **92.11%** | 2.07e-04 | 2414.7s |
|  8 | 0.2370 | 91.87% | 0.1389 | 95.30% | 94.25% | 93.88% | **93.95%** | 9.64e-05 | 1996.0s |
|  9 | 0.1981 | 93.01% | 0.1262 | 95.52% | 94.35% | 94.37% | **94.23%** **[BEST]** | 2.54e-05 | 1990.9s |
| 10 | 0.1635 | 94.61% | 0.1274 | 95.26% | 93.67% | 94.33% | **93.90%** | 1.00e-06 | 1989.2s |

---

## 5. Convergence Analysis & Best Epoch

- **Best Validation Macro-F1:** **94.23%** achieved at **Epoch 9**
- **Validation Accuracy at Best Epoch:** **95.52%**
- **Validation Macro Precision at Best Epoch:** **94.35%**
- **Validation Macro Recall at Best Epoch:** **94.37%**
- **Validation Loss at Best Epoch:** **0.1262**
- **Macro-F1 at Epoch 5:** **85.54%**
- **Best Macro-F1 in Epochs 6–10:** **94.23%** (if epochs 6–10 evaluated)

### Convergence Dynamics:
- **Did Macro-F1 improve after Epoch 5?** Yes
- **Lowest Validation Loss Epoch:** Epoch 9 (0.1262)
- **Overfitting Diagnosis:** No substantial overfitting was observed through the best checkpoint at Epoch 9. At Epoch 10, training loss continued to decrease while validation loss increased slightly and validation Macro-F1 decreased from 94.23% to 93.90%, indicating the onset of mild overfitting or convergence saturation. Therefore, Epoch 9 was selected as the best checkpoint using validation Macro-F1.

---

## 6. Comparison Against Frozen M7 Baseline

| Metric | Frozen M7 Baseline (Candidate A) | M7.3 Extended Training (10 Epochs) | Absolute Change |
| :--- | :---: | :---: | :---: |
| **Epoch Budget** | 5 epochs | 10 epochs | +5 epochs |
| **Best Epoch** | Epoch 5 | Epoch 9 | — |
| **Validation Macro-F1** | **88.16%** | **94.23%** | **+6.07 percentage points** |
| **Validation Accuracy** | **91.08%** | **95.52%** | **+4.44 percentage points** |
| **Validation Macro Precision**| **89.53%** | **94.35%** | **+4.82 percentage points** |
| **Validation Macro Recall** | **88.66%** | **94.37%** | **+5.71 percentage points** |

---

## 7. Runtime & Computational Efficiency

- **Total Training Duration:** **25547.2 s (7.10 hours)**
- **Average Duration per Epoch:** **2554.7 s (42.6 minutes)**
- **Hardware Platform:** CPU execution (single process)

---

## 8. Test Isolation & Checkpoint Integrity

- **Test Set Isolation:** Confirmed. `data/splits/test.csv` was never opened, loaded, or evaluated during any phase of M7.3.
- **Frozen Baseline Integrity:** Confirmed. Checkpoints in `results/m7_final_selection/` were never modified or overwritten.
- **M7.3 Checkpoint Locations:**
  - Best model checkpoint: `results/m7_3_full_training/checkpoints/best_model.pt`
  - Last checkpoint (Epoch 10): `results/m7_3_full_training/checkpoints/checkpoint_last.pt`

---

## 9. Scientific Conclusions & Interpretation

1. **Impact of Training Duration:** Extending the training duration from 5 to 10 epochs on the 100% training dataset yielded a validation Macro-F1 of **94.23%** at Epoch 9.
2. **Comparison with Baseline:** Compared to the frozen 5-epoch M7 baseline (88.16%), the extended training resulted in **+6.07 percentage points** change in validation Macro-F1.
3. **Model Selection Policy:** As specified by protocol, this model is NOT automatically promoted to the final project model. The official frozen model remains the M7 selected checkpoint until formal project review.
