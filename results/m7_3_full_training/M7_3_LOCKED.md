# M7.3 — EXPERIMENT FROZEN & LOCKED

**Lock Date:** 2026-10-08 05:55:00 UTC  
**Milestone:** M7.3 — Full-Training Convergence & Refinement (10 Epochs)  
**Status:** **FROZEN & LOCKED (READ-ONLY)**  

---

## 1. Lock Declaration

The M7.3 experimental directory (`results/m7_3_full_training/`) is hereby **FROZEN AND LOCKED**.
No further training, retraining, inference, re-evaluation, or file modifications may occur inside this directory, except explicitly documented read-only audit corrections.

---

## 2. Selected Checkpoint & Performance Specifications

| Parameter | Specification |
| :--- | :--- |
| **Selected Checkpoint Path** | `results/m7_3_full_training/checkpoints/best_model.pt` |
| **Selected Checkpoint Epoch** | **Epoch 9** |
| **Best Validation Macro-F1** | **94.23%** (94.2306%) |
| **Best Validation Accuracy** | **95.52%** (95.5213%) |
| **Best Validation Macro Precision** | **94.35%** (94.3487%) |
| **Best Validation Macro Recall** | **94.37%** (94.3652%) |
| **Validation Loss at Best** | **0.1262** |
| **Original Frozen M7 Baseline F1** | **88.16%** (88.1642%) |
| **Improvement over M7 Baseline** | **+6.07 percentage points** Macro-F1 |
| **Checkpoint File Size** | **97,278,348 bytes** (92.7719 MB) |
| **Checkpoint SHA-256 Checksum** | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` |
| **Model Parameters** | **8,041,354** trainable parameters |

---

## 3. Preservation of Historical Milestones & Model Governance

1. **Isolation from M7:** This checkpoint is **NOT** copied into `results/m7_final_selection/`. The official frozen M7 model remains untouched in its original location.
2. **Preservation of Preceding Milestones:**
   - `results/m7_final_selection/` (M7) is intact and frozen.
   - `results/m7_1_sto_refinement/` (M7.1) is intact and frozen.
   - `results/m7_2_progressive_sto/` (M7.2) is intact and frozen.
3. **Test Split Isolation:** The locked test set was never accessed during M7.3.
4. **Promotion Status:** This checkpoint represents the **M7.3 best validation checkpoint**. It is not automatically promoted to the final project model. Official promotion and independent locked test evaluation remain subject to formal project review.
