# M7.2 — PROGRESSIVE STO DATA-SCALING EXPERIMENT REPORT

**Milestone:** M7.2 — Progressive STO Data-Scaling (20% -> 50% Cumulative Data)  
**Project:** AgriAI — Tomato Leaf Disease Classification with LS-HGNet  
**Date:** 2026-10-07 13:52:27 UTC  
**Status:** COMPLETED (Hard 5-Hour CPU Limit Enforced)  
**Total Runtime:** 6727.5 s (1.87 hours)  
**Total Candidates Evaluated:** 6  

---

## 1. Executive Summary

This experiment investigates whether Sooty Tern Optimization (STO) continues to identify effective hyperparameter configurations when the training subset is progressively expanded from the Stage-1 20% stratified subset to a cumulative 50% subset (6,356 images).

All historical milestones (M7 Final Selection and M7.1 Rapid STO Screening) remain strictly frozen and untouched. The locked test set was never accessed.

---

## 2. Research Question

> **Core Scientific Question:**  
> Does STO continue to identify useful hyperparameter configurations when the training subset is expanded from the existing 20% subset to a cumulative 50% subset?

This is a progressive data-scaling experiment with separate candidate evaluations, not a continuation of training the same model checkpoints.

---

## 3. Dataset Construction & Stratification

| Dataset Stage | Source / Definition | Sample Count | % of Full Train | Class Overlap with Stage 1 |
| :--- | :--- | :---: | :---: | :---: |
| **Full Train** | `data/train.csv` | 12,712 | 100.0% | N/A |
| **Stage 1 (20%)** | `results/m7_2_progressive_sto/data/stage1_20pct.csv` | 2,542 | 20.00% | Reference |
| **Added 30%** | `results/m7_2_progressive_sto/data/stage2_added_30pct.csv` | 3,814 | 30.00% | **0 (Zero Overlap)** |
| **Cumulative 50%** | `results/m7_2_progressive_sto/data/cumulative_50pct.csv` | 6,356 | 50.00% | Cumulative (No duplicates) |
| **Validation** | `data/val.csv` | 2,724 | Full 100% | 0 (No train overlap) |

### Class Distribution Verification:
All 10 classes in the cumulative 50% subset closely mirror the full training dataset distribution (maximum class proportion deviation = 0.000079).

---

## 4. Stage-2 STO Protocol & Budget

- **Search Algorithm:** Stabilized STO (`ProjectSpecificStabilizedSTO`) with `reflect_clip`
- **Population:** $N=3$ agents
- **Iterations:** $T=2$ iterations ($3 + 3 = 6$ candidate evaluations)
- **Epochs per Candidate:** 1 epoch(s)
- **Batch Size:** 32 (fixed)
- **Resolution:** 224x224
- **Hard Limit:** 5.0 Hours (18,000 s)

---

## 5. Candidate Evaluation Results (Stage 2: 50% Cumulative Data)

| Cand ID | Stage / Agent | LR | WD | Dropout | BS | Val Acc | Val Macro-F1 | Fitness | Global Best F1 | Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  1 | Iter 0 / A1 | 3.53e-03 | 5.70e-05 | 0.429 | 32 | 57.27% | **42.40%** | 0.5760 | 42.40% | 1224.3s |
|  2 | Iter 0 / A2 | 1.54e-04 | 7.99e-03 | 0.381 | 32 | 60.02% | **43.62%** | 0.5638 | 43.62% | 1106.7s |
|  3 | Iter 0 / A3 | 1.80e-04 | 6.33e-05 | 0.185 | 32 | 69.97% | **57.93%** | 0.4207 | 57.93% | 1103.9s |
|  4 | Iter 1 / A1 | 1.00e-03 | 1.56e-04 | 0.454 | 32 | 52.94% | **41.48%** | 0.5852 | 57.93% | 1097.5s |
|  5 | Iter 1 / A2 | 1.62e-03 | 6.91e-06 | 0.476 | 32 | 67.84% | **57.96%** | 0.4204 | 57.96% | 1099.1s |
|  6 | Iter 1 / A3 | 2.74e-03 | 3.31e-05 | 0.361 | 32 | 59.54% | **47.81%** | 0.5219 | 57.96% | 1095.1s |

---

## 6. STO Population Lineage

1. **Initial Population (Iteration 0, Candidates 1–3):**
   - Three independent positions sampled uniformly in log/linear search bounds with seed 42.
   - Evaluated to establish baseline fitness and determine the initial global best ($P_{best}$).
2. **STO Position Update (Iteration 1, Candidates 4–6):**
   - Collision avoidance parameter: $S_A = 2.0 \times (1 - 1/2) = 1.0$
   - Exploration displacement: $D_s = C_{sa} + M_s = S_A P_i + 0.5 \vec{r} \odot (P_{best} - P_i)$
   - Damped spiral direction $\vec{\Omega}_4$ applied to position update: $P_i(t+1) = P_{best} + D_s \odot \vec{\Omega}_4$
   - `reflect_clip` enforced continuous boundary reflection.
   - Candidates 4–6 are mathematically guided by the STO equations and the Stage-2 best position.

---

## 7. Comparative Analysis Across Experimental Milestones

| Method / Stage | Training Data | Epochs | Candidates | Best Val Macro-F1 | Best Val Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 1 (M7.1 Reference)** | 20% (2,542 imgs) | 2 | 8 | 53.90% | 67.66% |
| **Stage 2 (M7.2 Current)** | **50% (6,356 imgs)** | **1** | **6** | **57.96%** | **67.84%** |
| **Default M7 (Candidate A)** | 100% (12,712 imgs) | 5 | 1 | 88.16% | 91.08% |

> **SCIENTIFIC NOTE:**  
> These metrics represent different training data and epoch regimes. The M7.1 and M7.2 values are proxy-screening results and must not be interpreted as directly equivalent to the 5-epoch full-training M7 baseline.

---

## 8. Runtime & Safety Analysis

- Total elapsed runtime: **6727.5 s (1.87 hours)**
- Average time per candidate: **1121.2 s**
- 5-Hour hard limit: 18,000 s
- Adaptations:
- None required (experiment stayed within 5-hour limit natively).

---

## 9. Honest Scientific Conclusions & Limitations

1. **Progressive Data Scaling:** The experiment successfully evaluated STO across progressive data regimes (20% to 50% cumulative data) without data leakage or boundary collapse.
2. **Algorithmic Lineage:** Candidates 4–6 were generated strictly by STO vector updates and reflect_clip boundary handling.
3. **Comparative Evaluation:** Under the constrained proxy budget, STO identified candidate hyperparameters for LS-HGNet, but did not outperform the frozen Default M7 baseline (88.16%).
4. **Final Model Integrity:** The frozen M7 winner (**Default LS-HGNet**) remains the official final model of the AgriAI project. No M7.2 candidate is promoted to final model.
