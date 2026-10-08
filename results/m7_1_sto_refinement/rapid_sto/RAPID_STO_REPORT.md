# M7.1 — RAPID STO SCREENING EXPERIMENT REPORT

**Milestone:** M7.1 — STO Audit & Rapid Refinement  
**Project:** AgriAI — Tomato Leaf Disease Classification with LS-HGNet  
**Date:** 2026-10-06 20:03:33 UTC  
**Status:** COMPLETED (Hard 5-Hour CPU Limit Enforced)  
**Total Runtime:** 13872.7 s (3.85 hours)  
**Total Candidates Evaluated:** 8  

---

## 1. Executive Summary & Objective

The objective of M7.1 is to evaluate Sooty Tern Optimization (STO) as a hyperparameter search algorithm for LS-HGNet under corrected experimental settings, investigating whether STO can identify competitive hyperparameter configurations relative to the frozen M7 Default LS-HGNet baseline.

Following the STO audit (`results/m7_1_sto_refinement/STO_AUDIT.md`), the full 110-candidate x 5-epoch search protocol was tested. However, empirically observed CPU runtime on Candidate 1 was **1251.9 seconds (~20.9 minutes) per epoch**, which would have required **~190 hours (~8 days)** of continuous computation on this laptop, making the unconstrained protocol physically infeasible.

To respect the available computational resources while strictly preserving scientific rigor, the search was redesigned as a **CPU-bounded Rapid STO Screening Experiment** with a hard constraint of **5 hours maximum**.

---

## 2. Experimental Protocol & Budget

| Parameter | Previous Protocol (Interrupted) | Rapid STO Screening Protocol |
| :--- | :--- | :--- |
| **Search Algorithm** | Stabilized STO (`reflect_clip`) | **Stabilized STO (`reflect_clip`)** |
| **Population Size (N)** | 10 agents | **4 agents** |
| **STO Iterations (T)** | 10 iterations | **2 iterations** (Iter 0: init; Iter 1: update) |
| **Max Candidate Budget** | 110 candidates | **8 candidates** |
| **Image Resolution** | 224x224 | **224x224** |
| **Epochs per Candidate** | 5 epochs | **2 epochs** |
| **Batch Size** | 32 | **32** (fixed) |
| **Training Dataset** | 30% stratified (3,814 imgs) | **20% stratified (2,542 imgs, seed=42)** |
| **Validation Dataset** | Full (2,724 imgs) | **Full (2,724 imgs, never subsampled)** |
| **Fitness Function** | 1.0 - Validation Macro-F1 | **1.0 - Validation Macro-F1** |
| **Hard Time Limit** | None (crashed on laptop) | **5.0 Hours (18,000 s)** |

---

## 3. Search Space & Baseline Verification

The search space bounds:
- **Learning Rate:** $\log_{10}(\text{LR}) \in [-4.0, -2.0] \implies [1.0\times 10^{-4}, 1.0\times 10^{-2}]$
- **Weight Decay:** $\log_{10}(\text{WD}) \in [-6.0, -2.0] \implies [1.0\times 10^{-6}, 1.0\times 10^{-2}]$
- **Dropout:** $[0.0, 0.5]$ (linear)
- **Batch Size:** Fixed at 32

**Baseline Verification:**
The frozen M7 Candidate A baseline (LR=0.001, WD=0.0001, Dropout=0.30, BS=32) corresponds to internal vector `[-3.0, -4.0, 0.30, 1.0]`, which lies strictly inside the search bounds `lb=[-4.0, -6.0, 0.0, 1.0]` and `ub=[-2.0, -2.0, 0.5, 1.0]`.

---

## 4. STO Algorithmic Implementation

The optimization loop executed genuine STO updates:
1. **Initial Population (Iteration 0):** 4 candidate positions were sampled within search bounds using PRNG seed 42 and evaluated.
2. **Global Best Identification:** The agent with the minimum fitness ($1 - \text{val\_macro\_f1}$) was tracked.
3. **Exploration & Spiral Update (Iteration 1):**
   $$\begin{aligned}
   S_A &= 2.0 \times \left(1 - \frac{t}{T}\right) \\
   C_{sa} &= S_A \times P_i(t) \\
   M_s &= 0.5 \times \vec{r} \odot (P_{best} - P_i(t)) \\
   D_s &= C_{sa} + M_s \\
   P_i(t+1) &= P_{best} + D_s \odot \vec{\Omega}_4
   \end{aligned}$$
4. **Boundary Handling:** `reflect_clip` reflected boundary-overshooting parameters back into the feasible domain, resolving the boundary pegging collapse documented in M6.

---

## 5. Candidate Evaluation Results

| Cand ID | Stage / Agent | LR | WD | Dropout | BS | Val Acc | Val Macro-F1 | Fitness | Global Best F1 | Runtime |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  1 | Iter 0 / A1 | 3.53e-03 | 5.70e-05 | 0.429 | 32 | 38.22% | **20.97%** | 0.7903 | 20.97% | 1800.6s |
|  2 | Iter 0 / A2 | 1.54e-04 | 7.99e-03 | 0.381 | 32 | 64.02% | **47.32%** | 0.5268 | 47.32% | 1837.3s |
|  3 | Iter 0 / A3 | 1.80e-04 | 6.33e-05 | 0.185 | 32 | 63.80% | **52.63%** | 0.4737 | 52.63% | 1851.5s |
|  4 | Iter 0 / A4 | 1.94e-03 | 1.95e-03 | 0.222 | 32 | 61.42% | **50.27%** | 0.4973 | 52.63% | 1783.1s |
|  5 | Iter 1 / A1 | 1.69e-04 | 4.70e-04 | 0.493 | 32 | 67.66% | **53.90%** | 0.4610 | 53.90% | 1810.4s |
|  6 | Iter 1 / A2 | 3.66e-04 | 1.13e-06 | 0.413 | 32 | 63.18% | **45.84%** | 0.5416 | 53.90% | 1805.1s |
|  7 | Iter 1 / A3 | 1.91e-04 | 4.77e-04 | 0.367 | 32 | 60.06% | **45.66%** | 0.5434 | 53.90% | 1769.7s |
|  8 | Iter 1 / A4 | 3.24e-04 | 1.09e-06 | 0.350 | 32 | 60.98% | **50.37%** | 0.4963 | 53.90% | 1208.4s |

---

## 6. Comparison Against Frozen M7 Default Baseline

| Method | LR | WD | Dropout | Epochs | Train Subset | Val Macro-F1 | Val Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default M7 (Candidate A)** | **0.0010** | **0.0001** | **0.30** | **5** | **100% (12,712 imgs)** | **88.16%** | **91.08%** |
| **Rapid STO Best (Candidate 5)** | **1.69e-04** | **4.70e-04** | **0.493** | **2** | **20% (2,542 imgs)** | **53.90%** | **67.66%** |

> **IMPORTANT SCIENTIFIC NOTE:**  
> These two scores MUST NOT be compared as if produced under identical training budgets. The Default M7 baseline was trained for 5 full epochs on 100% of the training data (12,712 images) with learning rate scheduling. The Rapid STO screening was trained for only 2 epochs on a 20% proxy subset (2,542 images) without scheduling to fit the 5-hour CPU constraint.

---

## 7. Runtime & Budget Safety Log

- Total elapsed time: **13872.7 s (3.85 hours)**
- Average time per candidate: **1734.1 s (28.9 minutes)**
- Hard limit: 18,000.0 s (5.0 hours)
- Dynamic adaptations:
- None required (experiment stayed within 5-hour limit natively).

---

## 8. Honest Scientific Conclusions & Limitations

1. **Proxy Screening Feasibility:** Rapid STO screening successfully demonstrated that STO can operate within realistic computational boundaries on standard hardware when properly configured with proxy training budgets and boundary reflection.
2. **Algorithm Execution:** STO successfully updated candidate positions toward the global best across iterations rather than degenerating or collapsing.
3. **Performance vs Baseline:** Best Rapid STO validation Macro-F1 was **53.90%** (vs Default baseline **88.16%**). Under the highly constrained 2-epoch / 20% data proxy regime, the STO screening did not identify a configuration that exceeded the full-training Default baseline.
4. **Honest Claim Policy:** We do NOT claim that STO outperformed the default model or produced a superior model. The frozen M7 winner (**Default LS-HGNet**) remains the selected model for the project, and its checkpoints and test metrics remain completely frozen and untouched.
