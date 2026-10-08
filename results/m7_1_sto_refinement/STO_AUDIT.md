# M7.1 — STO AUDIT REPORT
## AgriAI — Sooty Tern Optimization: Implementation & Protocol Audit

**Status:** DRAFT — Awaiting Approval Before Execution  
**Date:** 2026-10-06  
**Auditor:** Antigravity AI (AgriAI Research Assistant)  
**Scope:** `src/optimization/sto.py`, `src/optimization/fitness.py`, `scripts/run_sto.py`, `results/m6_sto_dev/`  
**Protected:** `results/m7_final_selection/` — NOT modified.  
**Test set:** `data/splits/test.csv` — NOT accessed.

---

## 1. Executive Summary

The M6 STO proxy search found a best-in-proxy F1 of **24.4%** (image_size=128, 3 epochs). When the
best candidate (LR≈0.005, WD≈1.83e-6, dropout=0.5, BS=32) was retrained at full scale (M7 Candidate B),
it achieved a validation Macro-F1 of **76.91%**, compared to **88.16%** for the Default (Candidate A).

This audit identifies **seven distinct root causes** that collectively explain this 11.25 pp gap. No single
cause alone explains the underperformance; all seven are present and contributing.

---

## 2. Audit Findings

### FINDING 1 — CRITICAL: Proxy Training Used Wrong Image Size

**Severity:** Critical  
**Evidence:** `results/m6_sto_dev/config.yaml`, line 3: `image_size: 128`  
**Default config:** image_size = 224

**Description:**  
The M6 STO proxy trained all 30 candidates at 128×128 pixels. The M7 full-scale training used 224×224 pixels.
This is a 3.06× difference in spatial resolution. The M4-final LS-HGNet architecture (3-stack hourglass, 4
encoder levels, base_channels=64) was designed and validated for 224×224 inputs. At 128×128, the deep
encoder/decoder stack has dramatically fewer spatial features at each pyramid level, causing the proxy fitness
landscape to be fundamentally different from the full-scale landscape.

**Impact:** A hyperparameter configuration that looks best at 128×128 may not be best at 224×224. In particular,
high dropout (0.5) and high LR (~0.005) may suppress underfitting at the small resolution but cause instability at
the larger resolution where the model has more capacity to overfit differently.

**Status:** Confirmed protocol flaw.

---

### FINDING 2 — CRITICAL: Proxy Used Only 3 Training Epochs

**Severity:** Critical  
**Evidence:** `results/m6_sto_dev/config.yaml`, line 1: `eval_epochs: 3`  
**M7 full training:** 5 epochs

**Description:**  
The M6 proxy trained each candidate for only 3 epochs. The M7 final training used 5 epochs. With 3 epochs, the
fitness landscape captures early-stage loss curves. High LR configurations that initially converge fast at 3 epochs
may diverge or plateau at 5 epochs. Low dropout configurations may not yet reveal their regularization advantage.

Combined with the image_size mismatch (Finding 1), the proxy effectively optimized for a (128px, 3-epoch)
world that does not transfer to the (224px, 5-epoch) world used for final evaluation.

**Status:** Confirmed protocol flaw.

---

### FINDING 3 — HIGH: STO Convergence Collapsed to a Narrow Region (LR≈0.005, Dropout=0.5)

**Severity:** High  
**Evidence:** `results/m6_sto_dev/candidate_results.json` — Candidates 19 through 30 (the last 12 evaluations)
all have theta_dropout=0.5 (boundary-pegged) and theta_lr in [-2.31, -2.29] (very narrow LR band).

| Candidate | LR        | Dropout | Proxy F1 |
|-----------|-----------|---------|----------|
| 19        | 0.004978  | 0.500   | 24.40%   |
| 20        | 0.004947  | 0.500   | ~9.6%    |
| 27        | 0.004958  | 0.500   | ~14.1%   |
| 28        | 0.004947  | 0.500   | ~19.4%   |
| 29        | 0.005005  | 0.500   | ~21.2%   |
| 30        | 0.005119  | 0.500   | ~8.3%    |

**Description:**  
The STO optimizer converged to a boundary-constrained degenerate solution. The search space has
`theta_3 in [0.0, 0.5]` for dropout. The optimizer's position update (`P(t+1) = P_best + D_s * Omega`)
caused dropout to lock to the upper bound (0.5) after iteration 3, and LR to cluster around log10^-2.3 ≈ 0.005.

The convergence curve (`sto_results.json`) shows the best fitness plateauing after iteration 3/5, with zero
improvement in iterations 4 and 5. This indicates **premature convergence** with no further exploration.

**Status:** Confirmed STO behavior flaw — premature convergence to boundary.

---

### FINDING 4 — HIGH: Position Update Creates Attraction to Search Space Boundary

**Severity:** High  
**Evidence:** `src/optimization/sto.py`, lines 427–441 (`_position_update`)

```python
# Additive update:
return best_pos + D_s * omega
```

Where `D_s = S_A * pos + 0.5 * r_vec * (best_pos - pos)`.

**Description:**  
The stabilized STO additive update `P_new = P_best + D_s * Omega` has a structural bias. When `P_best` is near a
boundary (as it was after iteration 3, with dropout=0.5 at the upper bound), `D_s` contains a term
`S_A * P_best` which is a positive scalar times the already-boundary-pegged position. This causes the new
position to be pushed further beyond the boundary. The subsequent clip operation then anchors all agents to
the boundary, and the process reinforces itself each iteration.

This is a known limitation of additive STO variants when the best solution is near a boundary: the swarm
collapses to the boundary and stops exploring.

**Status:** Confirmed structural algorithmic issue.

---

### FINDING 5 — MEDIUM: Weight Decay Also Pegged to Lower Boundary (WD ≈ 1.83e-6)

**Severity:** Medium  
**Evidence:** `results/m6_sto_dev/best_config.json`: `weight_decay: 1.8342e-06`  
**Search space WD lower bound:** 1e-6 (log theta_2 = -6.0)

**Description:**  
The STO-selected WD (1.83e-6) is effectively at the lower boundary of the WD search space (1e-6).
The best position theta_2 = -5.737 corresponds to WD=1.83e-6, very close to the lb of -6.0.
The Default WD is 1e-4, which is 54× larger. With high dropout=0.5 and near-zero WD=1.83e-6,
the proxy-selected configuration uses aggressive stochastic regularization but almost no L2 regularization
— a qualitatively different regime from the Default.

**Status:** Confirmed boundary pegging on WD as well.

---

### FINDING 6 — MEDIUM: Fitness Caching Had No Practical Effect

**Severity:** Medium  
**Evidence:** `src/optimization/fitness.py`, lines 194–237; `sto_results.json` all_fitnesses = 30 unique values.

**Description:**  
The cache key is based on decoded discrete HP values. Although caching was enabled, all 30 evaluations
returned unique theta vectors that decoded to unique (LR, WD, dropout, BS) tuples. No cache hits occurred.
Caching was ineffective in this run because the agent trajectories explored enough LR/WD variation to
always miss previously cached points (even while converging on a small region).

**Status:** Confirmed — caching neither helped nor harmed.

---

### FINDING 7 — LOW: Total Proxy Budget Was Insufficient for Reliable Ranking

**Severity:** Low  
**Evidence:** `results/m6_sto_dev/config.yaml`: `num_agents: 5`, `num_iterations: 5` → 30 evaluations total.

**Description:**  
With only 30 evaluations in a 4D continuous search space, the coverage is statistically sparse.
The Default hyperparameters (LR=1e-3, WD=1e-4, dropout=0.3, BS=32) map to theta = (-3.0, -4.0, 0.3, 1.0).
This point was **never explored** in the 30 evaluations, meaning the STO was unable to compete with a
hand-crafted baseline that it never directly sampled.

**Status:** Confirmed budget concern.

---

## 3. Root Cause Classification

| # | Finding | Root Cause Category | Severity |
|---|---------|---------------------|----------|
| 1 | Wrong image size in proxy (128 vs 224) | Protocol flaw | **Critical** |
| 2 | Too few proxy epochs (3 vs 5) | Protocol flaw | **Critical** |
| 3 | Premature convergence to dropout=0.5 boundary | STO behavior | **High** |
| 4 | Additive update boundary attraction | Algorithm structure | **High** |
| 5 | WD also pegged to lower boundary (1.83e-6) | Algorithm + search space | **Medium** |
| 6 | Caching had no practical effect | Implementation | **Low** |
| 7 | Proxy budget too small (30 evals) | Resource constraint | **Low** |

---

## 4. Code Correctness Assessment

The STO implementation is **algorithmically correct** relative to the documented specification
(`research_spec.md` Section 13.5, `src/optimization/sto.py` file header). Specifically:

- `OriginalStrictSTO._spiral_direction`: Correctly reproduces the Dhiman & Kaur (2019) scalar-broadcast.
- `ProjectSpecificStabilizedSTO._spiral_direction`: Correctly implements damped spiral (b=0.1), L2-normalizes
  the 3D vector, and cyclically extends to 4D.
- `ProjectSpecificStabilizedSTO._position_update`: Correctly implements the additive form (`P_best + D_s * Omega`).
- `SearchSpace.decode`: Correctly applies log-space decode for LR and WD, linear for dropout, NN-rounding for BS.
- `FitnessEvaluator.__call__`: Correctly uses validation split only; test set never accessed.
- Boundary handling (`clip`): Correctly clips to `[lb, ub]` after each position update.

**Conclusion:** There are **no bugs in the implementation**. The underperformance is entirely caused by
protocol flaws (Findings 1–2) and the STO algorithm's structural convergence behavior in the presence
of a near-boundary optimum (Findings 3–5).

---

## 5. What Did the M6 STO Actually Optimize?

The M6 STO effectively solved a *different problem*:

> **"Which LR/WD/dropout/BS gives the best validation Macro-F1 on 128×128 images after 3 training epochs?"**

The M7 final training solved:

> **"Which LR/WD/dropout/BS gives the best validation Macro-F1 on 224×224 images after 5 training epochs?"**

These two optimization problems are related but not identical. The transfer failure is expected
given the large proxy-target discrepancy.

---

## 6. Proposed Corrections (Summary)

Full details in `STO_REFINED_PROTOCOL.md`. In summary:

1. **Fix proxy image size** → 224×224 (matching M7 full-scale)
2. **Fix proxy epochs** → 5 epochs (matching M7 full-scale)
3. **Add reflected boundary handling** for dropout and WD to prevent swarm collapse
4. **Increase budget** → 10 agents × 10 iterations = 100 evaluations minimum
5. **Save all M7.1 outputs** only to `results/m7_1_sto_refinement/`

---

## 7. Protected Assets Checklist

| Asset | Status |
|-------|--------|
| `results/m7_final_selection/` | Not touched |
| `results/m6_sto_dev/` | Read-only (audit only) |
| `data/splits/test.csv` | Not accessed |
| `src/optimization/sto.py` | Not modified |
| `src/optimization/fitness.py` | Not modified |

---

*End of STO_AUDIT.md*
