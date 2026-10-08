# M7.1 — STO REFINED PROTOCOL
## AgriAI — Corrected Sooty Tern Optimization Search Protocol

**Status:** DRAFT — Awaiting Explicit User Approval Before Execution  
**Date:** 2026-10-06  
**Reference:** STO_AUDIT.md (M7.1 findings)  
**Output directory:** `results/m7_1_sto_refinement/`  
**M7 baseline (frozen):** `results/m7_final_selection/` — NOT modified.  
**Test set:** `data/splits/test.csv` — NOT accessed at any point.

---

## 1. Objective

Run a corrected STO hyperparameter search that:
1. Addresses all Critical and High findings from STO_AUDIT.md
2. Produces a fair, reproducible result that can be scientifically compared to the M7 Default
3. Does NOT modify any M7 frozen artifacts
4. Can be used to report what the STO would have found under a correct protocol

This is a **new experimental branch** (M7.1). Even if M7.1 STO outperforms the frozen Default,
the M7 winner (Default LS-HGNet) remains the selected final model for the project.

---

## 2. Corrections vs M6

| Issue | M6 Protocol | M7.1 Corrected Protocol |
|-------|------------|------------------------|
| Image size | 128×128 | **224×224** (matches M7 full-scale) |
| Proxy epochs | 3 | **5** (matches M7 full-scale) |
| Budget | 5 agents × 5 iter = 30 evals | **10 agents × 10 iter = 100 evals** |
| Dropout boundary | Clip only (collapses to 0.5) | **Reflect boundary handler for dropout** |
| WD boundary | Clip only (collapses to 1.83e-6) | **Reflect boundary handler for WD** |
| Variant | stabilized | **stabilized** (unchanged — no code bugs) |
| Output dir | `results/m6_sto_dev/` | `results/m7_1_sto_refinement/` |

---

## 3. Boundary Handling Correction

The clip-only boundary handler in `SearchSpace.clip` causes boundary attraction. For the M7.1 run,
we will use a **reflected boundary handler** for the dropout (dim 2) and weight decay (dim 1) dimensions,
which are the two dimensions that pegged to their bounds in M6.

### Reflection Rule (per dimension):
```
if new_pos[d] > ub[d]:
    new_pos[d] = 2 * ub[d] - new_pos[d]
    if new_pos[d] < lb[d]:  # double-exceeded
        new_pos[d] = lb[d]
elif new_pos[d] < lb[d]:
    new_pos[d] = 2 * lb[d] - new_pos[d]
    if new_pos[d] > ub[d]:  # double-exceeded
        new_pos[d] = ub[d]
```

This will be implemented as `SearchSpace.reflect_clip(theta)` — a **new method added to SearchSpace**
without modifying the existing `clip` method. The `run_m71_sto.py` script will pass a flag to use this
handler. The existing `clip` method is untouched (preserving backward compatibility with all M6 code).

---

## 4. Experiment Configuration

### 4.1 Fixed Architecture (M4 Final — UNCHANGED)
| Parameter | Value |
|-----------|-------|
| base_channels | 64 |
| num_stacks | 3 |
| encoder_levels | 4 |
| num_classes | 10 |
| in_channels | 3 |

### 4.2 Search Space (UNCHANGED from M6)
| Hyperparameter | Internal Range | Space Type |
|----------------|---------------|-----------|
| Learning rate | log₁₀ ∈ [-4, -2] → [1e-4, 1e-2] | Log-scale |
| Weight decay | log₁₀ ∈ [-6, -2] → [1e-6, 1e-2] | Log-scale |
| Dropout | [0.0, 0.5] | Linear |
| Batch size | {16, 32, 64} | Discrete NN |

### 4.3 STO Parameters (CORRECTED)
| Parameter | M6 Value | M7.1 Value |
|-----------|----------|-----------|
| num_agents | 5 | **10** |
| num_iterations | 5 | **10** |
| Total evaluations | 30 | **110** (10 init + 100 main) |
| eval_epochs | 3 | **5** |
| image_size | 128 | **224** |
| variant | stabilized | **stabilized** |
| seed | 42 | **42** |
| boundary_handler | clip | **reflect_clip** |

### 4.4 Training Protocol Per Candidate (matching M7 full-scale as closely as practical)
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Scheduler | None (matching M6/M7 proxy) |
| Loss | CrossEntropyLoss |
| num_workers | 0 |
| pin_memory | False |
| drop_last | False |
| Fitness signal | 1 - validation_macro_F1 |
| Best epoch selection | max(val_macro_f1) across all epochs |

> NOTE: No LR scheduler is used in the proxy to match the M6 protocol. The Default (M7 Candidate A)
> used ReduceLROnPlateau. This remains a minor proxy-target gap but is acceptable for a proxy run.
> Eliminating this gap entirely would require 5-epoch full-224 training with scheduler, which is
> identical to M7 training time (≈3 hours per candidate × 110 candidates ≈ prohibitive).

---

## 5. Data Protocol

| Split | Source | Usage |
|-------|--------|-------|
| Training | `data/train.csv` | Fitness evaluation training only |
| Validation | `data/val.csv` | Fitness signal (val Macro-F1) |
| Test | `data/splits/test.csv` | **LOCKED — NOT accessed** |

---

## 6. Implementation Plan

### Step 1: Add `reflect_clip` to `SearchSpace`
File: `src/optimization/sto.py`  
Action: Add new method `reflect_clip(theta)` alongside existing `clip(theta)`.
- Existing `clip` is NOT modified (backward compatible).
- Only dims 1 (WD) and 2 (dropout) use reflection; dims 0 (LR) and 3 (BS) use standard clip.
  (LR and BS did not exhibit boundary collapse in M6.)

### Step 2: Patch `_BaseSTOA.optimize()` to accept boundary handler argument
File: `src/optimization/sto.py`  
Action: Add optional `boundary_fn` parameter to `optimize()` with default `None` (falls back to `clip`).
- If `boundary_fn` is provided, it replaces the `ss.clip(new_pos)` call in the main loop.
- This is a backward-compatible addition — all existing M6 code calling `.optimize(fitness_fn)` continues to work.

### Step 3: Create `scripts/run_m71_sto.py`
New script (does not overwrite `scripts/run_sto.py`).
- Hardcoded M7.1 defaults: 224px, 5 epochs, 10 agents, 10 iters, reflect_clip
- Saves ALL outputs to `results/m7_1_sto_refinement/` only
- Logs `m7_1_sto_run.log` to the same directory

### Step 4: Run the search
Command (to be executed after user approval):
```powershell
.venv\Scripts\python scripts/run_m71_sto.py `
    --variant stabilized `
    --num_agents 10 `
    --num_iterations 10 `
    --eval_epochs 5 `
    --image_size 224 `
    --output_dir results/m7_1_sto_refinement `
    --seed 42 `
    --boundary_handler reflect_clip
```

### Step 5: Analyze and report
After the run completes:
- Compare M7.1 best val Macro-F1 to the M7 Default (88.16%)
- Generate `results/m7_1_sto_refinement/M7_1_REPORT.md`
- NO test set access. NO re-running M7.

---

## 7. Expected Runtime

At 224×224, 5 epochs, CPU only:
- Candidate A (M7) training = ~3.1 hours (11,231s for 5 epochs)
- Estimated per-eval time: ~2,250s (one-epoch scaling × 5)

With 110 evaluations:
- Worst case: 110 × 2250s / 3600 ≈ **69 hours** (full 224/5-epoch proxy)

**This is unacceptable for a proxy search.**

### Recommended Compromise: Reduced Training Subset + 224px

Use a stratified 30% training subset at 224×224 for 5 epochs:
- Train size: ~3,200 images (30% of ~10,700 training images)
- Estimated per-eval: ~675s (~11.25 min)
- Total 110 evals: ~675 × 110 / 3600 ≈ **20.6 hours**

This preserves the 224×224 image size correction (critical fix) while making the budget tractable.
The 5-epoch evaluation is fully preserved. Only the training set is subsampled (stratified).

**Alternative budget option:** If 20 hours is too long, reduce to 10 agents × 5 iterations = 60 evaluations:
- Total: ~675 × 60 / 3600 ≈ **11.25 hours**

> USER DECISION REQUIRED: Select between:
> - Option A: Full train set, 224px, 5 epochs — ~69 hours (most rigorous)
> - Option B: 30% stratified subset, 224px, 5 epochs — ~20 hours (recommended)
> - Option C: 30% stratified subset, 224px, 5 epochs, 60 evals — ~11 hours (fastest)

---

## 8. Success Criteria

The M7.1 run is considered scientifically successful if:
1. The corrected STO search completes without errors
2. All 100+ candidates are logged to `results/m7_1_sto_refinement/candidate_results.json`
3. The best proxy Macro-F1 is reported
4. A comparison table (M7.1 STO proxy-best vs M7 Default vs M7 STO-old) is written to the report
5. No test set access occurred at any point

Note: M7.1 is **NOT** expected to necessarily beat the Default. Its purpose is to determine
what STO *should* have found under a fair protocol.

---

## 9. Output Artifacts (M7.1 Only)

All artifacts written exclusively to `results/m7_1_sto_refinement/`:

| File | Description |
|------|-------------|
| `STO_AUDIT.md` | This audit (already created) |
| `STO_REFINED_PROTOCOL.md` | This document (already created) |
| `m7_1_sto_run.log` | Full execution log |
| `sto_results.json` | Summary of STO run |
| `best_config.json` | Best decoded hyperparameters found |
| `candidate_results.json` | Full record of all 110 evaluated candidates |
| `fitness_history.json` | Best-so-far fitness per evaluation |
| `convergence.png` | Convergence curve plot |
| `M7_1_REPORT.md` | Final analysis report |

---

## 10. Strict Do-Not-Touch List

| Protected Asset | Why |
|----------------|-----|
| `results/m7_final_selection/` | M7 frozen model selection |
| `results/m6_sto_dev/` | M6 original STO results (audit reference) |
| `results/m8_explainability/` | M8.1 Grad-CAM outputs |
| `data/splits/test.csv` | Locked test set |
| `scripts/run_sto.py` | Original STO script (not overwritten) |

---

**WAITING FOR EXPLICIT USER APPROVAL BEFORE ANY CODE IS EXECUTED OR MODIFIED.**

*End of STO_REFINED_PROTOCOL.md*
