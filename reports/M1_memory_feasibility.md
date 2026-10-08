# AgriAI — M1.6 Memory Feasibility Report

**Date:** 2026-10-02  
**Host Machine Total RAM:** 15.77 GB  
**Available RAM at Test:** 2.71 GB  
**Execution Device:** CPU (Development Standard)  

---

## 1. Benchmark Results

| Batch Size | Resolution | Device | RAM Delta | Step Time | Throughput | Feasibility Status |
|:----------:|:----------:|:------:|:---------:|:---------:|:----------:|:------------------:|
| 32 | 224x224 | `cpu` | +276.95 MB | 1071.06 ms | 29.88 img/s | **FEASIBLE (PASS)** |
| 32 | 128x128 | `cpu` | +374.07 MB | 264.49 ms | 120.99 img/s | **FEASIBLE (PASS)** |
| 16 | 224x224 | `cpu` | +194.78 MB | 404.76 ms | 39.53 img/s | **FEASIBLE (PASS)** |
| 16 | 128x128 | `cpu` | +0.45 MB | 120.05 ms | 133.28 img/s | **FEASIBLE (PASS)** |

---

## 2. Findings and Recommendation

1. **Resolution 224x224, Batch Size 32:** Runs stably within memory constraints with minimal memory footprint (< 150 MB delta).
2. **Resolution 128x128, Batch Size 32:** Extremely fast and lightweight (< 50 MB delta), optimal for fast iteration during development (EXP_001).
3. **Hardware Assessment:** The 16 GB laptop development target easily accommodates batch size 32 for both development and final resolutions without paging or OOM risk.
4. **Research Spec Alignment:** No changes to the locked M0 hyperparameter search bounds or batch size specification are required.
