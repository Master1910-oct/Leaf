# M10 — Final Performance Table

**Consolidated Performance Summary Across All Verified Milestones**

| Milestone / Stage | Evaluated Split | Sample Count | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Macro Precision (%) | Macro Recall (%) | Supporting Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M7 Frozen Baseline** | Validation (100%) | 2,724 | 91.08% | 88.16% | — | — | — | `results/m7_final_selection/` |
| **M7.3 Convergence (Epoch 9)** | Validation (100%) | 2,724 | 95.52% | 94.23% | 95.49% | 94.35% | 94.37% | `results/m7_3_full_training/metrics/metrics.json` |
| **M8.2 Clean Holdout** | Holdout Val (50%) | 1,362 | 96.11% | 95.12% | — | 95.31% | 95.18% | `results/m8_2_calibration_robustness/` |
| **M9 Final Locked Test** | **Locked Test (100%)** | **2,724** | **95.56%** | **94.06%** | **95.55%** | **94.00%** | **94.43%** | `results/m9_final_locked_test/metrics/test_metrics.json` |

*Note: Missing values (—) indicate metrics that were not part of the historical evaluation protocol for that specific stage. No values have been manufactured.*
