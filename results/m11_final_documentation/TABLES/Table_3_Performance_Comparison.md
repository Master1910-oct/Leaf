# Table 3: Multi-Stage Experimental Performance Comparison

| Milestone / Evaluation Phase | Evaluated Dataset Split | Sample Count | Overall Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | Macro Precision (%) | Macro Recall (%) | Verified Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **M7 Selection Baseline** | Validation (100%) | 2,724 | 91.08% | 88.16% | — | — | — | `results/m7_final_selection/` |
| **M7.3 Convergence (Epoch 9)** | Validation (100%) | 2,724 | 95.52% | 94.23% | 95.49% | 94.35% | 94.37% | `results/m7_3_full_training/metrics/` |
| **M8.2 Holdout Evaluation** | Holdout Val (50%) | 1,362 | 96.11% | 95.12% | — | 95.31% | 95.18% | `results/m8_2_calibration_robustness/` |
| **M9 Final Locked Test** | **Locked Test (100%)** | **2,724** | **95.56%** | **94.06%** | **95.55%** | **94.00%** | **94.43%** | `results/m9_final_locked_test/metrics/` |

*Note: All values reflect strictly evaluated artifacts. Em-dashes (—) indicate metrics that were not part of the historical logging schema for that milestone.*
