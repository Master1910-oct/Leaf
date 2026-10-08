# Table 6: Out-of-Sample Generalization Metric Analysis

| Comparison Baseline | Validation Metric | Locked Test Metric | Generalization Difference | Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Full Validation (M7.3) vs Test (M9)** | **95.52%** Accuracy | **95.56%** Accuracy | **-0.04 percentage points** | Test accuracy slightly exceeds validation |
| **Full Validation (M7.3) vs Test (M9)** | **94.23%** Macro-F1 | **94.06%** Macro-F1 | **+0.17 percentage points** | Minimal Macro-F1 drop across independent splits |
| **Holdout Val (M8.2) vs Test (M9)** | **96.11%** Accuracy | **95.56%** Accuracy | **+0.55 percentage points** | Tight concordance on unseen holdout |
| **Holdout Val (M8.2) vs Test (M9)** | **95.12%** Macro-F1 | **94.06%** Macro-F1 | **+1.06 percentage points** | Controlled 1-point difference |
| **M7 Selection Baseline vs Test (M9)** | **88.16%** Macro-F1 | **94.06%** Macro-F1 | **+5.90 percentage points** | Net gain preserved on held-out test data |
| **M7 Selection Baseline vs Test (M9)** | **91.08%** Accuracy | **95.56%** Accuracy | **+4.48 percentage points** | Net accuracy gain confirmed |
