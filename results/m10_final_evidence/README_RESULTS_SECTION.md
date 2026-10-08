# Final Experimental Results (README Summary)

The **AgriAI LS-HGNet** (Lightweight Stacked Hourglass Network) has completed all experimental and evaluation milestones, culminating in the **M9 Final Locked-Test Evaluation** on the previously untouched test dataset.

### Key Performance Highlights

| Metric | Final Locked Test (M9) | Validation (M7.3) | Baseline (M7) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **95.56%** | 95.52% | 91.08% |
| **Macro-F1** | **94.06%** | 94.23% | 88.16% |
| **Weighted-F1** | **95.55%** | 95.49% | — |
| **Macro Precision** | **94.00%** | 94.35% | — |
| **Macro Recall** | **94.43%** | 94.37% | — |
| **Parameters** | **8,041,354** | 8,041,354 | 8,041,354 |
| **CPU Latency (batch=1)** | **132.2 ms** | — | — |

- **Generalization Gap**: Minimal 0.17 percentage-point drop in Macro-F1 between validation (94.23%) and locked test (94.06%).
- **Top Performing Classes**:
  - *Tomato Yellow Leaf Curl Virus*: **99.12% F1**
  - *Healthy Leaf*: **98.94% F1** (100.0% Precision)
  - *Septoria Leaf Spot*: **96.30% F1**
- **Challenging Classes**:
  - *Target Spot*: **86.78% F1** (confused with Spider Mites)
  - *Early Blight*: **90.14% F1** (confused with Late Blight)

### Reproducibility & Artifact Links
- **Master Audit Report**: [`results/m10_final_evidence/M10_FINAL_REPORT.md`](results/m10_final_evidence/M10_FINAL_REPORT.md)
- **Frozen Model Checkpoint**: [`results/m7_3_full_training/checkpoints/best_model.pt`](results/m7_3_full_training/checkpoints/best_model.pt) (SHA-256: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`)
- **Test Metrics & Predictions**: [`results/m9_final_locked_test/metrics/`](results/m9_final_locked_test/metrics/)
- **Normalized Confusion Matrix**: [`results/m9_final_locked_test/plots/confusion_matrix_normalized.png`](results/m9_final_locked_test/plots/confusion_matrix_normalized.png)
