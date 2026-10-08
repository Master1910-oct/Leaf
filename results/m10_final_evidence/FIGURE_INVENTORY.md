# M10 — Final Figure Inventory

**Curated Catalog of Verified Plots and Visual Artifacts Across Milestones**

| Figure File | Stage | Purpose / Description | Suitable for Paper | Suitable for Presentation |
| :--- | :---: | :--- | :---: | :---: |
| `results/m7_3_full_training/plots/convergence_curve.png` | M7.3 | 10-epoch training/val loss and validation Macro-F1 convergence trajectory | Yes (Fig. 1) | Yes |
| `results/m7_3_full_training/plots/confusion_matrix.png` | M7.3 | Normalized validation confusion matrix heatmap (Epoch 9) | Reference | Reference |
| `results/m8_2_calibration_robustness/plots/reliability_before.png` | M8.2 | Reliability diagram before Temperature Scaling ($T=1.0$, ECE=1.34%) | Yes (Fig. 2A) | Yes |
| `results/m8_2_calibration_robustness/plots/reliability_after.png` | M8.2 | Reliability diagram after Temperature Scaling ($T=1.1558$, ECE=1.73%) | Yes (Fig. 2B) | Yes |
| `results/m8_2_calibration_robustness/plots/robustness_macro_f1.png` | M8.2 | Macro-F1 degradation under 6 image perturbations | Yes (Fig. 3) | Yes |
| `results/m8_2_calibration_robustness/plots/robustness_accuracy.png` | M8.2 | Accuracy degradation across 6 image perturbations | Appendix | Reference |
| `results/m8_2_calibration_robustness/plots/inference_latency_distribution.png` | M8.2 | CPU latency distribution (200 runs, mean=132.2 ms) | Yes (Fig. 4) | Yes |
| `results/m9_final_locked_test/plots/confusion_matrix.png` | M9 | Final locked-test confusion matrix with raw error counts | Appendix | Yes |
| `results/m9_final_locked_test/plots/confusion_matrix_normalized.png` | M9 | Final locked-test normalized confusion matrix | **Yes (Main Fig)** | **Yes (Key)** |
| `results/m9_final_locked_test/plots/confidence_distribution.png` | M9 | Distribution of prediction confidence (Correct vs Incorrect) | **Yes (Main Fig)** | **Yes (Key)** |

*Recommendation*: For publication, the primary figures should be:
1. `convergence_curve.png` (Training trajectory)
2. `confusion_matrix_normalized.png` (Final test error structure)
3. `confidence_distribution.png` (Prediction confidence separation)
4. `robustness_macro_f1.png` (Operational robustness limits)
