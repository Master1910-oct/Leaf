# M11 — Paper-Ready Figure Inventory and Captions

This catalog documents the verified figures copied into `results/m11_final_documentation/FIGURES/`.

---

### Figure 1: Convergence Trajectory of LS-HGNet
- **File**: `FIGURES/Figure_1_Convergence_Trajectory.png`
- **Source Artifact**: `results/m7_3_full_training/plots/convergence_curve.png`
- **Purpose**: Illustrates 10-epoch training and validation loss curves alongside validation Macro-F1.
- **Suggested Paper Caption**:  
  *Figure 1. Convergence trajectory of the Lightweight Stacked Hourglass Network (LS-HGNet) across 10 training epochs on 12,712 tomato leaf images. The validation Macro-F1 reached its peak of 94.23% at Epoch 9 (validation loss: 0.1262), which was selected as the frozen model checkpoint.*

---

### Figure 2: Confidence Calibration Reliability Diagrams
- **Files**:
  - `FIGURES/Figure_2A_Reliability_Before_Calibration.png` (Before Calibration, $T=1.0$)
  - `FIGURES/Figure_2B_Reliability_After_Calibration.png` (After Calibration, $T=1.1558$)
- **Source Artifacts**: `results/m8_2_calibration_robustness/plots/reliability_before.png`, `reliability_after.png`
- **Purpose**: Demonstrates empirical accuracy versus prediction confidence across 15 calibration bins.
- **Suggested Paper Caption**:  
  *Figure 2. Reliability diagrams on the 1,362-sample validation holdout set before (A) and after (B) Temperature Scaling ($T = 1.1558$). The raw model demonstrates strong initial calibration ($	ext{ECE} = 1.34\%$), with post-hoc scaling moderating overconfident predictions ($	ext{ECE} = 1.73\%$, mean confidence reduced from 95.87% to 94.89%) while strictly preserving classification rankings.*

---

### Figure 3: Model Robustness under Controlled Image Degradations
- **File**: `FIGURES/Figure_3_Robustness_Perturbation_Analysis.png`
- **Source Artifact**: `results/m8_2_calibration_robustness/plots/robustness_macro_f1.png`
- **Purpose**: Analyzes Macro-F1 sensitivity against six controlled synthetic perturbations.
- **Suggested Paper Caption**:  
  *Figure 3. Macro-F1 performance of the frozen LS-HGNet model across six synthetic image degradation conditions on the validation holdout split. The model retains high stability under contrast variation (+0.14%), brightness shifts (-2.24%), and JPEG compression (-2.62%), but exhibits marked sensitivity to sensor noise (-23.72%) and Gaussian blur (-47.82%).*

---

### Figure 4: CPU Inference Latency Distribution
- **File**: `FIGURES/Figure_4_CPU_Inference_Latency.png`
- **Source Artifact**: `results/m8_2_calibration_robustness/plots/inference_latency_distribution.png`
- **Purpose**: Profiles real-world unbatched CPU execution latency.
- **Suggested Paper Caption**:  
  *Figure 4. Single-image CPU inference latency distribution over 200 measured runs (input resolution: $224 \times 224$, batch size: 1). The model achieves a mean latency of $132.18 \pm 16.71$ ms (median: 128.16 ms, P95: 171.77 ms, throughput: 7.57 images/s).*

---

### Figure 5: Normalized Confusion Matrix on Locked Test Dataset
- **File**: `FIGURES/Figure_5_Normalized_Confusion_Matrix.png`
- **Source Artifact**: `results/m9_final_locked_test/plots/confusion_matrix_normalized.png`
- **Purpose**: Core multi-class evaluation figure illustrating diagonal classification rates.
- **Suggested Paper Caption**:  
  *Figure 5. Normalized confusion matrix of the frozen LS-HGNet evaluated on the locked test dataset (2,724 images across 10 classes). Diagonal values represent class recalls. The model exhibits high sensitivity across major classes (97.2%–98.6%), with primary confusion localized between Target Spot and Spider Mites (13.74% error rate).*

---

### Figure 6: Prediction Confidence Distribution (Correct vs. Incorrect)
- **File**: `FIGURES/Figure_6_Confidence_Distribution.png`
- **Source Artifact**: `results/m9_final_locked_test/plots/confidence_distribution.png`
- **Purpose**: Demonstrates uncertainty separation between accurate predictions and model errors.
- **Suggested Paper Caption**:  
  *Figure 6. Softmax prediction confidence distribution for correct predictions ($n = 2,603$, green) versus incorrect predictions ($n = 121$, red) on the locked test partition. Accurate classifications exhibit a mean confidence of 96.97% (median: 99.95%), while misclassifications display marked uncertainty with a mean confidence of 70.65% (median: 69.10%).*
