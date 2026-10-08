# M10 — Final Research Results Section (Manuscript Style)

## Results

### Model Convergence and Validation Performance
The Lightweight Stacked Hourglass Network (LS-HGNet; 8,041,354 parameters) was trained on the 12,712-image training split across 10 epochs using an initial learning rate of $10^{-3}$ and weight decay of $10^{-4}$. Model performance peaked at Epoch 9, attaining a validation accuracy of 95.52% and a Macro-F1 score of 94.23% (Macro Precision: 94.35%, Macro Recall: 94.37%, validation loss: 0.1262). This represents a substantial +6.07 percentage-point improvement in Macro-F1 over the 5-epoch baseline candidate (88.16% Macro-F1). The Epoch 9 checkpoint was permanently frozen for post-training audit and final locked-test evaluation.

### Post-Training Calibration and Robustness Profiling
Prior to locked-test evaluation, the model was analyzed on a 50/50 partition of the validation split (1,362 calibration samples and 1,362 holdout samples). On unperturbed holdout samples, the frozen model achieved 96.11% accuracy and 95.12% Macro-F1. Temperature scaling fitted on the calibration partition yielded an optimal temperature of $T = 1.1558$, moderating overconfident logits and shifting the Expected Calibration Error (ECE) from 1.34% to 1.73% while preserving argmax predictions. Controlled robustness testing across six image perturbations revealed resilience to moderate illumination changes (+0.14% $\Delta$F1 under contrast reduction, -2.24% under brightness increase) and JPEG compression (-2.62%), but identified sensitivity to sensor noise (-23.72% $\Delta$F1) and high-frequency edge loss via Gaussian blur (-47.82% $\Delta$F1). CPU deployment profiling established a single-image inference latency of $132.18 \pm 16.71$ ms (throughput: 7.57 images/s).

### Locked-Test Generalization
The final generalization capability of the frozen LS-HGNet checkpoint was evaluated in a single, deterministic pass on the previously untouched test dataset (2,724 images across 10 tomato leaf classes). The model achieved:
- **Test Accuracy**: **95.56%** (2,603 correct classifications out of 2,724 samples)
- **Macro-F1 Score**: **94.06%**
- **Weighted-F1 Score**: **95.55%**
- **Macro Precision**: **94.00%**
- **Macro Recall**: **94.43%**

Comparing the locked-test results against full validation performance reveals an exceptionally tight generalization gap: test accuracy exceeded validation accuracy by 0.04 percentage points (95.56% vs 95.52%), while Macro-F1 showed a minor decline of only 0.17 percentage points (94.06% vs 94.23%). Relative to the original frozen M7 baseline (88.16% Macro-F1), the final model demonstrates a +5.90 percentage-point net gain on independent test data.

### Per-Class Characteristics and Error Dynamics
Performance across individual pathologies remained high, with seven of the ten classes exceeding 95% F1-score. The highest scores were observed in *Tomato Yellow Leaf Curl Virus* (F1: 99.12%, support: 804), *Healthy leaves* (F1: 98.94%, precision: 100.0%, support: 238), and *Septoria leaf spot* (F1: 96.30%, support: 265). The lowest scores were recorded for *Target Spot* (F1: 86.78%, recall: 82.46%, support: 211) and *Early Blight* (F1: 90.14%, recall: 85.33%, support: 150).

Confusion matrix analysis indicated that the primary failure mode was the misclassification of *Target Spot* as *Two-Spotted Spider Mites* (29 instances, representing 13.74% of Target Spot samples and 24.0% of all test errors), driven by visual similarity between diffuse pinpoint lesions and severe mite stippling. Misclassifications between *Early Blight* and *Late Blight* accounted for 9 errors. Prediction confidence exhibited sharp discrimination between correct predictions (mean confidence: 96.97%, median: 99.95%) and misclassifications (mean confidence: 70.65%, median: 69.10%).
