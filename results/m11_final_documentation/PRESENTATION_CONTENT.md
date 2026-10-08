# M11 — Slide-by-Slide Technical Presentation Guide

**Target Audience**: Academic Reviewers, Department Faculty, Engineering Examiners  
**Duration**: 15–20 Minutes (18 Slides)

---

### Slide 1 — Title Slide
- **Title**: Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model
- **Subtitle**: Automated Diagnosis and Robust Evaluation of Tomato Leaf Pathologies
- **Presenter**: Engineering Project Team
- **Key Visual**: Sample multi-class tomato leaf images overlaid with LS-HGNet network schematic.
- **Presenter Notes**: *"Good morning respected members of the panel. Today we present our project on automated tomato leaf disease classification utilizing an adapted Lightweight Stacked Hourglass Network with empirical validation and strict data governance."*

---

### Slide 2 — Problem Statement
- **Points**:
  - Tomato is a vital cash crop susceptible to over 9 destructive foliar diseases.
  - Manual diagnosis requires agricultural pathology expertise and is slow, subjective, and error-prone.
  - Existing CNN models are computationally heavy and often lack rigorous generalization audits.
- **Key Visual**: Side-by-side comparison of visually ambiguous diseases (Target Spot vs Spider Mites).
- **Presenter Notes**: *"Foliar diseases cause severe global crop loss. Visual symptoms often share subtle overlapping features, making early detection critical yet challenging for farmers."*

---

### Slide 3 — Motivation & Objectives
- **Points**:
  - Adapt a multi-scale hourglass architecture to capture both macroscopic leaf lesions and microscopic fungal spots.
  - Benchmark metaheuristic hyperparameter optimization (Sooty Tern Optimization).
  - Enforce zero-leakage data isolation and locked test evaluation.
  - Profile model robustness under real-world image degradations.
- **Presenter Notes**: *"Our core motivation is to move beyond standard black-box classifiers to a calibrated, robustly profiled, and rigorously audited deep learning pipeline."*

---

### Slide 4 — Existing Systems vs. Proposed System
- **Points**:
  - *Existing*: Heavy backbones (ResNet-50, VGG-16) with >25M parameters; arbitrary hyperparameter tuning; evaluations lacking locked test protocols.
  - *Proposed*: LS-HGNet with 8.04M parameters; multi-dilation bottleneck design; structured multi-milestone evaluation from baseline (M7) to locked test (M9).
- **Table**: Existing vs Proposed feature matrix.
- **Presenter Notes**: *"We replace oversized architectures with a multi-scale hourglass network tailored for spatial multi-dilation feature reuse with under 8.1M parameters."*

---

### Slide 5 — Dataset & Stratification
- **Points**:
  - 18,160 curated tomato leaf images across 10 classes (9 diseases + healthy).
  - 70% Train (12,712), 15% Validation (2,724), 15% Locked Test (2,724).
  - Cryptographically confirmed zero sample overlap across all splits.
- **Table**: `TABLES/Table_1_Dataset_Distribution.md`
- **Presenter Notes**: *"We partitioned the 18,160 images into a stratified 70/15/15 split. The test set was locked prior to experiment start to prevent data leakage."*

---

### Slide 6 — LS-HGNet Architecture
- **Points**:
  - Entry Stem: $3 \times 3$ conv expanding to 64 channels.
  - 3 Recurrent Hourglass Stacks with intermediate supervision.
  - 4 Downsampling/Upsampling encoder levels per stack.
  - Multi-Dilation Bottlenecks: Dilation rates $(1, 2, 4)$ expanding receptive field coverage.
- **Presenter Notes**: *"The core innovation is our multi-dilation bottleneck within stacked hourglass modules, enabling the network to capture pinpoint fungal specks and broad blights simultaneously."*

---

### Slide 7 — Optimization & STO Integration
- **Points**:
  - Explored Sooty Tern Optimization (STO) for global hyperparameter exploration.
  - Tuned learning rate, weight decay, and dropout spaces on validation proxies.
  - Candidate A selected at M7; full convergence trained with AdamW and Cosine Annealing.
- **Presenter Notes**: *"We explored bio-inspired STO for hyperparameter search. We explicitly note that STO guided search space exploration, while AdamW updated weights during training."*

---

### Slide 8 — Full Training Convergence (Milestone M7.3)
- **Points**:
  - 10-epoch schedule on 100% training data (12,712 images).
  - Validation Accuracy reached **95.52%**; Validation Macro-F1 reached **94.23%**.
  - Peak attained at Epoch 9; checkpoint frozen (`best_model.pt`, SHA-256 verified).
- **Key Visual**: `FIGURES/Figure_1_Convergence_Trajectory.png`
- **Presenter Notes**: *"At Epoch 9, our validation Macro-F1 peaked at 94.23%, improving over our 5-epoch baseline candidate by over 6.0 percentage points. This checkpoint was frozen."*

---

### Slide 9 — Calibration & Reliability (Milestone M8.2)
- **Points**:
  - Temperature Scaling applied on validation calibration split ($T = 1.1558$).
  - Expected Calibration Error (ECE) held at a low 1.73%.
  - Moderated overconfident softmax probabilities while preserving exact classification decisions.
- **Key Visual**: `FIGURES/Figure_2B_Reliability_After_Calibration.png`
- **Presenter Notes**: *"Modern neural networks often exhibit overconfidence. Post-hoc temperature scaling adjusted our probabilities, ensuring calibrated uncertainty for agricultural decisions."*

---

### Slide 10 — Robustness Stress-Testing (Milestone M8.2)
- **Points**:
  - Evaluated under 6 synthetic perturbations on holdout validation samples.
  - Highly robust to contrast (+0.14%), brightness (-2.24%), and JPEG compression (-2.62%).
  - Vulnerability identified: Sensor noise (-23.72%) and Gaussian blur (-47.82%).
- **Key Visual**: `FIGURES/Figure_3_Robustness_Perturbation_Analysis.png`
- **Presenter Notes**: *"We systematically stressed the model. While it resists illumination and compression changes, loss of high-frequency edge detail causes notable degradation."*

---

### Slide 11 — CPU Deployment Latency
- **Points**:
  - Benchmarked on CPU: $224 \times 224$ resolution, batch size = 1.
  - Mean Latency: **132.18 ms**; P95 Latency: **171.77 ms**.
  - Throughput: **7.57 images/second** (unbatched, single worker).
- **Key Visual**: `FIGURES/Figure_4_CPU_Inference_Latency.png`
- **Presenter Notes**: *"A single-image latency of 132 ms allows our model to run smoothly on edge CPUs and mobile applications without demanding dedicated GPUs."*

---

### Slide 12 — Final Locked-Test Results (Milestone M9)
- **Points**:
  - One-shot evaluation on 2,724 previously untouched test images.
  - **Accuracy: 95.56%** (2,603 / 2,724 correct).
  - **Macro-F1: 94.06%** | **Weighted-F1: 95.55%**.
  - **Macro Precision: 94.00%** | **Macro Recall: 94.43%**.
- **Table**: `TABLES/Table_3_Performance_Comparison.md`
- **Presenter Notes**: *"Our locked test result demonstrates 95.56% accuracy and 94.06% Macro-F1 across 2,724 unseen images, confirming strong held-out generalization."*

---

### Slide 13 — Per-Class Pathology Highlights
- **Points**:
  - *Top Performers*: Tomato Yellow Leaf Curl Virus (99.12% F1), Healthy (98.94% F1).
  - *Healthy Leaf Specificity*: 100.0% precision — zero false diseased leaves called healthy.
  - *Challenging Classes*: Target Spot (86.78% F1), Early Blight (90.14% F1).
- **Table**: `TABLES/Table_4_Per_Class_Performance.md`
- **Presenter Notes**: *"Seven out of ten classes achieved above 95% F1. Healthy leaves were classified with 100% precision, preventing false alarms."*

---

### Slide 14 — Confusion Matrix & Error Dynamics
- **Points**:
  - Total test errors: 121 out of 2,724 samples (4.44% error rate).
  - Primary Error: Target Spot misclassified as Spider Mites (29 errors).
  - Cause: Small pinpoint fungal lesions visually mimic chlorotic mite stippling.
- **Key Visual**: `FIGURES/Figure_5_Normalized_Confusion_Matrix.png`
- **Presenter Notes**: *"Our normalized confusion matrix shows 29 Target Spot samples confused with Spider Mites. This aligns with pathological visual similarity in early lesion stages."*

---

### Slide 15 — Prediction Confidence Separation
- **Points**:
  - Correct predictions exhibit high certainty: Mean **96.97%** (Median **99.95%**).
  - Misclassifications exhibit marked uncertainty: Mean **70.65%** (Median **69.10%**).
  - Enables practical threshold gating: flag predictions under 75% for human review.
- **Key Visual**: `FIGURES/Figure_6_Confidence_Distribution.png`
- **Presenter Notes**: *"A critical safety property is confidence separation. When the model makes an error, its confidence drops to 70%, allowing easy human-in-the-loop triage."*

---

### Slide 16 — Generalization Gap Analysis
- **Points**:
  - Full Validation (94.23% Macro-F1) $\to$ Locked Test (**94.06% Macro-F1**).
  - Generalization Gap: **0.17 percentage points**.
  - Net Gain vs Frozen Baseline: **+5.90 percentage points**.
- **Table**: `TABLES/Table_6_Generalization_Analysis.md`
- **Presenter Notes**: *"The generalization gap is only 0.17 percentage points, proving that the model did not overfit the validation set."*

---

### Slide 17 — Limitations & Future Work
- **Points**:
  - *Limitations*: Lab-curated backgrounds; sensitivity to severe blur/noise; unbatched CPU runtime.
  - *Future Work*: In-field dataset validation (PlantDoc); INT8 quantization; UAV drone video pipeline.
- **Presenter Notes**: *"We transparently acknowledge our limitations: curated backgrounds require field validation, and severe lens blur degrades performance."*

---

### Slide 18 — Conclusion & Acknowledgments
- **Points**:
  - Successfully developed and validated LS-HGNet (8.04M params) for tomato disease diagnosis.
  - Confirmed 95.56% test accuracy and 94.06% Macro-F1 with locked-test integrity.
  - All artifacts, checkpoints, and reports are fully consolidated and reproducible.
- **Presenter Notes**: *"In conclusion, our LS-HGNet framework delivers accurate, balanced, and calibrated tomato disease diagnosis. Thank you, and we welcome your questions."*
