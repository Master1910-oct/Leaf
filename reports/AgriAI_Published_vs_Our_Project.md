# AgriAI Research Positioning: Published Work vs. Our Project

AgriAI is an independent adaptation and experimental extension of the LS-HGNet and Sooty Tern Optimization concepts from the published smart-farming work, applying them to tomato leaf disease classification and introducing a controlled comparison of default, STO-selected, and Random-Search configurations with strict validation-based model selection and independent test evaluation.

**Document Classification:** Comparative Research Analysis & Technical Positioning Report  
**Subject:** Methodological Comparison between Krishnan et al. (2024) and the AgriAI (Leaf) Project  
**Date:** October 6, 2026  
**Status:** M0–M7 Complete / Locked  

---

## 1. Overview and Intent

This document provides a formal, evidence-based comparative analysis between the published reference paper:
> **"Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model"**  
> V. Gokula Krishnan, B. Vikranth, M. Sumithra, B. Prathusha Laxmi, B. Shyamala Gowri  
> *International Journal of Experimental Research and Review (IJERR)*, Vol. 37, pp. 96–108 (March 2024), DOI: `10.52756/ijerr.2024.v37spl.008`

and our independently developed research project:
> **"STO-Optimized LS-HGNet for Explainable Crop Disease Classification in Smart Farming" (AgriAI / Leaf)**

The objective of this report is to transparently establish:
1. What the published paper actually investigated and claimed.
2. How the AgriAI project adapted and altered those concepts.
3. The exact technical, architectural, and experimental differences between the two works.
4. What constitutes genuine research adaptation versus unverified claims.
5. The precise boundary between completed engineering (M0–M7) and planned future extensions (M8).

---

## 2. In-Depth Analysis of the Published Reference Paper

### 2.1 The Reference Problem: UAV Maize Tassel Scouting
Despite the word *"Classification"* appearing in the title of Krishnan et al. (2024), a detailed reading of the paper reveals that the actual empirical problem studied was **UAV-based maize (corn) tassel detection and counting**:
- **Application:** Precision aerial phenotyping to count male maize inflorescences (tassels) to assist in flowering stage detection and harvest yield prediction.
- **Acquisition Hardware:** DJI Mavic Unmanned Aerial Vehicle (UAV).
- **Imaging Geometry:** Aerial nadir photography taken at 5-meter and 10-meter flight altitudes above the crop canopy.
- **Location:** Chengdu, Sichuan, China.
- **Dataset Scale:** 960 high-resolution RGB aerial images ($1920 \times 1080$).
- **Annotations:** Bounding boxes delineating approximately 41,232 individual tassels annotated using LabelImg.
- **Core Challenge:** Detecting tiny, dense, visually ambiguous tassels overlapping within large foliage canopies.

### 2.2 Conceptual Role of LS-HGNet
The authors introduced a **Lightweight Stacked Hourglass Network (LS-HGNet)** to handle the multi-scale spatial challenge of tassel detection:
- Standard Hourglass architectures (originally developed for human pose keypoint estimation by Newell et al., 2016) downsample features to capture global spatial context and repeatedly upsample them to regain high-resolution spatial maps.
- Krishnan et al. integrated **Pre-Activation Residual Blocks** and **Multi-Dilated Light Residual Blocks** using depthwise separable convolutions to extract multi-scale tassel features with fewer parameters than standard Hourglass-104 models.

### 2.3 Role and Mathematical Ambiguities of STOA in the Reference Work
The authors referenced the **Sooty Tern Optimization Algorithm (STOA)** introduced by Dhiman & Kaur (2019):
- **Nature Metaphor:** Simulates migration behavior (collision avoidance, movement towards best neighbor) and spiral attacking behaviors of sea-foraging sooty terns.
- **Theoretical Ambiguity in Reference Paper:** The reference paper asserted that STOA was used to optimize the network's parameters. In deep learning, optimizing the millions of continuous weights and biases of a convolutional network using gradient-free metaheuristics is computationally intractable and mathematically suboptimal compared to backpropagation with Adam/SGD. The exact mathematical coupling between STOA and LS-HGNet was left ambiguous in the published text, and no source code was published to verify the mechanism.

---

## 3. The AgriAI Adaptation: Rationale & Domain Shift

### 3.1 Why Adapt to Tomato Foliar Disease Classification?
Instead of replicating an aerial counting task, the AgriAI project adapted the architectural strengths of LS-HGNet to address a critical agricultural pathology challenge: **automated foliar disease diagnosis in tomato crops** (*Solanum lycopersicum*).

* **Pathological Complexity:** Foliar crop diseases manifest at radically different spatial scales—from tiny initial bacterial lesions (sub-millimeter spots) to large necrotic blights covering an entire leaf blade.
* **Hourglass Compatibility:** The multi-scale downsampling/upsampling and skip-connection mechanisms of Stacked Hourglass architectures are exceptionally well suited for preserving both minute lesion details and overall leaf context.
* **Target Dataset:** The **PlantVillage Tomato Dataset** (18,160 images, 10 diagnostic categories), providing an established, reproducible benchmark for comparative evaluation.

---

## 4. Detailed Technical Differences

The table below contrasts the fundamental technical specifications between the published paper and the AgriAI implementation:

| Technical Aspect | Published Work (Krishnan et al., 2024) | AgriAI Project Implementation (Leaf) |
| :--- | :--- | :--- |
| **Agricultural Target** | Maize tassels (*Zea mays*) | Tomato leaves (*Solanum lycopersicum*) |
| **Visual Pathology** | Tassel presence / density / count | 9 foliar disease classes + 1 healthy control |
| **Image Platform** | Drone aerial photography (DJI Mavic) | Handheld / proximal photography (PlantVillage) |
| **Image Count & Resolution** | 960 aerial images ($1920 \times 1080$) | 18,160 leaf images ($256 \times 256$ raw, $224 \times 224$ model input) |
| **Computer Vision Task** | Object detection & counting (bounding boxes) | Multi-class image classification (10 classes) |
| **Output Layer** | Bounding box coordinates $[x, y, w, h]$ | 10-dimensional class logits (Global Average Pooling + Linear) |
| **Backbone Pretraining** | Unspecified / custom | Trained strictly from scratch (no ImageNet pretraining) |
| **Optimization Target** | Described vaguely as model weights/parameters | Explicit 4-dimensional hyperparameter search space ($\eta, \lambda, p, B$) |
| **Optimization Space Mapping** | Assumed 3D kinematics | Formal 4D cyclic projection: $\Omega_4 = [\Omega_1, \Omega_2, \Omega_3, \Omega_1]$ |
| **Network Backpropagation** | Unspecified interaction with STOA | Pure gradient descent via AdamW with Cross-Entropy Loss |
| **Data Partitioning** | Conventional / unspecified split details | Deterministic 70% Train / 15% Val / 15% Test (Seed 42) |
| **Test Set Isolation** | Not formally audited; risk of repeated test tuning | Strict cryptographic test set locking; evaluated exactly once post-selection |
| **Baseline Architecture** | Compared against YOLO and VGG object detectors | Implemented 457K-parameter custom CNN baseline |
| **Optimization Comparison** | None (STOA presented in isolation) | Systematic 3-way study: Default vs. STO-Best vs. Random Search-Best |
| **Proxy Fidelity Testing** | None | 5-configuration Spearman rank correlation test ($\rho = 1.0$) |
| **Selection Criterion** | Accuracy / mAP | Validation Macro-F1 (to account for class imbalance) |
| **Code Availability** | Proprietary / closed source / no repository | Modular, documented PyTorch codebase with 94 passing unit tests |

---

## 5. Experimental Differences and Results

### 5.1 Optimization Benchmarking
A major limitation of many applied metaheuristic papers is the absence of fair control baselines (such as Random Search or default established configurations). AgriAI directly addressed this:
* **Published Paper:** Reported that STOA produced superior performance, but provided no comparison against standard random hyperparameter search or default learning rates under an identical compute budget.
* **AgriAI Project:** Designed a controlled experiment (M6) comparing Stabilized STO (30 proxy evaluations) against Random Search (25 proxy evaluations) on a standardized proxy budget (200 images, 128×128, 3 epochs).
* **The Empirical Reality (M7 Full Training):**
  * When the top configurations were retrained on the full 12,712 training images for 5 epochs:
    * **Candidate A (Default LS-HGNet):** **88.16% Validation Macro-F1** (Winner)
    * **Candidate C (Random Search Best):** **85.22% Validation Macro-F1**
    * **Candidate B (STO-Optimized Best):** **76.91% Validation Macro-F1**
  * *Scientific Value:* This represents an important **negative result**: an optimization algorithm tuning hyperparameters on a constrained, small-scale proxy may select hyperparameter regimes (e.g., high learning rate $\approx 0.005$, high dropout $0.50$) that perform well on small subsets but generalize poorly when scaled to the full dataset. Reporting this negative result demonstrates scientific rigor and prevents false claims about metaheuristic superiority.

### 5.2 Independent Test Evaluation
* **Published Paper:** Reported test metrics without detailing data isolation or safeguards against data leakage.
* **AgriAI Project:** Implemented strict test isolation. The 2,724 test images were evaluated **only once** after Candidate A was formally selected as the winner. The resulting test metrics are unbiased:
  * **Test Accuracy:** **90.71%**
  * **Test Macro-F1:** **87.40%**
  * **Test Precision:** **89.35%**
  * **Test Recall:** **87.59%**
  * **Test Loss:** 0.2801

---

## 6. What is Genuinely New in AgriAI?

When presenting this project to research examiners, the following distinct contributions should be emphasized:

1. **New Application Domain:** Adapting LS-HGNet and STOA from drone maize counting to 10-class tomato foliar pathology classification.
2. **Ground-Truth Modular PyTorch Implementation:** Fully implemented from first principles, including Pre-Activation Residual blocks, Multi-Dilated Light Residual Blocks ($d \in \{1, 2, 3\}$), Stacked Hourglass inter/intra-stack skip connections, and Global Average Pooling.
3. **Rigorous Mathematical Formulation of STOA for Hyperparameters:** Overcoming the 3D-space limitation of STOA by establishing a formal 4D cyclic projection mapping ($\Omega_4 = [\Omega_1, \Omega_2, \Omega_3, \Omega_1]$).
4. **Transparent Benchmark against Random Search:** Proving via empirical data that STO does not automatically guarantee superior hyperparameter discovery compared to random sampling.
5. **Strict Scientific Methodology:** Zero test set leakage, deterministic seed control, 94 unit/integration tests, and locked model weights (`results/m7_final_model/model_best.pt`).

---

## 7. Status of Project Components: Implemented vs. Future Work

To ensure complete honesty and avoid misrepresenting planned features as completed deliverables, the technical scope is partitioned as follows:

### Completed and Verified (M0 through M7)
- [x] Mathematical audit and specification of STOA and LS-HGNet (`M0`)
- [x] PlantVillage dataset partitioning, normalization, and leakage verification (`M1`)
- [x] Baseline 457K-parameter CNN implementation and profiling (`M3`)
- [x] Pure PyTorch 8.04M-parameter LS-HGNet implementation and verification (`M4`)
- [x] Vectorized STO algorithm and benchmark function testing suite (`M5`)
- [x] Proxy hyperparameter search (STO vs. Random Search) and fidelity analysis (`M6`)
- [x] Full-scale candidate retraining, validation Macro-F1 model selection, and single locked test evaluation (`M7`)
- [x] Comprehensive automated test suite: **94/94 unit and integration tests passing**

### Planned Future Work (Milestone 8 — M8)
- [ ] **Explainability:** Grad-CAM saliency maps on the final hourglass layer to explain foliar lesion detections to agronomists.
- [ ] **Calibration:** Post-hoc Temperature Scaling to align model output probabilities with true diagnostic accuracy.
- [ ] **Robustness:** Robustness stress testing against synthetic agricultural corruptions (motion blur, lighting variations, sensor noise).
- [ ] **Deployment:** REST API inference service via Flask paired with a modern React/Vite farmer web interface.

---

## 8. Summary for Project Evaluators

| Question | Short Answer |
| :--- | :--- |
| *Did we copy the published paper?* | **No.** We adapted the conceptual framework (LS-HGNet and STOA) to a completely different crop, pathology, dataset, and computer vision task. |
| *Is the code taken from the paper?* | **No.** The paper published no source code. Our codebase is an independent, modular PyTorch implementation. |
| *Did STO beat default hyperparameters?* | **No.** Under our experimental protocol, the Default LS-HGNet (88.16% Val Macro-F1) outperformed the STO-selected model (76.91% Val Macro-F1). This is reported honestly as an empirical finding. |
| *What is the final performance of the system?* | The frozen LS-HGNet model achieves **90.71% accuracy** and **87.40% Macro-F1** on 2,724 locked, unseen test images across 10 tomato health categories. |
| *Are explainability and deployment finished?* | **No.** They are formally scheduled as part of Milestone 8 (M8). |
