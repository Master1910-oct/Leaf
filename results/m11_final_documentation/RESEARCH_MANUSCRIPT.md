# Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model

**Authors**: AgriAI Research Team  
**Affiliation**: Department of Computer Science & Engineering / Precision Agriculture Laboratory  
**Target Venue**: IEEE / Elsevier Precision Agriculture & Applied Artificial Intelligence Journal  

---

## Abstract
Automated plant disease diagnosis is a cornerstone of digital smart farming, yet conventional deep convolutional architectures often demand tens of millions of parameters or struggle with multi-scale lesion variations. In this study, we adapt and evaluate the **Lightweight Stacked Hourglass Network (LS-HGNet)** for automated multi-class diagnosis across 10 tomato leaf conditions (9 foliar diseases and healthy leaves). The proposed architecture integrates three recurrent hourglass stacks with multi-dilation residual bottleneck blocks ($r \in \{1, 2, 4\}$) to capture fine-grained necrotic specks and broad chlorotic blights with 8,041,354 parameters. Global hyperparameter exploration was benchmarked using the bio-inspired Sooty Tern Optimization (STO) algorithm. Following 10-epoch full convergence training, the model checkpoint was permanently frozen and evaluated under strict data isolation on a locked held-out test partition comprising 2,724 images from the 18,160-image curated PlantVillage tomato dataset. On the locked test set, the frozen LS-HGNet achieved an **Overall Accuracy of 95.56%**, a **Macro-F1 score of 94.06%**, a **Weighted-F1 score of 95.55%**, **Macro Precision of 94.00%**, and **Macro Recall of 94.43%**. Out-of-sample generalization was exceptionally stable, with a validation-to-test Macro-F1 difference of only **0.17 percentage points** (94.23% vs 94.06%) and a net gain of **+5.90 percentage points** over the initial baseline model. Post-hoc Temperature Scaling ($T=1.1558$) achieved calibrated probability outputs with an Expected Calibration Error of 1.73%. Controlled stress-testing demonstrated strong resilience against illumination shifts and JPEG compression, while highlighting optical blur as a primary operational vulnerability. Single-image CPU inference latency averaged $132.18 \pm 16.71$ ms (7.57 images/second), confirming the feasibility of lightweight edge deployment for precision agricultural scouting.

**Keywords**: Precision Agriculture, Smart Farming, LS-HGNet, Stacked Hourglass Networks, Plant Pathology, Tomato Leaf Disease, Sooty Tern Optimization, Deep Learning, Generalization Audit.

---

## 1. Introduction

### 1.1 Background
Tomato (*Solanum lycopersicum*) represents one of the most widely cultivated and economically essential vegetable crops globally. Commercial yields, however, are persistently threatened by bacterial, fungal, and viral pathogens that attack foliage, diminish photosynthetic capacity, and ruin fruit harvests. Early, accurate detection of foliar symptoms is vital to implementing localized chemical and biological treatment before epidemics spread across entire acreage.

### 1.2 Problem Statement
Manual field scouting by agronomists is labor-intensive, costly, subjective, and impractical for smallholder or large-scale acreage. While computer vision systems have shown promise, current approaches present critical drawbacks:
1. **Architectural Overhead**: Heavy backbones (e.g., ResNet-50, VGG-16) require extensive computational memory, rendering edge deployment difficult.
2. **Multi-Scale Spatial Disconnect**: Foliar pathologies manifest across vastly different spatial scales—from minute bacterial spots (<2 mm) to sprawling late blight lesions. Standard CNNs with uniform receptive fields often fail to resolve both scales simultaneously.
3. **Lack of Rigorous Evaluation Governance**: Many published models lack locked test sets or calibration audits, risking over-optimistic generalization claims.

### 1.3 Motivation & Research Objectives
This study adapts a **Lightweight Stacked Hourglass Network (LS-HGNet)** designed to maintain multi-scale spatial receptive fields while restricting the parameter footprint to 8.04 million parameters. We integrate bio-inspired Sooty Tern Optimization (STO) exploration, post-hoc confidence calibration, systematic perturbation stress-testing, and a cryptographically verified locked-test protocol.

### 1.4 Primary Contributions
- **Architectural Adaptation**: Adapted LS-HGNet featuring 3 recurrent stacks and multi-dilation residual bottlenecks ($r \in \{1, 2, 4\}$) for multi-class foliar disease classification.
- **Empirical Generalization Verification**: Evaluated the frozen checkpoint on a strictly locked 2,724-image test set, achieving 95.56% accuracy and 94.06% Macro-F1 with a negligible 0.17 percentage-point generalization gap.
- **Calibrated Uncertainty & Deployment Profiling**: Validated post-hoc Temperature Scaling ($T=1.1558$, ECE=1.73%) and unbatched CPU inference latency ($132.18$ ms/image).
- **Vulnerability Mapping**: Identified fine-grained diagnostic failure modes (Target Spot vs Spider Mites) and quantified optical blur sensitivity.

---

## 2. Related Methodology

### 2.1 Plant Disease Classification
Deep convolutional neural networks have fundamentally transformed agricultural imaging. Early work by Hughes and Salathé (2015) established the PlantVillage benchmark. Subsequent studies applied standard transfer learning (e.g., MobileNets, DenseNets). However, standard models often sacrifice high-frequency spatial detail during rapid spatial pooling.

### 2.2 Stacked Hourglass Networks
Originally introduced by Newell et al. (2016) for human pose estimation, stacked hourglass networks repeatedly downsample and upsample feature representations, integrating multi-scale skip connections. This recurrent bottom-up, top-down processing enables the model to preserve fine local textures while capturing holistic image context.

### 2.3 Sooty Tern Optimization (STO)
The Sooty Tern Optimization Algorithm (Dhiman & Kaur, 2019) is a bio-inspired population metaheuristic simulating the migration and spiral attacking behaviors of seabirds. Krishnan et al. (2024) recently paired STO with an hourglass network for drone-based maize tassel detection. In our work, we adapt this framework to tomato foliar disease classification, employing STO for hyperparameter space exploration.

---

## 3. Proposed Method

### 3.1 Overall System Architecture
The classification framework processes input images through standard deterministic normalization, extracts multi-scale representations across 3 hourglass stacks, aggregates features through global pooling, and produces 10-class calibrated probabilities.

### 3.2 Preprocessing Pipeline
Input images are resized to $224 \times 224$ pixels in RGB format, converted to floating-point tensors, and normalized using standard ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 3.3 LS-HGNet Architecture Specification
- **Stem**: $3 \times 3$ Conv ($3 \to 64$ channels), Batch Normalization, ReLU.
- **Hourglass Stacks ($N=3$)**: Each stack downsamples spatial dimensions over 4 encoder levels via MaxPool2d (stride 2).
- **Multi-Dilation Bottleneck**: Within each bottleneck, three parallel branches apply $3 \times 3$ convolutions at dilation rates $r \in \{1, 2, 4\}$. Dilated features are concatenated and projected, expanding the receptive field without loss of spatial resolution.
- **Classification Head**: AdaptiveAvgPool2d collapses spatial dimensions to $1 \times 1 \times 64$, followed by a 0.30 Dropout layer and a Linear(64, 10) projection.

---

## 4. Experimental Setup

### 4.1 Dataset & Partitioning
The dataset comprises 18,160 curated tomato leaf images across 10 classes:
- **Training Set**: 12,712 images (70.0%)
- **Validation Set**: 2,724 images (15.0%)
- **Locked Test Set**: 2,724 images (15.0%)
All splits were stratified and cryptographically audited to ensure exactly 0 sample overlap.

### 4.2 Training Hyperparameters
Training was conducted over 10 epochs using AdamW with an initial learning rate of $1.0 \times 10^{-3}$, weight decay of $1.0 \times 10^{-4}$, batch size of 32, and a Cosine Annealing learning rate schedule.

---

## 5. Results & Discussion

### 5.1 Validation and Locked-Test Performance
Model convergence peaked at Epoch 9 with a validation accuracy of **95.52%** and a Macro-F1 score of **94.23%**. When evaluated in a single one-shot pass on the locked test partition (2,724 images), the frozen model achieved:
- **Test Accuracy**: **95.56%** (2,603 / 2,724 correct)
- **Macro-F1 Score**: **94.06%**
- **Weighted-F1 Score**: **95.55%**
- **Macro Precision**: **94.00%** | **Macro Recall**: **94.43%**

### 5.2 Generalization Gap Analysis
The generalization gap between full validation and locked test was:
- $\Delta \text{Accuracy} = 95.52\% - 95.56\% = -0.04$ percentage points.
- $\Delta \text{Macro-F1} = 94.23\% - 94.06\% = +0.17$ percentage points.
This tight concordance demonstrates robust out-of-sample stability.

### 5.3 Per-Class Performance & Error Dynamics
Seven of the ten classes achieved F1-scores exceeding 95%:
- *Tomato Yellow Leaf Curl Virus*: **99.12% F1** (Precision: 99.62%, Recall: 98.63%, Support: 804)
- *Tomato Healthy*: **98.94% F1** (Precision: 100.0%, Recall: 97.90%, Support: 238)
- *Septoria Leaf Spot*: **96.30% F1** (Precision: 94.55%, Recall: 98.11%, Support: 265)

The lowest scores occurred in *Target Spot* (**86.78% F1**, Recall: 82.46%) and *Early Blight* (**90.14% F1**, Recall: 85.33%). The primary confusion pair was *Target Spot* misclassified as *Two-Spotted Spider Mites* (29 errors, 13.74% of Target Spot), caused by visual similarity between diffuse pinpoint lesions and chlorotic mite stippling.

### 5.4 Confidence Calibration & Robustness Profiling
Post-hoc Temperature Scaling ($T=1.1558$) adjusted overconfident predictions, maintaining an Expected Calibration Error of 1.73%. Correct classifications displayed a mean confidence of **96.97%** (median: 99.95%), whereas misclassifications dropped to **70.65%** (median: 69.10%). Robustness testing revealed strong resilience against illumination shifts (+0.14% under contrast reduction) and JPEG compression (-2.62%), but identified vulnerability to Gaussian blur (-47.82% Macro-F1 drop).

---

## 6. Limitations & Future Work
1. **Curated Backgrounds**: Evaluated on controlled PlantVillage imagery; in-field performance under dense canopy occlusion requires further validation.
2. **Blur Sensitivity**: High-frequency edge sensitivity necessitates optical autofocus gating in field devices.
3. **Edge Optimization**: CPU latency averages 132.2 ms; future work will implement INT8 quantization and ONNX runtime integration for sub-30 ms embedded execution.

---

## 7. Conclusion
This study demonstrates that the Lightweight Stacked Hourglass Network (LS-HGNet), with 8.04 million parameters, delivers accurate (95.56% accuracy), balanced (94.06% Macro-F1), and calibrated disease classification across 10 tomato leaf conditions. Enforcing strict locked-test isolation confirmed a minimal 0.17 percentage-point generalization gap, establishing a reliable architectural foundation for precision smart farming diagnostics.

---

## References
1. **Krishnan, V. G., Vikranth, B., Sumithra, M., Laxmi, B. P., & Gowri, B. S. (2024).** Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model. *International Journal of Experimental Research and Review*, 37(spl), 96–108. https://doi.org/10.52756/ijerr.2024.v37spl.008
2. **Dhiman, G., & Kaur, A. (2019).** STOA: A bio-inspired based optimization algorithm for industrial engineering problems. *Engineering Applications of Artificial Intelligence*, 82, 148–174. https://doi.org/10.1016/j.engappai.2019.03.021
3. **Newell, A., Yang, K., & Deng, J. (2016).** Stacked Hourglass Networks for Human Pose Estimation. In *European Conference on Computer Vision (ECCV)*, pp. 483–499. arXiv:1603.06937.
4. **Hughes, D. P., & Salathé, M. (2015).** An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.
5. **Howard, A. G., et al. (2017).** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. *arXiv preprint arXiv:1704.04861*.
6. **Selvaraju, R. R., et al. (2017).** Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. In *IEEE International Conference on Computer Vision (ICCV)*, pp. 618–626. https://doi.org/10.1109/ICCV.2017.74
7. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** On Calibration of Modern Neural Networks. In *International Conference on Machine Learning (ICML)*, PMLR 70, pp. 1321–1330.
