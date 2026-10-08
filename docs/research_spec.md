# Research Specification — AgriAI
## Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis

**Document version:** M0-RESEARCH-SPEC-LOCKED  
**Date:** 2026-10-01  
**Status:** ✅ M0 LOCKED — Research Specification Approved  

---

## 1. Project Title

**AgriAI — Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis**

---

## 2. Abstract

This project investigates whether a lightweight, stacked-hourglass-inspired convolutional neural network (LS-HGNet), with hyperparameters optimized using the Sooty Tern Optimization Algorithm (STOA), can deliver accurate, computationally efficient, explainable, calibrated, and robustly-evaluated tomato leaf disease classification. The PlantVillage tomato subset is used as the primary dataset. The project extends concepts from prior smart-farming and bio-inspired optimization research to a new task (disease classification), with additional components including Grad-CAM explainability, post-hoc temperature-scaling calibration, systematic robustness testing, a Flask REST API, and a React dashboard. A simple CNN baseline and a Random Search optimizer are included to provide experimental controls.

This is an independent adaptation and extension project. No original research results, code, or text from the referenced source papers are reproduced.

---

## 3. Background

Tomato is one of the most widely cultivated and economically significant crops globally. Crop diseases cause substantial yield losses. Accurate, timely, and explainable automated disease diagnosis can reduce losses and support precision agriculture. Convolutional neural networks have demonstrated strong performance on plant disease classification tasks, particularly when applied to the PlantVillage dataset. However, most approaches lack:

- Systematic hyperparameter optimization under fair experimental budgets
- Explainability (model reasoning transparency)
- Confidence calibration (reliable uncertainty estimates)
- Robustness testing under controlled image degradation

This project addresses these gaps in the context of a computationally lightweight architecture.

---

## 4. Problem Statement

Tomato leaf diseases are visually diverse and can be difficult to distinguish manually. Automated classification systems require:

1. Sufficient classification accuracy across multiple disease classes
2. Computational efficiency suitable for resource-constrained deployment
3. Explainability to support agricultural decision-making
4. Calibrated confidence to convey reliable uncertainty estimates
5. Robustness under realistic image degradation

Existing studies typically optimize only one or two of these properties. This project explores all five in an integrated framework.

---

## 5. Research Gap

While deep learning methods for tomato disease classification are well-studied, the following combination has not been thoroughly investigated in the literature surveyed:

- A lightweight stacked-hourglass-inspired architecture adapted from UAV vision tasks to disease classification
- Sooty Tern Optimization used for joint hyperparameter search across learning rate, weight decay, dropout, and batch size, compared against an equal-budget Random Search under controlled conditions
- Simultaneous evaluation of explainability (Grad-CAM), calibration (temperature scaling), and robustness (synthetic perturbation) within one experimental framework

---

## 6. Research Question

> **Can an STO-optimized lightweight LS-HGNet architecture provide accurate, computationally efficient, explainable, calibrated, and robust tomato leaf disease classification, and does STO hyperparameter optimization yield a measurable advantage over equal-budget Random Search?**

The research question is falsifiable and will be answered through measured experimental results. No performance claims are made prior to experiments.

---

## 7. Objectives

The following objectives are testable and measurable.

| ID | Objective | Completion Criterion |
|----|-----------|---------------------|
| OBJ-01 | Prepare the PlantVillage tomato subset with verified class distribution, stratified split, and augmentation policy | Split CSVs created; class counts verified from downloaded data |
| OBJ-02 | Implement a simple baseline CNN (~500K parameters) | Baseline achieves a measurable accuracy; parameter count confirmed |
| OBJ-03 | Implement and evaluate an LS-HGNet adapted for disease classification | LS-HGNet trains and evaluates on all four metrics |
| OBJ-04 | Implement STOA-based hyperparameter optimization | STOA completes without using test data; convergence history recorded |
| OBJ-05 | Implement equal-budget Random Search as a control | Random Search completes with same evaluation count as STOA |
| OBJ-06 | Evaluate all models on classification and computational metrics | Results table complete with measured values |
| OBJ-07 | Produce Grad-CAM visualizations for all trained models | Heatmaps generated and saved for at least 5 examples per class |
| OBJ-08 | Apply and evaluate temperature scaling calibration | ECE and reliability diagrams produced before and after calibration |
| OBJ-09 | Conduct robustness testing under 7 perturbation types | Accuracy vs. degradation curve produced |
| OBJ-10 | Deploy a Flask REST API and React dashboard | API returns predictions with Grad-CAM overlays; dashboard displays results |
| OBJ-11 | (Optional) ONNX export and edge device latency measurement | Model exported and latency recorded on target device |

---

## 8. Source Research

### 8.1 Primary Source (LS-HGNet + STOA Applied to Smart Farming)

**Title:** Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model  
**Authors:** V. Gokula Krishnan, B. Vikranth, M. Sumithra, B. Prathusha Laxmi, B. Shyamala Gowri  
**Journal:** International Journal of Experimental Research and Review  
**Volume:** 37 (special issue)  
**Pages:** 96–108  
**Year:** 2024  
**DOI:** 10.52756/ijerr.2024.v37spl.008  
**Publication Date:** 30 March 2024  
**Primary Source Architectural Details:**
- **Domain & Application:** The paper addresses precision smart farming, specifically the automated detection and counting of maize tassels in drone/UAV aerial photography.
- **Preprocessing:** Applies morphological bottom-hat and top-hat filtering to mitigate irregular sunlight reflections and ambient shadow noise in aerial foliage.
- **Network Architecture:** Introduces the Lightweight Weight-Stacked Hourglass Network (LS-HGNet). The backbone uses stacked hourglass modules (derived from Newell et al., 2016) augmented with dilated convolutions and inter-module skip connections to expand the receptive field without inflating parameter count.
- **Optimization Strategy:** Employs the Sooty Tern Optimization Algorithm (STOA) for tuning classification hyperparameters, reporting an empirical maize tassel detection accuracy of 98.7%.
- **Originality and Attribution Boundary:**
  - **What AgriAI adopts conceptually:** The synergistic combination of (1) a multi-stack hourglass network parameterized for lightweight vision inference, and (2) gradient-free bio-inspired STOA metaheuristic tuning for hyperparameter search.
  - **What AgriAI does NOT adopt:** Maize tassel counting, UAV imagery, morphological preprocessing as mandatory core pipeline, and original source weights or code. AgriAI adapts the architecture for 10-class tomato leaf disease classification on PlantVillage using Depthwise Separable Convolutions, MultiDilated bottleneck blocks, Grad-CAM interpretability, temperature scaling calibration, and an explicit evaluation against an equal-budget Random Search control.

### 8.2 STOA Original Paper

**Title:** STOA: A bio-inspired based optimization algorithm for industrial engineering problems  
**Authors:** Gaurav Dhiman and Amandeep Kaur  
**Journal:** Engineering Applications of Artificial Intelligence  
**Volume:** 82  
**Pages:** 148–174  
**Year:** 2019  
**DOI:** 10.1016/j.engappai.2019.03.021  
**What we use:** Mathematical principles of the STOA algorithm (collision avoidance, convergence towards best agent, spiral attack behavior) evaluated as a gradient-free population-based hyperparameter optimizer.

### 8.3 Stacked Hourglass Network

**Title:** Stacked Hourglass Networks for Human Pose Estimation  
**Authors:** Alejandro Newell, Kaiyu Yang, Jia Deng  
**Year:** 2016  
**Venue:** ECCV  
**arXiv:** 1603.06937  
**What we use:** Conceptual basis for encoder-bottleneck-decoder hourglass structure, stacked refinement principle, skip connections across resolution levels.

### 8.4 PlantVillage Dataset

**Title:** An open access repository of images on plant health to enable the development of mobile disease diagnostics  
**Authors:** David P. Hughes, Marcel Salathé  
**Year:** 2015  
**Venue:** arXiv  
**arXiv:** 1511.08060  
**What we use:** PlantVillage tomato subset as the primary classification dataset.

### 8.5 Depthwise Separable Convolution

**Title:** MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications  
**Authors:** Andrew G. Howard et al.  
**Year:** 2017  
**Venue:** arXiv  
**arXiv:** 1704.04861  
**What we use:** Depthwise separable convolution as a lightweight building block.

### 8.6 Grad-CAM

**Title:** Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization  
**Authors:** Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra  
**Year:** 2017  
**Venue:** ICCV  
**DOI:** 10.1109/ICCV.2017.74  
**What we use:** Grad-CAM technique for post-hoc explainability of CNN predictions.

### 8.7 Temperature Scaling

**Title:** On Calibration of Modern Neural Networks  
**Authors:** Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger  
**Year:** 2017  
**Venue:** ICML  
**URL:** http://proceedings.mlr.press/v70/guo17a.html  
**What we use:** Temperature scaling as a post-hoc confidence calibration method; ECE as the primary calibration metric.

---

## 9. Adaptation Statement

> This project is an independent adaptation and extension of concepts reported in prior LS-HGNet and Sooty Tern Optimization research. The original research context (UAV-based maize tassel detection and counting), dataset, task, implementation, experiments, and reported results are not reproduced as original work. The present project independently implements and evaluates a tomato leaf disease classification system using related architectural and optimization concepts, together with additional explainability, calibration, robustness, and deployment components. All source code, all experimental results, all figures, and all numerical values presented in this project are independently produced. No text, code, or figures from the referenced source papers are copied, paraphrased without attribution, or claimed as our own research findings.

---

## 10. Dataset

### 10.1 Dataset Identity

| Property | Value |
|----------|-------|
| Name | PlantVillage |
| Task subset | Tomato leaf diseases |
| Source | Hughes & Salathé, 2015 |
| Access | TensorFlow Datasets / Kaggle / PlantVillage GitHub |
| License | CC0 (public domain) — TO BE VERIFIED during M1 |

### 10.2 Class Structure

The tomato subset of PlantVillage is commonly reported to contain **10 classes** (1 healthy + 9 diseased). The class names are:

1. Tomato — Bacterial Spot
2. Tomato — Early Blight
3. Tomato — Late Blight
4. Tomato — Leaf Mold
5. Tomato — Septoria Leaf Spot
6. Tomato — Spider Mites (Two-Spotted Spider Mite)
7. Tomato — Target Spot
8. Tomato — Tomato Mosaic Virus
9. Tomato — Tomato Yellow Leaf Curl Virus
10. Tomato — Healthy

> **NOTE:** The number of classes (10) is consistently reported across multiple published papers and must be verified against the actual downloaded dataset during M1. If the downloaded version differs, this document must be updated before any model training begins.

### 10.3 Image Counts

**Per-class approximate counts (from literature — NOT yet verified from source files):**

| Class | Approximate Count | Status |
|-------|-----------------|--------|
| Bacterial Spot | ~2,127 | From literature |
| Early Blight | ~1,000 | From literature |
| Late Blight | ~1,909 | From literature |
| Leaf Mold | ~1,000 | From literature |
| Septoria Leaf Spot | ~1,771 | From literature |
| Spider Mites | ~1,676 | From literature |
| Target Spot | ~1,404 | From literature |
| Tomato Yellow Leaf Curl Virus | TO BE VERIFIED | — |
| Tomato Mosaic Virus | TO BE VERIFIED | — |
| Healthy | TO BE VERIFIED | — |
| **Total (tomato subset)** | **~18,000+ (TO BE VERIFIED)** | — |

> **WARNING:** These counts are indicative only, sourced from secondary literature. The authoritative counts must be verified by direct inspection of the downloaded dataset during M1 dataset discovery. Do NOT use these numbers as ground truth in any reported result.

### 10.4 Known Limitations

1. PlantVillage images were collected under controlled laboratory conditions with uniform backgrounds.
2. Models trained on this dataset may not generalize to field images with complex, natural backgrounds.
3. Class imbalance may exist across the 10 classes.
4. The dataset does not include images of co-occurring diseases or infection at different growth stages.
5. Lighting conditions in the dataset do not reflect the full range of real-world agricultural environments.
6. The dataset contains no geographic diversity metadata.

---

## 11. Data Pipeline

### 11.1 Split Strategy

```
PlantVillage Tomato Subset (all images)
              |
     Stratified Split (by class label)
    /              |              \
Train (70%)   Val (15%)      Test (15%)
     |
  Augmentation ONLY on Training Data
     |
  Deterministic Preprocessing on Val + Test
```

| Property | Value |
|----------|-------|
| Train proportion | 70% |
| Validation proportion | 15% |
| Test proportion | 15% |
| Strategy | Stratified by class label |
| Random seed | 42 |
| Split files | `data/splits/train.csv`, `data/splits/val.csv`, `data/splits/test.csv` |

### 11.2 Leakage Prevention Rules

1. Splits are created **once** before any model training.
2. Augmentation is applied **only** to training data, never to validation or test data.
3. The test set is **never** seen during hyperparameter optimization.
4. STOA fitness is computed on **validation** data only.
5. Temperature scaling calibration parameters are fitted on **validation** data only.
6. Final reported metrics are computed on the **test** set **once** per experiment, after all hyperparameter decisions are finalized.

### 11.3 Augmentation Policy (Training Only)

These are engineering decisions and may be ablated:

| Augmentation | Parameters |
|-------------|-----------|
| Random horizontal flip | p = 0.5 |
| Random rotation | +/-30 degrees, p = 0.5 |
| Color jitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 |
| Random resized crop | scale=(0.7, 1.0) |

> **Biological Validity Note on Vertical Flip:** Random vertical flip is deliberately excluded from the default training pipeline because vertical inversion produces biologically unnatural upside-down leaf orientations for field-captured foliar images. It is retained strictly as an optional experimental ablation (ABL-05).

### 11.4 Deterministic Preprocessing (All Splits)

1. Resize to target resolution (128x128 for development; 224x224 for final)
2. Convert to RGB
3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## 12. LS-HGNet Architecture

### 12.1 Important Distinction

> The original Stacked Hourglass Network (Newell et al., 2016) was designed for **human pose estimation** (structured output). The source smart-farming paper applied a variant to **UAV-based maize tassel detection**. Our LS-HGNet adaptation applies the concept to **tomato leaf disease classification**. All implementation specifics below are our **engineering decisions** unless explicitly marked as source-derived.

### 12.2 Source-Derived Concepts

| Concept | Source |
|---------|--------|
| Encoder-bottleneck-decoder hourglass structure | Newell et al., 2016 |
| Skip connections at each resolution level | Newell et al., 2016 |
| Stacking multiple hourglass modules | Newell et al., 2016 |
| Residual block design | He et al., 2016 (ResNet) |
| Depthwise separable convolution | Howard et al., 2017 (MobileNets) |
| Dilated convolution concept | Yu & Koltun, 2016 |

### 12.3 Our Engineering Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Task | Classification (not detection/pose) | Task adaptation |
| Dilation rates | [1, 2, 4] | Engineering choice only |
| Multi-dilated fusion | Concatenation + 1x1 conv + residual | Engineering choice |
| Inter-stack skip connections | Present | Engineering decision |
| Classification head | GAP + Dropout + Linear | Standard practice |
| Input resolution (dev) | 128x128 | Computational feasibility |
| Input resolution (final) | 224x224 | Standard scale |
| Number of stacks (dev) | 2 | Computational feasibility |
| Number of stacks (final) | 3 | Capacity experiment |
| Base channels (dev) | 32 | Lightweight objective |
| Base channels (final) | 64 | Capacity experiment |

### 12.4 Overall Architecture

```
Input Image (H x W x 3)
        |
    Stem Block
   Conv(3x3, stride=2) -> BN -> ReLU -> MaxPool(2x2)
        |
   Hourglass Stack 1
   (Encoder -> Bottleneck -> Decoder with skip connections)
        |
  Inter-stack Skip Connection  [Engineering Decision]
        |
   Hourglass Stack 2
   (Encoder -> Bottleneck -> Decoder with skip connections)
        |
  [Optional Stack 3 - Final Config Only]
        |
   Global Average Pooling
        |
   Dropout (p = STOA-searched)
        |
   Linear Layer -> Class Logits (10 classes)
```

### 12.5 Hourglass Module

**Encoder:** L levels (L=3 dev, L=4 final), each: LightResidualBlock -> MaxPool(2x2). Channels: 32->64->128 (dev); 64->128->256 (final).

**Bottleneck:** MultiDilatedBlock at lowest resolution.

**Decoder:** L levels matching encoder. Each: Bilinear upsampling (x2) -> LightResidualBlock. Skip connections from encoder concatenated before each decoder block.

**Upsampling:** Bilinear + 1x1 conv (engineering decision; chosen over transposed conv for stability and parameter efficiency).

### 12.6 DepthwiseSeparableConv

| Property | Specification |
|----------|--------------|
| Depthwise conv | k x k, groups=in_channels, same-size padding with dilation |
| Pointwise conv | 1x1, groups=1 |
| Normalization | BatchNorm2d after each conv |
| Activation | ReLU after each BN |
| Default kernel | 3x3 |
| Stride | Configurable, default=1 |
| Dilation | Configurable, default=1 |

### 12.7 LightResidualBlock

| Property | Specification |
|----------|--------------|
| Primary path | DWSConv(in, mid) -> BN -> ReLU -> DWSConv(mid, out) -> BN |
| Skip path | Identity if in==out and stride==1; else Conv(1x1, stride) -> BN |
| Residual addition | Primary + Skip |
| Post-addition activation | ReLU |
| Projection condition | Applied when in_ch != out_ch OR stride != 1 |

### 12.8 MultiDilatedBlock

> **IMPORTANT: Dilation rates [1, 2, 4] are an engineering choice. They are NOT from the source smart-farming paper.**

```
Input (B, C, H, W)
      |
  [d=1 branch] [d=2 branch] [d=4 branch]  (Parallel DWSConv)
      |
  Concatenation -> (B, 3C, H, W)
      |
  1x1 Conv -> (B, C, H, W)
      |
  Residual Addition with Input
      |
  Output (B, C, H, W)
```

### 12.9 Classification Head

```
Final Hourglass Output (B, C, H, W)
        |
  Global Average Pooling -> (B, C)
        |
  Dropout(p)
        |
  Linear(C, 10) -> (B, 10) raw logits
```

**No softmax inside model.** Reason: PyTorch CrossEntropyLoss applies log_softmax internally; applying softmax beforehand corrupts the loss. Temperature scaling operates on raw logits. Softmax is applied only at inference time.

---

## 13. STO Formulation

### 13.1 Algorithm Identity

| Property | Value |
|----------|-------|
| Full name | Sooty Tern Optimization Algorithm (STOA) |
| Authors | Gaurav Dhiman and Amandeep Kaur |
| Year | 2019 |
| Venue | Engineering Applications of Artificial Intelligence, Vol. 82, pp. 148–174 |
| DOI | 10.1016/j.engappai.2019.03.021 |

### 13.2 Algorithm Overview

STOA is a population-based metaheuristic inspired by the migration and prey-attacking behavior of sooty terns. Two phases:

1. **Exploration (Migration):** Agents move while avoiding collisions with neighbors and converging toward the best-known position.
2. **Exploitation (Attacking):** Agents perform spiral motion toward the best position.

### 13.3 Original STOA — Source-Derived Formulation

> **Verification Status Notice:** **Exact primary-source equation verification remains unresolved.** While the conceptual mechanisms and general equations of the Sooty Tern Optimization Algorithm (STOA) are widely cited from the primary paper (Gaurav Dhiman and Amandeep Kaur, 2019, *Engineering Applications of Artificial Intelligence*, Vol. 82, pp. 148–174), independent primary-source verification of the exact mathematical indexing, variable definitions, and operator bounds from the original journal print article remains pending direct physical/PDF inspection. The equations below reflect the source-derived formulation as reported in the literature without modification, simplification, or stabilization.

#### 13.3.1 Migration Phase (Exploration)

During migration, sooty terns update positions to maintain collision avoidance and flock cohesion toward the fittest search agent:

1. **Collision Avoidance:**
   A collision-free position $\vec{C}_{sa}$ is computed using a linearly decreasing coefficient $S_A$:
   $$\vec{C}_{sa}(t) = S_A \cdot \vec{P}(t)$$
   $$S_A = C_f - \left(t \cdot \frac{C_f}{T_{max}}\right)$$
   where $t \in [0, T_{max}]$ is the current iteration, $T_{max}$ is the maximum iteration budget, and $C_f$ is a control parameter initialized to $2.0$ that decreases linearly to $0.0$. Thus, $S_A = 2 \cdot (1 - t / T_{max})$.

2. **Convergence Toward Best Individual:**
   Agents adjust their trajectory toward the best-so-far position $\vec{P}_{best}(t)$:
   $$\vec{M}_s(t) = B_s \cdot (\vec{P}_{best}(t) - \vec{P}(t))$$
   $$B_s = 0.5 \cdot \vec{r}$$
   where $\vec{r} \sim \text{Uniform}(0, 1)^D$ is a uniform random vector in $D$ dimensions.

3. **Combined Exploration Displacement:**
   $$\vec{D}_s(t) = \vec{C}_{sa}(t) + \vec{M}_s(t) = S_A \cdot \vec{P}(t) + B_s \cdot (\vec{P}_{best}(t) - \vec{P}(t))$$

#### 13.3.2 Attacking Phase (Exploitation via 3D Spiral)

In the hunting phase, agents execute a 3D spiral trajectory around the prey:
$$r = u \cdot \exp(k \cdot v)$$
$$x' = r \cdot \cos(k)$$
$$y' = r \cdot \sin(k)$$
$$z' = r \cdot k$$
where $k \in [0, 2\pi]$ is a randomly chosen angle, $u=1$ and $v=1$ define the default logarithmic spiral geometry, and $r$ is the spiral radius.

**Original Position Update (Dhiman & Kaur, 2019, Eq. 7–8):**
$$\vec{P}(t+1) = \left(\vec{D}_s(t) \odot (x' + y' + z')\right) \odot \vec{P}_{best}(t)$$
where $\odot$ denotes element-wise (Hadamard) product.

---

### 13.4 Critical Mathematical Analysis and Known Literature Failure Modes

Independent mathematical analyses of STOA (notably Kudela, 2022, *Swarm and Evolutionary Computation*; Kwok, 2025, arXiv:2511.17556) have demonstrated two critical numerical pathologies in the literal formulation:

1. **Exponential Magnitude Explosion:**
   Because $r = \exp(k)$ with $k \in [0, 2\pi]$, at $k = 2\pi$ the radius evaluates to $\exp(2\pi) \approx 535.49$. Multiplying spatial displacements by factors exceeding 500 instantaneously throws search candidates far outside bounded hyperparameter intervals (e.g., learning rate $[-4, -2]$ in log-space), forcing the algorithm into constant boundary-clamping saturation.
2. **Structural Zero-Bias (Coordinate-System Dependency):**
   The multiplicative position update $\vec{P}(t+1) \propto \vec{D}_s \odot \vec{P}_{best}$ causes an extreme collapse toward zero whenever search variables approach the origin. For hyperparameter search where optimal points are nonzero (e.g., $\log_{10}(\text{lr}) = -3.0$, batch size $= 32$), this multiplicative formulation severely degrades search efficiency and introduces coordinate-origin bias.

---

### 13.5 Project-Modified Stabilized/Additive STOA Variant

> **Project-Specific Modification Notice:** **This is a project-specific modification inspired by the original STOA. It is not presented as the original Dhiman & Kaur STOA formulation.** It must NOT be described as "primary-source verified STOA." All design choices below (damping, normalization, additive displacement, domain mappings) are engineering decisions introduced by this project to ensure practical numerical stability during neural network hyperparameter optimization.

To address the numerical explosion and coordinate zero-bias of the literal equations, AgriAI specifies the following **Project-Modified Stabilized/Additive STOA Variant**:

#### 13.5.1 Mathematical Specification (Option A: Consistent Best-Guided Formulation)

1. **Normalized and Damped Spiral Geometry:**
   The exponential growth is damped, and the resulting 3D spiral trajectory is normalized to a proper 3-dimensional unit direction vector:
   $$k \sim \text{Uniform}(0, 2\pi)$$
   $$r = \exp(b \cdot k) \quad \text{with project damping factor } b = 0.1$$
   $$x' = r \cdot \cos(k), \quad y' = r \cdot \sin(k), \quad z' = r \cdot k$$
   $$\vec{\Omega}_3 = \frac{[x', y', z']^T}{\|[x', y', z']\|_2 + \epsilon} = \frac{[x', y', z']^T}{\sqrt{(x')^2 + (y')^2 + (z')^2} + \epsilon} = [\Omega_1, \Omega_2, \Omega_3]^T$$
   where $[x', y', z']^T$ is the 3D spiral-direction vector, $\vec{\Omega}_3 \in \mathbb{R}^3$ is its normalized 3D direction vector, and $\epsilon = 10^{-8}$.

2. **Project-Specific Dimensional Extension ($\mathbb{R}^3 \to \mathbb{R}^D$):**
   The AgriAI hyperparameter search space has dimension $D = 4$ ($\theta_1$: learning rate, $\theta_2$: weight decay, $\theta_3$: dropout, $\theta_4$: batch size). To apply the 3D spiral direction to this 4-dimensional search space without introducing arbitrary bias or dimensional mismatch, the project-modified stabilized variant adopts a deterministic cyclic extension:
   $$\vec{\Omega}_D[i] = \vec{\Omega}_3[(i-1) \bmod 3 + 1], \quad i \in \{1, \dots, D\}$$
   For our $D = 4$ hyperparameter vector, this yields explicitly:
   $$\vec{\Omega}_4 = [\Omega_1, \Omega_2, \Omega_3, \Omega_1]^T \in \mathbb{R}^4$$
   This dimensional extension is an **explicit project-specific engineering modification** used solely by the stabilized variant, not part of the original Dhiman & Kaur STOA. It ensures that the element-wise perturbation $\vec{D}_s(t) \odot \vec{\Omega}_4$ is fully defined and dimensionally valid in $\mathbb{R}^4$.

3. **Consistent Best-Guided Displacement (Option A):**
   To resolve any ambiguity between swarm guidance strategies, this project strictly adopts **Option A (Preserve best-guided behavior)**: $\vec{P}_{best} \in \mathbb{R}^4$ is used consistently throughout the formulation. Search agents are guided toward the best-so-far solution $\vec{P}_{best}$ without unintended random-agent drift:
   $$\vec{C}_{sa}(t) = S_A(t) \cdot \vec{P}(t)$$
   $$\vec{M}_s(t) = 0.5 \cdot \vec{r} \odot (\vec{P}_{best}(t) - \vec{P}(t))$$
   $$\vec{D}_s(t) = \vec{C}_{sa}(t) + \vec{M}_s(t) = S_A(t) \cdot \vec{P}(t) + 0.5 \cdot \vec{r} \odot (\vec{P}_{best}(t) - \vec{P}(t))$$
   with linearly decaying collision parameter $S_A(t) = 2 \cdot (1 - t / T_{max})$ and $\vec{r} \sim \text{Uniform}(0, 1)^4$.

4. **Additive Position Update (Search Space Invariance):**
   Rather than multiplicative Hadamard collapse, the update applies an additive step scaled by the exploration displacement:
   $$\vec{P}(t+1) = \vec{P}_{best}(t) + \vec{D}_s(t) \odot \vec{\Omega}_4$$

4. **Hyperparameter Domain Mapping:**
   - **Learning rate:** Searched in log-space: $\theta_1 \in [-4.0, -2.0] \implies \text{lr} = 10^{\theta_1}$.
   - **Weight decay:** Searched in log-space: $\theta_2 \in [-6.0, -2.0] \implies \text{wd} = 10^{\theta_2}$.
   - **Dropout:** Searched in linear space: $\theta_3 \in [0.0, 0.5] \implies p_{drop} = \theta_3$.
   - **Batch size:** Searched in continuous index $\theta_4 \in [0, 2] \implies \text{batch\_size} \in \{16, 32, 64\}$ mapped via nearest neighbor.

5. **Boundary Handling:**
   $$\vec{P}_i = \text{clip}(\vec{P}_i, lb_i, ub_i)$$

---

### 13.6 The Three Optimizer Experimental Conditions

The AgriAI experimental framework explicitly distinguishes three optimizer conditions to ensure complete scientific clarity:

| Condition Name | Code Identifier | Description & Purpose |
|----------------|-----------------|----------------------|
| **Original/Strict STOA** | `original_strict` | Evaluates the source-derived STOA procedure (Dhiman & Kaur, 2019) as faithfully as possible to empirically document its behavior and known numerical properties under controlled conditions. |
| **Stabilized STOA** | `stabilized` | Evaluates the project-modified stabilized/additive STOA variant to test whether the proposed numerical corrections provide effective hyperparameter tuning. **(Default project optimizer)** |
| **Random Search Control** | `random_search` | Provides a non-metaheuristic search baseline evaluated under an identical evaluation budget (25 dev / 100 final evaluations) to test whether metaheuristic search provides any measurable advantage. |

> **Configuration Rule:** The default optimizer in configuration files is labeled as `optimizer.name: "stoa"` with `optimizer.variant: "stabilized"`. The configuration explicitly notes that this default is a project-modified stabilized variant, while retaining full configuration support for `original_strict` and `random_search`.

---

### 13.7 Critical Claims Rules

The following claims are STRICTLY PROHIBITED in all project documents:

- "STOA guarantees convergence to the global optimum"
- "STOA eliminates local minima"
- "STOA is provably better than Random Search or other metaheuristics"
- "STOA guarantees superior classification accuracy"

Required empirical framing:

> "STOA is used as a gradient-free population-based hyperparameter optimization method, and its effectiveness will be evaluated experimentally against a comparable Random Search baseline under an equal evaluation budget."

---

## 14. Hyperparameter Search Space

| Hyperparameter | Range | Scale |
|---------------|-------|-------|
| Learning rate | [1e-4, 1e-2] | Log |
| Weight decay | [1e-6, 1e-2] | Log |
| Dropout probability | [0.0, 0.5] | Linear |
| Batch size | {16, 32, 64} | Discrete |

All 4 hyperparameters optimized simultaneously.

---

## 15. Fitness Function

    fitness(h) = 1 - macro_F1(h, D_val)

- Minimizing (1 - macro_F1) is equivalent to maximizing macro F1.
- Macro F1 treats all classes equally; preferred over accuracy for imbalanced datasets.
- Test set is NEVER used during fitness evaluation.

---

## 16. Baseline CNN

### 16.1 Architecture

```
Input (H x W x 3)
      |
Block 1: Conv(3x3, 32) -> BN -> ReLU -> MaxPool(2x2)
      |
Block 2: Conv(3x3, 64) -> BN -> ReLU -> MaxPool(2x2)
      |
Block 3: Conv(3x3, 128) -> BN -> ReLU -> MaxPool(2x2)
      |
Global Average Pooling
      |
Linear(128, 10) -> Logits
```

Target: ~500K parameters (to be confirmed after implementation).

### 16.2 Purpose

Without a baseline:
- It is impossible to contextualize LS-HGNet's performance.
- The added architectural complexity cannot be justified experimentally.
- The project would compare LS-HGNet only against itself across optimizer conditions.

Baseline uses default hyperparameters; no hyperparameter optimization applied.

---

## 17. Random Search Control

| Property | Development | Final |
|----------|-------------|-------|
| STOA budget | 5 agents x 5 iters = 25 evals | 10 agents x 10 iters = 100 evals |
| Random Search budget | 25 evaluations | 100 evaluations |
| Search space | Same as STOA | Same as STOA |
| Fitness function | Same as STOA | Same as STOA |

**Why equal-budget:** If STOA uses 100 evaluations and Random Search uses 25, any difference is attributable to budget, not algorithm design. Equal-budget is the standard scientific protocol for optimizer comparison.

---

## 18. Experiment Matrix

| ID | Model | Optimizer / Hyperparameters | Purpose |
|----|-------|-----------------------------|---------|
| EXP_001 | Baseline CNN | Default hyperparameters | Experimental architecture control (~500K params) |
| EXP_002 | LS-HGNet | Default hyperparameters | Architectural baseline without hyperparameter tuning |
| EXP_003_original | LS-HGNet | Best from Original STOA (`original_strict`) | Evaluate source-derived STOA procedure as faithfully as possible |
| EXP_003_stabilized | LS-HGNet | Best from Stabilized STOA (`stabilized`) | Evaluate project-modified stabilized/additive STOA variant (Default) |
| EXP_004 | LS-HGNet | Best from Random Search (`random_search`) | Equal-budget non-metaheuristic search control (25 dev / 100 final) |

Models are NOT ranked before experiments. Results are measured, not predicted.

---

## 19. Evaluation Metrics

### 19.1 Classification

| Metric | Scope |
|--------|-------|
| Accuracy | Overall |
| Precision | Macro, Weighted, Per-class |
| Recall | Macro, Weighted, Per-class |
| F1 Score | Macro, Weighted, Per-class |
| Confusion Matrix | 10x10 |
| ROC-AUC | Per-class (one-vs-rest) |
| PR-AUC | Per-class |

### 19.2 Computational

| Metric | Unit |
|--------|------|
| Parameter count | Integer |
| Trainable parameter count | Integer |
| Model size on disk | MB |
| CPU inference latency | ms/image (batch=1) |
| Peak GPU memory | MB (if applicable) |

### 19.3 Optimization

| Metric | Description |
|--------|-------------|
| Best fitness | Best (1 - macro_F1) achieved |
| Number of evaluations | Integer |
| Optimization wall-clock time | Seconds |
| Convergence history | Fitness vs. evaluation number |

> **All metrics must be computed from actual model runs. No metric may be manually typed or fabricated.**

---

## 20. Explainability — Grad-CAM

### 20.1 Procedure

1. Forward pass: record activations A^k at last decoder block of final hourglass stack.
2. Backward pass: compute gradient dy^c / dA^k.
3. Neuron importance: alpha^c_k = (1/Z) * sum_{i,j} (dy^c / dA^k_{ij}).
4. Weighted combination: L^c = ReLU(sum_k alpha^c_k * A^k).
5. Upsample to input resolution.
6. Apply colormap and overlay on original image.

### 20.2 Target Layer

Output of the final decoder block of the last hourglass stack, before Global Average Pooling. (Subject to adjustment after implementation is confirmed.)

### 20.3 Limitations Disclosure

> Grad-CAM shows image regions contributing to the model's prediction. It does NOT prove that the model identified the true biological cause of the disease. Highlighted regions may reflect superficial image features. Outputs must be interpreted with appropriate caution.

---

## 21. Calibration

### 21.1 Temperature Scaling

Post-hoc method. A learned scalar T > 0 divides the logits:

    p_hat^k = exp(z^k / T) / sum_j exp(z^j / T)

### 21.2 Fitting Procedure

1. Train model to convergence.
2. Minimize NLL on validation data: T* = argmin_T NLL(D_val; T).
3. Apply T* at inference time.

The test set is NEVER used to fit T.

### 21.3 Metrics

| Metric | When |
|--------|------|
| ECE (Expected Calibration Error) | Before and after calibration |
| Reliability diagram | Before and after calibration |
| Mean raw confidence | vs. actual accuracy |
| Mean calibrated confidence | vs. actual accuracy |

---

## 22. Robustness Testing

### 22.1 Perturbations

| Perturbation | Description |
|-------------|-------------|
| Brightness increase | Pixel values multiplied by factor > 1 |
| Brightness decrease | Pixel values multiplied by factor < 1 |
| Contrast change | Linear contrast scaling |
| Gaussian noise | Zero-mean Gaussian noise, varying sigma |
| Gaussian blur | Varying kernel size |
| JPEG compression | Varying quality factor |
| Low-light transformation | Simulated dark/underexposed conditions |

Each perturbation is applied at multiple severity levels.

### 22.2 Robustness Claims Rule

> The model MUST NOT be described as "robust" unless measured results support that claim. This section presents accuracy-vs-severity degradation curves, not qualitative judgments.

---

## 23. Ablation Studies

Planned ablations (to be executed during M2/M3, not M0):

| ID | Ablation |
|----|---------|
| ABL-01 | MultiDilatedBlock vs. single-dilation bottleneck |
| ABL-02 | With vs. without inter-stack skip connections |
| ABL-03 | Stack count: 1 vs. 2 vs. 3 |
| ABL-04 | With vs. without morphological preprocessing (top-hat/bottom-hat) |
| ABL-05 | 128x128 vs. 224x224 input resolution |
| ABL-06 | Depthwise-separable vs. standard convolution |

**On morphological preprocessing (ABL-04):** Top-hat and bottom-hat morphological transforms may enhance disease lesion boundaries. This is treated as an experimental preprocessing branch, not part of the core pipeline. ABL-04 compares pipeline-with vs. pipeline-without morphological preprocessing. It is NOT automatically included in the main pipeline.

---

## 24. Deployment Concept

### 24.1 Flask REST API

- `POST /predict` — Input: leaf image. Output: JSON with class, confidence, calibrated confidence, Grad-CAM overlay (base64).
- `GET /health` — Health check.

### 24.2 React Dashboard

- Image upload interface
- Prediction display with confidence bar
- Grad-CAM heatmap visualization
- Disease information panel

### 24.3 Optional Extensions

- ONNX export for edge deployment
- IoT integration concept (not part of core evaluated deliverables)

---

## 25. Limitations

The following limitations must appear in all final reports and presentations:

1. PlantVillage images are from controlled conditions. Generalization to field images is not guaranteed.
2. Class imbalance may exist and must be measured and discussed.
3. Single-crop scope: trained only on tomato leaves.
4. STOA is a heuristic with no convergence guarantee to the global optimum.
5. Optimization quality depends on search space definition and evaluation budget.
6. Grad-CAM provides coarse localization and does not causally identify disease regions.
7. Calibration does not guarantee correctness; it adjusts confidence, not accuracy.
8. Experiments were conducted on a 16 GB laptop; results may differ on other hardware.
9. This is a research prototype. It is NOT a replacement for expert agricultural diagnosis.
10. No pesticide dosage recommendations are generated or implied.
11. IoT and edge deployment are optional and are not part of core evaluated deliverables.

---

## 26. Originality Statement

> This project is an independent adaptation and extension of concepts reported in prior LS-HGNet and Sooty Tern Optimization research. The original research context (UAV-based maize tassel detection and counting), dataset, task, implementation, experiments, and reported results are not reproduced as original work. The present project independently implements and evaluates a tomato leaf disease classification system using related architectural and optimization concepts, together with additional explainability (Grad-CAM), confidence calibration (temperature scaling), robustness testing, and deployment components (Flask API, React dashboard). All source code, experimental results, figures, and numerical values are independently produced. No text, code, or figures from the referenced source papers are copied, paraphrased without attribution, or claimed as our own findings.

---

## 27. Reproducibility Requirements

| Requirement | Status |
|-------------|--------|
| Random seeds documented (42, 123, 456 for final runs) | Set in M0 |
| Split CSV files committed before training | Pending M1 |
| requirements.txt / environment.yaml committed | Pending M1 |
| Configuration YAML files committed | Committed in M0 |
| Model weights stored outside Git | Policy set in M0 .gitignore |
| All results generated by code, not manually typed | Policy set in M0 |
| Convergence plots saved as artifacts | Pending M1 |
| Grad-CAM images saved per experiment | Pending M1 |

---

## 28. Open Questions and Verification Classification

### 28.1 Resolved Research Specifications

The following theoretical, architectural, and experimental specifications are formally resolved and defined:

- **RES-01 (STOA Primary Authorship):** Confirmed as Gaurav Dhiman and Amandeep Kaur (2019), *EAAI*, Vol. 82, pp. 148–174, DOI: 10.1016/j.engappai.2019.03.021.
- **RES-02 (Smart Farming Source Attribution):** Confirmed as Krishnan et al. (2024), *IJERR*, Vol. 37(spl), pp. 96–108, DOI: 10.52756/ijerr.2024.v37spl.008.
- **RES-03 (Originality and Adaptation Boundary):** Confirmed distinction between source UAV maize tassel detection with top-hat/bottom-hat filtering and our tomato leaf disease classification on PlantVillage.
- **RES-04 (Original vs. Stabilized STOA Distinction):** The source-derived formulation and our project-modified stabilized/additive variant are formally separated into distinct specifications.
- **RES-05 (Stabilized STOA Variant Design):** Project-specific modifications formalized with normalized perturbation $\vec{\Omega}$, damped spiral $b=0.1$, additive update, and consistent best-guided formulation (Option A: $\vec{P}_{best}$ throughout; no $\vec{P}_{rand}$).
- **RES-06 (Three Optimizer Controls):** Explicitly separated into `original_strict`, `stabilized`, and `random_search` with equal-budget protocol (25 dev / 100 final).
- **RES-07 (Data Split & Augmentation Policy):** Stratified 70/15/15 split finalized; random vertical flip excluded from default training pipeline on biological validity grounds (retained as optional ablation ABL-05).
- **RES-08 (Search Space & Fitness Objective):** Bounded hyperparameter space defined; fitness objective set to $1 - \text{macro\_F1}$ on validation set.
- **RES-09 (Original STOA Equation Verification — M0 Formal Resolution):** The STOA formulation in Section 13.3 is accepted as **literature-verified** under the following formal resolution:
  - The exact STOA equations ($S_A$, $\vec{C}_{sa}$, $\vec{M}_s$, $\vec{D}_s$, and spiral geometry) as documented are consistently reproduced across multiple independent secondary sources (ResearchGate, PubMed/NIH, AIP-affiliated repositories, peer-reviewed survey papers) citing Dhiman & Kaur (2019), *EAAI*, 82, 148–174.
  - The specific collision avoidance parameter $S_A = C_f - (z \times C_f / T_{max})$ with $C_f = 2$, the convergence coefficient $B_s = 0.5 \cdot \vec{r}$, and the 3D spiral geometry ($x' = r\cos k$, $y' = r\sin k$, $z' = rk$) match across all consulted secondary sources.
  - The known numerical pathologies (exponential explosion at $k = 2\pi$, multiplicative zero-bias) are independently corroborated by formal mathematical analyses in Kudela (2022) and Kwok (2025), providing additional indirect confirmation that our documented equations faithfully represent the original formulation.
  - **Direct full-text publisher PDF access to Dhiman & Kaur (2019) was not achieved during M0.** This limitation is acknowledged and formally accepted: the M0 lock proceeds on the basis of strong multi-source literature consensus. If discrepancies are discovered upon M1 implementation (e.g., via institutional library access), the equations and this record must be updated before publication.
  - **Resolution Status: Formally Accepted (Literature-Verified, Publisher-Unverified).** This is the final M0 classification for RES-09.

### 28.2 Deferred to Implementation Milestones (M1/M2)

The following questions are not research specification blockers. They are practical questions resolvable only during implementation and are formally deferred:

| ID | Deferred Question | Resolution Milestone |
|----|-------------------|---------------------|
| DEF-01 | PlantVillage exact per-class image counts and verified class distribution | M1: Count files on disk after download |
| DEF-02 | PlantVillage dataset licensing terms (CC0 vs. other) | M1: Inspect dataset README/LICENSE |
| DEF-03 | Hardware memory feasibility: 224×224 at batch size 32 on 16 GB RAM | M1: Memory profiling; reduce to batch 16 if needed |
| DEF-04 | Class-weighted Cross-Entropy loss benefit on macro F1 | M2: Ablation ABL-02 after measuring class imbalance |
| DEF-05 | Final dropout rate selection | M1/M2: Explored within STO search space [0.0, 0.5] |
| DEF-06 | Optional edge deployment ONNX runtime latency | OBJ-11 (optional): Benchmark during deployment phase |

---

## 29. References

1. Dhiman, G., & Kaur, A. (2019). STOA: A bio-inspired based optimization algorithm for industrial engineering problems. *Engineering Applications of Artificial Intelligence*, 82, 148–174. DOI: 10.1016/j.engappai.2019.03.021.

2. Krishnan, V. G., Vikranth, B., Sumithra, M., Laxmi, B. P., & Gowri, B. S. (2024). Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model. *International Journal of Experimental Research and Review*, 37(spl), 96–108. DOI: 10.52756/ijerr.2024.v37spl.008.

3. Newell, A., Yang, K., & Deng, J. (2016). Stacked hourglass networks for human pose estimation. *ECCV*, pp. 483-499. arXiv:1603.06937.

4. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.

5. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861.

6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, pp. 618-626. DOI: 10.1109/ICCV.2017.74.

7. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, pp. 1321-1330.

8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, pp. 770-778.

9. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR*. arXiv:1511.07122.

10. Kudela, J. (2022). A critical analysis of recent metaheuristics: origin bias and structural flaws. *Swarm and Evolutionary Computation*, 74, 101133.

11. Kwok, C. (2025). On the origin bias and numerical divergence in population metaheuristics: The case of Sooty Tern Optimization. arXiv:2511.17556.

---

*End of Research Specification — M0 LOCKED — Research Specification Approved 2026-10-01*
