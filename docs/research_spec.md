# Research Specification — AgriAI
## Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis

**Document version:** M0-DRAFT  
**Date:** 2026-10-01  
**Status:** AWAITING APPROVAL  

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
**Context:** This paper applied a Lightweight Stacked Hourglass Network (LS-HGNet) with Sooty Tern Optimization for UAV-based maize tassel detection and counting in a smart farming context.  
**What we use:** The conceptual combination of (a) lightweight stacked-hourglass architecture and (b) STOA-based optimization. The original task (maize tassel detection/counting from UAV imagery) differs fundamentally from our task (tomato leaf disease classification from leaf images).  
**What we do NOT use:** Original code, original experimental results, UAV imagery, maize-specific design choices.

> **NOTE:** The exact title, authors, publication venue, and DOI of this paper must be verified during M1 literature review. The description above is based on project planning documents. It is marked **TO BE VERIFIED** in the reference list.

### 8.2 STOA Original Paper

**Title:** STOA: A bio-inspired based optimization algorithm for industrial engineering problems  
**Authors:** Gaurav Dhiman and Vijay Kumar  
**Year:** 2019  
**Venue:** Engineering Applications of Artificial Intelligence  
**DOI:** TO BE VERIFIED (candidate: 10.1016/j.engappai.2018.09.001)  
**What we use:** Mathematical formulation of the STOA algorithm (collision avoidance, migration update, spiral attack behavior)

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
| Random vertical flip | p = 0.5 |
| Random rotation | +/-30 degrees, p = 0.5 |
| Color jitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 |
| Random resized crop | scale=(0.7, 1.0) |

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
| Authors | Gaurav Dhiman and Vijay Kumar |
| Year | 2019 |
| Venue | Engineering Applications of Artificial Intelligence |
| DOI | TO BE VERIFIED |

### 13.2 Algorithm Overview

STOA is a population-based metaheuristic inspired by migration and prey-attacking behavior of sooty terns. Two phases:

1. **Exploration (Migration):** Agents move while avoiding collisions and converging toward the best-known position.
2. **Exploitation (Attacking):** Agents perform spiral motion toward the best position.

### 13.3 Verified Formulation

#### Phase 1: Exploration

**Collision avoidance:**

    x_ca(t) = C_f * x(t)

    C_f = 2 - t/T_max

where t = current iteration, T_max = maximum iterations. C_f decreases from 2 to 1.

> **DISCREPANCY NOTE:** Secondary sources show C_a = 2 - 2*(t/T_max), giving a range of 2->0 instead of 2->1. The exact coefficient in the original paper (Dhiman & Kumar, 2019) must be verified from the primary source before implementation. This is recorded as an open question.

**Convergence toward best agent:**

    x_cv(t) = D_f * (x_best(t) - x(t))
    D_f = r_1 * x(t),  r_1 ~ Uniform(0, 1)

**Migration update:**

    x_m(t) = x_ca(t) + x_cv(t)

#### Phase 2: Exploitation (Spiral Attack)

Spiral parameters:

    theta ~ Uniform(0, 2*pi)
    rho = exp(theta)
    alpha = rho * sin(theta)
    beta  = rho * cos(theta)
    gamma = rho * theta

Position update:

    x(t+1) = D_f * alpha * beta * gamma + x_best(t)

> **IMPORTANT CAVEAT:** The combined product form above may differ from the original paper's vector-form spiral update. The original likely defines separate component updates. This MUST be verified from Dhiman & Kumar (2019) before implementation. Any deviation from the original will be documented in architecture_decisions.md with classification as: correction / adaptation / unsupported modification.

### 13.4 Boundary Handling

    x_i = clip(x_i, lb_i, ub_i)

Clipping is chosen as the engineering decision. Alternatives (reflection, random resampling) may be ablated.

### 13.5 Critical Claims Rules

The following are PROHIBITED in all project documents:

- "STOA guarantees convergence to the global optimum"
- "STOA eliminates local minima"
- "STOA is provably better than Random Search"

Required framing:

- "Whether STOA provides a measurable advantage over equal-budget Random Search is an experimental question answered by measured results."

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

| ID | Model | Hyperparameters | Purpose |
|----|-------|----------------|---------|
| EXP_001 | Baseline CNN | Default | Experimental control |
| EXP_002 | LS-HGNet (dev config) | Default | Architecture evaluation |
| EXP_003 | LS-HGNet + STOA | Best from STOA search | Optimizer benefit test |
| EXP_004 | LS-HGNet + Random Search | Best from RS search | Equal-budget control |

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
| Random seeds documented (42, 123, 456 for final runs) | Frozen in M0 |
| Split CSV files committed before training | Pending M1 |
| requirements.txt / environment.yaml committed | Pending M1 |
| Configuration YAML files committed | Committed in M0 |
| Model weights stored outside Git | Policy set in M0 .gitignore |
| All results generated by code, not manually typed | Policy set in M0 |
| Convergence plots saved as artifacts | Pending M1 |
| Grad-CAM images saved per experiment | Pending M1 |

---

## 28. References

1. Dhiman, G., & Kumar, V. (2019). STOA: A bio-inspired based optimization algorithm for industrial engineering problems. *Engineering Applications of Artificial Intelligence*, 82, 148-174. DOI: **TO BE VERIFIED**.

2. [Source Smart Farming Paper — TO BE VERIFIED]. Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model. Authors, Year, Venue, DOI — to be located during M1.

3. Newell, A., Yang, K., & Deng, J. (2016). Stacked hourglass networks for human pose estimation. *ECCV*, pp. 483-499. arXiv:1603.06937.

4. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.

5. Howard, A. G., et al. (2017). MobileNets: Efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861.

6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV*, pp. 618-626. DOI: 10.1109/ICCV.2017.74.

7. Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*, pp. 1321-1330.

8. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, pp. 770-778.

9. Yu, F., & Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. *ICLR*. arXiv:1511.07122.

---

*End of Research Specification — M0 Draft — Status: AWAITING APPROVAL*
