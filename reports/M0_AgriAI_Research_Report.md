# AgriAI — M0 Research Report

## STO-Optimized LS-HGNet for Explainable Crop Disease Classification in Smart Farming

---

| Field | Value |
|-------|-------|
| **Project Name** | AgriAI |
| **Project Title** | STO-Optimized LS-HGNet for Explainable Crop Disease Classification in Smart Farming |
| **Full Title** | Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis |
| **Research Stage** | M0 — Research Specification |
| **Status** | ✅ **LOCKED** |
| **Date** | 2026-10-01 (Locked) / 2026-10-02 (Final Verification & Archive) |
| **Team** | 3-member final-year project team |
| **Target Hardware** | 16 GB laptop (development); standard GPU optional for final experiments |

---

## 1. Abstract

Tomato is among the most widely cultivated and economically significant crops globally. Crop diseases cause substantial yield losses, and accurate, timely, and explainable automated disease diagnosis can reduce these losses and support precision agriculture. This project investigates whether a lightweight, stacked-hourglass-inspired convolutional neural network (LS-HGNet), with hyperparameters optimized using the Sooty Tern Optimization Algorithm (STOA), can deliver accurate, computationally efficient, explainable, calibrated, and robustly-evaluated tomato leaf disease classification.

The PlantVillage tomato subset is used as the primary dataset, targeting 10 classes (1 healthy + 9 diseased). The project extends concepts from prior smart-farming and bio-inspired optimization research to a new task (disease classification), with additional components including:

- **Grad-CAM explainability** — visual explanations of model predictions
- **Post-hoc temperature-scaling calibration** — reliable confidence estimates
- **Systematic robustness testing** — evaluation under 7 synthetic perturbation types
- **Flask REST API and React dashboard** — deployable inference interface

A simple CNN baseline and a Random Search optimizer are included to provide experimental controls. A project-modified stabilized/additive STOA variant is defined to address known numerical pathologies in the original STOA formulation.

This is an independent adaptation and extension project. No original research results, code, or text from the referenced source papers are reproduced.

---

## 2. Problem Statement

Tomato leaf diseases are visually diverse and can be difficult to distinguish manually. Automated classification systems require:

1. **Sufficient classification accuracy** across multiple disease classes
2. **Computational efficiency** suitable for resource-constrained deployment
3. **Explainability** to support agricultural decision-making
4. **Calibrated confidence** to convey reliable uncertainty estimates
5. **Robustness** under realistic image degradation

Existing studies typically optimize only one or two of these properties. This project explores all five in an integrated framework.

---

## 3. Research Gap

While deep learning methods for tomato disease classification are well-studied, the following combination has not been thoroughly investigated in the literature surveyed:

- A lightweight stacked-hourglass-inspired architecture adapted from UAV vision tasks to disease classification
- Sooty Tern Optimization used for joint hyperparameter search across learning rate, weight decay, dropout, and batch size, compared against an equal-budget Random Search under controlled conditions
- Simultaneous evaluation of explainability (Grad-CAM), calibration (temperature scaling), and robustness (synthetic perturbation) within one experimental framework

---

## 4. Research Question

> **Can an STO-optimized lightweight LS-HGNet architecture provide accurate, computationally efficient, explainable, calibrated, and robust tomato leaf disease classification, and does STO hyperparameter optimization yield a measurable advantage over equal-budget Random Search?**

The research question is falsifiable and will be answered through measured experimental results. No performance claims are made prior to experiments.

---

## 5. Research Objectives

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

## 6. Reference Research

### 6.1 Primary Source — LS-HGNet + STOA Applied to Smart Farming

| Field | Value |
|-------|-------|
| **Title** | Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model |
| **Authors** | V. Gokula Krishnan, B. Vikranth, M. Sumithra, B. Prathusha Laxmi, B. Shyamala Gowri |
| **Journal** | International Journal of Experimental Research and Review |
| **Volume** | 37 (special issue) |
| **Pages** | 96–108 |
| **Year** | 2024 |
| **DOI** | 10.52756/ijerr.2024.v37spl.008 |
| **Publication Date** | 30 March 2024 |

**Original application context:**

- **Domain:** Precision smart farming — automated detection and counting of maize tassels in drone/UAV aerial photography
- **Preprocessing:** Morphological bottom-hat and top-hat filtering to mitigate irregular sunlight reflections and ambient shadow noise in aerial foliage
- **Network:** Lightweight Weight-Stacked Hourglass Network (LS-HGNet) with stacked hourglass modules, dilated convolutions, and inter-module skip connections
- **Optimization:** Sooty Tern Optimization Algorithm (STOA) for tuning classification hyperparameters, reporting an empirical maize tassel detection accuracy of 98.7%

> **Critical distinction:** The original paper concerns UAV-based maize tassel detection and counting. AgriAI does NOT reproduce this task. AgriAI adapts the conceptual combination of (1) a multi-stack hourglass network and (2) STOA metaheuristic tuning to a completely different application: 10-class tomato leaf disease classification on the PlantVillage dataset.

**What AgriAI adopts conceptually:**

- The synergistic combination of a multi-stack hourglass network parameterized for lightweight vision inference and gradient-free bio-inspired STOA metaheuristic tuning for hyperparameter search

**What AgriAI does NOT adopt:**

- Maize tassel counting, UAV imagery, morphological preprocessing as mandatory core pipeline, and original source weights or code

### 6.2 STOA Original Paper

| Field | Value |
|-------|-------|
| **Title** | STOA: A bio-inspired based optimization algorithm for industrial engineering problems |
| **Authors** | Gaurav Dhiman and Amandeep Kaur |
| **Journal** | Engineering Applications of Artificial Intelligence |
| **Volume** | 82 |
| **Pages** | 148–174 |
| **Year** | 2019 |
| **DOI** | 10.1016/j.engappai.2019.03.021 |

### 6.3 Stacked Hourglass Network

| Field | Value |
|-------|-------|
| **Title** | Stacked Hourglass Networks for Human Pose Estimation |
| **Authors** | Alejandro Newell, Kaiyu Yang, Jia Deng |
| **Year** | 2016 |
| **Venue** | ECCV |
| **arXiv** | 1603.06937 |

### 6.4 PlantVillage Dataset

| Field | Value |
|-------|-------|
| **Title** | An open access repository of images on plant health to enable the development of mobile disease diagnostics |
| **Authors** | David P. Hughes, Marcel Salathé |
| **Year** | 2015 |
| **arXiv** | 1511.08060 |

### 6.5 Additional References

| Source | What we use |
|--------|-------------|
| Howard et al. (2017), MobileNets (arXiv:1704.04861) | Depthwise separable convolution as a lightweight building block |
| Selvaraju et al. (2017), Grad-CAM (ICCV, DOI: 10.1109/ICCV.2017.74) | Grad-CAM technique for post-hoc explainability |
| Guo et al. (2017), On Calibration of Modern Neural Networks (ICML) | Temperature scaling as post-hoc calibration; ECE metric |
| He et al. (2016), Deep Residual Learning (CVPR) | Residual block design |
| Yu & Koltun (2016), Dilated Convolutions (ICLR, arXiv:1511.07122) | Dilated convolution concept |
| Kudela (2022), Swarm and Evolutionary Computation, 74, 101133 | Independent mathematical critique of STOA numerical pathologies |
| Kwok (2025), arXiv:2511.17556 | Independent analysis of STOA origin bias and numerical divergence |

---

## 7. STOA Research Foundation

### 7.1 Algorithm Identity

The Sooty Tern Optimization Algorithm (STOA) is a population-based metaheuristic optimizer proposed by Gaurav Dhiman and Amandeep Kaur in 2019, published in *Engineering Applications of Artificial Intelligence*, Vol. 82, pp. 148–174 (DOI: 10.1016/j.engappai.2019.03.021).

STOA is inspired by the migration and prey-attacking behavior of sooty terns and consists of two phases:

1. **Exploration (Migration):** Agents move while avoiding collisions with neighbors and converging toward the best-known position.
2. **Exploitation (Attacking):** Agents perform spiral motion toward the best position.

### 7.2 Role in AgriAI

STOA is used as a gradient-free population-based hyperparameter optimization method. Its effectiveness is evaluated experimentally against a comparable Random Search baseline under an equal evaluation budget. STOA is NOT claimed to guarantee convergence to the global optimum, eliminate local minima, or be provably better than Random Search.

### 7.3 Original STOA Formulation (Source-Derived)

> **Verification Status: Literature-Verified, Publisher-Unverified**
>
> The documented STOA equations are consistent with independently consulted secondary literature (ResearchGate, PubMed/NIH, AIP-affiliated repositories, peer-reviewed survey papers). The original Dhiman & Kaur (2019) publisher PDF was not directly inspected during M0. Therefore, no claim of "publisher-verified" or "primary-source verified" status is made. If institutional/publisher access later reveals a discrepancy, the equations must be updated before publication.

#### Migration Phase (Exploration)

1. **Collision Avoidance:**

   $$\vec{C}_{sa}(t) = S_A \cdot \vec{P}(t)$$

   $$S_A = C_f - \left(t \cdot \frac{C_f}{T_{max}}\right)$$

   where $t \in [0, T_{max}]$ is the current iteration, $T_{max}$ is the maximum iteration budget, and $C_f = 2.0$ decreases linearly to $0.0$. Thus $S_A = 2 \cdot (1 - t / T_{max})$.

2. **Convergence Toward Best Individual:**

   $$\vec{M}_s(t) = B_s \cdot (\vec{P}_{best}(t) - \vec{P}(t))$$

   $$B_s = 0.5 \cdot \vec{r}$$

   where $\vec{r} \sim \text{Uniform}(0, 1)^D$.

3. **Combined Exploration Displacement:**

   $$\vec{D}_s(t) = \vec{C}_{sa}(t) + \vec{M}_s(t) = S_A \cdot \vec{P}(t) + B_s \cdot (\vec{P}_{best}(t) - \vec{P}(t))$$

#### Attacking Phase (Exploitation via 3D Spiral)

$$r = u \cdot \exp(k \cdot v)$$

$$x' = r \cdot \cos(k), \quad y' = r \cdot \sin(k), \quad z' = r \cdot k$$

where $k \in [0, 2\pi]$ is a randomly chosen angle, $u=1$ and $v=1$ define the default logarithmic spiral geometry.

**Original Position Update (Dhiman & Kaur, 2019, Eq. 7–8):**

$$\vec{P}(t+1) = \left(\vec{D}_s(t) \odot (x' + y' + z')\right) \odot \vec{P}_{best}(t)$$

where $\odot$ denotes element-wise (Hadamard) product.

### 7.4 Known Numerical Pathologies

Independent mathematical analyses (Kudela, 2022; Kwok, 2025) have demonstrated two critical numerical pathologies in the literal formulation:

1. **Exponential Magnitude Explosion:** $r = \exp(k)$ with $k = 2\pi$ evaluates to $\exp(2\pi) \approx 535.49$. Multiplying spatial displacements by factors exceeding 500 throws search candidates far outside bounded hyperparameter intervals, forcing constant boundary-clamping saturation.

2. **Structural Zero-Bias (Coordinate-System Dependency):** The multiplicative position update $\vec{P}(t+1) \propto \vec{D}_s \odot \vec{P}_{best}$ causes extreme collapse toward zero whenever search variables approach the origin, severely degrading search efficiency.

---

## 8. Original STOA vs Project-Modified STOA

This distinction is mandatory for scientific integrity.

| Component | Original STOA (Dhiman & Kaur, 2019) | AgriAI Project Modification |
|-----------|--------------------------------------|----------------------------|
| **Source** | Published formulation | — |
| **Verification status** | Literature-Verified, Publisher-Unverified | — |
| **Spiral radius** | $r = \exp(k)$, unbounded | $r = \exp(b \cdot k)$ with damping $b = 0.1$ |
| **Direction normalization** | Not applied | Normalized to unit direction: $\vec{\Omega}_3 = [x', y', z']^T / (\|[x', y', z']\|_2 + \epsilon)$ |
| **Position update** | Multiplicative Hadamard: $\vec{P}(t+1) = (\vec{D}_s \odot (x'+y'+z')) \odot \vec{P}_{best}$ | Additive: $\vec{P}(t+1) = \vec{P}_{best}(t) + \vec{D}_s(t) \odot \vec{\Omega}_4$ |
| **$\vec{P}_{best}$ guidance** | Source concept | Preserved (Option A: consistent best-guided) |
| **$P_{rand}$** | Not used in stabilized formulation | Not used |
| **3D → D-dimension mapping** | Not specified | Project-specific deterministic cyclic extension |
| **Boundary handling** | Not fully specified | Coordinate-wise clamping: $\vec{P}_i = \text{clip}(\vec{P}_i, lb_i, ub_i)$ |
| **Hyperparameter domain mapping** | Not specified | Log-space for LR and WD; linear for dropout; discrete index for batch size |

> **Attribution:** All modifications in the right column are **project-specific engineering decisions**. They are NOT attributed to the original Dhiman & Kaur STOA paper and must NOT be described as source-derived.

---

## 9. Stabilized STOA Mathematical Specification

> **Project-Specific Modification Notice:** This is a project-specific modification inspired by the original STOA. It is not presented as the original Dhiman & Kaur STOA formulation. It must NOT be described as "primary-source verified STOA." All design choices below are engineering decisions introduced by this project for practical numerical stability during neural network hyperparameter optimization.

### 9.1 Search Vector

The hyperparameter search space has dimension $D = 4$:

$$\vec{P} = [\theta_1, \theta_2, \theta_3, \theta_4]^T$$

where:
- $\theta_1$: learning rate (log-space)
- $\theta_2$: weight decay (log-space)
- $\theta_3$: dropout probability (linear space)
- $\theta_4$: batch size (continuous index, mapped to discrete set)

### 9.2 Normalized and Damped Spiral Geometry

$$k \sim \text{Uniform}(0, 2\pi)$$

$$r = \exp(b \cdot k) \quad \text{with project damping factor } b = 0.1$$

$$x' = r \cdot \cos(k), \quad y' = r \cdot \sin(k), \quad z' = r \cdot k$$

$$\vec{\Omega}_3 = \frac{[x', y', z']^T}{\|[x', y', z']\|_2 + \epsilon} = \frac{[x', y', z']^T}{\sqrt{(x')^2 + (y')^2 + (z')^2} + \epsilon} = [\Omega_1, \Omega_2, \Omega_3]^T$$

where $\vec{\Omega}_3 \in \mathbb{R}^3$ is the normalized 3D direction vector and $\epsilon = 10^{-8}$.

### 9.3 Project-Specific 3D → D-Dimension Cyclic Extension

Because the STOA spiral produces a 3D direction vector $\vec{\Omega}_3 \in \mathbb{R}^3$ while the AgriAI hyperparameter search space has $D = 4$ dimensions, a deterministic cyclic extension is applied:

$$\vec{\Omega}_D[i] = \vec{\Omega}_3[(i-1) \bmod 3 + 1], \quad i \in \{1, \dots, D\}$$

For $D = 4$, this yields explicitly:

$$\vec{\Omega}_4 = [\Omega_1, \Omega_2, \Omega_3, \Omega_1]^T \in \mathbb{R}^4$$

> **This 3D → 4D cyclic extension is a project-specific engineering modification of the stabilized variant and is not attributed to the original STOA paper.**

This ensures that the element-wise perturbation $\vec{D}_s(t) \odot \vec{\Omega}_4$ is fully defined and dimensionally valid in $\mathbb{R}^4$.

### 9.4 Consistent Best-Guided Displacement (Option A)

Search agents are guided toward the best-so-far solution $\vec{P}_{best}$ without unintended random-agent drift:

$$\vec{C}_{sa}(t) = S_A(t) \cdot \vec{P}(t)$$

$$\vec{M}_s(t) = 0.5 \cdot \vec{r} \odot (\vec{P}_{best}(t) - \vec{P}(t))$$

$$\vec{D}_s(t) = \vec{C}_{sa}(t) + \vec{M}_s(t) = S_A(t) \cdot \vec{P}(t) + 0.5 \cdot \vec{r} \odot (\vec{P}_{best}(t) - \vec{P}(t))$$

with linearly decaying collision parameter $S_A(t) = 2 \cdot (1 - t / T_{max})$ and $\vec{r} \sim \text{Uniform}(0, 1)^4$.

### 9.5 Additive Position Update

$$\vec{P}(t+1) = \vec{P}_{best}(t) + \vec{D}_s(t) \odot \vec{\Omega}_4$$

### 9.6 Hyperparameter Domain Mapping

| Hyperparameter | Search Variable | Domain | Mapping |
|---------------|----------------|--------|---------|
| Learning rate | $\theta_1 \in [-4.0, -2.0]$ | Log-space | $\text{lr} = 10^{\theta_1}$ |
| Weight decay | $\theta_2 \in [-6.0, -2.0]$ | Log-space | $\text{wd} = 10^{\theta_2}$ |
| Dropout | $\theta_3 \in [0.0, 0.5]$ | Linear | $p_{drop} = \theta_3$ |
| Batch size | $\theta_4 \in [0, 2]$ | Continuous index | $\text{batch\_size} \in \{16, 32, 64\}$ via nearest neighbor |

### 9.7 Boundary Handling

$$\vec{P}_i = \text{clip}(\vec{P}_i, lb_i, ub_i)$$

### 9.8 Fitness Function

$$\text{fitness}(\vec{h}) = 1 - \text{macro\_F1}(\vec{h}, D_{val})$$

- Minimizing $(1 - \text{macro\_F1})$ is equivalent to maximizing macro F1
- Macro F1 treats all classes equally; preferred over accuracy for imbalanced datasets
- **The test set is NEVER used during fitness evaluation**

---

## 10. LS-HGNet Architecture

### 10.1 Important Distinction

> The original Stacked Hourglass Network (Newell et al., 2016) was designed for **human pose estimation** (structured output). The source smart-farming paper applied a variant to **UAV-based maize tassel detection**. Our LS-HGNet adaptation applies the concept to **tomato leaf disease classification**. All implementation specifics below are our **engineering decisions** unless explicitly marked as source-derived.

### 10.2 Source-Derived Concepts

| Concept | Source |
|---------|--------|
| Encoder-bottleneck-decoder hourglass structure | Newell et al., 2016 |
| Skip connections at each resolution level | Newell et al., 2016 |
| Stacking multiple hourglass modules | Newell et al., 2016 |
| Residual block design | He et al., 2016 (ResNet) |
| Depthwise separable convolution | Howard et al., 2017 (MobileNets) |
| Dilated convolution concept | Yu & Koltun, 2016 |

### 10.3 Our Engineering Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Task | Classification (not detection/pose) | Task adaptation |
| Dilation rates | [1, 2, 4] | Engineering choice only |
| Multi-dilated fusion | Concatenation + 1×1 conv + residual | Engineering choice |
| Inter-stack skip connections | Present | Engineering decision |
| Classification head | GAP + Dropout + Linear | Standard practice |
| Input resolution (dev) | 128×128 | Computational feasibility |
| Input resolution (final) | 224×224 | Standard scale |
| Number of stacks (dev) | 2 | Computational feasibility |
| Number of stacks (final) | 3 | Capacity experiment |
| Base channels (dev) | 32 | Lightweight objective |
| Base channels (final) | 64 | Capacity experiment |

### 10.4 Overall Architecture

```
Input Image (H × W × 3)
        |
    Stem Block
   Conv(3×3, stride=2) → BN → ReLU → MaxPool(2×2)
        |
   Hourglass Stack 1
   (Encoder → Bottleneck → Decoder with skip connections)
        |
  Inter-stack Skip Connection  [Engineering Decision]
        |
   Hourglass Stack 2
   (Encoder → Bottleneck → Decoder with skip connections)
        |
  [Optional Stack 3 — Final Config Only]
        |
   Global Average Pooling
        |
   Dropout (p = STOA-searched)
        |
   Linear Layer → Class Logits (10 classes)
```

### 10.5 Hourglass Module

**Encoder:** L levels (L=3 dev, L=4 final), each: LightResidualBlock → MaxPool(2×2). Channels: 32→64→128 (dev); 64→128→256 (final).

**Bottleneck:** MultiDilatedBlock at lowest resolution.

**Decoder:** L levels matching encoder. Each: Bilinear upsampling (×2) → LightResidualBlock. Skip connections from encoder concatenated before each decoder block.

**Upsampling:** Bilinear + 1×1 conv (chosen over transposed conv for stability and parameter efficiency).

### 10.6 DepthwiseSeparableConv

| Property | Specification |
|----------|--------------|
| Depthwise conv | k × k, groups=in_channels, same-size padding with dilation |
| Pointwise conv | 1×1, groups=1 |
| Normalization | BatchNorm2d after each conv |
| Activation | ReLU after each BN |
| Default kernel | 3×3 |
| Stride | Configurable, default=1 |
| Dilation | Configurable, default=1 |

### 10.7 LightResidualBlock

| Property | Specification |
|----------|--------------|
| Primary path | DWSConv(in, mid) → BN → ReLU → DWSConv(mid, out) → BN |
| Skip path | Identity if in==out and stride==1; else Conv(1×1, stride) → BN |
| Residual addition | Primary + Skip |
| Post-addition activation | ReLU |
| Projection condition | Applied when in_ch != out_ch OR stride != 1 |

### 10.8 MultiDilatedBlock

> **Dilation rates [1, 2, 4] are an engineering choice. They are NOT from the source smart-farming paper.**

```
Input (B, C, H, W)
      |
  [d=1 branch] [d=2 branch] [d=4 branch]  (Parallel DWSConv)
      |
  Concatenation → (B, 3C, H, W)
      |
  1×1 Conv → (B, C, H, W)
      |
  Residual Addition with Input
      |
  Output (B, C, H, W)
```

---

## 11. Classification Adaptation

### 11.1 Task

10-class tomato leaf disease classification on the PlantVillage dataset. This is a standard multi-class classification task where each input image is assigned to exactly one of 10 classes.

### 11.2 Input Pipeline

```
Raw Image → Resize → Augmentation (train only) → Normalize → Tensor
```

### 11.3 Classification Head

```
Final Hourglass Output (B, C, H, W)
        |
  Global Average Pooling → (B, C)
        |
  Dropout(p)
        |
  Linear(C, 10) → (B, 10) raw logits
```

**No softmax inside the model.** PyTorch CrossEntropyLoss applies log_softmax internally; applying softmax beforehand corrupts the loss. Temperature scaling operates on raw logits. Softmax is applied only at inference time.

### 11.4 Class Structure

The tomato subset of PlantVillage is commonly reported to contain 10 classes:

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

> **NOTE:** The number of classes (10) is consistently reported across multiple published papers and must be verified against the actual downloaded dataset during M1.

---

## 12. Dataset Plan

### 12.1 Known from M0

| Property | Value |
|----------|-------|
| Name | PlantVillage |
| Task subset | Tomato leaf diseases |
| Source | Hughes & Salathé, 2015 |
| Access | TensorFlow Datasets / Kaggle / PlantVillage GitHub |
| Target classes | 10 (1 healthy + 9 diseased) |
| Approximate total | ~18,000+ images (from literature) |

### 12.2 To Be Verified During M1

| Item | Status |
|------|--------|
| Exact per-class image counts | TO BE VERIFIED from downloaded files |
| Exact total image count | TO BE VERIFIED |
| Class distribution / imbalance ratio | TO BE VERIFIED |
| Image file formats | TO BE VERIFIED |
| Image dimensions (original) | TO BE VERIFIED |
| License terms (CC0 or other) | TO BE VERIFIED |

### 12.3 Known Limitations

1. PlantVillage images were collected under controlled laboratory conditions with uniform backgrounds.
2. Models trained on this dataset may not generalize to field images with complex, natural backgrounds.
3. Class imbalance may exist across the 10 classes.
4. The dataset does not include images of co-occurring diseases or infection at different growth stages.
5. Lighting conditions do not reflect the full range of real-world agricultural environments.
6. The dataset contains no geographic diversity metadata.

---

## 13. Data Preprocessing

### 13.1 Split Strategy

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

### 13.2 Leakage Prevention Rules

1. Splits are created **once** before any model training.
2. Augmentation is applied **only** to training data, never to validation or test data.
3. The test set is **never** seen during hyperparameter optimization.
4. STOA fitness is computed on **validation** data only.
5. Temperature scaling calibration parameters are fitted on **validation** data only.
6. Final reported metrics are computed on the **test** set **once** per experiment, after all hyperparameter decisions are finalized.

### 13.3 Augmentation Policy (Training Only)

| Augmentation | Parameters |
|-------------|-----------|
| Random horizontal flip | p = 0.5 |
| Random rotation | ±30 degrees, p = 0.5 |
| Color jitter | brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1 |
| Random resized crop | scale=(0.7, 1.0) |

> **Vertical Flip Policy:** Random vertical flip is deliberately excluded from the default training pipeline because vertical inversion produces biologically unnatural upside-down leaf orientations for field-captured foliar images. It is retained strictly as an optional experimental ablation (ABL-05).

### 13.4 Deterministic Preprocessing (All Splits)

1. Resize to target resolution (128×128 for development; 224×224 for final)
2. Convert to RGB
3. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

---

## 14. Experimental Design

### 14.1 Experiment Matrix

| ID | Model | Optimizer / Hyperparameters | Purpose |
|----|-------|----------------------------|---------|
| EXP_001 | Baseline CNN | Default hyperparameters | Experimental architecture control (~500K params) |
| EXP_002 | LS-HGNet | Default hyperparameters | Architectural baseline without hyperparameter tuning |
| EXP_003_original | LS-HGNet | Best from Original STOA (`original_strict`) | Evaluate source-derived STOA procedure as faithfully as possible |
| EXP_003_stabilized | LS-HGNet | Best from Stabilized STOA (`stabilized`) | Evaluate project-modified stabilized/additive STOA variant (Default) |
| EXP_004 | LS-HGNet | Best from Random Search (`random_search`) | Equal-budget non-metaheuristic search control (25 dev / 100 final) |

Models are NOT ranked before experiments. Results are measured, not predicted.

### 14.2 Baseline CNN

```
Input (H × W × 3)
      |
Block 1: Conv(3×3, 32) → BN → ReLU → MaxPool(2×2)
      |
Block 2: Conv(3×3, 64) → BN → ReLU → MaxPool(2×2)
      |
Block 3: Conv(3×3, 128) → BN → ReLU → MaxPool(2×2)
      |
Global Average Pooling
      |
Linear(128, 10) → Logits
```

Target: ~500K parameters. Purpose: Without a baseline, the added architectural complexity of LS-HGNet cannot be justified experimentally. Baseline uses default hyperparameters; no hyperparameter optimization applied.

### 14.3 Three Optimizer Conditions

| Condition Name | Code Identifier | Description |
|----------------|----------------|-------------|
| **Original/Strict STOA** | `original_strict` | Evaluates source-derived STOA procedure as faithfully as possible to empirically document its behavior and known numerical properties |
| **Stabilized STOA** | `stabilized` | Evaluates the project-modified stabilized/additive STOA variant **(Default)** |
| **Random Search Control** | `random_search` | Provides a non-metaheuristic search baseline evaluated under identical budget (25 dev / 100 final evaluations) |

### 14.4 Random Search Budget

| Property | Development | Final |
|----------|-------------|-------|
| STOA budget | 5 agents × 5 iters = 25 evals | 10 agents × 10 iters = 100 evals |
| Random Search budget | 25 evaluations | 100 evaluations |
| Search space | Same as STOA | Same as STOA |
| Fitness function | Same as STOA | Same as STOA |

**Equal-budget is the standard scientific protocol for optimizer comparison.** If STOA uses 100 evaluations and Random Search uses 25, any difference is attributable to budget, not algorithm design.

---

## 15. Optimization Search Space

| Hyperparameter | Range | Scale | Search Variable |
|---------------|-------|-------|----------------|
| Learning rate | [1e-4, 1e-2] | Log | $\theta_1 \in [-4.0, -2.0]$ |
| Weight decay | [1e-6, 1e-2] | Log | $\theta_2 \in [-6.0, -2.0]$ |
| Dropout probability | [0.0, 0.5] | Linear | $\theta_3 \in [0.0, 0.5]$ |
| Batch size | {16, 32, 64} | Discrete | $\theta_4 \in [0, 2]$ mapped via nearest neighbor |

All 4 hyperparameters are optimized simultaneously.

### Evaluation Budgets

| Phase | STOA | Random Search |
|-------|------|---------------|
| Development | 5 agents × 5 iterations = 25 evaluations | 25 evaluations |
| Final | 10 agents × 10 iterations = 100 evaluations | 100 evaluations |

---

## 16. Evaluation Metrics

### 16.1 Classification Metrics

| Metric | Scope |
|--------|-------|
| Accuracy | Overall |
| Precision | Macro, Weighted, Per-class |
| Recall | Macro, Weighted, Per-class |
| F1 Score | Macro, Weighted, Per-class |
| Confusion Matrix | 10×10 |
| ROC-AUC | Per-class (one-vs-rest) |
| PR-AUC | Per-class |

### 16.2 Computational Metrics

| Metric | Unit |
|--------|------|
| Parameter count | Integer |
| Trainable parameter count | Integer |
| Model size on disk | MB |
| CPU inference latency | ms/image (batch=1) |
| Peak GPU memory | MB (if applicable) |

### 16.3 Optimization Metrics

| Metric | Description |
|--------|-------------|
| Best fitness | Best $(1 - \text{macro\_F1})$ achieved |
| Number of evaluations | Integer |
| Optimization wall-clock time | Seconds |
| Convergence history | Fitness vs. evaluation number |

> **All metrics must be computed from actual model runs. No metric may be manually typed or fabricated.**

---

## 17. Explainable AI — Grad-CAM

### 17.1 Purpose

Grad-CAM provides post-hoc visual explanations of CNN predictions by highlighting image regions that contributed most to the model's classification decision. This supports transparency and interpretability in agricultural decision-making contexts.

### 17.2 Procedure

1. Forward pass: record activations $A^k$ at last decoder block of final hourglass stack.
2. Backward pass: compute gradient $\partial y^c / \partial A^k$.
3. Neuron importance: $\alpha^c_k = \frac{1}{Z} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}$.
4. Weighted combination: $L^c = \text{ReLU}(\sum_k \alpha^c_k \cdot A^k)$.
5. Upsample to input resolution.
6. Apply colormap and overlay on original image.

### 17.3 Target Layer

Output of the final decoder block of the last hourglass stack, before Global Average Pooling.

### 17.4 Limitations

> Grad-CAM shows image regions contributing to the model's prediction. It does NOT prove that the model identified the true biological cause of the disease. Highlighted regions may reflect superficial image features (e.g., background texture, leaf shape) rather than genuine disease lesion patterns. Outputs must be interpreted with appropriate caution.

---

## 18. Confidence Calibration

### 18.1 Temperature Scaling

Post-hoc method. A learned scalar $T > 0$ divides the logits:

$$\hat{p}^k = \frac{\exp(z^k / T)}{\sum_j \exp(z^j / T)}$$

### 18.2 Fitting Procedure

1. Train model to convergence.
2. Minimize NLL on validation data: $T^* = \arg\min_T \text{NLL}(D_{val}; T)$.
3. Apply $T^*$ at inference time.

**The test set is NEVER used to fit $T$.**

### 18.3 Metrics

| Metric | When |
|--------|------|
| ECE (Expected Calibration Error) | Before and after calibration |
| Reliability diagram | Before and after calibration |
| Mean raw confidence | vs. actual accuracy |
| Mean calibrated confidence | vs. actual accuracy |

### 18.4 Limitations

> Calibration does not guarantee correctness; it adjusts confidence, not accuracy. Model confidence must NOT be equated with disease severity or treatment urgency.

---

## 19. Robustness Testing

### 19.1 Perturbation Types

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

### 19.2 Purpose

Robustness testing evaluates how model accuracy degrades under synthetic image perturbations that simulate real-world conditions (varying lighting, camera quality, compression artifacts). The output is accuracy-vs-severity degradation curves.

### 19.3 Robustness Claims Rule

> The model MUST NOT be described as "robust" unless measured results support that claim. This section presents accuracy-vs-severity degradation curves, not qualitative judgments.

---

## 20. Application Architecture

### 20.1 System Architecture

```
React + Vite Dashboard
         ↓ HTTP
    Flask REST API
         ↓
    AgriAI Model (PyTorch)
         ↓
Prediction + Grad-CAM Overlay + Calibrated Confidence
```

### 20.2 Flask REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Input: leaf image. Output: JSON with class, confidence, calibrated confidence, Grad-CAM overlay (base64) |
| `/health` | GET | Health check |

### 20.3 React Dashboard

- Image upload interface
- Prediction display with confidence bar
- Grad-CAM heatmap visualization
- Disease information panel

---

## 21. Optional Extensions

The following components are **OPTIONAL** and are NOT part of core evaluated deliverables:

| Extension | Status |
|-----------|--------|
| ONNX export for edge deployment | Optional (OBJ-11) |
| ESP32 / microcontroller integration | Optional |
| Soil/temperature/humidity sensors | Optional |
| MQTT IoT protocol | Optional |
| Multimodal sensor fusion | Optional |
| Drone / field imagery | Optional |
| PWA / offline functionality | Optional |

These must NOT be treated as M1 requirements.

---

## 22. Originality and Contribution

### 22.1 Originality Statement

This project is an independent adaptation and extension of concepts reported in prior LS-HGNet and Sooty Tern Optimization research. The original research context (UAV-based maize tassel detection and counting), dataset, task, implementation, experiments, and reported results are not reproduced as original work.

The present project independently implements and evaluates a tomato leaf disease classification system using related architectural and optimization concepts, together with additional explainability (Grad-CAM), confidence calibration (temperature scaling), robustness testing, and deployment components (Flask API, React dashboard).

**All source code, experimental results, figures, and numerical values are independently produced. No text, code, or figures from the referenced source papers are copied, paraphrased without attribution, or claimed as our own findings.**

### 22.2 Project-Specific Contributions

| Contribution | Type |
|-------------|------|
| Tomato leaf disease classification adaptation | Task adaptation |
| PlantVillage dataset application | Dataset choice |
| Classification head (GAP + Dropout + Linear) | Engineering decision |
| Selected channels, stack count, dilation rates | Engineering decisions |
| STOA hyperparameter search space definition | Engineering decision |
| Project-Modified Stabilized/Additive STOA variant | Project modification |
| 3D → 4D deterministic cyclic mapping | Project modification |
| Equal-budget Random Search control | Experimental design |
| Grad-CAM explainability integration | Project extension |
| Temperature scaling calibration | Project extension |
| Systematic robustness testing (7 perturbation types) | Project extension |
| Flask REST API + React dashboard | Deployment component |

---

## 23. Limitations

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
12. **Original STOA equation verification status: Literature-Verified, Publisher-Unverified.** Direct full-text publisher PDF access to Dhiman & Kaur (2019) was not achieved during M0. This limitation is acknowledged; the project proceeds on the basis of strong multi-source literature consensus. If discrepancies are discovered upon M1 implementation (e.g., via institutional library access), the equations must be updated before publication.

---

## 24. Research Workflow

```
M0 — Research Specification (LOCKED)
   ↓
M1 — Dataset & Environment Setup
   ↓
   Dataset Acquisition → Audit → Stratified Split
   ↓
M2 — Model Implementation
   ↓
   Baseline CNN → LS-HGNet → STOA / Random Search
   ↓
M3 — Training & Evaluation
   ↓
   EXP_001 through EXP_004 → Metrics Collection
   ↓
M4 — Analysis & Explainability
   ↓
   Grad-CAM → Calibration → Robustness Testing
   ↓
M5 — Deployment
   ↓
   Flask API → React Dashboard
   ↓
M6 — Optional Extensions
   ↓
   IoT / Edge / Drone (if pursued)
```

---

## 25. Team Responsibilities

### Member 1 — Machine Learning

- LS-HGNet architecture implementation
- Baseline CNN implementation
- Training pipeline
- Model evaluation
- Grad-CAM implementation

### Member 2 — Research & Optimization

- STOA implementation (original and stabilized variants)
- Random Search implementation
- Hyperparameter search experiments
- Calibration (temperature scaling)
- Robustness testing
- Research documentation

### Member 3 — Application

- Flask REST API development
- React dashboard development
- API integration with trained model
- User interface design
- Optional edge/IoT extensions

### Shared Responsibilities

- Data pipeline and preprocessing
- Experiment coordination
- Results analysis and reporting
- Documentation maintenance

---

## 26. M0 Decision Log

### Resolved Research Specifications

| ID | Item | Resolution |
|----|------|-----------|
| RES-01 | STOA Primary Authorship | Gaurav Dhiman and Amandeep Kaur (2019), *EAAI*, 82, 148–174 |
| RES-02 | Smart Farming Source Attribution | Krishnan et al. (2024), *IJERR*, 37(spl), 96–108 |
| RES-03 | Originality and Adaptation Boundary | Source: UAV maize tassel detection; Ours: tomato disease classification |
| RES-04 | Original vs. Stabilized STOA | Formally separated into distinct specifications |
| RES-05 | Stabilized STOA Variant Design | Normalized Ω, damped spiral b=0.1, additive update, Option A ($\vec{P}_{best}$ throughout) |
| RES-06 | Three Optimizer Controls | `original_strict`, `stabilized`, `random_search` with equal-budget protocol |
| RES-07 | Data Split & Augmentation | Stratified 70/15/15; vertical flip excluded from default pipeline |
| RES-08 | Search Space & Fitness | 4 hyperparameters; fitness = $1 - \text{macro\_F1}$ on validation set |
| RES-09 | Original STOA Equation Verification | **Literature-Verified, Publisher-Unverified** — Formally accepted |

### Ablation Studies (Planned for M2/M3)

| ID | Ablation |
|----|---------|
| ABL-01 | MultiDilatedBlock vs. single-dilation bottleneck |
| ABL-02 | With vs. without inter-stack skip connections |
| ABL-03 | Stack count: 1 vs. 2 vs. 3 |
| ABL-04 | With vs. without morphological preprocessing |
| ABL-05 | 128×128 vs. 224×224 input resolution |
| ABL-06 | Depthwise-separable vs. standard convolution |

---

## 27. Deferred Questions

### M0 LOCKED Decisions

All items in Sections 1–26 of this report are **locked** and frozen for implementation. They define the research foundation, architecture boundaries, optimization strategy, experimental design, evaluation methodology, and originality boundaries.

### Deferred to M1/M2 (NOT M0 Blockers)

| ID | Deferred Question | Resolution Milestone |
|----|-------------------|---------------------|
| DEF-01 | PlantVillage exact per-class image counts and verified class distribution | M1: Count files on disk after download |
| DEF-02 | PlantVillage dataset licensing terms (CC0 vs. other) | M1: Inspect dataset README/LICENSE |
| DEF-03 | Hardware memory feasibility: 224×224 at batch size 32 on 16 GB RAM | M1: Memory profiling; reduce to batch 16 if needed |
| DEF-04 | Class-weighted Cross-Entropy loss benefit on macro F1 | M2: Ablation after measuring class imbalance |
| DEF-05 | Final dropout rate selection | M1/M2: Explored within STOA search space [0.0, 0.5] |
| DEF-06 | Optional edge deployment ONNX runtime latency | OBJ-11 (optional): Benchmark during deployment phase |

These items are implementation-phase verification questions, not research specification blockers.

---

## 28. Reproducibility Requirements

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

## 30. Final M0 Declaration

> **M0 RESEARCH SPECIFICATION — LOCKED**
>
> The research foundation, architecture boundaries, optimization strategy, experimental design, evaluation methodology, and originality boundaries are frozen for implementation.
>
> The original STOA formulation is classified as **Literature-Verified, Publisher-Unverified**. This limitation is explicitly documented and does not block implementation.
>
> The next development stage is **M1 — Dataset & Environment Setup**.

---

*End of M0 Research Report — AgriAI — Locked 2026-10-01 — Archived 2026-10-02*
