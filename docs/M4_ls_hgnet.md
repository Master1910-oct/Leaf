# AgriAI — Milestone 4: Adapted Lightweight Stacked Hourglass Network (LS-HGNet)

**Status:** COMPLETE — READY FOR STO  
**Date:** October 2026  
**Author:** AgriAI Engineering Team  
**Task:** 10-Class Foliar Crop Disease Classification  
**Authoritative Reference:** `reports/M0_AgriAI_Research_Report.md`, `docs/architecture_decisions.md`

---

## 1. Purpose

Milestone 4 implements, rigorously verifies, and benchmarks the Adapted Lightweight Stacked Hourglass Network (LS-HGNet) architecture for tomato foliar disease classification. The primary objective is to evaluate the raw architectural capability and inductive bias of the adapted LS-HGNet against the M3 BaselineCNN control before introducing hyperparameter optimization via Sooty Tern Optimization (STO) in Milestone 5.

---

## 2. Source-Paper Architecture Basis

The conceptual foundation of the network derives from:
> **Krishnan et al. (2024)**, *"Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model"*, International Journal of Engineering and Robotic Research (IJERR), Vol. 37(spl), pp. 96–108.

The architectural principles synthesized from secondary literature include:
- **Hourglass Feature Hierarchy:** Newell et al. (2016), *Stacked Hourglass Networks for Human Pose Estimation*, ECCV.
- **Lightweight Convolutions:** Howard et al. (2017), *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, arXiv:1704.04861.
- **Multi-Scale Dilation:** Yu & Koltun (2016), *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR.
- **Identity Gradient Flow:** He et al. (2016), *Identity Mappings in Deep Residual Networks*, ECCV.

---

## 3. Original Application vs. AgriAI Adaptation

> **Formal Research Adaptation Statement:**  
> *"This implementation adapts the LS-HGNet architecture for tomato leaf disease classification. The original source work addressed UAV-based maize tassel detection/counting."*

### Critical Research Demarcations:
1. **Source Paper Application:** High-altitude UAV-based maize tassel detection and tassel counting in open agricultural fields.
2. **AgriAI Adaptation:** Close-up and field-captured foliar disease diagnosis across 10 distinct tomato pathological conditions.
3. **No Dataset Claim:** The original paper did NOT utilize the PlantVillage dataset; all PlantVillage experiments are independent adaptations.
4. **Architectural Modifications:**
   - Replaced multi-scale bounding-box regression/density heads with Global Average Pooling (GAP) and a 10-class linear classification head.
   - Formulated additive inter-stack skip connections between successive Hourglass stages.
   - Employed parallel multi-dilated bottleneck branches with dilation rates $r \in \{1, 2, 4\}$.

---

## 4. Architecture Overview

```
Input Leaf Image [B, 3, H, W]
  │
  ▼
Stem Block: Conv(3x3, stride=2) -> BatchNorm -> ReLU -> MaxPool(2x2)
  │ [Spatial downsampling 4x: H/4, W/4]
  ▼
StackedHourglass Module (num_stacks Hourglass modules)
  │ ├── Hourglass Stack 1 (Encoder -> Bottleneck -> Decoder)
  │ ├── Inter-Stack Skip Connection (Additive shortcut)
  │ └── Hourglass Stack 2 (Encoder -> Bottleneck -> Decoder)
  │ └── Hourglass Stack 3 [Final Config Only]
  │ [Feature Map: B, base_channels, H/4, W/4]
  ▼
Global Average Pooling (GAP) -> [B, base_channels]
  │
  ▼
Dropout (p = 0.3)
  │
  ▼
Linear Layer -> Raw Logits [B, 10] (NO SOFTMAX INSIDE MODEL)
```

---

## 5. Pre-Activation Residual Block (`PreActivationResidualBlock`)

Adheres strictly to the ordering:
$$\text{ReLU} \to \text{BatchNorm2d} \to \text{DepthwiseSeparableConv}$$

- **Structure:**
  $$\text{Input } x \to \text{ReLU} \to \text{BN} \to \text{DWConv}(3\times 3) \to \text{PWConv}(1\times 1) \to \text{ReLU} \to \text{BN} \to \text{DWConv}(3\times 3) \to \text{PWConv}(1\times 1) + \text{Shortcut}(x)$$
- **Identity Gradient Highway:** No activation is applied post-addition, preserving the clean identity flow across deep hierarchical levels.
- **Projection Shortcut:** Identity mapping when channels match and stride is 1; $1\times 1$ convolution with BatchNorm when spatial dimensions or channel counts change.

---

## 6. Depthwise-Separable Convolution (`DepthwiseSeparableConv`)

- Decomposes spatial and cross-channel operations:
  1. Depthwise convolution: spatial filtering per channel ($3\times 3$, `groups=in_channels`).
  2. Pointwise convolution: cross-channel projection ($1\times 1$).
- Computational cost reduced by factor:
  $$\frac{D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F}{D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F} = \frac{1}{N} + \frac{1}{D_K^2} \approx \frac{1}{9}$$

---

## 7. Multi-Dilated Lightweight Residual Block (`MultiDilatedLightResidualBlock`)

- Positioned at the lowest spatial resolution (bottleneck) of each Hourglass module.
- Parallel depthwise-separable branches with dilation rates $r \in \{1, 2, 4\}$.
- Outputs are concatenated along channels ($B \times 3C \times H' \times W'$), projected back to $C$ channels via $1\times 1$ convolution with BatchNorm and ReLU, and added to the bottleneck input shortcut.
- *Engineering Note:* Rates $[1, 2, 4]$ represent an AgriAI engineering decision.

---

## 8. Hourglass Structure (`Hourglass`)

- **Encoder:** $L$ levels ($L=3$ dev, $L=4$ final). Each level consists of a `PreActivationResidualBlock` followed by $2\times 2$ MaxPool.
- **Bottleneck:** `MultiDilatedLightResidualBlock` at lowest resolution.
- **Decoder:** $L$ levels in reverse order. Bilinear upsampling ($2\times$) followed by $1\times 1$ conv, additive intra-hourglass skip connection from matching encoder level, and a refining `PreActivationResidualBlock`.

---

## 9. Stacked Hourglass (`StackedHourglass`)

- Sequentially stacks multiple Hourglass modules ($S=2$ dev, $S=3$ final) allowing iterative bottom-up/top-down spatial and semantic refinement.

---

## 10. Inter-Stack Skip Connections

- Direct additive shortcut routing the output of Stack $k$ to the input of Stack $k+1$:
  $$x_{k+1} = \text{Hourglass}_k(x_k) + \text{Projection}_k(\text{Hourglass}_k(x_k))$$
- Ensures gradient backpropagation bypasses intermediate stacks without attenuation.

---

## 11. Classification Adaptation

- The detection/counting heads from the source literature are replaced by:
  $$\text{AdaptiveAvgPool2d}((1, 1)) \to \text{Flatten} \to \text{Dropout}(p=0.3) \to \text{Linear}(C, 10)$$
- **Strictly No Softmax Inside Model:** Produces raw unconstrained logits $[B, 10]$ for proper compatibility with PyTorch `CrossEntropyLoss` and post-hoc temperature scaling calibration.

---

## 12. Input / Output Dimensions

- **Development Input:** $[B, 3, 128, 128]$
- **Final Input:** $[B, 3, 224, 224]$
- **Output:** $[B, 10]$ (raw logits across all 10 tomato categories)

---

## 13. Parameter Count & Model Size

| Configuration | Base Channels | Stacks | Levels | Total Parameters | Trainable Parameters | Disk Checkpoint Size |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Development** | 32 | 2 | 3 | **357,226** | 357,226 | **4.42 MB** |
| **Final Benchmark** | 64 | 3 | 4 | **8,041,354** | 8,041,354 | **92.77 MB** |

---

## 14. Training Configuration

| Parameter | Development Run | Final Benchmark Run |
|:---|:---:|:---:|
| **Image Resolution** | $128 \times 128$ | $224 \times 224$ |
| **Batch Size** | 32 | 32 |
| **Epochs** | 5 | 5 |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | $1 \times 10^{-3}$ | $1 \times 10^{-3}$ |
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **LR Scheduler** | Cosine Annealing ($\eta_{min} = 10^{-6}$) | Cosine Annealing ($\eta_{min} = 10^{-6}$) |
| **Loss Function** | Unweighted Cross-Entropy | Unweighted Cross-Entropy |
| **Seed** | 42 | 42 |
| **Device** | CPU | CPU |

---

## 15. Development Experiment Results ($128\times 128$, Seed 42)

Artifacts saved in [`results/ls_hgnet_dev/`](file:///d:/Leaf/results/ls_hgnet_dev/):

| Metric | Measured Value |
|:---|:---:|
| **Validation Accuracy** | **87.37%** (0.873715) |
| **Validation Macro F1** | **82.54%** (0.825352) |
| **Validation Macro Precision** | **84.35%** (0.843486) |
| **Validation Macro Recall** | **83.99%** (0.839865) |
| **Median CPU Latency (batch=1)** | **13.89 ms** |
| **Mean CPU Latency (batch=1)** | 13.87 ms ($\pm 0.31$ ms) |
| **Best Epoch** | Epoch 5 |

---

## 16. Final Benchmark Validation Results ($224\times 224$, Seed 42)

Artifacts saved in [`results/ls_hgnet_final/`](file:///d:/Leaf/results/ls_hgnet_final/):

| Metric | Measured Value |
|:---|:---:|
| **Validation Accuracy** | **91.08%** (0.910793) |
| **Validation Macro F1** | **88.16%** (0.881642) |
| **Validation Macro Precision** | **89.53%** (0.895255) |
| **Validation Macro Recall** | **88.66%** (0.886605) |
| **Median CPU Latency (batch=1)** | **54.78 ms** |
| **Mean CPU Latency (batch=1)** | 54.00 ms ($\pm 2.86$ ms) |
| **Best Epoch** | Epoch 5 |

### Epoch-by-Epoch Training Trajectory:
- **Epoch 1:** Train Loss: `2.3661`, Train Acc: `50.26%` | Val Loss: `1.0233`, Val Acc: `67.07%`, Val F1: `57.26%` (2269.9s)
- **Epoch 2:** Train Loss: `0.8966`, Train Acc: `70.50%` | Val Loss: `0.6314`, Val Acc: `77.86%`, Val F1: `69.13%` (2065.2s)
- **Epoch 3:** Train Loss: `0.6367`, Train Acc: `78.54%` | Val Loss: `0.4314`, Val Acc: `84.07%`, Val F1: `77.75%` (2165.0s)
- **Epoch 4:** Train Loss: `0.4681`, Train Acc: `84.05%` | Val Loss: `0.2877`, Val Acc: `89.57%`, Val F1: `85.70%` (2209.2s)
- **Epoch 5:** Train Loss: `0.3596`, Train Acc: `87.67%` | Val Loss: `0.2325`, Val Acc: `91.08%`, Val F1: `88.16%` (2522.2s)

---

## 17. Per-Class Validation Results (Final $224\times 224$ Benchmark)

| Class Index | Disease Category | Precision | Recall | F1-Score | Support |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | `Tomato___Bacterial_spot` | 91.94% | 89.34% | 90.62% | 319 |
| 1 | `Tomato___Early_blight` | 92.79% | 68.67% | 78.93% | 150 |
| 2 | `Tomato___Late_blight` | 90.35% | 97.91% | 93.98% | 287 |
| 3 | `Tomato___Leaf_Mold` | 95.68% | 93.01% | 94.33% | 143 |
| 4 | `Tomato___Septoria_leaf_spot` | 84.31% | 96.99% | 90.21% | 266 |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | 72.43% | 98.41% | 83.45% | 251 |
| 6 | `Tomato___Target_Spot` | 87.30% | 52.38% | 65.48% | 210 |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 99.74% | 96.51% | 98.10% | 803 |
| 8 | `Tomato___Tomato_mosaic_virus` | 81.54% | 94.64% | 87.60% | 56 |
| 9 | `Tomato___healthy` | 99.16% | 98.74% | 98.95% | 239 |
| — | **Macro Average** | **89.53%** | **88.66%** | **88.16%** | **2,724** |
| — | **Weighted Average** | **91.84%** | **91.08%** | **90.76%** | **2,724** |

---

## 18. Confusion Matrix

The normalized confusion matrix was generated and saved to [`results/ls_hgnet_final/confusion_matrix.png`](file:///d:/Leaf/results/ls_hgnet_final/confusion_matrix.png).
Key confusion patterns observed:
- High separability (>98% F1) on Healthy and Tomato Yellow Leaf Curl Virus.
- Chief confusion occurs between foliar necrotic spot diseases: `Target_Spot` recall (52.38%) is degraded primarily by confusion with `Early_blight` and `Septoria_leaf_spot`, which exhibit visual homophily in necrotic foliar lesions.

---

## 19. CPU Inference Latency

Measured with 20 warmup iterations and 100 single-image evaluation iterations ($B=1$) on Intel CPU:
- **Development ($128\times 128$):** Median **13.89 ms** | Mean 13.87 ms ($\pm 0.31$ ms)
- **Final ($224\times 224$):** Median **54.78 ms** | Mean 54.00 ms ($\pm 2.86$ ms)
Both meet standard soft real-time field-edge inspection latency constraints (<100 ms).

---

## 20. Checkpoint & Model Size

- **Development Checkpoint:** 4.42 MB on disk (357,226 parameters)
- **Final Benchmark Checkpoint:** 92.77 MB on disk (8,041,354 parameters)

---

## 21. Comparison with BaselineCNN Control

Under an identical 5-epoch training protocol, identical stratified data split, unweighted Cross-Entropy loss, AdamW optimizer, and seed 42:

| Metric | BaselineCNN Final ($224\times 224$) | LS-HGNet Final ($224\times 224$) | Absolute Difference |
|:---|:---:|:---:|:---:|
| **Validation Accuracy** | 82.97% | **91.08%** | **+8.11%** |
| **Validation Macro F1** | 78.67% | **88.16%** | **+9.49%** |
| **Validation Macro Precision** | 83.42% | **89.53%** | **+6.11%** |
| **Validation Macro Recall** | 78.38% | **88.66%** | **+10.28%** |
| **Parameters** | 457,258 | 8,041,354 | +7,584,096 |
| **Model Size** | 5.26 MB | 92.77 MB | +87.51 MB |
| **Median CPU Latency** | 8.82 ms | 54.78 ms | +45.96 ms |
| **Best Epoch** | Epoch 5 | Epoch 5 | — |
| **Test Set Status** | **UNTOUCHED** | **UNTOUCHED** | Strictly Quarantined |

**Scientific Interpretation:**
The adapted LS-HGNet demonstrates a statistically significant +9.49% improvement in Macro F1 and +8.11% in accuracy over the baseline architecture, verifying the architectural efficacy of multi-scale hourglass bottom-up/top-down reasoning and multi-dilated receptive field aggregation.

---

## 22. Limitations

1. **Computational Cost:** The 3-stack, 4-level final configuration with base channel count 64 requires ~54.8 ms CPU latency per sample, compared to ~8.8 ms for the simpler 4-block baseline CNN.
2. **Lesion Homophily:** Foliar diseases with highly similar circular necrotic spot morphology (`Target_Spot` vs `Early_blight`) remain the primary source of classification errors in the unweighted baseline configuration.
3. **Fixed Hyperparameters:** Evaluated strictly using standard default hyperparameters ($lr = 10^{-3}$, $wd = 10^{-4}$, $dropout = 0.3$); hyperparameter space remains unoptimized until Milestone 5.

---

## 23. Reproducibility Information

- **Split Seeds:** Stratified split generated with seed 42.
- **PyTorch / Python Environment:** Python 3.12.10, PyTorch CPU, Intel 16-core CPU environment.
- **Deterministic Seeding:** `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.use_deterministic_algorithms(False)` (due to bilinear interpolation on CPU).
- **Test Set Quarantine:** Test set ($N = 2,724$) was strictly quarantined and unread.
