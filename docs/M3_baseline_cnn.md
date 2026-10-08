# AgriAI — M3 Baseline CNN Milestone Documentation

**Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis**  
**Milestone:** M3 — Baseline CNN Implementation  
**Status:** ✅ M3 COMPLETE (All Milestone Gates Passed)  
**Date:** 2026-10-02  

---

## 1. Purpose

The objective of Milestone 3 is to establish a rigorous, reproducible, lightweight **experimental control** using a conventional Convolutional Neural Network (BaselineCNN).

This baseline answers the core research question:
> *"How well can a lightweight conventional CNN perform on the PlantVillage tomato disease benchmark under identical dataset splits and training constraints?"*

The BaselineCNN serves as the benchmark against which later models will be evaluated:
1. Default LS-HGNet
2. STO-Optimized LS-HGNet
3. Random-Search-Optimized LS-HGNet

---

## 2. Model Architecture

The `BaselineCNN` is a 4-stage convolutional feed-forward network with batch normalization, non-linear activations, spatial pooling, global average pooling, and a fully connected classification head.

```text
Input Image: [B, 3, H, W]
  │
  ├── [Conv Block 1]
  │     ├── Conv2D: 3 -> 32 channels (3x3 kernel, stride 1, padding 1, bias=False)
  │     ├── BatchNorm2D(32)
  │     ├── ReLU(inplace=True)
  │     └── MaxPool2D(2x2 kernel, stride 2) -> Spatial reduction: H/2, W/2
  │
  ├── [Conv Block 2]
  │     ├── Conv2D: 32 -> 64 channels (3x3 kernel, stride 1, padding 1, bias=False)
  │     ├── BatchNorm2D(64)
  │     ├── ReLU(inplace=True)
  │     └── MaxPool2D(2x2 kernel, stride 2) -> Spatial reduction: H/4, W/4
  │
  ├── [Conv Block 3]
  │     ├── Conv2D: 64 -> 128 channels (3x3 kernel, stride 1, padding 1, bias=False)
  │     ├── BatchNorm2D(128)
  │     ├── ReLU(inplace=True)
  │     └── MaxPool2D(2x2 kernel, stride 2) -> Spatial reduction: H/8, W/8
  │
  ├── [Conv Block 4]
  │     ├── Conv2D: 128 -> 256 channels (3x3 kernel, stride 1, padding 1, bias=False)
  │     ├── BatchNorm2D(256)
  │     ├── ReLU(inplace=True)
  │     └── MaxPool2D(2x2 kernel, stride 2) -> Spatial reduction: H/16, W/16
  │
  ├── [Global Average Pooling]
  │     └── AdaptiveAvgPool2D((1, 1)) -> Output: [B, 256, 1, 1]
  │
  └── [Classifier Head]
        ├── Flatten() -> [B, 256]
        ├── Linear(256 -> 256, bias=True)
        ├── ReLU(inplace=True)
        ├── Dropout(p=0.2)
        └── Linear(256 -> 10, bias=True) -> Output Logits: [B, 10]
```

### Architectural Properties:
- **Input Channels:** 3 (RGB)
- **Output Classes:** 10 (PlantVillage tomato diseases + healthy)
- **Initialization:** Kaiming Normal (He) for convolutional and linear weights; constant 1.0 (weight) and 0.0 (bias) for BatchNorm.
- **Biases in Convolutions:** Omitted (`bias=False`) due to immediately following BatchNorm layers.

---

## 3. Parameter Count & Model Size

| Layer / Component | Specification | Trainable Parameters |
|:------------------|:--------------|:--------------------:|
| **Conv Block 1** | Conv2D(3, 32, 3x3) + BN(32) | 864 + 64 = **928** |
| **Conv Block 2** | Conv2D(32, 64, 3x3) + BN(64) | 18,432 + 128 = **18,560** |
| **Conv Block 3** | Conv2D(64, 128, 3x3) + BN(128) | 73,728 + 256 = **73,984** |
| **Conv Block 4** | Conv2D(128, 256, 3x3) + BN(256) | 294,912 + 512 = **295,424** |
| **Classifier Head** | Linear(256, 256) + Linear(256, 10) | 65,792 + 2,570 = **68,362** |
| **Total Parameters** | — | **457,258** (~457K parameters) |
| **Trainable Parameters** | — | **457,258** (100.0%) |
| **In-Memory FP32 Size** | Parameter & Buffer Tensors | **1.75 MB** |

The measured total of **457,258 parameters** satisfies the M0/M3 lightweight specification target of approximately 500K parameters.

---

## 4. Input Resolutions

The baseline architecture supports arbitrary spatial resolutions via `AdaptiveAvgPool2D`:
- **Development Resolution:** 128 × 128 pixels (for fast iteration, gradient verification, and pipeline validation).
- **Final Evaluation Resolution:** 224 × 224 pixels (locked final benchmark resolution).

---

## 5. Training Configuration

| Hyperparameter | Development Setting | Final Benchmark Setting |
|:---------------|:--------------------|:------------------------|
| **Device** | CPU (Intel Host standard) | CPU (Intel Host standard) |
| **Batch Size** | 32 | 32 |
| **Optimizer** | AdamW | AdamW |
| **Base Learning Rate** | 1.0e-3 | 1.0e-3 |
| **Weight Decay** | 1.0e-4 | 1.0e-4 |
| **LR Scheduler** | CosineAnnealingLR (T_max=epochs, eta_min=1e-6) | CosineAnnealingLR (T_max=epochs, eta_min=1e-6) |
| **Dropout Rate** | 0.2 | 0.2 |
| **Random Seed** | 42 | 42 |

---

## 6. Loss Function

In accordance with M0/M1 specifications:
- **Baseline v1 Criterion:** Standard unweighted `nn.CrossEntropyLoss()`.
- **Class Imbalance Policy:** The PlantVillage tomato dataset exhibits a 14.36:1 imbalance ratio. Standard cross-entropy is intentionally maintained to establish an authentic baseline. Any class-weighting or focal loss adjustments are strictly reserved for separate ablation studies.

---

## 7. Dataset Splits and Test Set Protection

The split was locked during M1 with seed = 42:

| Split | Number of Images | Percentage | Usage in M3 |
|:------|:----------------:|:----------:|:------------|
| **Train** | 12,712 | 70% | Model parameter optimization via SGD/AdamW |
| **Validation** | 2,724 | 15% | Per-epoch evaluation, checkpoint selection |
| **Test** | 2,724 | 15% | **STRICTLY UNTOUCHED & PROTECTED** |
| **Total** | 18,160 | 100% | — |

> **Security & Rigor Guarantee:**  
> The training scripts and DataLoaders used during M3 only access `train.csv` and `val.csv`. The test set manifest (`test.csv`) is never loaded into memory during baseline training or checkpoint selection.

---

## 8. Unit Test Verification (Test Suite Results)

All unit tests in `tests/test_baseline.py` passed cleanly:

```text
Ran 6 tests in 4.509s - OK

[Test 1] Forward Pass Dimension Verification:
  - Input: [4, 3, 128, 128] -> Logits: [4, 10] (PASS)
  - Input: [4, 3, 224, 224] -> Logits: [4, 10] (PASS)

[Test 2] Parameter Count Verification:
  - Measured Parameters: 457,258 (Target: ~500K) (PASS)

[Test 3] Gradient Flow Verification:
  - Backpropagation propagates non-zero gradients to all trainable parameters (PASS)

[Test 4] Tiny-Batch Overfit Sanity Check:
  - 8-sample overfit: Initial Loss 11.3962 -> Final Loss 0.00000, 100% Accuracy (PASS)

[Test 5] Checkpoint Save/Load Parity:
  - Saved model state loaded into fresh model instance; exact tensor match torch.allclose (PASS)

[Test 6] Seeded Reproducibility:
  - Two models seeded with identical seed produce identical weights and identical forward activations (PASS)
```

---

## 9. Development Run Results (128 × 128)

**Execution Settings:** Resolution: $128 \times 128$, Batch Size: 32, Epochs: 5, Device: CPU, Seed: 42.  
**Best Validation Epoch:** Epoch 5  
**Checkpoint Path:** `results/baseline_dev/best_model.pt` (5.26 MB)

### Macro Classification Metrics (Validation Set, N=2,724)
- **Validation Accuracy:** **85.13%** (0.851322)
- **Macro Precision:** **84.06%** (0.840552)
- **Macro Recall:** **80.44%** (0.804445)
- **Macro F1-Score:** **81.31%** (0.813099)

### Per-Class Validation Breakdown
| Class Index | Class Name | Support | Precision | Recall | F1-Score |
|:-----------:|:-----------|:-------:|:---------:|:------:|:--------:|
| 0 | `Tomato___Bacterial_spot` | 319 | 94.95% | 82.45% | 88.26% |
| 1 | `Tomato___Early_blight` | 150 | 76.60% | 48.00% | 59.02% |
| 2 | `Tomato___Late_blight` | 287 | 81.20% | 75.26% | 78.12% |
| 3 | `Tomato___Leaf_Mold` | 143 | 70.33% | 89.51% | 78.77% |
| 4 | `Tomato___Septoria_leaf_spot` | 266 | 80.52% | 80.83% | 80.68% |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | 251 | 72.56% | 94.82% | 82.21% |
| 6 | `Tomato___Target_Spot` | 210 | 85.71% | 60.00% | 70.59% |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 803 | 89.52% | 100.00% | 94.47% |
| 8 | `Tomato___Tomato_mosaic_virus` | 56 | 90.57% | 85.71% | 88.07% |
| 9 | `Tomato___healthy` | 239 | 98.59% | 87.87% | 92.92% |
| — | **Macro Average** | **2,724** | **84.06%** | **80.44%** | **81.31%** |

### Computational Performance (Measured on CPU)
- **Trainable Parameters:** 457,258
- **Checkpoint Size on Disk:** 5.26 MB
- **Single-Sample Inference Latency (Batch=1, Warmup=20, Timed=100):**
  - **Median Latency:** **3.40 ms**
  - **Mean Latency:** **3.40 ms** ($\pm 0.07\text{ ms}$)
  - **Min Latency:** 3.28 ms
  - **Max Latency:** 3.60 ms

### Epoch Progression
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro F1 | Epoch Time | Status |
|:-----:|:----------:|:---------:|:--------:|:-------:|:------------:|:----------:|:------:|
| 1 | 1.6318 | 52.38% | 1.0968 | 62.22% | 51.56% | 143.0s | Best |
| 2 | 1.0477 | 64.21% | 1.0198 | 63.47% | 53.21% | 140.4s | Best |
| 3 | 0.8616 | 70.55% | 0.9332 | 67.07% | 55.98% | 141.7s | Best |
| 4 | 0.6934 | 75.66% | 0.4895 | 81.83% | 76.95% | 142.2s | Best |
| 5 | 0.5790 | 80.24% | 0.4232 | 85.13% | 81.31% | 142.3s | **Best** |


---

## 10. Final Baseline Run Results (224 × 224)

**Execution Settings:** Resolution: $224 \times 224$, Batch Size: 32, Epochs: 5, Device: CPU, Seed: 42.  
**Best Validation Epoch:** Epoch 5  
**Checkpoint Path:** `results/baseline_final/best_model.pt` (5.26 MB)

### Macro Classification Metrics (Validation Set, N=2,724)
- **Validation Accuracy:** **82.97%** (0.829662)
- **Macro Precision:** **83.42%** (0.834196)
- **Macro Recall:** **78.38%** (0.783764)
- **Macro F1-Score:** **78.67%** (0.786657)

### Per-Class Validation Breakdown
| Class Index | Class Name | Support | Precision | Recall | F1-Score |
|:-----------:|:-----------|:-------:|:---------:|:------:|:--------:|
| 0 | `Tomato___Bacterial_spot` | 319 | 94.93% | 88.09% | 91.38% |
| 1 | `Tomato___Early_blight` | 150 | 87.50% | 51.33% | 64.71% |
| 2 | `Tomato___Late_blight` | 287 | 85.92% | 63.76% | 73.20% |
| 3 | `Tomato___Leaf_Mold` | 143 | 69.10% | 86.01% | 76.64% |
| 4 | `Tomato___Septoria_leaf_spot` | 266 | 82.96% | 69.55% | 75.66% |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | 251 | 58.29% | 95.22% | 72.31% |
| 6 | `Tomato___Target_Spot` | 210 | 91.67% | 41.90% | 57.52% |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 803 | 88.02% | 99.75% | 93.52% |
| 8 | `Tomato___Tomato_mosaic_virus` | 56 | 82.26% | 91.07% | 86.44% |
| 9 | `Tomato___healthy` | 239 | 93.55% | 97.07% | 95.28% |
| — | **Macro Average** | **2,724** | **83.42%** | **78.38%** | **78.67%** |

### Computational Performance (Measured on CPU)
- **Trainable Parameters:** 457,258
- **Checkpoint Size on Disk:** 5.26 MB
- **Single-Sample Inference Latency (Batch=1, Warmup=20, Timed=100):**
  - **Median Latency:** **8.82 ms**
  - **Mean Latency:** **8.81 ms** ($\pm 0.41\text{ ms}$)
  - **Min Latency:** 7.94 ms
  - **Max Latency:** 10.07 ms

### Epoch Progression
| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val Macro F1 | Epoch Time | Status |
|:-----:|:----------:|:---------:|:--------:|:-------:|:------------:|:----------:|:------:|
| 1 | 1.5325 | 54.04% | 0.9228 | 68.32% | 59.71% | 465.6s | Best |
| 2 | 1.0606 | 63.42% | 0.8448 | 69.13% | 61.46% | 466.6s | Best |
| 3 | 0.8948 | 69.34% | 0.7736 | 71.55% | 62.18% | 464.9s | Best |
| 4 | 0.7372 | 74.39% | 0.4816 | 82.38% | 77.92% | 485.5s | Best |
| 5 | 0.6179 | 78.84% | 0.4667 | 82.97% | 78.67% | 463.7s | **Best** |


---

## 11. Known Limitations of Baseline CNN

1. **Local Receptive Field:** Conventional 3x3 convolutions lack dynamic multi-scale contextual aggregation compared to stacked hourglass structures.
2. **Uniform Capacity:** Equal depth without intermediate multi-resolution supervision or cross-stage feature refinement.
3. **No Hyperparameter Optimization:** Uses heuristic hyperparameter defaults (lr=1e-3, batch=32) without STOA search.

---

## 12. Reproducibility Instructions

To reproduce the Baseline CNN experiments:

```powershell
# 1. Run unit verification test suite
.venv\Scripts\python -m unittest tests/test_baseline.py

# 2. Run development training (128x128)
.venv\Scripts\python scripts/train_baseline.py --config configs/development.yaml --epochs 5 --image_size 128 --batch_size 32 --output_dir results/baseline_dev

# 3. Run final baseline benchmark (224x224)
.venv\Scripts\python scripts/train_baseline.py --config configs/final.yaml --epochs 10 --image_size 224 --batch_size 32 --output_dir results/baseline_final
```
