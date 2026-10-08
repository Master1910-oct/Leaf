# M1 — Dataset & Environment Setup Documentation

**Project:** AgriAI — STO-Optimized LS-HGNet for Explainable Crop Disease Classification in Smart Farming  
**Milestone:** M1 — Dataset & Environment Setup  
**Date:** 2026-10-02  
**Status:** ✅ **M1 COMPLETE — READY FOR MODEL IMPLEMENTATION**

---

## 1. Environment Specification & Dependency Versions

### 1.1 Host Hardware & Python Environment
- **Operating System:** Windows 10/11 x86_64
- **Processor:** Intel(R) Core(TM) i5-14400 (10 cores, 16 logical threads, AVX2 support)
- **Host RAM:** 16 GB physical RAM
- **Python Version:** 3.12.10 (64-bit)
- **Virtual Environment:** Dedicated local `.venv/` at `d:/Leaf/.venv`

### 1.2 Installed Package Versions
| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | `2.2.2+cpu` | Core deep learning and tensor operations (CPU target) |
| `torchvision` | `0.17.2+cpu` | Image transforms and computer vision utilities |
| `numpy` | `1.26.4` | Numerical arrays and mathematical routines |
| `pandas` | `3.0.6` | Dataframe handling and split manifest processing |
| `scipy` | `1.17.1` | Scientific routines and statistical distributions |
| `scikit-learn` | `1.9.1` | Stratified data partitioning and evaluation metrics |
| `Pillow` | `12.3.0` | High-integrity image reading and validation |
| `PyYAML` | `6.0.3` | Configuration loading (`configs/*.yaml`) |
| `matplotlib` | `3.11.2` | Calibration, reliability, and convergence visualization |
| `seaborn` | `0.13.2` | Confusion matrices and statistical distribution plots |
| `tqdm` | `4.70.1` | Execution and iteration progress monitoring |
| `psutil` | `7.2.2` | Memory benchmarking and process resource tracking |

---

## 2. Dataset Acquisition (PlantVillage Tomato Subset)

### 2.1 Identity and Source
- **Dataset:** PlantVillage Tomato Disease Subset
- **Authoritative Reference:** Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics through machine learning and crowdsourcing.* arXiv:1511.08060.
- **Repository Source:** `https://github.com/spMohanty/PlantVillage-Dataset`
- **Acquisition Script:** `scripts/download_dataset.py` (via sparse git checkout)
- **License:** CC0 1.0 Universal (Public Domain Dedication)
- **Local Storage Path:** `data/raw/plantvillage_tomato/`
- **Target Crop:** *Solanum lycopersicum* (Tomato) foliar pathology

---

## 3. Dataset Audit Results (M1.3)

Audit executed via `scripts/audit_dataset.py`. Reports saved to `reports/M1_dataset_audit.md` and `reports/dataset_audit.json`.

### 3.1 Key Audit Metrics
| Metric | Measured Value | Spec Requirement |
|--------|----------------|------------------|
| **Total Images** | **18,160** | ~18,000+ verified |
| **Number of Classes** | **10** | Exactly 10 classes |
| **Corrupt / Unreadable Files** | **0** (0.00%) | 0 corrupt |
| **Image Formats** | **100.0% JPEG** | Verified standard |
| **Color Modes** | **100.0% RGB (3 channels)** | 3-channel color |
| **Native Dimensions** | **100.0% 256×256 pixels** | Uniform input |
| **Class Imbalance Ratio** | **14.36:1** | Quantified |

### 3.2 Verified Class Distribution
| Index | Class Name | Healthy / Diseased | Verified Count | Proportion |
|:-----:|:-----------|:------------------:|:--------------:|:----------:|
| 0 | `Tomato___Bacterial_spot` | Diseased (Bacterial) | 2,127 | 11.71% |
| 1 | `Tomato___Early_blight` | Diseased (Fungal) | 1,000 | 5.51% |
| 2 | `Tomato___Late_blight` | Diseased (Oomycete) | 1,909 | 10.51% |
| 3 | `Tomato___Leaf_Mold` | Diseased (Fungal) | 952 | 5.24% |
| 4 | `Tomato___Septoria_leaf_spot` | Diseased (Fungal) | 1,771 | 9.75% |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | Infested (Pest) | 1,676 | 9.23% |
| 6 | `Tomato___Target_Spot` | Diseased (Fungal) | 1,404 | 7.73% |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Diseased (Viral) | 5,357 | 29.50% |
| 8 | `Tomato___Tomato_mosaic_virus` | Diseased (Viral) | 373 | 2.05% |
| 9 | `Tomato___healthy` | Healthy (Negative control) | 1,591 | 8.76% |
| **Total** | **All 10 Classes** | — | **18,160** | **100.00%** |

---

## 4. Reproducible Stratified Split (M1.4)

Implemented in `scripts/split_dataset.py`.

### 4.1 Partition Parameters
- **Ratios:** Train = 70.0%, Validation = 15.0%, Test = 15.0%
- **Partition Methodology:** Stratified random sampling by class index
- **Random Seed:** `42` (Locked in M0 research specification)
- **Manifest CSV Locations:**
  - `data/train.csv` (12,712 samples)
  - `data/val.csv` (2,724 samples)
  - `data/test.csv` (2,724 samples)
  - `data/all_splits.csv` (18,160 samples master)
  - Mirrored in `data/splits/` for spec directory compatibility

### 4.2 Per-Class Stratification Breakdown
| Class Name | Total | Train (70%) | Val (15%) | Test (15%) |
|:-----------|:-----:|:-----------:|:---------:|:----------:|
| `Tomato___Bacterial_spot` | 2,127 | 1,489 (70.0%) | 319 (15.0%) | 319 (15.0%) |
| `Tomato___Early_blight` | 1,000 | 700 (70.0%) | 150 (15.0%) | 150 (15.0%) |
| `Tomato___Late_blight` | 1,909 | 1,336 (70.0%) | 287 (15.0%) | 286 (15.0%) |
| `Tomato___Leaf_Mold` | 952 | 666 (70.0%) | 143 (15.0%) | 143 (15.0%) |
| `Tomato___Septoria_leaf_spot` | 1,771 | 1,240 (70.0%) | 266 (15.0%) | 265 (15.0%) |
| `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | 1,173 (70.0%) | 251 (15.0%) | 252 (15.0%) |
| `Tomato___Target_Spot` | 1,404 | 983 (70.0%) | 210 (15.0%) | 211 (15.0%) |
| `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | 3,750 (70.0%) | 803 (15.0%) | 804 (15.0%) |
| `Tomato___Tomato_mosaic_virus` | 373 | 261 (70.0%) | 56 (15.0%) | 56 (15.0%) |
| `Tomato___healthy` | 1,591 | 1,114 (70.0%) | 239 (15.0%) | 238 (15.0%) |
| **Total** | **18,160** | **12,712** | **2,724** | **2,724** |

---

## 5. Image Preprocessing & Augmentation Pipeline (M1.5)

Implemented in `src/data/transforms.py` and `src/data/dataset.py`.

### 5.1 Training Preprocessing (`get_train_transforms`)
- **Random Resized Crop:** `scale=(0.7, 1.0)`
- **Random Horizontal Flip:** `p = 0.5`
- **Random Rotation:** `±30°`, `p = 0.5`
- **Color Jitter:** `brightness=0.3`, `contrast=0.3`, `saturation=0.3`, `hue=0.1`
- **Normalization:** ImageNet mean ($\mu = [0.485, 0.456, 0.406]$) and std ($\sigma = [0.229, 0.224, 0.225]$)
- **Biological Validity Policy on Vertical Flip:** Random vertical flip is deliberately excluded by default to avoid generating biologically unnatural upside-down foliar orientations; preserved strictly as an optional ablation (`ABL-05`).

### 5.2 Evaluation Preprocessing (`get_eval_transforms`)
- **Deterministic Resize:** Target square dimension (`128×128` for dev, `224×224` for final)
- **Tensor Conversion:** Float32 normalized to $[0.0, 1.0]$
- **Normalization:** ImageNet mean and std
- **Augmentation:** Strictly none.

### 5.3 Pipeline Validation
Verified via DataLoader forward pass:
- Train loader: 398 batches of shape `(32, 3, 128, 128)`, range $[-2.12, 2.64]$
- Validation loader: 86 batches of shape `(32, 3, 128, 128)`, range $[-2.12, 2.01]$
- Test loader: 86 batches of shape `(32, 3, 128, 128)`, range $[-2.07, 2.12]$

---

## 6. Practical Memory Feasibility Benchmark (M1.6)

Benchmarked via `scripts/memory_test.py`. Results saved in `reports/M1_memory_feasibility.md` and `reports/memory_feasibility.json`.

| Batch Size | Resolution | Execution Device | RAM Peak | RAM Delta | Step Time | Throughput | Feasibility |
|:----------:|:----------:|:----------------:|:--------:|:---------:|:---------:|:----------:|:-----------:|
| 32 | 224×224 | CPU (host) | 437.14 MB | +276.95 MB | 1,071.06 ms | 29.9 img/s | **FEASIBLE (PASS)** |
| 32 | 128×128 | CPU (host) | 811.21 MB | +374.07 MB | 264.49 ms | 121.0 img/s | **FEASIBLE (PASS)** |
| 16 | 224×224 | CPU (host) | 1,005.99 MB | +194.78 MB | 404.76 ms | 39.5 img/s | **FEASIBLE (PASS)** |
| 16 | 128×128 | CPU (host) | 1,006.45 MB | +0.45 MB | 120.05 ms | 133.3 img/s | **FEASIBLE (PASS)** |

**Conclusion:** Host machine easily handles batch size 32 at both 128×128 (dev) and 224×224 (final). No modification to the locked M0 hyperparameter search bounds or batch size grid is necessary.

---

## 7. Machine-Readable Manifest (M1.7)

Generated at `data/manifest.json`. Consolidates:
- Dataset name, citation, and CC0 license
- Verified 10 classes and per-class counts
- Train/Val/Test counts and manifest paths
- Seed (`42`) and stratification policy
- Complete preprocessing and augmentation parameters
- Zero corrupt files verification stamp

---

## 8. Resolution of Deferred M0 Questions

| ID | Item | Resolution in M1 |
|----|------|------------------|
| **DEF-01** | PlantVillage exact per-class counts | Fully verified: 18,160 total images across 10 classes |
| **DEF-02** | PlantVillage dataset licensing terms | CC0 1.0 Universal (Public Domain Dedication) confirmed |

---

## 9. Next Stage Transition

Milestone M1 is completely fulfilled. All dataset infrastructure, splitting manifests, image pipelines, and environment dependencies are verified. The repository is ready for **M2 — Model Implementation** (Baseline CNN & LS-HGNet).
