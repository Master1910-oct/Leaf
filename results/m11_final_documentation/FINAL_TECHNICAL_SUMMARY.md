# Final Technical Summary: Smart Farming LS-HGNet

**Project Title**: Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model  
**Repository**: `D:/Leaf`  
**Target Crop**: Solanum lycopersicum (Tomato) Foliage Disease Classification  
**Final Status**: **COMPLETED, LOCKED & CRYPTOGRAPHICALLY VERIFIED**

---

### 1. Problem & Core Solution
- **Problem**: Accurate identification of foliar tomato crop diseases is critical to precision agriculture, yet existing models often struggle with multi-scale lesion variations or impose high computational burdens.
- **Solution**: Adapted a **Lightweight Stacked Hourglass Network (LS-HGNet)** featuring 3 recurrent hourglass stacks, 4 encoder levels, and multi-dilation bottleneck blocks $(1, 2, 4)$, optimizing receptive field expansion with 8,041,354 parameters.

### 2. Dataset & Experimental Governance
- **Dataset**: 18,160 curated images partitioned into 70% Train (12,712), 15% Validation (2,724), and 15% Locked Test (2,724).
- **Class Coverage**: 10 distinct conditions (9 foliar diseases and healthy foliage).
- **Strict Data Governance**: Test split (`data/test.csv`) was kept locked throughout research development and evaluated strictly in a single one-shot pass in Milestone M9. Overlap across splits is verified as exactly 0.

### 3. Key Performance Metrics (M9 Final Locked Test)
- **Overall Test Accuracy**: **95.56%** (2,603 correct classifications out of 2,724 test images)
- **Macro-F1 Score**: **94.06%**
- **Weighted-F1 Score**: **95.55%**
- **Macro Precision**: **94.00%**
- **Macro Recall**: **94.43%**
- **Out-of-Sample Generalization Gap**: **0.17 percentage points** (Validation Macro-F1: 94.23% $\to$ Test Macro-F1: 94.06%)
- **Gain Over Frozen Baseline**: **+5.90 percentage points** (M7 Baseline: 88.16% $\to$ Final Test: 94.06%)

### 4. Per-Class Highlights & Diagnosed Errors
- **Top Pathology**: *Tomato Yellow Leaf Curl Virus* (F1: 99.12%, support: 804) and *Healthy Leaf* (F1: 98.94%, 100.0% precision).
- **Primary Vulnerability**: *Target Spot* (F1: 86.78%, recall: 82.46%), with 29 instances misclassified as *Two-Spotted Spider Mites* due to visual similarity between early pinpoint lesions and chlorotic mite stippling.
- **Prediction Confidence**: Correct predictions average 96.97% confidence (median: 99.95%), whereas misclassifications exhibit marked uncertainty (mean: 70.65%, median: 69.10%).

### 5. Deployment & Execution Profile
- **Single-Image CPU Latency**: $132.18 \pm 16.71$ ms (batch size = 1, throughput = 7.57 images/second).
- **Model Checkpoint**: `results/m7_3_full_training/checkpoints/best_model.pt` (SHA-256: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`).
