# Final-Year Engineering Project Report Content

## Project Title: Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model

**Department**: Computer Science & Engineering / Artificial Intelligence  
**Degree**: Bachelor of Technology / Bachelor of Engineering  
**Academic Year**: 2025–2026  

---

### Chapter 1: Introduction
Tomato (*Solanum lycopersicum*) is a major horticultural crop worldwide. Crop productivity is heavily compromised by viral, fungal, and bacterial leaf pathogens. Traditional agricultural scouting relies on visual inspection by pathologists, which is labor-intensive, localized, and subjective. Automated computer vision frameworks deployed via mobile devices or agricultural drones offer real-time diagnostic capability. This project presents an adapted **Lightweight Stacked Hourglass Network (LS-HGNet)** incorporating multi-dilation residual bottlenecks to classify 10 distinct tomato leaf conditions with high accuracy, low parameter footprint, and strict data governance.

### Chapter 2: Existing System & Literature Review
Conventional automated diagnosis typically deploys standard transfer-learning backbones (e.g., ResNet-50, VGG-16, Inception-v3). While effective, these models feature large parameter counts (>25 million), high memory requirements (>100 MB), and uniform spatial receptive fields that struggle with multi-scale lesion textures. Furthermore, existing studies often suffer from unverified data isolation, lacking independent locked-test verification.

### Chapter 3: Proposed System & Architecture
The proposed system adapts the **Lightweight Stacked Hourglass Network (LS-HGNet)**:
1. **Multi-Scale Hourglass Modules**: Three recurring stacks perform repeated bottom-up, top-down spatial inference.
2. **Multi-Dilation Residual Bottlenecks**: Incorporates parallel $3 \times 3$ dilated convolutions ($r \in \{1, 2, 4\}$) to capture fine-grained speckling and broad blights without downsampling loss.
3. **Parameter Efficiency**: Entire architecture contains exactly **8,041,354 parameters**, fitting within 92.77 MB.
4. **Bio-Inspired Optimization Exploration**: Sooty Tern Optimization (STO) was explored to evaluate hyperparameter search spaces.
5. **Calibrated Confidence**: Post-hoc Temperature Scaling ($T=1.1558$) moderates overconfidence.

### Chapter 4: System Requirements & Technical Specifications
- **Software**: Python 3.12.10, PyTorch 2.2.2+cpu, Torchvision 0.17.2, Scikit-learn 1.4, Pandas, Matplotlib.
- **Hardware**: Standard x86_64 CPU (Intel Core Architecture), 16 GB RAM.
- **Operating System**: Windows 11 / Linux compatible.

### Chapter 5: Dataset & Methodology
- **Dataset**: 18,160 curated tomato leaf images from PlantVillage.
- **Partitioning**: 70% Train (12,712), 15% Validation (2,724), 15% Locked Test (2,724). Overlap is verified as exactly 0.
- **Class Coverage**: 10 classes including 9 foliar diseases and healthy leaves.
- **Preprocessing**: Resize to $224 \times 224$, ToTensor, and ImageNet normalization.

### Chapter 6: Experimental Results & Analysis
- **Training Convergence**: 10-epoch full-training convergence (Milestone M7.3) reached global peak performance at Epoch 9: Validation Accuracy **95.52%**, Macro-F1 **94.23%**.
- **Final Locked-Test Performance (M9)**: One-shot evaluation on 2,724 unseen images yielded:
  - **Overall Accuracy**: **95.56%**
  - **Macro-F1 Score**: **94.06%**
  - **Weighted-F1 Score**: **95.55%**
  - **Macro Precision**: **94.00%** | **Macro Recall**: **94.43%**
- **Generalization Gap**: A minimal drop of **0.17 percentage points** between validation Macro-F1 (94.23%) and test Macro-F1 (94.06%).
- **Error Analysis**: 121 misclassifications (4.44% error rate). Chief confusion: Target Spot mistaken for Spider Mites (29 errors) due to visual similarity between early pinpoint lesions and chlorotic stippling.
- **Deployment Profile**: Single-image CPU inference latency of **$132.18 \pm 16.71$ ms** (throughput: 7.57 img/s).

### Chapter 7: Advantages & Limitations
- **Advantages**: High multi-class accuracy; balanced Macro-F1; lightweight footprint; calibrated uncertainty estimates; confirmed zero-leakage test validation.
- **Limitations**: Sensitivity to optical blur (-47.8% F1) and sensor noise (-23.7% F1); lab-curated leaf backgrounds require further validation under wild in-field canopy clutter.

### Chapter 8: Conclusion & Future Scope
The adapted LS-HGNet demonstrates that lightweight multi-scale hourglass architectures provide accurate, balanced, and robust crop disease diagnosis. Future work will extend the framework to INT8 edge quantization, in-field smartphone mobile applications, and aerial drone video analytics.
