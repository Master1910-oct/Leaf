# Smart Farming with LS-HGNet Tomato Leaf Disease Classification

[![Model](https://img.shields.io/badge/Architecture-LS--HGNet-blue.svg)](src/models/ls_hgnet.py)
[![Parameters](https://img.shields.io/badge/Parameters-8.04M-green.svg)](results/m10_final_evidence/FINAL_MODEL_SPECIFICATION.md)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.56%25-brightgreen.svg)](results/m9_final_locked_test/M9_FINAL_REPORT.md)
[![Macro F1](https://img.shields.io/badge/Macro%20F1-94.06%25-brightgreen.svg)](results/m9_final_locked_test/M9_FINAL_REPORT.md)
[![Checkpoint](https://img.shields.io/badge/Checkpoint-SHA256%20Verified-blueviolet.svg)](results/m7_3_full_training/checkpoints/best_model.pt)

## 📌 Project Overview
This repository implements and rigorously evaluates the **Lightweight Stacked Hourglass Network (LS-HGNet)** for automated diagnosis of tomato leaf diseases across 10 distinct pathology and healthy classes. The project features bio-inspired metaheuristic search exploration (Sooty Tern Optimization), post-hoc Temperature Scaling calibration, systematic synthetic robustness testing, and a strictly locked out-of-sample test evaluation.

---

## 🏆 Final Locked-Test Results (Milestone M9)

The final model checkpoint (`best_model.pt`, Epoch 9) was evaluated on the untouched 2,724-image test set:

| Evaluation Metric | Value |
| :--- | :---: |
| **Overall Accuracy** | **95.56%** (2,603 / 2,724 correct) |
| **Macro-F1 Score** | **94.06%** |
| **Weighted-F1 Score** | **95.55%** |
| **Macro Precision** | **94.00%** |
| **Macro Recall** | **94.43%** |
| **Validation-to-Test Generalization Gap** | **0.17 percentage points** |
| **Net Gain vs M7 Baseline** | **+5.90 percentage points** |
| **Single-Image CPU Latency (batch=1)** | **132.2 ms** (7.57 img/s) |

---

## 🔬 Architecture Highlights: LS-HGNet
- **3 Recurrent Hourglass Stacks**: Multi-scale bottom-up, top-down feature aggregation.
- **Multi-Dilation Bottleneck Blocks**: Dilation rates $r \in \{1, 2, 4\}$ for receptive field coverage without feature downsampling loss.
- **Parameter Efficiency**: 8,041,354 total parameters (92.77 MB checkpoint).

---

## 📂 Key Artifacts & Documentation
- **Research Manuscript**: [`results/m11_final_documentation/RESEARCH_MANUSCRIPT.md`](results/m11_final_documentation/RESEARCH_MANUSCRIPT.md)
- **Final Audit Report**: [`results/m11_final_documentation/M10_FINAL_REPORT.md`](results/m10_final_evidence/M10_FINAL_REPORT.md)
- **Technical Q&A / Viva Prep**: [`results/m11_final_documentation/VIVA_QA.md`](results/m11_final_documentation/VIVA_QA.md)
- **Frozen Checkpoint**: `results/m7_3_full_training/checkpoints/best_model.pt`  
  `SHA-256: 4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`
