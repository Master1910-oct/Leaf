# M11 — Methodology Consistency Audit

**Audit Date**: 2026-10-08 UTC  
**Auditor**: M11 Final Documentation Pipeline  
**Scope**: Systematic verification of repository claims against actual code, configs, logs, and artifacts.

---

## 1. Methodology Component Verification Matrix

| Claim / Component | Assessment Status | Concrete Repository Evidence | Audit Observations & Methodological Boundary |
| :--- | :---: | :--- | :--- |
| **LS-HGNet Implementation** | **SUPPORTED** | `src/models/ls_hgnet.py` | Full PyTorch implementation with 3 stacks, 4 encoder levels, and multi-dilation bottleneck blocks $(1, 2, 4)$. Parameter count verified as exactly 8,041,354. |
| **Sooty Tern Optimization (STO)** | **SUPPORTED (Screening Only)** | `src/optimization/`, `scripts/run_sto.py`, `results/m6_comparison/`, `results/m7_1_sto_refinement/` | Metaheuristic bio-inspired algorithm implemented and evaluated for hyperparameter exploration across learning rate, weight decay, dropout, and batch size. |
| **Influence of STO on Final Model** | **PARTIALLY SUPPORTED** | `results/m7_final_selection/`, `results/m7_3_full_training/` | **CRITICAL AUDIT FINDING**: STO was evaluated as a hyperparameter search algorithm on proxy data. Candidate A (learning rate $10^{-3}$, weight decay $10^{-4}$, dropout 0.3) was selected during M7. The final 10-epoch full convergence run (M7.3) used standard AdamW with Cosine Annealing. STO was NOT used to update backpropagation weights during full convergence. |
| **Dataset & Sample Count** | **SUPPORTED** | `data/train.csv`, `val.csv`, `test.csv` | Exact count verified: 18,160 total images (12,712 train, 2,724 val, 2,724 test). Zero sample overlap. |
| **Class Mapping** | **SUPPORTED** | Verified in CSVs and scripts | 10 distinct classes verified across all manifests with identical index-to-name mapping. |
| **Preprocessing Pipeline** | **SUPPORTED** | `scripts/run_m73_full_training.py`, `scripts/run_m9_final_locked_test.py` | ImageNet normalization at $224 \times 224$ pixels. Deterministic transforms during evaluation (no test-time augmentations). |
| **Final Checkpoint Integrity** | **SUPPORTED** | `results/m7_3_full_training/checkpoints/best_model.pt` | Verified SHA-256: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`. Size: 97,278,348 bytes. |
| **Locked Test Evaluation (M9)** | **SUPPORTED** | `results/m9_final_locked_test/` | Evaluated strictly once under `torch.no_grad()`. Test accuracy: 95.56%, Macro-F1: 94.06%. Zero test leakage. |
| **Real-time Field / UAV Deployment** | **NOT VERIFIED (Proposed Only)** | Conceptual in `docs/research_spec.md` | **AUDIT BOUNDARY**: No physical UAV imagery, flight hardware, or field trials were executed in this repository. All experiments used curated PlantVillage laboratory/field leaf imagery. UAV integration is strictly a proposed future extension. |

---

## 2. Explicit Scientific Clarifications

1. **Role of STO**: The title describes an *"STO based LS-HGNet Classification Model"*. The repository confirms that STO was implemented and benchmarked as a global hyperparameter exploration mechanism (M6, M7, M7.1, M7.2) to discover effective parameter spaces, but standard gradient descent (AdamW) performed all convolutional weight updates. Papers and presentations must describe STO as a *hyperparameter optimization mechanism*, not an end-to-end backpropagation replacement.
2. **Domain Boundary**: The primary research paper from Krishnan et al. (2024) studied maize tassel detection in UAV imagery. This repository adapts the LS-HGNet architecture specifically to **tomato leaf disease classification**.
