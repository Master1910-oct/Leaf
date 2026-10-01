# AgriAI — Adaptive Smart Farming Framework

**Final-Year Research Project**  
**Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis**

---

## Project Status: M0 — AWAITING APPROVAL

This repository is at the **M0 Research Lock** milestone. No implementation code has been written. The M0 documents require explicit approval before M1 (dataset discovery and implementation) begins.

---

## What This Project Does

AgriAI investigates whether a lightweight stacked-hourglass-inspired neural network (LS-HGNet), with hyperparameters optimized using the Sooty Tern Optimization Algorithm (STOA), can deliver:

- Accurate tomato leaf disease classification (PlantVillage dataset, 10 classes)
- Computational efficiency (lightweight architecture, parameter-efficient design)
- Explainability (Grad-CAM visual explanations)
- Calibrated confidence (temperature scaling)
- Robustness under image degradation (7 perturbation types)
- Deployable inference (Flask REST API + React dashboard)

---

## Originality Disclaimer

> This is an **independent adaptation and extension** of concepts from prior LS-HGNet and Sooty Tern Optimization research. The original research task (UAV-based maize tassel detection), dataset, code, results, and figures are not reproduced. All experimental results, source code, figures, and numerical values in this project are independently produced.

---

## Repository Structure (M0)

```
smart-farming-agri-ai/
|
+-- docs/
|   +-- research_spec.md          <- Primary research specification (M0 locked)
|   +-- architecture_decisions.md <- All engineering decisions documented
|
+-- configs/
|   +-- development.yaml          <- Development configuration
|   +-- final.yaml                <- Final experiment configuration
|
+-- .env.example                  <- Environment variable template
+-- .gitignore                    <- Protects datasets, weights, secrets
+-- README.md                     <- This file
```

Implementation directories (src/, data/, experiments/, models/, frontend/, api/) will be created in M1 after M0 is approved.

---

## M0 Documents

| Document | Purpose | Status |
|----------|---------|--------|
| [docs/research_spec.md](docs/research_spec.md) | Complete research specification | M0 Draft |
| [docs/architecture_decisions.md](docs/architecture_decisions.md) | All engineering decision records | M0 Draft |
| [configs/development.yaml](configs/development.yaml) | Development phase configuration | Frozen |
| [configs/final.yaml](configs/final.yaml) | Final experiment configuration | Frozen |

---

## Key Technical Decisions (Summary)

| Component | Decision |
|-----------|---------|
| Dataset | PlantVillage — Tomato subset (10 classes) |
| Architecture | LS-HGNet: Lightweight Stacked Hourglass CNN adapted for classification |
| Optimizer | STOA (Sooty Tern Optimization Algorithm) for hyperparameter search |
| Baseline model | Simple 3-block CNN (~500K parameters) |
| Comparison | Equal-budget Random Search (25 dev / 100 final evaluations) |
| Explainability | Grad-CAM on final decoder block |
| Calibration | Temperature scaling fitted on validation data |
| Robustness | 7 synthetic perturbation types on test set |
| Deployment | Flask REST API + React dashboard |
| Dev input size | 128 x 128 |
| Final input size | 224 x 224 |
| Dev stacks/channels | 2 stacks, 32 base channels |
| Final stacks/channels | 3 stacks, 64 base channels |

---

## Research Question

> **Can an STO-optimized lightweight LS-HGNet architecture provide accurate, computationally efficient, explainable, calibrated, and robust tomato leaf disease classification, and does STO hyperparameter optimization yield a measurable advantage over equal-budget Random Search?**

---

## Experiment Matrix

| ID | Model | Hyperparameters | Purpose |
|----|-------|----------------|---------|
| EXP_001 | Baseline CNN | Default | Experimental control |
| EXP_002 | LS-HGNet | Default | Architecture evaluation |
| EXP_003 | LS-HGNet + STOA | Best from STOA | Optimizer benefit test |
| EXP_004 | LS-HGNet + Random Search | Best from RS | Equal-budget control |

---

## Open Questions Before M1

1. Exact STOA C_f coefficient range (2->1 vs. 2->0) — must read Dhiman & Kumar (2019).
2. Source smart-farming paper exact citation — literature search needed.
3. PlantVillage tomato per-class exact image counts — verify after download.
4. PlantVillage license — verify from dataset README.
5. Whether 224x224 at batch 32 fits 16 GB RAM — memory profiling in M1.

---

## Important Rules

- **Test set is NEVER used during hyperparameter optimization or calibration fitting.**
- **No metric is manually typed. All values are computed from code.**
- **No performance claims are made before experiments.**
- **STOA is not claimed to guarantee a global optimum.**
- **This system is a research prototype, not an agricultural diagnostic service.**

---

## References

1. Dhiman, G., & Kumar, V. (2019). STOA. *Engineering Applications of Artificial Intelligence*, 82.
2. Newell, A., Yang, K., & Deng, J. (2016). Stacked Hourglass Networks. *ECCV*. arXiv:1603.06937.
3. Hughes, D. P., & Salathé, M. (2015). PlantVillage. arXiv:1511.08060.
4. Howard, A. G., et al. (2017). MobileNets. arXiv:1704.04861.
5. Selvaraju, R. R., et al. (2017). Grad-CAM. *ICCV*. DOI:10.1109/ICCV.2017.74.
6. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
7. He, K., et al. (2016). Deep Residual Learning. *CVPR*.
8. Yu, F., & Koltun, V. (2016). Multi-scale Context Aggregation. *ICLR*. arXiv:1511.07122.

Full references with complete citations are in [docs/research_spec.md](docs/research_spec.md).

---

*M0 Research Lock — 2026-10-01 — Status: AWAITING APPROVAL*
