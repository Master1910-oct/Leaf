# AgriAI — Adaptive Smart Farming Framework

**Final-Year Research Project**  
**Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis**

---

## Project Status: M0 Research Lock — Pending Final Verification

**Status: NOT READY — CRITICAL VERIFICATION REMAINS**  
Critical verification remains for the exact primary-source STOA mathematical formulation and source LS-HGNet architectural specifications. Implementation code must NOT begin until M0 is verified and explicitly approved.

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
| [docs/research_spec.md](docs/research_spec.md) | Complete research specification | M0 Draft — Pending Final Verification |
| [docs/architecture_decisions.md](docs/architecture_decisions.md) | All engineering decision records | M0 Draft — Pending Final Verification |
| [configs/development.yaml](configs/development.yaml) | Development phase configuration | Pending Final Verification |
| [configs/final.yaml](configs/final.yaml) | Final experiment configuration | Pending Final Verification |

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

## Open Questions

### Critical — Must Be Resolved Before M1 Implementation
1. **STOA mathematical formulation:** Primary source verification of exact collision-avoidance coefficient schedule ($S_A = 2 - 2t/T_{max}$ vs. alternatives), spiral parameters ($\rho$ calculation and boundary stabilization), and position-update equation from Dhiman & Kaur (2019).
2. **Source LS-HGNet architectural parameters:** Verification of channel progression, block topologies, and stem structure from the primary smart-farming paper.
3. **Data split & augmentation protocol:** Stratified 70/15/15 split confirmed; vertical flip excluded from default pipeline (retained only as optional ablation).

### Non-Critical — Experimentally Resolved in M1/M2
4. **PlantVillage exact per-class counts & licensing:** Inspect physical image files and license documentation upon download in M1.
5. **Hardware memory feasibility:** Confirm whether 224x224 at batch 32 fits within 16 GB RAM via memory profiling during M1.
6. **Loss function class weighting:** Measure class imbalance on disk and test if weighted Cross-Entropy improves macro F1 over standard CE.

---

## Important Rules

- **Test set is NEVER used during hyperparameter optimization or calibration fitting.**
- **No metric is manually typed. All values are computed from code.**
- **No performance claims are made before experiments.**
- **STOA is not claimed to guarantee a global optimum.**
- **This system is a research prototype, not an agricultural diagnostic service.**

---

## References

1. Dhiman, G., & Kaur, A. (2019). STOA: A bio-inspired based optimization algorithm for industrial engineering problems. *Engineering Applications of Artificial Intelligence*, 82, 148–174. DOI: 10.1016/j.engappai.2019.03.021.
2. Krishnan, V. G., Vikranth, B., Sumithra, M., Laxmi, B. P., & Gowri, B. S. (2024). Smart Farming with Sooty Tern Optimization based LS-HGNet Classification Model. *International Journal of Experimental Research and Review*, 37(spl), 96–108. DOI: 10.52756/ijerr.2024.v37spl.008.
3. Newell, A., Yang, K., & Deng, J. (2016). Stacked Hourglass Networks for Human Pose Estimation. *ECCV*. arXiv:1603.06937.
4. Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv:1511.08060.
5. Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv:1704.04861.
6. Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*. DOI: 10.1109/ICCV.2017.74.
7. Guo, C., et al. (2017). On Calibration of Modern Neural Networks. *ICML*.
8. He, K., et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
9. Yu, F., & Koltun, V. (2016). Multi-scale Context Aggregation by Dilated Convolutions. *ICLR*. arXiv:1511.07122.

Full references with complete citations are in [docs/research_spec.md](docs/research_spec.md).

---

*M0 Research Lock — 2026-10-01 — Status: NOT READY — CRITICAL VERIFICATION REMAINS*
