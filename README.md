# AgriAI — Adaptive Smart Farming Framework

**Final-Year Research Project**  
**Adaptive Smart Farming Framework using STO-Optimized LS-HGNet with Explainable Crop Disease Analysis**

---

## Project Status: ✅ M0 LOCKED — Research Specification Approved

**Status: M0 LOCKED (2026-10-01)**  
Primary literature identities for STOA (Dhiman & Kaur, 2019) and the LS-HGNet smart farming framework (Krishnan et al., 2024) are confirmed. The original STOA formulation and the project-modified stabilized/additive variant are explicitly separated into distinct specifications. The STOA equation formulation is formally accepted as literature-verified (multi-source consensus; see RES-09 in `docs/research_spec.md`). All architecture and experiment design decisions are locked in `docs/architecture_decisions.md`. **M1 — Dataset & Environment Setup may now begin.**

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
|   +-- research_spec.md          <- Primary research specification (M0 LOCKED)
|   +-- architecture_decisions.md <- All engineering decision records (M0 LOCKED)
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
| [docs/research_spec.md](docs/research_spec.md) | Complete research specification | ✅ M0 LOCKED |
| [docs/architecture_decisions.md](docs/architecture_decisions.md) | All engineering decision records | ✅ M0 LOCKED |
| [configs/development.yaml](configs/development.yaml) | Development phase configuration | ✅ M0 LOCKED |
| [configs/final.yaml](configs/final.yaml) | Final experiment configuration | ✅ M0 LOCKED |

---

## Key Technical Decisions (Summary)

| Component | Decision |
|-----------|---------|
| Dataset | PlantVillage — Tomato subset (10 classes) |
| Architecture | LS-HGNet: Lightweight Stacked Hourglass CNN adapted for classification |
| Optimizer | Project-Modified Stabilized STOA (Default); controls: `original_strict`, `random_search` |
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

| ID | Model | Optimizer / Hyperparameters | Purpose |
|----|-------|-----------------------------|---------|
| EXP_001 | Baseline CNN | Default hyperparameters | Experimental architecture control (~500K params) |
| EXP_002 | LS-HGNet | Default hyperparameters | Architectural baseline without hyperparameter tuning |
| EXP_003_original | LS-HGNet | Best from Original STOA (`original_strict`) | Evaluate source-derived STOA procedure as faithfully as possible |
| EXP_003_stabilized | LS-HGNet | Best from Stabilized STOA (`stabilized`) | Evaluate project-modified stabilized/additive STOA variant (Default) |
| EXP_004 | LS-HGNet | Best from Random Search (`random_search`) | Equal-budget non-metaheuristic search control (25 dev / 100 final) |

---

## Open Questions Status

### Resolved Specifications (All M0 Items)

1. **STOA Authors & Paper Identity:** Gaurav Dhiman and Amandeep Kaur (2019), *EAAI*, Vol. 82, pp. 148–174.
2. **Original vs. Stabilized Distinction:** Original source-derived STOA and project-modified stabilized/additive variant are formally separated.
3. **Stabilized Formulation Consistency:** Project modification adopts Option A (preserves best-guided convergence using $\vec{P}_{best}$ consistently throughout; no $\vec{P}_{rand}$).
4. **Three Optimizer Controls:** Explicitly separated into `original_strict`, `stabilized` (default), and `random_search` with equal-budget protocol (25 dev / 100 final).
5. **Data Split & Augmentation:** Stratified 70/15/15 split confirmed; vertical flip excluded from default pipeline on biological validity grounds (retained as optional ablation ABL-05).
6. **LS-HGNet Architecture Adaptation:** Hourglass topology, DWSConv block, MultiDilated bottleneck rates [1, 2, 4], and GAP+Linear classification head finalized.
7. **Original STOA Equation Verification (RES-09):** Formally accepted as Literature-Verified (multi-source consensus). Publisher PDF access not achieved; if discrepancies discovered via institutional access in M1, equations must be updated before publication.

### Deferred to M1/M2 (Not M0 Blockers)

| ID | Question | Milestone |
|----|----------|-----------|
| DEF-01 | PlantVillage exact per-class counts & licensing | M1: Dataset download |
| DEF-02 | Hardware memory feasibility at 224×224 / batch 32 | M1: Memory profiling |
| DEF-03 | Class-weighted CE benefit on macro F1 | M2: Ablation ABL-02 |
| DEF-04 | Final dropout rate | M1/M2: STO search |
| DEF-05 | Optional edge deployment latency | OBJ-11 (optional) |

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
10. Kudela, J. (2022). A critical analysis of recent metaheuristics: origin bias and structural flaws. *Swarm and Evolutionary Computation*, 74, 101133.
11. Kwok, C. (2025). On the origin bias and numerical divergence in population metaheuristics: The case of Sooty Tern Optimization. arXiv:2511.17556.

Full references with complete citations are in [docs/research_spec.md](docs/research_spec.md).

---

*M0 Research Specification — 2026-10-01 — Status: ✅ LOCKED — Research Specification Approved*
