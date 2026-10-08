# M10 — Experimental Timeline Reconstruction

**Audit Timestamp**: 2026-10-08 UTC  
**Project**: Tomato Leaf Disease Classification (AgriAI / Leaf)  
**Scope**: Progression from research inception (M0) through final locked-test evaluation (M9) and evidence consolidation (M10).

---

## 1. Experimental Chronology & Milestone Map

```
M0: Research Specification & Mathematical Formulation
 │  (Defined LS-HGNet, STO optimization algorithm, and 10-class PlantVillage tomato dataset)
 ▼
M1: Dataset Pipeline & Multi-Split Stratification
 │  (18,160 images partitioned into 70% Train [12,712], 15% Val [2,724], 15% Locked Test [2,724])
 ▼
M2–M5: Model Architecture Development & Baselines
 │  (Implemented Standard CNN baseline and custom LS-HGNet multi-scale architecture)
 ▼
M6: Optimization Exploration (STO vs Random Search)
 │  (Pilot hyperparameter screening comparing Social Spider Optimization against Random Search)
 ▼
M7: Model Selection & Baseline Lock
 │  (Screened Candidates A-E; Candidate A achieved 88.16% Macro-F1; frozen as official baseline)
 ▼
M7.1: Rapid STO Refinement
 │  (CPU-bounded rapid screening exploring boundary handling and population budgets)
 ▼
M7.2: Progressive STO Data-Scaling
 │  (Evaluated hyperparameter responsiveness under progressive data subsets: 20% → 50%)
 ▼
M7.3: Full-Training Convergence & Global Checkpoint Lock
 │  (10-epoch convergence on 100% training data; Epoch 9 achieved 94.23% Validation Macro-F1; Checkpoint frozen)
 ▼
M8.2: Calibration, Robustness & Deployment Profiling
 │  (Temperature Scaling T=1.1558; 6-condition robustness evaluation; CPU latency profiling; Test set locked)
 ▼
M9: Final Locked-Test Evaluation (One-Shot)
 │  (Single evaluation pass on untouched data/test.csv; Test Accuracy 95.56%, Macro-F1 94.06%; Zero leakage)
 ▼
M10: Final Evidence Consolidation & Research Package
    (Formal audit, cross-verification, timeline reconstruction, and master artifact synthesis)
```

---

## 2. Milestone-by-Milestone Evidence Log

### Milestone M0 — Research & Architectural Foundations
- **Status**: Confirmed Evidence
- **Key Artifacts**: `docs/research_spec.md`, `docs/architecture_decisions.md`
- **Focus**: Established mathematical formulation of Lightweight Stacked Hourglass Networks (LS-HGNet) with multi-scale dilated residual bottlenecks and Social Spider Optimization (STO).

### Milestone M1 — Dataset Curation & Partitioning
- **Status**: Confirmed Evidence
- **Key Artifacts**: `data/train.csv`, `data/val.csv`, `data/test.csv`, `docs/M1_dataset_setup.md`
- **Focus**: Partitioned 18,160 tomato leaf images across 10 classes into 12,712 training (70%), 2,724 validation (15%), and 2,724 locked test samples (15%). Split overlap verified as exactly 0.

### Milestones M2–M5 — Baseline CNN & LS-HGNet Core Implementation
- **Status**: Confirmed Evidence
- **Key Artifacts**: `src/models/baseline_cnn.py`, `src/models/ls_hgnet.py`, `results/baseline_final/`
- **Focus**: Constructed LS-HGNet with 3 stacks, 4 encoder levels, dilation rates (1, 2, 4), and 64 base channels (8,041,354 parameters).

### Milestone M6 — Search Strategy Benchmarking
- **Status**: Confirmed Evidence
- **Key Artifacts**: `scripts/run_sto.py`, `scripts/run_random_search.py`, `results/m6_comparison/`
- **Focus**: Explored STO spider dynamics against random hyperparameter exploration on proxy validation subsets.

### Milestone M7 — Hyperparameter Selection & Baseline Freeze
- **Status**: Confirmed Evidence
- **Key Artifacts**: `results/m7_final_selection/M7_FINAL_SELECTION_REPORT.md`
- **Focus**: Candidate A selected with 91.08% Accuracy and 88.16% Validation Macro-F1. Frozen as official historical baseline.

### Milestones M7.1 & M7.2 — Rapid Screening & Progressive Data-Scaling
- **Status**: Confirmed Evidence
- **Key Artifacts**: `results/m7_1_sto_refinement/`, `results/m7_2_progressive_sto/`
- **Focus**: Investigated convergence dynamics on 20% and 50% training subsets under strict runtime limits.

### Milestone M7.3 — Full-Training Convergence & Master Model Freeze
- **Status**: Confirmed Evidence
- **Key Artifacts**: `results/m7_3_full_training/checkpoints/best_model.pt`, `results/m7_3_full_training/M7_3_FINAL_AUDIT.md`
- **Focus**: Trained LS-HGNet on 100% training data for 10 epochs. Epoch 9 reached peak performance:
  - Validation Accuracy: **95.52%**
  - Validation Macro-F1: **94.23%**
  - Checkpoint SHA-256: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`
  - Checkpoint permanently frozen.

### Milestone M8.2 — Calibration, Robustness & Deployment Profiling
- **Status**: Confirmed Evidence
- **Key Artifacts**: `results/m8_2_calibration_robustness/M8_2_REPORT.md`
- **Focus**: Validation set partitioned into 50% calibration (1,362) and 50% holdout (1,362).
  - Temperature Scaling fitted on calibration subset: learned $T = 1.155831$.
  - Clean Holdout Accuracy: **96.11%**, Macro-F1: **95.12%**.
  - Robustness tested across 6 perturbations (Gaussian noise, blur, compression, illumination).
  - CPU inference latency profiled: Mean 132.18 ms/image (7.57 img/s).
  - Test partition remained completely locked.

### Milestone M9 — Final Locked-Test Evaluation
- **Status**: Confirmed Evidence
- **Key Artifacts**: `results/m9_final_locked_test/M9_FINAL_REPORT.md`, `results/m9_final_locked_test/metrics/test_metrics.json`
- **Focus**: One-shot evaluation on untouched `data/test.csv` (2,724 images):
  - Test Accuracy: **95.56%**
  - Test Macro-F1: **94.06%**
  - Generalization Gap: **+0.17 percentage points** vs M7.3 Full Val.
  - Zero test leakage; zero threshold or hyperparameter modification.

### Milestone M10 — Final Evidence Consolidation & Package Synthesis
- **Status**: Active Milestone
- **Focus**: Comprehensive auditing, cross-verification, and compilation of the definitive research package.
