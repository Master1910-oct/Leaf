# M11 — Claims & Evidence Verification Matrix

This matrix establishes the boundary between **verified empirical facts** and **unsubstantiated claims** to prevent overclaiming in publications, reports, and viva examinations.

| Claim | Evidence Source | Verification Status | Allowed Scientific Phrasing | Prohibited / Overstated Phrasing |
| :--- | :--- | :---: | :--- | :--- |
| **Final Checkpoint is Frozen** | `checkpoints/best_model.pt` SHA-256: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` | **VERIFIED** | "The model checkpoint was frozen at Epoch 9 with verified cryptographic hash." | "The model was continually updated during test evaluation." |
| **Test Set Contains 2,724 Samples** | `data/test.csv`, `manifests/m9_test_audit.json` | **VERIFIED** | "The locked test partition comprises 2,724 independent images." | Any claim of dynamic or re-sampled test sets. |
| **Zero Train/Test Leakage** | `results/m10_final_evidence/DATASET_AUDIT.md` | **VERIFIED** | "Data split separation was verified with zero sample overlap across train, val, and test." | "Evaluated across completely distinct geographic farms." |
| **Test Accuracy is 95.56%** | `results/m9_final_locked_test/metrics/test_metrics.json` | **VERIFIED** | "The model achieved 95.56% overall accuracy on the held-out test split." | "Achieved 100% infallible plant health detection." |
| **Test Macro-F1 is 94.06%** | `results/m9_final_locked_test/metrics/test_metrics.json` | **VERIFIED** | "The test Macro-F1 score is 94.06%, reflecting balanced multi-class recognition." | Rounding up to "95% Macro-F1" without qualification. |
| **Parameter Count is 8,041,354** | Layer inspection in `FINAL_MODEL_SPECIFICATION.md` | **VERIFIED** | "LS-HGNet contains 8,041,354 trainable parameters." | Claiming the network is under 1 million parameters. |
| **Generalization Gap is 0.17 percentage points** | `results/m10_final_evidence/GENERALIZATION_ANALYSIS.md` | **VERIFIED** | "Macro-F1 changed by only 0.17 percentage points between validation (94.23%) and test (94.06%)." | "Proves universal generalization across all agricultural domains." |
| **UAV / Drone Real-Time Flight** | No drone flight logs exist in repository | **PROPOSED ONLY** | "Drone-based aerial acquisition represents a proposed future extension." | "We deployed an autonomous drone in the field to classify crops." |
| **Real-Time Edge Deployment** | Benchmarked on CPU: 132.2 ms/image | **PARTIALLY VERIFIED** | "Single-image CPU inference latency was measured at 132.2 ms (7.57 img/s)." | "The model runs in real-time at 60 FPS on edge microcontrollers." |
| **Role of STO Algorithm** | Implemented in `src/optimization/`, M6/M7 logs | **VERIFIED (Screening)** | "STO was utilized for hyperparameter search space exploration." | "STO replaced gradient descent for neural network backpropagation." |
