# M8.2 -- Calibration, Robustness & Deployment Readiness Report

**Date**: 2026-10-08 07:28 UTC

**Project**: D:/Leaf  |  **Experiment**: M8.2

---

## 1. Executive Summary

This report documents **M8.2 -- Calibration, Robustness & Deployment Readiness**, a post-training analysis of the frozen M7.3 LS-HGNet model (Epoch 9). The checkpoint achieved **94.23% validation Macro-F1** and **95.52% validation accuracy** during M7.3. **The locked test set was not accessed during M8.2.**

Temperature scaling was applied (learned T = 1.1558), moderating mean confidence from 95.87% to 94.89% (holdout ECE: 0.0134 -> 0.0173). Maximum Macro-F1 degradation under mild perturbations: 47.82% (blur). Mean CPU latency (batch=1): 132.2 ms (P95 = 171.8 ms).

## 2. Input Model

| Field | Value |
|---|---|
| Architecture | LS-HGNet |
| Experiment | M7.3 Full-Training Convergence |
| Best Epoch | 9 |
| Checkpoint | `results/m7_3_full_training/checkpoints/best_model.pt` |
| SHA-256 | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` |
| Parameter Count | 8,041,354 |
| Checkpoint Size | 92.77 MB |
| Val Accuracy (M7.3) | 95.52% |
| Val Macro-F1 (M7.3) | 94.23% |
| M7 Frozen Baseline Macro-F1 | 88.16% |

## 3. Calibration Method

The validation set (2,724 samples) was split into a **calibration subset (1362 samples, 50%)** and a **calibration holdout (1362 samples, 50%)** using `StratifiedShuffleSplit` (seed=42). No test data was used.

**Temperature scaling** (Guo et al., 2017): a single scalar temperature T was fitted on the calibration subset by minimising NLL via L-BFGS (100 iterations, lr=0.01). **Model weights were NOT modified.**

**Learned temperature**: T = 1.155831

## 4. Reliability Analysis

### Before / After Temperature Scaling (Holdout Set)

| Metric | Before | After |
|---|---|---|
| NLL | 0.122237 | 0.124716 |
| Brier Score | 0.060023 | 0.060751 |
| ECE | 0.013430 | 0.017325 |
| Mean Confidence | 95.87% | 94.89% |
| Accuracy | 96.11% | 96.11% |

**Plots**: `plots/reliability_before.png`, `plots/reliability_after.png`

## 5. Robustness Evaluation

Evaluated on 1362 holdout samples (validation only, no test data). All six conditions applied to the **same samples** for paired comparison.

| Condition | Description | Accuracy | Macro-F1 | F1 Drop |
|---|---|---|---|---|
| 1_clean | No perturbation (clean images) | 96.11% | 95.12% | 0.00% |
| 2_brightness | Mild brightness increase (factor=1.3) | 94.71% | 92.88% | 2.24% |
| 3_contrast | Mild contrast reduction (factor=0.75) | 96.33% | 95.26% | -0.14% |
| 4_gaussian_noise | Mild Gaussian noise (std=0.05 in [0,1] tensor space) | 76.73% | 71.39% | 23.72% |
| 5_gaussian_blur | Mild Gaussian blur (radius=1.5) | 58.30% | 47.30% | 47.82% |
| 6_jpeg_compression | Mild JPEG compression (quality=60) | 94.20% | 92.50% | 2.62% |

**Maximum Macro-F1 degradation**: 47.82%

**Plots**: `plots/robustness_macro_f1.png`, `plots/robustness_accuracy.png`

## 6. Deployment Benchmark

| Field | Value |
|---|---|
| Checkpoint Size | 92.77 MB |
| Total Parameters | 8,041,354 |
| Device | CPU |
| Image Size | 224x224 |
| Batch Size | 1 |
| Warmup Runs | 10 |
| Measured Runs | 200 |
| Mean Latency | 132.18 ms |
| Median Latency | 128.16 ms |
| Std Dev | 16.71 ms |
| Min | 109.50 ms |
| Max | 221.75 ms |
| P95 | 171.77 ms |
| P99 | 202.67 ms |
| Throughput | 7.57 images/sec |
| Process RSS | 1122.9 MB |
| PyTorch | 2.2.2+cpu |
| Python | 3.12.10 |
| CPU | Intel64 Family 6 Model 191 Stepping 2, GenuineIntel |

**Plot**: `plots/inference_latency_distribution.png`

## 7. Limitations

- **CPU-only benchmark**: GPU latency not claimed or measured.
- **Single-image benchmark**: batch_size=1; batch throughput will be higher.
- **Calibration scope**: Temperature scaling fitted on 50% val. Calibration quality on test set unknown.
- **Synthetic perturbations**: Mild and controlled; real deployment may exhibit different degradation patterns.
- **Memory measurement**: Process RSS includes Python interpreter overhead.

## 8. Test Set Status

> **The locked test set was not accessed during M8.2.**

- `data/test.csv` was NOT opened.
- No test images were loaded.
- No test predictions were generated.
- No test metrics were computed.
- No parameters were tuned using test data.

## 9. Scientific Interpretation

The M7.3 checkpoint achieves **94.23% validation Macro-F1** -- a +6.07 point improvement over the M7 frozen baseline of 88.16%. This is strictly **validation performance**; test set generalisation remains unknown until final evaluation.

Temperature scaling adjusts probability confidence (ECE: 0.0134 -> 0.0173) but does not change classification decisions -- accuracy before (96.11%) and after (96.11%) scaling is identical. Temperature scaling is a probability-calibration method only.

Robustness analysis shows a maximum Macro-F1 drop of 47.82% under mild perturbations, indicating notable sensitivity to image quality changes.

CPU latency of 132.2 ms/image (P95=171.8 ms) is practical for offline or low-throughput workflows. High-throughput production deployment would require GPU or batched inference optimisation.

## 10. Readiness for Final Evaluation

M8.2 has produced all required artefacts:

- [x] SHA-256 verified checkpoint (`PASS`)
- [x] Calibration split manifests
- [x] Temperature scaling metrics (before/after)
- [x] Reliability diagrams
- [x] Robustness metrics (6 conditions)
- [x] Robustness plots
- [x] CPU deployment benchmark
- [x] Latency distribution plot
- [x] Data isolation audit
- [x] Reproducibility config

**The project is ready for the final locked-test evaluation (M9) when the team decides to proceed.**

> IMPORTANT: The final test evaluation must be performed exactly once, deliberately, on the frozen M7.3 Epoch-9 checkpoint (`best_model.pt`). It was NOT performed in M8.2.
