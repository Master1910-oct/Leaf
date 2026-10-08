# M10 — Reproducibility & Governance Audit

**Audit Date**: 2026-10-08 UTC  
**Scope**: Full reproducibility pipeline for the frozen M7.3 checkpoint and M9 locked-test evaluation.

---

## 1. Checkpoint Identification & Cryptographic Provenance

| Verification Item | Specification / Value | Verification Status |
| :--- | :--- | :--- |
| **Model Architecture** | `LS-HGNet` | Verified exact |
| **Primary Checkpoint** | `results/m7_3_full_training/checkpoints/best_model.pt` | Verified exact |
| **Target Epoch** | Epoch 9 (of 10-epoch convergence schedule) | Verified exact |
| **File Size** | 97,278,348 bytes (92.77 MB) | Verified exact |
| **SHA-256 Checksum** | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` | **PASS (Cryptographically Verified)** |
| **State Dict Loading** | `strict=True` | **PASS (0 missing, 0 unexpected keys)** |
| **Total Parameters** | 8,041,354 | Verified exact |

---

## 2. Evaluation Protocol & Environment

| Component | Detail |
| :--- | :--- |
| **Input Resolution** | $224 \times 224$ pixels |
| **Preprocessing Transform** | Deterministic `Resize((224, 224))` $\to$ `ToTensor()` $\to$ `Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])` |
| **Augmentations** | **NONE** (Standard deterministic evaluation) |
| **Batch Size** | 32 |
| **Execution Mode** | `model.eval()`, `torch.no_grad()` |
| **Hardware** | Intel64 Family 6 Model 191 Stepping 2, GenuineIntel (CPU) |
| **Python Version** | 3.12.10 |
| **PyTorch Version** | 2.2.2+cpu |

---

## 3. Test Governance & Isolation Certification

```text
M9 TEST DATA GOVERNANCE AUDIT:
1. data/test.csv remained strictly locked throughout M0 to M8.2.
2. M9 performed evaluation exactly once (one-shot inference).
3. Zero training, fine-tuning, or model weight adjustments took place.
4. Classification thresholds were not optimized or tuned using test labels.
5. Temperature Scaling was NOT fitted on test data (frozen T=1.1558 from M8.2 validation was applied strictly for probability reporting).
6. M9 test inference is complete. No post-test tuning was performed.
7. No additional training was performed after M9.
```

**Governance Verdict**: **STRICT EXPERIMENTAL INTEGRITY PASS.**
