# M12.1 — External Dataset Evaluation Report

**Frozen LS-HGNet | M7.3 Epoch-9 Checkpoint**
Generated: 2026-10-08 18:20:45

---

## 1. Objective

This milestone evaluates the already-frozen, already-trained LS-HGNet model on an
independently sourced external image collection. The purpose is to understand
generalization behavior on data that was not part of the original train/validation/test pipeline.

The model was **NOT retrained, fine-tuned, or modified** in any way.

---

## 2. Dataset

**Location:** `D:\Leaf\test_data\archive\Dataset`

| Category | Folders | Images |
|----------|---------|--------|
| Tomato (in-distribution) | 10 | 16,011 |
| Non-Tomato OOD | 13 | 19,714 |
| **Total** | **23** | **35,725** |

> The frozen 10-class LS-HGNet model was evaluated on an external Tomato-only subset
> containing classes corresponding to its trained label space.
> Non-Tomato images were evaluated as an out-of-distribution stress test because
> their classes are outside the model's 10-class label space.

---

## 3. Model

| Property | Value |
|----------|-------|
| Architecture | LS-HGNet |
| Parameters | 8,041,354 |
| Classes | 10 (Tomato diseases) |
| Input Resolution | 224 × 224 |
| Checkpoint | `results/m7_3_full_training/checkpoints/best_model.pt` |
| SHA-256 | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` |
| Hash Verified | ✅ PASS |
| Weights Modified | ❌ NO |

---

## 4. Preprocessing

Identical to M9 evaluation pipeline:
- Resize to 224 × 224
- ToTensor
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- No augmentation, no test-time augmentation

---

## 5. Tomato External Results

### Overall

| Metric | Value |
|--------|-------|
| Total Images | 16,011 |
| Correct | 15,321 |
| Incorrect | 690 |
| **Accuracy** | **95.691%** |
| **Macro Precision** | **95.021%** |
| **Macro Recall** | **95.274%** |
| **Macro F1** | **95.034%** |
| **Weighted F1** | **95.666%** |
| Mean Confidence | 95.59% |
| Median Confidence | 99.88% |
| Mean Confidence (Correct) | 96.71% |
| Mean Confidence (Incorrect) | 70.57% |

### Per-Class

| Class | Support | Correct | Accuracy | Precision | Recall | F1 |
|-------|---------|---------|----------|-----------|--------|----|
| Bacterial Spot | 2127 | 1982 | 93.2% | 98.2% | 93.2% | 95.6% |
| Early Blight | 1000 | 896 | 89.6% | 96.1% | 89.6% | 92.8% |
| Late Blight | 1909 | 1882 | 98.6% | 94.7% | 98.6% | 96.6% |
| Leaf Mold | 952 | 922 | 96.8% | 98.3% | 96.8% | 97.6% |
| Septoria Spot | 1771 | 1732 | 97.8% | 96.0% | 97.8% | 96.9% |
| Spider Mites | 1676 | 1655 | 98.8% | 87.1% | 98.8% | 92.6% |
| Target Spot | 1404 | 1153 | 82.1% | 92.2% | 82.1% | 86.8% |
| TYLCV | 3208 | 3168 | 98.8% | 99.4% | 98.8% | 99.1% |
| Mosaic Virus | 373 | 369 | 98.9% | 89.3% | 98.9% | 93.9% |
| Healthy | 1591 | 1562 | 98.2% | 98.9% | 98.2% | 98.5% |


---

## 6. M9 vs External

| Metric | M9 Locked Test | External Tomato | Difference |
|--------|---------------|-----------------|------------|
| Accuracy | 95.558% | 95.691% | +0.13 pp |
| Macro Precision | 94.003% | 95.021% | +1.02 pp |
| Macro Recall | 94.427% | 95.274% | +0.85 pp |
| Macro F1 | 94.063% | 95.034% | +0.97 pp |
| Weighted F1 | 95.548% | 95.666% | +0.12 pp |

*M9 = original locked test set. External = separate external dataset. pp = percentage-point.*

---

## 7. Confusion Analysis

See `figures/tomato_confusion_matrix.png` and `figures/tomato_confusion_matrix_normalized.png`.

---

## 8. OOD Stress Test

| Statistic | Value |
|-----------|-------|
| Total OOD images | 19,714 |
| Mean max confidence | 82.23% |
| Median max confidence | 90.12% |
| % above 50% conf | 91.57% |
| % above 80% conf | 63.57% |
| % above 90% conf | 50.20% |

---

## 9. Visual Results

| Figure | Description |
|--------|-------------|
| `figures/tomato_per_class_accuracy.png` | Per-class accuracy bar chart |
| `figures/tomato_per_class_f1.png` | Per-class F1 bar chart |
| `figures/tomato_confidence_distribution.png` | Confidence split: correct vs incorrect |
| `figures/tomato_correct_vs_incorrect.png` | Count comparison |
| `figures/tomato_confusion_matrix.png` | Raw confusion matrix |
| `figures/tomato_confusion_matrix_normalized.png` | Normalized confusion matrix |
| `figures/prediction_examples.png` | 6 correct + 6 incorrect examples |
| `figures/ood_predicted_class_distribution.png` | OOD predicted-class histogram |
| `figures/ood_confidence_distribution.png` | OOD confidence histogram |

---

## 10. Limitations

- External dataset may have different image acquisition conditions than the original training data.
- Dataset provenance relative to the original training/test data could not be independently established from filenames alone.
- External images may differ in camera, resolution, lighting, and background characteristics.
- This is not field validation or deployment testing.
- This is not universal real-world accuracy measurement.
- OOD confidence statistics do not represent OOD detection accuracy.
- The model has no rejection mechanism; it always assigns one of 10 Tomato labels.

---

## 11. Research Integrity

| Rule | Status |
|------|--------|
| Model retrained | ❌ NO |
| Model fine-tuned | ❌ NO |
| Weights modified | ❌ NO |
| M9 modified | ❌ NO |
| M10 modified | ❌ NO |
| M11 modified | ❌ NO |
| Threshold tuning | ❌ NO |
| External dataset added to train/val/test | ❌ NO |
| SHA-256 verified | ✅ YES — MATCH |
