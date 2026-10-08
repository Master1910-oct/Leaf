# M10 — Final Per-Class Performance Analysis

**Source Artifact**: `results/m9_final_locked_test/metrics/per_class_metrics.csv`  
**Dataset**: Locked Test Set (`data/test.csv`, 2,724 samples)

---

## 1. Complete Per-Class Metrics Table

| Class Index | Pathology / Condition | Precision (%) | Recall (%) | F1-Score (%) | Support | Error Count |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `Tomato___Bacterial_spot` | 97.08% | 93.73% | 95.37% | 319 | 20 |
| 1 | `Tomato___Early_blight` | 95.52% | 85.33% | 90.14% | 150 | 22 |
| 2 | `Tomato___Late_blight` | 93.98% | 98.25% | 96.07% | 286 | 5 |
| 3 | `Tomato___Leaf_Mold` | 97.12% | 94.41% | 95.74% | 143 | 8 |
| 4 | `Tomato___Septoria_leaf_spot` | 94.55% | 98.11% | 96.30% | 265 | 5 |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | 85.96% | 97.22% | 91.25% | 252 | 7 |
| 6 | `Tomato___Target_Spot` | 91.58% | 82.46% | 86.78% | 211 | 37 |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 99.62% | 98.63% | 99.12% | 804 | 11 |
| 8 | `Tomato___Tomato_mosaic_virus` | 84.62% | 98.21% | 90.91% | 56 | 1 |
| 9 | `Tomato___healthy` | 100.00% | 97.90% | 98.94% | 238 | 5 |

---

## 2. Strongest Performing Classes

1. **`Tomato___Tomato_Yellow_Leaf_Curl_Virus`**:
   - **F1-Score**: **99.12%** (Precision: 99.62%, Recall: 98.63%, Support: 804)
   - *Characteristics*: Strong distinctive phenotypic signatures (prominent upward leaf curling and chlorosis) combined with robust training support enable near-perfect identification.
2. **`Tomato___healthy`**:
   - **F1-Score**: **98.94%** (Precision: 100.00%, Recall: 97.90%, Support: 238)
   - *Characteristics*: Flawless precision (0 false positives). Absence of necrotic lesions and uniform chlorosis provides strong negative separation.
3. **`Tomato___Septoria_leaf_spot`**:
   - **F1-Score**: **96.30%** (Precision: 94.55%, Recall: 98.11%, Support: 265)
   - *Characteristics*: Tiny circular lesions with distinct gray centers and dark borders are sharply resolved by the dilated multi-scale receptive field.

---

## 3. Most Vulnerable Classes

1. **`Tomato___Target_Spot`**:
   - **F1-Score**: **86.78%** (Precision: 91.58%, Recall: 82.46%, Support: 211)
   - *Vulnerability*: Lowest recall across all classes (82.46%). Target spot lesions share concentric ring patterns with early blight and diffuse chlorotic speckling with spider mites.
2. **`Tomato___Early_blight`**:
   - **F1-Score**: **90.14%** (Precision: 95.52%, Recall: 85.33%, Support: 150)
   - *Vulnerability*: Moderate recall (85.33%). Irregular brown lesions on older leaves exhibit overlapping visual traits with late blight and target spot.
3. **`Tomato___Tomato_mosaic_virus`**:
   - **F1-Score**: **90.91%** (Precision: 84.62%, Recall: 98.21%, Support: 56)
   - *Vulnerability*: Lowest precision across all classes (84.62%) despite high recall (98.21%). Low sample support (56 images) leads to slight over-prediction when diffuse mottling appears in other pathologies.
