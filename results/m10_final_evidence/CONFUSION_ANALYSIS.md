# M10 — Final Confusion Matrix Analysis

**Source Artifacts**:
- `results/m9_final_locked_test/metrics/confusion_matrix.csv`
- `results/m9_final_locked_test/plots/confusion_matrix.png`
- `results/m9_final_locked_test/plots/confusion_matrix_normalized.png`

**Total Test Predictions**: 2,724  
**Correct Predictions**: 2,603 (95.56%)  
**Total Misclassifications**: 121 (4.44%)  

---

## 1. Top Observed Confusion Pairs

| Rank | Ground Truth Pathology | Model Predicted Pathology | Error Count | Share of True Class (%) |
| :---: | :--- | :--- | :---: | :---: |
| 1 | `Tomato___Target_Spot` | `Tomato___Spider_mites Two-spotted_spider_mite` | **29** | 13.74% |
| 2 | `Tomato___Bacterial_spot` | `Tomato___Target_Spot` | **10** | 3.13% |
| 3 | `Tomato___Early_blight` | `Tomato___Late_blight` | **9** | 6.00% |
| 4 | `Tomato___Spider_mites Two-spotted_spider_mite` | `Tomato___Tomato_mosaic_virus` | **5** | 1.98% |
| 5 | `Tomato___Target_Spot` | `Tomato___Septoria_leaf_spot` | **5** | 2.37% |
| 6 | `Tomato___Early_blight` | `Tomato___Bacterial_spot` | **4** | 2.67% |
| 7 | `Tomato___Leaf_Mold` | `Tomato___Early_blight` | **4** | 2.80% |
| 8 | `Tomato___Bacterial_spot` | `Tomato___Septoria_leaf_spot` | **4** | 1.25% |

---

## 2. Technical Analysis of Model Confusion Patterns

### Pattern A: Target Spot $\to$ Two-Spotted Spider Mite (29 errors)
- **Observed Behavior**: The single largest source of error in the entire evaluation (29 out of 121 total errors, 24.0% of all errors).
- **Visual Overlap**: Severe Two-Spotted Spider Mite infestation produces yellow stippling and speckled leaf chlorosis. When Target Spot manifests as numerous small pinpoint lesions rather than mature concentric target rings, the high-frequency pattern closely resembles mite stippling.

### Pattern B: Early Blight $\to$ Late Blight (9 errors)
- **Observed Behavior**: 9 Early Blight samples were classified as Late Blight.
- **Visual Overlap**: Both conditions produce expanding dark necrotic leaf lesions. At advanced infection stages, water-soaked margins in Late Blight can visually mimic the irregular necrotic margins of Early Blight.

### Pattern C: Bacterial Spot $\to$ Target Spot (10 errors)
- **Observed Behavior**: 10 Bacterial Spot samples classified as Target Spot.
- **Visual Overlap**: Early Bacterial Spot lesions are small, dark brown-to-black spots often surrounded by a narrow chlorotic halo, sharing visual characteristics with early Target Spot lesions.

### Pattern D: Healthy Leaves (238 samples)
- **Observed Behavior**: Zero non-healthy leaves were misclassified as healthy (100.0% precision). Only 5 healthy samples were misclassified (2 as Leaf Mold, 2 as Spider Mites, 1 as Bacterial Spot), demonstrating high specificity for normal vegetation.
