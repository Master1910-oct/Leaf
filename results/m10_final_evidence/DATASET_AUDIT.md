# M10 — Dataset & Class Mapping Audit

**Dataset**: Tomato Leaf Disease Dataset (PlantVillage Curation)  
**Total Images**: 18,160  
**Audit Scope**: Manifests `data/train.csv`, `data/val.csv`, `data/test.csv`.

---

## 1. Split Allocation & Data Partitioning

| Split Manifest | Sample Count | Percentage | Missing Files | Duplicates | Null Values |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `data/train.csv` | **12,712** | 70.00% | 0 | 0 | 0 |
| `data/val.csv` | **2,724** | 15.00% | 0 | 0 | 0 |
| `data/test.csv` | **2,724** | 15.00% | 0 | 0 | 0 |
| **Total** | **18,160** | **100.00%** | **0** | **0** | **0** |

---

## 2. Split Isolation & Leakage Verification

Set intersection across filepaths was computed independently:
- **Train $\cap$ Test**: `len(train_fps & test_fps) == 0` (**0 overlapping samples**)
- **Val $\cap$ Test**: `len(val_fps & test_fps) == 0` (**0 overlapping samples**)
- **Train $\cap$ Val**: `len(train_fps & val_fps) == 0` (**0 overlapping samples**)

**Leakage Verdict**: **ZERO DATA LEAKAGE DETECTED (STRICT ISOLATION CONFIRMED)**.

---

## 3. Class Mapping & Per-Split Distribution

| Index | Pathology / Class Name | Train Support | Val Support | Test Support | Total Count | % of Dataset |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `Tomato___Bacterial_spot` | 1489 | 319 | 319 | 2127 | 11.71% |
| 1 | `Tomato___Early_blight` | 700 | 150 | 150 | 1000 | 5.51% |
| 2 | `Tomato___Late_blight` | 1336 | 287 | 286 | 1909 | 10.51% |
| 3 | `Tomato___Leaf_Mold` | 666 | 143 | 143 | 952 | 5.24% |
| 4 | `Tomato___Septoria_leaf_spot` | 1240 | 266 | 265 | 1771 | 9.75% |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` | 1173 | 251 | 252 | 1676 | 9.23% |
| 6 | `Tomato___Target_Spot` | 983 | 210 | 211 | 1404 | 7.73% |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 3750 | 803 | 804 | 5357 | 29.5% |
| 8 | `Tomato___Tomato_mosaic_virus` | 261 | 56 | 56 | 373 | 2.05% |
| 9 | `Tomato___healthy` | 1114 | 239 | 238 | 1591 | 8.76% |

---

## 4. Class Balance Observation

- The largest class is `Tomato___Tomato_Yellow_Leaf_Curl_Virus` with 5,357 total samples (29.50% of dataset).
- The smallest class is `Tomato___Tomato_mosaic_virus` with 373 total samples (2.05% of dataset).
- The stratified 70/15/15 sampling accurately preserved class frequency proportions across all three partitions.
