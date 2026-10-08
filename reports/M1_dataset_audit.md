# AgriAI — M1.3 Dataset Audit Report

**Date:** 2026-10-02  
**Dataset:** PlantVillage Tomato Disease Subset  
**Raw Directory:** `d:\Leaf\data\raw\plantvillage_tomato`  
**Status:** PASS (0 corrupt files)

---

## 1. Executive Summary

| Metric | Measured Value |
|--------|----------------|
| **Total Images** | 18,160 |
| **Number of Classes** | 10 |
| **Corrupt / Unreadable Files** | 0 |
| **Class Imbalance Ratio** | 14.36:1 (Max: 5,357 / Min: 373) |
| **Duplicate Filenames Across Classes** | 0 |

---

## 2. Per-Class Image Distribution

| Index | Class Name | Image Count | Percentage |
|:-----:|:-----------|:-----------:|:----------:|
| 1 | `Tomato___Bacterial_spot` | 2,127 | 11.71% |
| 2 | `Tomato___Early_blight` | 1,000 | 5.51% |
| 3 | `Tomato___Late_blight` | 1,909 | 10.51% |
| 4 | `Tomato___Leaf_Mold` | 952 | 5.24% |
| 5 | `Tomato___Septoria_leaf_spot` | 1,771 | 9.75% |
| 6 | `Tomato___Spider_mites Two-spotted_spider_mite` | 1,676 | 9.23% |
| 7 | `Tomato___Target_Spot` | 1,404 | 7.73% |
| 8 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | 5,357 | 29.50% |
| 9 | `Tomato___Tomato_mosaic_virus` | 373 | 2.05% |
| 10 | `Tomato___healthy` | 1,591 | 8.76% |
| **Total** | **All 10 Classes** | **18,160** | **100.00%** |

---

## 3. Image Dimensions & Color Channels

| Dimension (W x H) | Count | Percentage |
|:-----------------:|:-----:|:----------:|
| 256x256 | 18,160 | 100.00% |

### Color Modes

- **Mode `RGB`**: 18,160 images (100.00%)

### Image Formats

- **Format `JPEG`**: 18,160 images (100.00%)

---

## 4. Integrity and Data Quality

- **Corrupt Files:** 0 files failed PIL image verification.
- **Duplicate Filenames:** 0 distinct filenames appear in multiple class folders (typical for PlantVillage due to original camera numbering convention e.g., `image (1).JPG`, distinguished by parent class directory).
- **Modification Note:** No files were altered, resized, or moved during this audit.

---

## 5. Audit Conclusion

The dataset satisfies all M0 research specification criteria:
1. All 10 expected tomato classes (1 healthy, 9 diseased) are present.
2. Verified total image count is confirmed from filesystem inspection.
3. Class imbalance is quantified (ratio 14.36:1), justifying stratified splitting and macro-averaged evaluation metrics.
4. All images are readable and uncorrupted.
