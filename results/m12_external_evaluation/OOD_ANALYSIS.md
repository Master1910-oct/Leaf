# Out-of-Distribution (OOD) Stress Test Analysis — M12.1

## Purpose
The frozen LS-HGNet model has exactly **10 Tomato disease classes**.
Non-Tomato images (Apple, Corn, Pepper, Potato) are by definition **out-of-distribution**.
This analysis examines how the classifier behaves when presented with images outside its label space.

> **IMPORTANT:** OOD confidence statistics are NOT accuracy measurements.
> The model cannot classify Apple/Corn/Pepper/Potato images correctly because
> those classes are absent from its label space.

## OOD Dataset
| Species | Folders | Images |
|---------|---------|--------|
| Apple___Apple_scab | 1 | 2016 |
| Apple___Black_rot | 1 | 1987 |
| Apple___Cedar_apple_rust | 1 | 1760 |
| Apple___healthy | 1 | 2008 |
| Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot | 1 | 1642 |
| Corn_(maize)___Common_rust_ | 1 | 1907 |
| Corn_(maize)___Northern_Leaf_Blight | 1 | 1908 |
| Corn_(maize)___healthy | 1 | 1859 |
| Pepper__bell___Bacterial_spot | 1 | 997 |
| Pepper__bell___healthy | 1 | 1478 |
| Potato___Early_blight | 1 | 1000 |
| Potato___Late_blight | 1 | 1000 |
| Potato___healthy | 1 | 152 |


## OOD Confidence Statistics
| Statistic | Value |
|-----------|-------|
| Total OOD images | 19,714 |
| Mean max-softmax confidence | 82.23% |
| Median max-softmax confidence | 90.12% |
| % predictions above 50% confidence | 91.57% |
| % predictions above 80% confidence | 63.57% |
| % predictions above 90% confidence | 50.20% |

## Interpretation
High OOD confidence (mean > 50%) indicates the model is poorly calibrated for OOD detection.
This is expected for a softmax classifier without an explicit rejection class.
The model will always assign one of the 10 Tomato labels to any image presented to it.
An OOD detector or confidence thresholding would be needed for production deployment.

## Figures
- `figures/ood_predicted_class_distribution.png`
- `figures/ood_confidence_distribution.png`
