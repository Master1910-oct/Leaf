# Model Provenance — M12.1 External Evaluation

## Checkpoint
| Field | Value |
|-------|-------|
| Path | `D:\Leaf\results\m7_3_full_training\checkpoints\best_model.pt` |
| Expected SHA-256 | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` |
| Actual SHA-256 | `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8` |
| Hash Match | ✅ VERIFIED |

## Architecture
| Parameter | Value |
|-----------|-------|
| Model | LS-HGNet |
| Parameters | 8,041,354 |
| Input Resolution | 224 × 224 |
| num_classes | 10 |
| base_channels | 64 |
| num_stacks | 3 |
| encoder_levels | 4 |
| dilation_rates | (1, 2, 4) |
| dropout | 0.30 |

## Preprocessing (M9-identical)
- Resize: 224 × 224
- ToTensor
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- No random augmentation
- No test-time augmentation

## Class Mapping
| Index | Class Name |
|-------|-----------|
| 0 | `Tomato___Bacterial_spot` |
| 1 | `Tomato___Early_blight` |
| 2 | `Tomato___Late_blight` |
| 3 | `Tomato___Leaf_Mold` |
| 4 | `Tomato___Septoria_leaf_spot` |
| 5 | `Tomato___Spider_mites Two-spotted_spider_mite` |
| 6 | `Tomato___Target_Spot` |
| 7 | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` |
| 8 | `Tomato___Tomato_mosaic_virus` |
| 9 | `Tomato___healthy` |


## Research Integrity
- Model weights: **NOT MODIFIED**
- Checkpoint: **NOT MODIFIED**
- M9: **NOT MODIFIED**
- M10: **NOT MODIFIED**
- M11: **NOT MODIFIED**
- No threshold tuning performed.
