# M8.1 — Grad-CAM Explainability Report for Frozen LS-HGNet

**Project:** AgriAI — STO-Optimized LS-HGNet for Explainable Crop Disease Classification in Smart Farming  
**Date:** 2026-10-06  
**Phase:** Milestone 8.1 (Model Explainability & Visual Attribution)  
**Status:** COMPLETE / FROZEN ARTIFACTS VERIFIED  

---

## 1. Objective

The objective of Milestone 8.1 is to implement and quantitatively validate **Gradient-weighted Class Activation Mapping (Grad-CAM)** for the frozen final LS-HGNet model.

In agricultural smart-farming applications, diagnostic transparency is paramount: farmers and agronomists must be confident that a deep neural network makes disease classifications based on genuine pathological foliar symptoms (lesions, fungal sporulation, chlorosis, viral mottling) rather than superficial background artifacts (soil, bench surface, leaf petioles, sensor noise).

---

## 2. Frozen Model & Checkpoint Integrity

- **Selected Model:** Candidate A — Default LS-HGNet (M7 Winner)
- **Architecture:** Adapted LS-HGNet (3 Stacks, 4 Encoder Levels, Dilations [1, 2, 4], Base Channels 64)
- **Total Parameters:** 8,041,354
- **Checkpoint Path:** `results/m7_final_selection/candidate_a_default/best_model.pt`
- **Checkpoint SHA-256:** `caef534f6fe0e03c90513443454d5fc4fa4c84e9edd1757eb8cd7cc402c8ab9c`
- **Checkpoint Verification:** Bitwise identical before and after Grad-CAM execution. Zero model weights modified.

---

## 3. Target Layer Specification

```text
Target layer: stacked_hourglass.stacks.2.decoder_blocks.3
Module path: model.stacked_hourglass.stacks.2.decoder_blocks.3
Why this layer was selected: Per Selvaraju et al. (2017) and the AgriAI architecture decisions (M0/M4), this module represents the final spatial decoder block of the last Hourglass stack, immediately before Global Average Pooling (GAP). It retains the deepest, most class-discriminative semantic representations while preserving 56x56 spatial resolution.
Feature-map shape: [1, 64, 56, 56]
```

### Architectural Justification
The selected layer `stacked_hourglass.stacks.2.decoder_blocks.3` is the final pre-activation residual block in the decoder of the third (and final) Hourglass stack. As established in the project specification and architectural decisions:
1. It is the deepest convolutional feature representation before the Global Average Pooling (GAP) layer.
2. It encapsulates features refined through all repeated downsampling, dilated multi-scale bottleneck aggregation, and intra/inter-stack residual skip connections.
3. It preserves a rich spatial grid of $56 \times 56$ resolution, allowing fine-grained spatial attribution of foliar lesions when projected back to the input image ($224 \times 224$).

---

## 4. Dataset & Data Isolation Policy

- **Split Used:** **Validation split exclusively** (`data/splits/val.csv`).
- **Independent Test Set Isolation:** **Strictly preserved.** The 2,724-image locked test set (`data/splits/test.csv`) was **NOT accessed**, loaded, or preprocessed during Grad-CAM development.
- **Total Images Analyzed:** 30 validation images.
- **Classes Represented:** All 10 diagnostic classes (Tomato pathologies + healthy control).

---

## 5. Grad-CAM Methodological Pipeline

The Grad-CAM pipeline was implemented from first principles in PyTorch:
1. **Forward Hook:** Attached to `stacked_hourglass.stacks.2.decoder_blocks.3` to capture activation feature maps $A \in \mathbb{R}^{1 \times 64 \times 56 \times 56}$.
2. **Forward Inference:** Evaluates raw logits $y \in \mathbb{R}^{1 \times 10}$ and softmax probabilities.
3. **Score Selection:** Selects raw logit $y^c$ corresponding to the target class $c$ (default: predicted class $\arg\max(y)$).
4. **Backward Hook:** Computes gradient tensor $G = \frac{\partial y^c}{\partial A} \in \mathbb{R}^{1 \times 64 \times 56 \times 56}$.
5. **Importance Weighting:** Computes channel importance weights via Global Average Pooling:
   $$\alpha_k^c = \frac{1}{56 \times 56} \sum_{i=1}^{56} \sum_{j=1}^{56} \frac{\partial y^c}{\partial A_{i, j}^k}$$
6. **Linear Combination & Rectification:**
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^{64} \alpha_k^c A^k \right)$$
7. **Bilinear Upsampling:** Projects the $56 \times 56$ activation map to input dimensions ($224 \times 224$).
8. **Min-Max Normalization:** Rescales spatial activations into $[0.0, 1.0]$.
9. **Visual Overlay:** Blends the jet-colormap heatmap with the original RGB image (alpha blending weight $\alpha = 0.5$).
10. **Guaranteed Cleanup:** Hooks are safely deregistered via a `finally` block to prevent GPU/CPU memory leaks.

---

## 6. Quantitative Validation & Diagnostics Summary

- **Total Samples Analyzed:** 30
- **Correct Predictions:** 21 / 30 (70.0%)
- **Mean Prediction Confidence:** 91.4%
- **Heatmap Value Range:** Strictly bounded in $[0.0, 1.0]$ across all processed samples.
- **Gradient Health:** All samples produced strictly non-zero gradient norms (mean norm $> 0$).
- **Activation Health:** Zero NaN or Inf values detected.

### Per-Class Diagnostic Summary

| Class Slug | Full Class Name | Samples Analyzed | Accuracy | Mean Confidence |
| :--- | :--- | :---: | :---: | :---: |
| `bacterial_spot` | Bacterial_spot | 3 | 66.7% | 80.6% |
| `early_blight` | Early_blight | 3 | 66.7% | 95.9% |
| `late_blight` | Late_blight | 3 | 66.7% | 86.2% |
| `leaf_mold` | Leaf_Mold | 3 | 66.7% | 84.6% |
| `septoria` | Septoria_leaf_spot | 3 | 66.7% | 94.4% |
| `spider_mites` | Spider_mites Two-spotted_spider_mite | 3 | 66.7% | 99.3% |
| `target_spot` | Target_Spot | 3 | 66.7% | 89.9% |
| `yellow_leaf_curl` | Tomato_Yellow_Leaf_Curl_Virus | 3 | 66.7% | 97.6% |
| `mosaic` | Tomato_mosaic_virus | 3 | 66.7% | 85.6% |
| `healthy` | healthy | 3 | 100.0% | 100.0% |

---

## 7. Analysis of Correct Predictions (Positive Alignments)

In the vast majority of correctly classified validation samples, Grad-CAM heatmaps demonstrate clear biological alignment with pathological symptoms:

- **Bacterial Spot (`bacterial_spot`):** Saliency is sharply localized over clustered punctate necrotic lesions on the leaflet surfaces, ignoring healthy margins.
- **Late Blight (`late_blight`):** The network focuses on broad, water-soaked necrotic zones and brown margins characteristic of *Phytophthora infestans* infection.
- **Septoria Leaf Spot (`septoria`):** Attention peaks tightly over small circular spots with distinct dark borders.
- **Tomato Yellow Leaf Curl Virus (`yellow_leaf_curl`):** Attention covers upward-curled leaf edges and chlorotic margins where viral stunting is most pronounced.
- **Healthy Foliage (`healthy`):** Attention is diffuse across the homogenous green leaf surface without concentrated focal points, confirming the absence of localized necrotic triggers.

---

## 8. Honest Analysis of Misclassifications (Diagnostic Error Modes)

To maintain strict scientific integrity, representative misclassifications were intentionally included and analyzed:

- **Image `c3ac7b7a-a626-4004-ae2b-58720f38f027___GCREC_Bact.Sp 6299.JPG`**:
  - **True Class:** `Tomato___Bacterial_spot`
  - **Predicted Class:** `Tomato___Spider_mites Two-spotted_spider_mite` (Confidence: 41.7%)
  - **Observation:** Low model confidence (41.7%). The bacterial lesions appear as diffuse microscopic punctations across the leaflet, closely resembling the fine yellow-white stippling caused by two-spotted spider mite feeding. Grad-CAM confirms attention is distributed across the stippled texture.

- **Image `d53bdcff-1ea7-46c7-9d46-701d104bf130___RS_Erly.B 8353.JPG`**:
  - **True Class:** `Tomato___Early_blight`
  - **Predicted Class:** `Tomato___Late_blight` (Confidence: 88.0%)
  - **Observation:** The lesion in this sample is in an advanced necrotic stage covering a large segment of the leaf margin. Without distinct concentric target rings, the broad necrotic blight morphology triggered high activation for Late Blight (*Phytophthora infestans*).

- **Image `36193f11-7789-416d-b0dd-0cae49f74836___GHLB Leaf 23.2 Day 18.JPG`**:
  - **True Class:** `Tomato___Late_blight`
  - **Predicted Class:** `Tomato___Tomato_Yellow_Leaf_Curl_Virus` (Confidence: 58.6%)
  - **Observation:** The leaf margin exhibits curling and peripheral chlorosis alongside water-soaking. Saliency focuses primarily on the chlorotic curled leaf edges rather than the dark necrotic center, leading to a TYLCV misprediction.

- **Image `4badfb88-91f8-49e0-baad-553c18b6bede___Crnl_L.Mold 6904.JPG`**:
  - **True Class:** `Tomato___Leaf_Mold`
  - **Predicted Class:** `Tomato___Tomato_mosaic_virus` (Confidence: 53.8%)
  - **Observation:** In this specimen, the olive-green fungal sporulation is subtle, while the upper leaf surface exhibits irregular light-green and yellow mottling that closely mimics the systemic mosaic pattern of Tomato Mosaic Virus.

- **Image `3ddb7094-9ee7-473d-8a01-b48ce8eda506___Matt.S_CG 6719.JPG`**:
  - **True Class:** `Tomato___Septoria_leaf_spot`
  - **Predicted Class:** `Tomato___Spider_mites Two-spotted_spider_mite` (Confidence: 83.4%)
  - **Observation:** High-density pinpoint Septoria lesions peppered across the lamina create a textured speckled appearance. The network's attention focuses on the cumulative speckled pattern, mistaking the aggregated necrotic spots for severe spider mite bronzing.

- **Image `7a4ac14c-d097-4a03-aa13-46e47a3d6bef___Com.G_SpM_FL 1298.JPG`**:
  - **True Class:** `Tomato___Spider_mites Two-spotted_spider_mite`
  - **Predicted Class:** `Tomato___Tomato_mosaic_virus` (Confidence: 98.0%)
  - **Observation:** Severe feeding damage caused extensive yellow chlorotic patches without visible spider webbing. The network assigned high confidence to Mosaic Virus based on broad laminar discoloration.

- **Image `eabe0c3f-2339-4a4c-a472-8f4b0fe36e97___Com.G_TgS_FL 1088.JPG`**:
  - **True Class:** `Tomato___Target_Spot`
  - **Predicted Class:** `Tomato___Spider_mites Two-spotted_spider_mite` (Confidence: 77.0%)
  - **Observation:** Early-stage Target Spot lesions lack fully defined concentric halos and appear as small scattered brown spots. Saliency is concentrated on the tiny spots, leading to spider mite misclassification.

- **Image `41a298af-bb2d-4870-8a7e-a02f26d46187___UF.GRC_YLCV_Lab 08454.JPG`**:
  - **True Class:** `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
  - **Predicted Class:** `Tomato___Bacterial_spot` (Confidence: 92.7%)
  - **Observation:** Secondary necrotic punctations present on the leaflet surface attracted dense Grad-CAM focus, overshadowing the primary viral upward leaf curling.

- **Image `65226438-efaa-4bf4-91e7-e459e3528565___PSU_CG 2143.JPG`**:
  - **True Class:** `Tomato___Tomato_mosaic_virus`
  - **Predicted Class:** `Tomato___Leaf_Mold` (Confidence: 57.5%)
  - **Observation:** Chlorotic yellow patches without distinct mottled borders were attended by the model, matching the diffuse yellow spotting typical of early Leaf Mold infection.

---

## 9. Limitations of Saliency-Based Interpretability

> [!IMPORTANT]
> **Scientific Transparency on Explainability:**  
> Grad-CAM provides a post-hoc gradient attribution map showing which spatial regions contributed highest mathematical weight to the model's output logit. **It does NOT prove biological causality.**  
> A highlighted region indicates statistical correlation within the learned feature manifold, but does not guarantee that the network understands the pathogen's biological etiology. In operational smart farming, Grad-CAM outputs should serve as an assistive visual second opinion for agronomists, not an autonomous medical verdict.

---

## 10. Conclusion & Milestone Status

- **Implementation Status:** Grad-CAM for the frozen LS-HGNet model is **FULLY IMPLEMENTED and VERIFIED**.
- **Output Verification:** 3-panel visual explanations, JSON quantitative logs, and target layer documentation successfully generated.
- **Integrity Verification:** Checkpoint bitwise identical before and after execution; zero test images accessed.
- **Readiness:** Milestone 8.1 is **COMPLETE**. The model is fully prepared for Milestone 8.2 (Confidence Calibration via Temperature Scaling).