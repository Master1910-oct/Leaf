# M10 — Final Model Specification: LS-HGNet

**Model Name**: Lightweight Stacked Hourglass Network (LS-HGNet)  
**Implementation Source**: `src/models/ls_hgnet.py`  
**Frozen Checkpoint**: `results/m7_3_full_training/checkpoints/best_model.pt`  
**Verified Checkpoint SHA-256**: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`  
**Checkpoint File Size**: 97,278,348 bytes (92.77 MB)  
**Total Parameter Count**: 8,041,354  

---

## 1. Architectural Configuration & Hyperparameters

Cross-verification against `src/models/ls_hgnet.py`, `scripts/run_m73_full_training.py`, and `results/m9_final_locked_test/config/m9_config.json` confirms the exact specifications:

| Parameter | Value | Verification Status |
| :--- | :--- | :--- |
| **Model Class** | `LS_HGNet` | Verified exact |
| **Input Channels** | `3` (RGB) | Verified exact |
| **Input Spatial Resolution** | `224 × 224` pixels | Verified exact |
| **Output Classes** | `10` | Verified exact |
| **Base Channels ($C_{base}$)** | `64` | Verified exact |
| **Number of Hourglass Stacks ($N_{stacks}$)** | `3` | Verified exact |
| **Encoder Levels per Stack ($N_{levels}$)** | `4` | Verified exact |
| **Bottleneck Dilation Rates** | `(1, 2, 4)` | Verified exact |
| **Bottleneck Reduction Ratio** | `2` (channels halved in inner convs) | Verified exact |
| **Dropout Probability** | `0.30` | Verified exact |
| **Intermediate Supervision** | Inter-stack projection heads with residual connections | Verified exact |
| **Total Trainable Parameters** | **8,041,354** | Verified exact (`sum(p.numel())`) |

---

## 2. Layer-by-Layer Architectural Breakdown

1. **Stem (Entry Module)**:
   - Initial $3	imes 3$ convolution: 3 input channels $	o$ 64 channels, stride 1, padding 1.
   - Batch Normalization + ReLU activation.
   - Feature map remains $224 	imes 224 	imes 64$.

2. **Hourglass Modules (3 Recurring Stacks)**:
   - **Encoder Path (4 Levels)**:
     - Multi-level spatial downsampling via MaxPool2d (kernel 2, stride 2).
     - Each level applies a multi-dilation residual bottleneck:
       - Parallel $3	imes 3$ dilated depthwise/pointwise branches at dilation rates $r \in \{1, 2, 4\}$.
       - Dilated features are aggregated, maintaining multi-scale receptive field coverage without feature loss.
   - **Bottleneck**:
     - Deepest receptive field processing with multi-dilation aggregation.
   - **Decoder Path**:
     - Upsampling via nearest-neighbor interpolation ($2	imes$) followed by residual addition with corresponding encoder skip connections.
   - **Inter-Stack Connectivity**:
     - Residual sum of stack input and stack output feature maps passed to the next stack.

3. **Classification Head**:
   - Adaptive Average Pooling: collapses spatial dimensions $H 	imes W 	o 1 	imes 1$.
   - Flatten layer: 64-dimensional feature vector.
   - Dropout: $p = 0.30$.
   - Fully Connected Linear Classifier: $64 	o 10$ logits.

---

## 3. Discrepancy & Consistency Audit

- **Specification vs Implementation Check**:
  - `ARCH_BASE_CHANNELS = 64`: Matches checkpoint weight tensor `classifier.weight` shape `[10, 64]`.
  - `ARCH_NUM_STACKS = 3`: Matches state dict keys `stacks.0`, `stacks.1`, `stacks.2`.
  - `ARCH_ENCODER_LEVELS = 4`: Matches state dict encoder blocks 0, 1, 2, 3 per stack.
  - `ARCH_DILATION_RATES = (1, 2, 4)`: Matches state dict bottleneck branch weights (branch 0, 1, 2).
  - Parameter count: Exactly **8,041,354**; 0 missing keys, 0 unexpected keys when loaded with `strict=True`.
- **Audit Conclusion**: **ZERO DISCREPANCIES DETECTED.**
