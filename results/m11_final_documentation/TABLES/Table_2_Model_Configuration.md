# Table 2: LS-HGNet Architectural & Training Hyperparameter Specification

| Hyperparameter / Component | Specification / Value | Source / Verification |
| :--- | :--- | :--- |
| **Model Backbone** | Lightweight Stacked Hourglass Network (LS-HGNet) | `src/models/ls_hgnet.py` |
| **Total Trainable Parameters** | **8,041,354** | Exact `sum(p.numel())` count |
| **Input Spatial Resolution** | $224 \times 224$ pixels, 3 channels (RGB) | Verified deterministic transform |
| **Base Channels ($C_{base}$)** | 64 | Top-level classifier input: `[10, 64]` |
| **Hourglass Stacks ($N_{stacks}$)** | 3 recurrent stacks | `stacks.0`, `stacks.1`, `stacks.2` |
| **Encoder Levels per Stack** | 4 multi-scale downsampling stages | MaxPool2d stride 2 at each level |
| **Bottleneck Dilation Rates** | $(1, 2, 4)$ parallel multi-dilation branches | Receptive field expansion without pooling |
| **Bottleneck Channel Reduction** | Ratio 2 ($C_{inner} = C_{base} // 2 = 32$) | Parameter-efficient computation |
| **Dropout Probability** | 0.30 | Applied prior to final linear projection |
| **Classification Head** | AdaptiveAvgPool2d $\to$ Flatten $\to$ Linear(64, 10) | Single global linear mapping |
| **Optimizer & Base LR** | AdamW, Initial Learning Rate = $1.0 \times 10^{-3}$ | Verified from `m7_3_config.json` |
| **Weight Decay** | $1.0 \times 10^{-4}$ | Regularization penalty |
| **LR Scheduler** | Cosine Annealing Schedule | Min LR: $2.54 \times 10^{-5}$ at epoch 9 |
| **Batch Size** | 32 | Evaluation and full training |
| **Loss Function** | Cross-Entropy Loss with Softmax | Standard multi-class objective |
