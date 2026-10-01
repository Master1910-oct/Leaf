# Architecture Decisions Log — AgriAI
## Adaptive Smart Farming Framework using STO-Optimized LS-HGNet

**Document version:** M0-DRAFT  
**Date:** 2026-10-01  
**Status:** AWAITING APPROVAL  

---

## Purpose

This document records every significant engineering decision made during the design of the AgriAI system. Each entry distinguishes between what is supported by source literature and what is independently decided by this project. This log is a scientific integrity record and must be updated whenever an architecture, training, or experimental decision changes.

---

## Decision: Input Resolution

### Decision
Development configuration: 128 x 128 pixels.  
Final configuration: 224 x 224 pixels.

### Source basis
None. Neither the source smart-farming paper nor the Newell et al. hourglass paper specifies this resolution for disease classification. The source smart-farming paper used UAV imagery at its own resolution.

### Our engineering choice
128 x 128 for development (fast iteration), 224 x 224 for final experiments (standard classification resolution consistent with ImageNet-pretrained conventions).

### Reason
- 128 x 128 enables rapid prototyping on a 16 GB laptop without GPU memory exhaustion.
- 224 x 224 is the standard input size for most ImageNet-scale classification research, providing a fair comparison with published baselines.

### Alternatives considered
- 64 x 64: too small; likely to lose disease-discriminative texture detail.
- 256 x 256: feasible but increases training time without a clear benefit at this scale.
- 320 x 320: would require GPU with >16 GB VRAM for batch training.

### Trade-offs
- 128 x 128 reduces training cost but may limit fine-grained texture discrimination.
- 224 x 224 provides richer features but increases memory and compute requirements.
- If 224 x 224 is computationally infeasible during final experiments, this decision must be revisited and re-documented.

---

## Decision: Number of Hourglass Stacks

### Decision
Development configuration: 2 stacks.  
Final configuration: 3 stacks.

### Source basis
The stacked hourglass concept (Newell et al., 2016) used up to 8 stacks for human pose estimation. The source smart-farming paper's exact stack count is TO BE VERIFIED.

### Our engineering choice
2 stacks for development; 3 stacks for final. The optimal number is an experimental question answered by ablation ABL-03.

### Reason
- 2 stacks are tractable on 16 GB RAM during development.
- 3 stacks provide more refinement stages, potentially improving classification accuracy.
- Beyond 3 stacks would likely exceed memory constraints and add diminishing returns for a 10-class classification task.

### Alternatives considered
- 1 stack: loss of refinement benefit; too close to a plain encoder-decoder.
- 4+ stacks: likely memory-prohibitive on target hardware; added complexity may not benefit a classification head.

### Trade-offs
- More stacks: more representational capacity, higher memory usage, longer training.
- Fewer stacks: faster training, lower memory, potentially lower accuracy.

---

## Decision: Base Channel Count

### Decision
Development: base channels = 32.  
Final: base channels = 64.

### Source basis
Not specified in Newell et al. for this task. Source smart-farming paper's channel counts are TO BE VERIFIED.

### Our engineering choice
32 channels for development; 64 for final. Channel doubling at each encoder level.

### Reason
- 32 base channels produce a lightweight model suitable for rapid development and fitting within memory constraints.
- 64 base channels provide more expressive feature representations for final experiments.

### Alternatives considered
- 16 base channels: likely too limited for 10-class discrimination.
- 128 base channels: would significantly increase parameter count and memory.

### Trade-offs
- Fewer channels: faster, less memory, potentially weaker feature representations.
- More channels: richer features but higher computational cost.

---

## Decision: Dilation Rates in MultiDilatedBlock

### Decision
Dilation rates: [1, 2, 4] in parallel branches.

### Source basis
**None.** These dilation rates are NOT from the source smart-farming paper or from Newell et al. They are inspired by multi-scale dilated convolution literature (DeepLab: Chen et al., 2017; Multi-scale context aggregation: Yu & Koltun, 2016).

### Our engineering choice
[1, 2, 4] chosen to capture features at three spatial scales without excessive parameter increase.

### Reason
- Rate 1: standard spatial convolution.
- Rate 2: 2x expanded receptive field.
- Rate 4: 4x expanded receptive field.
- Together they provide multi-scale spatial context at the bottleneck.

### Alternatives considered
- [1, 2]: only two scales; less diversity.
- [1, 2, 4, 8]: four scales; increases concatenated channels by 4x; memory-intensive.
- [1, 3, 5]: odd-rate spacing; no strong principled motivation.
- No dilation (standard bottleneck): valid baseline; covered by ABL-01.

### Trade-offs
- More rates: richer multi-scale context, but concatenation increases channel count by factor of number-of-branches before the 1x1 fusion.
- The 1x1 fusion after concatenation partially mitigates the channel explosion.
- [1, 2, 4] is a common and well-validated choice in semantic segmentation literature.

---

## Decision: Inter-stack Skip Connections

### Decision
Present: the output feature map of each hourglass stack is added to (or concatenated with) the input of the next stack.

### Source basis
Newell et al. (2016) used intermediate supervision between stacks (loss at each stack's output) but did not explicitly describe inter-stack feature-level skip connections as a primary design element. The source smart-farming paper's exact implementation is TO BE VERIFIED.

### Our engineering choice
Include inter-stack skip connections to promote feature reuse across stacks and facilitate gradient flow.

### Reason
- Skip connections between stacks allow later stacks to refine earlier representations rather than learning from scratch.
- This is analogous to the dense connection principle (Huang et al., 2017, DenseNet) applied at the stack level.

### Alternatives considered
- No inter-stack skip: each stack operates independently; covered by ABL-02.
- Intermediate supervision only: add auxiliary loss at each stack's output (as in Newell et al.); adds training complexity.

### Trade-offs
- With skip: improves gradient flow, may improve accuracy, adds minor complexity.
- Without skip: each stack independently learns; potentially easier to train but may limit refinement.

---

## Decision: Upsampling Method in Hourglass Decoder

### Decision
Bilinear interpolation followed by a 1x1 convolution to adjust channels.

### Source basis
Newell et al. (2016) used nearest-neighbor upsampling in the original hourglass. The source smart-farming paper's upsampling method is TO BE VERIFIED.

### Our engineering choice
Bilinear upsampling preferred over transposed convolution.

### Reason
- Bilinear upsampling avoids checkerboard artifacts common with transposed convolutions.
- No additional learnable parameters for upsampling itself (only the subsequent 1x1 conv).
- Reduces parameter count and training instability.

### Alternatives considered
- Transposed convolution: learnable upsampling but prone to checkerboard artifacts and higher parameter count.
- Nearest-neighbor upsampling: simpler, but bilinear produces smoother feature maps.
- Pixel shuffle: more complex implementation without clear benefit at this scale.

### Trade-offs
- Bilinear: simpler, artifact-free, slightly less expressive than transposed conv.
- Transposed conv: more flexible but more parameters and training sensitivity.

---

## Decision: Residual Block Structure

### Decision
Post-activation: DWSConv -> BN -> ReLU -> DWSConv -> BN -> Residual Add -> ReLU.

### Source basis
He et al. (2016) introduced post-activation residual blocks (original ResNet). Pre-activation residual blocks (BN -> ReLU -> Conv) were proposed in He et al. (2016, Identity Mappings). Both are established designs.

### Our engineering choice
Post-activation structure as default, matching the original ResNet formulation.

### Reason
- Post-activation is the most widely reproduced and well-understood residual design.
- Pre-activation may improve gradient flow in very deep networks; our shallow hourglass stacks are unlikely to see this benefit.

### Alternatives considered
- Pre-activation (BN -> ReLU -> Conv): potentially better gradient properties in deep networks.
- No residual connection: would reduce the model to a plain network, removing the skip-connection benefit.

### Trade-offs
- Post-activation: simpler, well-tested.
- Pre-activation: marginal benefit for our depth; adds implementation complexity.

---

## Decision: Classifier Head Design

### Decision
Global Average Pooling -> Dropout -> Linear -> Raw Logits.

### Source basis
GAP-based classification heads are standard practice (Lin et al., 2014, Network in Network). No softmax inside the model.

### Our engineering choice
No softmax inside the model. Dropout rate is a STOA-searched hyperparameter.

### Reason
- PyTorch's nn.CrossEntropyLoss applies log_softmax internally. Applying softmax before CE loss causes incorrect loss computation.
- Temperature scaling (calibration) operates on raw logits. Including softmax inside the model would prevent direct temperature scaling.
- Keeping logits as output is consistent with PyTorch conventions and supports both calibration and flexible loss functions.

### Alternatives considered
- Fully connected layers instead of GAP: significantly more parameters; less spatial invariance.
- Average pooling + multiple FC layers: adds parameters without demonstrated benefit for this task.
- Softmax inside model: prevents correct CE loss and temperature scaling.

### Trade-offs
- GAP + single linear: simple, parameter-efficient, spatially invariant.
- Removing softmax: requires users to apply softmax explicitly at inference time (minor inconvenience, clearly documented).

---

## Decision: Optimizer

### Decision
Adam optimizer.

### Source basis
No specific optimizer prescribed by source papers for this task.

### Our engineering choice
Adam with STOA-searched learning rate and weight decay (applied as AdamW decoupled weight decay).

### Reason
- Adam is the standard choice for image classification training with reasonable convergence properties across a wide learning rate range.
- AdamW provides decoupled L2 regularization, which is more principled than L2 in Adam.

### Alternatives considered
- SGD with momentum: requires careful learning rate scheduling; more sensitive to hyperparameter choices.
- RMSProp: similar to Adam; no strong reason to prefer over Adam here.

### Trade-offs
- Adam/AdamW: fast convergence, less sensitive to learning rate, well-understood behavior.
- SGD: can generalize better with proper tuning but requires more careful scheduling.

---

## Decision: Learning Rate Scheduler

### Decision
CosineAnnealingLR scheduler.

### Source basis
No specific scheduler prescribed by source papers for this task.

### Our engineering choice
Cosine annealing with warm restart (CosineAnnealingLR or CosineAnnealingWarmRestarts).

### Reason
- Cosine annealing smoothly reduces learning rate, allowing the optimizer to fine-tune near convergence.
- Warm restarts can help escape local minima during early training.

### Alternatives considered
- StepLR: simple but abrupt drops; less smooth optimization trajectory.
- ReduceLROnPlateau: adaptive but adds monitoring complexity.
- Constant LR: simplest but often suboptimal.

### Trade-offs
- Cosine annealing: smooth, well-tested, parameter-light.
- Warm restarts: adds one hyperparameter (T_0 period); set to a fixed value (e.g., 10 epochs) as an engineering choice.

---

## Decision: Loss Function

### Decision
Cross-Entropy Loss (nn.CrossEntropyLoss).

### Source basis
Standard classification loss.

### Our engineering choice
Standard CE loss as default. If class imbalance is significant (confirmed during M1), class-weighted CE or focal loss will be considered as ablation.

### Reason
- CE loss is the standard and most interpretable classification loss.
- nn.CrossEntropyLoss includes log_softmax internally; raw logits must be passed (consistent with our classifier head decision).

### Alternatives considered
- Focal loss: addresses class imbalance by down-weighting easy examples; more complex to configure.
- Label smoothing CE: reduces overconfidence; adds one hyperparameter.
- Weighted CE: addresses class imbalance; weights computed from class frequencies.

### Trade-offs
- Standard CE: simple, well-understood; may be suboptimal under class imbalance.
- Weighted CE: better under imbalance but requires careful weight computation.

---

## Decision: Data Augmentation Policy

### Decision
Random horizontal flip (p=0.5), random vertical flip (p=0.5), random rotation (+/-30 deg), color jitter, random resized crop. Applied to training data only.

### Source basis
No augmentation policy specified by source papers for tomato classification.

### Our engineering choice
Standard augmentation set commonly used in plant disease classification literature.

### Reason
- Flips and rotations increase rotational invariance; leaf orientation in field images is arbitrary.
- Color jitter accounts for lighting variation.
- Random resized crop simulates partial occlusion and viewpoint variation.

### Alternatives considered
- No augmentation: likely to overfit on smaller classes.
- CutMix / MixUp: more aggressive augmentations; may be added as ablation.
- AutoAugment: computationally expensive policy search.

### Trade-offs
- More augmentation: better generalization, slower training.
- Less augmentation: faster training, higher overfitting risk.

---

## Decision: STO Search Space

### Decision
Four hyperparameters: learning rate [1e-4, 1e-2] log, weight decay [1e-6, 1e-2] log, dropout [0.0, 0.5] linear, batch size in {16, 32, 64}.

### Source basis
No STO search space prescribed in source papers.

### Our engineering choice
Search space designed based on commonly used ranges in CNN training literature.

### Reason
- Learning rate range covers orders of magnitude appropriate for Adam.
- Weight decay range includes near-zero (minimal regularization) to moderate (0.01).
- Dropout range [0, 0.5]: above 0.5 typically harms classification performance.
- Batch size: 16 is minimum for stable BN; 64 is maximum given 16 GB RAM constraint.

### Alternatives considered
- Including number of epochs in search space: not included to keep evaluation time fixed.
- Including scheduler parameters: adds search space dimensionality; left as fixed engineering choices.
- Wider LR range [1e-5, 1e-1]: too wide for STOA to explore efficiently within 25/100 evaluations.

### Trade-offs
- Narrow search space: faster convergence, risk of missing optimal region.
- Wide search space: more exploration, but inefficient with limited evaluation budget.

---

## Decision: STO Evaluation Budget

### Decision
Development: 5 agents x 5 iterations = 25 evaluations.  
Final: 10 agents x 10 iterations = 100 evaluations.

### Source basis
Budget is not prescribed by STOA original paper for this task.

### Our engineering choice
Budgets chosen based on computational feasibility on a 16 GB laptop.

### Reason
- 25 evaluations (development): fast enough for rapid iteration; large enough to detect obvious hyperparameter effects.
- 100 evaluations (final): standard for metaheuristic comparison experiments in hyperparameter optimization literature.

### Alternatives considered
- 200 evaluations: doubles cost; may provide marginally better optimization but exceeds practical constraints.
- 10 evaluations: too few to meaningfully evaluate STOA's search behavior.

### Trade-offs
- More evaluations: better optimization, much higher runtime.
- Fewer evaluations: faster, but optimization quality may be limited by budget rather than algorithm quality.

---

## Decision: Random Search Budget

### Decision
Equal to STOA budget at all stages: 25 (dev) and 100 (final).

### Source basis
Equal-budget comparison is the standard experimental protocol in metaheuristic optimization literature.

### Our engineering choice
Enforced equal budget as a scientific requirement.

### Reason
Any performance difference between STOA and Random Search must be attributable to algorithm design, not budget. Without equal budget, results cannot be interpreted as evidence of STOA's effectiveness.

### Alternatives considered
- Random Search with fewer evaluations: would bias results in favor of STOA; scientifically invalid.
- Random Search with more evaluations: would bias against STOA; also invalid.

### Trade-offs
- Equal budget: scientifically valid, clear interpretation.
- Unequal budget: simpler to implement but results are uninterpretable.

---

## Decision: Calibration Method

### Decision
Temperature scaling (Guo et al., 2017).

### Source basis
Guo et al. (2017) demonstrated temperature scaling as the most practical post-hoc calibration method for modern CNNs.

### Our engineering choice
Temperature scaling fitted on validation data; applied at inference time.

### Reason
- Single-parameter method: easy to implement, computationally trivial.
- Does not change model accuracy (only confidence).
- Fitted on validation data only; no test-set leakage.

### Alternatives considered
- Platt scaling: binary version; temperature scaling is the multiclass extension.
- Isotonic regression: non-parametric; tends to overfit on small calibration sets.
- Histogram binning: non-parametric; calibration quality depends on bin size.

### Trade-offs
- Temperature scaling: simple, effective for modern overconfident CNNs; assumes a global temperature applies equally to all classes.
- Isotonic regression: more flexible but may overfit.

---

## Decision: Grad-CAM Target Layer

### Decision
Output of the final decoder block of the last hourglass stack, immediately before Global Average Pooling.

### Source basis
Selvaraju et al. (2017) recommend the last convolutional layer before the classification head for Grad-CAM, as it retains the most abstract spatial features.

### Our engineering choice
Final decoder block output selected because it (a) is the deepest spatial feature map before the head, (b) contains class-discriminative information after all hourglass refinement, and (c) retains spatial resolution for meaningful localization.

### Reason
Earlier layers retain finer spatial detail but less class-discriminative information. Later layers (after GAP) lose spatial dimensions entirely. The final decoder output is the optimal balance.

### Alternatives considered
- Output of bottleneck: very low spatial resolution; heatmaps would be too coarse.
- Output of intermediate decoder levels: less refined features.
- Multiple layers (multi-layer Grad-CAM): possible but adds complexity; kept as an optional extension.

### Trade-offs
- Final decoder output: best class-discriminative + spatial balance.
- Earlier layers: finer detail, less discriminative.

---

## Decision: Robustness Test Perturbations

### Decision
Seven perturbation types: brightness increase, brightness decrease, contrast change, Gaussian noise, Gaussian blur, JPEG compression, low-light transformation. Applied to test set only.

### Source basis
Robustness evaluation under synthetic perturbations is a standard evaluation protocol (see Hendrycks & Dietterich, 2019, ImageNet-C).

### Our engineering choice
Selection of seven perturbations representing common real-world image degradation scenarios relevant to agricultural imagery.

### Reason
- Brightness and contrast: common in outdoor agricultural imaging due to sunlight variation.
- Gaussian noise: sensor noise in cameras.
- Gaussian blur: motion blur or out-of-focus capture.
- JPEG compression: lossy compression used in mobile devices.
- Low-light: night or indoor imaging scenarios.

### Alternatives considered
- More perturbations (e.g., fog, rain, elastic distortion): increases test coverage but also evaluation cost.
- Fewer perturbations: less comprehensive coverage.

### Trade-offs
- More types: more comprehensive; more implementation and evaluation time.
- Seven types: practical balance of coverage vs. effort for a final-year project.

---

## Decision: Morphological Preprocessing Treatment

### Decision
Top-hat and bottom-hat morphological transforms are treated as an **ablation experimental branch** (ABL-04), NOT as part of the core pipeline.

### Source basis
Morphological image processing for plant disease image enhancement is referenced in some agricultural vision papers, but is not a core component of the source smart-farming paper or the LS-HGNet architecture.

### Our engineering choice
Include as an optional ablation to determine whether morphological preprocessing improves performance on this task.

### Reason
- Top-hat enhances bright structures (potential lesion areas) against dark backgrounds.
- Bottom-hat enhances dark structures on bright backgrounds.
- These may or may not benefit tomato disease classification depending on the disease type.
- Including them as a mandatory preprocessing step without experimental justification would be an unverified assumption.

### Alternatives considered
- Include in all experiments: would make it impossible to know whether results were due to the preprocessing or the model.
- Exclude entirely: would miss a potentially useful preprocessing technique.

### Trade-offs
- Ablation branch: principled; allows measuring the effect of morphological preprocessing.
- Mandatory inclusion: confounds architecture and preprocessing effects.

---

## Research Decision Summary Table

| Decision | Status | Evidence | Final Choice |
|----------|--------|----------|-------------|
| Dataset | Literature-verified (counts TO BE VERIFIED from files) | Hughes & Salathé, 2015 | PlantVillage Tomato Subset |
| Classes | 10 classes widely reported (must verify from download) | Multiple published papers | 10 classes (1 healthy + 9 diseased) |
| Input resolution | Engineering decision | Computational constraint | 128x128 (dev), 224x224 (final) |
| LS-HGNet structure | Adapted from Newell et al. 2016 | Hourglass paper | Encoder-Bottleneck-Decoder with skip connections |
| STO equations | Partially verified; discrepancy in C_f range noted | Dhiman & Kumar 2019 + secondary sources | To be resolved from primary paper in M1 |
| Dilation rates | Engineering decision only | No source support | [1, 2, 4] (ablation ABL-01 planned) |
| Inter-stack skip | Engineering decision | No direct source support | Present (ablation ABL-02 planned) |
| Upsampling method | Engineering decision | Newell et al. used nearest-neighbor | Bilinear + 1x1 conv |
| Residual structure | Adapted from He et al. 2016 | ResNet paper | Post-activation DWSConv residual |
| Classifier head | Standard practice | Lin et al. 2014 + PyTorch conventions | GAP + Dropout + Linear, no softmax |
| Optimizer | Engineering decision | Standard DL practice | AdamW |
| Scheduler | Engineering decision | Standard DL practice | CosineAnnealingLR |
| Loss function | Engineering decision | Standard DL practice | CrossEntropyLoss (weighted if needed) |
| Augmentation | Engineering decision | Plant disease classification literature | Standard set (flip, rotate, jitter, crop) |
| STO search space | Engineering decision | Experiment design | 4 hyperparameters, ranges as specified |
| STO budget | Engineering decision | Computational constraint | 25 (dev), 100 (final) |
| Random Search budget | Scientific requirement | Equal-budget protocol | Equal to STO at all stages |
| Fitness function | Engineering decision | Research design | 1 - macro_F1 on validation set |
| Baseline model | Experimental control | Scientific requirement | 3-layer CNN, ~500K params, default HPs |
| Grad-CAM target | Adapted from Selvaraju et al. 2017 | Grad-CAM paper | Final decoder block output |
| Calibration method | Adapted from Guo et al. 2017 | Calibration paper | Temperature scaling on validation data |
| Robustness tests | Engineering decision | Hendrycks & Dietterich 2019 protocol | 7 perturbation types, multiple severities |
| Morphological preprocessing | Engineering decision (ablation) | No source requirement | ABL-04; NOT in core pipeline |
| Number of stacks | Engineering decision | Computational constraint + ablation planned | 2 (dev), 3 (final) |
| Base channels | Engineering decision | Lightweight objective | 32 (dev), 64 (final) |

---

## Open Questions (To Be Resolved in M1)

| ID | Question | Resolution Path |
|----|----------|----------------|
| OQ-01 | Exact STO C_f formulation: does range go 2->1 or 2->0? | Read Dhiman & Kumar (2019) primary paper |
| OQ-02 | Exact spiral update equation from primary paper | Read Dhiman & Kumar (2019) primary paper |
| OQ-03 | Source smart-farming paper exact title, authors, DOI | Literature search during M1 |
| OQ-04 | PlantVillage exact per-class image counts | Download dataset during M1; count files |
| OQ-05 | PlantVillage license (CC0 vs. other) | Inspect dataset README/LICENSE file |
| OQ-06 | Source paper's exact LS-HGNet architecture details | Read source paper during M1 |
| OQ-07 | Whether 224x224 final config fits 16 GB RAM at batch 32 | Memory profiling during M1 |
| OQ-08 | Whether class-weighted CE loss is needed | Measure class imbalance during M1 |

---

*End of Architecture Decisions Log — M0 Draft*
