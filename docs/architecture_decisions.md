# Architecture Decisions Log — AgriAI
## Adaptive Smart Farming Framework using STO-Optimized LS-HGNet

**Document version:** M0-ARCHITECTURE-LOG-LOCKED  
**Date:** 2026-10-01  
**Status:** ✅ M0 LOCKED — Architecture Decisions Approved  

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
Random horizontal flip (p=0.5), random rotation (+/-30 deg, p=0.5), color jitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), random resized crop (scale=0.7 to 1.0). Applied to training data only. **Random vertical flip is deliberately excluded from the default/final pipeline.**

### Source basis
No augmentation policy specified by source papers for tomato classification.

### Our engineering choice
Standard biologically appropriate augmentation set commonly used in plant pathology vision literature. Vertical flipping is excluded by default because upside-down leaf orientations are biologically non-standard for natural photographic capture of foliage. Vertical flip is retained strictly as an optional ablation experiment (ABL-05).

### Reason
- Horizontal flips and rotations increase rotational invariance while maintaining biologically realistic leaf orientations.
- Color jitter accounts for lighting and illumination variations.
- Random resized crop simulates varying camera distances and partial occlusions.

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

## Decision: Project-Modified Stabilized/Additive STOA Variant and Three Experimental Controls

### Decision
AgriAI adopts the **Project-Modified Stabilized/Additive STOA Variant** as the default search algorithm for M1 hyperparameter optimization, while preserving an optional `variant: "original_strict"` toggle for empirical ablation and replication, and an equal-budget `variant: "random_search"` control.

### Source basis
- Dhiman & Kaur (2019), *EAAI*, Vol. 82, pp. 148–174.
- Independent critique & proofs: Kudela (2022, *Swarm Evol. Comput.*) and Kwok (2025, arXiv:2511.17556).

### Our engineering choice
- **Clear Project Modification Notice:** This is a project-specific modification inspired by the original STOA. It is not presented as the original Dhiman & Kaur STOA formulation.
- **Damped and Normalized Spiral:** Replace exponential spiral divergence $r = \exp(k)$ (which produces $\approx 535.49$ at $k=2\pi$) with a project damping factor $b = 0.1$ and a proper normalized 3D direction vector $\vec{\Omega}_3 = [x', y', z']^T / (\|[x', y', z']\|_2 + \epsilon) = [\Omega_1, \Omega_2, \Omega_3]^T$.
- **Project-Specific 3D → D-Dimension Cyclic Extension:** Because the STOA spiral produces a 3D direction vector $\vec{\Omega}_3 \in \mathbb{R}^3$ while the AgriAI hyperparameter search space has $D = 4$ dimensions, a deterministic cyclic extension is applied: $\vec{\Omega}_D[i] = \vec{\Omega}_3[(i-1) \bmod 3 + 1]$ for $i \in \{1, \dots, D\}$. For $D = 4$, this yields $\vec{\Omega}_4 = [\Omega_1, \Omega_2, \Omega_3, \Omega_1]^T \in \mathbb{R}^4$. This ensures $\vec{D}_s \odot \vec{\Omega}_4$ is dimensionally valid in $\mathbb{R}^4$. **This cyclic extension is an explicit project-specific engineering modification, not part of the original Dhiman & Kaur (2019) STOA.**
- **Option A (Consistent Best-Guided Formulation):** Search agents are guided toward the best-so-far solution $\vec{P}_{best}$ throughout the formulation without unintended random-agent drift:
  $$\vec{D}_s(t) = S_A(t) \cdot \vec{P}(t) + 0.5 \cdot \vec{r} \odot (\vec{P}_{best}(t) - \vec{P}(t))$$
- **Additive Update:** Replace multiplicative Hadamard update $\vec{P}(t+1) \propto \vec{D}_s \odot \vec{P}_{best}$ (which induces severe origin zero-bias) with an additive displacement: $\vec{P}(t+1) = \vec{P}_{best}(t) + \vec{D}_s(t) \odot \vec{\Omega}_4$.
- **Three Optimizer Controls:**
  1. `original_strict`: Evaluates the source-derived STOA procedure as faithfully as possible to empirically document its behavior and known numerical properties.
  2. `stabilized`: Evaluates the project-modified stabilized/additive STOA variant (Default).
  3. `random_search`: Provides an equal-budget (25 dev / 100 final) non-metaheuristic search baseline.
- **Domain Clamping:** Clamp all hyperparameter coordinates directly to valid domain bounds $[lb_i, ub_i]$.

### Reason
Literal implementation of original equations leads to numerical overflow, immediate boundary saturation, and severe search degradation toward the coordinate origin. The stabilized additive formulation preserves the conceptual bio-inspired balance of collision avoidance and spiral pursuit while ensuring stable, valid convergence over hyperparameter spaces.

### Alternatives considered
- Literal Dhiman & Kaur equations only: causes numerical explosion and artificial zero-collapse.
- Plain Random Search only: lacks metaheuristic evaluation objective.
- Option B (deliberate $P_{rand}$ drift): rejected in favor of Option A to preserve principled best-guided convergence.

### Trade-offs
- Stabilized Additive STOA departs from literal 2019 equations in favor of numerical stability; this deviation is explicitly acknowledged and documented to maintain scientific integrity.

---

## Decision: LS-HGNet Architectural Adaptation vs. Source Paper

### Decision
Adapt the Lightweight Stacked Hourglass Network (LS-HGNet) conceptual framework from UAV maize tassel detection (Krishnan et al., 2024) to 10-class tomato leaf disease classification on PlantVillage.

### Source basis
- Krishnan et al. (2024), *IJERR*, Vol. 37(spl), pp. 96–108.
- Newell et al. (2016), *ECCV*, arXiv:1603.06937.

### Our engineering choice
- Utilize Depthwise Separable Convolutions (DWSConv) in LightResidualBlocks.
- Implement parallel MultiDilatedBlock bottleneck with rates [1, 2, 4] and 1x1 conv fusion.
- Use Bilinear upsampling + 1x1 conv for decoder stages.
- Replace detection heads with Global Average Pooling -> Dropout -> Linear(10) classifier head.
- Exclude morphological top-hat/bottom-hat transforms from the core pipeline (retained only as ablation ABL-04).

### Reason
- Aerial tassel counting requires dense spatial localization under drone vibration and lighting glare, whereas foliar pathology diagnosis requires robust multi-scale texture classification and post-hoc visual attribution.
- DWSConv drastically minimizes parameter count for edge/laptop execution.
- GAP avoids spatial flattening and eliminates millions of fully connected parameters.

### Alternatives considered
- Full ResNet-50 / ConvNeXt: significantly more parameters; defeats lightweight research objective.
- Literal replication of maize detection network: inappropriate task framing for classification.

### Trade-offs
- Adapting the architecture requires novel design choices (dilation rates, head structure); these are cleanly identified as engineering choices rather than source-prescribed features.

---

## Research Decision Summary Table

| Decision | Status | Evidence | Final Choice |
|----------|--------|----------|-------------|
| Dataset | Literature-verified (counts to verify in M1) | Hughes & Salathé, 2015 | PlantVillage Tomato Subset |
| Classes | 10 classes verified across literature | Multiple published papers | 10 classes (1 healthy + 9 diseased) |
| Input resolution | Engineering decision | Computational constraint | 128x128 (dev), 224x224 (final) |
| LS-HGNet structure | Adapted from Newell et al. & Krishnan et al. | Hourglass & Smart Farming papers | Encoder-Bottleneck-Decoder with skip connections |
| STO equations | Original unverified; project-modified stabilized variant defined | Dhiman & Kaur 2019; Kudela 2022 | Project-Modified Stabilized STOA (default); 3 controls: `original_strict`, `stabilized`, `random_search` |
| Dilation rates | Engineering decision only | No source support | [1, 2, 4] (ablation ABL-01 planned) |
| Inter-stack skip | Engineering decision | Inspired by Newell et al. / DenseNet | Present (ablation ABL-02 planned) |
| Upsampling method | Engineering decision | Newell et al. used nearest-neighbor | Bilinear + 1x1 conv |
| Residual structure | Adapted from He et al. 2016 | ResNet paper | Post-activation DWSConv residual |
| Classifier head | Standard practice | Lin et al. 2014 + PyTorch conventions | GAP + Dropout + Linear, no softmax |
| Optimizer | Engineering decision | Standard DL practice | AdamW |
| Scheduler | Engineering decision | Standard DL practice | CosineAnnealingLR |
| Loss function | Engineering decision | Standard DL practice | CrossEntropyLoss (weighted if needed) |
| Augmentation | Engineering decision | Plant disease classification literature | Standard set (flip, rotate, jitter, crop; NO vertical flip) |
| STO search space | Engineering decision | Experiment design | 4 hyperparameters, ranges as specified |
| STO budget | Engineering decision | Computational constraint | 25 (dev), 100 (final) |
| Random Search budget | Scientific requirement | Equal-budget protocol | Equal to STO at all stages (25 dev / 100 final) |
| Fitness function | Engineering decision | Research design | 1 - macro_F1 on validation set |
| Baseline model | Experimental control | Scientific requirement | 3-layer CNN, ~500K params, default HPs |
| Grad-CAM target | Adapted from Selvaraju et al. 2017 | Grad-CAM paper | Final decoder block output |
| Calibration method | Adapted from Guo et al. 2017 | Calibration paper | Temperature scaling on validation data |
| Robustness tests | Engineering decision | Hendrycks & Dietterich 2019 protocol | 7 perturbation types, multiple severities |
| Morphological preprocessing | Engineering decision (ablation) | No source requirement | ABL-04; NOT in core pipeline |
| Number of stacks | Engineering decision | Computational constraint + ablation planned | 2 (dev), 3 (final) |
| Base channels | Engineering decision | Lightweight objective | 32 (dev), 64 (final) |

---

## Open Questions and Verification Classification

### Resolved Research Specifications

The following theoretical, architectural, and experimental specifications are formally resolved and defined:

| ID | Item | Resolution Summary |
|----|------|-------------------|
| RES-01 | STOA Primary Author Attribution | Confirmed: Gaurav Dhiman and Amandeep Kaur (2019), *EAAI*, 82, 148–174, DOI: 10.1016/j.engappai.2019.03.021 |
| RES-02 | Smart Farming Source Paper Metadata | Confirmed: Krishnan, V. G., Vikranth, B., Sumithra, M., Laxmi, B. P., & Gowri, B. S. (2024). *IJERR*, 37(spl), 96–108. DOI: 10.52756/ijerr.2024.v37spl.008 |
| RES-03 | Scope & Originality Boundary | Confirmed: Source paper performs UAV maize tassel detection; our project is an independent adaptation for tomato leaf disease classification |
| RES-04 | Original vs. Stabilized STOA Distinction | Original source-derived formulation and project-modified stabilized variant are formally distinguished in documentation and configs |
| RES-05 | Stabilized STOA Variant Design | Project-specific modifications formalized: normalized perturbation $\vec{\Omega}$, damped spiral $b=0.1$, additive update, and consistent Option A best-guided formulation ($\vec{P}_{best}$ throughout) |
| RES-06 | Three Optimizer Controls | Explicitly separated into `original_strict`, `stabilized`, and `random_search` with equal-budget protocol (25 dev / 100 final) |
| RES-07 | Vertical Flip Contradiction | Removed random vertical flip from default configuration on biological grounds; documented strictly as optional ablation ABL-05 |
| RES-08 | LS-HGNet Architecture Specification | Hourglass topology, DWSConv block, MultiDilated bottleneck rates [1, 2, 4], and GAP+Linear classification head finalized |
| RES-09 | Original STOA Equation Verification — M0 Formal Resolution | **Formally Accepted (Literature-Verified, Publisher-Unverified).** The documented equations ($S_A$, $\vec{C}_{sa}$, $\vec{M}_s$, $\vec{D}_s$, 3D spiral) are consistently reproduced across multiple independent secondary sources citing Dhiman & Kaur (2019). Direct full-text publisher PDF access was not achieved; this limitation is acknowledged. If discrepancies are found via institutional access in M1, the equations must be updated before publication. |

### Deferred to Implementation Milestones (M1/M2)

The following questions are not architecture or research specification blockers. They are practical questions resolvable only during implementation and are formally deferred:

| ID | Deferred Question | Resolution Milestone |
|----|-------------------|---------------------|
| DEF-01 | PlantVillage exact per-class image counts and verified class distribution | M1: Count files on disk after download |
| DEF-02 | PlantVillage dataset licensing terms (CC0 vs. other) | M1: Inspect dataset README/LICENSE |
| DEF-03 | Hardware memory feasibility: 224×224 at batch size 32 on 16 GB RAM | M1: Memory profiling; reduce to batch 16 if needed |
| DEF-04 | Class-weighted Cross-Entropy loss benefit on macro F1 | M2: Ablation ABL-02 after measuring class imbalance |
| DEF-05 | Final dropout rate selection | M1/M2: Explored within STO search space [0.0, 0.5] |
| DEF-06 | Optional edge deployment ONNX runtime latency | OBJ-11 (optional): Benchmark during deployment phase |

---

*End of Architecture Decisions Log — M0 LOCKED — Architecture Decisions Approved 2026-10-01*
