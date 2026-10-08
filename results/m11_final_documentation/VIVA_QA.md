# M11 — Comprehensive Technical Viva Voce Preparation Guide

**Scope**: 95 Rigorous Technical Questions and Verified Answers across All Domains  
**Audience**: Final-Year Engineering Students & Research Defense Candidates

---

## Section 1: Core Methodology & Motivation (15 Questions)

#### Q1: What is the main objective of this project?
**Answer**: To adapt, optimize, and rigorously evaluate a Lightweight Stacked Hourglass Network (LS-HGNet) for automated 10-class tomato leaf disease classification, enforcing zero-leakage data isolation and a locked test evaluation.

#### Q2: Why choose tomato crop pathology for this study?
**Answer**: Tomato (*Solanum lycopersicum*) is an economically vital crop vulnerable to over nine distinct foliar diseases causing major yield loss. Early automated detection supports precision agriculture and targeted fungicide intervention.

#### Q3: Why LS-HGNet instead of standard architectures like ResNet-50 or VGG-16?
**Answer**: ResNet-50 (25.6M params) and VGG-16 (138M params) are computationally heavy. LS-HGNet uses 3 recurrent hourglass stacks with multi-dilation residual bottlenecks to achieve multi-scale receptive field coverage with only 8.04M parameters, balancing efficiency and discriminability.

#### Q4: Why not MobileNetV3 or EfficientNet-B0?
**Answer**: While MobileNets are lightweight, their progressive downsampling can discard high-frequency spatial context. LS-HGNet’s hourglass structure repeatedly downsamples and upsamples with skip connections, preserving spatial lesion details across resolutions.

#### Q5: What is the primary source reference for this architecture?
**Answer**: The conceptual integration of LS-HGNet and Sooty Tern Optimization for smart farming originated from Krishnan et al. (2024, *IJERR*, DOI: 10.52756/ijerr.2024.v37spl.008), who studied maize tassel detection. We adapted the backbone to multi-class tomato disease diagnosis.

#### Q6: What is the role of Sooty Tern Optimization (STO)?
**Answer**: STO is a bio-inspired population metaheuristic (Dhiman & Kaur, 2019) modeling collision avoidance and spiral attack behaviors of seabirds. We utilized STO for exploring the hyperparameter search space (learning rate, weight decay, dropout, batch size).

#### Q7: Did STO update the weights of the neural network?
**Answer**: No. STO was used strictly as a hyperparameter search algorithm. All internal convolutional neural network weights were updated via backpropagation using the AdamW optimizer.

#### Q8: What was Candidate A from the M7 selection phase?
**Answer**: Candidate A was the hyperparameter configuration selected from the search phase ($lr = 10^{-3}$, weight decay $= 10^{-4}$, dropout $= 0.3$), which achieved 88.16% Macro-F1 on baseline training and served as the foundation for 10-epoch convergence.

#### Q9: What is the difference between classification accuracy and Macro-F1?
**Answer**: Overall accuracy measures the percentage of total correct predictions across all samples, which can be dominated by majority classes. Macro-F1 computes the harmonic mean of precision and recall independently per class and averages them equally, penalizing poor performance on minority classes.

#### Q10: Why is Macro-F1 the primary metric rather than overall accuracy?
**Answer**: The dataset exhibits natural class support imbalance (from 804 samples in Yellow Leaf Curl Virus down to 56 samples in Mosaic Virus). Macro-F1 ensures that rare disease classes are weighted equally in model evaluation.

#### Q11: What is data leakage, and how was it prevented?
**Answer**: Data leakage occurs when test data informs training or model selection. We prevented leakage by performing stratified partitioning prior to experiments, locking `data/test.csv`, and running test inference strictly once without post-hoc tuning.

#### Q12: What does a 0.17 percentage-point generalization gap signify?
**Answer**: The difference between full validation Macro-F1 (94.23%) and locked-test Macro-F1 (94.06%) was only 0.17 percentage points. This tight concordance confirms that the model learned genuine diagnostic features rather than overfitting the validation set.

#### Q13: Does 95.56% test accuracy prove real-world field readiness?
**Answer**: No. The evaluation was conducted on curated imagery under laboratory/field conditions. Field deployment requires testing under complex canopy clutter, direct sun glare, and multi-disease co-infections.

#### Q14: What was the purpose of post-hoc Temperature Scaling?
**Answer**: Deep networks are often overconfident in their raw softmax probabilities. Temperature Scaling fits a single scalar parameter $T$ on validation logits to align confidence with empirical accuracy without altering class predictions.

#### Q15: What is the learned temperature value from M8.2?
**Answer**: $T = 1.155831$, fitted on the validation calibration subset via L-BFGS, reducing overconfidence and maintaining an Expected Calibration Error of 1.73%.

---

## Section 2: Dataset & Preprocessing (10 Questions)

#### Q16: How many total images are in the dataset?
**Answer**: 18,160 images across 10 tomato leaf classes from the curated PlantVillage dataset.

#### Q17: What are the exact sample counts for each split?
**Answer**: Training: 12,712 images (70.0%); Validation: 2,724 images (15.0%); Locked Test: 2,724 images (15.0%).

#### Q18: Are there any overlapping images between splits?
**Answer**: No. Cryptographic and set intersection checks confirmed exactly 0 duplicate filepaths and 0 overlap across all three splits.

#### Q19: Which class has the largest number of test samples?
**Answer**: `Tomato___Tomato_Yellow_Leaf_Curl_Virus` with 804 test images (29.52% of the test set).

#### Q20: Which class has the smallest number of test samples?
**Answer**: `Tomato___Tomato_mosaic_virus` with 56 test images (2.06% of the test set).

#### Q21: What input resolution was used, and why?
**Answer**: $224 \times 224$ pixels in RGB format. This standard vision resolution preserves fine lesion textures while maintaining fast computation on CPU.

#### Q22: What normalization transform was applied?
**Answer**: Standard ImageNet channel normalization: mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

#### Q23: Were data augmentations applied during test evaluation?
**Answer**: No. Test evaluation used strictly deterministic preprocessing: Resize to $224 \times 224$, convert to tensor, and normalize. No random flips, crops, or test-time augmentations were used.

#### Q24: Were corrupt or missing images detected in the test set?
**Answer**: An automated PIL verification pass confirmed 0 missing files and 0 corrupt/unreadable images among all 2,724 test samples.

#### Q25: How does the healthy class compare to diseased classes?
**Answer**: The test set contains 238 healthy tomato leaf images (8.74% support), providing a strong negative control class.

---

## Section 3: Architecture & Deep Learning Details (15 Questions)

#### Q26: What are the exact structural components of LS-HGNet?
**Answer**: An entry stem ($3 \times 3$ conv $\to$ BN $\to$ ReLU), followed by 3 stacked hourglass modules with intermediate residual connections, an adaptive average pooling layer, a 0.30 dropout layer, and a linear classifier.

#### Q27: How many trainable parameters does LS-HGNet possess?
**Answer**: Exactly 8,041,354 parameters.

#### Q28: What is an hourglass module?
**Answer**: An encoder-decoder architecture that downsamples feature maps via pooling to capture global context, processes features at a bottleneck, and upsamples them while combining them with multi-scale skip connections from the encoder.

#### Q29: What is the purpose of stacking multiple hourglass modules?
**Answer**: Stacking allows repeated bottom-up, top-down spatial inference. Subsequent stacks refine initial feature predictions, enabling the network to re-evaluate lesion context and fine boundaries.

#### Q30: What is a multi-dilation bottleneck block?
**Answer**: A residual block containing parallel $3 \times 3$ convolutional branches with dilation rates $r \in \{1, 2, 4\}$. Their outputs are aggregated, expanding the effective receptive field without spatial downsampling or parameter inflation.

#### Q31: What is dilated (atrous) convolution?
**Answer**: A convolution with spaces between kernel elements. A dilation rate of $r=2$ expands a $3 \times 3$ kernel's receptive field to $5 \times 5$ without increasing the number of weights.

#### Q32: What are the base channels of the network?
**Answer**: $C_{base} = 64$. The classifier projection matrix has shape $[10, 64]$.

#### Q33: How does the network pool spatial features before classification?
**Answer**: Using `AdaptiveAvgPool2d((1, 1))`, which computes the global average over the $H \times W$ feature map, yielding a 64-dimensional feature vector.

#### Q34: What dropout rate was used, and where is it placed?
**Answer**: A dropout rate of 0.30 placed immediately between the flattened global average pooling vector and the final fully connected layer.

#### Q35: What loss function was used during training?
**Answer**: Standard multi-class Cross-Entropy Loss with Softmax.

#### Q36: What optimizer and learning rate schedule were used for full convergence?
**Answer**: AdamW with an initial learning rate of $1.0 \times 10^{-3}$, weight decay of $1.0 \times 10^{-4}$, and a Cosine Annealing schedule decaying to $2.54 \times 10^{-5}$ at Epoch 9.

#### Q37: How many epochs was the full model trained?
**Answer**: 10 epochs on 100% of the training dataset (Milestone M7.3).

#### Q38: Why was Epoch 9 selected rather than Epoch 10?
**Answer**: Validation Macro-F1 peaked at Epoch 9 (94.23%) with a validation loss of 0.1262. At Epoch 10, loss slightly rose to 0.1274, indicating Epoch 9 as the global optimum.

#### Q39: What is the cryptographic SHA-256 hash of the final checkpoint?
**Answer**: `4E4145378F2DE852AB7EABE1E72FE7AE7AFA31F608E0F0E246E06C41A7F098A8`.

#### Q40: What happens if `load_state_dict` is called with `strict=True` on the final checkpoint?
**Answer**: It loads with 0 missing keys and 0 unexpected keys, confirming exact architectural matching.

---

## Section 4: Experimental Results & Analysis (15 Questions)

#### Q41: What was the final overall accuracy on the locked test set?
**Answer**: **95.56%** (2,603 correct classifications out of 2,724 test samples).

#### Q42: What was the final Macro-F1 score on the locked test set?
**Answer**: **94.06%**.

#### Q43: What was the final Weighted-F1 score?
**Answer**: **95.55%**.

#### Q44: What were the Macro Precision and Macro Recall scores?
**Answer**: Macro Precision: **94.00%**; Macro Recall: **94.43%**.

#### Q45: How many total test misclassifications occurred?
**Answer**: 121 errors out of 2,724 images (4.44% error rate).

#### Q46: Which class achieved the highest F1-score?
**Answer**: `Tomato___Tomato_Yellow_Leaf_Curl_Virus` with an F1-score of **99.12%** (Precision: 99.62%, Recall: 98.63%, Support: 804).

#### Q47: How did the healthy leaf class perform?
**Answer**: It achieved **100.0% Precision** and an F1-score of **98.94%** (Recall: 97.90%, Support: 238), with zero false positives.

#### Q48: Which class had the lowest F1-score?
**Answer**: `Tomato___Target_Spot` with an F1-score of **86.78%** (Precision: 91.58%, Recall: 82.46%, Support: 211).

#### Q49: What was the single most frequent confusion pair?
**Answer**: `Tomato___Target_Spot` misclassified as `Tomato___Spider_mites Two-spotted_spider_mite` (29 errors, representing 13.74% of Target Spot test samples).

#### Q50: Why are Target Spot and Spider Mites confused?
**Answer**: Severe spider mite infestation causes chlorotic speckling and yellow stippling. Early target spot lesions manifest as small pinpoint brown specks that visually resemble mite feeding damage.

#### Q51: What other notable confusion pairs occurred?
**Answer**: Bacterial Spot confused with Target Spot (10 errors), and Early Blight confused with Late Blight (9 errors), due to overlapping necrotic lesion margins.

#### Q52: What was the mean prediction confidence for correct classifications?
**Answer**: **96.97%** (median: **99.95%**).

#### Q53: What was the mean prediction confidence for misclassifications?
**Answer**: **70.65%** (median: **69.10%**).

#### Q54: What is the practical value of this confidence separation?
**Answer**: The 26.3 percentage-point difference enables uncertainty thresholding: predictions below 75% confidence can be automatically routed to agricultural extension specialists for human verification.

#### Q55: What was the measured single-image CPU inference latency?
**Answer**: $132.18 \pm 16.71$ ms on Intel x86 CPU (7.57 images/second at batch size 1).

---

## Section 5: Robustness, Limitations & Future Work (15 Questions)

#### Q56: How was model robustness tested in Milestone M8.2?
**Answer**: By evaluating the frozen model across six synthetic image perturbation conditions (clean, brightness $\times 1.3$, contrast $\times 0.75$, Gaussian noise $\sigma=0.05$, Gaussian blur $r=1.5$, JPEG compression $Q=60$) on 1,362 holdout validation samples.

#### Q57: How did the model perform under JPEG compression?
**Answer**: It showed high stability, with Macro-F1 dropping by only 2.62 percentage points (from 95.12% to 92.50%).

#### Q58: How did the model perform under illumination shifts?
**Answer**: It proved resilient, retaining 95.26% Macro-F1 (+0.14%) under reduced contrast and 92.88% (-2.24%) under increased brightness.

#### Q59: What was the most severe robustness vulnerability discovered?
**Answer**: Gaussian blur ($r = 1.5$), which reduced Macro-F1 by 47.82 percentage points down to 47.30%, indicating high sensitivity to optical defocusing.

#### Q60: Why does blur degrade LS-HGNet so significantly?
**Answer**: The multi-dilation residual bottlenecks rely on high-frequency edge gradients to resolve lesion borders. Severe blur smooths out these micro-textures.

#### Q61: What was the effect of Gaussian sensor noise?
**Answer**: Sensor noise ($\sigma = 0.05$) decreased Macro-F1 by 23.72 percentage points down to 71.39%.

#### Q62: What are the main limitations of using the PlantVillage dataset?
**Answer**: Leaf images are primarily captured under controlled lighting against neutral backgrounds. They do not fully capture natural canopy clutter, severe direct sun glare, or co-occurring multi-disease infections.

#### Q63: Can this system be deployed directly onto an aerial drone?
**Answer**: Not without additional adaptations. Real-time drone flight requires a foliage localization stage (e.g., YOLO object detection) and hardware acceleration (e.g., TensorRT on Jetson Nano).

#### Q64: What hardware was used for this project?
**Answer**: Standard laptop CPU execution environment (Python 3.12.10, PyTorch 2.2.2+cpu).

#### Q65: How can the model be accelerated for edge deployment?
**Answer**: Through INT8 post-training quantization, ONNX Runtime optimization, and batching, which can reduce latency to sub-30 ms on edge microprocessors.

#### Q66: What is the benefit of intermediate supervision in stacked hourglasses?
**Answer**: It provides auxiliary gradient signals directly to early stacks, counteracting vanishing gradients and encouraging intermediate feature representations to converge faster.

#### Q67: What does the Expected Calibration Error (ECE) metric represent?
**Answer**: ECE measures the average absolute difference between predicted confidence and actual accuracy across binned probability intervals. A lower ECE denotes a more trustworthy probabilistic output.

#### Q68: What was the ECE of our model?
**Answer**: 1.34% before calibration and 1.73% after Temperature Scaling, both comfortably below the standard 5% calibration threshold.

#### Q69: How would this project fit into a real-world smart farming workflow?
**Answer**: A mobile or tractor-mounted camera captures leaf foliage, the LS-HGNet edge model infers pathology with calibrated confidence, and predictions above 80% generate automated fungicide recommendations via a farmer dashboard.

#### Q70: What is the single most important scientific takeaway from this project?
**Answer**: A lightweight 8.04M-parameter multi-scale hourglass network can deliver high accuracy (95.56%) and robust generalization (0.17 percentage points gap) for crop disease recognition without requiring multi-tens-of-millions parameter models.
