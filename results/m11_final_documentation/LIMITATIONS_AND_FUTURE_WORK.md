# M11 — Technical Limitations & Future Work

---

## 1. Verified Technical Limitations

1. **Curated Laboratory Imagery**:
   - The primary dataset consists of curated PlantVillage imagery. Leaves are predominantly centered against neutral or controlled backgrounds. Natural field conditions (complex soil/foliage backgrounds, variable canopy shadows, overlapping leaves, and direct sunlight glare) were not comprehensively evaluated.
2. **Frequency & Blur Sensitivity (M8.2 Findings)**:
   - Systematic robustness stress-testing revealed that Gaussian blur ($r = 1.5$) caused a 47.82 percentage-point drop in Macro-F1 (falling from 95.12% to 47.30%), and Gaussian noise caused a 23.72 percentage-point drop. The multi-dilation receptive field is sensitive to high-frequency edge loss.
3. **Foliar Pathological Confusion**:
   - Visual overlap between early *Target Spot* necrotic specks and *Two-Spotted Spider Mite* stippling caused 29 misclassifications (13.74% of Target Spot test samples). Furthermore, *Early Blight* and *Late Blight* share necrotic lesion features leading to cross-class confusion.
4. **Single-Image CPU Latency**:
   - Single-image CPU inference averages $132.18 \pm 16.71$ ms (7.57 images/second). While practical for mobile applications, high-throughput aerial video scanning requires GPU acceleration or model quantization (e.g., INT8/ONNX).
5. **Class Imbalance in Natural Support**:
   - The dataset exhibits natural support imbalance, ranging from 804 test images for *Tomato Yellow Leaf Curl Virus* to 56 test images for *Tomato Mosaic Virus*.

---

## 2. Structured Future Enhancements

1. **In-Field Wild Agricultural Dataset Benchmark**:
   - Evaluate model transferability on in-field open-access datasets (e.g., PlantDoc, AI Challenger) to quantify out-of-distribution domain shifts.
2. **Edge Hardware Acceleration & Quantization**:
   - Export LS-HGNet to TensorRT, ONNX Runtime, and INT8 post-training quantization to reduce memory footprint below 30 MB and achieve sub-30 ms latency on embedded boards (e.g., Raspberry Pi 5, NVIDIA Jetson Orin Nano).
3. **Multi-Scale Attention Integration**:
   - Introduce lightweight spatial attention modules (e.g., CBAM or Coordinate Attention) to disambiguate fine-grained textural differences between Target Spot lesions and Spider Mite chlorotic stippling.
4. **Autonomous UAV Flight & Aerial Video Pipeline**:
   - Interface the classification backbone with real-time object detection (e.g., YOLOv8-tiny) for simultaneous plant localization and foliage health assessment from low-altitude drone footage.
