# Object Detection and Severity Classification Model Architecture

This document is to be completed by the project partner to detail the deep learning models and post-processing structures.

## 1. Primary Detector: YOLOv11
Describe the YOLOv11 network architecture.

*   **Backbone**: What feature extraction backbone does YOLOv11 use?
*   **Neck**: Explain the feature fusion mechanism (e.g. PANet / BiFPN).
*   **Head**: Explain the decoupled anchor-free detection head.

---

## 2. Baseline Model: YOLOv8
Detail the baseline YOLOv8 structure and key differences compared to YOLOv11.

---

## 3. Severity Classification Module
Detail how potholes will be classified into Minor, Moderate, and Severe categories after localization.

*   **Decision Logic**: Will it be based on the bounding box area in pixels? Or a secondary classifier?
*   **Thresholding**: Outline the proposed mathematical logic or threshold boundaries for classifying severity.
