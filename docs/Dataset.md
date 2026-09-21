# UAV Pothole Recognition Dataset Details and Integrity Audit

This document provides a comprehensive technical audit of the aerial road defect dataset, analyzing annotation formatting, structural consistency, missing file diagnostics, and class imbalance mitigation strategies.

---

## 1. Annotation Format: YOLO Text Standard

The dataset conforms to the standardized **YOLO object detection format**. Each image in the dataset corresponds to a single `.txt` file containing space-delimited text annotations.

### 1.1 Specification
Each line inside a label file represents exactly one localized bounding box and follows the schema:

$$\langle\text{class\_id}\rangle \quad \langle x_{\text{center}}\rangle \quad \langle y_{\text{center}}\rangle \quad \langle w\rangle \quad \langle h\rangle$$

```
0 0.512345 0.384512 0.084120 0.045610
1 0.721450 0.812300 0.154200 0.021000
```

### 1.2 Coordinate Index Definitions
* **`class_id` $\in \{0, 1\}$:** Integer indicating defect category:
  * `0`: `agujero` (Pothole / Depression)
  * `1`: `grietas` (Crack / Surface fracture)
* **$x_{\text{center}}$ (Normalized Horizontal Center):** Horizontal coordinate of the bounding box centroid divided by total image width:
  $$x_{\text{center}} = \frac{X_{\text{center\_pixel}}}{W_{\text{image}}} \in [0.0, 1.0]$$
* **$y_{\text{center}}$ (Normalized Vertical Center):** Vertical coordinate of the bounding box centroid divided by total image height:
  $$y_{\text{center}} = \frac{Y_{\text{center\_pixel}}}{H_{\text{image}}} \in [0.0, 1.0]$$
* **$w$ (Normalized Bounding Box Width):** Total width of the bounding box divided by image width:
  $$w = \frac{W_{\text{box\_pixel}}}{W_{\text{image}}} \in [0.0, 1.0]$$
* **$h$ (Normalized Bounding Box Height):** Total height of the bounding box divided by image height:
  $$h = \frac{H_{\text{box\_pixel}}}{H_{\text{image}}} \in [0.0, 1.0]$$

**Advantages for Aerial UAV Imagery:**
1. **Resolution Independence:** Bounding boxes remain mathematically valid regardless of whether images are processed at 4K ($3840 \times 2160$) or resized to $640 \times 640$.
2. **Minimal Storage Footprint:** Text files require minimal disk space compared to bulky XML/PASCAL VOC formats.

---

## 2. Dataset Audit & Structural Verification Summary

A rigorous audit conducted via [`data/check_dataset.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/data/check_dataset.py) and [`data/dataset_statistics.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/data/dataset_statistics.py) yielded the following empirical findings:

### 2.1 Summary Metrics
* **Total Image Files Discovered:** 122 images (`.png` format)
* **Total Annotation Files Discovered:** 492 label files (`.txt` format, excluding `classes.txt`)
* **Original Acquisition Resolution:** $3840 \times 2160$ pixels (4K Ultra-High Definition)
* **Mean File Size per Image:** $16.61\text{ MB}$ (totaling $\approx 2.02\text{ GB}$)
* **Fully Matched Image-Label Pairs:** **111 pairs**
* **Corrupt Image Files:** 0 (all 122 image headers successfully validated via PIL without byte corruption)

---

### 2.2 Missing Files & Discrepancy Diagnostics

The automated audit revealed two structural anomalies:

```
Total Discovered Files:
Images: 122 ──────────┐
                       ├──► Fully Matched Pairs: 111
Labels: 492 ──────────┤
                       ├──► Orphan Labels (No Image): 381
                       └──► Unannotated Images: 11
```

#### 1. Orphaned Labels ($n = 381$)
* **Root Cause:** 381 label files exist in the repository without matching `.png` images. This occurs when an aerial flight session's annotation set is exported in bulk, but only a subset of raw 4K frames are transferred due to storage or network transmission constraints.
* **Impact on Model Training:** If an object detection pipeline attempts to load labels without corresponding image matrices, standard training loops (like PyTorch / Ultralytics) will crash with `FileNotFoundError`.
* **Resolution:** Our dataset loader isolates these files into an orphan registry, training exclusively on the 111 fully verified pairs.

#### 2. Unannotated Images ($n = 11$)
* **Root Cause:** 11 image files have no corresponding `.txt` file.
* **Impact on Model Training:** In YOLO frameworks, an image without a label file can be interpreted as either:
  1. A **Negative Sample (Background):** A pristine road section containing zero defects. Negative samples are valuable for training the network to suppress false positive detections on asphalt aggregate or manholes.
  2. An **Incomplete Annotation:** A defective road section that was simply missed by annotators. Including unannotated defect images would severely confuse gradient descent by treating true potholes as background.
* **Resolution:** These 11 images are sequestered until manual inspection verifies whether they represent pristine road surfaces (negative background samples) or unannotated defect frames.

---

## 3. Class Distribution & Imbalance Analysis

### 3.1 Empirical Annotation Counts

Across the 111 fully matched annotation files, a total of **3,804 bounding box instances** were verified:

| Class Index | Class Label | Common Name | Total Bounding Boxes | Percentage of Total | Images Containing Class |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | `agujero` | Pothole / Depression | **3,477** | **91.40%** | 95 images |
| **1** | `grietas` | Crack / Fissure | **327** | **8.60%** | 49 images |
| **Total** | — | — | **3,804** | **100.00%** | 111 images |

```
Class Distribution:
agujero (91.4%) [████████████████████████████████████████████████░] 3,477
grietas (8.6%)  [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]   327
```

---

### 3.2 Impact of Severe Class Imbalance
The 10.6 : 1 ratio between potholes and cracks presents a severe optimization challenge:
1. **Majority Class Domination:** The network's loss function will be dominated by `agujero` gradients. A degenerate model predicting exclusively `agujero` would still achieve $>91\%$ raw classification accuracy while failing completely to detect cracks.
2. **False Negatives for Cracks:** `grietas` instances often have low contrast and narrow pixel footprints. Without intervention, recall for cracks will drop dramatically.

---

### 3.3 Mitigation Strategies

To achieve robust detection across both classes, the following three mitigation techniques are incorporated into the training configuration:

#### 1. Loss Function Adaptation: Focal Loss
Standard Cross-Entropy treats all samples equally. We implement **Focal Loss**, which applies a modulating factor $(1 - p_t)^\gamma$ to down-weight easy examples (majority potholes) and focus training gradients on hard-to-detect minority examples (cracks):

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma = 2.0$ (focusing parameter) and $\alpha_t$ dynamically balances class frequencies.

#### 2. Class-Weighted Loss Computation
In the YOLO classification loss term, inverse frequency weights are assigned to each class:

$$w_c = \frac{N_{\text{total}}}{2 \times N_c}$$

$$w_{\text{agujero}} = \frac{3804}{2 \times 3477} \approx 0.547, \qquad w_{\text{grietas}} = \frac{3804}{2 \times 327} \approx 5.816$$

Every misclassification of a crack penalizes the network $\approx 10.6\times$ more heavily than an equivalent error on a pothole.

#### 3. Targeted Minority-Class Augmentations
During batch generation, images containing `grietas` undergo targeted offline and online data augmentation:
- **Random Mosaic & Mixup:** Blending crack regions with varied road textures.
- **Random Affine Transformations:** Rotation ($\pm 15^\circ$), shear, and horizontal/vertical flips to simulate various UAV flight headings.
- **HSV Color Space Jitter:** Random variations in saturation and value to simulate varying solar illumination without corrupting geometric edge definitions.
