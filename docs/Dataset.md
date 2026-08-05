# UAV Pothole Recognition Dataset Details

This document is to be completed by the project partner to analyze the dataset structure and data quality findings.

## 1. Annotation Format
Describe the YOLO text annotation file format.

*   Explain each index of: `<class_id> <x_center> <y_center> <width> <height>`
*   Show how coordinate values are normalized between `0` and `1`.

---

## 2. Dataset Audit Summary
Discuss the dataset findings from running `scripts/run_checks.py`.

*   **Total Images**: 122 (PNG)
*   **Total Labels**: 492 (TXT)
*   **Missing Files Analysis**: Why are there 381 label files without matching images? Why are 11 images missing label files? How does this affect model training?
*   **Corruptions**: Were any corrupted images found?

---

## 3. Class Imbalance and Mitigation
Analyze the distribution of annotations:
*   `agujero` (Pothole): 3,477 boxes (91.40%)
*   `grietas` (Crack): 327 boxes (8.60%)

Explain how this class imbalance might affect training and outline at least three mitigation strategies (e.g. Focal Loss, data augmentation, class weighting).
