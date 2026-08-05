# UAV Road Inspection Preprocessing Pipeline

This document is to be completed by the project partner to explain the image enhancement and normalization pipeline.

## 1. Contrast Limited Adaptive Histogram Equalization (CLAHE)
Explain how CLAHE works and why it is applied on aerial UAV images.

*   **Color Space Conversion**: Why do we convert from BGR to LAB color space? What is the role of the L* channel?
*   **Clip Limit**: What is the impact of the clip limit (default: `2.0`)? How does it limit noise amplification?
*   **Tile Grid Size**: What is the impact of the tile grid size (default: `8x8`)?

### Processing Results
*(Teammate: Add sample images comparing original vs. CLAHE preprocessed road surfaces.)*

---

## 2. Image Resizing and Normalization
Explain why images are normalized to 640x640 resolution.

*   **Interpolation**: What interpolation method was used (e.g. Bilinear, Bicubic) and why?
*   **YOLO Bounding Box Coordinates**: Explain why bounding box labels do not need coordinate scaling when the image aspect ratio changes during resizing (hint: discuss normalized ratios).

---

## 3. Preprocessing Execution Guide
Provide instructions on how to run `preprocessing/clahe.py` and `preprocessing/resize.py` with different clip limits and sizes.
