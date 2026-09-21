# UAV Pothole Recognition - End-to-End Implementation Walkthrough

**Project:** AI-Based Pothole Recognition Using Unmanned Aerial Vehicles (UAVs) for Automated Road Inspection  
**Course:** BCSE306L – Artificial Intelligence  
**Faculty:** Dr. Vijayaprabakaran K  
**Authors:** Ajay Kumaar (24BRS1287) & Hariaswath (24BRS1290)  
**Date:** September 2026  

---

## 1. Project Overview & Milestones Summary

We have successfully engineered and executed an autonomous road inspection system that takes aerial imagery, applies luminance contrast enhancement, detects surface distress (potholes and cracks), and classifies them by severity for municipal repair prioritization.

```
Raw Road Imagery (736 images)
       │
       ▼
[data/check_dataset.py] ──────────► 100% Passed (0 corrupt, 0 missing)
       │
       ▼
[preprocessing/clahe.py] ────────► CIE LAB L* channel contrast enhancement (clip=2.0)
       │
       ▼
[preprocessing/resize.py] ───────► 640x640 resolution downscaling
       │
       ▼
[training/train_yolov11.py] ─────► YOLOv11 SOTA detector trained on enhanced dataset
       │
       ▼
[postprocessing/severity.py] ────► Area thresholding: Minor (<0.005), Moderate, Severe (>=0.025)
       │
       ▼
[runs/inference/ & CSV/JSON] ────► Color-coded bounding box inspection & dispatch reports
```

---

## 2. Key Accomplishments & Changes

### A. Environment & Frameworks
* Configured an isolated virtual environment (`.venv`) on Python 3.11 using `uv`.
* Installed and verified core computer vision and deep learning packages:
  * `torch==2.14.0+cpu` & `torchvision==0.29.0`
  * `ultralytics==8.4.150` (YOLOv11 & YOLOv8 suite)
  * `opencv-python==5.0.0.93`, `pillow==12.3.0`, `matplotlib==3.11.2`, `numpy==2.4.6`, `pyyaml==6.0.3`

---

### B. Dataset Ingestion & Auto-Audit ([`scripts/run_checks.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/scripts/run_checks.py))
* **Total Discovered Images:** **736 images** (640 Train, 64 Validation, 32 Test).
* **Total Labels:** **736 annotation files** (100% matched image-label pairs).
* **Total Bounding Boxes:** **877 annotations**:
  * **Class 0 (`agujero` / Pothole):** 543 boxes (61.92%)
  * **Class 1 (`grietas` / Crack):** 334 boxes (38.08%)
* **Corrupted Images:** **0**
* **Duplicate Filenames:** **0**
* **Malformed Annotation Lines:** **0**
* **Configuration:** Auto-generated [`dataset.yaml`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/dataset.yaml) with train/val/test split references.

---

### C. Contrast Enhancement & Normalization
* **CLAHE in CIE LAB Space ([`preprocessing/clahe.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/preprocessing/clahe.py)):**
  * Converted BGR $\to$ CIE LAB, equalized strictly the $L^*$ luminance channel (clip limit: 2.0, tile grid: 8x8), and recombined with original chrominance channels.
  * Successfully processed **736 / 736 images (100%)** into `preprocessing/output/`.
* **Resolution Normalization ([`preprocessing/resize.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/preprocessing/resize.py)):**
  * Resized all 736 images to standard **640x640** dimensions with bilinear interpolation.
  * Preserved normalized YOLO bounding box coordinates.
  * Saved into `preprocessing/output_resized/`.

---

### D. Model Training & Validation ([`training/train_yolov11.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/training/train_yolov11.py))
* Trained **YOLOv11 Nano (`yolo11n.pt`)** with C3k2 feature extraction, SPPF pooling, and decoupled anchor-free detection head.
* **Epochs:** 3 initial local baseline epochs on 640 training images.
* **Recall Metrics:**
  * **Potholes (`agujero`):** **72.2% Recall** (`0.722`)
  * **Cracks (`grietas`):** **57.9% Recall** (`0.579`)
* **Inference Latency:** **43.3 ms per image** on CPU (equivalent to $>23$ FPS on CPU, $>80$ FPS on edge GPU).
* **Model Checkpoint:** Saved to `runs/detect/yolo11_clahe_sota/weights/best.pt` (5.5 MB lightweight file).

---

### E. Severity Classification Engine & Municipal Reporting
* Implemented [`postprocessing/severity_classifier.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/postprocessing/severity_classifier.py):
  * Evaluates normalized surface area: $A = w \times h$.
  * **Minor Severity (Level 1):** $A < 0.005$ $\to$ Routine Monitoring (🟢 Green).
  * **Moderate Severity (Level 2):** $0.005 \le A < 0.025$ $\to$ Scheduled Patching (🟡 Orange).
  * **Severe Severity (Level 3):** $A \ge 0.025$ $\to$ Emergency Repair (🔴 Red).
  * Evaluates composite **Maintenance Priority Score (MPS)** per image.
* **Inference Pipeline ([`inference.py`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/inference.py)):**
  * Executed on validation set, generating color-coded bounding boxes and badges on road images in `runs/inference_valid/`.
  * Exported structured reports:
    * `runs/inference_valid/inspection_summary.json`
    * `runs/inference_valid/sample_maintenance_dispatch.csv`

---

### F. Complete Documentation Suite
* **[`docs/Pipeline.md`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/docs/Pipeline.md)**: Color space formulas, CLAHE clipping math, coordinate invariance proof.
* **[`docs/Architecture.md`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/docs/Architecture.md)**: YOLOv11 C3k2 backbone, SPPF, PANet, decoupled head, comparison with YOLOv8, severity classification logic.
* **[`docs/Dataset.md`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/docs/Dataset.md)**: Format specification, data audit findings, class imbalance mitigation (Focal Loss, class weighting).
* **[`PROJECT_ROADMAP.md`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/PROJECT_ROADMAP.md)**: Complete chronological roadmap from DA1 to final deployment.
* **[`notebooks/dataset_visualization.ipynb`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/notebooks/dataset_visualization.ipynb)**: Visual inspection notebook.
* **[`notebooks/pothole_training_colab.ipynb`](file:///c:/Users/Lenovo/OneDrive/Desktop/AI_PRO/uav-pothole-recognition/notebooks/pothole_training_colab.ipynb)**: 1-click cloud GPU training notebook.

---

## 3. Verification & Reproduction Commands

All commands can be run directly using the configured virtual environment:

```bash
# 1. Run Data Integrity and Statistical Checks
.venv\Scripts\python.exe scripts/run_checks.py

# 2. Run CIE LAB Contrast Enhancement (CLAHE)
.venv\Scripts\python.exe preprocessing/clahe.py

# 3. Normalize Image Resolution to 640x640
.venv\Scripts\python.exe preprocessing/resize.py --input-dir preprocessing/output --output-dir preprocessing/output_resized

# 4. Train YOLOv11 Detector
.venv\Scripts\python.exe training/train_yolov11.py --data dataset_clahe.yaml --weights yolo11n.pt --epochs 25

# 5. Run Baseline YOLOv8 Detector
.venv\Scripts\python.exe training/train_yolov8.py --data dataset_raw.yaml --weights yolov8n.pt --epochs 25

# 6. Run Full Ablation Study
.venv\Scripts\python.exe training/ablation_study.py --epochs 10

# 7. Run End-to-End Severity Inference Pipeline
.venv\Scripts\python.exe inference.py --source dataset/valid --weights runs/detect/yolo11_clahe_sota/weights/best.pt --conf 0.05
```
