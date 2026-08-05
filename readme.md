# AI-Based Pothole Recognition Using Unmanned Aerial Vehicles (UAVs) for Automated Road Inspection

## Project Overview
Road potholes and surface cracks pose significant hazards to vehicle safety, infrastructure longevity, and traffic efficiency. Traditional manual road inspection methods are time-consuming, resource-intensive, and represent safety risks for field surveyors. 

This project implements an automated, artificial intelligence-based road inspection system utilizing aerial imagery captured by Unmanned Aerial Vehicles (UAVs). By combining advanced computer vision architectures (YOLOv11 and baseline YOLOv8) with image enhancement techniques, the system automates pothole detection and surface defect recognition. In addition to localizing potholes and cracks, the framework supports severity classification (Minor, Moderate, Severe) to help road maintenance authorities prioritize repairs.

This repository serves as the professional research codebase for the B.Tech Artificial Intelligence project, prepared to DA1 milestones.

---

## Objectives
1. Develop an automated object detection system using UAV-captured RGB images.
2. Integrate Contrast Limited Adaptive Histogram Equalization (CLAHE) to improve detection robustness under varying shadows, lighting conditions, and wet surfaces.
3. Classify surface distress into distinct categories: Potholes (agujero) and Cracks (grietas).
4. Resize images to standard training dimensions (640x640) while maintaining relative annotation coordinates.
5. Formulate a complete baseline evaluation against YOLOv8.

---

## Methodology
The system architecture follows a modular pipeline:
1. **UAV Image Acquisition**: Capturing high-resolution (4K) aerial imagery from nadir or near-nadir viewpoints.
2. **Dataset Audit and Discovery**: Auto-scanning directory structures to check for missing files, duplicate stems, and format integrity.
3. **Contrast Enhancement (CLAHE)**: Transforming BGR images into the CIE LAB color space, applying adaptive histogram equalization to the L* (luminance) channel to preserve natural colors, and returning the image to BGR space.
4. **Resolution Normalization**: Downscaling high-resolution images to 640x640 to match standard deep learning input boundaries, maintaining annotations directly through normalized coordinates.
5. **YOLO Detection**: Running bounding box regression and class probability extraction.
6. **Maintenance Mapping**: Exporting severity levels and counts.

---

## Repository Structure
```
uav-pothole-recognition/
├── .gitignore                     # Git ignore rules for caches, datasets, and weights
├── requirements.txt               # Minimal list of project dependencies
├── dataset.yaml                   # Auto-generated YOLO configuration file
│
├── data/
│   ├── dataset_loader.py          # Auto-discovery, structural inspection, and YAML generation
│   ├── check_dataset.py           # Verification of missing files, corruption, and imbalances
│   └── dataset_statistics.py      # Statistical distributions of sizes and classes
│
├── preprocessing/
│   ├── clahe.py                   # CIE LAB color-space CLAHE preprocessing utility
│   ├── resize.py                  # Image resizing and annotation mapping utility
│   └── output/                    # Target directory for CLAHE outputs (git ignored)
│
├── scripts/
│   ├── install.sh                 # Linux shell environment install script
│   ├── install.bat                # Windows PowerShell/cmd install batch script
│   └── run_checks.py              # Master script running all data checks sequentially
│
├── notebooks/
│   └── dataset_visualization.ipynb # Teammate task: Visual data inspection notebook
│
├── docs/                          # Teammate task: Pipeline and architecture writeups
│   ├── Pipeline.md
│   ├── Architecture.md
│   └── Dataset.md
│
└── README.md                      # Project documentation and guide
```

---

## Dataset Description
The system automatically discovers and inspects the local dataset structure. A full integrity audit of the `dataset/` directory revealed the following:
* **Format**: YOLOv8/YOLOv11 text annotation format.
* **Image Extension**: `.png`
* **Original Resolution**: 3840x2160 pixels (4K UHD).
* **Classes Defined**: 
  - Class 0: `agujero` (Pothole)
  - Class 1: `grietas` (Crack)
* **Dataset Audit Summary**:
  - Total Images: 122
  - Total Labels: 492 (excluding `classes.txt`)
  - Missing Labels: 11 images have no annotations.
  - Missing Images: 381 label files have no matching image (images not in directory).
  - Fully Matched Pairs: 111 image-label pairs.
  - File Size: Average file size of 16.61 MB, totaling ~2.0 GB.
* **Class Distribution**:
  - `agujero` (ID 0): 3,477 bounding boxes (91.40% of annotations, present in 95 images).
  - `grietas` (ID 1): 327 bounding boxes (8.60% of annotations, present in 49 images).
  - **Class Imbalance Alert**: Class `grietas` is significantly underrepresented. Standard class weights or focal loss should be implemented during training.

---

## Installation

### Prerequisites
* Python 3.10 or Python 3.11

### Windows Installation
Double-click `scripts/install.bat` or run:
```cmd
scripts\install.bat
```

### Linux Installation
Run the following script to set up a virtual environment and dependencies:
```bash
chmod +x scripts/install.sh
./scripts/install.sh
```

---

## Requirements
The project uses a minimal set of dependencies defined in `requirements.txt`:
* `opencv-python` (Image resizing, color-space mapping, and CLAHE filter application)
* `pillow` (Image header auditing and validation)
* `numpy` (Coordinate matrix and numerical math)
* `matplotlib` (Image and bounding box plotting)
* `pyyaml` (YOLO configuration serialization)
* `tqdm` (Progress bar tracking in preprocessing)

---

## Usage

### 1. Verify Dataset and Statistics
Run the master script to verify dataset integrity and output analytical stats:
```bash
python scripts/run_checks.py
```
This runs `dataset_loader.py` (which creates `dataset.yaml`), `dataset_statistics.py`, and `check_dataset.py` sequentially.

### 2. Apply Contrast Enhancement (CLAHE)
Run the preprocessing script to apply luminance-level contrast enhancement:
```bash
python preprocessing/clahe.py
```
*Outputs are saved to `preprocessing/output/`.*

### 3. Normalize Image Resolution (Resize)
Downscale preprocessed images to standard YOLO training sizes:
```bash
python preprocessing/resize.py --input-dir preprocessing/output --output-dir preprocessing/output_resized
```
*Outputs are saved to `preprocessing/output_resized/`.*

---

## Current Progress (DA1)
- [x] Converted workspace into a professional research repository.
- [x] Implemented automatic dataset discovery utilizing relative `pathlib` tracking.
- [x] Completed dataset statistics calculator (resolutions, file sizes, classes).
- [x] Completed dataset checker for corruptions, duplicates, and missing pairs.
- [x] Created `dataset.yaml` generator for YOLO integration.
- [x] Implemented CLAHE contrast enhancement in LAB color space.
- [x] Implemented YOLO-compliant image resizing (640x640).
- [x] Developed master execution check script and multi-platform installers.

---

## Teammate Tasks
To divide project responsibilities, the following assignments are allocated for the project partner to complete DA1 milestones:

### 1. Data Visualization (`notebooks/dataset_visualization.ipynb`)
- Open the Jupyter Notebook and write Python code to load 6 random images from the dataset.
- Draw bounding boxes using coordinates from the matching `.txt` labels.
- Overlay class labels (`agujero` and `grietas`) and draw legend boxes.
- Verify that annotations align perfectly with potholes and cracks.

### 2. Project Documentation (`docs/`)
Write the brief methodologies inside the `docs/` folder:
- **`Pipeline.md`**: Detail the preprocessing pipeline (CIE LAB space conversions, CLAHE parameters, and interpolation parameters during resizing).
- **`Architecture.md`**: Explain YOLOv11 model layers (backbone, neck, and head structures) and contrast it with the YOLOv8 baseline.
- **`Dataset.md`**: Summarize the data structure, bounding box normalization formula, and discuss strategies for addressing the `grietas` class imbalance.

---

## Future Work
- Deploy the YOLOv11 object detector on the enhanced dataset.
- Measure detection speed (FPS) and bounding box regression accuracy.
- Evaluate standard metrics (mAP@50 and mAP@50-95).
- Deploy severity classification thresholds based on bounding box areas.

---

## License
This repository is prepared for academic purposes as part of a B.Tech Artificial Intelligence course project. Source code and documentations are restricted to educational and research uses.

---

## Contributors
* **Ajay Kumaar** (Registration Number: 24BRS1287)
* **Hariaswath** (Registration Number: 24BRS1290)