# AI-Based Pothole Recognition Using Unmanned Aerial Vehicles (UAVs) for Automated Road Inspection

## Overview

Road potholes are a major cause of vehicle damage, traffic congestion, and road safety hazards. Conventional inspection methods rely on manual surveys or vehicle-mounted systems, which are time-consuming, labor-intensive, and often limited in coverage.

This project proposes an artificial intelligence-based system for automatic pothole recognition using aerial images captured by Unmanned Aerial Vehicles (UAVs). The system employs a YOLOv11 object detection model combined with image enhancement techniques to improve pothole detection under varying environmental conditions. In addition to detecting potholes, the proposed framework classifies their severity into three levels—Minor, Moderate, and Severe—to support maintenance prioritization.

This repository contains the implementation, datasets, experimental setup, and documentation for the project.

---

## Objectives

The objectives of this project are:

- Develop an automated pothole detection system using UAV-captured RGB images.
- Improve detection robustness through image enhancement using Contrast Limited Adaptive Histogram Equalization (CLAHE).
- Classify detected potholes into Minor, Moderate, and Severe categories.
- Compare the proposed YOLOv11-based framework with a YOLOv8 baseline.
- Evaluate the proposed system using standard object detection metrics.

---

## Proposed Methodology

The proposed system follows the pipeline shown below.

```
UAV Image Acquisition
        │
        ▼
Dataset Preparation
        │
        ▼
Image Preprocessing
(Resize, CLAHE, Normalization, Data Augmentation)
        │
        ▼
YOLOv11 Object Detection
        │
        ▼
Post Processing
(Confidence Threshold and Non-Maximum Suppression)
        │
        ▼
Severity Classification
(Minor, Moderate, Severe)
        │
        ▼
GPS Mapping (Optional)
        │
        ▼
Road Inspection Report
```

---

## Proposed Contributions

The primary contributions of this work include:

- Integration of CLAHE-based image enhancement prior to object detection.
- Implementation of a YOLOv11-based pothole detection framework.
- Severity classification for maintenance-oriented decision support.
- Comparative evaluation with a YOLOv8 baseline.
- Application to UAV-based road inspection using aerial imagery.

---

## Repository Structure

```
AI-UAV-Pothole-Recognition/
│
├── data/
│   ├── dataset_loader.py
│   ├── dataset.yaml
│   └── download_dataset.py
│
├── datasets/
│
├── preprocessing/
│   └── clahe.py
│
├── models/
│   └── yolov11/
│
├── notebooks/
│   └── dataset_visualization.ipynb
│
├── docs/
│   ├── pipeline_diagram.png
│   └── architecture_diagram.png
│
├── results/
│
├── requirements.txt
│
└── README.md
```

---

## Dataset

The project will utilize publicly available road damage datasets, including:

- Road Damage Dataset 2022 (RDD2022)
- China_Drone subset of RD22
- Additional UAV pothole datasets where applicable

All datasets will be converted into YOLO-compatible annotation format for training and evaluation.

---

## Model

### Primary Detector

- YOLOv11

### Baseline

- YOLOv8

### Preprocessing

- Image Resizing
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Normalization
- Data Augmentation

### Post-processing

- Confidence Thresholding
- Non-Maximum Suppression (NMS)

---

## Evaluation Metrics

The proposed framework will be evaluated using the following performance metrics:

- Precision
- Recall
- F1-score
- Mean Average Precision (mAP@50)
- Mean Average Precision (mAP@50:95)
- Inference Speed (Frames Per Second)
- Severity Classification Accuracy

---

## Planned Experiments

The experimental evaluation will include:

- Performance comparison between YOLOv8 and YOLOv11.
- Effect of CLAHE preprocessing on detection accuracy.
- Impact of data augmentation on model generalization.
- Evaluation of the proposed severity classification module.
- Ablation study of the complete detection pipeline.

---

## Development Environment

### Programming Language

- Python 3.10+

### Deep Learning Framework

- PyTorch
- Ultralytics YOLO

### Libraries

- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML

### Development Tools

- Visual Studio Code
- Jupyter Notebook
- Git
- GitHub

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<username>/<repository>.git
cd <repository>
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Status

Current Phase: **Design and Planning (DA1)**

Completed:

- Literature Survey
- Research Gap Identification
- Problem Statement
- Proposed System Architecture
- Experimental Design
- Repository Setup

Upcoming:

- Dataset Preparation
- Model Training
- Performance Evaluation
- Ablation Studies
- Final Report

---

## Team Members

| Name | Registration Number |
|------|----------------------|
| Ajay kumaar | 24BRS1287 |
| Hariaswath  | 24BRS1290 |

---

## License

This repository has been created for academic purposes as part of a Bachelor of Technology (Artificial Intelligence) course project. The source code and documentation are intended solely for educational and research use.