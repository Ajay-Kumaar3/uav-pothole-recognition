"""YOLOv11 SOTA Training Script for UAV Pothole & Crack Recognition.

This script trains the proposed state-of-the-art YOLOv11 model (yolo11n or yolo11s)
with C3k2 feature extraction, PANet neck, and decoupled anchor-free head.
Supports training on both raw imagery and CLAHE preprocessed datasets for ablation studies.
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("train_yolov11")

def train_yolov11(
    data_yaml: str = "dataset_clahe.yaml",
    weights: str = "yolo11n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 16,
    project_name: str = "runs/detect",
    run_name: str = "yolo11_clahe_sota",
    device: str = "cpu"
):
    """Train YOLOv11 detector and evaluate validation metrics."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics is not installed. Run: pip install ultralytics")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / data_yaml
    if not data_path.exists():
        logger.error(f"Dataset config not found at: {data_path}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("STARTING YOLOV11 SOTA ROAD INSPECTION TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model Architecture: YOLOv11 (Weights: {weights})")
    logger.info(f"Dataset Config:     {data_path}")
    logger.info(f"Epochs:             {epochs}")
    logger.info(f"Batch Size:         {batch_size}")
    logger.info(f"Input Image Size:   {imgsz}x{imgsz}")
    logger.info(f"Compute Device:     {device}")
    logger.info("=" * 60)

    # Initialize YOLOv11 model
    model = YOLO(weights)

    # Train
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=str(project_root / project_name),
        name=run_name,
        exist_ok=True,
        plots=True,
        verbose=True
    )

    logger.info("=" * 60)
    logger.info("YOLOV11 TRAINING COMPLETED")
    logger.info(f"Best model saved at: {project_root / project_name / run_name / 'weights' / 'best.pt'}")
    logger.info("=" * 60)

    # Run validation on best model checkpoint
    logger.info("Running validation on best model checkpoint...")
    val_results = model.val()
    
    # Log key metrics
    metrics = val_results.results_dict
    logger.info("Validation Results Summary:")
    for k, v in metrics.items():
        logger.info(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 detector on UAV road defect dataset.")
    parser.add_argument("--data", type=str, default="dataset_clahe.yaml", help="Path to dataset YAML (default: dataset_clahe.yaml)")
    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Pretrained weights (yolo11n.pt, yolo11s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', '0', etc.)")
    parser.add_argument("--name", type=str, default="yolo11_clahe_sota", help="Run name under runs/detect/")

    args = parser.parse_args()
    train_yolov11(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        run_name=args.name,
        device=args.device
    )

if __name__ == "__main__":
    main()
