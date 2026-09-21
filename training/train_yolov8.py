"""YOLOv8 Baseline Training Script for UAV Pothole Recognition.

This script trains a baseline YOLOv8 model (yolov8n or yolov8s) on the road defect dataset.
Supports CLI arguments for epochs, batch size, image dimensions, and dataset configuration.
Logs training metrics and evaluates on the validation split.
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
logger = logging.getLogger("train_yolov8")

def train_yolov8(
    data_yaml: str = "dataset.yaml",
    weights: str = "yolov8n.pt",
    epochs: int = 25,
    imgsz: int = 640,
    batch_size: int = 16,
    project_name: str = "runs/detect",
    run_name: str = "yolov8_baseline",
    device: str = "cpu"
):
    """Train YOLOv8 baseline detector and evaluate performance."""
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
    logger.info("STARTING YOLOV8 BASELINE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Model Weights:  {weights}")
    logger.info(f"Dataset Config: {data_path}")
    logger.info(f"Epochs:         {epochs}")
    logger.info(f"Batch Size:     {batch_size}")
    logger.info(f"Image Size:     {imgsz}")
    logger.info(f"Compute Device: {device}")
    logger.info("=" * 60)

    # Initialize model
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
    logger.info("YOLOV8 TRAINING COMPLETED")
    logger.info(f"Best model weights saved at: {project_root / project_name / run_name / 'weights' / 'best.pt'}")
    logger.info("=" * 60)

    # Validate
    logger.info("Running validation on best model checkpoint...")
    val_results = model.val()
    
    # Log key metrics
    metrics = val_results.results_dict
    logger.info("Validation Results Summary:")
    for k, v in metrics.items():
        logger.info(f"  - {k}: {v:.4f}" if isinstance(v, float) else f"  - {k}: {v}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Train baseline YOLOv8 on UAV road defect dataset.")
    parser.add_argument("--data", type=str, default="dataset.yaml", help="Path to dataset YAML (relative to project root)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pretrained weights (yolov8n.pt, yolov8s.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs (default: 25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image dimension (default: 640)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', '0', etc.)")
    parser.add_argument("--name", type=str, default="yolov8_baseline", help="Run name under runs/detect/")

    args = parser.parse_args()
    train_yolov8(
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
