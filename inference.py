"""End-to-End UAV Road Inspection and Severity Inference Pipeline.

This script executes the complete pipeline:
1. Loads aerial UAV road images.
2. Applies CIE LAB CLAHE contrast enhancement dynamically.
3. Performs object detection via trained YOLO (v11 or v8).
4. Runs severity classification (Minor / Moderate / Severe).
5. Draws color-coded severity bounding boxes and exports municipal repair reports.
"""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

# Add parent to path for internal imports
sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing.clahe import apply_clahe
from postprocessing.severity_classifier import SeverityClassifier

def run_inference(
    source_path: str,
    weights: str = "yolov8n.pt",
    conf_thresh: float = 0.25,
    apply_enhancement: bool = True,
    output_dir: str = "runs/inference"
):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics is required for inference. Run: pip install ultralytics")
        sys.exit(1)

    project_root = Path(__file__).resolve().parent
    source = Path(source_path)
    out_dir = project_root / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UAV ROAD INSPECTION INFERENCE PIPELINE")
    print("=" * 60)
    print(f"Input Source:       {source}")
    print(f"Model Weights:      {weights}")
    print(f"Confidence Thresh:  {conf_thresh}")
    print(f"CLAHE Enhancement:  {apply_enhancement}")
    print(f"Output Directory:   {out_dir}")
    print("=" * 60)

    # Collect images
    if source.is_dir():
        image_paths = sorted(list(source.glob("*.jpg")) + list(source.glob("*.png")) + list(source.glob("*.jpeg")))
    elif source.is_file():
        image_paths = [source]
    else:
        print(f"Error: Input source '{source}' does not exist.")
        sys.exit(1)

    if not image_paths:
        print(f"No image files found in '{source}'.")
        sys.exit(1)

    # Load YOLO detector
    print(f"Loading object detector: {weights}...")
    model = YOLO(weights)
    classifier = SeverityClassifier()

    total_defects_detected = 0
    all_reports = {}

    for idx, img_path in enumerate(image_paths, 1):
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        h, w = raw_bgr.shape[:2]

        # Step 1: Contrast Enhancement if enabled
        if apply_enhancement:
            processed_bgr = apply_clahe(img_path, clip_limit=2.0, grid_size=8)
        else:
            processed_bgr = raw_bgr

        # Step 2: YOLO Detection
        results = model.predict(processed_bgr, conf=conf_thresh, verbose=False)
        boxes = results[0].boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                confidence = float(b.conf[0].item())
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                
                # Normalized w and h
                box_w = (xyxy[2] - xyxy[0]) / w
                box_h = (xyxy[3] - xyxy[1]) / h

                detections.append({
                    "class_id": cls_id,
                    "bbox": tuple(xyxy),
                    "norm_wh": (box_w, box_h),
                    "conf": confidence
                })

        # Step 3: Severity Classification
        report = classifier.process_predictions(detections)
        report["image_name"] = img_path.name
        all_reports[img_path.name] = report
        total_defects_detected += report["total_defects"]

        # Step 4: Draw Color-Coded Visual Overlays
        annotated_img = classifier.draw_severity_overlays(raw_bgr, report)

        # Step 5: Save Output Image
        out_img_file = out_dir / f"annotated_{img_path.name}"
        cv2.imwrite(str(out_img_file), annotated_img)

        print(f"[{idx}/{len(image_paths)}] {img_path.name}: {report['pothole_count']} Potholes, {report['crack_count']} Cracks | MPS: {report['maintenance_priority_score']}")

    # Export consolidated reports
    summary_data = {
        "total_images_inspected": len(image_paths),
        "total_defects_detected": total_defects_detected,
        "inspections": all_reports
    }
    classifier.export_report_json(summary_data, out_dir / "inspection_summary.json")

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print(f"Annotated Images Saved To: {out_dir}")
    print(f"Full Report Saved To:      {out_dir / 'inspection_summary.json'}")
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end UAV road defect inference and severity classification.")
    parser.add_argument("--source", type=str, default="dataset/test", help="Path to input image or directory (default: dataset/test)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Model weights (yolo11n.pt, yolov8n.pt, or path to best.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE preprocessing")
    parser.add_argument("--output", type=str, default="runs/inference", help="Directory for output visualizations and reports")

    args = parser.parse_args()
    run_inference(
        source_path=args.source,
        weights=args.weights,
        conf_thresh=args.conf,
        apply_enhancement=not args.no_clahe,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()
