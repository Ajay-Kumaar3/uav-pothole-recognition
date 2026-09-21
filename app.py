"""Interactive UAV Pothole & Crack Inspection Web Dashboard.

Provides an interactive web interface for evaluators and municipal inspectors:
- Live sample selection & custom image uploads
- CIE LAB CLAHE luminance enhancement comparison
- YOLOv11 deep learning defect localization
- Severity classification (Minor, Moderate, Severe)
- Maintenance Priority Score (MPS) HUD & report exports
"""

import base64
import os
import sys
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
import cv2
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from preprocessing.clahe import apply_clahe
from postprocessing.severity_classifier import SeverityClassifier
from ultralytics import YOLO

app = Flask(__name__, static_folder="static")

# Load YOLO model
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "yolo11_clahe_sota" / "weights" / "best.pt"
if not MODEL_PATH.exists():
    # Fallback to pretrained nano
    MODEL_PATH = "yolo11n.pt"

print(f"[INFO] Initializing YOLO model from: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
classifier = SeverityClassifier()

def cv2_to_base64(img_bgr: np.ndarray) -> str:
    """Encode OpenCV BGR image to base64 JPEG string."""
    _, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return base64.b64encode(buffer).decode("utf-8")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/samples", methods=["GET"])
def get_samples():
    """Return a curated list of sample test/validation images."""
    sample_dirs = [
        PROJECT_ROOT / "dataset" / "valid",
        PROJECT_ROOT / "dataset" / "test",
        PROJECT_ROOT / "dataset" / "train"
    ]
    
    samples = []
    for sdir in sample_dirs:
        if sdir.exists():
            for p in sorted(list(sdir.glob("*.jpg")) + list(sdir.glob("*.png")))[:12]:
                samples.append({
                    "filename": p.name,
                    "split": sdir.name,
                    "rel_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
                })
        if len(samples) >= 8:
            break

    return jsonify({"samples": samples[:8]})

@app.route("/api/analyze", methods=["POST"])
def analyze_image():
    """Run full inspection: CLAHE + YOLOv11 + Severity Grading."""
    conf_thresh = float(request.form.get("conf", 0.020))
    apply_enh = request.form.get("apply_clahe", "true").lower() == "true"
    
    img_bgr = None
    filename = "uploaded_image.jpg"

    # Check for file upload
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        filename = file.filename
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Check for sample selection
    elif "rel_path" in request.form and request.form.get("rel_path"):
        rel_path = request.form.get("rel_path")
        target_path = PROJECT_ROOT / rel_path
        if target_path.exists():
            filename = target_path.name
            img_bgr = cv2.imread(str(target_path))
    # Check for base64 image (from city survey inspect button)
    elif "image_base64" in request.form and request.form.get("image_base64"):
        b64_str = request.form.get("image_base64")
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(b64_str)
            img_bgr = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            filename = request.form.get("image_name", "inspected_road.jpg")
        except Exception as err:
            print(f"[ERROR] Failed to decode base64 input image: {err}")

    if img_bgr is None:
        return jsonify({"error": "Failed to read or locate input image."}), 400

    orig_h, orig_w = img_bgr.shape[:2]

    # Step 1: CLAHE Enhancement in LAB Space
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    clahe_bgr = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)

    inference_input = clahe_bgr if apply_enh else img_bgr

    # Step 2: YOLO Detection with IoU NMS
    results = model.predict(inference_input, conf=conf_thresh, iou=0.35, imgsz=640, verbose=False)
    boxes = results[0].boxes

    detections = []
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls_id = int(b.cls[0].item())
            confidence = float(b.conf[0].item())
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            
            box_w = (xyxy[2] - xyxy[0]) / orig_w
            box_h = (xyxy[3] - xyxy[1]) / orig_h

            detections.append({
                "class_id": cls_id,
                "bbox": tuple(xyxy),
                "norm_wh": (box_w, box_h),
                "conf": confidence,
                "nx1": xyxy[0] / orig_w,
                "nx2": xyxy[2] / orig_w
            })

    # Step 3: Severity Classification with tree/shadow filter and crater analysis
    report = classifier.process_predictions(detections, image=img_bgr)
    report["image_name"] = filename

    # Step 4: Generate Visual Overlays
    annotated_bgr = classifier.draw_severity_overlays(img_bgr, report)

    # Encode images to base64
    response_data = {
        "image_name": filename,
        "dimensions": f"{orig_w}x{orig_h}",
        "raw_image": f"data:image/jpeg;base64,{cv2_to_base64(img_bgr)}",
        "clahe_image": f"data:image/jpeg;base64,{cv2_to_base64(clahe_bgr)}",
        "annotated_image": f"data:image/jpeg;base64,{cv2_to_base64(annotated_bgr)}",
        "total_defects": report["total_defects"],
        "pothole_count": report["pothole_count"],
        "crack_count": report["crack_count"],
        "severity_breakdown": report["severity_breakdown"],
        "maintenance_priority_score": report["maintenance_priority_score"],
        "defects": report["defects"]
    }

    return jsonify(response_data)

@app.route("/api/batch-survey", methods=["POST"])
def batch_survey():
    """Analyze multiple surveyed roads across a city and rank them from worst to best."""
    conf_thresh = float(request.form.get("conf", 0.020))
    apply_enh = request.form.get("apply_clahe", "true").lower() == "true"
    
    road_items = []
    
    # If multiple files uploaded
    uploaded_files = request.files.getlist("files")
    if uploaded_files and len(uploaded_files) > 0 and uploaded_files[0].filename != "":
        for idx, f in enumerate(uploaded_files, 1):
            bytes_data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(bytes_data, cv2.IMREAD_COLOR)
            if img is not None:
                road_items.append({
                    "name": f.filename if f.filename else f"Road Sector {idx}",
                    "img": img
                })
    else:
        # Load sample city road sectors from valid/test
        valid_dir = PROJECT_ROOT / "dataset" / "valid"
        sample_paths = sorted(list(valid_dir.glob("*.jpg")) + list(valid_dir.glob("*.png")))[:10]
        sector_names = [
            "North Highway Corridor (KM 14.2)",
            "Downtown Market Arterial Road",
            "Industrial Bypass Sector 4",
            "Airport Link Expressway",
            "East Ring Road Flyover Approach",
            "University Campus Main Access",
            "Metro Station Feeder Road",
            "Tech Park Outer Boulevard",
            "Old Harbor Freight Avenue",
            "South Suburban Residential Link"
        ]
        for idx, p in enumerate(sample_paths):
            img = cv2.imread(str(p))
            if img is not None:
                sname = sector_names[idx] if idx < len(sector_names) else f"City Sector {idx+1}"
                road_items.append({
                    "name": sname,
                    "img": img,
                    "rel_path": str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
                })

    ranked_roads = []
    total_city_potholes = 0
    total_city_cracks = 0
    emergency_count = 0
    scheduled_count = 0

    for item in road_items:
        img_bgr = item["img"]
        h, w = img_bgr.shape[:2]
        
        # CLAHE
        if apply_enh:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            clahe_bgr = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
            infer_img = clahe_bgr
        else:
            infer_img = img_bgr
            
        results = model.predict(infer_img, conf=conf_thresh, iou=0.35, imgsz=640, verbose=False)
        boxes = results[0].boxes
        
        detections = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls[0].item())
                confidence = float(b.conf[0].item())
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                detections.append({
                    "class_id": cls_id,
                    "bbox": tuple(xyxy),
                    "norm_wh": ((xyxy[2] - xyxy[0]) / w, (xyxy[3] - xyxy[1]) / h),
                    "conf": confidence,
                    "nx1": xyxy[0] / w,
                    "nx2": xyxy[2] / w
                })
                
        report = classifier.process_predictions(detections, image=img_bgr)
        report["image_name"] = item["name"]
        annotated_bgr = classifier.draw_severity_overlays(img_bgr, report)
        
        mps = report["maintenance_priority_score"]
        potholes = report["pothole_count"]
        cracks = report["crack_count"]
        severe_count = report["severity_breakdown"].get("Severe", 0)
        total_def = report["total_defects"]

        if mps >= 50.0 or potholes >= 3 or severe_count >= 1:
            status = "EMERGENCY DISPATCH"
            urgency_tier = "CRITICAL"
            emergency_count += 1
        elif mps >= 15.0 or total_def >= 1:
            status = "SCHEDULED REPAIR"
            urgency_tier = "MODERATE"
            scheduled_count += 1
        else:
            status = "ROUTINE MONITOR"
            urgency_tier = "LOW"
            
        total_city_potholes += report["pothole_count"]
        total_city_cracks += report["crack_count"]
        
        # Make small thumbnail for table
        thumb = cv2.resize(annotated_bgr, (160, 100))
        
        ranked_roads.append({
            "road_name": item["name"],
            "mps": round(mps, 1),
            "status": status,
            "urgency_tier": urgency_tier,
            "total_defects": report["total_defects"],
            "potholes": report["pothole_count"],
            "cracks": report["crack_count"],
            "severity_breakdown": report["severity_breakdown"],
            "thumbnail": f"data:image/jpeg;base64,{cv2_to_base64(thumb)}",
            "annotated_full": f"data:image/jpeg;base64,{cv2_to_base64(annotated_bgr)}",
            "raw_image": f"data:image/jpeg;base64,{cv2_to_base64(img_bgr)}",
            "clahe_image": f"data:image/jpeg;base64,{cv2_to_base64(clahe_bgr if apply_enh else img_bgr)}",
            "rel_path": item.get("rel_path", ""),
            "defects": report["defects"]
        })
        
    # SORT BY MPS DESCENDING (Highest priority repair at Rank #1!)
    ranked_roads.sort(key=lambda r: r["mps"], reverse=True)
    
    # Assign Ranks 1..N
    for i, r in enumerate(ranked_roads, 1):
        r["rank"] = i
        
    return jsonify({
        "total_roads_surveyed": len(ranked_roads),
        "total_city_potholes": total_city_potholes,
        "total_city_cracks": total_city_cracks,
        "emergency_roads_count": emergency_count,
        "scheduled_roads_count": scheduled_count,
        "ranked_roads": ranked_roads
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)

