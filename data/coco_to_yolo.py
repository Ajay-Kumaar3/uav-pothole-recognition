"""Convert Roboflow COCO Segmentation annotations to YOLO format.

This script parses _annotations.coco.json in train, valid, and test folders,
extracts bounding boxes for potholes and cracks, normalizes coordinates,
and writes individual .txt annotation files for every image.
"""

import json
from pathlib import Path

# Mapping COCO category_id to YOLO class_id
# COCO category 2: 'pothole' -> YOLO class 0 ('agujero')
# COCO category 1: 'crack'   -> YOLO class 1 ('grietas')
CAT_MAP = {
    2: 0,  # pothole -> agujero (Class 0)
    1: 1   # crack   -> grietas (Class 1)
}

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

def convert_split(split_dir: Path):
    json_path = split_dir / "_annotations.coco.json"
    if not json_path.exists():
        print(f"Skipping {split_dir.name}: _annotations.coco.json not found.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Index images by id: {id: (file_name, width, height)}
    images = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco["images"]}

    # Group annotations by image_id
    img_annos = {img_id: [] for img_id in images}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        cat_id = ann["category_id"]
        if cat_id not in CAT_MAP:
            continue
        
        yolo_class = CAT_MAP[cat_id]
        bbox = ann["bbox"]  # [x_min, y_min, width, height] in pixels
        img_annos.setdefault(img_id, []).append((yolo_class, bbox))

    # Write .txt file for each image
    count = 0
    total_boxes = 0
    for img_id, (file_name, img_w, img_h) in images.items():
        stem = Path(file_name).stem
        txt_path = split_dir / f"{stem}.txt"
        
        lines = []
        for yolo_cls, (x_min, y_min, w, h) in img_annos.get(img_id, []):
            # Calculate normalized center, width, height
            x_center = (x_min + w / 2.0) / img_w
            y_center = (y_min + h / 2.0) / img_h
            norm_w = w / img_w
            norm_h = h / img_h

            # Clamp coordinates to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            lines.append(f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            total_boxes += 1

        with open(txt_path, "w", encoding="utf-8") as out_f:
            out_f.write("\n".join(lines) + ("\n" if lines else ""))
        count += 1

    print(f"[{split_dir.name.upper()}] Converted {count} images with {total_boxes} bounding boxes.")

def main():
    print(f"Converting COCO annotations in: {DATASET_DIR}")
    for split in ["train", "valid", "test"]:
        split_path = DATASET_DIR / split
        if split_path.is_dir():
            convert_split(split_path)
    print("COCO to YOLO conversion completed successfully!")

if __name__ == "__main__":
    main()
