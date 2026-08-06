"""Image Resizing Utility for UAV Pothole Recognition.

This script resizes images in the discovered dataset to 640x640 (standard YOLO size)
and copies/maintains their matching YOLO annotations. Because YOLO annotations
contain normalized bounding box coordinates, they remain valid when the image is resized.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

# Try importing cv2
try:
    import cv2
except ImportError:
    print("Error: OpenCV (opencv-python) is required but not installed.")
    sys.exit(1)

# Set up path to allow importing data modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
# pyrefly: ignore [missing-import]
from dataset_loader import DatasetInspector, tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("resize_preprocessing")


def resize_image(img_path: Path, width: int = 640, height: int = 640) -> cv2.Mat:
    """Read an image and resize it to specified width and height.
    
    Args:
        img_path: Path to the image file.
        width: Target width in pixels.
        height: Target height in pixels.
        
    Returns:
        The resized image.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
        
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_img


def process_resize(
    width: int = 640,
    height: int = 640,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    copy_labels: bool = True
) -> None:
    """Resize dataset images and copy matching annotations."""
    inspector = DatasetInspector()
    if not inspector.discover_dataset():
        logger.error("Dataset auto-discovery failed. Cannot resize.")
        sys.exit(1)

    project_root = inspector.root_dir
    
    # Determine input directory (either custom, or default to auto-detected dataset)
    if input_dir:
        in_path = Path(input_dir)
        logger.info(f"Using custom input directory: {in_path}")
        # Find images in custom directory
        images_set = set()
        for ext in inspector.IMAGE_EXTENSIONS:
            images_set.update(in_path.rglob(f"*{ext}"))
            images_set.update(in_path.rglob(f"*{ext.upper()}"))
        images = sorted(list(images_set))
        
        # Find labels in custom directory
        labels = list(in_path.rglob("*.txt"))
        label_map = {lbl.stem: lbl for lbl in labels if lbl.name != "classes.txt"}
        dataset_origin = in_path
    else:
        images = inspector.image_paths
        label_map = {lbl.stem: lbl for lbl in inspector.label_paths if lbl.name != "classes.txt"}
        dataset_origin = inspector.dataset_root
        in_path = inspector.dataset_root

    if not images:
        logger.error(f"No images found in input path: {in_path}")
        sys.exit(1)

    # Determine output directory
    if output_dir:
        out_path = Path(output_dir)
    else:
        out_path = project_root / "preprocessing" / "output_resized"

    logger.info(f"Resizing {len(images)} images to {width}x{height}...")
    logger.info(f"Saving resized dataset to: {out_path}")

    # Clear/create output directory
    if out_path.exists():
        logger.info(f"Cleaning existing directory: {out_path}")
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0

    for img_path in tqdm(images, desc="Resizing images"):
        try:
            # Maintain subfolder hierarchy relative to input root
            rel_path = img_path.relative_to(dataset_origin)
            out_img_path = out_path / rel_path
            
            # Ensure subdirectories exist
            out_img_path.parent.mkdir(parents=True, exist_ok=True)

            # Resize
            resized = resize_image(img_path, width, height)
            
            # Save resized image
            cv2.imwrite(str(out_img_path), resized)

            # Copy corresponding labels if requested
            if copy_labels:
                lbl_path = label_map.get(img_path.stem)
                if lbl_path and lbl_path.exists():
                    out_lbl_path = out_path / rel_path.parent / f"{img_path.stem}.txt"
                    shutil.copy2(lbl_path, out_lbl_path)
                    
            success_count += 1
        except Exception as e:
            logger.error(f"Error resizing image {img_path.name}: {e}", exc_info=True)
            skipped_count += 1

    # Copy classes.txt if it exists to maintain standard YOLO dataset structure
    classes_file = list(in_path.rglob("classes.txt"))
    if classes_file and classes_file[0].exists():
        try:
            shutil.copy2(classes_file[0], out_path / "classes.txt")
        except Exception as e:
            logger.warning(f"Could not copy classes.txt to output folder: {e}")
    elif inspector.classes_file and inspector.classes_file.exists():
        try:
            shutil.copy2(inspector.classes_file, out_path / "classes.txt")
        except Exception as e:
            logger.warning(f"Could not copy classes.txt to output folder: {e}")

    logger.info(f"Resizing completed. Processed: {success_count}, Skipped/Failed: {skipped_count}")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Resize dataset images to YOLO standard dimensions.")
    parser.add_argument("--width", type=int, default=640, help="Target image width (default: 640)")
    parser.add_argument("--height", type=int, default=640, help="Target image height (default: 640)")
    parser.add_argument("--input-dir", type=str, default=None, help="Custom input directory (defaults to auto-detected dataset)")
    parser.add_argument("--output-dir", type=str, default=None, help="Custom output directory (default: preprocessing/output_resized/)")
    parser.add_argument("--no-labels", action="store_true", help="Do not copy labels to output folder")
    
    args = parser.parse_args()
    process_resize(
        width=args.width,
        height=args.height,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        copy_labels=not args.no_labels
    )


if __name__ == "__main__":
    main()
