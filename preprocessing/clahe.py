"""CLAHE Preprocessing for UAV Pothole Recognition.

This script applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
to color images by converting to LAB color space, equalizing the L* channel,
and saving the outputs to the specified preprocessing directory.
It also mirrors the matching YOLO labels alongside the preprocessed images.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Try importing cv2
try:
    import cv2
except ImportError:
    print("Error: OpenCV (opencv-python) is required but not installed.")
    sys.exit(1)

# Set up path to allow importing data modules
sys.path.append(str(Path(__file__).resolve().parent.parent / "data"))
from dataset_loader import DatasetInspector, tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("clahe_preprocessing")


def apply_clahe(img_path: Path, clip_limit: float = 2.0, grid_size: int = 8) -> cv2.Mat:
    """Read an image, convert to LAB space, apply CLAHE to L channel, and convert back.
    
    Args:
        img_path: Path to the image file.
        clip_limit: Threshold for contrast limiting.
        grid_size: Size of grid for histogram equalization (grid_size x grid_size).
        
    Returns:
        The preprocessed image in BGR format.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")

    # Convert from BGR to LAB color space (L: Luminance, A/B: color channels)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    
    # Apply CLAHE to L channel
    l_enhanced = clahe.apply(l_channel)

    # Merge channels back and convert to BGR
    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_bgr


def process_dataset(clip_limit: float = 2.0, grid_size: int = 8, copy_labels: bool = True) -> None:
    """Run CLAHE preprocessing on the automatically discovered dataset."""
    # Find dataset
    inspector = DatasetInspector()
    if not inspector.discover_dataset():
        logger.error("Dataset auto-discovery failed. Cannot preprocess.")
        sys.exit(1)

    project_root = inspector.root_dir
    dataset_root = inspector.dataset_root
    output_root = project_root / "preprocessing" / "output"
    
    logger.info(f"Using clip limit: {clip_limit}, grid size: {grid_size}x{grid_size}")
    logger.info(f"Saving outputs to: {output_root}")

    # Clear/create output directory
    if output_root.exists():
        logger.info(f"Cleaning existing directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # Create mapping of image stems to label paths for faster lookup
    label_map = {lbl.stem: lbl for lbl in inspector.label_paths if lbl.name != "classes.txt"}

    # Process each image
    success_count = 0
    skipped_count = 0

    for img_path in tqdm(inspector.image_paths, desc="Applying CLAHE"):
        try:
            # Maintain subfolder hierarchy relative to dataset root
            rel_path = img_path.relative_to(dataset_root)
            out_img_path = output_root / rel_path
            
            # Ensure subdirectories exist
            out_img_path.parent.mkdir(parents=True, exist_ok=True)

            # Apply CLAHE
            enhanced_img = apply_clahe(img_path, clip_limit, grid_size)
            
            # Save enhanced image
            cv2.imwrite(str(out_img_path), enhanced_img)

            # Copy corresponding labels if requested
            if copy_labels:
                lbl_path = label_map.get(img_path.stem)
                if lbl_path and lbl_path.exists():
                    out_lbl_path = output_root / rel_path.parent / f"{img_path.stem}.txt"
                    shutil.copy2(lbl_path, out_lbl_path)
                    
            success_count += 1
        except Exception as e:
            logger.error(f"Error processing image {img_path.name}: {e}", exc_info=True)
            skipped_count += 1

    # Copy classes.txt if it exists to maintain standard YOLO dataset structure
    if inspector.classes_file and inspector.classes_file.exists():
        try:
            shutil.copy2(inspector.classes_file, output_root / "classes.txt")
        except Exception as e:
            logger.warning(f"Could not copy classes.txt to output folder: {e}")

    logger.info(f"Preprocessing completed. Processed: {success_count}, Skipped/Failed: {skipped_count}")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Apply CLAHE enhancement on the discovered dataset.")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--grid-size", type=int, default=8, help="CLAHE grid size (default: 8)")
    parser.add_argument("--no-labels", action="store_true", help="Do not copy labels to output folder")
    
    args = parser.parse_args()
    process_dataset(clip_limit=args.clip_limit, grid_size=args.grid_size, copy_labels=not args.no_labels)


if __name__ == "__main__":
    main()
