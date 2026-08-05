"""Dataset Statistics Extractor for UAV Pothole Recognition.

This script calculates and prints dataset statistics including:
- Total image and label counts
- Min, Max, and Average image dimensions (pixels)
- Min, Max, and Average image file sizes (MB)
- Bounding box counts per class
- Image counts per class (images containing each class)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Set

# Set up path to allow importing dataset_loader
sys.path.append(str(Path(__file__).resolve().parent))
from dataset_loader import DatasetInspector, tqdm, HAS_PIL, Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dataset_statistics")


class DatasetStatistics:
    """Calculates granular statistics of the dataset."""

    def __init__(self, inspector: DatasetInspector) -> None:
        """Initialize statistics generator.
        
        Args:
            inspector: An initialized DatasetInspector instance.
        """
        self.inspector = inspector
        self.image_paths = inspector.image_paths
        self.label_paths = inspector.label_paths
        self.classes = inspector.classes

    def calculate_statistics(self) -> None:
        """Analyze all image and label files to generate a detailed report."""
        if not self.image_paths:
            logger.error("No images found to analyze.")
            return

        total_images = len(self.image_paths)
        total_labels = len(self.label_paths)

        # Image sizes (file sizes)
        file_sizes = [p.stat().st_size for p in self.image_paths]
        min_file_size_mb = min(file_sizes) / (1024 * 1024)
        max_file_size_mb = max(file_sizes) / (1024 * 1024)
        avg_file_size_mb = (sum(file_sizes) / len(file_sizes)) / (1024 * 1024)

        # Image dimensions
        widths: List[int] = []
        heights: List[int] = []
        
        if HAS_PIL:
            for img_path in tqdm(self.image_paths, desc="Analyzing image dimensions"):
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        widths.append(w)
                        heights.append(h)
                except Exception as e:
                    logger.warning(f"Error opening image {img_path.name}: {e}")

        # Label/Bounding Box statistics
        class_box_counts: Dict[int, int] = {}
        class_image_counts: Dict[int, Set[str]] = {}  # Set of existing image stems per class
        total_boxes = 0
        image_stems = {img.stem for img in self.image_paths}

        for lbl_path in self.label_paths:
            if lbl_path.name == "classes.txt":
                continue
            
            stem = lbl_path.stem
            is_present = stem in image_stems
            
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                class_box_counts[class_id] = class_box_counts.get(class_id, 0) + 1
                                if is_present:
                                    class_image_counts.setdefault(class_id, set()).add(stem)
                                total_boxes += 1
                            except ValueError:
                                pass
            except OSError:
                pass

        # Print statistics report
        print("\n" + "=" * 50)
        print("               DATASET STATISTICS")
        print("=" * 50)
        print(f"Number of Images:        {total_images}")
        print(f"Number of Labels:        {total_labels}")
        
        # Dimensions summary
        if widths and heights:
            min_w, max_w = min(widths), max(widths)
            min_h, max_h = min(heights), max(heights)
            avg_w = sum(widths) / len(widths)
            avg_h = sum(heights) / len(heights)
            
            print(f"Min Image Dimensions:    {min_w}x{min_h}")
            print(f"Max Image Dimensions:    {max_w}x{max_h}")
            print(f"Average Image Dimensions:{avg_w:.1f}x{avg_h:.1f}")
        else:
            print("Image Dimensions:        Unknown (Pillow not installed or error reading headers)")
            
        print(f"Min Image File Size:     {min_file_size_mb:.2f} MB")
        print(f"Max Image File Size:     {max_file_size_mb:.2f} MB")
        print(f"Average Image File Size: {avg_file_size_mb:.2f} MB")
        
        print("\nClasses Defined in classes.txt:")
        if self.classes:
            for idx, cname in enumerate(self.classes):
                print(f"  - ID {idx}: {cname}")
        else:
            print("  - None")

        print("\nSamples / Bounding Boxes per Class:")
        if class_box_counts:
            for cid in sorted(class_box_counts.keys()):
                cname = self.classes[cid] if self.classes and cid < len(self.classes) else f"Class_{cid}"
                box_count = class_box_counts[cid]
                img_count = len(class_image_counts.get(cid, set()))
                box_pct = (box_count / total_boxes * 100) if total_boxes > 0 else 0
                img_pct = (img_count / total_images * 100) if total_images > 0 else 0
                
                print(f"  - {cname} (ID {cid}):")
                print(f"      * Total Bounding Boxes:  {box_count} ({box_pct:.2f}% of boxes)")
                print(f"      * Images with Class:     {img_count}/{total_images} ({img_pct:.2f}% of images)")
        else:
            print("  - No bounding boxes found in label files.")
            
        print(f"\nTotal Annotations (Bounding Boxes): {total_boxes}")
        print("=" * 50 + "\n")


def main() -> None:
    """Main execution block."""
    inspector = DatasetInspector()
    if not inspector.discover_dataset():
        logger.error("Dataset auto-discovery failed.")
        sys.exit(1)
        
    stats = DatasetStatistics(inspector)
    stats.calculate_statistics()


if __name__ == "__main__":
    main()
