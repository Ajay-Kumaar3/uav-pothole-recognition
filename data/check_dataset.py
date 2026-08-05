"""Dataset Integrity Checker for UAV Pothole Recognition.

This script verifies dataset quality by checking for:
- Missing labels (images without annotation files)
- Missing images (annotation files without corresponding images)
- Empty or malformed annotation files
- Corrupted images (unable to open or read metadata)
- Duplicate filenames
- Class distribution and imbalance alerts
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Set up path to allow importing dataset_loader
sys.path.append(str(Path(__file__).resolve().parent))
from dataset_loader import DatasetInspector, tqdm, HAS_PIL, Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("check_dataset")


class DatasetChecker:
    """Verifies dataset files for potential data quality issues."""

    def __init__(self, inspector: DatasetInspector) -> None:
        """Initialize the checker.
        
        Args:
            inspector: An initialized DatasetInspector instance.
        """
        self.inspector = inspector
        self.dataset_root = inspector.dataset_root
        self.image_paths = inspector.image_paths
        self.label_paths = inspector.label_paths
        self.classes = inspector.classes

        # Setup sets for faster lookup
        self.image_stems = {img.stem: img for img in self.image_paths}
        self.label_stems = {lbl.stem: lbl for lbl in self.label_paths}

    def check_missing_labels(self) -> List[Path]:
        """Find images that do not have a corresponding label file.
        
        Returns:
            List of image paths with missing labels.
        """
        missing = []
        for stem, path in self.image_stems.items():
            if stem not in self.label_stems:
                missing.append(path)
        return missing

    def check_missing_images(self) -> List[Path]:
        """Find label files that do not have a corresponding image file.
        
        Returns:
            List of label paths with missing images.
        """
        missing = []
        for stem, path in self.label_stems.items():
            # Skip classes.txt since it is a metadata file, not an annotation
            if stem == "classes":
                continue
            if stem not in self.image_stems:
                missing.append(path)
        return missing

    def check_empty_and_malformed_annotations(self) -> Tuple[List[Path], List[Tuple[Path, str]]]:
        """Check labels for empty contents or malformed YOLO formats.
        
        Returns:
            A tuple of (empty_label_paths, malformed_label_records)
        """
        empty_labels: List[Path] = []
        malformed_labels: List[Tuple[Path, str]] = []
        
        for lbl_path in tqdm(self.label_paths, desc="Checking annotations"):
            if lbl_path.name == "classes.txt":
                continue
                
            try:
                if lbl_path.stat().st_size == 0:
                    empty_labels.append(lbl_path)
                    continue

                with open(lbl_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                if not lines or all(not line.strip() for line in lines):
                    empty_labels.append(lbl_path)
                    continue

                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        malformed_labels.append(
                            (lbl_path, f"Line {i}: Expected 5 elements, got {len(parts)} ('{line}')")
                        )
                        continue
                    
                    # Validate types
                    try:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                    except ValueError:
                        malformed_labels.append(
                            (lbl_path, f"Line {i}: Non-numeric bounding box values ('{line}')")
                        )
                        continue
                    
                    # Validate coords are in bounds [0, 1]
                    if any(coord < 0.0 or coord > 1.0 for coord in coords):
                        malformed_labels.append(
                            (lbl_path, f"Line {i}: Normalized coordinates outside [0, 1] ('{line}')")
                        )
                        continue
                    
                    # Validate class_id matches classes.txt if classes are known
                    if self.classes and (class_id < 0 or class_id >= len(self.classes)):
                        malformed_labels.append(
                            (lbl_path, f"Line {i}: Class ID {class_id} out of range [0, {len(self.classes)-1}]")
                        )

            except Exception as e:
                malformed_labels.append((lbl_path, f"Read error: {e}"))
                
        return empty_labels, malformed_labels

    def check_corrupted_images(self) -> List[Path]:
        """Try opening each image file to verify it is not corrupted.
        
        Returns:
            List of corrupted image paths.
        """
        corrupted = []
        if not HAS_PIL:
            logger.warning("PIL not installed; skipping corruption check.")
            return corrupted

        for img_path in tqdm(self.image_paths, desc="Checking image corruption"):
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Verify image headers
                
                # Double-check by actually loading pixels of a tiny sample, or just opening
                with Image.open(img_path) as img:
                    img.load()
            except Exception as e:
                corrupted.append(img_path)
                logger.warning(f"Corrupted image detected: {img_path.name}. Error: {e}")
                
        return corrupted

    def check_duplicates(self) -> Dict[str, List[Path]]:
        """Identify duplicate image or label filenames across the directory tree.
        
        Returns:
            Dict mapping duplicate base names to their multiple path locations.
        """
        seen: Dict[str, List[Path]] = {}
        duplicates: Dict[str, List[Path]] = {}
        
        # Traverse dataset directory
        for p in self.dataset_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (self.inspector.IMAGE_EXTENSIONS | {".txt"}):
                if p.name == "classes.txt" or "preprocessing" in p.parts:
                    continue
                seen.setdefault(p.name, []).append(p)
                
        for name, paths in seen.items():
            if len(paths) > 1:
                duplicates[name] = paths
                
        return duplicates

    def check_class_imbalance(self) -> Tuple[Dict[int, int], List[str]]:
        """Count standard annotation counts per class and identify severe imbalances.
        
        Returns:
            Tuple of (class_counts, alerts)
        """
        counts: Dict[int, int] = {i: 0 for i in range(len(self.classes))} if self.classes else {}
        alerts: List[str] = []
        
        for lbl_path in self.label_paths:
            if lbl_path.name == "classes.txt":
                continue
            try:
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if parts:
                            try:
                                cid = int(parts[0])
                                counts[cid] = counts.get(cid, 0) + 1
                            except ValueError:
                                pass
            except OSError:
                pass

        if not counts:
            return counts, ["No class distributions could be extracted."]

        total_annotations = sum(counts.values())
        if total_annotations == 0:
            return counts, ["No annotations found."]

        for cid, count in counts.items():
            class_name = self.classes[cid] if cid < len(self.classes) else f"Class_{cid}"
            percentage = (count / total_annotations) * 100
            
            # Alert if class has less than 15% of the data in binary/multi-class setup
            if percentage < 15.0 and len(counts) > 1:
                alerts.append(
                    f"Class '{class_name}' (ID {cid}) is heavily underrepresented: "
                    f"only {count} boxes ({percentage:.2f}%)"
                )
                
        return counts, alerts

    def run_all_checks(self) -> None:
        """Run all data checks and print a detailed report."""
        print("\n" + "=" * 50)
        print("               DATASET INTEGRITY REPORT")
        print("=" * 50)

        # Missing files checks
        missing_lbls = self.check_missing_labels()
        missing_imgs = self.check_missing_images()
        print(f"Missing Labels: {len(missing_lbls)}")
        if missing_lbls:
            for item in missing_lbls[:10]:
                print(f"  - Image '{item.name}' lacks annotation file.")
            if len(missing_lbls) > 10:
                print(f"  - ... and {len(missing_lbls) - 10} more.")

        print(f"Missing Images: {len(missing_imgs)}")
        if missing_imgs:
            for item in missing_imgs[:10]:
                print(f"  - Label '{item.name}' lacks image file.")
            if len(missing_imgs) > 10:
                print(f"  - ... and {len(missing_imgs) - 10} more.")

        # Empty/Malformed label checks
        empty_lbls, malformed_lbls = self.check_empty_and_malformed_annotations()
        print(f"Empty Annotation Files: {len(empty_lbls)}")
        if empty_lbls:
            for item in empty_lbls[:10]:
                print(f"  - Empty file: {item.name}")
            if len(empty_lbls) > 10:
                print(f"  - ... and {len(empty_lbls) - 10} more.")

        print(f"Malformed Annotation Lines: {len(malformed_lbls)}")
        if malformed_lbls:
            for path, err in malformed_lbls[:10]:
                print(f"  - Malformed box in {path.name}: {err}")
            if len(malformed_lbls) > 10:
                print(f"  - ... and {len(malformed_lbls) - 10} more.")

        # Corrupted images checks
        corrupted_imgs = self.check_corrupted_images()
        print(f"Corrupted Images: {len(corrupted_imgs)}")
        if corrupted_imgs:
            for item in corrupted_imgs[:10]:
                print(f"  - Corrupt image: {item.name}")
            if len(corrupted_imgs) > 10:
                print(f"  - ... and {len(corrupted_imgs) - 10} more.")

        # Duplicate checks
        duplicates = self.check_duplicates()
        print(f"Duplicate Filenames: {len(duplicates)}")
        if duplicates:
            for name, paths in list(duplicates.items())[:10]:
                print(f"  - Duplicate base file: '{name}' in multiple locations:")
                for p in paths:
                    print(f"      * {p.relative_to(self.inspector.root_dir)}")
            if len(duplicates) > 10:
                print(f"  - ... and {len(duplicates) - 10} more duplicates.")

        # Class imbalance checks
        class_counts, imbalance_alerts = self.check_class_imbalance()
        print("\nClass Distribution:")
        total_boxes = sum(class_counts.values())
        for cid, count in class_counts.items():
            cname = self.classes[cid] if cid < len(self.classes) else f"Class_{cid}"
            pct = (count / total_boxes * 100) if total_boxes > 0 else 0
            print(f"  - Class {cid} ({cname}): {count} bounding boxes ({pct:.2f}%)")

        print(f"\nClass Imbalance Alerts: {len(imbalance_alerts)}")
        for alert in imbalance_alerts:
            print(f"  - WARNING: {alert}")

        print("=" * 50 + "\n")


def main() -> None:
    """Main execution block."""
    inspector = DatasetInspector()
    if not inspector.discover_dataset():
        logger.error("Dataset auto-discovery failed. Cannot check dataset.")
        sys.exit(1)
        
    checker = DatasetChecker(inspector)
    checker.run_all_checks()


if __name__ == "__main__":
    main()
