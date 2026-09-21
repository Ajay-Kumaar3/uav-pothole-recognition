"""Dataset Loader and Auto-Inspector for UAV Pothole Recognition.

This script automatically scans the project directory to locate image datasets,
extracts statistics, infers structure (flat or train/val/test splits), and logs reports.
Designed to run modularly and be imported by other validation scripts.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dataset_loader")

# Try importing PIL for image dimensions; define fallback if not available
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None
    logger.warning("PIL (Pillow) is not installed. Image resolutions will not be extracted from headers.")

# Try importing tqdm for progress indicators
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        """Fallback if tqdm is not available."""
        return iterable


class DatasetInspector:
    """Automated inspector for local datasets in the repository."""

    IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    IGNORE_DIRS: Set[str] = {
        ".git", ".venv", "venv", "node_modules", "preprocessing",
        "notebooks", "docs", "models", "results", "__pycache__", "scratch", ".gemini"
    }

    def __init__(self, start_dir: Optional[Path] = None) -> None:
        """Initialize the inspector.
        
        Args:
            start_dir: Path to start scanning from. Defaults to project root (parent of data/).
        """
        if start_dir is None:
            # Assume project root is the parent of the data folder
            self.root_dir = Path(__file__).resolve().parent.parent
        else:
            self.root_dir = Path(start_dir)
        
        self.dataset_root: Optional[Path] = None
        self.image_paths: List[Path] = []
        self.label_paths: List[Path] = []
        self.classes: List[str] = []
        self.structure_type: str = "Unknown"
        self.splits: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}
        self.classes_file: Optional[Path] = None

    def discover_dataset(self) -> bool:
        """Scan the workspace recursively to detect images and labels.
        
        Returns:
            True if a dataset was successfully detected, False otherwise.
        """
        logger.info(f"Scanning directory: {self.root_dir} for datasets...")
        
        # Step 1: Find directories containing images
        image_folders: Dict[Path, List[Path]] = {}
        
        try:
            self._recursive_search(self.root_dir, image_folders)
        except Exception as e:
            logger.error(f"Error during recursive workspace search: {e}", exc_info=True)
            return False

        if not image_folders:
            logger.error("No image folders found in the workspace.")
            return False

        # Step 2: Determine which directory contains the main dataset
        # We select the directory (or the root of directories) with the most images
        best_dir = max(image_folders.keys(), key=lambda k: len(image_folders[k]))
        logger.info(f"Inferred image directory: {best_dir} (contains {len(image_folders[best_dir])} images)")
        
        # Determine dataset root. If the best directory is 'dataset' or 'datasets', use that
        if best_dir.name.lower() in ("dataset", "datasets"):
            self.dataset_root = best_dir
        else:
            # Check parent folder
            parent = best_dir.parent
            if parent.name.lower() in ("dataset", "datasets") or parent != self.root_dir:
                self.dataset_root = parent
            else:
                self.dataset_root = best_dir

        logger.info(f"Inferred dataset root: {self.dataset_root}")

        # Step 3: Analyze structure (splits, flat structure, classes.txt)
        self.image_paths = []
        for path in self.dataset_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.IMAGE_EXTENSIONS:
                # Exclude output images in preprocessing
                if "preprocessing" not in path.parts:
                    self.image_paths.append(path)

        # Look for annotations (.txt files for YOLO)
        self.label_paths = [
            p for p in self.dataset_root.rglob("*.txt")
            if p.is_file() and p.name != "classes.txt" and not p.name.startswith("README") and "preprocessing" not in p.parts
        ]

        # Look for classes.txt
        classes_files = list(self.dataset_root.rglob("classes.txt"))
        if classes_files:
            self.classes_file = classes_files[0]
            try:
                with open(self.classes_file, "r", encoding="utf-8") as f:
                    self.classes = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded classes from {self.classes_file.name}: {self.classes}")
            except Exception as e:
                logger.error(f"Failed to read classes.txt: {e}")
        
        # Check folder split structure
        # Check if train/val/test subdirectories exist under the dataset root
        has_splits = False
        for p in self.dataset_root.iterdir():
            if p.is_dir() and p.name.lower() in ("train", "val", "valid", "validation", "test"):
                has_splits = True
                split_name = "val" if p.name.lower() in ("val", "valid", "validation") else p.name.lower()
                # Find images in this split
                split_imgs = [
                    img for img in self.image_paths if p in img.parents
                ]
                self.splits[split_name] = split_imgs
                logger.info(f"Detected split folder '{p.name}' with {len(split_imgs)} images.")

        if has_splits:
            self.structure_type = "Split (train/val/test subfolders)"
        else:
            self.structure_type = "Flat (all files in single directory)"
            self.splits["train"] = self.image_paths  # Assume all as train for flat

        return len(self.image_paths) > 0

    def _recursive_search(self, current_dir: Path, image_folders: Dict[Path, List[Path]]) -> None:
        """Helper to recursively find folders with images, skipping ignored directories."""
        if current_dir.name in self.IGNORE_DIRS:
            return

        try:
            subdirs: List[Path] = []
            current_imgs: List[Path] = []
            
            for item in current_dir.iterdir():
                if item.is_dir():
                    subdirs.append(item)
                elif item.is_file() and item.suffix.lower() in self.IMAGE_EXTENSIONS:
                    current_imgs.append(item)

            if current_imgs:
                image_folders[current_dir] = current_imgs

            for subdir in subdirs:
                self._recursive_search(subdir, image_folders)
        except PermissionError:
            # Skip folders where we don't have permissions
            pass

    def get_resolutions(self) -> Dict[str, int]:
        """Read image sizes and return resolution distributions.
        
        Returns:
            A dictionary mapping resolution strings to their counts.
        """
        resolutions: Dict[str, int] = {}
        if not HAS_PIL or not self.image_paths:
            return resolutions

        for img_path in tqdm(self.image_paths, desc="Checking image resolutions"):
            try:
                with Image.open(img_path) as img:
                    res_str = f"{img.width}x{img.height}"
                    resolutions[res_str] = resolutions.get(res_str, 0) + 1
            except Exception as e:
                logger.warning(f"Could not open image {img_path.name} to check size: {e}")
        
        return resolutions

    def get_dataset_size_bytes(self) -> int:
        """Calculate the total size of the dataset folder in bytes.
        
        Returns:
            Total size in bytes.
        """
        total_size = 0
        if not self.dataset_root:
            return 0
        
        for p in self.dataset_root.rglob("*"):
            if p.is_file() and "preprocessing" not in p.parts:
                try:
                    total_size += p.stat().st_size
                except OSError:
                    pass
        return total_size

    def print_report(self) -> None:
        """Print a professional dataset status report to the terminal."""
        if not self.dataset_root:
            print("Dataset not discovered. Run discover_dataset() first.")
            return

        size_mb = self.get_dataset_size_bytes() / (1024 * 1024)
        resolutions = self.get_resolutions()
        res_summary = ", ".join([f"{k} ({v} images)" for k, v in resolutions.items()]) if resolutions else "Unknown"

        # Unique image extensions
        extensions = sorted(list({p.suffix.lower() for p in self.image_paths}))
        ext_summary = ", ".join(extensions)

        print("\n" + "=" * 50)
        print("                 DATASET REPORT")
        print("=" * 50)
        print(f"Dataset Name:     {self.dataset_root.name}")
        print(f"Dataset Path:     {self.dataset_root}")
        print(f"Folder Structure: {self.structure_type}")
        print(f"Train Split:      {len(self.splits.get('train', []))} images")
        print(f"Validation Split: {len(self.splits.get('val', []))} images")
        print(f"Test Split:       {len(self.splits.get('test', []))} images")
        print(f"Total Images:     {len(self.image_paths)}")
        print(f"Total Labels:     {len(self.label_paths)}")
        print(f"Image Extensions: {ext_summary}")
        print(f"Classes:          {self.classes if self.classes else 'None (classes.txt missing)'}")
        print(f"Image Resolution: {res_summary}")
        print(f"Dataset Size:     {size_mb:.2f} MB")
        print("=" * 50 + "\n")

    def generate_yolo_yaml(self) -> None:
        """Automatically generate dataset.yaml in the project root if YOLO format is detected."""
        if not self.dataset_root:
            return

        # Check if dataset is YOLO formatted
        if not self.label_paths or not self.classes:
            logger.info("Dataset does not appear to be YOLO formatted (missing labels or classes). Skipping dataset.yaml generation.")
            return

        yaml_path = self.root_dir / "dataset.yaml"
        logger.info(f"Generating YOLO configuration: {yaml_path}")
        
        # Calculate dataset root path relative to project root
        try:
            rel_path = self.dataset_root.relative_to(self.root_dir)
            rel_path_str = f"./{rel_path.as_posix()}"
        except ValueError:
            rel_path_str = self.dataset_root.as_posix()

        yaml_content = f"""# YOLOv11 / YOLOv8 Dataset Configuration
# Generated automatically by dataset_loader

path: {rel_path_str} # dataset root dir
train: .            # train images (relative to path)
val: .              # val images (relative to path)
test:               # test images (optional)

# Class Names
names:
"""
        for idx, cname in enumerate(self.classes):
            yaml_content += f"  {idx}: {cname}\n"

        try:
            with open(yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            logger.info("Successfully generated dataset.yaml")
        except Exception as e:
            logger.error(f"Failed to write dataset.yaml: {e}")


def main() -> None:
    """Main execution block."""
    inspector = DatasetInspector()
    if inspector.discover_dataset():
        inspector.print_report()
        inspector.generate_yolo_yaml()
    else:
        logger.error("Dataset auto-discovery failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

