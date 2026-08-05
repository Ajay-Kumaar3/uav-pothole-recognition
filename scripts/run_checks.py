"""Master Dataset Verification Script for UAV Pothole Recognition.

This script executes:
1. data/dataset_loader.py (dataset structure and yaml generation)
2. data/dataset_statistics.py (image sizes and class distributions)
3. data/check_dataset.py (dataset integrity audit)

Sequentially, reporting execution success or failure for each check.
"""

import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("run_checks")


def run_script(script_path: Path) -> bool:
    """Run a python script as a subprocess and stream its output.
    
    Args:
        script_path: Path to the python script to run.
        
    Returns:
        True if the script ran successfully (exit code 0), False otherwise.
    """
    logger.info(f"Starting execution of: {script_path.name}")
    print("\n" + "=" * 60)
    print(f" RUNNING: {script_path.name}")
    print("=" * 60)
    
    try:
        # Run subprocess and redirect output to stdout/stderr
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent.parent),
            text=True,
            check=False
        )
        
        print("=" * 60)
        if result.returncode == 0:
            logger.info(f"Successfully completed: {script_path.name}")
            return True
        else:
            logger.error(f"Script {script_path.name} failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"Error executing {script_path.name}: {e}")
        return False


def main() -> None:
    """Main verification sequence."""
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    
    scripts_to_run = [
        data_dir / "dataset_loader.py",
        data_dir / "dataset_statistics.py",
        data_dir / "check_dataset.py"
    ]
    
    logger.info("Starting UAV Pothole Recognition Dataset verification suite...")
    
    success = True
    results = {}
    
    for script in scripts_to_run:
        if not script.exists():
            logger.error(f"Required script not found: {script}")
            results[script.name] = "Not Found"
            success = False
            continue
            
        status = run_script(script)
        results[script.name] = "PASSED" if status else "FAILED"
        if not status:
            success = False
            
    print("\n" + "=" * 40)
    print("             SUMMARY OF CHECKS")
    print("=" * 40)
    for name, res in results.items():
        print(f"  {name:<25}: {res}")
    print("=" * 40)
    
    if success:
        logger.info("All dataset validation checks passed successfully.")
        sys.exit(0)
    else:
        logger.error("Some dataset validation checks failed. Please review outputs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
