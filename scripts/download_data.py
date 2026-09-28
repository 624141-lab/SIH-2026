"""Download and extraction script for authentic IO-VNBD dataset.

Authoritative source: onyekpeu/IO-VNBD
Dataset paper: 'IO-VNBD: Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning'

NON-NEGOTIABLE REQUIREMENT:
The production pipeline must NEVER fall back to synthetic or mock data.
If authentic IO-VNBD cannot be located or downloaded, this script fails loudly.
"""

import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_real_iovnbd import download_file, extract_dataset, DATASET_ZIP_URL, RAW_DIR, ZIP_DEST, EXTRACT_DIR
from scripts.validate_iovnbd import main as run_validation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_iovnbd(output_dir: Path):
    """Download and extract authentic IO-VNBD dataset."""
    logger.info("Checking authentic IO-VNBD dataset...")
    
    # Check if real CSV files already exist
    real_csvs = list(EXTRACT_DIR.rglob("S-*.csv"))
    if len(real_csvs) >= 70:
        logger.info(f"Authentic IO-VNBD dataset already present ({len(real_csvs)} S-files found).")
    else:
        logger.info(f"Downloading authentic IO-VNBD archive from {DATASET_ZIP_URL}...")
        download_file(DATASET_ZIP_URL, ZIP_DEST)
        extract_dataset(ZIP_DEST, EXTRACT_DIR)

    # Validate dataset
    logger.info("Running post-download dataset validation...")
    run_validation()
    logger.info("Authentic IO-VNBD ingestion complete and verified.")


def main():
    try:
        download_iovnbd(RAW_DIR)
    except Exception as e:
        logger.error(
            f"ERROR:\nOriginal IO-VNBD dataset could not be downloaded or verified: {e}\n"
            "Expected authentic dataset in data/raw/iovnbd.\n"
            "No synthetic fallback will be used.\n"
            "Please ensure network access to GitHub Git LFS media or place the authentic dataset into data/raw/iovnbd."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
