"""Authentic IO-VNBD downloader and extractor.

Authoritative source: onyekpeu/IO-VNBD via Git LFS media endpoints.
"""

import logging
from pathlib import Path
import urllib.request
import zipfile
from tqdm import tqdm

logger = logging.getLogger(__name__)

DATASET_ZIP_URL = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets.zip"
RAW_DIR = Path("data/raw")
ZIP_DEST = RAW_DIR / "Synchronised V abd S datasets.zip"
EXTRACT_DIR = RAW_DIR / "iovnbd"


def download_file(url: str, dest: Path):
    """Download a file with visual progress tracking."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading from {url} to {dest}...")

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f_out:
        total_size = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc=dest.name) as pbar:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
                pbar.update(len(chunk))

    logger.info(f"Downloaded {dest.name} ({dest.stat().st_size} bytes)")


def extract_dataset(zip_path: Path, extract_to: Path):
    """Extract zip archive cleanly into destination."""
    logger.info(f"Extracting {zip_path} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    logger.info(f"Extracted dataset to {extract_to}")


def main():
    if not (EXTRACT_DIR.exists() and len(list(EXTRACT_DIR.rglob("S-*.csv"))) >= 70):
        download_file(DATASET_ZIP_URL, ZIP_DEST)
        extract_dataset(ZIP_DEST, EXTRACT_DIR)
    else:
        logger.info(f"Authentic dataset already present in {EXTRACT_DIR}")


if __name__ == "__main__":
    main()
