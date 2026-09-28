"""Preprocessing pipeline for authentic IO-VNBD dataset.

Performs:
- Drive loading and S/V synchronization
- Sliding window feature extraction (IMU -> forward velocity)
- Window-level leakage prevention across drive splits (Rule 31 & 32)
- Generation of reports/window_leakage_report.json
- Export to data/processed/{train,val,test}_data.npz
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import CONFIG, DatasetConfig, REPORTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
from .loader import load_drive_pair, IOVNBDrive

logger = logging.getLogger(__name__)


class IDRWindowDataset(Dataset):
    """PyTorch Dataset yielding IMU sliding windows and supervision targets."""

    def __init__(self, windows: np.ndarray, targets_vel: np.ndarray, targets_imu: np.ndarray):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        self.targets_vel = torch.tensor(targets_vel, dtype=torch.float32)
        self.targets_imu = torch.tensor(targets_imu, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.targets_vel[idx], self.targets_imu[idx]


def create_sliding_windows(
    imu_data: np.ndarray,
    speed_data: np.ndarray,
    timestamps: np.ndarray,
    drive_id: str,
    window_size: int = 50,
    stride: int = 10,
    clean_imu_ref: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """Slice continuous drive time-series into sliding windows strictly within drive boundaries.
    
    Returns:
        X_windows: shape (N, 6, window_size) - channel first for 1D convolutions
        y_vel: shape (N,) - reference forward speed (m/s)
        y_imu: shape (N, 6) - IMU target
        metadata: list of dicts with drive_id, start_time, end_time
    """
    N_samples = len(imu_data)
    if clean_imu_ref is None:
        clean_imu_ref = imu_data

    X_list = []
    y_vel_list = []
    y_imu_list = []
    meta_list = []

    for start in range(0, N_samples - window_size + 1, stride):
        end = start + window_size
        win = imu_data[start:end].T  # shape (6, window_size)
        
        # Target velocity is the velocity at the end of the window (causal supervision)
        target_v = speed_data[end - 1]
        target_imu = clean_imu_ref[end - 1]

        X_list.append(win)
        y_vel_list.append(target_v)
        y_imu_list.append(target_imu)
        meta_list.append({
            "drive_id": drive_id,
            "start_idx": start,
            "end_idx": end,
            "start_time": float(timestamps[start]),
            "end_time": float(timestamps[end - 1]),
        })

    if not X_list:
        return np.empty((0, 6, window_size)), np.empty((0,)), np.empty((0, 6)), []

    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_vel_list, dtype=np.float32),
        np.array(y_imu_list, dtype=np.float32),
        meta_list,
    )


def preprocess_dataset(
    raw_dir: Path = None,
    output_dir: Path = None,
    config: DatasetConfig = None,
) -> Dict[str, Path]:
    """Preprocess authentic IO-VNBD drives and export train/val/test NPZ files."""
    if config is None:
        config = CONFIG["dataset"]
    if raw_dir is None:
        raw_dir = RAW_DATA_DIR / "iovnbd"
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": list(config.train_drives),
        "val": list(config.val_drives),
        "test": list(config.test_drives),
    }

    # Rule 31: Enforce zero split contamination
    assert set(splits["train"]).isdisjoint(splits["test"]), "Test drive in train set!"
    assert set(splits["val"]).isdisjoint(splits["test"]), "Test drive in val set!"
    assert set(splits["train"]).isdisjoint(splits["val"]), "Train and val drives overlap!"

    saved_paths = {}
    split_metadata = {"train": [], "val": [], "test": []}

    for split_name, drive_ids in splits.items():
        all_X, all_y_vel, all_y_imu = [], [], []
        logger.info(f"Processing '{split_name}' split with {len(drive_ids)} drives: {drive_ids}")

        for drive_id in drive_ids:
            try:
                drive = load_drive_pair(raw_dir, drive_id)
                phone_imu, gt_gps, v_speed, t = drive.get_synced_data()
                
                X_w, y_v, y_i, metas = create_sliding_windows(
                    phone_imu,
                    v_speed,
                    timestamps=t,
                    drive_id=drive_id,
                    window_size=config.window_size,
                    stride=config.window_stride,
                )

                if len(X_w) > 0:
                    all_X.append(X_w)
                    all_y_vel.append(y_v)
                    all_y_imu.append(y_i)
                    split_metadata[split_name].extend(metas)
                    logger.info(f"  Drive {drive_id}: generated {len(X_w)} windows ({len(phone_imu)} samples).")
            except Exception as e:
                logger.error(f"Error processing drive {drive_id}: {e}")
                raise RuntimeError(f"FATAL: Preprocessing failed on authentic drive {drive_id}: {e}") from e

        if all_X:
            concat_X = np.concatenate(all_X, axis=0)
            concat_y_vel = np.concatenate(all_y_vel, axis=0)
            concat_y_imu = np.concatenate(all_y_imu, axis=0)

            out_file = output_dir / f"{split_name}_data.npz"
            np.savez_compressed(
                out_file,
                windows=concat_X,
                targets_vel=concat_y_vel,
                targets_imu=concat_y_imu,
            )
            saved_paths[split_name] = out_file
            logger.info(f"Saved {split_name} split ({len(concat_X)} windows) to {out_file}")
        else:
            raise RuntimeError(f"FATAL: No data generated for split '{split_name}'.")

    # Rule 32: Window Leakage Test & Verification Report
    train_drives_used = set(m["drive_id"] for m in split_metadata["train"])
    val_drives_used = set(m["drive_id"] for m in split_metadata["val"])
    test_drives_used = set(m["drive_id"] for m in split_metadata["test"])

    train_test_overlap = len(train_drives_used & test_drives_used)
    val_test_overlap = len(val_drives_used & test_drives_used)
    train_val_overlap = len(train_drives_used & val_drives_used)
    total_cross_split_overlaps = train_test_overlap + val_test_overlap + train_val_overlap

    leakage_report = {
        "dataset": "IO-VNBD",
        "window_size": config.window_size,
        "window_stride": config.window_stride,
        "sampling_rate_hz": CONFIG["dataset"].window_size,
        "splits": {
            "train": {
                "drives": sorted(list(train_drives_used)),
                "total_windows": len(split_metadata["train"]),
            },
            "val": {
                "drives": sorted(list(val_drives_used)),
                "total_windows": len(split_metadata["val"]),
            },
            "test": {
                "drives": sorted(list(test_drives_used)),
                "total_windows": len(split_metadata["test"]),
            },
        },
        "leakage_verification": {
            "train_test_drive_overlap": train_test_overlap,
            "val_test_drive_overlap": val_test_overlap,
            "train_val_drive_overlap": train_val_overlap,
            "cross_split_overlaps": total_cross_split_overlaps,
            "window_boundary_leakage": "NONE (windows strictly partitioned per drive)",
            "verdict": "ZERO_LEAKAGE_VERIFIED" if total_cross_split_overlaps == 0 else "CONTAMINATED",
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    leakage_file = REPORTS_DIR / "window_leakage_report.json"
    with open(leakage_file, "w", encoding="utf-8") as f:
        json.dump(leakage_report, f, indent=2)
    logger.info(f"Saved window leakage report to {leakage_file} (cross_split_overlaps={total_cross_split_overlaps})")

    return saved_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    preprocess_dataset()
