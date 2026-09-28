"""Global configuration, hyperparameters, seed settings, and constants."""

from pathlib import Path
from dataclasses import dataclass, field
import os
import random
from typing import Tuple, List, Dict
import numpy as np
import yaml

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
OSM_DIR = DATA_DIR / "osm"
CONFIG_DIR = BASE_DIR / "config"

for d in [
    DATA_DIR, RAW_DATA_DIR, INTERMEDIATE_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, FIGURES_DIR, RESULTS_DIR, RESULTS_FIGURES_DIR, OSM_DIR, CONFIG_DIR
]:
    d.mkdir(parents=True, exist_ok=True)

# Hardware & Deterministic Seeds (Rule 33)
SEED = 42
SAMPLING_RATE_HZ = 10.0  # 10 Hz smartphone target
DT = 1.0 / SAMPLING_RATE_HZ


def set_seed(seed: int = SEED):
    """Ensure strict determinism across Python, NumPy, and PyTorch (Rule 33)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_splits_yaml() -> Dict[str, List[str]]:
    """Load canonical drive splits from config/splits.yaml (Rule 30)."""
    splits_file = CONFIG_DIR / "splits.yaml"
    if splits_file.exists():
        with open(splits_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            splits = data.get("splits", {})
            train_ids = splits.get("train", [])
            val_ids = splits.get("val", [])
            test_ids = splits.get("test", [])
            
            # Rule 31: Enforce zero cross-split contamination
            assert set(train_ids).isdisjoint(test_ids), f"Test contamination in train: {set(train_ids) & set(test_ids)}"
            assert set(val_ids).isdisjoint(test_ids), f"Test contamination in val: {set(val_ids) & set(test_ids)}"
            assert set(train_ids).isdisjoint(val_ids), f"Train and val overlap: {set(train_ids) & set(val_ids)}"
            return {"train": train_ids, "val": val_ids, "test": test_ids}
    
    # Fallback default if yaml not found
    return {
        "train": ["M_M", "S1", "S2", "S3a", "S3c", "Vta1a", "Vta2", "Vta16", "Vta29", "Vtb1", "Vtb5", "Vw1", "Vw2", "Vw4"],
        "val": ["Y1"],
        "test": ["Vfa01", "Vfa02"],
    }


_splits = load_splits_yaml()


@dataclass
class DatasetConfig:
    """IO-VNBD Dataset Split Configuration."""
    train_drives: tuple = field(default_factory=lambda: tuple(_splits["train"]))
    val_drives: tuple = field(default_factory=lambda: tuple(_splits["val"]))
    test_drives: tuple = field(default_factory=lambda: tuple(_splits["test"]))
    window_size: int = 50  # 5 seconds at 10 Hz
    window_stride: int = 10  # 1 second stride (80% overlap)


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    # IMU Denoise Net
    denoise_in_channels: int = 6  # ax, ay, az, gx, gy, gz
    denoise_out_channels: int = 6
    denoise_hidden_dim: int = 64
    
    # Velocity Estimator Net
    vel_in_channels: int = 6
    vel_hidden_dim: int = 64
    vel_num_layers: int = 2
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 20
    weight_decay: float = 1e-4


@dataclass
class FilterConfig:
    """EKF / UKF sensor fusion tuning parameters."""
    pos_noise: float = 5.0        # GNSS horizontal position standard deviation (m)
    vel_noise: float = 0.5        # GNSS velocity standard deviation (m/s)
    accel_noise: float = 0.2      # Accelerometer process noise (m/s^2)
    gyro_noise: float = 0.02      # Gyro process noise (rad/s)
    nhc_lat_noise: float = 0.05   # Non-holonomic lateral velocity pseudo-measurement noise (m/s)
    nhc_vert_noise: float = 0.05  # Non-holonomic vertical velocity pseudo-measurement noise (m/s)


CONFIG = {
    "dataset": DatasetConfig(),
    "model": ModelConfig(),
    "filter": FilterConfig(),
}
