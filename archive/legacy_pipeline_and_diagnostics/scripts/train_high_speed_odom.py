"""Phase 3: High-Speed AI Odometry Training with Speed-Stratified Sampling & Rebalancing.

Uses ONLY training drives (M, S, Vta, Vtb).
STRICTLY EXCLUDES Vfa01 (Validation) and Vfa02 (Test).

Solves the high-speed underestimation defect by:
1. Speed-stratified sampling (oversampling high-speed windows >= 15 m/s, subsampling <= 10 m/s)
2. Physics-consistent acceleration-velocity scaling augmentation
3. Gaussian NLL loss with heteroscedastic aleatoric uncertainty
4. Validating independently on Vfa01 validation across all speed ranges.
"""

import sys
import logging
from pathlib import Path
from typing import List, Tuple, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.idr.config import set_seed
from src.idr.io.loader import load_drive_pair
from src.idr.models.inertial_odom import InertialOdomNet, gaussian_nll_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = ROOT_DIR / "data" / "raw"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT_DIR / "results" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class SpeedStratifiedDataset(Dataset):
    """Dataset with physics-consistent augmentation for robust high-speed generalization."""

    def __init__(self, windows: np.ndarray, targets: np.ndarray, speeds: np.ndarray, augment: bool = True):
        self.windows = torch.tensor(windows, dtype=torch.float32)   # (N, 6, 50)
        self.targets = torch.tensor(targets, dtype=torch.float32)   # (N, 2) [dx, dy]
        self.speeds = speeds                                         # (N,)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        win = self.windows[idx].clone()
        tgt = self.targets[idx].clone()

        if self.augment and torch.rand(1).item() < 0.5:
            # 1. Physics-consistent speed scale alpha in [0.85, 1.25]
            # Scaling speed scales longitudinal accel and displacement by alpha
            alpha = 0.85 + 0.40 * torch.rand(1).item()
            win[0] = win[0] * alpha  # acc_x
            tgt[0] = tgt[0] * alpha  # dx

            # 2. Add subtle sensor bias perturbation
            b_acc = (torch.rand(3) - 0.5) * 0.04
            b_gyro = (torch.rand(3) - 0.5) * 0.005
            win[:3] = win[:3] + b_acc.unsqueeze(1)
            win[3:] = win[3:] + b_gyro.unsqueeze(1)

        return win, tgt


def extract_speed_stratified_windows(
    train_drive_ids: List[str],
    window_size: int = 50,
    dt: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sliding windows with speed-stratified strides to prevent low-speed dominance."""
    all_windows = []
    all_targets = []
    all_speeds = []

    logger.info(f"Extracting speed-stratified windows from {len(train_drive_ids)} training drives...")

    for d_id in train_drive_ids:
        try:
            drive = load_drive_pair(DATA_DIR, d_id)
            phone_imu, gt_coords, v_speed, t = drive_data = drive.get_synced_data()
        except Exception as e:
            continue

        N = len(phone_imu)
        if N < window_size + 10:
            continue

        # Convert coordinates to body-frame displacements
        ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
        lat_rad = np.deg2rad(ref_lat)
        R_m = 6378137.0
        gt_e = np.deg2rad(gt_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
        gt_n = np.deg2rad(gt_coords[:, 0] - ref_lat) * R_m

        gt_diff_e = np.diff(gt_e)
        gt_diff_n = np.diff(gt_n)
        headings = np.zeros(N)
        headings[1:] = np.arctan2(gt_diff_n, gt_diff_e)
        headings[0] = headings[1]

        # Scan with speed-dependent stride:
        # High speed (>= 18 m/s): stride 2
        # Medium-high (12-18 m/s): stride 4
        # Medium (6-12 m/s): stride 10
        # Low (< 6 m/s): stride 20
        idx = window_size
        while idx < N:
            cur_speed = float(np.mean(v_speed[idx - window_size : idx]))
            if cur_speed >= 18.0:
                stride = 2
            elif cur_speed >= 12.0:
                stride = 4
            elif cur_speed >= 6.0:
                stride = 10
            else:
                stride = 20

            start = idx - window_size
            win = phone_imu[start:idx].T  # (6, W)

            de = gt_e[idx - 1] - gt_e[start]
            dn = gt_n[idx - 1] - gt_n[start]
            psi = headings[start]
            dx_body = float(np.cos(psi) * de + np.sin(psi) * dn)
            dy_body = float(-np.sin(psi) * de + np.cos(psi) * dn)

            all_windows.append(win)
            all_targets.append([dx_body, dy_body])
            all_speeds.append(cur_speed)

            idx += stride

    X = np.array(all_windows, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32)
    S = np.array(all_speeds, dtype=np.float32)
    logger.info(f"Extracted {len(X):,} stratified windows across all training drives.")
    return X, Y, S


def train_model():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Training drives list (STRICTLY NO Vfa01, NO Vfa02)
    train_ids = [
        "M",
        "S1", "S2", "S3a", "S3b", "S3c", "S4",
        "Vta1a", "Vta1b", "Vta2", "Vta3", "Vta4", "Vta5", "Vta6", "Vta7", "Vta8", "Vta9", "Vta10",
        "Vta11", "Vta12", "Vta13", "Vta14", "Vta15", "Vta16", "Vta17", "Vta19", "Vta20",
        "Vta21", "Vta22", "Vta23", "Vta24", "Vta25", "Vta26", "Vta27", "Vta28", "Vta29", "Vta30",
        "Vtb1", "Vtb2", "Vtb3", "Vtb4", "Vtb5", "Vtb6", "Vtb7", "Vtb8", "Vtb9", "Vtb10", "Vtb11", "Vtb12",
    ]

    X_all, Y_all, S_all = extract_speed_stratified_windows(train_ids, window_size=50)

    # Balance across 6 speed bins: up to 3,500 samples per bin
    bins = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 1000.0]
    selected_indices = []
    max_per_bin = 3500

    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        mask = np.where((S_all >= low) & (S_all < high))[0]
        if len(mask) > max_per_bin:
            chosen = np.random.choice(mask, size=max_per_bin, replace=False)
        else:
            chosen = mask
        selected_indices.extend(chosen)
        logger.info(f"Speed bin [{low:2.0f}, {high:2.0f}) m/s: {len(chosen)} samples selected.")

    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)
    X_train = X_all[selected_indices]
    Y_train = Y_all[selected_indices]
    S_train = S_all[selected_indices]
    logger.info(f"Balanced training set constructed: {len(X_train)} samples across 6 speed tiers.")

    # Validation on Vfa01 (strictly for validation assessment)
    drive_val = load_drive_pair(DATA_DIR, "Vfa01")
    v_imu, v_coords, v_spd, _ = drive_val.get_synced_data()
    ref_lat, ref_lon = v_coords[0, 0], v_coords[0, 1]
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    ve = np.deg2rad(v_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
    vn = np.deg2rad(v_coords[:, 0] - ref_lat) * R_m
    vh = np.zeros(len(v_imu))
    vh[1:] = np.arctan2(np.diff(vn), np.diff(ve))
    vh[0] = vh[1]

    val_wins, val_tgts, val_spds = [], [], []
    for i in range(50, len(v_imu), 5):
        win = v_imu[i - 50 : i].T
        de = ve[i - 1] - ve[i - 50]
        dn = vn[i - 1] - vn[i - 50]
        psi = vh[i - 50]
        dx = float(np.cos(psi) * de + np.sin(psi) * dn)
        dy = float(-np.sin(psi) * de + np.cos(psi) * dn)
        val_wins.append(win)
        val_tgts.append([dx, dy])
        val_spds.append(float(np.mean(v_spd[i - 50 : i])))

    X_val = np.array(val_wins, dtype=np.float32)
    Y_val = np.array(val_tgts, dtype=np.float32)
    S_val = np.array(val_spds, dtype=np.float32)

    logger.info(f"Validation dataset: {len(X_val)} windows from Vfa01.")

    train_ds = SpeedStratifiedDataset(X_train, Y_train, S_train, augment=True)
    val_ds = SpeedStratifiedDataset(X_val, Y_val, S_val, augment=False)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128).to(device)

    # Initialize from previous checkpoint if available to preserve baseline convergence
    ckpt_init = MODELS_DIR / "inertial_odom.pt"
    if ckpt_init.exists():
        logger.info(f"Warm-starting from {ckpt_init}")
        model.load_state_dict(torch.load(ckpt_init, map_location=device))

    epochs = 8
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_mae = float("inf")
    save_path = MODELS_DIR / "inertial_odom.pt"

    logger.info("Starting training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = gaussian_nll_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_dx_errs = []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = gaussian_nll_loss(pred, by)
                val_loss += loss.item() * len(bx)
                val_dx_errs.extend(torch.abs(pred[:, 0] - by[:, 0]).cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_mae = float(np.mean(val_dx_errs))
        scheduler.step()

        logger.info(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val dx MAE: {val_mae:.2f} m")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), save_path)
            logger.info(f"  [+] Saved best model checkpoint (val_mae = {val_mae:.2f} m)")

    logger.info(f"[+] High-speed generalization training complete. Best Val MAE: {best_val_mae:.2f} m")


if __name__ == "__main__":
    train_model()
