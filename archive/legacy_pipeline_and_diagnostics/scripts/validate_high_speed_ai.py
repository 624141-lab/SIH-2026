"""Phase 3C: Independent AI Odometry Validation on Vfa01 (VALIDATION ONLY).

Computes:
1. Displacement MAE & RMSE (meters)
2. Velocity MAE & Bias (m/s)
3. Performance breakdown across speed ranges:
   0-5, 5-10, 10-15, 15-20, 20-25, 25-30+ m/s
4. Dedicated evaluation for high speeds (> 15 m/s)
5. Aleatoric uncertainty calibration (1-sigma, 2-sigma, 3-sigma empirical coverage).
"""

import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from src.idr.config import set_seed
from src.idr.io.loader import load_drive_pair
from src.idr.models.inertial_odom import InertialOdomNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_ai_model():
    set_seed(42)
    logger.info("Starting Phase 3C: Independent AI Validation on Vfa01...")

    model_path = ROOT_DIR / "models" / "inertial_odom.pt"
    model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    data_dir = ROOT_DIR / "data" / "raw"
    drive = load_drive_pair(data_dir, "Vfa01")
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1

    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    gt_e = np.deg2rad(gt_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
    gt_n = np.deg2rad(gt_coords[:, 0] - ref_lat) * R_m

    gt_diff_e = np.diff(gt_e)
    gt_diff_n = np.diff(gt_n)
    gt_heading = np.zeros(len(t))
    gt_heading[1:] = np.arctan2(gt_diff_n, gt_diff_e)
    gt_heading[0] = gt_heading[1]

    # Extract all consecutive 50-step windows across Vfa01
    N = len(phone_imu)
    window_size = 50
    windows = []
    targets_dx = []
    targets_dy = []
    mean_speeds = []

    for i in range(window_size, N, 2):
        start = i - window_size
        win = phone_imu[start:i].T
        de = gt_e[i - 1] - gt_e[start]
        dn = gt_n[i - 1] - gt_n[start]
        psi = gt_heading[start]
        dx = float(np.cos(psi) * de + np.sin(psi) * dn)
        dy = float(-np.sin(psi) * de + np.cos(psi) * dn)
        avg_spd = float(np.mean(v_speed[start:i]))

        windows.append(win)
        targets_dx.append(dx)
        targets_dy.append(dy)
        mean_speeds.append(avg_spd)

    X = torch.tensor(np.array(windows), dtype=torch.float32)
    Y_dx = np.array(targets_dx)
    Y_dy = np.array(targets_dy)
    Speeds = np.array(mean_speeds)

    logger.info(f"Running inference on {len(X)} validation windows...")
    batch_size = 128
    preds = []
    with torch.no_grad():
        for b_start in range(0, len(X), batch_size):
            b_end = min(b_start + batch_size, len(X))
            out = model(X[b_start:b_end])
            preds.append(out.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    pred_dx = preds[:, 0]
    pred_dy = preds[:, 1]
    log_var_x = preds[:, 2]
    log_var_y = preds[:, 3]
    sigma_x = np.exp(0.5 * log_var_x)

    # Convert 5-second displacement to equivalent forward speed (m/s)
    # dx over 50 steps at dt=0.1s is dx / 5.0s
    pred_speed = pred_dx / (window_size * dt)
    true_speed = Y_dx / (window_size * dt)

    dx_err = pred_dx - Y_dx
    spd_err = pred_speed - true_speed

    # 1. Overall Metrics
    disp_mae = float(np.mean(np.abs(dx_err)))
    disp_rmse = float(np.sqrt(np.mean(dx_err ** 2)))
    spd_mae = float(np.mean(np.abs(spd_err)))
    spd_bias = float(np.mean(spd_err))

    # 2. Speed Range Breakdown
    bins = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 1000.0]
    bin_labels = [
        "0–5 m/s",
        "5–10 m/s",
        "10–15 m/s",
        "15–20 m/s",
        "20–25 m/s",
        "25–30+ m/s",
    ]

    range_records = []
    for i in range(len(bin_labels)):
        low, high = bins[i], bins[i + 1]
        mask = (Speeds >= low) if i == len(bin_labels) - 1 else ((Speeds >= low) & (Speeds < high))
        n_m = int(np.sum(mask))
        if n_m == 0:
            continue

        b_dx_err = dx_err[mask]
        b_spd_err = spd_err[mask]
        b_sigma_x = sigma_x[mask]

        # Empirical uncertainty coverage
        cov_1sig = float(np.mean(np.abs(b_dx_err) <= b_sigma_x) * 100.0)
        cov_2sig = float(np.mean(np.abs(b_dx_err) <= 2.0 * b_sigma_x) * 100.0)
        cov_3sig = float(np.mean(np.abs(b_dx_err) <= 3.0 * b_sigma_x) * 100.0)

        range_records.append({
            "Speed Range": bin_labels[i],
            "Samples": n_m,
            "Disp MAE (m)": round(float(np.mean(np.abs(b_dx_err))), 2),
            "Disp RMSE (m)": round(float(np.sqrt(np.mean(b_dx_err ** 2))), 2),
            "Speed MAE (m/s)": round(float(np.mean(np.abs(b_spd_err))), 2),
            "Speed Bias (m/s)": round(float(np.mean(b_spd_err)), 2),
            "1-Sigma Cov (%)": round(cov_1sig, 1),
            "2-Sigma Cov (%)": round(cov_2sig, 1),
        })

    df_range = pd.DataFrame(range_records)
    out_csv = RESULTS_DIR / "ai_odom_independent_validation_vfa01.csv"
    df_range.to_csv(out_csv, index=False)

    # 3. High Speed Focus (> 15 m/s)
    high_mask = Speeds > 15.0
    high_dx_err = dx_err[high_mask]
    high_spd_err = spd_err[high_mask]
    high_spd_mae = float(np.mean(np.abs(high_spd_err)))
    high_spd_bias = float(np.mean(high_spd_err))

    # Overall uncertainty calibration
    cov_1sig_all = float(np.mean(np.abs(dx_err) <= sigma_x) * 100.0)
    cov_2sig_all = float(np.mean(np.abs(dx_err) <= 2.0 * sigma_x) * 100.0)
    cov_3sig_all = float(np.mean(np.abs(dx_err) <= 3.0 * sigma_x) * 100.0)

    print("\n" + "=" * 75)
    print("PHASE 3C: INDEPENDENT AI ODOMETRY VALIDATION ON Vfa01")
    print("=" * 75)
    print(f"Overall Displacement MAE:  {disp_mae:.2f} m")
    print(f"Overall Displacement RMSE: {disp_rmse:.2f} m")
    print(f"Overall Velocity MAE:      {spd_mae:.2f} m/s")
    print(f"Overall Velocity Bias:     {spd_bias:+.2f} m/s")
    print(f"High-Speed (>15 m/s) MAE:  {high_spd_mae:.2f} m/s")
    print(f"High-Speed (>15 m/s) Bias: {high_spd_bias:+.2f} m/s")
    print(f"Uncertainty Calibration:   1-sig: {cov_1sig_all:.1f}% (ideal 68.3%) | 2-sig: {cov_2sig_all:.1f}% (ideal 95.5%) | 3-sig: {cov_3sig_all:.1f}% (ideal 99.7%)")
    print("\nBreakdown by Speed Range:")
    print(df_range.to_string(index=False))


if __name__ == "__main__":
    evaluate_ai_model()
