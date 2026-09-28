"""Phase 5 Evaluation: Phone-to-Vehicle Alignment under Synthetic 3D Rotations.

Verifies that PhoneToVehicleAligner actively estimates and corrects mounting rotations:
- Yaw: 20°, 45°
- Pitch: 15°, 30°
- Roll: 10°, 20°

Compares WITHOUT alignment vs WITH alignment across speed MAE, heading error,
final position error, and drift %.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch

from src.idr.config import set_seed
from src.idr.io.loader import load_drive_pair
from src.idr.calib.alignment import PhoneToVehicleAligner, compute_rotation_matrix
from src.idr.filters.ekf import ExtendedKalmanFilter
from src.idr.filters.nhc import apply_adaptive_nhc_update
from src.idr.filters.vibration_filter import VibrationMotionFilter
from src.idr.models.inertial_odom import InertialOdomNet

RESULTS_DIR = ROOT_DIR / "results" / "forensic_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_alignment_benchmark():
    set_seed(42)
    print("=" * 70)
    print("PHASE 5: ACTIVE PHONE-TO-VEHICLE ALIGNMENT BENCHMARK")
    print("=" * 70)

    # 1. Load authentic Vfa01
    data_dir = ROOT_DIR / "data" / "raw"
    drive_data = load_drive_pair(data_dir, "Vfa01")
    phone_imu, gt_coords, v_speed, t = drive_data.get_synced_data()
    dt = 0.1

    # Convert GT to local ENU
    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    gt_e = np.deg2rad(gt_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
    gt_n = np.deg2rad(gt_coords[:, 0] - ref_lat) * R_m
    gt_xy = np.column_stack([gt_e, gt_n])

    # Pre-blackout heading from displacement
    gt_diff_e = np.diff(gt_e)
    gt_diff_n = np.diff(gt_n)
    gt_heading = np.zeros(len(gt_xy))
    gt_heading[1:] = np.arctan2(gt_diff_n, gt_diff_e)
    gt_heading[0] = gt_heading[1]

    # Evaluate on standard 60s scenario (Vfa01_t220s_d60s)
    start_s = 220
    dur_s = 60
    start_idx = int(start_s / dt)
    end_idx = start_idx + int(dur_s / dt)
    N_seg = end_idx - start_idx

    p0 = gt_xy[start_idx].copy()
    v0 = float(v_speed[start_idx])
    psi0 = float(gt_heading[start_idx])
    gt_seg_xy = gt_xy[start_idx:end_idx]
    gt_seg_speed = v_speed[start_idx:end_idx]
    gt_seg_heading = gt_heading[start_idx:end_idx]
    dist_gt = float(np.sum(gt_seg_speed) * dt)

    # Load trained InertialOdomNet
    cand_b_path = ROOT_DIR / "models" / "inertial_odom.pt"
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(cand_b_path, map_location="cpu"))
    odom_model.eval()

    # Pre-blackout calibration window (t = 160s to 220s)
    calib_start = max(0, start_idx - 600)
    calib_imu = phone_imu[calib_start:start_idx]

    # Synthetic mounting configurations (Phase 5 specifications)
    test_cases = [
        {"name": "Nominal (0°)", "yaw": 0.0, "pitch": 0.0, "roll": 0.0},
        {"name": "Yaw +20°", "yaw": 20.0, "pitch": 0.0, "roll": 0.0},
        {"name": "Yaw +45°", "yaw": 45.0, "pitch": 0.0, "roll": 0.0},
        {"name": "Pitch +15°", "yaw": 0.0, "pitch": 15.0, "roll": 0.0},
        {"name": "Pitch +30°", "yaw": 0.0, "pitch": 30.0, "roll": 0.0},
        {"name": "Roll +10°", "yaw": 0.0, "pitch": 0.0, "roll": 10.0},
        {"name": "Roll +20°", "yaw": 0.0, "pitch": 0.0, "roll": 20.0},
    ]

    alignment_records = []

    for tc in test_cases:
        r_rad = np.deg2rad(tc["roll"])
        p_rad = np.deg2rad(tc["pitch"])
        y_rad = np.deg2rad(tc["yaw"])
        R_synth = compute_rotation_matrix(r_rad, p_rad, y_rad)

        # Apply synthetic rotation to phone IMU
        # acc_phone_synth = acc_phone @ R_synth.T
        calib_imu_synth = calib_imu.copy()
        calib_imu_synth[:, :3] = calib_imu_synth[:, :3] @ R_synth.T
        calib_imu_synth[:, 3:] = calib_imu_synth[:, 3:] @ R_synth.T

        seg_imu_synth = phone_imu[start_idx:end_idx].copy()
        seg_imu_synth[:, :3] = seg_imu_synth[:, :3] @ R_synth.T
        seg_imu_synth[:, 3:] = seg_imu_synth[:, 3:] @ R_synth.T

        # --- Sub-run 1: WITHOUT ALIGNMENT ---
        def run_filter(imu_data, use_aligner=False):
            if use_aligner:
                aligner = PhoneToVehicleAligner()
                # Stationary gravity from first 100 samples
                aligner.update_gravity_estimate(calib_imu_synth[:100, :3])
                aligner.update_forward_motion(calib_imu_synth[100:, :3])
                acc_corr, gyro_corr = aligner.transform_imu(imu_data[:, :3], imu_data[:, 3:])
            else:
                acc_corr = imu_data[:, :3]
                gyro_corr = imu_data[:, 3:]

            ekf = ExtendedKalmanFilter(dt=dt)
            ekf.x[0:2] = p0
            ekf.x[3] = v0 * np.cos(psi0)
            ekf.x[4] = v0 * np.sin(psi0)
            ekf.x[6] = psi0

            vfilter = VibrationMotionFilter(window_size=10, dt=dt)
            v_track = v0
            traj = np.zeros((N_seg, 2), dtype=np.float64)
            traj[0] = p0

            # AI windows
            windows = np.zeros((N_seg, 6, 50), dtype=np.float32)
            for i in range(N_seg):
                glob_i = start_idx + i
                w_start = max(0, glob_i - 50)
                win = imu_data[w_start - start_idx:glob_i - start_idx].T if glob_i - start_idx >= 50 else imu_data[:50].T
                if win.shape[1] < 50:
                    win = np.pad(win, ((0, 0), (50 - win.shape[1], 0)), mode="edge")
                windows[i] = win

            with torch.no_grad():
                out = odom_model(torch.from_numpy(windows)).cpu().numpy()

            headings = [psi0]
            speeds = [v0]

            for i in range(1, N_seg):
                s_acc, s_gyro, state, cov_scale = vfilter.process(acc_corr[i], gyro_corr[i])
                fwd_acc = float(s_acc[0])
                yaw_rate = float(s_gyro[2])

                ekf.predict(fwd_acc, yaw_rate)
                apply_adaptive_nhc_update(ekf, yaw_rate, sigma_lat_base=0.05 * np.sqrt(cov_scale), k_turn=0.5)

                dx_b = float(out[i, 0])
                v_ai = max(0.0, dx_b / (50.0 * dt))
                sig_x = float(np.exp(0.5 * out[i, 2]))
                v_track = 0.85 * (v_track + fwd_acc * dt) + 0.15 * v_ai
                ekf.update_velocity(v_track, R_speed=max(0.1, sig_x * np.sqrt(cov_scale)))

                traj[i] = ekf.x[0:2]
                headings.append(ekf.x[6])
                speeds.append(np.sqrt(ekf.x[3]**2 + ekf.x[4]**2))

            pos_err = float(np.linalg.norm(traj[-1] - gt_seg_xy[-1]))
            drift = (pos_err / dist_gt) * 100.0
            speed_mae = float(np.mean(np.abs(np.array(speeds) - gt_seg_speed)))
            h_err = np.abs(np.rad2deg((np.array(headings) - gt_seg_heading + np.pi) % (2 * np.pi) - np.pi))
            mean_head_err = float(np.mean(h_err))

            return pos_err, drift, speed_mae, mean_head_err

        # Run both
        err_no_align, drift_no_align, spd_no_align, head_no_align = run_filter(seg_imu_synth, use_aligner=False)
        err_with_align, drift_with_align, spd_with_align, head_with_align = run_filter(seg_imu_synth, use_aligner=True)

        print(f"[{tc['name']}]:")
        print(f"  WITHOUT Alignment: Pos Err = {err_no_align:.2f} m | Drift = {drift_no_align:.2f}% | Head Err = {head_no_align:.2f}°")
        print(f"  WITH Alignment:    Pos Err = {err_with_align:.2f} m | Drift = {drift_with_align:.2f}% | Head Err = {head_with_align:.2f}°")

        alignment_records.append({
            "Misalignment": tc["name"],
            "Yaw_deg": tc["yaw"],
            "Pitch_deg": tc["pitch"],
            "Roll_deg": tc["roll"],
            "Error_Without_Align_m": round(err_no_align, 2),
            "Drift_Without_Align_pct": round(drift_no_align, 2),
            "Speed_MAE_Without_Align_mps": round(spd_no_align, 2),
            "Heading_Err_Without_Align_deg": round(head_no_align, 2),
            "Error_With_Align_m": round(err_with_align, 2),
            "Drift_With_Align_pct": round(drift_with_align, 2),
            "Speed_MAE_With_Align_mps": round(spd_with_align, 2),
            "Heading_Err_With_Align_deg": round(head_with_align, 2),
            "Improvement_Factor": round(err_no_align / max(1e-2, err_with_align), 2),
        })

    df_align = pd.DataFrame(alignment_records)
    csv_path = RESULTS_DIR / "synthetic_alignment_benchmark.csv"
    df_align.to_csv(csv_path, index=False)
    print(f"\n[+] Alignment benchmark saved to {csv_path}")
    return df_align


if __name__ == "__main__":
    run_alignment_benchmark()
