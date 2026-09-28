"""Phase 2D: Active Alignment Test on Vfa01 Validation.

Runs the complete navigation pipeline across all Vfa01 validation scenarios under:
Configuration A: Identity Alignment (R = I)
Configuration B: Old Aligner (assumed phone_imu[:100] was stationary + used v_speed)
Configuration C: Corrected Aligner (explicit stationarity detection + zero v_speed)

Compares:
- Heading Error (deg)
- Position Error (m)
- Drift (%)
- Cross-track Error (m)
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
from src.idr.filters.ekf import ExtendedKalmanFilter
from src.idr.filters.nhc import apply_adaptive_nhc_update
from src.idr.filters.vibration_filter import VibrationMotionFilter
from src.idr.calib.alignment import PhoneToVehicleAligner, AlignmentStatus

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_active_alignment_test():
    set_seed(42)
    logger.info("Starting Phase 2D: Active Alignment Test on Vfa01 Validation...")

    data_dir = ROOT_DIR / "data" / "raw"
    drive_data = load_drive_pair(data_dir, "Vfa01")
    phone_imu, gt_coords, v_speed, t = drive_data.get_synced_data()
    dt = 0.1
    N_total = len(t)

    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    gt_e = np.deg2rad(gt_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
    gt_n = np.deg2rad(gt_coords[:, 0] - ref_lat) * R_m
    gt_xy = np.column_stack([gt_e, gt_n])

    gt_diff_e = np.diff(gt_e)
    gt_diff_n = np.diff(gt_n)
    gt_heading = np.zeros(len(t))
    gt_heading[1:] = np.arctan2(gt_diff_n, gt_diff_e)
    gt_heading[0] = gt_heading[1]

    # GNSS velocity pre-blackout (from discrete positions)
    gnss_vel = np.zeros((len(t), 2))
    gnss_vel[1:, 0] = gt_diff_e / dt
    gnss_vel[1:, 1] = gt_diff_n / dt
    gnss_vel[0] = gnss_vel[1]

    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(ROOT_DIR / "models" / "inertial_odom.pt", map_location="cpu"))
    odom_model.eval()

    # Define validation scenarios for Vfa01
    durations = [15, 30, 60]
    stride_step = 200
    scenarios = []

    for dur in durations:
        dur_steps = int(dur / dt)
        if dur == 60:
            start_indices = [100, 300, 500, 700, 900, 1100, 1300, 1500, 450, 2200]
            max_sc = 10
        else:
            start_indices = list(range(100, N_total - dur_steps - 20, stride_step))
            max_sc = 8

        count = 0
        for start_idx in start_indices:
            if count >= max_sc:
                break
            end_idx = start_idx + dur_steps
            seg_speeds = v_speed[start_idx:end_idx]
            dist_gt = float(np.sum(seg_speeds) * dt)
            if dist_gt < 50.0 or np.mean(seg_speeds) < 3.0:
                continue

            scenarios.append({
                "scenario_id": f"Vfa01_t{int(start_idx*dt)}s_d{dur}s",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "dist_gt": dist_gt,
                "dur": dur,
            })
            count += 1

    logger.info(f"Loaded {len(scenarios)} validation scenarios.")

    # -----------------------------------------------------------------
    # Setup the three Aligner Configurations:
    # -----------------------------------------------------------------
    # Configuration A: Identity Aligner
    aligner_A = PhoneToVehicleAligner()
    aligner_A.R_phone_to_vehicle = np.eye(3, dtype=np.float64)

    # Configuration B: Old Aligner (assumed 0:100 was stationary + passed v_speed)
    # Recreate the old flawed behavior:
    class OldFlawedAligner:
        def __init__(self):
            # Blindly took samples 0:100 as gravity
            mean_g = np.mean(phone_imu[:100, :3], axis=0)
            z_p = mean_g / np.linalg.norm(mean_g)
            m_arr = phone_imu[100:400, :3]
            dyn_acc = m_arr - np.outer(m_arr @ z_p, z_p)
            cov = np.cov(dyn_acc, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            x_p = eigvecs[:, np.argmax(eigvals)]
            # Used v_speed
            vel_tail = v_speed[100:400]
            dv = np.gradient(vel_tail)
            proj = dyn_acc @ x_p
            if np.sum(dv * proj) < 0:
                x_p = -x_p
            y_p = np.cross(z_p, x_p)
            y_p /= np.linalg.norm(y_p)
            x_p = np.cross(y_p, z_p)
            x_p /= np.linalg.norm(x_p)
            self.R = np.vstack([x_p, y_p, z_p])

        def transform_imu(self, acc, gyro):
            return (self.R @ acc.T).T, (self.R @ gyro.T).T

    aligner_B = OldFlawedAligner()

    # Configuration C: Corrected Aligner
    # Scans pre-blackout data with automatic stationarity detector; zero reliance on v_speed
    aligner_C = PhoneToVehicleAligner()
    # In Vfa01, scan pre-drive stream (e.g. 0 to 500)
    aligner_C.estimate_attitude(
        phone_imu[:500, :3],
        phone_imu[:500, 3:],
        gnss_vel=gnss_vel[:500],
    )
    logger.info(f"Aligner C status: {aligner_C.status} | Euler: {aligner_C.get_euler_angles()} | Uncertainty: {aligner_C.yaw_uncertainty_deg}°")

    configs = {
        "Config_A_Identity": aligner_A,
        "Config_B_Old_Flawed": aligner_B,
        "Config_C_Corrected": aligner_C,
    }

    results = []

    for cfg_name, aligner in configs.items():
        logger.info(f"Evaluating {cfg_name}...")
        cfg_drifts = []
        cfg_pos_errors = []
        cfg_head_errors = []
        cfg_cross_errors = []

        for sc in scenarios:
            s_idx = sc["start_idx"]
            e_idx = sc["end_idx"]
            N = e_idx - s_idx
            dist_gt = sc["dist_gt"]

            p0 = gt_xy[s_idx].copy()
            v0 = float(v_speed[s_idx])
            psi0 = float(gt_heading[s_idx])
            gt_seg = gt_xy[s_idx:e_idx]
            gt_h_seg = gt_heading[s_idx:e_idx]

            raw_acc = phone_imu[s_idx:e_idx, :3]
            raw_gyro = phone_imu[s_idx:e_idx, 3:]
            acc_v, gyro_v = aligner.transform_imu(raw_acc, raw_gyro)

            # AI windows
            windows = np.zeros((N, 6, 50), dtype=np.float32)
            for i in range(N):
                glob_i = s_idx + i
                w_start = max(0, glob_i - 50)
                win = phone_imu[w_start:glob_i].T
                if win.shape[1] < 50:
                    win = np.pad(win, ((0, 0), (50 - win.shape[1], 0)), mode="edge")
                windows[i] = win

            with torch.no_grad():
                out = odom_model(torch.from_numpy(windows)).cpu().numpy()
                ai_displacements = out[:, :2]
                ai_log_vars = out[:, 2:]

            # Run EKF + Adaptive NHC
            ekf = ExtendedKalmanFilter(dt=dt)
            ekf.x[0:2] = p0
            ekf.x[3] = v0 * np.cos(psi0)
            ekf.x[4] = v0 * np.sin(psi0)
            ekf.x[6] = psi0
            vfilter = VibrationMotionFilter(window_size=10, dt=dt)
            v_track = v0

            traj = np.zeros((N, 2), dtype=np.float64)
            traj[0] = p0
            headings = [psi0]

            for i in range(1, N):
                s_acc, s_gyro, state, cov_scale = vfilter.process(acc_v[i], gyro_v[i])
                fwd_acc = float(s_acc[0])
                yaw_rate = float(s_gyro[2])
                ekf.predict(fwd_acc, yaw_rate)
                apply_adaptive_nhc_update(ekf, yaw_rate, sigma_lat_base=0.05 * np.sqrt(cov_scale), k_turn=0.5)

                dx_b = float(ai_displacements[i, 0])
                v_ai = max(0.0, dx_b / (50.0 * dt))
                sig_x = float(np.exp(0.5 * ai_log_vars[i, 0]))
                v_track = 0.85 * (v_track + fwd_acc * dt) + 0.15 * v_ai
                ekf.update_velocity(v_track, R_speed=max(0.1, sig_x * np.sqrt(cov_scale)))

                traj[i] = ekf.x[0:2]
                headings.append(ekf.x[6])

            err_vec = traj[-1] - gt_seg[-1]
            pos_err = float(np.linalg.norm(err_vec))
            drift = (pos_err / dist_gt) * 100.0

            final_h_err = float(np.abs((headings[-1] - gt_h_seg[-1] + np.pi) % (2 * np.pi) - np.pi) * (180.0 / np.pi))

            psi_f = gt_h_seg[-1]
            u_cross = np.array([-np.sin(psi_f), np.cos(psi_f)])
            cross_err = float(np.abs(np.dot(err_vec, u_cross)))

            cfg_drifts.append(drift)
            cfg_pos_errors.append(pos_err)
            cfg_head_errors.append(final_h_err)
            cfg_cross_errors.append(cross_err)

        results.append({
            "Configuration": cfg_name,
            "Median_Drift_pct": round(float(np.median(cfg_drifts)), 2),
            "Mean_Drift_pct": round(float(np.mean(cfg_drifts)), 2),
            "P90_Drift_pct": round(float(np.percentile(cfg_drifts, 90)), 2),
            "Mean_Pos_Error_m": round(float(np.mean(cfg_pos_errors)), 2),
            "Mean_Heading_Error_deg": round(float(np.mean(cfg_head_errors)), 2),
            "Mean_Cross_Track_m": round(float(np.mean(cfg_cross_errors)), 2),
            "Pass_Rate_10pct": f"{float(np.mean(np.array(cfg_drifts) < 10.0)) * 100:.1f}%",
        })

    df_res = pd.DataFrame(results)
    out_csv = RESULTS_DIR / "active_alignment_comparison_vfa01.csv"
    df_res.to_csv(out_csv, index=False)
    logger.info(f"[+] Active alignment comparison saved to {out_csv}")
    print("\n--- PHASE 2D: ACTIVE ALIGNMENT COMPARISON (Vfa01 VALIDATION) ---")
    print(df_res.to_string())


if __name__ == "__main__":
    run_active_alignment_test()
