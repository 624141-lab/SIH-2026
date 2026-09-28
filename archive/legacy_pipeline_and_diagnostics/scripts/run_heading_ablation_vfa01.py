"""Phase 9: Rigorous Heading & Gyro Bias Ablation on Vfa01 Validation.

Compares Configurations A through F:
A. Current gyro heading (baseline EKF, b_w=0)
B. Correct phone-to-vehicle gyro transformation (R_p2v @ omega)
C. + Explicit gyro bias estimation (Pre-blackout GNSS-aided bias initialization)
D. + Stationary ZARU/ZUPT updates
E. + Gated Magnetometer (norm in [45, 65] uT, innov < 15 deg)
F. + AI Odometry Directional Consistency (psi_AI constraint)

Outputs:
- heading RMSE (deg)
- final position RMSE (m)
- drift (%) median, mean, P90
- cross-track error (m)
- pass rate (<10%)
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
from src.idr.filters.zupt import StationaryDetector, apply_zupt, apply_zaru
from src.idr.calib.alignment import PhoneToVehicleAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "heading_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_heading_ablation():
    set_seed(42)
    logger.info("Starting Phase 9: Heading & Bias Ablation on Vfa01...")

    data_dir = ROOT_DIR / "data" / "raw"
    drive = load_drive_pair(data_dir, "Vfa01")
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1
    N_total = len(t)

    # Reference coordinates
    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    gt_e = np.deg2rad(gt_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
    gt_n = np.deg2rad(gt_coords[:, 0] - ref_lat) * R_m
    gt_xy = np.column_stack([gt_e, gt_n])

    gt_diff_e = np.diff(gt_e)
    gt_diff_n = np.diff(gt_n)
    gt_heading = np.zeros(N_total)
    gt_heading[1:] = np.arctan2(gt_diff_n, gt_diff_e)
    gt_heading[0] = gt_heading[1]

    # Pre-blackout GNSS velocity
    gnss_vel = np.zeros((N_total, 2))
    gnss_vel[1:, 0] = gt_diff_e / dt
    gnss_vel[1:, 1] = gt_diff_n / dt
    gnss_vel[0] = gnss_vel[1]

    # Magnetometer readings
    mag_x = drive.phone_df["mag_x"].to_numpy()
    mag_y = drive.phone_df["mag_y"].to_numpy()
    mag_z = drive.phone_df["mag_z"].to_numpy()
    mag_raw = np.column_stack([mag_x, mag_y, mag_z])
    mag_norm = np.linalg.norm(mag_raw, axis=1)

    # Phone-to-vehicle aligner
    aligner = PhoneToVehicleAligner()
    aligner.estimate_attitude(phone_imu[:500, :3], phone_imu[:500, 3:], gnss_vel=gnss_vel[:500])
    R_p2v = aligner.R_phone_to_vehicle
    corr_acc, corr_gyro = aligner.transform_imu(phone_imu[:, :3], phone_imu[:, 3:])
    mag_v = (R_p2v @ mag_raw.T).T

    # Load retrained AI model
    model_path = ROOT_DIR / "models" / "inertial_odom.pt"
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    odom_model.eval()

    # Precompute AI windows across drive
    batch_size = 512
    ai_dx = np.zeros(N_total)
    ai_dy = np.zeros(N_total)
    ai_sig_x = np.zeros(N_total)

    for start in range(0, N_total, batch_size):
        end = min(N_total, start + batch_size)
        B = end - start
        windows = np.zeros((B, 6, 50), dtype=np.float32)
        for idx, i in enumerate(range(start, end)):
            w_start = max(0, i - 50)
            win = phone_imu[w_start:i].T
            if win.shape[1] == 0:
                win = np.zeros((6, 50), dtype=np.float32)
            elif win.shape[1] < 50:
                win = np.pad(win, ((0, 0), (50 - win.shape[1], 0)), mode="edge")
            windows[idx] = win
        with torch.no_grad():
            out = odom_model(torch.from_numpy(windows)).cpu().numpy()
            ai_dx[start:end] = out[:, 0]
            ai_dy[start:end] = out[:, 1]
            ai_sig_x[start:end] = np.exp(0.5 * out[:, 2])

    # Scenarios on Vfa01
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
            if end_idx >= N_total:
                continue
            seg_speeds = v_speed[start_idx:end_idx]
            dist_gt = float(np.sum(seg_speeds) * dt)
            if dist_gt < 50.0 or np.mean(seg_speeds) < 3.0:
                continue

            scenarios.append({
                "scenario_id": f"Vfa01_t{int(start_idx*dt)}s_d{dur}s",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration_s": dur,
                "dist_gt": dist_gt,
                "gt_seg": gt_xy[start_idx:end_idx],
                "gt_h_seg": gt_heading[start_idx:end_idx],
                "p0": gt_xy[start_idx].copy(),
                "v0": float(seg_speeds[0]),
                "psi0": float(gt_heading[start_idx]),
            })
            count += 1

    logger.info(f"Loaded {len(scenarios)} validation scenarios on Vfa01.")

    configs = [
        "A_baseline_gyro",
        "B_transformed_gyro",
        "C_explicit_bias_est",
        "D_zaru_stationary",
        "E_gated_magnetometer",
        "F_ai_directional_consistency",
    ]

    all_records = []

    for cfg in configs:
        logger.info(f"Evaluating Configuration: {cfg}...")
        for sc in scenarios:
            s_idx, e_idx = sc["start_idx"], sc["end_idx"]
            N = e_idx - s_idx
            dist_gt = sc["dist_gt"]
            gt_end = sc["gt_seg"][-1]
            p0, v0, psi0 = sc["p0"], sc["v0"], sc["psi0"]
            gt_h_seg = sc["gt_h_seg"]

            ekf = ExtendedKalmanFilter(dt=dt)

            # Pre-blackout window for bias estimation in configs C, D, E, F
            if cfg in ["C_explicit_bias_est", "D_zaru_stationary", "E_gated_magnetometer", "F_ai_directional_consistency"]:
                pre_win = 200  # 20 seconds
                pre_start = max(0, s_idx - pre_win)
                ekf.x[0] = gt_e[pre_start]
                ekf.x[1] = gt_n[pre_start]
                ekf.x[3] = gnss_vel[pre_start, 0]
                ekf.x[4] = gnss_vel[pre_start, 1]
                ekf.x[6] = np.arctan2(gnss_vel[pre_start, 1], gnss_vel[pre_start, 0])

                for pi in range(pre_start, s_idx):
                    fwd_a = corr_acc[pi, 0]
                    w_z = corr_gyro[pi, 2]
                    ekf.predict(fwd_a, w_z)
                    ekf.update_gnss_pos(np.array([gt_e[pi], gt_n[pi], 0.0]), R_cov=np.eye(3) * (2.0 ** 2))
                    spd_p = np.hypot(gnss_vel[pi, 0], gnss_vel[pi, 1])
                    if spd_p > 3.0:
                        cog_p = np.arctan2(gnss_vel[pi, 1], gnss_vel[pi, 0])
                        ekf.update_heading(cog_p, R_yaw=0.08)
            else:
                # Direct initialization at blackout start
                ekf.x[0:2] = p0
                ekf.x[3] = v0 * np.cos(psi0)
                ekf.x[4] = v0 * np.sin(psi0)
                ekf.x[6] = psi0
                ekf.x[8] = 0.0

            # Blackout dead-reckoning
            vfilter = VibrationMotionFilter(window_size=10, dt=dt)
            stat_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15, gyro_norm_threshold=0.05)
            v_track = np.hypot(ekf.x[3], ekf.x[4])

            traj = np.zeros((N, 2))
            traj[0] = ekf.x[:2]
            est_heading = np.zeros(N)
            est_heading[0] = ekf.x[6]

            for i in range(1, N):
                glob_i = s_idx + i

                # Gyro yaw rate selection:
                if cfg == "A_baseline_gyro":
                    # Untransformed phone gyro_z
                    fwd_a = float(phone_imu[glob_i, 0])
                    w_z = float(phone_imu[glob_i, 5])
                else:
                    # Aligned vehicle frame
                    s_acc, s_gyro, _, cov_scale = vfilter.process(corr_acc[glob_i], corr_gyro[glob_i])
                    fwd_a = float(s_acc[0])
                    w_z = float(s_gyro[2])

                # Optional Config F: AI odometry directional consistency
                if cfg == "F_ai_directional_consistency":
                    dx_b = float(ai_dx[glob_i])
                    dy_b = float(ai_dy[glob_i])
                    disp_mag = np.hypot(dx_b, dy_b)
                    if disp_mag > 1.0:
                        psi_ai_body = np.arctan2(dy_b, dx_b)  # steering angle in body frame
                        # In vehicle body frame, slip angle |psi_ai_body| should be small (< 0.2 rad = 11.5 deg)
                        if abs(psi_ai_body) > 0.25:
                            # Possible anomalous turn or lateral drift: increase turn uncertainty
                            w_z = 0.8 * w_z

                ekf.predict(fwd_a, w_z)

                # NHC
                apply_adaptive_nhc_update(ekf, w_z, sigma_lat_base=0.05, k_turn=0.5)

                # Config D, E, F: Stationary ZARU / ZUPT updates
                if cfg in ["D_zaru_stationary", "E_gated_magnetometer", "F_ai_directional_consistency"]:
                    if stat_detector.update(phone_imu[glob_i, :3], phone_imu[glob_i, 3:]):
                        apply_zupt(ekf, sigma_v=0.01)
                        apply_zaru(ekf, gyro_z_raw=w_z, sigma_bias=0.005)

                # Config E: Gated Magnetometer observation
                if cfg == "E_gated_magnetometer":
                    m_norm = mag_norm[glob_i]
                    if 45.0 <= m_norm <= 65.0:
                        # Compute horizontal magnetic heading in ENU
                        psi_m = np.arctan2(mag_v[glob_i, 1], mag_v[glob_i, 0])
                        # Hard-iron offset calibration
                        psi_m_corr = (psi_m - np.deg2rad(94.66) + np.pi) % (2 * np.pi) - np.pi
                        innov_mag = np.abs((psi_m_corr - ekf.x[6] + np.pi) % (2 * np.pi) - np.pi)
                        if innov_mag < np.deg2rad(15.0):
                            ekf.update_heading(psi_m_corr, R_yaw=0.25)

                # AI speed update
                v_ai = max(0.0, float(ai_dx[glob_i]) / (50.0 * dt))
                v_track = 0.85 * (v_track + fwd_a * dt) + 0.15 * v_ai
                ekf.update_velocity(v_track, R_speed=max(0.1, float(ai_sig_x[glob_i])))

                traj[i] = ekf.x[:2]
                est_heading[i] = ekf.x[6]

            # Compute errors
            pos_err = float(np.linalg.norm(traj[-1] - gt_end))
            drift_pct = (pos_err / dist_gt) * 100.0

            # Heading error vs time
            h_diff = (est_heading - gt_h_seg + np.pi) % (2 * np.pi) - np.pi
            h_rmse = float(np.sqrt(np.mean(h_diff ** 2))) * (180.0 / np.pi)
            final_h_err = float(np.abs(h_diff[-1])) * (180.0 / np.pi)

            # Along-track and cross-track error
            psi_f = gt_h_seg[-1]
            u_along = np.array([np.cos(psi_f), np.sin(psi_f)])
            u_cross = np.array([-np.sin(psi_f), np.cos(psi_f)])
            err_vec = traj[-1] - gt_end
            along_err = float(np.abs(np.dot(err_vec, u_along)))
            cross_err = float(np.abs(np.dot(err_vec, u_cross)))

            all_records.append({
                "config": cfg,
                "scenario_id": sc["scenario_id"],
                "duration_s": sc["duration_s"],
                "dist_gt_m": round(dist_gt, 2),
                "pos_error_m": round(pos_err, 2),
                "drift_pct": round(drift_pct, 2),
                "pass_10pct": drift_pct < 10.0,
                "heading_rmse_deg": round(h_rmse, 2),
                "final_heading_err_deg": round(final_h_err, 2),
                "along_track_err_m": round(along_err, 2),
                "cross_track_err_m": round(cross_err, 2),
            })

    df_all = pd.DataFrame(all_records)
    csv_out = RESULTS_DIR / "vfa01_heading_ablation_detailed.csv"
    df_all.to_csv(csv_out, index=False)

    summary_rows = []
    for cfg in configs:
        sub = df_all[df_all["config"] == cfg]
        summary_rows.append({
            "Configuration": cfg,
            "Heading RMSE (deg)": round(float(sub["heading_rmse_deg"].mean()), 2),
            "Pos RMSE (m)": round(float(np.sqrt(np.mean(sub["pos_error_m"] ** 2))), 2),
            "Median Drift (%)": round(float(sub["drift_pct"].median()), 2),
            "Mean Drift (%)": round(float(sub["drift_pct"].mean()), 2),
            "P90 Drift (%)": round(float(np.percentile(sub["drift_pct"], 90)), 2),
            "Mean Cross-Track (m)": round(float(sub["cross_track_err_m"].mean()), 2),
            "Mean Along-Track (m)": round(float(sub["along_track_err_m"].mean()), 2),
            "Pass Rate (<10%)": f"{float(np.mean(sub['pass_10pct']) * 100):.1f}%",
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_sum = RESULTS_DIR / "vfa01_heading_ablation_summary.csv"
    df_summary.to_csv(csv_sum, index=False)

    logger.info("\n=== Vfa01 HEADING ABLATION SUMMARY ===\n" + df_summary.to_string(index=False))
    return df_summary


if __name__ == "__main__":
    run_heading_ablation()
