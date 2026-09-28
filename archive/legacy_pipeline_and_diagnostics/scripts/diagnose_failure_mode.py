"""Detailed mathematical diagnosis of worst-case scenario Vfa01_t45s_d60s.
Traces IMU -> vibration filter -> alignment -> yaw rate -> acceleration ->
AI displacement -> EKF -> NHC -> map matching at 10 Hz resolution.
Generates all 11 forensic diagnosis curves and identifies the exact failure timestamp.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.idr.config import set_seed
from src.idr.io.loader import load_drive_pair
from src.idr.models.inertial_odom import InertialOdomNet
from src.idr.filters.ekf import ExtendedKalmanFilter
from src.idr.filters.nhc import apply_nhc_update, compute_lateral_velocity
from src.idr.filters.vibration_filter import VibrationMotionFilter
from src.idr.mapmatch.osm_graph import OSMGraphLoader
from src.idr.mapmatch.hmm_matcher import HMMMapMatcher

DIAG_DIR = ROOT_DIR / "results" / "forensic_diagnosis"
DIAG_DIR.mkdir(parents=True, exist_ok=True)


def diagnose_worst_case():
    set_seed(42)
    print("=" * 70)
    print("PHASE 2: MATHEMATICAL FAILURE MODE DIAGNOSIS FOR Vfa01_t45s_d60s")
    print("=" * 70)

    # 1. Load authentic Vfa01
    data_dir = ROOT_DIR / "data" / "raw"
    drive_data = load_drive_pair(data_dir, "Vfa01")
    phone_imu, gt_coords, v_speed, t = drive_data.get_synced_data()
    dt = 0.1

    # Convert GT coordinates to local ENU
    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    gt_e = np.deg2rad(gt_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
    gt_n = np.deg2rad(gt_coords[:, 0] - ref_lat) * R_m
    gt_xy = np.column_stack([gt_e, gt_n])

    # Compute GT heading from displacement
    gt_diff_e = np.diff(gt_e)
    gt_diff_n = np.diff(gt_n)
    gt_heading = np.zeros(len(gt_xy))
    gt_heading[1:] = np.arctan2(gt_diff_n, gt_diff_e)
    gt_heading[0] = gt_heading[1]

    # Compute GT yaw rate
    gt_yaw_rate = np.zeros(len(gt_xy))
    d_psi = np.diff(gt_heading)
    d_psi = (d_psi + np.pi) % (2 * np.pi) - np.pi
    gt_yaw_rate[1:] = d_psi / dt
    gt_yaw_rate[0] = gt_yaw_rate[1]

    # Load OSM road network & HMM matcher
    osm_loader = OSMGraphLoader(cache_dir=ROOT_DIR / "data" / "osm")
    road_graph = osm_loader.load_offline_graph("Vfa01", ref_lat=ref_lat, ref_lon=ref_lon)
    matcher = HMMMapMatcher(road_graph, sigma_z=8.0, beta=4.0, max_search_radius=60.0)

    # Load trained InertialOdomNet
    cand_b_path = ROOT_DIR / "models" / "inertial_odom.pt"
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(cand_b_path, map_location="cpu"))
    odom_model.eval()

    # Scenario parameters
    start_s = 45
    dur_s = 60
    start_idx = int(start_s / dt)
    end_idx = start_idx + int(dur_s / dt)
    N_seg = end_idx - start_idx
    time_rel = np.arange(N_seg) * dt

    p0 = gt_xy[start_idx].copy()
    v0 = float(v_speed[start_idx])
    psi0 = float(gt_heading[start_idx])
    gt_seg_xy = gt_xy[start_idx:end_idx]
    gt_seg_speed = v_speed[start_idx:end_idx]
    gt_seg_heading = gt_heading[start_idx:end_idx]
    gt_seg_yaw_rate = gt_yaw_rate[start_idx:end_idx]

    # Pre-extract AI odometry windows (strictly causal t-50 to t)
    windows = np.zeros((N_seg, 6, 50), dtype=np.float32)
    for i in range(N_seg):
        glob_i = start_idx + i
        w_start = max(0, glob_i - 50)
        win = phone_imu[w_start:glob_i].T
        if win.shape[1] < 50:
            win = np.pad(win, ((0, 0), (50 - win.shape[1], 0)), mode="edge")
        windows[i] = win

    with torch.no_grad():
        out = odom_model(torch.from_numpy(windows)).cpu().numpy()
        ai_displacements = out[:, :2]
        ai_log_vars = out[:, 2:]

    # Run step-by-step EKF with detailed internal telemetry
    ekf = ExtendedKalmanFilter(dt=dt)
    ekf.x[0:2] = p0
    ekf.x[3] = v0 * np.cos(psi0)
    ekf.x[4] = v0 * np.sin(psi0)
    ekf.x[6] = psi0

    vfilter = VibrationMotionFilter(window_size=10, dt=dt)
    v_tracking = v0

    # Telemetry storage
    telemetry = []

    for i in range(N_seg):
        curr_t = start_s + i * dt
        glob_i = start_idx + i

        ax_raw = float(phone_imu[glob_i, 0])
        ay_raw = float(phone_imu[glob_i, 1])
        az_raw = float(phone_imu[glob_i, 2])
        gx_raw = float(phone_imu[glob_i, 3])
        gy_raw = float(phone_imu[glob_i, 4])
        wz_raw = float(phone_imu[glob_i, 5])

        if i == 0:
            # t = 0
            pos_err = 0.0
            heading_err_deg = 0.0
            telemetry.append({
                "time_rel_s": 0.0,
                "time_abs_s": curr_t,
                "gt_heading_deg": np.rad2deg(psi0),
                "est_heading_deg": np.rad2deg(psi0),
                "heading_err_deg": 0.0,
                "gt_yaw_rate_deg_s": np.rad2deg(gt_seg_yaw_rate[0]),
                "meas_yaw_rate_deg_s": np.rad2deg(wz_raw),
                "gt_speed_mps": v0,
                "ai_speed_mps": v0,
                "ekf_speed_mps": v0,
                "long_accel_mps2": ax_raw,
                "lat_accel_mps2": ay_raw,
                "centripetal_accel_mps2": v0 * wz_raw,
                "nhc_innovation_mps": 0.0,
                "nhc_K_heading": 0.0,
                "nhc_gated": False,
                "gyro_bias_deg_s": 0.0,
                "pos_err_pre_map_m": 0.0,
                "est_x": float(p0[0]),
                "est_y": float(p0[1]),
            })
            continue

        # 1. Vibration filter
        s_acc, s_gyro, state, cov_scale = vfilter.process(phone_imu[glob_i, :3], phone_imu[glob_i, 3:])
        fwd_acc = float(s_acc[0])
        yaw_rate = float(s_gyro[2])

        # 2. EKF predict
        ekf.predict(fwd_acc, yaw_rate)

        # 3. NHC Innovation and Kalman Gain inspection
        psi = ekf.x[6]
        ve, vn = ekf.x[3], ekf.x[4]
        v_lat_pred = -np.sin(psi) * ve + np.cos(psi) * vn
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)

        H = np.zeros((2, ekf.dim_x), dtype=np.float64)
        H[0, 3] = -sin_psi
        H[0, 4] = cos_psi
        H[0, 6] = -cos_psi * ve - sin_psi * vn
        H[1, 5] = 1.0

        sigma_lat = 0.05 * np.sqrt(cov_scale)
        R_nhc = np.diag([sigma_lat**2, 0.05**2])
        S = H @ ekf.P @ H.T + R_nhc
        S_inv = np.linalg.inv(S)
        K = ekf.P @ H.T @ S_inv
        K_heading_lat = float(K[6, 0])  # heading correction gain per m/s lateral innovation

        # Apply actual NHC update
        y_nhc = -v_lat_pred
        nis = float(y_nhc**2 * S_inv[0, 0])
        passed, v_before, v_after = apply_nhc_update(ekf, sigma_lat=sigma_lat, sigma_vert=0.05)

        # 4. AI Odometry update
        dx_b = float(ai_displacements[i, 0])
        v_ai = max(0.0, dx_b / (50.0 * dt))
        sig_x = float(np.exp(0.5 * ai_log_vars[i, 0]))
        v_tracking = 0.85 * (v_tracking + fwd_acc * dt) + 0.15 * v_ai
        ekf.update_velocity(v_tracking, R_speed=max(0.1, sig_x * np.sqrt(cov_scale)))

        # Telemetry recording
        pos_err = float(np.linalg.norm(ekf.x[0:2] - gt_seg_xy[i]))
        h_diff = (ekf.x[6] - gt_seg_heading[i] + np.pi) % (2 * np.pi) - np.pi
        heading_err_deg = float(np.rad2deg(h_diff))
        v_curr_fwd = float(np.sqrt(ekf.x[3]**2 + ekf.x[4]**2))

        if np.isnan(ekf.x).any() or np.isinf(ekf.x).any():
            print(f"[!] FATAL: NaN/Inf detected in ekf.x at step {i} (t = {curr_t}s): {ekf.x}")
            break

        telemetry.append({
            "time_rel_s": round(i * dt, 2),
            "time_abs_s": round(curr_t, 2),
            "gt_heading_deg": round(float(np.rad2deg(gt_seg_heading[i])), 2),
            "est_heading_deg": round(float(np.rad2deg(ekf.x[6])), 2),
            "heading_err_deg": round(heading_err_deg, 2),
            "gt_yaw_rate_deg_s": round(float(np.rad2deg(gt_seg_yaw_rate[i])), 2),
            "meas_yaw_rate_deg_s": round(float(np.rad2deg(yaw_rate)), 2),
            "gt_speed_mps": round(float(gt_seg_speed[i]), 2),
            "ai_speed_mps": round(float(v_ai), 2),
            "ekf_speed_mps": round(v_curr_fwd, 2),
            "long_accel_mps2": round(fwd_acc, 3),
            "lat_accel_mps2": round(ay_raw, 3),
            "centripetal_accel_mps2": round(v_curr_fwd * yaw_rate, 3),
            "nhc_innovation_mps": round(float(y_nhc), 3),
            "nhc_K_heading": round(K_heading_lat, 4),
            "nhc_gated": bool(not passed),
            "gyro_bias_deg_s": round(float(np.rad2deg(ekf.x[8])), 4),
            "pos_err_pre_map_m": round(pos_err, 2),
            "est_x": float(ekf.x[0]),
            "est_y": float(ekf.x[1]),
        })

    df_tel = pd.DataFrame(telemetry)

    # Compute map-matched trajectory
    est_xy = np.column_stack([df_tel["est_x"].values, df_tel["est_y"].values])
    matched_xy = matcher.match_trajectory(est_xy)
    map_pos_err = np.linalg.norm(matched_xy - gt_seg_xy, axis=1)
    df_tel["pos_err_post_map_m"] = np.round(map_pos_err, 2)
    df_tel.drop(columns=["est_x", "est_y"], inplace=True)

    csv_path = DIAG_DIR / "vfa01_t45s_telemetry.csv"
    df_tel.to_csv(csv_path, index=False)
    print(f"[+] Saved high-resolution telemetry to {csv_path}")

    # Identify first significant failure timestamp
    # Criteria: pos_err > 10m or abs(heading_err) > 5 deg
    sig_fail = df_tel[(df_tel["pos_err_pre_map_m"] > 10.0) | (np.abs(df_tel["heading_err_deg"]) > 5.0)]
    if len(sig_fail) > 0:
        first_row = sig_fail.iloc[0]
        first_t = first_row["time_rel_s"]
        first_abs_t = first_row["time_abs_s"]
        print(f"\n[!] FIRST SIGNIFICANT FAILURE DETECTED AT t = {first_t}s (abs t = {first_abs_t}s):")
        print(f"    Position Error: {first_row['pos_err_pre_map_m']} m")
        print(f"    Heading Error:  {first_row['heading_err_deg']} deg")
        print(f"    GT Yaw Rate:    {first_row['gt_yaw_rate_deg_s']} deg/s")
        print(f"    Meas Yaw Rate:  {first_row['meas_yaw_rate_deg_s']} deg/s")
        print(f"    NHC Innovation: {first_row['nhc_innovation_mps']} m/s")
        print(f"    NHC K_heading:  {first_row['nhc_K_heading']}")
    else:
        first_t = None
        print("\n[+] No significant error threshold exceeded.")

    # -------------------------------------------------------------
    # GENERATE ALL 11 DIAGNOSIS PLOTS
    # -------------------------------------------------------------
    t_axis = df_tel["time_rel_s"].values

    fig, axes = plt.subplots(6, 2, figsize=(16, 20), dpi=150)
    axes = axes.flatten()

    # 1. Ground-truth heading vs estimated heading
    axes[0].plot(t_axis, df_tel["gt_heading_deg"], label="Ground Truth Heading", color="#10B981", lw=2)
    axes[0].plot(t_axis, df_tel["est_heading_deg"], label="Estimated Heading (EKF)", color="#EF4444", lw=2, linestyle="--")
    axes[0].set_title("1. Heading: Ground Truth vs Estimated", fontweight="bold")
    axes[0].set_ylabel("Heading (deg)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # 2. Ground-truth yaw rate vs measured yaw rate
    axes[1].plot(t_axis, df_tel["gt_yaw_rate_deg_s"], label="GT Yaw Rate (d_psi/dt)", color="#10B981", lw=2)
    axes[1].plot(t_axis, df_tel["meas_yaw_rate_deg_s"], label="Phone Gyro Yaw Rate", color="#3B82F6", lw=1.5, alpha=0.8)
    axes[1].set_title("2. Yaw Rate: Ground Truth vs Measured", fontweight="bold")
    axes[1].set_ylabel("Yaw Rate (deg/s)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # 3. Ground-truth speed vs AI speed vs EKF speed
    axes[2].plot(t_axis, df_tel["gt_speed_mps"], label="Vehicle GT Speed (CAN)", color="#10B981", lw=2)
    axes[2].plot(t_axis, df_tel["ai_speed_mps"], label="InertialOdomNet Speed", color="#3B82F6", lw=1.5, alpha=0.8)
    axes[2].plot(t_axis, df_tel["ekf_speed_mps"], label="Fused EKF Speed", color="#F59E0B", lw=2, linestyle="--")
    axes[2].set_title("3. Forward Speed Comparison", fontweight="bold")
    axes[2].set_ylabel("Speed (m/s)")
    axes[2].legend()
    axes[2].grid(True, linestyle=":", alpha=0.6)

    # 4. Longitudinal acceleration
    axes[3].plot(t_axis, df_tel["long_accel_mps2"], label="Forward Accel (ax)", color="#6366F1", lw=1.5)
    axes[3].set_title("4. Longitudinal Acceleration", fontweight="bold")
    axes[3].set_ylabel("Accel (m/s²)")
    axes[3].legend()
    axes[3].grid(True, linestyle=":", alpha=0.6)

    # 5. Lateral acceleration
    axes[4].plot(t_axis, df_tel["lat_accel_mps2"], label="Lateral Accel (ay)", color="#EC4899", lw=1.5)
    axes[4].set_title("5. Measured Lateral Acceleration", fontweight="bold")
    axes[4].set_ylabel("Accel (m/s²)")
    axes[4].legend()
    axes[4].grid(True, linestyle=":", alpha=0.6)

    # 6. Centripetal acceleration
    axes[5].plot(t_axis, df_tel["centripetal_accel_mps2"], label="Centripetal Accel (v * omega)", color="#8B5CF6", lw=1.5)
    axes[5].plot(t_axis, df_tel["lat_accel_mps2"], label="Measured ay", color="#EC4899", lw=1.0, alpha=0.5, linestyle=":")
    axes[5].set_title("6. Centripetal Acceleration vs Measured ay", fontweight="bold")
    axes[5].set_ylabel("Centripetal Accel (m/s²)")
    axes[5].legend()
    axes[5].grid(True, linestyle=":", alpha=0.6)

    # 7. NHC Innovation
    axes[6].plot(t_axis, df_tel["nhc_innovation_mps"], label="NHC Innovation (-v_lat_pred)", color="#EF4444", lw=1.5)
    axes[6].axhline(0.0, color="black", linestyle=":", lw=1)
    axes[6].set_title("7. NHC Innovation (Lateral Velocity)", fontweight="bold")
    axes[6].set_ylabel("Innovation (m/s)")
    axes[6].legend()
    axes[6].grid(True, linestyle=":", alpha=0.6)

    # 8. NHC Kalman Gain on Heading
    axes[7].plot(t_axis, df_tel["nhc_K_heading"], label="NHC Kalman Gain K_psi", color="#F59E0B", lw=1.5)
    axes[7].set_title("8. NHC Kalman Gain Coupling into Heading", fontweight="bold")
    axes[7].set_ylabel("K[6, 0] (rad / (m/s))")
    axes[7].legend()
    axes[7].grid(True, linestyle=":", alpha=0.6)

    # 9. Gyro bias estimate
    axes[8].plot(t_axis, df_tel["gyro_bias_deg_s"], label="Estimated Gyro Bias b_w", color="#14B8A6", lw=1.5)
    axes[8].set_title("9. Gyroscope Bias Evolution", fontweight="bold")
    axes[8].set_ylabel("Bias (deg/s)")
    axes[8].legend()
    axes[8].grid(True, linestyle=":", alpha=0.6)

    # 10. Position error before map matching
    axes[9].plot(t_axis, df_tel["pos_err_pre_map_m"], label="Pre-Map Position Error (System 5)", color="#F97316", lw=2)
    axes[9].set_title("10. Position Error Before Map Matching", fontweight="bold")
    axes[9].set_ylabel("Error (m)")
    axes[9].legend()
    axes[9].grid(True, linestyle=":", alpha=0.6)

    # 11. Position error after map matching
    axes[10].plot(t_axis, df_tel["pos_err_pre_map_m"], label="Before Map (Sys 5)", color="#F97316", lw=1.5, linestyle="--")
    axes[10].plot(t_axis, df_tel["pos_err_post_map_m"], label="After Map Matching (Sys 6)", color="#EF4444", lw=2)
    axes[10].set_title("11. Position Error After Map Matching", fontweight="bold")
    axes[10].set_ylabel("Error (m)")
    axes[10].legend()
    axes[10].grid(True, linestyle=":", alpha=0.6)

    # 12. Heading Error vs Time
    axes[11].plot(t_axis, df_tel["heading_err_deg"], label="Heading Error (psi_est - psi_gt)", color="#DC2626", lw=2)
    axes[11].axhline(0.0, color="black", linestyle=":", lw=1)
    if first_t is not None:
        axes[11].axvline(first_t, color="black", linestyle="--", label=f"First Failure (t={first_t}s)")
    axes[11].set_title("12. Absolute Heading Error (deg)", fontweight="bold")
    axes[11].set_ylabel("Error (deg)")
    axes[11].legend()
    axes[11].grid(True, linestyle=":", alpha=0.6)

    for ax in axes:
        ax.set_xlabel("Blackout Duration (seconds)")

    plt.tight_layout()
    plot_path = DIAG_DIR / "vfa01_t45s_11_curves_diagnosis.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"[+] Saved 11-curve diagnosis plot to {plot_path}")

    # Generate summary report dictionary
    summary = {
        "scenario_id": "Vfa01_t45s_d60s",
        "first_failure_timestamp_s": first_t,
        "first_failure_abs_timestamp_s": first_abs_t if first_t else None,
        "first_failure_pos_err_m": float(first_row["pos_err_pre_map_m"]) if first_t else None,
        "first_failure_heading_err_deg": float(first_row["heading_err_deg"]) if first_t else None,
        "peak_heading_err_deg": float(df_tel["heading_err_deg"].abs().max()),
        "final_heading_err_deg": float(df_tel["heading_err_deg"].iloc[-1]),
        "final_pos_err_pre_map_m": float(df_tel["pos_err_pre_map_m"].iloc[-1]),
        "final_pos_err_post_map_m": float(df_tel["pos_err_post_map_m"].iloc[-1]),
        "nhc_gated_steps": int(df_tel["nhc_gated"].sum()),
        "max_meas_yaw_rate_deg_s": float(df_tel["meas_yaw_rate_deg_s"].abs().max()),
        "max_gt_yaw_rate_deg_s": float(df_tel["gt_yaw_rate_deg_s"].abs().max()),
        "max_centripetal_accel_mps2": float(df_tel["centripetal_accel_mps2"].abs().max()),
    }
    with open(DIAG_DIR / "failure_diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Summary saved to {DIAG_DIR / 'failure_diagnosis_summary.json'}")
    return summary


if __name__ == "__main__":
    import json
    diagnose_worst_case()
