"""Phase 1 & 2: Forensic Heading Error and Gyro Axis Analysis on Vfa02.

For every Vfa02 scenario (all 24 scenarios):
- Initial heading, final heading, ground-truth heading
- Heading error vs time
- Gyro z / yaw-rate bias (measured vs true vehicle yaw rate)
- Yaw-rate RMSE
- Integrated yaw error
- Speed
- Along-track error vs cross-track error
- Correlation between heading error and position failure.

Also verifies:
- Which phone gyro axis corresponds to vehicle yaw
- Whether omega_vehicle = R_phone_to_vehicle @ omega_phone correctly extracts vehicle yaw rate
"""

import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.idr.config import set_seed
from src.idr.io.loader import load_drive_pair
from src.idr.calib.alignment import PhoneToVehicleAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "forensic_heading"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_forensic_heading_analysis():
    set_seed(42)
    logger.info("Starting Phase 1: Forensic Heading Error Analysis on Vfa02...")

    data_dir = ROOT_DIR / "data" / "raw"
    drive = load_drive_pair(data_dir, "Vfa02")
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1
    N_total = len(t)

    # True vehicle yaw rate from vehicle ECU (converted deg/s -> rad/s in loader)
    if "yaw_rate" in drive.vehicle_df.columns:
        gt_yaw_rate = drive.vehicle_df["yaw_rate"].to_numpy(dtype=np.float64)[:N_total]
    else:
        gt_yaw_rate = np.zeros(N_total)

    # Coordinates in local ENU
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

    # Phase 2: Check phone alignment matrix
    aligner = PhoneToVehicleAligner()
    aligner.estimate_attitude(phone_imu[:500, :3], phone_imu[:500, 3:], gnss_vel=gnss_vel[:500])
    R_p2v = aligner.R_phone_to_vehicle
    r_deg, p_deg, y_deg = aligner.get_euler_angles()
    logger.info(f"Phone alignment on Vfa02: Roll={r_deg:.2f}°, Pitch={p_deg:.2f}°, Yaw={y_deg:.2f}°")
    logger.info(f"R_p2v matrix:\n{R_p2v}")

    # Transform full gyro stream into vehicle frame
    # gyro is [gx, gy, gz] = [roll_rate, pitch_rate, yaw_rate]
    raw_gyro = phone_imu[:, 3:].astype(np.float64)
    v_gyro = (R_p2v @ raw_gyro.T).T

    # Inspect gyro axes vs true vehicle yaw rate
    corr_gx = np.corrcoef(raw_gyro[:, 0], gt_yaw_rate)[0, 1]
    corr_gy = np.corrcoef(raw_gyro[:, 1], gt_yaw_rate)[0, 1]
    corr_gz = np.corrcoef(raw_gyro[:, 2], gt_yaw_rate)[0, 1]
    corr_vg = np.corrcoef(v_gyro[:, 2], gt_yaw_rate)[0, 1]
    logger.info(f"Correlation with True Vehicle Yaw Rate: raw_gx={corr_gx:.3f}, raw_gy={corr_gy:.3f}, raw_gz={corr_gz:.3f}, v_gyro_z={corr_vg:.3f}")

    # Check overall gyro bias across whole drive
    bias_raw_gz = float(np.mean(raw_gyro[:, 2] - gt_yaw_rate))
    bias_vg_z = float(np.mean(v_gyro[:, 2] - gt_yaw_rate))
    logger.info(f"Mean Gyro Bias across Vfa02: raw_gz bias = {bias_raw_gz:+.4f} rad/s ({np.rad2deg(bias_raw_gz):+.2f} deg/s)")
    logger.info(f"Mean Gyro Bias across Vfa02: v_gyro_z bias = {bias_vg_z:+.4f} rad/s ({np.rad2deg(bias_vg_z):+.2f} deg/s)")

    # 24 scenarios definition
    durations = [15, 30, 60]
    stride_step = 200
    scenarios = []

    for dur in durations:
        dur_steps = int(dur / dt)
        start_indices = list(range(100, N_total - dur_steps - 20, stride_step))
        count = 0
        for start_idx in start_indices:
            if count >= 8:
                break
            end_idx = start_idx + dur_steps
            seg_speeds = v_speed[start_idx:end_idx]
            dist_gt = float(np.sum(seg_speeds) * dt)
            if dist_gt < 50.0 or np.mean(seg_speeds) < 3.0:
                continue

            scenarios.append({
                "scenario_id": f"Vfa02_t{int(start_idx*dt)}s_d{dur}s",
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration_s": dur,
                "dist_gt": dist_gt,
            })
            count += 1

    # Load previously computed final Vfa02 metrics to correlate
    prev_metrics_file = ROOT_DIR / "results" / "vfa02_final_hardened_metrics.csv"
    if prev_metrics_file.exists():
        df_prev = pd.read_csv(prev_metrics_file)
        err_lookup = dict(zip(df_prev["scenario_id"], df_prev["post_map_pos_error_m"]))
        drift_lookup = dict(zip(df_prev["scenario_id"], df_prev["post_drift_pct"]))
    else:
        err_lookup = {}
        drift_lookup = {}

    forensic_records = []
    sample_time_series = None

    for sc in scenarios:
        s_idx, e_idx = sc["start_idx"], sc["end_idx"]
        N = e_idx - s_idx
        dur = sc["duration_s"]
        dist_gt = sc["dist_gt"]

        gt_h_seg = gt_heading[s_idx:e_idx]
        gt_yr_seg = gt_yaw_rate[s_idx:e_idx]
        raw_gz_seg = raw_gyro[s_idx:e_idx, 2]
        vg_z_seg = v_gyro[s_idx:e_idx, 2]
        spd_seg = v_speed[s_idx:e_idx]
        gt_xy_seg = gt_xy[s_idx:e_idx]

        # Integrate yaw from raw gz and from v_gyro_z
        psi0 = float(gt_h_seg[0])
        raw_yaw = np.zeros(N)
        vg_yaw = np.zeros(N)
        raw_yaw[0] = psi0
        vg_yaw[0] = psi0

        for i in range(1, N):
            raw_yaw[i] = (raw_yaw[i - 1] + raw_gz_seg[i] * dt + np.pi) % (2 * np.pi) - np.pi
            vg_yaw[i] = (vg_yaw[i - 1] + vg_z_seg[i] * dt + np.pi) % (2 * np.pi) - np.pi

        # Angular errors
        raw_h_err = np.abs((raw_yaw - gt_h_seg + np.pi) % (2 * np.pi) - np.pi) * (180.0 / np.pi)
        vg_h_err = np.abs((vg_yaw - gt_h_seg + np.pi) % (2 * np.pi) - np.pi) * (180.0 / np.pi)

        # Gyro yaw-rate errors
        yr_err_raw = raw_gz_seg - gt_yr_seg
        yr_err_vg = vg_z_seg - gt_yr_seg
        raw_bias = float(np.mean(yr_err_raw))
        vg_bias = float(np.mean(yr_err_vg))
        raw_yr_rmse = float(np.sqrt(np.mean(yr_err_raw ** 2)))
        vg_yr_rmse = float(np.sqrt(np.mean(yr_err_vg ** 2)))

        final_pos_err = err_lookup.get(sc["scenario_id"], 0.0)
        final_drift = drift_lookup.get(sc["scenario_id"], 0.0)

        # Decompose position error into along-track and cross-track
        # End vector:
        psi_f = gt_h_seg[-1]
        u_along = np.array([np.cos(psi_f), np.sin(psi_f)])
        u_cross = np.array([-np.sin(psi_f), np.cos(psi_f)])

        # Dead-reckoning position integration with vg_yaw and true speed
        # To isolate heading contribution:
        dr_xy = np.zeros((N, 2))
        dr_xy[0] = gt_xy_seg[0]
        for i in range(1, N):
            dr_xy[i, 0] = dr_xy[i - 1, 0] + spd_seg[i] * np.cos(vg_yaw[i]) * dt
            dr_xy[i, 1] = dr_xy[i - 1, 1] + spd_seg[i] * np.sin(vg_yaw[i]) * dt

        err_vec = dr_xy[-1] - gt_xy_seg[-1]
        along_err = float(np.dot(err_vec, u_along))
        cross_err = float(np.dot(err_vec, u_cross))

        forensic_records.append({
            "scenario_id": sc["scenario_id"],
            "duration_s": dur,
            "dist_gt_m": round(dist_gt, 1),
            "mean_speed_mps": round(float(np.mean(spd_seg)), 2),
            "init_heading_deg": round(float(np.rad2deg(psi0)), 2),
            "final_gt_heading_deg": round(float(np.rad2deg(psi_f)), 2),
            "final_raw_h_err_deg": round(float(raw_h_err[-1]), 2),
            "final_vg_h_err_deg": round(float(vg_h_err[-1]), 2),
            "mean_vg_h_err_deg": round(float(np.mean(vg_h_err)), 2),
            "vg_bias_deg_s": round(float(np.rad2deg(vg_bias)), 3),
            "vg_yr_rmse_deg_s": round(float(np.rad2deg(vg_yr_rmse)), 3),
            "actual_pos_err_m": round(final_pos_err, 2),
            "actual_drift_pct": round(final_drift, 2),
            "heading_only_cross_err_m": round(cross_err, 2),
            "heading_only_along_err_m": round(along_err, 2),
        })

        if sc["scenario_id"] == "Vfa02_t50s_d60s":
            sample_time_series = {
                "t": np.arange(N) * dt,
                "gt_heading_deg": np.rad2deg(gt_h_seg),
                "est_heading_deg": np.rad2deg(vg_yaw),
                "h_err_deg": vg_h_err,
                "gt_yaw_rate": np.rad2deg(gt_yr_seg),
                "meas_yaw_rate": np.rad2deg(vg_z_seg),
                "yr_err": np.rad2deg(yr_err_vg),
                "pos_err_t": np.linalg.norm(dr_xy - gt_xy_seg, axis=1),
                "along_t": [np.dot(dr_xy[k] - gt_xy_seg[k], np.array([np.cos(gt_h_seg[k]), np.sin(gt_h_seg[k])])) for k in range(N)],
                "cross_t": [np.dot(dr_xy[k] - gt_xy_seg[k], np.array([-np.sin(gt_h_seg[k]), np.cos(gt_h_seg[k])])) for k in range(N)],
            }

    df_forensic = pd.DataFrame(forensic_records)
    out_csv = RESULTS_DIR / "vfa02_forensic_heading_summary.csv"
    df_forensic.to_csv(out_csv, index=False)
    logger.info(f"[+] Forensic heading summary saved to {out_csv}")

    # Generate the 5 required forensic plots:
    if sample_time_series is not None:
        st = sample_time_series
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # 1. Heading error vs time
        axes[0, 0].plot(st["t"], st["h_err_deg"], "r-", linewidth=2.0)
        axes[0, 0].set_title("1. Heading Error vs Time (Vfa02_t50s_d60s)")
        axes[0, 0].set_xlabel("Time in Blackout (s)")
        axes[0, 0].set_ylabel("Heading Error (deg)")
        axes[0, 0].grid(True)

        # 2. Gyro bias vs time (cumulative running mean of error)
        running_bias = np.cumsum(st["yr_err"]) / (np.arange(len(st["yr_err"])) + 1)
        axes[0, 1].plot(st["t"], running_bias, "b-", linewidth=2.0)
        axes[0, 1].axhline(0, color="k", linestyle="--")
        axes[0, 1].set_title("2. Estimated Gyro Bias vs Time")
        axes[0, 1].set_xlabel("Time in Blackout (s)")
        axes[0, 1].set_ylabel("Running Mean Yaw-Rate Bias (deg/s)")
        axes[0, 1].grid(True)

        # 3. Yaw-rate true vs estimated
        axes[1, 0].plot(st["t"], st["gt_yaw_rate"], "k-", label="True Yaw Rate (CAN)", linewidth=1.5)
        axes[1, 0].plot(st["t"], st["meas_yaw_rate"], "g--", label="Phone Vehicle-Frame Yaw Rate", linewidth=1.5)
        axes[1, 0].set_title("3. Yaw-Rate: Ground Truth vs Estimated")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Yaw Rate (deg/s)")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # 4. Position error vs time
        axes[1, 1].plot(st["t"], st["pos_err_t"], "m-", linewidth=2.0)
        axes[1, 1].set_title("4. Position Error vs Time")
        axes[1, 1].set_xlabel("Time in Blackout (s)")
        axes[1, 1].set_ylabel("Position Error (m)")
        axes[1, 1].grid(True)

        # 5. Along-track vs Cross-track error
        axes[2, 0].plot(st["t"], st["along_t"], "c-", label="Along-Track Error (m)", linewidth=1.8)
        axes[2, 0].plot(st["t"], st["cross_t"], "r-", label="Cross-Track Error (m)", linewidth=1.8)
        axes[2, 0].set_title("5. Along-Track vs Cross-Track Error")
        axes[2, 0].set_xlabel("Time in Blackout (s)")
        axes[2, 0].set_ylabel("Error (m)")
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        # 6. Correlation scatter across all 24 scenarios: Heading Error vs Position Error
        axes[2, 1].scatter(df_forensic["final_vg_h_err_deg"], df_forensic["actual_pos_err_m"], c="purple", s=50, edgecolors="k")
        corr = df_forensic["final_vg_h_err_deg"].corr(df_forensic["actual_pos_err_m"])
        axes[2, 1].set_title(f"6. Heading Error vs Position Error (r = {corr:.3f})")
        axes[2, 1].set_xlabel("Final Heading Error (deg)")
        axes[2, 1].set_ylabel("Final Position Error (m)")
        axes[2, 1].grid(True)

        plt.tight_layout()
        plot_path = RESULTS_DIR / "vfa02_forensic_heading_diagnosis.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.info(f"[+] Saved 6-panel forensic diagnosis plot to {plot_path}")

    # Compute correlations
    corr_pos_head = df_forensic["actual_pos_err_m"].corr(df_forensic["final_vg_h_err_deg"])
    corr_drift_head = df_forensic["actual_drift_pct"].corr(df_forensic["final_vg_h_err_deg"])
    corr_cross_head = df_forensic["heading_only_cross_err_m"].abs().corr(df_forensic["final_vg_h_err_deg"])

    print("\n" + "=" * 80)
    print("PHASE 1 FORENSIC FINDINGS: CORRELATION OF POSITION FAILURE WITH HEADING DRIFT")
    print("=" * 80)
    print(f"Correlation: Position Error vs Final Heading Error: r = {corr_pos_head:.3f}")
    print(f"Correlation: Drift % vs Final Heading Error:       r = {corr_drift_head:.3f}")
    print(f"Correlation: Cross-Track Error vs Heading Error:   r = {corr_cross_head:.3f}")
    print(f"Mean Final Heading Error across 24 scenarios:       {df_forensic['final_vg_h_err_deg'].mean():.2f}°")
    print(f"Mean Gyro Bias across 24 scenarios:                {df_forensic['vg_bias_deg_s'].mean():+.3f} deg/s")
    print(f"Mean Heading-induced Cross-Track Error:            {df_forensic['heading_only_cross_err_m'].abs().mean():.2f} m")
    print("\nSummary by Duration:")
    for dur in [15, 30, 60]:
        sub = df_forensic[df_forensic["duration_s"] == dur]
        print(f"  Duration {dur:2d}s: Mean Heading Error = {sub['final_vg_h_err_deg'].mean():5.2f}°, Mean Pos Err = {sub['actual_pos_err_m'].mean():6.2f}m, Cross Err = {sub['heading_only_cross_err_m'].abs().mean():6.2f}m")


if __name__ == "__main__":
    run_forensic_heading_analysis()
