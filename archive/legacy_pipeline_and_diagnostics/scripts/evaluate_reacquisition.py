"""Requirement 3 Evaluation: Seamless GNSS Reacquisition Blending.

Compares unmitigated instant GNSS reacquisition (step teleport / velocity spike)
versus the integrated C1 continuous cosine smoother during GNSS restoration.
Generates quantitative metrics (position jump, velocity spike, continuity)
and publication-grade verification plots.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.idr.config import RAW_DATA_DIR, RESULTS_DIR
from src.idr.io.loader import load_drive_pair
from src.idr.filters.fusion import GNSSINSFusion
from src.idr.filters.ekf import ExtendedKalmanFilter
from src.idr.eval.transition import ReacquisitionSmoother


def run_reacquisition_comparison(drive_id: str = "Vfa01"):
    print(f"[*] Loading authentic IO-VNBD drive: {drive_id}...")
    drive = load_drive_pair(RAW_DATA_DIR, drive_id)
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1
    N = min(len(phone_imu), len(gt_coords))

    # Blackout window: 30 seconds (300 steps)
    start_step = 600  # at 60s
    end_step = 900    # at 90s
    recovery_step = end_step

    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]

    # Convert GT to local ENU
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    d_lat = np.deg2rad(gt_coords[:N, 0] - ref_lat)
    d_lon = np.deg2rad(gt_coords[:N, 1] - ref_lon)
    gt_north = d_lat * R_m
    gt_east = d_lon * R_m * np.cos(lat_rad)

    # 1. Run Baseline Unmitigated Snap (instant reset upon recovery)
    fusion_snap = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
    snap_traj = []

    for k in range(N):
        fwd_acc = float(phone_imu[k, 0])
        yaw_rate = float(phone_imu[k, 5])
        
        is_blackout = (start_step <= k < end_step)
        gnss_pos = None if is_blackout else (gt_coords[k, 0], gt_coords[k, 1])

        # Step prediction
        fusion_snap.ekf.predict(fwd_acc, yaw_rate)

        if not is_blackout:
            east, north, _ = fusion_snap.latlon_to_enu(gnss_pos[0], gnss_pos[1])
            if k == recovery_step:
                # Instant hard snap overwrite
                fusion_snap.ekf.x[0] = east
                fusion_snap.ekf.x[1] = north
            else:
                fusion_snap.ekf.update_gnss_pos(np.array([east, north, 0.0]))
        else:
            # Dead reckoning during outage
            pass

        snap_traj.append(fusion_snap.ekf.x.copy())

    snap_traj = np.array(snap_traj)

    # 2. Run Integrated C1 Smooth Reacquisition
    fusion_smooth = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
    smooth_traj = []

    for k in range(N):
        fwd_acc = float(phone_imu[k, 0])
        yaw_rate = float(phone_imu[k, 5])

        is_blackout = (start_step <= k < end_step)
        gnss_pos = None if is_blackout else (gt_coords[k, 0], gt_coords[k, 1])

        # Pass to integrated fusion pipeline (which uses ReacquisitionSmoother and GNSSDeficitDetector)
        state = fusion_smooth.step(
            fwd_accel=fwd_acc,
            yaw_rate=yaw_rate,
            gnss_pos=gnss_pos,
            is_gnss_denied=is_blackout,
            use_nhc=True,
            hdop=1.0 if not is_blackout else 50.0,
            pos_std=2.0 if not is_blackout else 100.0,
        )
        smooth_traj.append(state.copy())

    smooth_traj = np.array(smooth_traj)

    # 3. Compute Metrics at and around Reacquisition Epoch
    # Jump at recovery frame
    snap_pos_before = snap_traj[recovery_step - 1, :2]
    snap_pos_after = snap_traj[recovery_step, :2]
    snap_jump = float(np.linalg.norm(snap_pos_after - snap_pos_before))

    smooth_pos_before = smooth_traj[recovery_step - 1, :2]
    smooth_pos_after = smooth_traj[recovery_step, :2]
    smooth_step_jump = float(np.linalg.norm(smooth_pos_after - smooth_pos_before))

    # Velocity spike around recovery (from recovery_step - 5 to recovery_step + 20)
    eval_window = slice(recovery_step - 5, recovery_step + 25)
    t_eval = t[eval_window]

    snap_speeds = np.linalg.norm(snap_traj[eval_window, 3:5], axis=1)
    # Estimate finite difference velocity spike if hard overwrite occurs
    snap_eff_vel = np.linalg.norm(np.diff(snap_traj[eval_window, :2], axis=0), axis=1) / dt
    snap_max_vel_spike = float(np.max(snap_eff_vel))

    smooth_eff_vel = np.linalg.norm(np.diff(smooth_traj[eval_window, :2], axis=0), axis=1) / dt
    smooth_max_vel_spike = float(np.max(smooth_eff_vel))

    # Continuity metric: max second derivative (apparent acceleration)
    snap_acc_spike = float(np.max(np.abs(np.diff(snap_eff_vel) / dt)))
    smooth_acc_spike = float(np.max(np.abs(np.diff(smooth_eff_vel) / dt)))

    # Error vs GT
    gt_xy = np.column_stack([gt_east[:N], gt_north[:N]])
    snap_errors = np.linalg.norm(snap_traj[:, :2] - gt_xy, axis=1)
    smooth_errors = np.linalg.norm(smooth_traj[:, :2] - gt_xy, axis=1)

    recovery_error_snap = float(snap_errors[recovery_step + 15])  # 1.5s after recovery
    recovery_error_smooth = float(smooth_errors[recovery_step + 15])

    metrics = {
        "drive_id": drive_id,
        "outage_duration_s": (end_step - start_step) * dt,
        "snap_jump_m": round(snap_jump, 3),
        "smooth_step_jump_m": round(smooth_step_jump, 3),
        "jump_reduction_pct": round((1.0 - smooth_step_jump / max(1e-3, snap_jump)) * 100.0, 1),
        "snap_max_vel_spike_mps": round(snap_max_vel_spike, 2),
        "smooth_max_vel_spike_mps": round(smooth_max_vel_spike, 2),
        "snap_acc_spike_mps2": round(snap_acc_spike, 2),
        "smooth_acc_spike_mps2": round(smooth_acc_spike, 2),
        "correction_duration_s": 1.5,
        "recovery_error_snap_m": round(recovery_error_snap, 2),
        "recovery_error_smooth_m": round(recovery_error_smooth, 2),
    }

    out_json = RESULTS_DIR / "reacquisition_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Reacquisition metrics saved to {out_json}")
    print(json.dumps(metrics, indent=2))

    # 4. Generate Publication-Grade Verification Plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    plt.subplots_adjust(hspace=0.35)

    plot_window = slice(recovery_step - 20, recovery_step + 40)
    t_plot = t[plot_window] - t[recovery_step]

    # Plot 1: Position East vs Time around Recovery
    axes[0].plot(t_plot, gt_east[plot_window], "k--", label="Ground Truth (GNSS)", linewidth=2.0)
    axes[0].plot(t_plot, snap_traj[plot_window, 0], "r-.", label="Instant Hard Snap (Baseline)", linewidth=1.8)
    axes[0].plot(t_plot, smooth_traj[plot_window, 0], "b-", label="C1 Smooth Reacquisition (Ours)", linewidth=2.2)
    axes[0].axvline(0.0, color="orange", linestyle=":", label="GNSS Reacquired")
    axes[0].axvspan(0.0, 1.5, color="green", alpha=0.15, label="1.5s C1 Blend Interval")
    axes[0].set_ylabel("East Position (m)", fontsize=11, fontweight="bold")
    axes[0].set_title("Requirement 3: Position Continuity during GNSS Reacquisition", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left")

    # Plot 2: Velocity / Apparent Speed vs Time
    t_vel_plot = (t[plot_window][:-1] + t[plot_window][1:]) / 2.0 - t[recovery_step]
    snap_plot_vel = np.linalg.norm(np.diff(snap_traj[plot_window, :2], axis=0), axis=1) / dt
    smooth_plot_vel = np.linalg.norm(np.diff(smooth_traj[plot_window, :2], axis=0), axis=1) / dt
    axes[1].plot(t_vel_plot, snap_plot_vel, "r-.", label="Instant Snap Speed Spike", linewidth=1.8)
    axes[1].plot(t_vel_plot, smooth_plot_vel, "b-", label="C1 Smooth Velocity Profile", linewidth=2.2)
    axes[1].plot(t[plot_window], v_speed[plot_window], "k:", label="Vehicle Reference Speed", linewidth=1.5)
    axes[1].axvline(0.0, color="orange", linestyle=":")
    axes[1].set_ylabel("Apparent Speed (m/s)", fontsize=11, fontweight="bold")
    axes[1].set_title("Velocity Spike Elimination (No Trajectory Teleportation)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    # Plot 3: Error vs Time across Entire Run
    axes[2].plot(t[:N], snap_errors, "r-.", label="Instant Reset Position Error", alpha=0.8)
    axes[2].plot(t[:N], smooth_errors, "b-", label="Smooth Reacquisition Error", linewidth=2.0)
    axes[2].axvspan(t[start_step], t[end_step], color="red", alpha=0.1, label="30s GNSS Outage")
    axes[2].axvline(t[recovery_step], color="green", linestyle="--", label="Recovery Epoch")
    axes[2].set_xlabel("Elapsed Time (s)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Position Error (m)", fontsize=11, fontweight="bold")
    axes[2].set_title("Overall Trajectory Position Error vs Time", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left")

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_plot = plot_dir / "reacquisition_comparison.png"
    plt.savefig(out_plot, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Reacquisition plots saved to {out_plot}")


if __name__ == "__main__":
    run_reacquisition_comparison("Vfa01")
