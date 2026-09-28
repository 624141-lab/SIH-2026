"""Requirement 7 Evaluation: Phone-to-Vehicle Alignment & Rotation Observability.

Benchmarks PhoneToVehicleAligner under challenging synthetic mount orientations:
- yaw = 20°, 45°
- pitch = 15°, 30°
- roll = 10°, 20°

Verifies:
1. ALIGNMENT_UNCERTAIN state when vehicle is stationary (yaw unobservable).
2. Convergence to ALIGNED state once vehicle forward acceleration is detected.
3. Recovered Euler angle accuracy (roll, pitch, yaw errors).
4. Trajectory drift comparison: WITHOUT alignment vs WITH alignment.
"""

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.idr.config import RAW_DATA_DIR, RESULTS_DIR
from src.idr.io.loader import load_drive_pair
from src.idr.calib.alignment import (
    PhoneToVehicleAligner,
    AlignmentStatus,
    compute_rotation_matrix,
    rotation_matrix_to_euler,
)
from src.idr.filters.fusion import GNSSINSFusion


def run_alignment_benchmark(drive_id: str = "Vfa01"):
    print(f"[*] Loading drive {drive_id} for alignment benchmarking...")
    drive = load_drive_pair(RAW_DATA_DIR, drive_id)
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1
    N = min(len(phone_imu), len(gt_coords))

    # Identify stationary and accelerating segments in Vfa01
    # First 15 seconds: stationary engine idle
    stat_slice = slice(0, 150)
    # Next 25 seconds: forward acceleration
    motion_slice = slice(150, 400)

    # Synthetic test orientations (roll_deg, pitch_deg, yaw_deg)
    test_cases = [
        {"name": "Moderate 3D Tilt", "roll": 10.0, "pitch": 15.0, "yaw": 20.0},
        {"name": "Severe 3D Tilt", "roll": 20.0, "pitch": 30.0, "yaw": 45.0},
        {"name": "Pure Yaw Rotation", "roll": 0.0, "pitch": 0.0, "yaw": 45.0},
        {"name": "Pure Pitch Tilt", "roll": 0.0, "pitch": 30.0, "yaw": 0.0},
        {"name": "Pure Roll Tilt", "roll": 20.0, "pitch": 0.0, "yaw": 0.0},
    ]

    results = []
    plot_comparison_data = None

    for tc in test_cases:
        r_deg, p_deg, y_deg = tc["roll"], tc["pitch"], tc["yaw"]
        # Ground truth rotation matrix mapping vehicle to phone frame
        R_veh_to_phone = compute_rotation_matrix(np.deg2rad(r_deg), np.deg2rad(p_deg), np.deg2rad(y_deg))
        # Inverse: phone to vehicle
        R_phone_to_veh_true = R_veh_to_phone.T

        # Create synthetically rotated IMU signals: v_phone = R_v_to_p @ v_veh
        imu_rotated = phone_imu.copy()
        imu_rotated[:, :3] = (R_veh_to_phone @ phone_imu[:, :3].T).T
        imu_rotated[:, 3:6] = (R_veh_to_phone @ phone_imu[:, 3:6].T).T

        # 1. Test Stationary Observability (Requirement: ALIGNMENT_UNCERTAIN)
        aligner = PhoneToVehicleAligner()
        stat_acc = imu_rotated[stat_slice, :3]
        aligner.update_gravity_estimate(stat_acc)
        status_after_stat = aligner.status

        # 2. Test Motion Observability (Transition to ALIGNED)
        motion_acc = imu_rotated[motion_slice, :3]
        aligner.update_forward_motion(motion_acc)
        status_after_motion = aligner.status

        # 3. Recovered Rotation & Euler angles
        R_est = aligner.R_phone_to_vehicle
        est_r, est_p, est_y = aligner.get_euler_angles()

        # Rotation error metric: Frobenius norm and angle errors
        frob_err = float(np.linalg.norm(R_est - R_phone_to_veh_true))
        r_err = float(abs(est_r - (-r_deg)))
        p_err = float(abs(est_p - (-p_deg)))
        y_err = float(abs(est_y - (-y_deg)))

        # 4. Trajectory Drift Comparison: WITHOUT vs WITH Alignment during 30s blackout
        eval_slice = slice(400, 700)  # 30 seconds
        eval_len = 300
        ref_lat, ref_lon = gt_coords[400, 0], gt_coords[400, 1]

        # Case A: WITHOUT Alignment (naive IMU integration)
        fusion_no_align = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
        traj_no_align = []
        for k in range(400, 700):
            # Feeding unaligned phone IMU directly as forward accel
            fwd_a = float(imu_rotated[k, 0])
            yaw_r = float(imu_rotated[k, 5])
            st = fusion_no_align.step(fwd_accel=fwd_a, yaw_rate=yaw_r, gnss_pos=None, use_nhc=True)
            traj_no_align.append(st[:2])

        # Case B: WITH Alignment (rotate phone IMU to vehicle frame)
        fusion_aligned = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
        traj_aligned = []
        for k in range(400, 700):
            acc_v, gyro_v = aligner.transform_imu(imu_rotated[k:k+1, :3], imu_rotated[k:k+1, 3:6])
            fwd_a = float(acc_v[0, 0])
            yaw_r = float(gyro_v[0, 2])
            st = fusion_aligned.step(fwd_accel=fwd_a, yaw_rate=yaw_r, gnss_pos=None, use_nhc=True)
            traj_aligned.append(st[:2])

        # GT trajectory in local ENU
        lat_rad = np.deg2rad(ref_lat)
        R_m = 6378137.0
        gt_e = np.deg2rad(gt_coords[400:700, 1] - ref_lon) * R_m * np.cos(lat_rad)
        gt_n = np.deg2rad(gt_coords[400:700, 0] - ref_lat) * R_m
        gt_xy = np.column_stack([gt_e, gt_n])

        drift_no_align = float(np.linalg.norm(traj_no_align[-1] - gt_xy[-1]))
        drift_aligned = float(np.linalg.norm(traj_aligned[-1] - gt_xy[-1]))
        dist_travelled = float(np.sum(v_speed[400:700]) * dt)

        tc_metrics = {
            "case_name": tc["name"],
            "true_angles_deg": {"roll": r_deg, "pitch": p_deg, "yaw": y_deg},
            "status_stationary": status_after_stat.value,
            "status_motion": status_after_motion.value,
            "recovered_frobenius_norm_err": round(frob_err, 4),
            "drift_without_alignment_m": round(drift_no_align, 1),
            "drift_with_alignment_m": round(drift_aligned, 1),
            "drift_reduction_pct": round((1.0 - drift_aligned / max(1e-2, drift_no_align)) * 100.0, 1),
            "drift_pct_with_alignment": round((drift_aligned / max(1.0, dist_travelled)) * 100.0, 2),
        }
        results.append(tc_metrics)
        print(f"  [{tc['name']}]: Drift Without={drift_no_align:.1f}m -> With={drift_aligned:.1f}m "
              f"({tc_metrics['drift_reduction_pct']}% reduction) | Status: {status_after_stat} -> {status_after_motion}")

        if plot_comparison_data is None and tc["name"] == "Severe 3D Tilt":
            plot_comparison_data = {
                "gt_xy": gt_xy,
                "traj_no_align": np.array(traj_no_align),
                "traj_aligned": np.array(traj_aligned),
                "t": t[400:700] - t[400],
            }

    out_json = RESULTS_DIR / "alignment_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Alignment benchmark metrics saved to {out_json}")

    # Generate Publication Plots
    if plot_comparison_data is not None:
        d = plot_comparison_data
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: 2D Trajectory comparison
        axes[0].plot(d["gt_xy"][:, 0], d["gt_xy"][:, 1], "k-", label="Ground Truth Path", linewidth=2.5)
        axes[0].plot(d["traj_no_align"][:, 0], d["traj_no_align"][:, 1], "r--", label="Without Alignment (Corrupted Frame)", linewidth=2.0)
        axes[0].plot(d["traj_aligned"][:, 0], d["traj_aligned"][:, 1], "b-", label="With Alignment (Calibrated Frame)", linewidth=2.0)
        axes[0].plot(d["gt_xy"][0, 0], d["gt_xy"][0, 1], "go", markersize=8, label="Blackout Start")
        axes[0].set_xlabel("East Position (m)", fontsize=11, fontweight="bold")
        axes[0].set_ylabel("North Position (m)", fontsize=11, fontweight="bold")
        axes[0].set_title("Requirement 7: Trajectory Drift With vs Without Alignment", fontsize=12, fontweight="bold")
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].axis("equal")
        axes[0].legend(loc="upper left")

        # Plot 2: Position Error vs Time
        err_no_align = np.linalg.norm(d["traj_no_align"] - d["gt_xy"], axis=1)
        err_aligned = np.linalg.norm(d["traj_aligned"] - d["gt_xy"], axis=1)
        axes[1].plot(d["t"], err_no_align, "r--", label="Without Alignment Error", linewidth=2.0)
        axes[1].plot(d["t"], err_aligned, "b-", label="With Alignment Error", linewidth=2.0)
        axes[1].set_xlabel("Blackout Duration (s)", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Position Error (m)", fontsize=11, fontweight="bold")
        axes[1].set_title("Position Error Over 30s Outage (Severe 3D Tilt)", fontsize=12, fontweight="bold")
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(loc="upper left")

        plot_path = RESULTS_DIR / "plots" / "alignment_evaluation.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[+] Alignment evaluation plot saved to {plot_path}")


if __name__ == "__main__":
    run_alignment_benchmark("Vfa01")
