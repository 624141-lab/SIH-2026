"""Requirements 8 & 9 Evaluation: Non-Holonomic Constraints (NHC) and HMM Map-Matching.

Benchmarks:
1. NHC (Requirement 9):
   - Mathematically verifies v_lat = -sin(ψ)*vE + cos(ψ)*vN ≈ 0.
   - Evaluates lateral velocity before vs after NHC update.
   - Chi-square innovation gating (NIS).
   - Quantifies cross-track drift reduction from NHC heading observability.

2. HMM Map Matching (Requirement 8):
   - Evaluates independent OSM road graph matching.
   - Measures cross-track error before vs after map matching.
   - Measures candidate generation, road likelihood, route continuity, and latency.

Generates quantitative metrics and publication-grade verification plots.
"""

import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.idr.config import RAW_DATA_DIR, RESULTS_DIR
from src.idr.io.loader import load_drive_pair
from src.idr.filters.fusion import GNSSINSFusion
from src.idr.filters.ekf import ExtendedKalmanFilter
from src.idr.filters.nhc import apply_nhc_update, compute_lateral_velocity
from src.idr.mapmatch.osm_graph import OSMGraphLoader
from src.idr.mapmatch.hmm_matcher import HMMMapMatcher


def evaluate_nhc_and_mapmatch(drive_id: str = "Vfa01"):
    print(f"[*] Loading drive {drive_id} for NHC and Map Matching evaluation...")
    drive = load_drive_pair(RAW_DATA_DIR, drive_id)
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1
    N = min(len(phone_imu), len(gt_coords))

    # Evaluate on a 60-second blackout segment (600 steps)
    start_step = 400
    end_step = 1000
    eval_len = end_step - start_step

    ref_lat, ref_lon = gt_coords[start_step, 0], gt_coords[start_step, 1]

    # Convert Ground Truth to local ENU
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    gt_e = np.deg2rad(gt_coords[start_step:end_step, 1] - ref_lon) * R_m * np.cos(lat_rad)
    gt_n = np.deg2rad(gt_coords[start_step:end_step, 0] - ref_lat) * R_m
    gt_xy = np.column_stack([gt_e, gt_n])
    dist_travelled = float(np.sum(v_speed[start_step:end_step]) * dt)

    # -------------------------------------------------------------
    # 1. EVALUATE NHC (Requirement 9)
    # -------------------------------------------------------------
    print("[*] Running NHC mathematical verification...")
    # System A: EKF WITHOUT NHC
    ekf_no_nhc = ExtendedKalmanFilter(dt=dt)
    ekf_no_nhc.x[3] = v_speed[start_step] * np.cos(0.0)
    ekf_no_nhc.x[4] = v_speed[start_step] * np.sin(0.0)

    # System B: EKF WITH NHC
    ekf_with_nhc = ExtendedKalmanFilter(dt=dt)
    ekf_with_nhc.x[3] = v_speed[start_step] * np.cos(0.0)
    ekf_with_nhc.x[4] = v_speed[start_step] * np.sin(0.0)

    v_lat_no_nhc = []
    v_lat_before_nhc = []
    v_lat_after_nhc = []
    gated_count = 0

    traj_no_nhc = []
    traj_with_nhc = []

    for k in range(start_step, end_step):
        ax = float(phone_imu[k, 0])
        wz = float(phone_imu[k, 5])

        # A: Predict without NHC
        ekf_no_nhc.predict(ax, wz)
        # Add slight speed anchor to isolate heading/lateral drift
        ekf_no_nhc.update_velocity(float(v_speed[k]), R_speed=0.5)
        v_lat_no_nhc.append(compute_lateral_velocity(ekf_no_nhc))
        traj_no_nhc.append(ekf_no_nhc.x[:2].copy())

        # B: Predict with NHC
        ekf_with_nhc.predict(ax, wz)
        ekf_with_nhc.update_velocity(float(v_speed[k]), R_speed=0.5)
        passed, v_before, v_after = apply_nhc_update(ekf_with_nhc, sigma_lat=0.05, sigma_vert=0.05)
        if not passed:
            gated_count += 1
        v_lat_before_nhc.append(v_before)
        v_lat_after_nhc.append(v_after)
        traj_with_nhc.append(ekf_with_nhc.x[:2].copy())

    traj_no_nhc = np.array(traj_no_nhc)
    traj_with_nhc = np.array(traj_with_nhc)

    mean_v_lat_no_nhc = float(np.mean(np.abs(v_lat_no_nhc)))
    mean_v_lat_after_nhc = float(np.mean(np.abs(v_lat_after_nhc)))
    max_v_lat_no_nhc = float(np.max(np.abs(v_lat_no_nhc)))
    max_v_lat_after_nhc = float(np.max(np.abs(v_lat_after_nhc)))

    # Cross-track drift relative to GT
    drift_no_nhc = float(np.linalg.norm(traj_no_nhc[-1] - gt_xy[-1]))
    drift_with_nhc = float(np.linalg.norm(traj_with_nhc[-1] - gt_xy[-1]))
    nhc_drift_reduction = float((1.0 - drift_with_nhc / max(1e-2, drift_no_nhc)) * 100.0)

    # -------------------------------------------------------------
    # 2. EVALUATE HMM MAP MATCHING (Requirement 8)
    # -------------------------------------------------------------
    print("[*] Running HMM Map Matching verification on independent OSM graph...")
    osm_loader = OSMGraphLoader()
    graph = osm_loader.load_offline_graph(drive_id, ref_lat, ref_lon)
    matcher = HMMMapMatcher(graph=graph, sigma_z=8.0, beta=4.0, max_search_radius=60.0)

    # Run HMM on the dead-reckoned trajectory (traj_with_nhc)
    t0_hmm = time.perf_counter()
    matched_traj = matcher.match_trajectory(traj_with_nhc[:, :2])
    total_hmm_time = time.perf_counter() - t0_hmm
    mean_hmm_latency_ms = (total_hmm_time / eval_len) * 1000.0

    matched_traj = np.array(matched_traj)

    # Cross-track errors before vs after map matching relative to GT path
    # For each point, find distance to nearest GT road segment
    def compute_cross_track_errors(path, ref_path):
        errs = []
        for p in path:
            dists = np.linalg.norm(ref_path - p, axis=1)
            errs.append(np.min(dists))
        return np.array(errs)

    xerr_before_mm = compute_cross_track_errors(traj_with_nhc, gt_xy)
    xerr_after_mm = compute_cross_track_errors(matched_traj, gt_xy)

    mean_xerr_before = float(np.mean(xerr_before_mm))
    mean_xerr_after = float(np.mean(xerr_after_mm))
    max_xerr_before = float(np.max(xerr_before_mm))
    max_xerr_after = float(np.max(xerr_after_mm))
    xerr_reduction_pct = float((1.0 - mean_xerr_after / max(1e-2, mean_xerr_before)) * 100.0)

    final_drift_before_mm = float(np.linalg.norm(traj_with_nhc[-1] - gt_xy[-1]))
    final_drift_after_mm = float(np.linalg.norm(matched_traj[-1] - gt_xy[-1]))
    final_drift_pct = (final_drift_after_mm / max(1.0, dist_travelled)) * 100.0

    metrics = {
        "drive_id": drive_id,
        "evaluation_duration_s": eval_len * dt,
        "distance_travelled_m": round(dist_travelled, 1),
        "nhc_evaluation": {
            "mean_v_lat_no_nhc_mps": round(mean_v_lat_no_nhc, 3),
            "mean_v_lat_after_nhc_mps": round(mean_v_lat_after_nhc, 3),
            "max_v_lat_no_nhc_mps": round(max_v_lat_no_nhc, 3),
            "max_v_lat_after_nhc_mps": round(max_v_lat_after_nhc, 3),
            "v_lat_suppression_pct": round((1.0 - mean_v_lat_after_nhc / max(1e-4, mean_v_lat_no_nhc)) * 100.0, 1),
            "chi2_gating_events": gated_count,
            "drift_without_nhc_m": round(drift_no_nhc, 1),
            "drift_with_nhc_m": round(drift_with_nhc, 1),
            "nhc_drift_reduction_pct": round(nhc_drift_reduction, 1),
        },
        "map_matching_evaluation": {
            "mean_cross_track_error_before_mm_m": round(mean_xerr_before, 2),
            "mean_cross_track_error_after_mm_m": round(mean_xerr_after, 2),
            "max_cross_track_error_before_mm_m": round(max_xerr_before, 2),
            "max_cross_track_error_after_mm_m": round(max_xerr_after, 2),
            "cross_track_reduction_pct": round(xerr_reduction_pct, 1),
            "final_position_drift_after_mm_m": round(final_drift_after_mm, 1),
            "final_drift_pct_after_mm": round(final_drift_pct, 2),
            "mean_hmm_step_latency_ms": round(mean_hmm_latency_ms, 2),
        },
    }

    out_json = RESULTS_DIR / "nhc_mapmatch_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] NHC & Map Matching metrics saved to {out_json}")
    print(json.dumps(metrics, indent=2))

    # -------------------------------------------------------------
    # 3. GENERATE PUBLICATION-GRADE PLOTS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(11, 12))
    plt.subplots_adjust(hspace=0.32)
    t_eval = t[start_step:end_step] - t[start_step]

    # Plot 1: Lateral Velocity vs Time (Requirement 9)
    axes[0].plot(t_eval, v_lat_no_nhc, "r--", label="Without NHC (v_lateral drift unchecked)", linewidth=1.8)
    axes[0].plot(t_eval, v_lat_after_nhc, "b-", label="With NHC (v_lateral ≈ 0 enforced)", linewidth=2.0)
    axes[0].axhline(0.0, color="k", linestyle=":", alpha=0.7)
    axes[0].set_ylabel("Lateral Velocity (m/s)", fontsize=10, fontweight="bold")
    axes[0].set_title(f"Requirement 9: NHC Lateral Slip Suppression (Mean: {mean_v_lat_no_nhc:.2f} -> {mean_v_lat_after_nhc:.3f} m/s)", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Plot 2: Cross-Track Error vs Time (Requirement 8)
    axes[1].plot(t_eval, xerr_before_mm, "r--", label=f"Before Map Matching (Mean: {mean_xerr_before:.1f}m)", linewidth=1.8)
    axes[1].plot(t_eval, xerr_after_mm, "g-", label=f"After HMM Map Matching (Mean: {mean_xerr_after:.1f}m)", linewidth=2.0)
    axes[1].set_ylabel("Cross-Track Error (m)", fontsize=10, fontweight="bold")
    axes[1].set_title(f"Requirement 8: Road Cross-Track Error Reduction ({xerr_reduction_pct:.1f}% reduction)", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")

    # Plot 3: 2D Spatial Trajectory Comparison on Road Network
    axes[2].plot(gt_xy[:, 0], gt_xy[:, 1], "k-", label="Ground Truth Vehicle Path", linewidth=2.5)
    axes[2].plot(traj_no_nhc[:, 0], traj_no_nhc[:, 1], "r:", label="Dead Reckoning (No NHC)", linewidth=1.5)
    axes[2].plot(traj_with_nhc[:, 0], traj_with_nhc[:, 1], "b--", label="Dead Reckoning (With NHC)", linewidth=1.8)
    axes[2].plot(matched_traj[:, 0], matched_traj[:, 1], "g-", label="HMM Road-Constrained Output", linewidth=2.2)
    axes[2].plot(gt_xy[0, 0], gt_xy[0, 1], "go", markersize=8, label="Blackout Start")
    axes[2].set_xlabel("East (m)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("North (m)", fontsize=10, fontweight="bold")
    axes[2].set_title("2D Trajectory: Raw INS vs NHC vs HMM Road Network Constraint", fontsize=11, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].axis("equal")
    axes[2].legend(loc="upper left")

    plot_path = RESULTS_DIR / "plots" / "nhc_mapmatch_evaluation.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] NHC & Map Matching evaluation plot saved to {plot_path}")


if __name__ == "__main__":
    evaluate_nhc_and_mapmatch("Vfa01")
