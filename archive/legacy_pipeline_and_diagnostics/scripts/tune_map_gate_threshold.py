"""Tune and validate Map Matching Out-of-Corridor Gating Threshold strictly on Vfa01 (VALIDATION).

Tests candidate thresholds:
10m, 15m, 20m, 25m, 30m, 40m, 50m, and ungated (inf).
Selects the threshold that optimizes lateral lane-keeping while preventing off-corridor snapping.
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
from src.idr.mapmatch.osm_graph import OSMGraphLoader
from src.idr.mapmatch.causal_matcher import CausalHMMMapMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_gate_thresholds():
    set_seed(42)
    logger.info("Evaluating Map Gating Thresholds on Vfa01 (VALIDATION ONLY)...")

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

    osm_loader = OSMGraphLoader(cache_dir=ROOT_DIR / "data" / "osm")
    road_graph = osm_loader.load_offline_graph("Vfa01", ref_lat=ref_lat, ref_lon=ref_lon)

    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(ROOT_DIR / "models" / "inertial_odom.pt", map_location="cpu"))
    odom_model.eval()

    # Pre-generate unconstrained dead-reckoning trajectories (System 5: AI + Adaptive NHC) for Vfa01 scenarios
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

            N = end_idx - start_idx
            gt_seg = gt_xy[start_idx:end_idx]
            gt_h_seg = gt_heading[start_idx:end_idx]
            p0 = gt_seg[0].copy()
            v0 = float(seg_speeds[0])
            psi0 = float(gt_h_seg[0])

            # AI windows
            windows = np.zeros((N, 6, 50), dtype=np.float32)
            for i in range(N):
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

            # Run EKF + Adaptive NHC (raw phone frame / nominal)
            ekf = ExtendedKalmanFilter(dt=dt)
            ekf.x[0:2] = p0
            ekf.x[3] = v0 * np.cos(psi0)
            ekf.x[4] = v0 * np.sin(psi0)
            ekf.x[6] = psi0
            vfilter = VibrationMotionFilter(window_size=10, dt=dt)
            v_track = v0
            traj_s5 = np.zeros((N, 2), dtype=np.float64)
            traj_s5[0] = p0

            for i in range(1, N):
                s_acc, s_gyro, state, cov_scale = vfilter.process(phone_imu[start_idx + i, :3], phone_imu[start_idx + i, 3:])
                fwd_acc = float(s_acc[0])
                yaw_rate = float(s_gyro[2])
                ekf.predict(fwd_acc, yaw_rate)
                apply_adaptive_nhc_update(ekf, yaw_rate, sigma_lat_base=0.05 * np.sqrt(cov_scale), k_turn=0.5)

                dx_b = float(ai_displacements[i, 0])
                v_ai = max(0.0, dx_b / (50.0 * dt))
                sig_x = float(np.exp(0.5 * ai_log_vars[i, 0]))
                v_track = 0.85 * (v_track + fwd_acc * dt) + 0.15 * v_ai
                ekf.update_velocity(v_track, R_speed=max(0.1, sig_x * np.sqrt(cov_scale)))
                traj_s5[i] = ekf.x[0:2]

            scenarios.append({
                "scenario_id": f"Vfa01_t{int(start_idx*dt)}s_d{dur}s",
                "gt_xy": gt_seg,
                "gt_heading": gt_h_seg,
                "dist_gt": dist_gt,
                "traj_s5": traj_s5,
            })
            count += 1

    logger.info(f"Generated {len(scenarios)} validation scenario dead-reckoning trajectories.")

    # Candidate gate thresholds to test
    thresholds = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 1000.0]
    tuning_results = []

    for gate_d in thresholds:
        logger.info(f"Testing max_distance_gate = {gate_d} m...")
        # Configure causal matcher with candidate threshold
        matcher = CausalHMMMapMatcher(
            road_graph,
            sigma_z=8.0,
            beta=4.0,
            max_search_radius=60.0,
            lag_steps=15,
            max_distance_gate=gate_d,
            min_road_confidence=0.20,
        )

        drifts = []
        errors = []
        crosses = []
        rejections = 0
        total_steps = 0

        for sc in scenarios:
            pred_traj, stats = matcher.match_trajectory_causal_with_stats(sc["traj_s5"])
            err_vec = pred_traj[-1] - sc["gt_xy"][-1]
            pos_err = float(np.linalg.norm(err_vec))
            drift = (pos_err / sc["dist_gt"]) * 100.0

            psi_f = sc["gt_heading"][-1]
            u_cross = np.array([-np.sin(psi_f), np.cos(psi_f)])
            cross_err = float(np.abs(np.dot(err_vec, u_cross)))

            drifts.append(drift)
            errors.append(pos_err)
            crosses.append(cross_err)
            rejections += stats.get("rejected_steps", 0)
            total_steps += len(sc["traj_s5"])

        drifts = np.array(drifts)
        errors = np.array(errors)
        crosses = np.array(crosses)

        tuning_results.append({
            "Gate_Threshold_m": "Ungated (inf)" if gate_d >= 999.0 else f"{gate_d:.0f} m",
            "Threshold_Value": gate_d,
            "Median_Drift_pct": round(float(np.median(drifts)), 2),
            "Mean_Drift_pct": round(float(np.mean(drifts)), 2),
            "P90_Drift_pct": round(float(np.percentile(drifts, 90)), 2),
            "Mean_Final_Error_m": round(float(np.mean(errors)), 2),
            "Mean_Cross_Track_m": round(float(np.mean(crosses)), 2),
            "Pass_Rate_10pct": f"{float(np.mean(drifts < 10.0)) * 100:.1f}%",
            "Within_15pct": f"{float(np.mean(drifts <= 15.0)) * 100:.1f}%",
            "Within_20pct": f"{float(np.mean(drifts <= 20.0)) * 100:.1f}%",
            "Rejection_Rate_pct": f"{rejections / total_steps * 100:.1f}%",
        })

    df_tune = pd.DataFrame(tuning_results)
    csv_out = RESULTS_DIR / "map_gate_threshold_tuning_vfa01.csv"
    df_tune.to_csv(csv_out, index=False)
    logger.info(f"[+] Tuning table saved to {csv_out}")
    print("\n--- MAP GATE THRESHOLD TUNING (Vfa01 VALIDATION) ---")
    print(df_tune.to_string())


if __name__ == "__main__":
    evaluate_gate_thresholds()
