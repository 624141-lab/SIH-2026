"""Phase 8: Causal Map Matching 4-Way Ablation on Vfa01_t220s_d60s.

Compares:
A: AI odometry only
B: AI + NHC
C: AI + Causal Map Matching
D: AI + NHC + Causal Map Matching
Against the Old Retrospective HMM.
Computes Final Error, Drift %, Along-Track Error, Cross-Track Error, and Causal? status.
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
from src.idr.models.inertial_odom import InertialOdomNet
from src.idr.filters.ekf import ExtendedKalmanFilter
from src.idr.filters.nhc import apply_adaptive_nhc_update, apply_nhc_update
from src.idr.filters.vibration_filter import VibrationMotionFilter
from src.idr.mapmatch.osm_graph import OSMGraphLoader
from src.idr.mapmatch.hmm_matcher import HMMMapMatcher
from src.idr.mapmatch.causal_matcher import CausalHMMMapMatcher

RESULTS_DIR = ROOT_DIR / "results" / "forensic_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_causal_ablation():
    set_seed(42)
    print("=" * 70)
    print("PHASE 8: CAUSAL MAP-MATCHING 4-WAY ABLATION (Vfa01_t220s_d60s)")
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

    # Scenario parameters
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

    # Load OSM road network
    osm_loader = OSMGraphLoader(cache_dir=ROOT_DIR / "data" / "osm")
    road_graph = osm_loader.load_offline_graph("Vfa01", ref_lat=ref_lat, ref_lon=ref_lon)
    old_matcher = HMMMapMatcher(road_graph, sigma_z=8.0, beta=4.0, max_search_radius=60.0)
    causal_matcher = CausalHMMMapMatcher(road_graph, sigma_z=8.0, beta=4.0, max_search_radius=60.0, lag_steps=15)

    # Load trained InertialOdomNet
    cand_b_path = ROOT_DIR / "models" / "inertial_odom.pt"
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(cand_b_path, map_location="cpu"))
    odom_model.eval()

    # Pre-extract AI odometry windows
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

    def run_filter_pipeline(use_nhc=True, use_ai=True):
        ekf = ExtendedKalmanFilter(dt=dt)
        ekf.x[0:2] = p0
        ekf.x[3] = v0 * np.cos(psi0)
        ekf.x[4] = v0 * np.sin(psi0)
        ekf.x[6] = psi0

        vfilter = VibrationMotionFilter(window_size=10, dt=dt)
        v_tracking = v0
        traj = np.zeros((N_seg, 2), dtype=np.float64)
        traj[0] = p0

        for i in range(1, N_seg):
            glob_i = start_idx + i
            s_acc, s_gyro, state, cov_scale = vfilter.process(phone_imu[glob_i, :3], phone_imu[glob_i, 3:])
            fwd_acc = float(s_acc[0])
            yaw_rate = float(s_gyro[2])

            ekf.predict(fwd_acc, yaw_rate)

            if use_nhc:
                apply_adaptive_nhc_update(ekf, yaw_rate, sigma_lat_base=0.05 * np.sqrt(cov_scale), k_turn=0.5)

            if use_ai:
                dx_b = float(ai_displacements[i, 0])
                v_ai = max(0.0, dx_b / (50.0 * dt))
                sig_x = float(np.exp(0.5 * ai_log_vars[i, 0]))
                v_tracking = 0.85 * (v_tracking + fwd_acc * dt) + 0.15 * v_ai
                ekf.update_velocity(v_tracking, R_speed=max(0.1, sig_x * np.sqrt(cov_scale)))
            else:
                v_tracking = max(0.0, v_tracking + fwd_acc * dt)
                ekf.update_velocity(v_tracking, R_speed=0.5)

            traj[i] = ekf.x[0:2]

        return traj

    # Generate trajectories
    traj_A = run_filter_pipeline(use_nhc=False, use_ai=True)  # AI only
    traj_B = run_filter_pipeline(use_nhc=True, use_ai=True)   # AI + NHC

    # Map matched trajectories
    # Causal Zero-Lag (Look-ahead = 0s)
    traj_C_causal = causal_matcher.match_trajectory_causal(traj_A, use_finalized=False)
    traj_D_causal = causal_matcher.match_trajectory_causal(traj_B, use_finalized=False)

    # Old Retrospective (Non-causal)
    traj_C_old = old_matcher.match_trajectory(traj_A)
    traj_D_old = old_matcher.match_trajectory(traj_B)

    # Unit vectors for along-track and cross-track decomposition
    # Final GT heading
    psi_gt_final = gt_seg_heading[-1]
    u_along = np.array([np.cos(psi_gt_final), np.sin(psi_gt_final)])
    u_cross = np.array([-np.sin(psi_gt_final), np.cos(psi_gt_final)])

    def evaluate_errors(pred_traj):
        err_vec = pred_traj[-1] - gt_seg_xy[-1]
        final_err = float(np.linalg.norm(err_vec))
        drift = (final_err / dist_gt) * 100.0
        along_track = float(np.dot(err_vec, u_along))
        cross_track = float(np.dot(err_vec, u_cross))
        return round(final_err, 2), round(drift, 2), round(abs(along_track), 2), round(abs(cross_track), 2)

    configs = [
        ("Config A: AI Odometry Only", traj_A, "YES (Causal Filter)"),
        ("Config B: AI + Adaptive NHC", traj_B, "YES (Causal Filter)"),
        ("Config C: AI + Causal Map (Zero-Lag)", traj_C_causal, "YES (Causal Online)"),
        ("Config D: AI + NHC + Causal Map (Zero-Lag)", traj_D_causal, "YES (Causal Online)"),
        ("Reference: AI + NHC + Old Retrospective HMM", traj_D_old, "NO (Retrospective Batch)"),
    ]

    ablation_rows = []
    for name, tr, causal_status in configs:
        f_err, drift, along, cross = evaluate_errors(tr)
        ablation_rows.append({
            "Configuration": name,
            "Final Error (m)": f_err,
            "Drift (%)": drift,
            "Along-Track Error (m)": along,
            "Cross-Track Error (m)": cross,
            "Causal?": causal_status,
        })
        print(f"{name:45s} | Error: {f_err:5.2f}m | Drift: {drift:4.2f}% | Cross-Track: {cross:4.2f}m | {causal_status}")

    df_ablation = pd.DataFrame(ablation_rows)
    csv_path = RESULTS_DIR / "causal_map_matching_ablation.csv"
    df_ablation.to_csv(csv_path, index=False)
    print(f"\n[+] Causal ablation saved to {csv_path}")
    return df_ablation


if __name__ == "__main__":
    run_causal_ablation()
