"""Phase 10 & 11: Final Evaluation of Heading-Hardened Pipeline on Untouched Vfa02.

Evaluates the complete heading-hardened pipeline on all 24 untouched Vfa02 scenarios:
- Full 3D rotation omega_v = R_phone_to_vehicle @ omega_phone
- Explicit EKF gyro bias state propagation
- Causal pre-blackout bias initialization (robust straight-line COG gating)
- Stationary ZARU / ZUPT updates during stops
- Vibration motion filtering & Adaptive NHC
- Safe 30m gated causal map matching
- Error decomposition: Along-Track vs Cross-Track, Heading RMSE
- Category breakdowns:
  * Straight roads (< 15 deg heading change)
  * Moderate maneuvers (15 - 45 deg heading change)
  * Complex turns (> 45 deg heading change)
  * Mapped corridors (within 30m of OSM road)
  * Unmapped corridors (outside 30m)
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
from src.idr.mapmatch.osm_graph import OSMGraphLoader
from src.idr.mapmatch.causal_matcher import CausalHMMMapMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "heading_hardened"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_heading_hardened_vfa02():
    set_seed(42)
    logger.info("Executing Phase 10: Untouched Vfa02 Final Evaluation...")

    data_dir = ROOT_DIR / "data" / "raw"
    drive = load_drive_pair(data_dir, "Vfa02")
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1
    N_total = len(t)

    # Coordinates in ENU
    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
    lat_rad = np.deg2rad(ref_lat)
    R_m = 6378137.0
    gt_e = np.deg2rad(gt_coords[:, 1] - ref_lon) * R_m * np.cos(lat_rad)
    gt_n = np.deg2rad(gt_coords[:, 0] - ref_lat) * R_m
    gt_xy = np.column_stack([gt_e, gt_n])

    vel_e = np.zeros_like(gt_e)
    vel_n = np.zeros_like(gt_n)
    vel_e[1:] = np.diff(gt_e) / dt
    vel_n[1:] = np.diff(gt_n) / dt

    # Ground truth heading from coordinate diffs
    gt_heading = np.zeros(N_total)
    gt_heading[1:] = np.arctan2(vel_n[1:], vel_e[1:])
    gt_heading[0] = gt_heading[1]

    # Pre-blackout GNSS velocity for alignment
    gnss_vel = np.column_stack([vel_e, vel_n])

    # 1. Aligner (Frozen)
    aligner = PhoneToVehicleAligner()
    aligner.estimate_attitude(phone_imu[:500, :3], phone_imu[:500, 3:], gnss_vel=gnss_vel[:500])
    corr_acc, corr_gyro = aligner.transform_imu(phone_imu[:, :3], phone_imu[:, 3:])

    # 2. Map Matcher
    osm_loader = OSMGraphLoader(cache_dir=ROOT_DIR / "data" / "osm")
    road_graph = osm_loader.load_offline_graph("Vfa02", ref_lat=ref_lat, ref_lon=ref_lon)
    causal_matcher = CausalHMMMapMatcher(road_graph, max_search_radius=60.0, max_distance_gate=30.0)

    # 3. AI Model
    model_path = ROOT_DIR / "models" / "inertial_odom.pt"
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    odom_model.eval()

    # Define standard 24 scenarios on Vfa02
    durations = [15, 30, 60]
    scenarios = []

    for dur in durations:
        dur_steps = int(dur / dt)
        start_indices = list(range(100, N_total - dur_steps - 20, 200))
        count = 0
        for s_idx in start_indices:
            if count >= 8:
                break
            e_idx = s_idx + dur_steps
            seg_spds = v_speed[s_idx:e_idx]
            dist_gt = float(np.sum(seg_spds) * dt)
            if dist_gt < 50.0 or np.mean(seg_spds) < 3.0:
                continue

            # Heading change in scenario to classify road maneuver
            h_seg = gt_heading[s_idx:e_idx]
            h_diff_total = float(np.abs((h_seg[-1] - h_seg[0] + np.pi) % (2 * np.pi) - np.pi) * (180.0 / np.pi))

            if h_diff_total < 15.0:
                road_type = "straight"
            elif h_diff_total <= 45.0:
                road_type = "moderate_maneuver"
            else:
                road_type = "complex_turn"

            scenarios.append({
                "scenario_id": f"Vfa02_t{int(s_idx*dt)}s_d{dur}s",
                "start_idx": s_idx,
                "end_idx": e_idx,
                "duration_s": dur,
                "dist_gt": dist_gt,
                "road_type": road_type,
                "total_turn_deg": round(h_diff_total, 2),
                "p0": gt_xy[s_idx].copy(),
                "v0": float(seg_spds[0]),
                "psi0": float(gt_heading[s_idx]),
                "gt_end": gt_xy[e_idx - 1].copy(),
                "gt_seg": gt_xy[s_idx:e_idx],
                "gt_h_seg": h_seg,
            })
            count += 1

    logger.info(f"Loaded {len(scenarios)} Vfa02 validation scenarios (Target: 24).")

    # Load baseline metrics for comparison
    baseline_file = ROOT_DIR / "results" / "vfa02_final_hardened_metrics.csv"
    if baseline_file.exists():
        df_base = pd.read_csv(baseline_file)
        base_drift_map = dict(zip(df_base["scenario_id"], df_base["post_drift_pct"]))
        base_err_map = dict(zip(df_base["scenario_id"], df_base["post_map_pos_error_m"]))
    else:
        base_drift_map = {}
        base_err_map = {}

    scenario_results = []

    for sc in scenarios:
        s_idx, e_idx = sc["start_idx"], sc["end_idx"]
        N = e_idx - s_idx
        dist_gt = sc["dist_gt"]
        gt_end = sc["gt_end"]
        p0, v0, psi0 = sc["p0"], sc["v0"], sc["psi0"]
        gt_h_seg = sc["gt_h_seg"]

        # Phase 5: Robust pre-blackout bias initialization (causally available)
        pre_win = 100  # 10s pre-blackout
        pre_s = max(0, s_idx - pre_win)
        w_pre = corr_gyro[pre_s:s_idx, 2]
        cog_pre = np.unwrap(np.arctan2(vel_n[pre_s:s_idx], vel_e[pre_s:s_idx]))
        cog_rate = np.diff(cog_pre) / dt
        spd_pre = np.hypot(vel_e[pre_s:s_idx - 1], vel_n[pre_s:s_idx - 1])
        mask_st = (spd_pre > 4.0) & (np.abs(cog_rate) < 0.03)

        if np.sum(mask_st) >= 15:
            # Gated: vehicle must be driving straight to extract bias cleanly
            raw_b = float(np.median(w_pre[:-1][mask_st] - cog_rate[mask_st]))
            # Physical sanity clamp for smartphone MEMS gyro bias (|b| <= 0.05 rad/s = 2.86 deg/s)
            b_w_init = float(np.clip(raw_b, -0.05, 0.05))
        else:
            b_w_init = 0.0

        # Precompute AI windows
        windows = np.zeros((N, 6, 50), dtype=np.float32)
        for k in range(N):
            glob_i = s_idx + k
            w_start = max(0, glob_i - 50)
            win = phone_imu[w_start:glob_i].T
            if win.shape[1] == 0:
                win = np.zeros((6, 50), dtype=np.float32)
            elif win.shape[1] < 50:
                win = np.pad(win, ((0, 0), (50 - win.shape[1], 0)), mode="edge")
            windows[k] = win

        with torch.no_grad():
            out = odom_model(torch.from_numpy(windows)).cpu().numpy()
            ai_dx = out[:, 0]
            ai_sig = np.exp(0.5 * out[:, 2])

        # Initialize EKF
        ekf = ExtendedKalmanFilter(dt=dt)
        ekf.x[0:2] = p0
        ekf.x[3] = v0 * np.cos(psi0)
        ekf.x[4] = v0 * np.sin(psi0)
        ekf.x[6] = psi0
        ekf.x[8] = b_w_init

        vfilter = VibrationMotionFilter(window_size=10, dt=dt)
        stat_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15, gyro_norm_threshold=0.05)
        v_track = v0

        traj_pre = np.zeros((N, 2))
        traj_pre[0] = p0
        est_headings = np.zeros(N)
        est_headings[0] = psi0

        for k in range(1, N):
            glob_i = s_idx + k
            s_acc, s_gyro, _, cov_scale = vfilter.process(corr_acc[glob_i], corr_gyro[glob_i])
            fwd_a = float(s_acc[0])
            w_z = float(s_gyro[2])
            ekf.predict(fwd_a, w_z)

            # Adaptive NHC
            apply_adaptive_nhc_update(ekf, w_z, sigma_lat_base=0.05 * np.sqrt(cov_scale), k_turn=0.5)

            # Phase 4: Stationary ZARU / ZUPT updates
            if stat_detector.update(phone_imu[glob_i, :3], phone_imu[glob_i, 3:]):
                apply_zupt(ekf, sigma_v=0.01)
                apply_zaru(ekf, gyro_z_raw=w_z, sigma_bias=0.005)

            # AI forward speed update
            v_ai = max(0.0, float(ai_dx[k]) / (50.0 * dt))
            v_track = 0.85 * (v_track + fwd_a * dt) + 0.15 * v_ai
            ekf.update_velocity(v_track, R_speed=max(0.1, float(ai_sig[k]) * np.sqrt(cov_scale)))

            traj_pre[k] = ekf.x[:2]
            est_headings[k] = ekf.x[6]

        # Phase 1: Safe Gated Causal Map Matching
        traj_post, stats = causal_matcher.match_trajectory_causal_with_stats(traj_pre)

        # Metrics computation
        pre_err = float(np.linalg.norm(traj_pre[-1] - gt_end))
        post_err = float(np.linalg.norm(traj_post[-1] - gt_end))
        pre_drift = (pre_err / dist_gt) * 100.0
        post_drift = (post_err / dist_gt) * 100.0

        # Heading error & decomposition
        h_diff = (est_headings - gt_h_seg + np.pi) % (2 * np.pi) - np.pi
        h_rmse = float(np.sqrt(np.mean(h_diff ** 2))) * (180.0 / np.pi)
        final_h_err = float(np.abs(h_diff[-1])) * (180.0 / np.pi)

        psi_f = gt_h_seg[-1]
        u_along = np.array([np.cos(psi_f), np.sin(psi_f)])
        u_cross = np.array([-np.sin(psi_f), np.cos(psi_f)])
        err_vec = traj_post[-1] - gt_end
        along_err = float(np.abs(np.dot(err_vec, u_along)))
        cross_err = float(np.abs(np.dot(err_vec, u_cross)))

        accepted_ratio = stats.get("accepted_ratio", 0.0)
        is_mapped = accepted_ratio > 0.05
        corridor_type = "mapped" if is_mapped else "unmapped"

        scenario_results.append({
            "scenario_id": sc["scenario_id"],
            "duration_s": sc["duration_s"],
            "dist_gt_m": round(dist_gt, 2),
            "road_type": sc["road_type"],
            "corridor_type": corridor_type,
            "total_turn_deg": sc["total_turn_deg"],
            "b_w_init_deg_s": round(float(np.rad2deg(b_w_init)), 3),
            "b_w_final_deg_s": round(float(np.rad2deg(ekf.x[8])), 3),
            "heading_rmse_deg": round(h_rmse, 2),
            "final_heading_err_deg": round(final_h_err, 2),
            "baseline_drift_pct": base_drift_map.get(sc["scenario_id"], np.nan),
            "baseline_pos_err_m": base_err_map.get(sc["scenario_id"], np.nan),
            "hardened_drift_pct": round(post_drift, 2),
            "hardened_pos_err_m": round(post_err, 2),
            "along_track_err_m": round(along_err, 2),
            "cross_track_err_m": round(cross_err, 2),
            "pass_sih_10pct": post_drift < 10.0,
        })

    df_out = pd.DataFrame(scenario_results)
    csv_path = RESULTS_DIR / "vfa02_heading_hardened_metrics.csv"
    df_out.to_csv(csv_path, index=False)

    logger.info("\n=== Vfa02 ALL SCENARIOS COMPARISON ===")
    logger.info(df_out[[
        "scenario_id", "duration_s", "dist_gt_m", "road_type", "heading_rmse_deg",
        "baseline_drift_pct", "hardened_drift_pct", "hardened_pos_err_m", "pass_sih_10pct"
    ]].to_string(index=False))

    return df_out


if __name__ == "__main__":
    evaluate_heading_hardened_vfa02()
