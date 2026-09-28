"""Phase 4: Validation Ablation Study on Vfa01 (VALIDATION ONLY).

Evaluates the 5 exact configurations specified in the engineering task:
A. AI only
B. AI + adaptive NHC
C. AI + causal map
D. AI + adaptive NHC + gated map
E. corrected alignment + AI + adaptive NHC + gated map

Reports:
- Median Drift (%)
- Mean Drift (%)
- P90 Drift (%)
- Pass Rate (<10%)
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
from src.idr.calib.alignment import PhoneToVehicleAligner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT_DIR / "results" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_vfa01_ablation():
    set_seed(42)
    logger.info("Starting Phase 4: Validation Ablation Study on Vfa01...")

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

    # Pre-blackout GNSS velocity
    gnss_vel = np.zeros((len(t), 2))
    gnss_vel[1:, 0] = gt_diff_e / dt
    gnss_vel[1:, 1] = gt_diff_n / dt
    gnss_vel[0] = gnss_vel[1]

    # Load offline OSM graph for Vfa01
    osm_loader = OSMGraphLoader(cache_dir=ROOT_DIR / "data" / "osm")
    road_graph = osm_loader.load_offline_graph("Vfa01", ref_lat=ref_lat, ref_lon=ref_lon)

    # Load high-speed retrained InertialOdomNet
    model_path = ROOT_DIR / "models" / "inertial_odom.pt"
    odom_model = InertialOdomNet(in_channels=6, window_size=50, hidden_dim=128)
    odom_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    odom_model.eval()

    # Calibrate Corrected Aligner (Phase 2)
    corrected_aligner = PhoneToVehicleAligner()
    corrected_aligner.estimate_attitude(phone_imu[:500, :3], phone_imu[:500, 3:], gnss_vel=gnss_vel[:500])

    # Standard scenarios on Vfa01
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
                "gt_seg": gt_xy[start_idx:end_idx],
                "gt_h_seg": gt_heading[start_idx:end_idx],
                "v0": float(seg_speeds[0]),
                "p0": gt_xy[start_idx].copy(),
                "psi0": float(gt_heading[start_idx]),
            })
            count += 1

    logger.info(f"Loaded {len(scenarios)} validation scenarios.")

    # 5 Ablation Configurations:
    # A. AI only (no NHC, no map)
    # B. AI + adaptive NHC (no map)
    # C. AI + causal map (ungated map, no NHC)
    # D. AI + adaptive NHC + gated map
    # E. corrected alignment + AI + adaptive NHC + gated map

    config_names = [
        "A. AI only",
        "B. AI + adaptive NHC",
        "C. AI + causal map",
        "D. AI + adaptive NHC + gated map",
        "E. Corrected alignment + AI + adaptive NHC + gated map",
    ]

    ablation_drifts = {cfg: [] for cfg in config_names}

    for sc in scenarios:
        s_idx, e_idx = sc["start_idx"], sc["end_idx"]
        N = e_idx - s_idx
        dist_gt = sc["dist_gt"]
        gt_end = sc["gt_seg"][-1]
        p0, v0, psi0 = sc["p0"], sc["v0"], sc["psi0"]

        raw_imu = phone_imu[s_idx:e_idx]
        corr_acc, corr_gyro = corrected_aligner.transform_imu(raw_imu[:, :3], raw_imu[:, 3:])

        # AI inference
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
            ai_dx = out[:, 0]
            ai_sig_x = np.exp(0.5 * out[:, 2])

        # Helper to run EKF
        def run_filter(acc_in, gyro_in, use_nhc=True):
            ekf = ExtendedKalmanFilter(dt=dt)
            ekf.x[0:2] = p0
            ekf.x[3] = v0 * np.cos(psi0)
            ekf.x[4] = v0 * np.sin(psi0)
            ekf.x[6] = psi0
            vfilter = VibrationMotionFilter(window_size=10, dt=dt)
            v_track = v0
            traj = np.zeros((N, 2), dtype=np.float64)
            traj[0] = p0

            for i in range(1, N):
                s_acc, s_gyro, _, cov_scale = vfilter.process(acc_in[i], gyro_in[i])
                fwd_a = float(s_acc[0])
                w_z = float(s_gyro[2])
                ekf.predict(fwd_a, w_z)

                if use_nhc:
                    apply_adaptive_nhc_update(ekf, w_z, sigma_lat_base=0.05 * np.sqrt(cov_scale), k_turn=0.5)

                v_ai = max(0.0, float(ai_dx[i]) / (50.0 * dt))
                v_track = 0.85 * (v_track + fwd_a * dt) + 0.15 * v_ai
                ekf.update_velocity(v_track, R_speed=max(0.1, float(ai_sig_x[i]) * np.sqrt(cov_scale)))
                traj[i] = ekf.x[0:2]

            return traj

        # Raw IMU runs
        traj_ai_only = run_filter(raw_imu[:, :3], raw_imu[:, 3:], use_nhc=False)
        traj_ai_nhc = run_filter(raw_imu[:, :3], raw_imu[:, 3:], use_nhc=True)

        # Corrected aligner run
        traj_corr_ai_nhc = run_filter(corr_acc, corr_gyro, use_nhc=True)

        # Map matchers:
        # Ungated (inf) for Config C
        matcher_ungated = CausalHMMMapMatcher(road_graph, sigma_z=8.0, beta=4.0, max_search_radius=60.0, lag_steps=15, max_distance_gate=1000.0)
        traj_map_c = matcher_ungated.match_trajectory_causal(traj_ai_only)

        # Gated (30m) for Config D & E
        matcher_gated = CausalHMMMapMatcher(road_graph, sigma_z=8.0, beta=4.0, max_search_radius=60.0, lag_steps=15, max_distance_gate=30.0, min_road_confidence=0.20)
        traj_map_d = matcher_gated.match_trajectory_causal(traj_ai_nhc)
        traj_map_e = matcher_gated.match_trajectory_causal(traj_corr_ai_nhc)

        for cfg, traj in zip(
            config_names,
            [traj_ai_only, traj_ai_nhc, traj_map_c, traj_map_d, traj_map_e]
        ):
            err = float(np.linalg.norm(traj[-1] - gt_end))
            drift = (err / dist_gt) * 100.0
            ablation_drifts[cfg].append(drift)

    rows = []
    for cfg in config_names:
        dr = np.array(ablation_drifts[cfg])
        rows.append({
            "Configuration": cfg,
            "Median Drift": f"{np.median(dr):.2f}%",
            "Mean Drift": f"{np.mean(dr):.2f}%",
            "P90": f"{np.percentile(dr, 90):.2f}%",
            "Pass <10%": f"{np.mean(dr < 10.0) * 100:.1f}%",
            "med_num": np.median(dr),
            "mean_num": np.mean(dr),
            "p90_num": np.percentile(dr, 90),
            "pass_num": np.mean(dr < 10.0) * 100,
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "vfa01_ablation_study_results.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 80)
    print("PHASE 4: VALIDATION ABLATION STUDY (Vfa01)")
    print("=" * 80)
    print(df[["Configuration", "Median Drift", "Mean Drift", "P90", "Pass <10%"]].to_string(index=False))


if __name__ == "__main__":
    run_vfa01_ablation()
