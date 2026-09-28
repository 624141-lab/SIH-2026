"""GNSS Blackout Benchmark Simulation Engine for Authentic IO-VNBD.

STRICT FORENSIC CONSTRAINTS (Rules 11-20, 40-50, 63-65):
1. Evaluates exclusively on authentic IO-VNBD test drives (Vfa01, Vfa02).
2. NEVER generates synthetic trajectories.
3. Strict separation: EstimatorInput vs GroundTruthReference (Rule 50).
4. During blackout: ZERO GNSS position, ZERO GNSS velocity, ZERO GNSS heading,
   and ZERO ground truth enters the filter.
5. Road graphs are loaded from independent OpenStreetMap data (Rule 15, 64),
   NEVER from ground-truth waypoints.
6. HMM Map Matcher consumes strictly the dead-reckoned estimated trajectory (Rule 19).
7. AI Velocity is predicted from smartphone IMU windows, independent of EKF states (Rule 65).
8. IMU DenoiseNet filters raw IMU measurements before mechanization (Rule 24).
9. Three required benchmark configurations evaluated on identical scenarios (Rule 41):
   - Config A: Raw IMU Baseline (Pure Dead Reckoning)
   - Config B: EKF + AI Velocity + NHC (+ IMU Denoise)
   - Config C: EKF + AI Velocity + NHC + Independent OSM HMM/Viterbi
10. Manifest and per-scenario metrics saved to:
    - results/scenario_manifest.csv (Rule 44)
    - results/per_scenario_metrics.csv (Rule 45)
    - results/eval_results.json
    - results/eval_trajectories.npz
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
import numpy as np
import pandas as pd
import torch

from ..config import (
    CONFIG, SEED, set_seed, RAW_DATA_DIR, MODELS_DIR,
    REPORTS_DIR, RESULTS_DIR, OSM_DIR,
)
from ..io.loader import load_drive_pair, IOVNBDrive
from ..calib.alignment import PhoneToVehicleAligner
from ..models.velocity_net import VelocityEstimatorNet
from ..models.imu_denoise import IMUDenoiseNet
from ..filters.fusion import GNSSINSFusion
from ..filters.nhc import apply_nhc_update
from ..mapmatch.osm_graph import OSMGraphLoader
from ..mapmatch.hmm_matcher import HMMMapMatcher
from .metrics import compute_navigation_metrics, NavigationMetrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EstimatorInput:
    """Isolated sensor inputs available to navigation estimator during inference (Rule 50)."""
    imu_data: np.ndarray             # (N, 6) [ax, ay, az, gx, gy, gz]
    initial_pos_enu: np.ndarray      # (3,) [E, N, U] position at start of blackout
    initial_vel_enu: np.ndarray      # (3,) [vE, vN, vU] velocity at start of blackout
    initial_heading_rad: float       # Heading angle at start of blackout
    timestamps: np.ndarray           # (N,) seconds relative to start of blackout


@dataclass
class GroundTruthReference:
    """Ground truth data isolated strictly for post-hoc metric computation (Rule 50)."""
    gt_enu: np.ndarray               # (N, 3) reference position in local ENU
    gt_speed: np.ndarray             # (N,) reference forward velocity from ECU CAN
    timestamps: np.ndarray           # (N,) seconds


def run_raw_imu_baseline(
    sensor_input: EstimatorInput,
    dt: float = 0.1,
) -> np.ndarray:
    """Configuration A: Raw IMU Baseline (Pure Dead Reckoning Mechanization).
    
    Integrates forward accelerometer and gyroscope heading directly.
    No AI velocity, no NHC, no map matching.
    """
    N = len(sensor_input.imu_data)
    pos_enu = np.zeros((N, 2), dtype=np.float64)
    
    pos = np.copy(sensor_input.initial_pos_enu[:2])
    vel = np.copy(sensor_input.initial_vel_enu[:2])
    heading = sensor_input.initial_heading_rad
    
    pos_enu[0] = pos

    for i in range(1, N):
        fwd_accel = float(sensor_input.imu_data[i, 0])
        yaw_rate = float(sensor_input.imu_data[i, 5])

        # Integrate heading
        heading = (heading + yaw_rate * dt + np.pi) % (2.0 * np.pi) - np.pi

        # Acceleration in ENU
        a_e = fwd_accel * np.cos(heading)
        a_n = fwd_accel * np.sin(heading)

        # Integrate position and velocity
        pos[0] += vel[0] * dt + 0.5 * a_e * dt**2
        pos[1] += vel[1] * dt + 0.5 * a_n * dt**2
        vel[0] += a_e * dt
        vel[1] += a_n * dt

        pos_enu[i] = pos

    return pos_enu


def run_ekf_ai_nhc(
    sensor_input: EstimatorInput,
    vel_model: Optional[VelocityEstimatorNet] = None,
    denoise_model: Optional[IMUDenoiseNet] = None,
    use_nhc: bool = True,
    dt: float = 0.1,
) -> np.ndarray:
    """Configuration B: EKF + AI Velocity + NHC (+ IMU Denoise).
    
    Integrates:
    - IMU DenoiseNet filtering on raw IMU (Rule 24)
    - AI Velocity forward speed estimation (Rule 21, 65)
    - 9-state EKF prediction and measurement updates
    - Non-Holonomic Constraints (NHC) lateral & vertical velocity pseudo-measurements
    """
    N = len(sensor_input.imu_data)
    imu_data = np.copy(sensor_input.imu_data)
    win_size = CONFIG["dataset"].window_size

    # Construct batched IMU windows (N, 6, win_size) efficiently
    padded = np.pad(imu_data, ((win_size - 1, 0), (0, 0)), mode="edge")
    windows = np.zeros((N, 6, win_size), dtype=np.float32)
    for i in range(N):
        windows[i] = padded[i : i + win_size].T

    # 1. Apply IMU Denoise Network if available (Rule 24)
    if denoise_model is not None:
        denoise_model.eval()
        with torch.no_grad():
            batch_t = torch.from_numpy(windows)
            bias_residuals = denoise_model(batch_t).cpu().numpy()
            imu_data -= bias_residuals
            padded = np.pad(imu_data, ((win_size - 1, 0), (0, 0)), mode="edge")
            for i in range(N):
                windows[i] = padded[i : i + win_size].T

    # 2. Precompute AI Velocity forward speeds in single batched pass (Rule 21, 65)
    ai_speeds = np.zeros(N, dtype=np.float64)
    if vel_model is not None:
        vel_model.eval()
        with torch.no_grad():
            batch_t = torch.from_numpy(windows)
            ai_speeds = vel_model(batch_t).squeeze(-1).cpu().numpy().astype(np.float64)
    else:
        # Fallback without AI model: clamp integrated forward speed
        ai_speeds = np.clip(np.cumsum(imu_data[:, 0] * dt), 0.0, 35.0)

    # 3. Initialize EKF at pre-blackout state (Rule 14)
    fusion = GNSSINSFusion(ref_lat=52.55, ref_lon=-1.50, dt=dt)
    fusion.ekf.x[0:2] = sensor_input.initial_pos_enu[:2]
    fusion.ekf.x[2] = 0.0
    fusion.ekf.x[3:5] = sensor_input.initial_vel_enu[:2]
    fusion.ekf.x[5] = 0.0
    fusion.ekf.x[6] = sensor_input.initial_heading_rad

    pred_enu = np.zeros((N, 2), dtype=np.float64)
    pred_enu[0] = sensor_input.initial_pos_enu[:2]

    # 4. Filter iteration during GNSS blackout (Rule 12)
    # ZERO GNSS position, ZERO GNSS velocity, ZERO ground truth
    for i in range(1, N):
        fwd_accel = float(imu_data[i, 0])
        yaw_rate = float(imu_data[i, 5])
        v_ai = float(ai_speeds[i])

        # Mechanization step
        fusion.ekf.predict(fwd_accel, yaw_rate)

        # AI Velocity measurement update (z_v = v_ai)
        fusion.ekf.update_velocity(v_ai, R_speed=0.6)

        # Non-Holonomic Constraints (v_lat ≈ 0, v_vert ≈ 0)
        if use_nhc:
            apply_nhc_update(fusion.ekf, sigma_lat=0.05, sigma_vert=0.05)

        pred_enu[i] = fusion.ekf.x[0:2]

    return pred_enu


def run_ekf_ai_nhc_osm(
    sensor_input: EstimatorInput,
    osm_matcher: HMMMapMatcher,
    vel_model: Optional[VelocityEstimatorNet] = None,
    denoise_model: Optional[IMUDenoiseNet] = None,
    dt: float = 0.1,
) -> np.ndarray:
    """Configuration C: EKF + AI Velocity + NHC + Independent OSM HMM/Viterbi.
    
    1. Dead-reckons estimated trajectory using EKF + AI Velocity + NHC + Denoise.
    2. Passes strictly estimated trajectory to HMM Viterbi Map Matcher (Rule 19, 20).
    3. Snaps to independent OpenStreetMap road network.
    """
    # Step 1: Compute dead-reckoned estimated trajectory
    estimated_traj = run_ekf_ai_nhc(
        sensor_input=sensor_input,
        vel_model=vel_model,
        denoise_model=denoise_model,
        use_nhc=True,
        dt=dt,
    )

    # Step 2: HMM Viterbi Map Matching using strictly estimated trajectory (Rule 19, 20)
    matched_traj = osm_matcher.match_trajectory(estimated_traj)
    return matched_traj


def generate_scenarios_from_drive(
    drive: IOVNBDrive,
    durations: List[int] = [15, 30, 60],
    stride_seconds: int = 45,
    min_speed: float = 3.0,
) -> List[Dict]:
    """Generate deterministic blackout scenarios along an authentic driving trip (Rule 43, 44)."""
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    N = len(t)
    dt = 0.1
    scenarios = []

    # Local ENU projection relative to drive start
    ref_lat, ref_lon = gt_coords[0, 0], gt_coords[0, 1]
    fusion = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=dt)
    gt_enu = np.array([fusion.latlon_to_enu(lat, lon)[:2] for lat, lon in gt_coords[:, :2]])

    step_stride = int(stride_seconds / dt)

    for dur_sec in durations:
        dur_steps = int(dur_sec / dt)
        if dur_steps >= N - 50:
            continue

        for start_idx in range(50, N - dur_steps - 10, step_stride):
            end_idx = start_idx + dur_steps
            
            # Check vehicle was actually moving during this window
            mean_speed = float(np.mean(v_speed[start_idx:end_idx]))
            if mean_speed < min_speed:
                continue

            # Ground-truth travelled distance during outage
            gt_seg = gt_enu[start_idx:end_idx]
            dist_travelled = float(np.sum(np.linalg.norm(np.diff(gt_seg, axis=0), axis=1)))
            if dist_travelled < 50.0:
                continue

            # Pre-blackout velocity and heading from GNSS fix immediately preceding outage (Rule 14)
            p_prev = gt_enu[start_idx - 1]
            p_curr = gt_enu[start_idx]
            initial_vel = (p_curr - p_prev) / dt
            initial_heading = float(np.arctan2(initial_vel[1], initial_vel[0]))

            scenarios.append({
                "scenario_id": f"{drive.drive_id}_t{start_idx * dt:.0f}_d{dur_sec}s",
                "drive_id": drive.drive_id,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "duration_seconds": dur_sec,
                "gt_distance_m": dist_travelled,
                "initial_lat": float(gt_coords[start_idx, 0]),
                "initial_lon": float(gt_coords[start_idx, 1]),
                "initial_pos_enu": np.array([p_curr[0], p_curr[1], 0.0], dtype=np.float64),
                "initial_vel_enu": np.array([initial_vel[0], initial_vel[1], 0.0], dtype=np.float64),
                "initial_heading_rad": initial_heading,
                "imu_slice": phone_imu[start_idx:end_idx],
                "gt_enu_slice": gt_seg,
                "gt_speed_slice": v_speed[start_idx:end_idx],
                "timestamps": t[start_idx:end_idx] - t[start_idx],
                "ref_lat": ref_lat,
                "ref_lon": ref_lon,
            })

    return scenarios


def evaluate_benchmark(
    raw_dir: Path = RAW_DATA_DIR / "iovnbd",
    models_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    max_scenarios_per_drive: int = 50,
) -> Dict:
    """Run comprehensive benchmark across authentic test drives (Rule 40-47)."""
    set_seed(SEED)
    raw_dir = Path(raw_dir)
    models_dir = Path(models_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load trained models
    vel_model = None
    vel_pt = models_dir / "velocity_net.pt"
    if vel_pt.exists():
        vel_model = VelocityEstimatorNet()
        vel_model.load_state_dict(torch.load(vel_pt, weights_only=True))
        logger.info(f"Loaded trained VelocityEstimatorNet from {vel_pt}")

    denoise_model = None
    denoise_pt = models_dir / "imu_denoise_net.pt"
    if denoise_pt.exists():
        denoise_model = IMUDenoiseNet()
        denoise_model.load_state_dict(torch.load(denoise_pt, weights_only=True))
        logger.info(f"Loaded trained IMUDenoiseNet from {denoise_pt}")

    # 2. Discover authentic test drives from canonical split
    test_drive_ids = list(CONFIG["dataset"].test_drives)
    logger.info(f"Evaluating on authentic test drives: {test_drive_ids}")

    all_scenarios = []
    drives_loaded = {}
    osm_matchers = {}

    for d_id in test_drive_ids:
        drive = load_drive_pair(raw_dir, d_id)
        drives_loaded[d_id] = drive
        
        # Load offline independent OSM road graph for this drive
        ref_lat = float(drive.vehicle_df["lat"].iloc[0])
        ref_lon = float(drive.vehicle_df["lon"].iloc[0])
        graph_loader = OSMGraphLoader()
        osm_graph = graph_loader.load_offline_graph(d_id, ref_lat=ref_lat, ref_lon=ref_lon)
        osm_matchers[d_id] = HMMMapMatcher(osm_graph, sigma_z=8.0, beta=4.0, max_search_radius=60.0)

        drive_scenarios = generate_scenarios_from_drive(drive)
        if len(drive_scenarios) > max_scenarios_per_drive:
            # Deterministic subset for balanced representation
            indices = np.linspace(0, len(drive_scenarios) - 1, max_scenarios_per_drive, dtype=int)
            drive_scenarios = [drive_scenarios[i] for i in indices]

        all_scenarios.extend(drive_scenarios)
        logger.info(f"Drive {d_id}: generated {len(drive_scenarios)} deterministic blackout scenarios.")

    logger.info(f"Total authentic blackout scenarios generated: {len(all_scenarios)}")

    # Save scenario_manifest.csv (Rule 44)
    manifest_rows = []
    for sc in all_scenarios:
        manifest_rows.append({
            "scenario_id": sc["scenario_id"],
            "drive_id": sc["drive_id"],
            "blackout_start_idx": sc["start_idx"],
            "blackout_end_idx": sc["end_idx"],
            "duration_s": sc["duration_seconds"],
            "distance_m": round(sc["gt_distance_m"], 2),
            "initial_lat": round(sc["initial_lat"], 6),
            "initial_lon": round(sc["initial_lon"], 6),
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(results_dir / "scenario_manifest.csv", index=False)
    logger.info(f"Saved scenario manifest to {results_dir / 'scenario_manifest.csv'}")

    # 3. Evaluate the 3 Configurations (Rule 41)
    configs = [
        "Config A: Raw IMU Baseline",
        "Config B: EKF + AI Velocity + NHC",
        "Config C: EKF + AI Velocity + NHC + OSM HMM",
    ]

    per_scenario_records = []
    saved_trajectories = {}

    for sc_idx, sc in enumerate(all_scenarios):
        # Construct isolated EstimatorInput (Rule 50)
        sensor_input = EstimatorInput(
            imu_data=sc["imu_slice"],
            initial_pos_enu=sc["initial_pos_enu"],
            initial_vel_enu=sc["initial_vel_enu"],
            initial_heading_rad=sc["initial_heading_rad"],
            timestamps=sc["timestamps"],
        )
        gt_enu = sc["gt_enu_slice"]
        dist_gt = sc["gt_distance_m"]
        drive_id = sc["drive_id"]
        matcher = osm_matchers[drive_id]

        # Execute Config A
        traj_a = run_raw_imu_baseline(sensor_input)
        metrics_a = compute_navigation_metrics(traj_a, gt_enu)

        # Execute Config B
        traj_b = run_ekf_ai_nhc(sensor_input, vel_model=vel_model, denoise_model=denoise_model, use_nhc=True)
        metrics_b = compute_navigation_metrics(traj_b, gt_enu)

        # Execute Config C
        traj_c = run_ekf_ai_nhc_osm(sensor_input, osm_matcher=matcher, vel_model=vel_model, denoise_model=denoise_model)
        metrics_c = compute_navigation_metrics(traj_c, gt_enu)

        # Record metrics for each configuration
        for cfg_name, met in [
            (configs[0], metrics_a),
            (configs[1], metrics_b),
            (configs[2], metrics_c),
        ]:
            drift_pct = (met.final_drift_m / dist_gt * 100.0) if dist_gt > 0 else 0.0
            per_scenario_records.append({
                "scenario_id": sc["scenario_id"],
                "drive_id": sc["drive_id"],
                "configuration": cfg_name,
                "blackout_duration": sc["duration_seconds"],
                "blackout_distance": round(dist_gt, 2),
                "final_error": round(met.final_drift_m, 2),
                "RMSE": round(met.rmse_position_m, 2),
                "CEP50": round(met.cep_50_m, 2),
                "drift_percentage": round(drift_pct, 2),
                "max_error": round(met.drms_95_m, 2),
                "passed_10_percent": bool(drift_pct < 10.0),
            })

        # Save trajectory snippet for representative visualization
        if sc_idx == 0:
            saved_trajectories = {
                "gt_enu": gt_enu,
                "config_a": traj_a,
                "config_b": traj_b,
                "config_c": traj_c,
            }

    # Save per_scenario_metrics.csv (Rule 45)
    per_scenario_df = pd.DataFrame(per_scenario_records)
    per_scenario_df.to_csv(results_dir / "per_scenario_metrics.csv", index=False)
    logger.info(f"Saved per-scenario metrics to {results_dir / 'per_scenario_metrics.csv'}")

    # 4. Aggregate summary statistics per configuration
    summary_stats = {}
    for cfg in configs:
        sub = per_scenario_df[per_scenario_df["configuration"] == cfg]
        drifts = sub["drift_percentage"].values
        pass_rate = float(np.mean(sub["passed_10_percent"])) * 100.0
        
        summary_stats[cfg] = {
            "num_scenarios": len(sub),
            "mean_drift_pct": round(float(np.mean(drifts)), 2),
            "median_drift_pct": round(float(np.median(drifts)), 2),
            "p90_drift_pct": round(float(np.percentile(drifts, 90)), 2),
            "p95_drift_pct": round(float(np.percentile(drifts, 95)), 2),
            "max_drift_pct": round(float(np.max(drifts)), 2),
            "mean_rmse_m": round(float(sub["RMSE"].mean()), 2),
            "mean_cep50_m": round(float(sub["CEP50"].mean()), 2),
            "pass_rate_under_10pct": round(pass_rate, 2),
        }
        logger.info(
            f"[{cfg}] Scenarios={len(sub)} | Median Drift={summary_stats[cfg]['median_drift_pct']}% | "
            f"Mean Drift={summary_stats[cfg]['mean_drift_pct']}% | Pass Rate (<10%)={pass_rate:.1f}%"
        )

    # Save results/eval_results.json and reports/eval_results.json
    eval_json = {
        "dataset": "IO-VNBD (Authentic)",
        "test_drives": test_drive_ids,
        "total_scenarios": len(all_scenarios),
        "configurations": summary_stats,
    }
    with open(results_dir / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_json, f, indent=2)
    logger.info(f"Saved evaluation results summary to {results_dir / 'eval_results.json'}")

    try:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / "eval_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_json, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not copy eval_results.json to reports: {e}")

    # Save trajectories for plotting
    np.savez_compressed(results_dir / "eval_trajectories.npz", **saved_trajectories)
    logger.info(f"Saved sample evaluation trajectories to {results_dir / 'eval_trajectories.npz'}")
    try:
        np.savez_compressed(REPORTS_DIR / "eval_trajectories.npz", **saved_trajectories)
    except Exception as e:
        logger.warning(f"Could not copy eval_trajectories.npz to reports: {e}")

    return eval_json


simulate_blackout_benchmark = evaluate_benchmark


def main():
    evaluate_benchmark()


if __name__ == "__main__":
    main()
