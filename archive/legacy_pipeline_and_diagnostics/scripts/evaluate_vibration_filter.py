"""Requirement 6 Evaluation: Non-Navigation Motion, Shock, and Vibration Filter.

Benchmarks the multi-stage VibrationMotionFilter against controlled disturbances:
1. Normal cruising
2. Synthetic pothole shock impulses (>2.5g vertical spike)
3. High-frequency chassis/engine vibration
4. Harsh sudden braking deceleration
5. In-cabin phone handling / mount rotation

Quantifies velocity jump reduction, state classification correctness,
and covariance gating behavior. Generates verification plots.
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
from src.idr.filters.vibration_filter import VibrationMotionFilter, MotionState


def evaluate_vibration_filter_pipeline(drive_id: str = "Vfa01"):
    print(f"[*] Loading drive {drive_id} for vibration filter benchmarking...")
    drive = load_drive_pair(RAW_DATA_DIR, drive_id)
    phone_imu, gt_coords, v_speed, t = drive.get_synced_data()
    dt = 0.1
    N = 800  # 80-second evaluation slice

    acc_raw = phone_imu[:N, :3].copy()
    gyro_raw = phone_imu[:N, 3:6].copy()

    # Inject controlled disturbances into specific intervals:
    # Interval 1: Normal Driving (t = 0 to 20s, steps 0 to 200) - untouched
    
    # Interval 2: Synthetic Potholes (t = 25s, 30s, 35s - steps 250, 300, 350)
    # Impulsive acceleration spikes of 25 m/s^2 (~2.5g) for 2 timesteps
    pothole_steps = [250, 300, 350]
    for p_step in pothole_steps:
        acc_raw[p_step:p_step+2, 2] += 26.0  # Sharp vertical bump
        acc_raw[p_step:p_step+2, 0] += 12.0  # Forward shock

    # Interval 3: High-frequency engine/chassis vibration (t = 45s to 55s, steps 450 to 550)
    # 25 Hz synthetic vibration added to 10 Hz sampling (aliased high-variance noise)
    noise_vib = np.random.randn(100, 3) * 3.5
    acc_raw[450:550] += noise_vib

    # Interval 4: Harsh sudden braking (t = 60s to 65s, steps 600 to 650)
    acc_raw[600:650, 0] -= 3.2  # -3.2 m/s^2 forward decel

    # Interval 5: Phone mount handling / rotation disturbance (t = 70s to 75s, steps 700 to 750)
    gyro_raw[700:750, 0] += 2.0 * np.sin(np.linspace(0, 4*np.pi, 50))
    gyro_raw[700:750, 1] += 1.8 * np.cos(np.linspace(0, 4*np.pi, 50))

    # Evaluate:
    # 1. Unfiltered integration (naive dead reckoning)
    # 2. Filtered integration with VibrationMotionFilter
    vfilter = VibrationMotionFilter(window_size=10, dt=dt)

    vel_unfiltered = [0.0]
    vel_filtered = [0.0]
    states_detected = []
    cov_scales = []
    acc_filtered_log = []

    for k in range(N):
        a_in = acc_raw[k]
        g_in = gyro_raw[k]

        # Naive: integrate raw forward accel
        v_unfilt = max(0.0, vel_unfiltered[-1] + a_in[0] * dt)
        vel_unfiltered.append(v_unfilt)

        # Multi-stage filter
        s_acc, s_gyro, state, cov_scale = vfilter.process(a_in, g_in)
        states_detected.append(state.value)
        cov_scales.append(cov_scale)
        acc_filtered_log.append(s_acc)

        # Filtered integration: clamp shock impact and weight
        v_filt = max(0.0, vel_filtered[-1] + s_acc[0] * dt)
        vel_filtered.append(v_filt)

    vel_unfiltered = np.array(vel_unfiltered[1:])
    vel_filtered = np.array(vel_filtered[1:])
    acc_filtered_log = np.array(acc_filtered_log)

    # Compute jump metrics around potholes
    unfilt_jumps = []
    filt_jumps = []
    for p_step in pothole_steps:
        # Jump across 5 steps around pothole
        uj = abs(vel_unfiltered[p_step + 4] - vel_unfiltered[p_step - 1])
        fj = abs(vel_filtered[p_step + 4] - vel_filtered[p_step - 1])
        unfilt_jumps.append(uj)
        filt_jumps.append(fj)

    mean_unfilt_jump = float(np.mean(unfilt_jumps))
    mean_filt_jump = float(np.mean(filt_jumps))
    jump_reduction = float((1.0 - mean_filt_jump / max(1e-4, mean_unfilt_jump)) * 100.0)

    # State classification breakdown
    unique_states, counts = np.unique(states_detected, return_counts=True)
    state_distribution = dict(zip(unique_states.tolist(), [int(c) for c in counts]))

    metrics = {
        "drive_id": drive_id,
        "evaluation_samples": N,
        "mean_pothole_velocity_jump_unfiltered_mps": round(mean_unfilt_jump, 2),
        "mean_pothole_velocity_jump_filtered_mps": round(mean_filt_jump, 2),
        "shock_jump_reduction_pct": round(jump_reduction, 1),
        "potholes_detected_as_shock": int(np.sum([s == MotionState.SHOCK_POTHOLE.value for s in states_detected])),
        "phone_motion_detected": int(np.sum([s == MotionState.PHONE_MOTION.value for s in states_detected])),
        "vibration_detected": int(np.sum([s == MotionState.VIBRATION_DISTURBANCE.value for s in states_detected])),
        "state_distribution": state_distribution,
    }

    out_json = RESULTS_DIR / "vibration_filter_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[+] Vibration Filter metrics saved to {out_json}")
    print(json.dumps(metrics, indent=2))

    # Generate Publication Plots
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    plt.subplots_adjust(hspace=0.25)
    t_axis = np.arange(N) * dt

    # Plot 1: Acceleration Raw vs Sanitized
    axes[0].plot(t_axis, acc_raw[:, 0], "r-", label="Raw IMU Accel (with Potholes & Vibration)", alpha=0.6, linewidth=1.2)
    axes[0].plot(t_axis, acc_filtered_log[:, 0], "b-", label="Sanitized Forward Accel", linewidth=1.8)
    axes[0].set_ylabel("Forward Accel (m/s²)", fontsize=10, fontweight="bold")
    axes[0].set_title("Requirement 6: Multi-Stage Non-Navigation Motion & Shock Filter", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Plot 2: Velocity Integration Jump Prevention
    axes[1].plot(t_axis, vel_unfiltered, "r--", label="Unfiltered Velocity (Severe Drift & Jumps)", linewidth=1.8)
    axes[1].plot(t_axis, vel_filtered, "b-", label="Filtered Velocity (Stable Dead Reckoning)", linewidth=2.0)
    for p_step in pothole_steps:
        axes[1].axvline(p_step * dt, color="purple", linestyle=":", label="Pothole Shock" if p_step == pothole_steps[0] else None)
    axes[1].axvspan(45.0, 55.0, color="orange", alpha=0.15, label="Chassis Vibration")
    axes[1].axvspan(70.0, 75.0, color="green", alpha=0.15, label="Phone Handling")
    axes[1].set_ylabel("Integrated Speed (m/s)", fontsize=10, fontweight="bold")
    axes[1].set_title("Velocity Jump Prevention during Controlled Disturbances", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")

    # Plot 3: Covariance Inflation & State Classifier
    axes[2].plot(t_axis, cov_scales, "m-", label="EKF Covariance Inflation Factor γ", linewidth=1.8)
    axes[2].set_yscale("log")
    axes[2].set_xlabel("Elapsed Time (s)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Covariance Scale (log)", fontsize=10, fontweight="bold")
    axes[2].set_title("Adaptive Covariance Gating (De-weights Corrupt Sensor Samples)", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right")

    plot_path = RESULTS_DIR / "plots" / "vibration_filter_evaluation.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Vibration filter evaluation plot saved to {plot_path}")


if __name__ == "__main__":
    evaluate_vibration_filter_pipeline("Vfa01")
