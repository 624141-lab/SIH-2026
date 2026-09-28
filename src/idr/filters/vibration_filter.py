"""Multi-stage Non-Navigation Motion, Shock, and Vibration Filter.

Satisfies SIH Requirement 6:
- Separates static gravity from dynamic vehicle acceleration
- Calculates rolling variance, jerk, and kurtosis to identify transient shocks
- Classifies motion states: STATIONARY, NORMAL_DRIVING, ACCELERATION, BRAKING,
  TURNING, VIBRATION_DISTURBANCE, SHOCK_POTHOLE, PHONE_MOTION
- Applies adaptive covariance inflation and shock clamping to prevent velocity/position jumps.
"""

from enum import Enum
from typing import Dict, Tuple, Optional
import numpy as np


class MotionState(str, Enum):
    STATIONARY = "STATIONARY"
    NORMAL_DRIVING = "NORMAL_DRIVING"
    ACCELERATION = "ACCELERATION"
    BRAKING = "BRAKING"
    TURNING = "TURNING"
    VIBRATION_DISTURBANCE = "VIBRATION_DISTURBANCE"
    SHOCK_POTHOLE = "SHOCK_POTHOLE"
    PHONE_MOTION = "PHONE_MOTION"


class VibrationMotionFilter:
    """Multi-stage signal processor and motion-state classifier for smartphone IMU."""

    def __init__(
        self,
        window_size: int = 10,
        dt: float = 0.1,
        shock_acc_thresh: float = 20.0,       # m/s^2 (~2g impulse)
        kurtosis_shock_thresh: float = 5.0,    # Kurtosis threshold for transient outlier
        var_vibration_thresh: float = 2.5,     # Rolling variance for engine/chassis vibration
    ):
        self.window_size = window_size
        self.dt = dt
        self.shock_acc_thresh = shock_acc_thresh
        self.kurtosis_shock_thresh = kurtosis_shock_thresh
        self.var_vibration_thresh = var_vibration_thresh

        # History buffers
        self.acc_hist = []
        self.gyro_hist = []
        self.gravity_est = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
        self.alpha_gravity = 0.95  # Low-pass filter coefficient for gravity tracking

        self.last_acc = None
        self.current_state = MotionState.STATIONARY
        self.confidence = 1.0

    def process(
        self,
        acc_3d: np.ndarray,
        gyro_3d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, MotionState, float]:
        """Process 3D accelerometer and gyroscope sample.
        
        Args:
            acc_3d: (3,) [ax, ay, az] in m/s^2
            gyro_3d: (3,) [gx, gy, gz] in rad/s
            
        Returns:
            sanitized_acc: (3,) filtered acceleration
            sanitized_gyro: (3,) filtered gyroscope
            motion_state: MotionState classification
            cov_scale: Covariance inflation factor for EKF (1.0 = clean, >10.0 = disturbance)
        """
        acc = np.array(acc_3d, dtype=np.float64)
        gyro = np.array(gyro_3d, dtype=np.float64)

        # 1. Low-pass gravity separation
        self.gravity_est = self.alpha_gravity * self.gravity_est + (1.0 - self.alpha_gravity) * acc
        dyn_acc = acc - self.gravity_est

        # 2. Maintain sliding window history
        self.acc_hist.append(acc)
        self.gyro_hist.append(gyro)
        if len(self.acc_hist) > self.window_size:
            self.acc_hist.pop(0)
            self.gyro_hist.pop(0)

        acc_arr = np.array(self.acc_hist)
        gyro_arr = np.array(self.gyro_hist)

        # 3. Compute statistical features
        acc_mag = np.linalg.norm(acc)
        dyn_mag = np.linalg.norm(dyn_acc)
        gyro_mag = np.linalg.norm(gyro)

        # Jerk (rate of change of acceleration)
        if self.last_acc is not None:
            jerk = np.linalg.norm(acc - self.last_acc) / self.dt
        else:
            jerk = 0.0
        self.last_acc = acc.copy()

        # Rolling variance
        acc_var = float(np.mean(np.var(acc_arr, axis=0))) if len(acc_arr) >= 4 else 0.0
        gyro_var = float(np.mean(np.var(gyro_arr, axis=0))) if len(gyro_arr) >= 4 else 0.0

        # Kurtosis along vertical/magnitude axis (detects heavy-tailed shock spikes)
        if len(acc_arr) >= 6:
            diff = acc_arr[:, 2] - np.mean(acc_arr[:, 2])
            std_z = np.std(acc_arr[:, 2])
            if std_z > 1e-4:
                kurtosis_z = float(np.mean(diff**4) / (std_z**4))
            else:
                kurtosis_z = 3.0
        else:
            kurtosis_z = 3.0

        # 4. Motion State Classification
        cov_scale = 1.0
        sanitized_acc = acc.copy()
        sanitized_gyro = gyro.copy()

        # Check Stationary (engine off or idling at stoplight)
        if dyn_mag < 0.25 and gyro_mag < 0.04 and acc_var < 0.08:
            state = MotionState.STATIONARY
            cov_scale = 1.0

        # Check Shock / Pothole impulse (transient acceleration spike or high kurtosis)
        elif acc_mag > self.shock_acc_thresh or jerk > 80.0 or kurtosis_z > self.kurtosis_shock_thresh:
            state = MotionState.SHOCK_POTHOLE
            # Clamp the shock impulse so it does not corrupt the dead reckoning velocity integration
            norm_dyn = np.linalg.norm(dyn_acc)
            if norm_dyn > 5.0:
                sanitized_acc = self.gravity_est + (dyn_acc / norm_dyn) * 5.0
            cov_scale = 50.0  # Inflate measurement covariance by 50x during pothole

        # Check Phone Motion / Mount disturbance (sudden 3D gyro tumble)
        elif gyro_mag > 1.2 and (np.abs(gyro[0]) > 0.8 or np.abs(gyro[1]) > 0.8):
            state = MotionState.PHONE_MOTION
            cov_scale = 100.0  # Phone is being handled; isolate from vehicle heading

        # Check High-Frequency Vibration Disturbance (rough cobblestone / engine shake)
        elif acc_var > self.var_vibration_thresh:
            state = MotionState.VIBRATION_DISTURBANCE
            cov_scale = 5.0

        # Check Turning
        elif np.abs(gyro[2]) > 0.12:
            state = MotionState.TURNING
            cov_scale = 1.0

        # Check Harsh Braking
        elif dyn_acc[0] < -1.8:
            state = MotionState.BRAKING
            cov_scale = 1.0

        # Check Acceleration
        elif dyn_acc[0] > 1.0:
            state = MotionState.ACCELERATION
            cov_scale = 1.0

        else:
            state = MotionState.NORMAL_DRIVING
            cov_scale = 1.0

        self.current_state = state
        self.confidence = 1.0 / cov_scale

        return sanitized_acc, sanitized_gyro, state, cov_scale
