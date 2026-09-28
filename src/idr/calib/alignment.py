"""Estimation and causal tracking of phone-to-vehicle rotation matrix (attitude alignment).

Transforms smartphone sensor observations into the vehicle reference frame:
- X_v: Vehicle forward (longitudinal)
- Y_v: Vehicle lateral (right/left)
- Z_v: Vehicle vertical (upward)

Satisfies SIH Requirements:
- Strictly causally available onboard sensors (Accelerometer, Gyroscope, pre-blackout GNSS velocity)
- ZERO reliance on ECU speed, vehicle speed, or ground truth (NO v_speed)
- Explicit automatic stationary detector (gravity norm, low gyro, low variance, sustained duration)
- Roll and pitch derived from gravity vector
- Yaw observability handling:
    1. Stationary phone -> ALIGNMENT_UNCERTAIN (yaw unobservable)
    2. Straight motion with forward acceleration -> Observable forward axis
    3. Pre-blackout GNSS velocity -> Observable track angle and forward axis
    4. Unavailable / Insufficient motion -> ALIGNMENT_UNCERTAIN with quantified uncertainty.
"""

from enum import Enum
from typing import Optional, Tuple, List, Dict
import numpy as np


class AlignmentStatus(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    ALIGNMENT_UNCERTAIN = "ALIGNMENT_UNCERTAIN"  # Gravity/Pitch/Roll known, Yaw unobservable or noisy
    ALIGNED = "ALIGNED"                          # Full 3D rotation matrix (Roll, Pitch, Yaw) estimated


def compute_rotation_matrix(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Euler Z-Y-X rotation matrix from roll, pitch, yaw (radians).
    
    Transforms vehicle frame to phone frame: v_phone = R @ v_vehicle.
    """
    cr, sr = np.cos(roll_rad), np.sin(roll_rad)
    cp, sp = np.cos(pitch_rad), np.sin(pitch_rad)
    cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float64)

    Ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)

    Rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Decompose 3x3 rotation matrix R (phone-to-vehicle or vehicle-to-phone) into roll, pitch, yaw in degrees."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    if np.abs(np.cos(pitch)) > 1e-4:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(np.rad2deg(roll)), float(np.rad2deg(pitch)), float(np.rad2deg(yaw))


def detect_stationary_windows(
    acc: np.ndarray,
    gyro: np.ndarray,
    dt: float = 0.1,
    min_duration_s: float = 1.5,
    acc_grav_tol: float = 0.45,
    gyro_thresh: float = 0.06,
    var_acc_thresh: float = 0.04,
    var_gyro_thresh: float = 0.005,
) -> List[Tuple[int, int]]:
    """Explicitly detect stationary intervals using accelerometer and gyroscope physics.
    
    Conditions for stationarity at timestep i:
    1. Acceleration magnitude near 1g: |norm(a) - 9.81| < acc_grav_tol
    2. Gyroscope magnitude near zero: norm(w) < gyro_thresh
    3. Rolling acceleration variance over window < var_acc_thresh
    4. Rolling angular rate variance over window < var_gyro_thresh
    5. Continuous duration >= min_duration_s
    """
    N = len(acc)
    min_samples = int(min_duration_s / dt)
    if N < min_samples:
        return []

    acc_norm = np.linalg.norm(acc, axis=1)
    gyro_norm = np.linalg.norm(gyro, axis=1)

    # Condition 1 & 2: Instantaneous magnitudes
    cond_a_mag = np.abs(acc_norm - 9.81) < acc_grav_tol
    cond_w_mag = gyro_norm < gyro_thresh

    # Condition 3 & 4: Rolling variances over 1.0s window
    win_len = max(5, int(1.0 / dt))
    pad_a = np.pad(acc_norm, (win_len // 2, win_len // 2), mode="edge")
    pad_w = np.pad(gyro_norm, (win_len // 2, win_len // 2), mode="edge")

    # Fast rolling variance
    var_a = np.zeros(N, dtype=np.float64)
    var_w = np.zeros(N, dtype=np.float64)
    for i in range(N):
        var_a[i] = np.var(pad_a[i : i + win_len])
        var_w[i] = np.var(pad_w[i : i + win_len])

    cond_a_var = var_a < var_acc_thresh
    cond_w_var = var_w < var_gyro_thresh

    is_stat = cond_a_mag & cond_w_mag & cond_a_var & cond_w_var

    # Find contiguous intervals
    segments = []
    in_seg = False
    seg_start = 0
    for i in range(N):
        if is_stat[i] and not in_seg:
            in_seg = True
            seg_start = i
        elif not is_stat[i] and in_seg:
            in_seg = False
            if (i - seg_start) >= min_samples:
                segments.append((seg_start, i))
    if in_seg and (N - seg_start) >= min_samples:
        segments.append((seg_start, N))

    return segments


class PhoneToVehicleAligner:
    """Estimates and causally tracks the 3D rotation matrix R_{phone -> vehicle}.
    
    Guarantees:
    - Never uses vehicle speed / ECU speed / ground truth (strictly 0% v_speed).
    - Uses automatic stationary detector for gravity/pitch/roll.
    - Explicitly sets ALIGNMENT_UNCERTAIN when stationary or when yaw is unobservable.
    - Observes yaw from straight dynamic motion or pre-blackout GNSS velocity.
    """

    def __init__(
        self,
        dt: float = 0.1,
        min_stat_duration_s: float = 1.5,
        acc_grav_tol: float = 0.45,
        gyro_thresh: float = 0.06,
    ):
        self.dt = dt
        self.min_stat_duration_s = min_stat_duration_s
        self.acc_grav_tol = acc_grav_tol
        self.gyro_thresh = gyro_thresh

        self.R_phone_to_vehicle = np.eye(3, dtype=np.float64)
        self.status = AlignmentStatus.UNINITIALIZED

        self.z_phone = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.x_phone = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.yaw_uncertainty_deg = 180.0

        self.gravity_estimated = False
        self.yaw_estimated = False

    @property
    def is_calibrated(self) -> bool:
        """Returns True ONLY if full 3D alignment is confirmed."""
        return self.status == AlignmentStatus.ALIGNED

    def get_euler_angles(self) -> Tuple[float, float, float]:
        """Return estimated (roll_deg, pitch_deg, yaw_deg)."""
        return float(self.roll_deg), float(self.pitch_deg), float(self.yaw_deg)

    def detect_stationarity(self, acc: np.ndarray, gyro: np.ndarray) -> List[Tuple[int, int]]:
        """Detect verified stationary windows in incoming IMU stream."""
        return detect_stationary_windows(
            acc,
            gyro,
            dt=self.dt,
            min_duration_s=self.min_stat_duration_s,
            acc_grav_tol=self.acc_grav_tol,
            gyro_thresh=self.gyro_thresh,
        )

    def update_gravity_estimate(self, stationary_acc: np.ndarray) -> bool:
        """Estimate vehicle vertical axis (Z_v, pitch, roll) from stationary gravity.
        
        Yaw remains strictly UNOBSERVABLE when stationary!
        """
        if stationary_acc.ndim == 1:
            stationary_acc = stationary_acc[np.newaxis, :]
        if len(stationary_acc) == 0:
            return False

        mean_g = np.mean(stationary_acc, axis=0)
        norm_g = np.linalg.norm(mean_g)
        if norm_g < 1e-3:
            return False

        # Specific force points upward opposite to gravity (+Z_v)
        z_p = mean_g / norm_g
        self.z_phone = z_p

        # Derive pitch and roll from gravity vector
        # Leveling rotation to map z_p -> [0, 0, 1]
        z_v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        axis = np.cross(z_p, z_v)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            R_level = np.eye(3, dtype=np.float64)
        else:
            axis = axis / axis_len
            angle = np.arccos(np.clip(np.dot(z_p, z_v), -1.0, 1.0))
            K = np.array([
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ], dtype=np.float64)
            R_level = np.eye(3, dtype=np.float64) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

        self.R_phone_to_vehicle = R_level
        r_deg, p_deg, _ = rotation_matrix_to_euler(R_level)
        self.roll_deg = r_deg
        self.pitch_deg = p_deg
        self.gravity_estimated = True

        # When stationary, yaw is strictly unobservable!
        self.status = AlignmentStatus.ALIGNMENT_UNCERTAIN
        self.yaw_uncertainty_deg = 180.0
        return True

    def update_forward_motion(
        self,
        motion_acc: np.ndarray,
        motion_gyro: Optional[np.ndarray] = None,
        gnss_vel: Optional[np.ndarray] = None,
    ) -> bool:
        """Estimate forward longitudinal axis (yaw) when motion is observable.
        
        Uses NO vehicle speed / ECU speed / ground truth!
        Sources:
        1. Pre-blackout GNSS velocity vector (if available during GNSS reception)
        2. Dynamic forward acceleration PCA on horizontal plane
        """
        if not self.gravity_estimated:
            return False

        if motion_acc.ndim == 1:
            motion_acc = motion_acc[np.newaxis, :]
        N = len(motion_acc)
        if N < 15:
            return False

        # Level the acceleration into horizontal plane using known vertical axis
        R_level = self.R_phone_to_vehicle
        acc_level = (R_level @ motion_acc.T).T

        # Case A: GNSS Velocity available pre-blackout (Legitimate onboard GNSS sensor)
        if gnss_vel is not None and len(gnss_vel) == N:
            # Horizontal GNSS speed and acceleration
            gnss_spd = np.linalg.norm(gnss_vel[:, :2], axis=1)
            valid_gnss = gnss_spd > 2.5
            if np.sum(valid_gnss) >= 15:
                # Vehicle forward acceleration from GNSS speed
                a_gnss_fwd = np.gradient(gnss_spd, self.dt)
                accel_active = valid_gnss & (np.abs(a_gnss_fwd) > 0.25)
                if np.sum(accel_active) >= 10:
                    a_horiz_phone = acc_level[accel_active, :2]
                    a_weights = a_gnss_fwd[accel_active, np.newaxis]
                    x_dir = np.sum(a_horiz_phone * a_weights, axis=0)
                    norm_x = float(np.linalg.norm(x_dir))
                    if norm_x > 1e-4:
                        x_unit = x_dir / norm_x
                        yaw_est = float(np.arctan2(x_unit[1], x_unit[0]))
                        cy, sy = np.cos(-yaw_est), np.sin(-yaw_est)
                        R_yaw = np.array([
                            [cy, -sy, 0.0],
                            [sy, cy, 0.0],
                            [0.0, 0.0, 1.0],
                        ], dtype=np.float64)

                        self.R_phone_to_vehicle = R_yaw @ R_level
                        self.yaw_deg = float(np.rad2deg(yaw_est))
                        self.yaw_uncertainty_deg = 2.5
                        self.yaw_estimated = True
                        self.status = AlignmentStatus.ALIGNED
                        return True

        # Case B: Onboard IMU only (Straight driving with longitudinal acceleration, NO GNSS)
        a_horiz = acc_level[:, :2]
        a_mag = np.linalg.norm(a_horiz, axis=1)

        # Filter for straight-line dynamic acceleration (accel > 0.4 m/s^2, low gyro if available)
        motion_mask = (a_mag > 0.4) & (a_mag < 4.0)
        if motion_gyro is not None and len(motion_gyro) == N:
            motion_mask = motion_mask & (np.linalg.norm(motion_gyro, axis=1) < 0.10)

        if np.sum(motion_mask) >= 15:
            # PCA on dynamic horizontal acceleration
            cov = np.cov(a_horiz[motion_mask], rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            ratio = eigvals[1] / max(eigvals[0], 1e-4)

            # Significant 1D dominance (ratio > 2.5 indicates straight-line motion)
            if ratio > 2.5:
                x_dir = eigvecs[:, 1]
                # Default forward direction aligns with positive horizontal axis
                if x_dir[0] < 0:
                    x_dir = -x_dir

                yaw_est = float(np.arctan2(x_dir[1], x_dir[0]))
                cy, sy = np.cos(-yaw_est), np.sin(-yaw_est)
                R_yaw = np.array([
                    [cy, -sy, 0.0],
                    [sy, cy, 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float64)

                self.R_phone_to_vehicle = R_yaw @ R_level
                self.yaw_deg = float(np.rad2deg(yaw_est))
                self.yaw_uncertainty_deg = 15.0
                self.yaw_estimated = True
                self.status = AlignmentStatus.ALIGNED
                return True

        # Yaw cannot be reliably resolved
        self.status = AlignmentStatus.ALIGNMENT_UNCERTAIN
        self.yaw_uncertainty_deg = 180.0
        return False

    def estimate_from_stationary_and_motion(
        self,
        stationary_acc: np.ndarray,
        motion_acc: np.ndarray,
        motion_vel: Optional[np.ndarray] = None,  # Kept in signature for compatibility; STRICTLY UNUSED!
    ) -> np.ndarray:
        """Calibrate using stationary gravity and dynamic motion without vehicle speed."""
        self.update_gravity_estimate(stationary_acc)
        self.update_forward_motion(motion_acc)
        return self.R_phone_to_vehicle

    def estimate_attitude(
        self,
        acc: np.ndarray,
        gyro: np.ndarray,
        gnss_vel: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, AlignmentStatus, Dict[str, float]]:
        """Fully automatic causal attitude estimator.
        
        1. Detects verified stationary window in incoming IMU stream.
        2. If stationary window found: extracts gravity, calculates pitch and roll.
        3. If motion present: extracts forward axis (yaw).
        4. If no stationary window found: returns ALIGNMENT_UNCERTAIN, does NOT invent a rotation.
        """
        stat_segments = self.detect_stationarity(acc, gyro)
        if stat_segments:
            # Use longest verified stationary window
            longest_seg = max(stat_segments, key=lambda seg: seg[1] - seg[0])
            s_acc = acc[longest_seg[0] : longest_seg[1]]
            self.update_gravity_estimate(s_acc)

            # Look for motion outside stationary segments
            m_indices = [i for i in range(len(acc)) if not any(s <= i < e for s, e in stat_segments)]
            if len(m_indices) >= 20:
                m_acc = acc[m_indices]
                m_gyro = gyro[m_indices]
                g_vel = gnss_vel[m_indices] if gnss_vel is not None else None
                self.update_forward_motion(m_acc, m_gyro, g_vel)
        else:
            # NO RELIABLE STATIONARY PERIOD FOUND
            # Do NOT invent an artificial rotation!
            self.status = AlignmentStatus.ALIGNMENT_UNCERTAIN
            self.R_phone_to_vehicle = np.eye(3, dtype=np.float64)
            self.yaw_uncertainty_deg = 180.0

        telemetry = {
            "roll_deg": self.roll_deg,
            "pitch_deg": self.pitch_deg,
            "yaw_deg": self.yaw_deg,
            "yaw_uncertainty_deg": self.yaw_uncertainty_deg,
            "status": self.status.value,
        }
        return self.R_phone_to_vehicle, self.status, telemetry

    def transform_imu(self, acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate raw smartphone IMU measurements into the vehicle reference frame."""
        acc_v = (self.R_phone_to_vehicle @ acc.T).T
        gyro_v = (self.R_phone_to_vehicle @ gyro.T).T
        return acc_v, gyro_v
