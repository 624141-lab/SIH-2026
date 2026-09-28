"""High-level GNSS+INS Fusion Engine orchestrating EKF, NHC, and AI-velocity updates."""

from typing import List, Optional, Tuple
import numpy as np

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from .ekf import ExtendedKalmanFilter
from .nhc import apply_nhc_update
from .zupt import StationaryDetector, apply_zupt, apply_zaru
from .gnss_monitor import GNSSDeficitDetector, GNSSStatus
from ..eval.transition import ReacquisitionSmoother

class GNSSINSFusion:
    """Manages coordinate transformations, automatic GNSS deficit detection, blackout transitions, and Dead Reckoning."""

    def __init__(self, ref_lat: float, ref_lon: float, dt: float = 0.1):
        self.dt = dt
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self.stationary_detector = StationaryDetector(window_size=10, acc_var_threshold=0.15)
        self.gnss_detector = GNSSDeficitDetector(dt=dt)
        self.reacquisition_smoother = ReacquisitionSmoother(blend_duration_sec=1.5, dt=dt)
        
        # Local ENU projection centered at initial GPS point
        if HAS_PYPROJ:
            self.proj_enu = pyproj.Proj(
                proj="aeqd",
                lat_0=ref_lat,
                lon_0=ref_lon,
                datum="WGS84",
                units="m"
            )
        else:
            self.proj_enu = None

        self.ekf = ExtendedKalmanFilter(dt=dt)
        self.in_blackout = False
        self.trajectory_history: List[np.ndarray] = []
        self.prev_gnss_enu: Optional[Tuple[float, float]] = None

    def latlon_to_enu(self, lat: float, lon: float) -> Tuple[float, float, float]:
        """Convert WGS84 lat/lon to local East-North-Up (m)."""
        if self.proj_enu is not None:
            x, y = self.proj_enu(lon, lat)
            return float(x), float(y), 0.0
        # High-precision WGS84 flat-Earth geodesic projection
        lat_rad = np.deg2rad(self.ref_lat)
        R_m = 6378137.0
        d_lat = np.deg2rad(lat - self.ref_lat)
        d_lon = np.deg2rad(lon - self.ref_lon)
        north = d_lat * R_m
        east = d_lon * R_m * np.cos(lat_rad)
        return float(east), float(north), 0.0

    def enu_to_latlon(self, east: float, north: float) -> Tuple[float, float]:
        """Convert local East-North-Up to WGS84 lat/lon."""
        if self.proj_enu is not None:
            lon, lat = self.proj_enu(east, north, inverse=True)
            return float(lat), float(lon)
        lat_rad = np.deg2rad(self.ref_lat)
        R_m = 6378137.0
        d_lat = north / R_m
        d_lon = east / (R_m * np.cos(lat_rad))
        lat = self.ref_lat + np.rad2deg(d_lat)
        lon = self.ref_lon + np.rad2deg(d_lon)
        return float(lat), float(lon)

    def step(
        self,
        fwd_accel: float,
        yaw_rate: float,
        gnss_pos: Optional[Tuple[float, float]] = None,
        ai_velocity: Optional[object] = None,
        ai_uncertainty: Optional[np.ndarray] = None,
        is_gnss_denied: Optional[bool] = None,
        use_nhc: bool = True,
        acc_3d: Optional[np.ndarray] = None,
        gyro_3d: Optional[np.ndarray] = None,
        hdop: float = 1.0,
        pos_std: float = 2.0,
    ) -> np.ndarray:
        """Single 10 Hz filter integration step.
        
        Args:
            fwd_accel: Forward acceleration in vehicle body frame (m/s^2)
            yaw_rate: Yaw rate in vehicle body frame (rad/s)
            gnss_pos: (lat, lon) or None
            ai_velocity: Forward velocity (float) or 2D velocity [v_fwd, v_lat] from AI model
            is_gnss_denied: Optional manual flag (overrides automatic detection if provided)
            use_nhc: Whether to apply non-holonomic constraints
            acc_3d: Optional 3-axis accel for stationary ZUPT detection
            gyro_3d: Optional 3-axis gyro for stationary ZARU detection
            hdop: Horizontal Dilution of Precision
            pos_std: Positional standard deviation in meters
        """
        # 1. IMU Prediction step
        self.ekf.predict(fwd_accel, yaw_rate)

        # 2. Check stationary detector for ZUPT and ZARU
        if acc_3d is not None and gyro_3d is not None:
            if self.stationary_detector.update(acc_3d, gyro_3d):
                apply_zupt(self.ekf, sigma_v=0.01)
                apply_zaru(self.ekf, gyro_z_raw=float(gyro_3d[2]) if len(gyro_3d) > 2 else yaw_rate)

        # 3. Automatic GNSS Deficit & Reacquisition Detection (Requirement 1 & 3)
        gnss_enu = None
        innov_vec = None
        innov_cov = None
        if gnss_pos is not None:
            east, north, up = self.latlon_to_enu(gnss_pos[0], gnss_pos[1])
            gnss_enu = np.array([east, north, up], dtype=np.float64)
            innov_vec = gnss_enu - self.ekf.x[0:3]
            innov_cov = self.ekf.P[0:3, 0:3] + np.eye(3) * (pos_std**2)

        self.gnss_detector.update(
            gnss_pos=gnss_pos,
            hdop=hdop,
            pos_std=pos_std,
            innovation_vector=innov_vec,
            innovation_cov=innov_cov,
            force_denied=is_gnss_denied,
        )

        # 4. Measurement updates based on automatic detector state
        if not self.gnss_detector.is_blackout() and gnss_enu is not None:
            # Check if transitioning from blackout to healthy (Requirement 3: Reacquisition)
            if self.in_blackout or self.gnss_detector.is_recovering():
                if self.in_blackout:
                    self.in_blackout = False
                    self.reacquisition_smoother.trigger_reacquisition(self.ekf.x[:2], gnss_enu[:2])

                # Apply smooth cosine bell blend (eliminates position teleports & velocity spikes)
                smoothed_xy = self.reacquisition_smoother.apply_smoothing(self.ekf.x[:2], gnss_enu[:2])
                gnss_to_update = np.array([smoothed_xy[0], smoothed_xy[1], gnss_enu[2]])
            else:
                self.in_blackout = False
                gnss_to_update = gnss_enu

            self.ekf.update_gnss_pos(gnss_to_update, R_cov=np.eye(3) * (pos_std**2))

            # Course Over Ground (COG) heading and GNSS velocity updates
            east, north = float(gnss_to_update[0]), float(gnss_to_update[1])
            if self.prev_gnss_enu is not None:
                de = east - self.prev_gnss_enu[0]
                dn = north - self.prev_gnss_enu[1]
                dist = np.hypot(de, dn)
                if dist > 0.1:  # Moving threshold
                    cog = np.arctan2(dn, de)
                    self.ekf.update_heading(cog, R_yaw=0.03)
                    v_gnss = np.array([de / self.dt, dn / self.dt, 0.0])
                    self.ekf.update_gnss_vel(v_gnss, R_cov=np.eye(3, dtype=np.float64) * (0.1**2))
            self.prev_gnss_enu = (east, north)
        else:
            self.in_blackout = True
            self.prev_gnss_enu = None

            # Non-Holonomic Constraints (NHC) during blackout
            if use_nhc:
                apply_nhc_update(self.ekf, sigma_lat=0.05, sigma_vert=0.05)

            # AI Velocity update (scalar or 2D displacement/velocity)
            if ai_velocity is not None:
                if isinstance(ai_velocity, (tuple, list, np.ndarray)) and len(ai_velocity) >= 2:
                    r_cov = None
                    if ai_uncertainty is not None:
                        if isinstance(ai_uncertainty, (tuple, list, np.ndarray)) and len(ai_uncertainty) >= 2:
                            r_cov = np.diag([float(max(1e-4, ai_uncertainty[0])), float(max(1e-4, ai_uncertainty[1]))])
                    self.ekf.update_velocity_2d(float(ai_velocity[0]), float(ai_velocity[1]), R_cov=r_cov)
                else:
                    r_speed = float(ai_uncertainty) if (ai_uncertainty is not None and np.isscalar(ai_uncertainty)) else 0.5
                    self.ekf.update_velocity(float(ai_velocity), R_speed=max(0.01, r_speed))

        current_state = self.ekf.x.copy()
        self.trajectory_history.append(current_state)
        return current_state
