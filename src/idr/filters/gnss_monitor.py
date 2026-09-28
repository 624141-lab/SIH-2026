"""Automatic GNSS Quality and Deficit Detector for IDR System (Requirement 1).

Monitors GNSS signal health and transitions between states:
NORMAL_GNSS -> DEGRADED -> GNSS_DENIED -> RECOVERING -> NORMAL_GNSS

Features:
- Fix availability and update timeout monitoring.
- Horizontal Dilution of Precision (HDOP) and positional variance monitoring.
- Chi-square Innovation / Mahalanobis distance gating (Normalized Innovation Squared - NIS).
- Hysteresis and debouncing to prevent chatter on sporadic multipath.
"""

from enum import Enum
from typing import Optional, Tuple
import numpy as np


class GNSSStatus(Enum):
    NORMAL_GNSS = "NORMAL_GNSS"
    DEGRADED = "DEGRADED"
    GNSS_DENIED = "GNSS_DENIED"
    RECOVERING = "RECOVERING"


class GNSSDeficitDetector:
    """State machine detector for automatic GNSS blackout and degradation detection."""

    def __init__(
        self,
        dt: float = 0.1,
        max_fix_age_sec: float = 0.5,
        hdop_threshold: float = 3.5,
        pos_sigma_threshold: float = 8.0,
        nis_gate_threshold: float = 11.34,  # Chi-square 3-DOF at p=0.01
        degrade_hysteresis_count: int = 3,
        recover_hysteresis_count: int = 5,
    ):
        self.dt = dt
        self.max_fix_age_sec = max_fix_age_sec
        self.hdop_threshold = hdop_threshold
        self.pos_sigma_threshold = pos_sigma_threshold
        self.nis_gate_threshold = nis_gate_threshold
        self.degrade_hysteresis_count = degrade_hysteresis_count
        self.recover_hysteresis_count = recover_hysteresis_count

        self.status = GNSSStatus.NORMAL_GNSS
        self.time_since_last_fix = 0.0
        self.degrade_counter = 0
        self.recover_counter = 0

        # Diagnostics metrics
        self.total_frames = 0
        self.outage_events = 0
        self.recovery_events = 0

    def update(
        self,
        gnss_pos: Optional[Tuple[float, float]],
        hdop: float = 1.0,
        pos_std: float = 2.0,
        innovation_vector: Optional[np.ndarray] = None,
        innovation_cov: Optional[np.ndarray] = None,
        force_denied: Optional[bool] = None,
    ) -> GNSSStatus:
        """Process one time step and update GNSS status.
        
        Args:
            gnss_pos: (lat, lon) or None
            hdop: Horizontal Dilution of Precision
            pos_std: Estimated 1-sigma horizontal position accuracy in meters
            innovation_vector: (3,) [delta_east, delta_north, delta_up] from EKF
            innovation_cov: (3, 3) Innovation covariance S from EKF
            force_denied: Optional manual override for controlled experiments
        """
        self.total_frames += 1

        # Check explicit manual injection if supplied (for controlled testing)
        if force_denied is True:
            gnss_pos = None

        # 1. Update temporal fix freshness
        if gnss_pos is not None:
            self.time_since_last_fix = 0.0
        else:
            self.time_since_last_fix += self.dt

        # 2. Compute Normalized Innovation Squared (NIS) if available
        nis = 0.0
        if innovation_vector is not None and innovation_cov is not None:
            try:
                inv_S = np.linalg.inv(innovation_cov[:2, :2])
                nis = float(innovation_vector[:2] @ inv_S @ innovation_vector[:2])
            except np.linalg.LinAlgError:
                nis = 0.0

        # 3. Quality evaluation
        is_fresh = self.time_since_last_fix <= self.max_fix_age_sec
        is_hdop_good = hdop <= self.hdop_threshold
        is_acc_good = pos_std <= self.pos_sigma_threshold
        is_nis_good = (nis <= self.nis_gate_threshold) if (innovation_vector is not None) else True

        is_healthy = is_fresh and is_hdop_good and is_acc_good and is_nis_good

        # 4. State Machine with Hysteresis
        if self.status == GNSSStatus.NORMAL_GNSS:
            if not is_healthy:
                self.degrade_counter += 1
                if not is_fresh or self.degrade_counter >= self.degrade_hysteresis_count:
                    self.status = GNSSStatus.GNSS_DENIED
                    self.outage_events += 1
                    self.degrade_counter = 0
                else:
                    self.status = GNSSStatus.DEGRADED
            else:
                self.degrade_counter = 0

        elif self.status == GNSSStatus.DEGRADED:
            if is_healthy:
                self.status = GNSSStatus.NORMAL_GNSS
                self.degrade_counter = 0
            else:
                self.degrade_counter += 1
                if not is_fresh or self.degrade_counter >= self.degrade_hysteresis_count:
                    self.status = GNSSStatus.GNSS_DENIED
                    self.outage_events += 1
                    self.degrade_counter = 0

        elif self.status == GNSSStatus.GNSS_DENIED:
            if is_healthy:
                self.recover_counter += 1
                if self.recover_counter >= self.recover_hysteresis_count:
                    self.status = GNSSStatus.RECOVERING
                    self.recovery_events += 1
                    self.recover_counter = 0
            else:
                self.recover_counter = 0

        elif self.status == GNSSStatus.RECOVERING:
            if is_healthy:
                self.recover_counter += 1
                # Smooth transition window before declaring NORMAL_GNSS
                if self.recover_counter >= self.recover_hysteresis_count:
                    self.status = GNSSStatus.NORMAL_GNSS
                    self.recover_counter = 0
            else:
                # Regressed during recovery
                self.status = GNSSStatus.GNSS_DENIED
                self.recover_counter = 0

        return self.status

    def is_blackout(self) -> bool:
        """Returns True if the system must operate in Dead Reckoning mode."""
        return self.status in [GNSSStatus.GNSS_DENIED, GNSSStatus.DEGRADED]

    def is_recovering(self) -> bool:
        """Returns True if the system is currently smoothing the transition back to GNSS."""
        return self.status == GNSSStatus.RECOVERING
