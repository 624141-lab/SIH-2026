"""Non-Holonomic Constraints (NHC) for ground vehicle dead reckoning.

Ground vehicles satisfy two kinematic constraints in the body frame:
1. No lateral slip: v_lateral ≈ 0 (vehicle doesn't slide sideways)
2. No vertical velocity: v_vertical ≈ 0 (vehicle stays on ground plane)

These are formulated as EKF pseudo-measurement updates that provide
heading observability through the cross-covariance d(v_lat)/d(ψ).
Includes Mahalanobis Chi-square innovation gating (Requirement 9)
to prevent corrupting the filter during high-slip or dynamic cornering maneuvers.
"""

from typing import Tuple
import numpy as np
from .ekf import ExtendedKalmanFilter


def compute_lateral_velocity(ekf: ExtendedKalmanFilter) -> float:
    """Compute current vehicle body-frame lateral velocity v_lat."""
    psi = ekf.x[6]
    ve = ekf.x[3]
    vn = ekf.x[4]
    return float(-np.sin(psi) * ve + np.cos(psi) * vn)


def apply_nhc_update(
    ekf: ExtendedKalmanFilter,
    sigma_lat: float = 0.05,
    sigma_vert: float = 0.05,
    chi2_gate: float = 9.21,  # Chi-square 2-DOF 99% gating threshold
) -> Tuple[bool, float, float]:
    """Apply non-holonomic constraint pseudo-measurements with innovation gating.

    The lateral velocity in body frame is:
        v_lat = -sin(ψ) * vE + cos(ψ) * vN

    Its Jacobian with respect to heading ψ is:
        d(v_lat)/d(ψ) = -cos(ψ) * vE - sin(ψ) * vN

    This term is CRITICAL: it couples the NHC innovation to the heading
    state, providing heading observability during GNSS denial.
    
    Returns:
        (gated_passed, v_lat_before, v_lat_after)
    """
    psi = ekf.x[6]
    ve = ekf.x[3]
    vn = ekf.x[4]
    vu = ekf.x[5]

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # Predicted body-frame velocities
    v_lat_pred = -sin_psi * ve + cos_psi * vn
    v_vert_pred = vu
    v_lat_before = float(v_lat_pred)

    # Innovation: measurement (0) - prediction
    y = np.array([-v_lat_pred, -v_vert_pred], dtype=np.float64)

    # Measurement Jacobian H (2 x dim_x)
    H = np.zeros((2, ekf.dim_x), dtype=np.float64)

    # d(v_lat)/d(vE) = -sin(ψ)
    H[0, 3] = -sin_psi
    # d(v_lat)/d(vN) = cos(ψ)
    H[0, 4] = cos_psi
    # d(v_lat)/d(ψ) = -cos(ψ)*vE - sin(ψ)*vN  ← heading observability term
    H[0, 6] = -cos_psi * ve - sin_psi * vn

    # d(v_vert)/d(vU) = 1
    H[1, 5] = 1.0

    R = np.diag([sigma_lat**2, sigma_vert**2])

    S = H @ ekf.P @ H.T + R
    S_inv = np.linalg.inv(S)

    # Mahalanobis Chi-Square Innovation Gating (Requirement 9)
    nis = float(y.T @ S_inv @ y)
    if nis > chi2_gate:
        # High slip / extreme dynamic maneuver: gate update or inflate R to prevent corruption
        return False, v_lat_before, v_lat_before

    K = ekf.P @ H.T @ S_inv

    # Full Kalman update
    ekf.x = ekf.x + K @ y
    ekf.x[6] = (ekf.x[6] + np.pi) % (2 * np.pi) - np.pi  # normalize heading

    I = np.eye(ekf.dim_x)
    ekf.P = (I - K @ H) @ ekf.P @ (I - K @ H).T + K @ R @ K.T

    # Compute lateral velocity after update
    v_lat_after = float(-np.sin(ekf.x[6]) * ekf.x[3] + np.cos(ekf.x[6]) * ekf.x[4])
    return True, v_lat_before, v_lat_after


def apply_adaptive_nhc_update(
    ekf: ExtendedKalmanFilter,
    yaw_rate: float,
    sigma_lat_base: float = 0.05,
    sigma_vert_base: float = 0.05,
    k_turn: float = 0.5,
    chi2_gate: float = 9.21,
) -> Tuple[bool, float, float, float]:
    """Centripetal-aware adaptive Non-Holonomic Constraint (Phases 3 & 4).
    
    Dynamically inflates lateral covariance during cornering maneuvers:
        sigma_lat^2 = sigma_lat_base^2 + k_turn * (v_fwd * |yaw_rate|)^2
    
    This preserves the tight heading observability of NHC on straight corridors
    (where centripetal acceleration is ~0) while preventing heading corruption
    and filter divergence during high-speed turns.
    
    Returns:
        (passed, v_lat_before, v_lat_after, sigma_lat_adapted)
    """
    psi = ekf.x[6]
    ve = ekf.x[3]
    vn = ekf.x[4]
    vu = ekf.x[5]

    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)

    # Forward and lateral velocity prediction
    v_fwd_pred = cos_psi * ve + sin_psi * vn
    v_lat_pred = -sin_psi * ve + cos_psi * vn
    v_vert_pred = vu
    v_lat_before = float(v_lat_pred)

    # Centripetal acceleration a_c = |v_fwd * yaw_rate|
    a_centripetal = abs(float(v_fwd_pred * yaw_rate))

    # Adaptive lateral noise variance based on tyre-slip physics
    sigma_lat_adapted = float(np.sqrt(sigma_lat_base**2 + k_turn * (a_centripetal**2)))

    # Innovation: measurement (0) - prediction
    y = np.array([-v_lat_pred, -v_vert_pred], dtype=np.float64)

    H = np.zeros((2, ekf.dim_x), dtype=np.float64)
    H[0, 3] = -sin_psi
    H[0, 4] = cos_psi
    H[0, 6] = -cos_psi * ve - sin_psi * vn
    H[1, 5] = 1.0

    R = np.diag([sigma_lat_adapted**2, sigma_vert_base**2])

    S = H @ ekf.P @ H.T + R
    S_inv = np.linalg.inv(S)

    nis = float(y.T @ S_inv @ y)
    if nis > chi2_gate:
        return False, v_lat_before, v_lat_before, sigma_lat_adapted

    K = ekf.P @ H.T @ S_inv

    ekf.x = ekf.x + K @ y
    ekf.x[6] = (ekf.x[6] + np.pi) % (2 * np.pi) - np.pi

    I = np.eye(ekf.dim_x)
    ekf.P = (I - K @ H) @ ekf.P @ (I - K @ H).T + K @ R @ K.T

    v_lat_after = float(-np.sin(ekf.x[6]) * ekf.x[3] + np.cos(ekf.x[6]) * ekf.x[4])
    return True, v_lat_before, v_lat_after, sigma_lat_adapted

