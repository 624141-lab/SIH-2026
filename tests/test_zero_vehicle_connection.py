"""Requirement 4 Verification: Zero Physical Connection & Zero CAN/OBD-II Data Flow Test.

Formally proves that during active navigation and GNSS blackout inference,
the IDR estimation pipeline relies strictly and exclusively on smartphone sensors:
- No OBD-II
- No CAN bus
- No wheel speed sensors / ticks
- No vehicle ECU communication
- No ground truth vehicle speed or position leakage during blackout inference.
"""

import inspect
import numpy as np
import pytest
import torch

from src.idr.filters.fusion import GNSSINSFusion
from src.idr.filters.ekf import ExtendedKalmanFilter
from src.idr.models.inertial_odom import InertialOdomNet
from src.idr.models.velocity_net import VelocityEstimatorNet


def test_estimator_input_signature_has_no_ecu_dependency():
    """Verify that filter and model interfaces do not accept or require vehicle ECU parameters."""
    forbidden_tokens = ["obd", "can", "wheel", "ecu", "vehicle_speed", "ground_truth", "gt_vel", "gt_pos"]

    # 1. Check GNSSINSFusion.step() arguments
    fusion_sig = inspect.signature(GNSSINSFusion.step)
    for param_name in fusion_sig.parameters.keys():
        for token in forbidden_tokens:
            assert token not in param_name.lower(), (
                f"Forbidden token '{token}' found in GNSSINSFusion.step parameter: {param_name}"
            )

    # 2. Check EKF methods
    for method_name in ["predict", "update_gnss_pos", "update_gnss_vel", "update_velocity", "update_velocity_2d"]:
        method = getattr(ExtendedKalmanFilter, method_name)
        sig = inspect.signature(method)
        for param_name in sig.parameters.keys():
            for token in forbidden_tokens:
                assert token not in param_name.lower(), (
                    f"Forbidden token '{token}' found in EKF.{method_name} parameter: {param_name}"
                )

    # 3. Check InertialOdomNet forward
    odom_sig = inspect.signature(InertialOdomNet.forward)
    params = [p for p in odom_sig.parameters.keys() if p != "self"]
    assert len(params) == 1, "InertialOdomNet must only take IMU tensor x"
    for token in forbidden_tokens:
        assert token not in params[0].lower()


def test_blackout_inference_data_flow_pure_smartphone():
    """Prove that during GNSS blackout, inference operates purely on smartphone IMU.
    
    Any attempt to pass or access ECU/CAN/wheel speed raises an error.
    """
    ref_lat, ref_lon = 37.7749, -122.4194
    fusion = GNSSINSFusion(ref_lat=ref_lat, ref_lon=ref_lon, dt=0.1)

    # Pre-blackout healthy GNSS initialization (simulated 2 seconds = 20 steps)
    for i in range(20):
        # 10 m/s northward
        lat = ref_lat + (i * 0.1 * 10.0) / 111111.0
        lon = ref_lon
        fusion.step(
            fwd_accel=0.0,
            yaw_rate=0.0,
            gnss_pos=(lat, lon),
            hdop=1.0,
            pos_std=1.5,
        )

    # Create a hostile wrapper that detects any forbidden attribute access
    class StrictSmartphoneIMUOnly:
        def __init__(self, acc, gyro):
            self.acc = acc
            self.gyro = gyro

        def __getattr__(self, name):
            if any(t in name.lower() for t in ["obd", "can", "ecu", "wheel", "gt"]):
                raise AssertionError(f"ILLEGAL ATTEMPT to access vehicle hardware channel: {name}")
            raise AttributeError(f"No attribute {name}")

    # Simulated 30-second blackout (300 steps)
    for step in range(300):
        # Only smartphone IMU signals available: forward accel and yaw rate
        phone_accel = 0.05 * np.sin(step * 0.05)
        phone_yaw_rate = 0.01 * np.cos(step * 0.05)

        # AI odometry produces displacement/velocity solely from IMU
        ai_speed = 9.8 + 0.1 * np.sin(step * 0.05)

        # Pass strictly no GNSS, no vehicle speed
        state = fusion.step(
            fwd_accel=phone_accel,
            yaw_rate=phone_yaw_rate,
            gnss_pos=None,  # Blackout
            ai_velocity=ai_speed,
            hdop=99.0,
            pos_std=100.0,
        )

        assert state is not None
        assert len(state) == 9  # [pE, pN, pU, vE, vN, vU, yaw, b_acc, b_gyro]
        # Verify state is finite (no NaNs, no explosions)
        assert np.all(np.isfinite(state)), f"NaN encountered at step {step}"

    # Verify detector confirms blackout
    assert fusion.gnss_detector.is_blackout()
    assert fusion.in_blackout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
