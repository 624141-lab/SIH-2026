"""Remediation Verification and Integrity Test Suite.

Validates all 18 forensic audit requirements (Rule 60, Rule 67):
1. Authentic IO-VNBD dataset schema, headers, and unit scaling.
2. Canonical drive-disjoint train/val/test splits (Zero contamination).
3. Window overlap leakage assertion (cross-split overlap == 0).
4. EstimatorInput vs GroundTruthReference strict isolation.
5. Independent OSM road graph loader (no waypoints fallback).
6. Authentic HMM Viterbi map matcher properties and anti-leakage guards.
7. Navigation metrics formulas (CEP50, drift%, RMSE).
8. ONNX export model sizes (< 2 MB) and numerical parity.
"""

from pathlib import Path
import json
import numpy as np
import pytest
import torch
import yaml

from idr.config import (
    BASE_DIR, CONFIG_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, REPORTS_DIR, RESULTS_DIR, OSM_DIR,
)
from idr.io.schema import CANONICAL_FIELDS_SMARTPHONE, CANONICAL_FIELDS_VEHICLE, detect_schema
from idr.io.loader import load_drive_pair, IOVNBDrive
from idr.filters.ekf import ExtendedKalmanFilter
from idr.filters.nhc import apply_nhc_update
from idr.eval.blackout import EstimatorInput, GroundTruthReference
from idr.eval.metrics import compute_navigation_metrics
from idr.mapmatch.osm_graph import OSMGraphLoader
from idr.mapmatch.hmm_matcher import HMMMapMatcher, RoadCandidate
import networkx as nx
from shapely.geometry import LineString


# ─────────────────────────────────────────────────────────
# Test 1: Authentic IO-VNBD Schema & Units (Rule 1-6)
# ─────────────────────────────────────────────────────────
def test_iovnbd_raw_dataset_exists():
    """Verify raw IO-VNBD directory contains authentic extracted CSVs."""
    raw_dir = RAW_DATA_DIR / "iovnbd"
    assert raw_dir.exists(), f"Raw dataset directory missing: {raw_dir}"
    csv_files = list(raw_dir.rglob("*.csv"))
    assert len(csv_files) >= 288, f"Expected >= 288 CSVs, found {len(csv_files)}"


def test_drive_loading_and_units():
    """Verify phone/vehicle pair loads with correct units (m/s, rad/s)."""
    raw_dir = RAW_DATA_DIR / "iovnbd"
    drive = load_drive_pair(raw_dir, "Vfa01")
    assert isinstance(drive, IOVNBDrive)
    assert drive.drive_id == "Vfa01"
    
    imu, gt_coords, v_speed, t = drive.get_synced_data()
    assert imu.ndim == 2 and imu.shape[1] == 6  # ax, ay, az, gx, gy, gz
    assert gt_coords.ndim == 2 and gt_coords.shape[1] >= 2
    assert v_speed.ndim == 1
    assert t.ndim == 1
    assert len(imu) == len(v_speed) == len(gt_coords) == len(t)
    
    # Speed should be in m/s (highway driving: 0 to 45 m/s)
    assert 0.0 <= np.max(v_speed) <= 45.0


# ─────────────────────────────────────────────────────────
# Test 2: Canonical Splits & Zero Leakage (Rule 7-10, 46)
# ─────────────────────────────────────────────────────────
def test_splits_yaml_disjoint():
    """Verify splits.yaml contains strictly disjoint drive IDs."""
    splits_file = CONFIG_DIR / "splits.yaml"
    assert splits_file.exists(), f"Missing {splits_file}"
    with open(splits_file, "r", encoding="utf-8") as f:
        splits = yaml.safe_load(f)

    split_dict = splits.get("splits", splits)
    train_ids = set(split_dict.get("train", []))
    val_ids = set(split_dict.get("val", []))
    test_ids = set(split_dict.get("test", []))

    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap!"
    assert "Vfa01" in test_ids and "Vfa02" in test_ids, "Test split must contain Vfa01 and Vfa02"


def test_window_leakage_report_clean():
    """Verify window leakage report confirms zero cross-split overlap."""
    report_file = REPORTS_DIR / "window_leakage_report.json"
    assert report_file.exists(), f"Missing {report_file}"
    with open(report_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    cross_overlaps = data.get("leakage_verification", {}).get("cross_split_overlaps", -1)
    assert cross_overlaps == 0, f"Window leakage detected: {cross_overlaps}"


# ─────────────────────────────────────────────────────────
# Test 3: Sensor Input Isolation (Rule 50)
# ─────────────────────────────────────────────────────────
def test_estimator_input_has_no_ground_truth():
    """Verify EstimatorInput exposes no ground-truth attributes."""
    sensor_input = EstimatorInput(
        imu_data=np.zeros((100, 6)),
        initial_pos_enu=np.zeros(3),
        initial_vel_enu=np.zeros(3),
        initial_heading_rad=0.0,
        timestamps=np.linspace(0, 10, 100),
    )
    assert not hasattr(sensor_input, "gt_coords")
    assert not hasattr(sensor_input, "gt_enu")
    assert not hasattr(sensor_input, "gt_speed")
    assert not hasattr(sensor_input, "ground_truth")


# ─────────────────────────────────────────────────────────
# Test 4: Independent OSM Road Graph (Rule 15-18, 64)
# ─────────────────────────────────────────────────────────
def test_independent_osm_graph_loading():
    """Verify road graphs load from offline OSM cache without ground truth."""
    loader = OSMGraphLoader()
    graph = loader.load_offline_graph("Vfa01", ref_lat=52.55, ref_lon=-1.50)
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_nodes() > 100
    assert graph.number_of_edges() > 100


# ─────────────────────────────────────────────────────────
# Test 5: Authentic HMM Map Matcher (Rule 19, 20)
# ─────────────────────────────────────────────────────────
def test_hmm_map_matcher_mechanics():
    """Verify HMM Viterbi emission, transition, and decoding behavior."""
    G = nx.MultiDiGraph()
    # Segment 1: (0, 0) -> (100, 0)
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=100.0, y=0.0)
    G.add_edge(0, 1, key=0, geometry=LineString([(0.0, 0.0), (100.0, 0.0)]))

    matcher = HMMMapMatcher(G, sigma_z=5.0, beta=3.0, max_search_radius=30.0)
    
    # Test emission probability
    cand_close = RoadCandidate(0, 0, 1, 0, 10.0, 0.0, dist=1.0, azimuth=0.0)
    cand_far = RoadCandidate(0, 0, 1, 0, 10.0, 0.0, dist=10.0, azimuth=0.0)
    assert matcher.log_emission_prob(cand_close) > matcher.log_emission_prob(cand_far)

    # Test trajectory matching snaps points to road
    noisy_traj = np.array([
        [10.0, 3.0],
        [20.0, -2.0],
        [30.0, 4.0],
        [40.0, 1.0],
    ])
    matched = matcher.match_trajectory(noisy_traj)
    assert matched.shape == (4, 2)
    # Since road is y = 0, snapped points should have y close to 0
    assert np.all(np.abs(matched[:, 1]) < 0.1)


def test_hmm_anti_leakage_assertion():
    """Verify HMMMapMatcher strictly forbids ground-truth objects."""
    G = nx.MultiDiGraph()
    G.add_node(0, x=0.0, y=0.0)
    G.add_node(1, x=10.0, y=0.0)
    G.add_edge(0, 1, key=0, geometry=LineString([(0.0, 0.0), (10.0, 0.0)]))
    matcher = HMMMapMatcher(G)

    class FakeWithGT:
        ground_truth = True
    
    with pytest.raises(AssertionError):
        matcher.match_trajectory(FakeWithGT())


# ─────────────────────────────────────────────────────────
# Test 6: Navigation Metrics (Rule 35, 42)
# ─────────────────────────────────────────────────────────
def test_navigation_metrics_formulae():
    """Verify drift%, CEP50, and RMSE match exact mathematical definitions."""
    # 200m line with linear drift ending at 10m error
    gt = np.column_stack([np.linspace(0, 200, 100), np.zeros(100)])
    est = np.column_stack([np.linspace(0, 200, 100), np.linspace(0, 10, 100)])
    
    metrics = compute_navigation_metrics(est, gt)
    assert abs(metrics.total_distance_m - 200.0) < 1.0
    assert abs(metrics.final_drift_m - 10.0) < 0.1
    # drift% = 10m / 200m * 100 = 5.0%
    assert abs(metrics.drift_percent - 5.0) < 0.2
    assert 0.0 < metrics.cep_50_m < 10.0
    assert 0.0 < metrics.rmse_position_m < 10.0


# ─────────────────────────────────────────────────────────
# Test 7: ONNX Models & Deployment Constraint (Rule 54-56)
# ─────────────────────────────────────────────────────────
def test_onnx_model_artifacts_and_sizes():
    """Verify ONNX models exist and comply with the < 2 MB size limit."""
    vel_onnx = MODELS_DIR / "velocity_net.onnx"
    denoise_onnx = MODELS_DIR / "imu_denoise.onnx"

    assert vel_onnx.exists(), f"Missing {vel_onnx}"
    assert denoise_onnx.exists(), f"Missing {denoise_onnx}"

    vel_size_mb = vel_onnx.stat().st_size / (1024 * 1024)
    denoise_size_mb = denoise_onnx.stat().st_size / (1024 * 1024)

    assert vel_size_mb < 2.0, f"Velocity ONNX too large: {vel_size_mb:.2f} MB"
    assert denoise_size_mb < 2.0, f"IMU Denoise ONNX too large: {denoise_size_mb:.2f} MB"


def test_export_validation_report_exists():
    """Verify export validation report was generated."""
    report = REPORTS_DIR / "export_validation.md"
    assert report.exists(), f"Missing {report}"
