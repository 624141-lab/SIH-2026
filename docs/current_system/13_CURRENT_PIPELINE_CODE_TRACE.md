# 13 — CURRENT PIPELINE CODE TRACE: DEVELOPER NAVIGATION MAP

---

## 1. Master Execution Call Stack

When you run the primary benchmark ([`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py)) or the proposal figure generator ([`scripts/generate_proposal_figures.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/generate_proposal_figures.py)), execution follows this exact call tree:

```text
scripts/evaluate_hardened_pipeline.py
  │
  ├──► run_hardened_evaluation()                          [Line 202]
  │     │
  │     ├──► load_drive_pair()                            [src/idr/io/loader.py:160]
  │     │     ├──► detect_schema()                        [src/idr/io/loader.py:32]
  │     │     ├──► standardize_phone_dataframe()          [src/idr/io/loader.py:65]
  │     │     └──► standardize_vehicle_dataframe()        [src/idr/io/loader.py:100]
  │     │
  │     ├──► PhoneToVehicleAligner()                      [src/idr/calib/alignment.py:175]
  │     │     ├──► compute_initial_alignment()            [src/idr/calib/alignment.py:283]
  │     │     │     ├──► detect_stationary_windows()      [src/idr/calib/alignment.py:73]
  │     │     │     └──► update_gravity_vector()          [src/idr/calib/alignment.py:199]
  │     │     └──► transform_imu()                        [src/idr/calib/alignment.py:382]
  │     │
  │     ├──► OSMGraphLoader.load_offline_graph()          [src/idr/mapmatch/osm_loader.py]
  │     │     └──► CausalHMMMapMatcher()                  [src/idr/mapmatch/causal_matcher.py:50]
  │     │
  │     ├──► InertialOdomNet()                            [src/idr/models/inertial_odom.py:52]
  │     │     └──► torch.load("models/inertial_odom.pt")
  │     │
  │     └──► Loop over 50 scenarios:
  │           │
  │           └──► run_single_hardened_scenario()         [scripts/evaluate_hardened_pipeline.py:50]
  │                 │
  │                 ├──► Baseline 1: Raw IMU Mechanization  [Lines 76–86]
  │                 │
  │                 ├──► Baseline 2: Standard EKF          [Lines 91–105]
  │                 │     └──► ExtendedKalmanFilter()     [src/idr/filters/ekf.py:5]
  │                 │
  │                 ├──► Baseline 3: EKF + Rigid NHC      [Lines 110–118]
  │                 │     └──► apply_nhc_update()         [src/idr/filters/nhc.py:26]
  │                 │
  │                 ├──► Baseline 4: EKF + NHC + Map      [Line 124]
  │                 │     └──► match_trajectory_causal()  [src/idr/mapmatch/causal_matcher.py:427]
  │                 │
  │                 ├──► System 5: AI + Adaptive NHC      [Lines 129–178]
  │                 │     ├──► VibrationMotionFilter()    [src/idr/filters/vibration_filter.py:27]
  │                 │     │     └──► vfilter.process()    [src/idr/filters/vibration_filter.py:54]
  │                 │     ├──► ekf.predict()              [src/idr/filters/ekf.py:37]
  │                 │     ├──► apply_adaptive_nhc_update()[src/idr/filters/nhc.py:100]
  │                 │     ├──► odom_model(windows)        [src/idr/models/inertial_odom.py:92]
  │                 │     └──► ekf.update_velocity()      [src/idr/filters/ekf.py:120]
  │                 │
  │                 └──► System 6: Full Hardened Pipeline [Line 183]
  │                       └──► match_trajectory_causal()  [src/idr/mapmatch/causal_matcher.py:427]
  │                             └──► causal_matcher.step()[src/idr/mapmatch/causal_matcher.py:255]
  │
  └──► Compute Metrics & Save CSV                         [Lines 340–435]
        └──► "results/hardened/hardened_per_scenario_metrics.csv"
```

---

## 2. Code Navigation Directory (Where to Find What)

| System Stage | File Path | Primary Class / Function | What to Look For |
| :--- | :--- | :--- | :--- |
| **Data Ingestion** | [`src/idr/io/loader.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/io/loader.py) | `load_drive_pair()` | Timestamp interpolation, CSV schema normalization |
| **Attitude Alignment**| [`src/idr/calib/alignment.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/calib/alignment.py) | `PhoneToVehicleAligner` | Leveling Rodrigues rotation, dynamic forward yaw tracking |
| **Vibration Filter** | [`src/idr/filters/vibration_filter.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/vibration_filter.py) | `VibrationMotionFilter` | Rolling variance, shock damping, `cov_scale` calculation |
| **AI Speed Model** | [`src/idr/models/inertial_odom.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/models/inertial_odom.py) | `InertialOdomNet` | 1D-CNN dilated blocks, causal GRU, regression head |
| **Kalman Filter** | [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) | `ExtendedKalmanFilter` | 9-state ENU vector, process Jacobian $F$, measurement updates |
| **Non-Holonomic** | [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py) | `apply_adaptive_nhc_update()` | Centripetal variance inflation, Chi-square gate ($NIS \le 9.21$) |
| **Road Map Matcher** | [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py) | `CausalHMMMapMatcher` | Forward trellis, 30m corridor gate, `use_finalized=False` |
| **GNSS Monitor** | [`src/idr/filters/gnss_monitor.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/gnss_monitor.py) | `GNSSDeficitDetector` | 4-state FSM, HDOP thresholds, outage debouncing |
| **Transition Filter**| [`src/idr/eval/transition.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/eval/transition.py) | `ReacquisitionSmoother` | $C^1$ cosine bell transition over 3.5 seconds |
| **Benchmark Runner** | [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) | `run_hardened_evaluation()` | Canonical 50-scenario evaluation across B1–S6 |
| **Figure Generator** | [`scripts/generate_proposal_figures.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/generate_proposal_figures.py) | `main()` | Renders `fig1`–`fig5` for SIH proposal slides |
