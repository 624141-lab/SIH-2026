# 22 — CODE-TO-CONCEPT INDEX: QUICK REPOSITORY LOOKUP

---

## Developer Quick Lookup Matrix

When you or an evaluator asks a specific question about an equation or algorithm, use this index to find the exact source file and line number immediately:

| Concept / Algorithmic Question | Source File | Class / Function | Exact Line / Reference |
| :--- | :--- | :--- | :--- |
| **Where is the raw dataset loaded?** | [`src/idr/io/loader.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/io/loader.py) | `load_drive_pair()` | Lines 160–198 |
| **Where are phone and vehicle clocks aligned?** | [`src/idr/io/loader.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/io/loader.py) | `load_drive_pair()` | Lines 183–190 (`np.interp`) |
| **Where is stationary vehicle detected?** | [`src/idr/calib/alignment.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/calib/alignment.py) | `detect_stationary_windows()` | Lines 73–140 |
| **Where is the phone leveling rotation computed?** | [`src/idr/calib/alignment.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/calib/alignment.py) | `update_gravity_vector()` | Lines 215–231 (Rodrigues formula) |
| **Where is the phone forward yaw axis found?** | [`src/idr/calib/alignment.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/calib/alignment.py) | `update_forward_motion()` | Lines 268–310 |
| **Where are gravity and chassis vibrations separated?**| [`src/idr/filters/vibration_filter.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/vibration_filter.py) | `VibrationMotionFilter.process()`| Lines 74–77 ($\alpha=0.95$ low-pass) |
| **Where is dynamic covariance scale ($s_{\text{cov}}$) calculated?**| [`src/idr/filters/vibration_filter.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/vibration_filter.py) | `process()` | Lines 120–145 |
| **Where is the AI odometry architecture defined?** | [`src/idr/models/inertial_odom.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/models/inertial_odom.py) | `InertialOdomNet` | Lines 52–91 (1D-CNN + GRU) |
| **Where is AI forward pass called during testing?** | [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) | `run_single_hardened_scenario()` | Lines 149–153 |
| **Where is AI displacement converted to velocity?** | [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) | `run_single_hardened_scenario()` | Line 172: `v_ai = dx_b / (50*dt)` |
| **Where is AI velocity injected into the EKF?** | [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) | `run_single_hardened_scenario()` | Line 175: `ekf.update_velocity()` |
| **Where is the EKF state vector initialized?** | [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) | `__init__()` | Lines 18–36 (9-state ENU) |
| **Where are the EKF mechanization equations?** | [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) | `predict()` | Lines 37–65 ($a_E, a_N, v_E, v_N$) |
| **Where is the EKF process Jacobian ($F$) computed?** | [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) | `predict()` | Lines 66–78 |
| **Where is the AI velocity measurement update equation?**| [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) | `update_velocity()` | Lines 120–147 |
| **Where is the adaptive NHC variance formula?** | [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py) | `apply_adaptive_nhc_update()` | Lines 134–138 ($\sigma^2 \propto a_c^2$) |
| **Where is the NHC heading observability Jacobian?** | [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py) | `apply_adaptive_nhc_update()` | Line 146: $H[0,6] = -\cos\psi v_E - \sin\psi v_N$ |
| **Where is the NHC Chi-square gate evaluated?** | [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py) | `apply_adaptive_nhc_update()` | Lines 154–156 ($NIS > 9.21$) |
| **Where is the spatial road KD-Tree constructed?** | [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py) | `_extract_road_segments()` | Lines 76–115 (`cKDTree`) |
| **Where is the Causal HMM forward step executed?** | [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py) | `step()` | Lines 255–425 |
| **Where is the 30 m map corridor gate checked?** | [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py) | `step()` | Line 361: `prov_c.dist > 30.0` |
| **Where is zero look-ahead enforced in map matching?** | [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) | `run_single_hardened_scenario()` | Line 183: `use_finalized=False` |
| **Where is the GNSS outage detector state machine?** | [`src/idr/filters/gnss_monitor.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/gnss_monitor.py) | `GNSSDeficitDetector.update()` | Lines 56–125 |
| **Where is the cosine-bell reacquisition smoother?** | [`src/idr/eval/transition.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/eval/transition.py) | `ReacquisitionSmoother` | Lines 45–51 ($C^1$ cosine weight) |
| **Where are the 5 proposal figures generated?** | [`scripts/generate_proposal_figures.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/generate_proposal_figures.py) | `main()` | Lines 170–370 |
| **Where is the official 50-scenario benchmark run?** | [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) | `run_hardened_evaluation()` | Lines 202–465 |
| **Where are final drift and errors calculated?** | [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) | `run_single_hardened_scenario()` | Lines 328–360 |
