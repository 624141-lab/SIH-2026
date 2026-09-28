# TABLE H — ERROR SOURCE & FAILURE MODE ANALYSIS

### Overview
This table documents the primary physical and algorithmic error sources identified through empirical ablation and scenario decomposition across the 50 evaluated scenarios.

---

### Experimentally Verified Failure Mechanisms

| Failure Mechanism | Physical & Algorithmic Cause | Empirical Evidence from Benchmark | Affected Scenarios | Severity / Impact on Drift | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Open-Loop Gyroscope Drift** | Unmodeled MEMS gyro bias integration ($\sim 0.1\text{–}0.5^\circ/\text{s}$) rotates velocity vector over time | Causes 98–145% drift in B1–B3 baselines | Universal across all unconstrained runs | **HIGH (>100% drift if uncorrected)** | Adaptive NHC + OSM Road Map heading alignment |
| **Map Snapping in Unmapped Rural Areas** | HMM snaps dead-reckoning trajectory to false road segments when roads are missing in OSM | S6 error increases from 241 m to 350 m on `Vfa02` (62.5% of `Vfa02` degraded) | Drive `Vfa02` rural scenarios (e.g. `Vfa02_t10s_d30s`) | **HIGH (Up to 311% drift in worst case)** | 30 m Corridor gating; switch to Visual-Inertial Odometry (Phase 2) |
| **Along-Track Speed Scale Lag** | AI neural network slightly underestimates peak highway acceleration / deceleration | Produces 30–80 m longitudinal offset while cross-track error remains < 4 m | Highway runs (e.g. `Vfa01_t70s_d60s`, along-track 90.7 m) | **MODERATE (5–8% drift on 1 km outages)** | GRU window temporal expansion + zero-lag residual scaling |
| **Stationary Bias Accumulation** | Stopped vehicle continues integrating noisy accelerometer bias without ZUPT | `Vfa02_t10s_d30s` spent 8.8 s at 0 km/h; accumulated 546 m error | Scenarios with traffic stops / intersections | **CRITICAL (Generates false velocity drift)** | Zero Velocity Update (ZUPT) detector triggered when $|\mathbf{a}| \approx g$ and $\|\boldsymbol{\omega}\| \approx 0$ |
| **Centripetal Slip during Cornering** | Classical NHC assumes zero lateral velocity ($v_{\text{lat}}=0$), which is violated during sharp cornering | Conventional EKF distorts vehicle heading when lateral acceleration exceeds $4\text{ m/s}^2$ | High yaw-rate curves (>20 deg/s) | **MODERATE (Distorts heading estimate)** | Adaptive centripetal NHC noise inflation: $\sigma_{\text{lat}}^2 \propto (v_{\text{fwd}} \cdot \|\omega_z\|)^2$ |

---

### Dominant Error Decomposition on Mapped 1.14 km Outage (`Vfa01_t70s_d60s`)
- **Total Position Error:** $90.84\text{ m}$ ($7.97\%$ drift).
- **Along-Track (Longitudinal) Error:** $90.72\text{ m}$ (**99.8% of residual error** due to neural odometry speed integration lag).
- **Cross-Track (Lateral) Error:** $4.15\text{ m}$ (**0.2% of residual error**; road map geometry bounds lateral divergence).
