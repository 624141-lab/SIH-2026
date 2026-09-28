# SIH_IDR_Prototype_Compliance_Audit_Report
**Technical Compliance, Algorithmic Integrity, and Mobile Deployment Audit**  
**Problem Statement**: SIH 26168 — Intelligent Dead Reckoning (IDR) System with GNSS Fusion (ISRO)  
**Evaluator Roles**: Senior Navigation/INS Engineer, AI/ML Engineer, Mobile Systems Engineer, SIH Technical Evaluator  
**Audit Date**: September 28, 2026  
**Repository**: `varshithkuravati/SIH` (`AI_gemini_pro`)

---

## 1. Executive Summary

### 1.1 Prototype Maturity Assessment
The repository represents a **high-quality, mathematically sound algorithmic research and desktop simulation prototype (Maturity Level B+)** with verified ONNX model exports. It demonstrates rigorous compliance with dataset integrity rules, zero ground-truth leakage, and authentic empirical benchmarking on the official IEEE IO-VNBD dataset.

However, from an **operational and deployment standpoint (Maturity Level D/E)**, the prototype has critical gaps that would be flagged by a strict SIH technical evaluation panel:
1. **No Mobile Application**: There is zero native mobile code (no Android/Kotlin or iOS/Swift application, no UI, no live sensor background service).
2. **AI Speed Estimation Uncertainty**: `VelocityEstimatorNet` achieves an MAE of 11.51 m/s (41.45 km/h) and $R^2 = -2.22$ on held-out test drives when relying purely on IMU vibration without wheel speed or odometry history, proving that unaccelerated highway cruising speed cannot be observed from vibration amplitude alone.
3. **Dead Reckoning Drift Threshold**: While the system achieves a **25.0% pass rate** (<10% drift) across 100 authentic scenarios and slashed the 1 km target outage drift from 57.55% down to **15.83%** (a 72.5% error reduction), the $<10\%$ drift criterion is not yet uniformly met across 60-second outages.
4. **Disconnected Alignment & Smoothing Modules**: Key engineering modules—specifically `PhoneToVehicleAligner` (attitude calibration) and `ReacquisitionSmoother` (cosine blend transition)—exist in the codebase but are **not integrated** into the active evaluation pipeline.
5. **No External IMU Abstraction**: The IO pipeline is tightly coupled to IO-VNBD CSV headers; generic streaming interfaces (Serial, UDP, ROS2) for external/edge IMUs are absent.

### 1.2 Summary Scorecard
- **Satisfied Requirements**: 4 / 13 (Authentic Dataset, Independent OSM HMM Map Matching, Core Sensor Fusion, Model Export & Footprint)
- **Partially Satisfied Requirements**: 6 / 13 (GNSS/INS Transition, Attitude Alignment, AI Speed Filter, Dead Reckoning Performance, 10 Hz Processing, Kinematic NHC)
- **Not Satisfied Requirements**: 3 / 13 (Real-Time Mobile Application UI, External IMU Hardware Abstraction, Automatic GNSS Deficit Detector)

---

## 2. Requirement Compliance Matrix

| SIH Requirement | Status | Concrete Evidence | Current Implementation | Technical Gap | Required Technical Fix | Verification Method |
|---|---|---|---|---|---|---|
| **1. GNSS $\to$ INS Blackout Switching** | **PARTIALLY SATISFIED** | `src/idr/filters/fusion.py:75-135` | Boolean `is_gnss_denied` controls EKF measurement updates. | No automatic signal quality / SNR / HDOP monitoring; caller must flag blackout. | Implement `GNSSSignalMonitor` tracking satellite count, HDOP, and residual variance. | Inject synthetic multipath & SNR drops in real drive streams. |
| **2. Road-Constrained Navigation** | **SATISFIED** | `src/idr/mapmatch/hmm_matcher.py`, `src/idr/eval/blackout.py:213-235` | Offline OSM road networks cached; closed-loop centerline guidance + HMM Viterbi trellis. | Offline bounding box limited to test routes; no 3D vertical ramp disambiguation. | Add multi-layer elevation tags and spatial graph tiling. | Test intersection branching on dense urban road network. |
| **3. Seamless GNSS Re-acquisition** | **PARTIALLY SATISFIED** | `src/idr/eval/transition.py` | `ReacquisitionSmoother` implemented with cosine bell blend ($C^1$ continuous). | `ReacquisitionSmoother` is **not integrated** into `fusion.py`; filter experiences discontinuous position snap upon first fix. | Integrate `ReacquisitionSmoother` into `GNSSINSFusion.step()` with Mahalanobis gating. | Measure position derivative (velocity spike) upon re-acquisition. |
| **4. Zero Vehicle Physical Connection** | **SATISFIED** | `src/idr/io/loader.py`, `src/idr/eval/blackout.py` | Estimator strictly consumes smartphone 6-axis IMU; vehicle ECU speed is strictly isolated. | None for algorithm; relies on smartphone sensors. | Maintain strict isolation in mobile deployment. | Static code inspection (`EstimatorInput` isolation). |
| **5. AI Speed Estimation from IMU** | **PARTIALLY SATISFIED** | `src/idr/models/velocity_net.py`, `results/velocity_predictions.csv` | Dilated 1D-CNN + GRU + Stationary Gate predicts speed from 50-step IMU window. | Speed estimation fails during unaccelerated cruising (MAE=11.51 m/s, $R^2=-2.22$); requires pre-outage speed anchoring. | Transition to 2D Displacement Odometry (`InertialOdomNet`) with heteroscedastic uncertainty. | Evaluate test MAE on varied vehicle platforms. |
| **6. Non-Navigation Motion Rejection** | **PARTIALLY SATISFIED** | `src/idr/filters/zupt.py`, `src/idr/models/velocity_net.py:53` | Two-stage variance detector (`acc_var < 0.15`, `gyro_norm < 0.05`) + softplus/sigmoid gate. | Rejects idle stationary vibration, but shock from potholes/bumps can trigger false velocity changes. | Add peak kurtosis and frequency-domain spectral notch filtering. | Test with simulated pothole impulse signals ($>2g$). |
| **7. Phone Mounting & Attitude Alignment** | **PARTIALLY SATISFIED** | `src/idr/calib/alignment.py` | Gravity vector (stationary) + PCA of longitudinal acceleration for forward axis. | **Not called in evaluation pipeline**; assumes pre-aligned phone cradle from IO-VNBD; yaw unobservable when stationary. | Integrate `PhoneToVehicleAligner` into pipeline initialization; fuse magnetometer/COG. | Run benchmark with synthetic phone rotation matrix ($R_{pitch=30^\circ, roll=15^\circ}$). |
| **8. Map Matching & OSM Constraints** | **SATISFIED** | `src/idr/mapmatch/` | Offline Overpass OSM graph + Newson & Krumm HMM Viterbi matching. | Trellis latency scales with edge count (156k edges in Vfa02); needs spatial R-tree caching. | Pre-index directed edges into uniform spatial grid cells. | Measure HMM query latency across 100 scenarios. |
| **9. Non-Holonomic Constraints (NHC)** | **SATISFIED** | `src/idr/filters/nhc.py`, `src/idr/eval/blackout.py:207-211` | Lateral velocity constrained ($v_{lat} \equiv 0$); kinematic heading integration. | Standard EKF centripetal term caused divergence; solved via kinematic vehicle mechanization. | Port kinematic car model into unified Error-State Kalman Filter (ESKF). | Verify lateral drift $\le 3.0\text{ m}$ on straight highway. |
| **10. AI/ML GNSS+INS Sensor Fusion** | **SATISFIED** | `src/idr/filters/`, `src/idr/models/` | 9-state EKF + kinematic projection + AI velocity anchoring + ZUPT/ZARU. | AI model output currently enters as scalar forward speed instead of full covariance displacement. | Feed `[dx, dy]` with learned covariance into EKF measurement update. | Benchmark covariance consistency and NEES. |
| **11. On-Device Mobile Execution** | **PARTIALLY SATISFIED** | `models/velocity_net.onnx`, `models/imu_denoise.onnx` | ONNX Opset 14 exports verified (<320 KB each, <1e-5 parity error, <1 ms latency). | No mobile application wrapper (no Android APK, no iOS IPA). | Develop Android native application with ONNX Runtime Mobile (`onnxruntime-android`). | Measure APK size, frame rate, and battery drain on Android device. |
| **12. Smartphone IMU Input Support** | **PARTIALLY SATISFIED** | `src/idr/io/schema.py` | Parses authentic smartphone IMU data from IO-VNBD (10 Hz). | Read from pre-recorded CSV; no Android `SensorEventListener` live streaming harness. | Write Android JNI / Kotlin service reading `TYPE_ACCELEROMETER` and `TYPE_GYROSCOPE`. | Test live sensor latency on physical smartphone. |
| **13. External IMU / Edge Deployable Engine** | **NOT SATISFIED** | `src/idr/io/` | Schema is hardcoded to IO-VNBD CSV format. | No generic streaming API, no serial/socket reader, no 200 Hz FOG IMU interface. | Create `BaseIMUStream` interface with configurable sampling rate, axes, and units. | Stream simulated 200 Hz FOG data through a local UDP socket. |

---

## 3. System Architecture Audit

### 3.1 Current Prototype Architecture
```
[Raw IO-VNBD CSVs]
       │
       ▼
[loader.py / preprocessor.py] (Latin-1, Unit Normalization, Canonical Splits)
       │
       ▼
[EstimatorInput] ─── (Isolated: NO Ground Truth, NO GNSS during Outage)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
[Pre-Blackout GNSS Fixes]                 [Phone IMU (ax..gz)]
       │                                         │
[Butterworth 2nd-Order LPF]                      ▼
       │                                  [StationaryDetector (ZUPT/ZARU)]
       ├────────────────────────┐                │
       ▼                        ▼                ▼
[Acc Bias b_a]          [Gyro Bias b_w]   [Kinematic Velocity Mechanization]
                                                 │
                                                 ▼
                                     [Closed-Loop Road Guidance] (every 1.0s)
                                                 │
                                                 ▼
                                     [HMM Viterbi Map Matcher] (Offline OSM Graph)
                                                 │
                                                 ▼
                                     [Estimated ENU Trajectory]
```

### 3.2 Recommended Target Architecture for SIH Final Deployment
```
                    ┌────────────────────────────────────────────────────────┐
                    │                    SENSOR INGESTION LAYER              │
                    │  ┌──────────────────────┐    ┌──────────────────────┐  │
                    │  │ Android SensorManager│    │ External IMU Stream  │  │
                    │  │ (100 Hz Accel/Gyro)  │    │ (200 Hz FOG via UDP) │  │
                    │  └──────────┬───────────┘    └──────────┬───────────┘  │
                    └─────────────┼───────────────────────────┼──────────────┘
                                  ▼                           ▼
                    ┌────────────────────────────────────────────────────────┐
                    │               PREPROCESSING & CALIBRATION               │
                    │  - Unified Sensor Stream Abstraction (Hz, Units, Axes) │
                    │  - PhoneToVehicleAligner (Attitude Quaternion q_p2v)   │
                    │  - Dual-Threshold Stationary Detector (ZUPT/ZARU)      │
                    └─────────────────────────────┬──────────────────────────┘
                                                  ▼
                    ┌────────────────────────────────────────────────────────┐
                    │               AI/ML DEEP INERTIAL ODOMETRY              │
                    │  - InertialOdomNet (ONNX Runtime Mobile, INT8 / FP16)  │
                    │  - Input: 50-step IMU window (100 Hz)                  │
                    │  - Output: Body-frame displacement [dx, dy] + Sigma    │
                    └─────────────────────────────┬──────────────────────────┘
                                                  ▼
                    ┌────────────────────────────────────────────────────────┐
                    │           ERROR-STATE KALMAN FILTER (ESKF-15)          │
                    │  - States: [delta_p (3), delta_v (3), delta_theta (3), │
                    │             delta_ba (3), delta_bg (3)]                │
                    │  - Updates: GNSS (when valid), NHC (v_lat=0, v_up=0),  │
                    │             AI Odometry [dx, dy], ZUPT, ZARU           │
                    │  - GNSS Deficit Detector (HDOP, SNR, Chi-Square Test)  │
                    │  - ReacquisitionSmoother (C1 Continuous Cosine Blend)  │
                    └─────────────────────────────┬──────────────────────────┘
                                                  ▼
                    ┌────────────────────────────────────────────────────────┐
                    │              ADVANCED MAP-MATCHING ENGINE              │
                    │  - Spatial R-Tree Grid Indexing of Local OSM GeoJSON   │
                    │  - Hidden Markov Model (Newson & Krumm Viterbi Trellis)│
                    │  - Closed-Loop Heading & Centerline Orthogonal Snapping│
                    └─────────────────────────────┬──────────────────────────┘
                                                  ▼
                    ┌────────────────────────────────────────────────────────┐
                    │             MOBILE USER INTERFACE (MAPVIEW)            │
                    │  - Mapbox / MapLibre Native Offline Vector Map         │
                    │  - 60 FPS Interpolated Vehicle Heading/Position Marker │
                    │  - Live Navigation HUD (Speed, Outage Status, Drift)   │
                    └────────────────────────────────────────────────────────┘
```

---

## 4. AI/ML Model Audit

### Model 1: `VelocityEstimatorNet`
- **Purpose**: Estimate scalar forward velocity from smartphone IMU vibrations.
- **Architecture**: 3-layer Dilated 1D-CNN (receptive field = 15 steps) + 2-layer GRU (64 hidden units) + Linear regressor + Sigmoid motion gate.
- **Input**: `(B, 6, 50)` — 6-axis IMU window (5.0s at 10 Hz).
- **Output**: `(B, 1)` — Estimated forward speed ($m/s$).
- **Training Loss**: Smooth L1 (Huber) Loss vs ECU CAN wheel speed.
- **Model Parameters**: 79,394 parameters.
- **Deployed Artifact**: `models/velocity_net.onnx` (314.9 KB, Opset 14).
- **Desktop Latency**: Mean = **0.926 ms**, P95 = **1.930 ms**.
- **Empirical Performance on Held-Out Test Drives (`Vfa01`, `Vfa02`)**:
  - **MAE**: **11.51 m/s (41.45 km/h)**
  - **RMSE**: **12.98 m/s (46.71 km/h)**
  - **Mean Relative Error**: **60.18%**
  - **Mean Bias**: **-10.93 m/s** (systematic underestimation)
  - **$R^2$ Score**: **-2.22**
- **Engineering Diagnosis**: Unaccelerated cruising speed has zero linear acceleration. The vibration spectrum depends primarily on road roughness and engine mount stiffness, not speed. A network given only raw IMU without wheel integration regresses to the training distribution mean (~40 km/h).
- **Required Fix**: Upgrade to `InertialOdomNet` predicting 2D relative displacement `[dx, dy]` with learned uncertainty.

### Model 2: `IMUDenoiseNet`
- **Purpose**: Filter non-navigation noise and estimate sensor bias residuals.
- **Architecture**: 3-layer 1D CNN with residual bypass connections.
- **Input**: `(B, 6, 50)` — 6-axis IMU window.
- **Output**: `(B, 6)` — Sensor bias residuals $[\Delta a_x, \Delta a_y, \Delta a_z, \Delta \omega_x, \Delta \omega_y, \Delta \omega_z]$.
- **Parameters**: 53,414 parameters.
- **Deployed Artifact**: `models/imu_denoise.onnx` (209.2 KB, Opset 14).
- **Desktop Latency**: Mean = **0.358 ms**, P95 = **0.692 ms**.
- **Audit Finding**: When pre-blackout GNSS is available, Butterworth low-pass filtering on pre-blackout GNSS positions provides superior physical bias calibration ($\hat{b}_a, \hat{b}_\omega$). Subtracting `IMUDenoiseNet` residuals concurrently with low-pass biases caused double-bias correction, degrading performance from 37% to 62% median drift. Fixed by making neural denoising conditional on the absence of pre-blackout GNSS.

---

## 5. IO-VNBD Evaluation & Dataset Integrity Audit

### 5.1 Dataset Authenticity & Separation
- **Data Source**: Authentic IEEE IO-VNBD dataset (Onyekpeu et al., Data in Brief 2021).
- **Integrity**: All 288 CSV files verified with Latin-1 encoding, trimmed headers, and canonical unit conversions (km/h $\to$ m/s, deg/s $\to$ rad/s).
- **Split Rigor**:
  - **Train Drives (14)**: `M_M`, `S1`, `S2`, `S3a`, `S3c`, `Vta1a`, `Vta2`, `Vta16`, `Vta29`, `Vtb1`, `Vtb5`, `Vw1`, `Vw2`, `Vw4`
  - **Validation Drives (1)**: `Y1`
  - **Test Drives (2)**: `Vfa01`, `Vfa02` (Held-out trips)
- **Window Leakage Audit**: Tested across 83,020 windows using `scripts/validate_iovnbd.py`: **0 cross-split window overlaps** (disjoint routes and sessions).
- **Ground Truth Isolation**: The dead-reckoning engine consumes strictly `EstimatorInput`. Zero ground-truth coordinates or velocities enter during simulated outages.

---

## 6. Performance Benchmark Results

### 6.1 Quantitative Multi-Scenario Benchmark (100 Authentic Outages)
Evaluated across 100 deterministic blackout intervals (15s, 30s, 60s) on held-out test drives `Vfa01` and `Vfa02`:

| Configuration | Scenarios | Median Drift (%) | Mean Drift (%) | P90 Drift (%) | Mean RMSE (m) | Mean CEP50 (m) | Pass Rate (<10%) |
|---|---|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 100 | **29.18%** | 38.66% | 64.67% | 126.48 m | 88.76 m | **16.0% (16/100)** |
| **Config B: EKF + Kinematic NHC** | 100 | **37.17%** | 45.12% | 93.56% | 169.62 m | 114.82 m | **9.0% (9/100)** |
| **Config C: EKF + NHC + OSM HMM** | 100 | **22.66%** | **38.48%** | **87.97%** | **135.15 m** | **112.78 m** | **25.0% (25/100)** |

### 6.2 Target 1 km Outage Performance (`Vfa01_t725_d60s`, 60-Second Outage)

| Configuration | Distance Travelled | Final Position Error | Drift % | RMSE | CEP50 | Status (<10%) |
|---|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 1012.54 m | 472.43 m | 46.66% | 231.84 m | 185.20 m | FAIL |
| **Config B: EKF + Kinematic NHC** | 1012.54 m | 639.03 m | 63.11% | 275.60 m | 210.45 m | FAIL |
| **Config C: EKF + NHC + OSM HMM** | 1012.54 m | **160.32 m** | **15.83%** | **78.42 m** | **62.15 m** | FAIL (Near Pass) |

*Progress Note*: On this exact 1 km test, Config C final drift was reduced from **582.76 m (57.55%)** down to **160.32 m (15.83%)**—a **72.5% error reduction**.

### 6.3 1 km Range Average (800 m – 1200 m Outages, 12 Scenarios)
- **Config A**: Mean Distance = 935.28 m | Mean Error = 299.69 m | Mean Drift = 30.52%
- **Config B**: Mean Distance = 935.28 m | Mean Error = 415.13 m | Mean Drift = 42.58%
- **Config C**: Mean Distance = 935.28 m | Mean Error = **256.62 m** | Mean Drift = **27.00%** (Down from 675.04 m / 69.89%)

### 6.4 Runtime & Compute Benchmark (Measured on Local CPU)

| Subsystem Component | Execution Rate | Mean Latency | P95 Latency | Memory (RSS) | Model Size | Mobile Constraint (<2MB) |
|---|---|---|---|---|---|---|
| **VelocityEstimatorNet** | 10 Hz | 0.926 ms | 1.930 ms | ~45 MB | 314.9 KB | ✅ PASS |
| **IMUDenoiseNet** | 10 Hz | 0.358 ms | 0.692 ms | ~30 MB | 209.2 KB | ✅ PASS |
| **EKF Kinematic Predict/Update** | 10 Hz | 0.180 ms | 0.334 ms | < 5 MB | N/A | ✅ PASS |
| **HMM Spatial Query & Viterbi** | 1 Hz / 10 Hz | 1.712 ms | 2.572 ms | ~65 MB | N/A | ✅ PASS |
| **Full Pipeline Integration** | **10 Hz** | **3.176 ms** | **5.528 ms** | **~145 MB** | **524.1 KB** | ✅ PASS |

*Verification*: The complete desktop pipeline executes in **3.18 ms** per 100 ms epoch (3.18% CPU duty cycle), theoretically supporting up to **300 Hz** on desktop x86_64.

---

## 7. Mathematical & Algorithmic Audit

### 7.1 Coordinate Systems & Frame Transformations
- **Sensor/Body Frame**: Defined as $X_b$ (forward), $Y_b$ (lateral right), $Z_b$ (vertical down or up depending on convention).
- **World Frame**: Local East-North-Up (ENU) tangent plane centered at pre-blackout GNSS origin using high-precision flat-Earth geodesic equations:
  $$E = R_e \Delta\lambda \cos\phi_0, \quad N = R_e \Delta\phi$$
- **Finding**: Coordinate transformations in `latlon_to_enu` and `enu_to_latlon` are mathematically exact. However, in `ExtendedKalmanFilter.predict`, yaw was treated as heading from East (counter-clockwise), whereas standard automotive Course Over Ground (COG) is clockwise from North. This created an axis mismatch that was corrected by standardizing on counter-clockwise angle from East ($\psi = 0 \to \text{East}, \psi = \pi/2 \to \text{North}$).

### 7.2 EKF Mechanization vs. Kinematic NHC
- **Original EKF Formulation**:
  $$a_e = a_{fwd} \cos\psi - v_{fwd} \omega \sin\psi, \quad a_n = a_{fwd} \sin\psi + v_{fwd} \omega \cos\psi$$
  $$\dot{v}_e = a_e, \quad \dot{v}_n = a_n$$
- **Vulnerability**: The centripetal term $-v_{fwd} \omega \sin\psi$ couples yaw rate errors directly into linear velocity integration. Any small gyro bias residual causes $v_e$ and $v_n$ to diverge away from the true vehicle heading vector.
- **Remediation**: Implemented planar kinematic vehicle velocity constraint:
  $$v_e = v \cos\psi, \quad v_n = v \sin\psi$$
  $$\dot{p}_e = v \cos\psi, \quad \dot{p}_n = v \sin\psi$$
  This matches the physical reality of a passenger car on asphalt where lateral tire slip is virtually zero ($v_{lat} \approx 0$).

---

## 8. Mobile & Edge Deployment Audit

### 8.1 Current Deployment Level
- **Level A (Offline Research Prototype)**: SATISFIED (IO-VNBD scripts, PyTorch models, Jupyter exploration).
- **Level B (Real-Time Desktop Prototype)**: SATISFIED (Synchronous 10 Hz streaming simulation, ONNX Runtime evaluation, independent OSM matching).
- **Level C (Mobile Inference Ready)**: SATISFIED (Models exported to ONNX Opset 14, <320 KB, FP32/INT8 ready, numerical parity verified with ONNX Runtime Mobile).
- **Level D (Working Mobile Application)**: **NOT SATISFIED** (No Android APK or iOS project exists in repository).
- **Level E (Fully Integrated Mobile IDR Demo)**: **NOT SATISFIED** (No real-world in-vehicle smartphone field test demonstrated).

### 8.2 External IMU Support Audit
- The software currently expects pre-packaged NumPy arrays structured according to IO-VNBD column indices.
- **Missing Interface**: No `AbstractIMUInterface` exists to ingest generic byte streams over USB/Serial (UART), Bluetooth Low Energy (BLE), or UDP sockets.
- **Required Refactoring**: Implement a decoupled sensor publisher-subscriber pattern (e.g. lightweight ZeroMQ or standard callback interface).

---

## 9. Critical Technical Problems & Vulnerabilities

### Problem 1: No Live Mobile Application (Android/iOS)
- **Severity**: HIGH (Directly required by Problem Statement).
- **Evidence**: Repository contains zero `.kt`, `.java`, `.swift`, or `AndroidManifest.xml` files.
- **Root Cause**: Development focused entirely on Python algorithmic simulation.
- **Impact**: Evaluators will penalize the submission if a working on-device UI is claimed without code.
- **Fix**: Build an Android prototype using Kotlin + Jetpack Compose + ONNX Runtime Android (`com.microsoft.onnxruntime:onnxruntime-android:1.17.0`) wrapping MapLibre Native for offline vector maps.

### Problem 2: Speed Underestimation by Vibration Network
- **Severity**: HIGH (Core cause of along-track drift).
- **Evidence**: `results/velocity_predictions.csv` shows test MAE of 11.51 m/s (41.45 km/h) and systematic negative bias of -10.93 m/s.
- **Root Cause**: Steady highway driving at 85 km/h produces near-zero specific force. Vibration amplitude is uninformative of absolute speed.
- **Impact**: Filter lags behind ground truth by hundreds of meters during long highway outages.
- **Fix**: Replace scalar speed regression with 2D relative displacement odometry (`InertialOdomNet`) conditioned on pre-blackout GNSS speed anchoring.

### Problem 3: Disconnected Re-acquisition Smoothing
- **Severity**: MEDIUM.
- **Evidence**: `ReacquisitionSmoother` is implemented in `src/idr/eval/transition.py` but never imported or called in `src/idr/filters/fusion.py`.
- **Root Cause**: Module was written as a standalone utility but omitted from the active pipeline.
- **Impact**: When GNSS returns after a 60s outage, position jumps abruptly by 100–160 meters in a single 0.1s step, causing UI cursor teleports and velocity spikes.
- **Fix**: Call `ReacquisitionSmoother.trigger_reacquisition()` on the first valid GNSS fix and blend output over 1.5 seconds.

### Problem 4: Hardcoded Sensor Pipeline
- **Severity**: MEDIUM.
- **Evidence**: `src/idr/io/schema.py` explicitly looks for string `"ACCELEROMETER X"` from IO-VNBD CSVs.
- **Root Cause**: Parser was designed exclusively for the benchmark dataset.
- **Impact**: Violates Requirement 13 ("support external IMU data through an edge-deployable software engine").
- **Fix**: Implement a generic `IMUFrame(timestamp, ax, ay, az, gx, gy, gz)` data class with configurable units and axis remapping.

---

## 10. Prioritized Roadmap & Action Items

### Must Fix Before Proposal Submission
1. **Accurate Proposal Wording**: Clearly disclose that the prototype is a verified desktop/ONNX mobile-ready engine, with mobile application development in Phase 2.
2. **Include Exact Empirical Benchmark Tables**: Report the verified 25.0% pass rate, 22.66% median drift, and 160.32 m (15.83%) 1 km performance.
3. **Integrate Reacquisition Smoother**: Connect `transition.py` into `fusion.py` to document seamless, jump-free recovery.

### Should Fix Before Grand Finale
1. **Android Reference App**: Implement a minimal Android application reading live IMU sensors and running `velocity_net.onnx` via ONNX Runtime Mobile.
2. **Attitude Alignment Integration**: Wire `PhoneToVehicleAligner` into pipeline initialization to handle arbitrary phone orientations.
3. **External IMU UDP Daemon**: Implement a lightweight socket listener that ingests generic IMU packets at 100–200 Hz.

---

## 11. Proposal Claims Guidance

### What We Can SAFELY Claim in the SIH Proposal
- ✅ Evaluated on authentic, non-synthetic IEEE IO-VNBD dataset across 25.08 hours of driving.
- ✅ Zero ground-truth leakage verified by strict `EstimatorInput` encapsulation.
- ✅ 100% disjoint train/val/test split with 0 cross-split window overlap.
- ✅ Independent OpenStreetMap road networks downloaded via Overpass API and matched via true HMM Viterbi dynamic programming.
- ✅ Models exported to ONNX with <320 KB memory footprint (<2 MB constraint satisfied) and verified numerical parity ($<10^{-5}$ error).
- ✅ Achieved 25.0% pass rate (<10% drift) across 100 authentic scenarios, reducing 1 km target blackout drift to 160.32 m (15.83%), a 72.5% reduction.
- ✅ Desktop execution latency of 3.18 ms per epoch (300 Hz theoretical throughput).

### What We Must NOT Claim Yet
- ❌ Do NOT claim a fully functional mobile application is running on smartphones today (it is currently a desktop/ONNX mobile-ready engine).
- ❌ Do NOT claim lane-level (<1 meter) accuracy during long blackouts (current CEP50 is ~60–80 m on 60s outages).
- ❌ Do NOT claim external 200 Hz FOG IMU hardware has been physically interfaced (it is an edge-compatible software design).
- ❌ Do NOT claim arbitrary phone orientations are fully solved in live evaluation (the aligner module is written but not yet integrated into the blackout benchmark).
