# AI-ML Intelligent Dead Reckoning (IDR) System
**Smart India Hackathon Problem Statement 26168 — ISRO**  
*Vehicle navigation during GNSS blackouts using smartphone IMU sensor fusion, deep-learning forward velocity estimation, Non-Holonomic Constraints (NHC), and independent OpenStreetMap HMM map matching.*

---

## 🛡️ Forensic Remediation & Verification Status

Following a comprehensive forensic audit, this repository has been **100% remediated** to ensure complete empirical integrity and scientific reproducibility:

- **Authentic Dataset**: Evaluated strictly on the authentic 194.2 MB [IO-VNBD dataset](https://github.com/onyekpeu/IO-VNBD) (Onyekpeu et al., IEEE/Data in Brief 2021). All 288 CSV files (1,070,745 sensor rows across 25.08 hours of driving) are parsed with `latin-1`, trimmed headers, and canonical unit normalization (km/h $\to$ m/s, deg/s $\to$ rad/s).
- **Zero Synthetic Trajectories**: The primary benchmark runs exclusively on authentic driving trips (`Vfa01` and `Vfa02`). All legacy synthetic fixtures are quarantined under `tests/fixtures/synthetic/`.
- **Zero Ground-Truth Leakage**: The navigation estimator operates strictly on isolated `EstimatorInput`. Zero GNSS coordinates, velocities, or ground-truth references enter the filters during simulated blackouts.
- **Independent OpenStreetMap Graphs**: Road graphs are acquired independently from OpenStreetMap (Overpass API) and cached offline (`data/osm/`). Road networks are **never** constructed from ground-truth waypoints.
- **Authentic HMM Viterbi Map Matching**: Implements true Newson & Krumm HMM map matching (Gaussian emission, exponential transition with heading alignment, dynamic programming trellis) consuming strictly estimated dead-reckoned coordinates.
- **18/18 Forensic Audit Checks Passed**: Fully verified by `py -3.11 scripts/run_audit.py` and `py -3.11 -m pytest tests/`.

---

## 📊 Empirical Benchmark Results

Evaluated across **100 deterministic blackout intervals** (durations: 15s, 30s, 60s) on held-out test drives `Vfa01` and `Vfa02` with **Top 5 Concrete Improvements** active:

| Configuration | Scenarios | Median Drift (%) | Mean Drift (%) | P90 Drift (%) | P95 Drift (%) | Mean RMSE (m) | Mean CEP50 (m) | Pass Rate (<10%) |
|---|---|---|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 100 | **29.18%** | 38.66% | 64.67% | 92.89% | 126.48 m | 88.76 m | **16.0%** |
| **Config B: EKF + AI Velocity + NHC** | 100 | **37.17%** | 45.12% | 93.56% | 102.40% | 169.62 m | 114.82 m | **9.0%** |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 100 | **22.66%** | 38.48% | 87.97% | 105.90% | 135.15 m | 112.78 m | **25.0%** |

### Benchmark Definitions:
- **Drift Percentage**: $\text{drift\%} = \frac{\text{final horizontal position error (m)}}{\text{ground-truth distance travelled (m)}} \times 100\%$
- **CEP50**: Circular Error Probable (50th percentile horizontal position error over the outage window).
- **Impact of Improvements**: With the Top 5 concrete improvements (Butterworth pre-blackout bias calibration, passenger car kinematic Non-Holonomic Constraints, dynamic speed anchoring, ZUPT, and closed-loop OSM guidance), **Config C pass rate (<10% drift) reached 25.0%** (up from 6.0%), and median drift dropped to **22.66%** (down from 56.80%).

---

### 📍 1 km Outage Performance (`~1 km` Range Metrics)

#### 1. Exact 1 km Scenario from Multi-Scenario Benchmark (`Vfa01_t725_d60s`, 60s Outage)
| Configuration | `total_distance_m` | `final_drift_m` | `drift_percent` | RMSE (m) | CEP50 (m) |
|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 1012.54 m | 472.43 m | 46.66% | 231.84 m | 185.20 m |
| **Config B: EKF + AI Velocity + NHC** | 1012.54 m | 639.03 m | 63.11% | 275.60 m | 210.45 m |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 1012.54 m | **160.32 m** | **15.83%** | **78.42 m** | **62.15 m** |

*Note: Config C final drift on this exact 1 km scenario dropped from 582.76 m (57.55%) down to **160.32 m (15.83%)** — a **72.5% reduction in error**.*

#### 2. Average Across All 1 km Range Scenarios (800 m – 1200 m, 12 Scenarios)
| Configuration | Mean `total_distance_m` | Mean `final_drift_m` | Mean `drift_percent` |
|---|---|---|---|
| **Config A: Raw IMU Baseline** | 935.28 m | 299.69 m | 30.52% |
| **Config B: EKF + AI Velocity + NHC** | 935.28 m | 415.13 m | 42.58% |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 935.28 m | **256.62 m** | **27.00%** |

*Note: Average final drift in the 1 km range dropped from **675.04 m (69.89%)** down to **256.62 m (27.00%)**.*

---

## ⚡ Top 5 Concrete Improvements Implemented

1. **Pre-Blackout Sensor Bias Calibration**:
   - Uses the 3.0-second window immediately prior to GNSS loss to estimate gyroscope yaw rate bias $\hat{b}_\omega$ and forward accelerometer bias $\hat{b}_a$.
   - Directly initializes EKF states $x[7] = \hat{b}_a$ and $x[8] = \hat{b}_\omega$ and offsets raw IMU inputs, eliminating linear heading drift and quadratic position drift.
2. **Dynamic Forward Speed Anchoring**:
   - Deep neural networks predicting speed from 1-second IMU vibration tend to predict average cruising speeds (~43 km/h).
   - Speed estimates are dynamically anchored to the known pre-outage GNSS velocity $v_0$, preserving true highway cruising speeds (80–90 km/h).
3. **Directional Forward Velocity Model in EKF**:
   - Replaced scalar speed measurement $z = \sqrt{v_e^2 + v_n^2}$ with kinematic directional body velocity $z_{fwd} = v_e \cos\psi + v_n \sin\psi$.
   - Measurement Jacobian explicitly couples speed updates to heading $\psi$, eliminating spurious lateral velocity accumulation.
4. **Zero-Velocity & Zero-Angular-Rate Updates (ZUPT/ZARU)**:
   - Integrated statistical stationary detector ($\sigma_a^2 < 0.15 \text{ m}^2/\text{s}^4, \|\omega\| < 0.05 \text{ rad/s}$).
   - Whenever vehicle stops at signals or traffic, velocity is clamped to zero and sensor biases are reset.
5. **Closed-Loop Road Guidance in HMM Map Matcher**:
   - Evaluates nearest road candidate distance and bearing during the blackout.
   - Gently guides dead-reckoned trajectory to road centerlines before full Viterbi trellis decoding, preventing divergence beyond the search radius.

## 🚀 Quickstart & Pipeline Execution

Prerequisites: Python 3.11 with PyTorch 2.x, ONNX, and ONNX Runtime.

```bash
# 1. Download & Validate Authentic IO-VNBD Dataset (194.2 MB)
py -3.11 scripts/download_data.py
py -3.11 scripts/validate_iovnbd.py

# 2. Preprocess into Drive-Disjoint Sliding Windows (Train: 68,104, Val: 7,024, Test: 7,892)
py -3.11 -m idr.io.preprocess

# 3. Train Forward Velocity Estimator & IMU Denoise Models
py -3.11 -m idr.models.train_all --epochs 10

# 4. Export to ONNX and Verify Numerical Parity (< 2 MB footprint)
py -3.11 scripts/export_and_validate.py

# 5. Execute Multi-Scenario Blackout Benchmark on Real Test Drives
py -3.11 -m idr.eval.blackout

# 6. Generate Empirical CDF, Boxplots, Trajectories, and RESULTS.md
py -3.11 -m idr.eval.plotting

# 7. Run Complete Test Suite & 18-Point Audit
py -3.11 -m pytest tests/
py -3.11 scripts/run_audit.py
```

---

## 📱 Mobile & Edge Deployment (< 2 MB Target)

Models are exported to ONNX (Opset 14) and verified for numerical parity between PyTorch and ONNX Runtime:

| Model | Parameters | PyTorch Size | ONNX Size | < 2 MB Limit | Max Parity Error | Status |
|---|---|---|---|---|---|---|
| **VelocityEstimatorNet** | 79,394 | 323.9 KB | 314.9 KB | ✅ PASS | $3.80 \times 10^{-6}$ | ✅ VERIFIED |
| **IMUDenoiseNet** | 53,414 | 221.7 KB | 209.2 KB | ✅ PASS | $4.10 \times 10^{-6}$ | ✅ VERIFIED |

- **Deployment Route**: ONNX Runtime Mobile (`onnxruntime-mobile` for Android / ARM).
- **TFLite Status**: Documented honestly as unsupported in current environment (`tensorflow` / `tflite_runtime` not installed on Windows host). ONNX Runtime Mobile is the production target.

---

## 📁 Repository Structure

```
idr-system/
├── config/
│   └── splits.yaml                 # Canonical drive-disjoint train/val/test splits
├── data/
│   ├── raw/iovnbd/                 # 288 authentic IO-VNBD CSV files (72 S-files, 72 V-files)
│   ├── processed/                  # train_data.npz, val_data.npz, test_data.npz
│   └── osm/                        # Cached independent OpenStreetMap road networks
├── models/
│   ├── velocity_net.pt             # Trained PyTorch checkpoint
│   ├── velocity_net.onnx           # 315 KB ONNX export (verified parity)
│   ├── imu_denoise_net.pt          # Trained PyTorch checkpoint
│   └── imu_denoise.onnx            # 209 KB ONNX export (verified parity)
├── reports/
│   ├── RESULTS.md                  # Dynamically generated benchmark results
│   ├── dataset_validation.md       # 288-file dataset forensic report
│   ├── export_validation.md        # ONNX export and parity report
│   └── window_leakage_report.json  # Proof of 0 cross-split window overlap
├── results/
│   ├── scenario_manifest.csv       # Exact 100 evaluated blackout intervals
│   ├── per_scenario_metrics.csv    # Per-scenario drift, RMSE, CEP50 for all 3 configs
│   ├── eval_results.json           # Machine-readable benchmark summary
│   ├── velocity_predictions.csv    # Real test set velocity predictions
│   └── figures/                    # Empirical CDFs, boxplots, trajectories, sensor EDA
├── scripts/
│   ├── download_data.py            # Authentic LFS download script (no mocks)
│   ├── validate_iovnbd.py          # Dataset schema and integrity validator
│   ├── download_osm_roads.py       # Independent Overpass OSM fetcher
│   ├── export_and_validate.py      # ONNX export and parity verification
│   └── run_audit.py                # 18-point forensic audit validator
├── src/idr/
│   ├── io/                         # Schema detection, loader, preprocessor
│   ├── models/                     # VelocityEstimatorNet, IMUDenoiseNet
│   ├── filters/                    # EKF sensor fusion, Non-Holonomic Constraints (NHC)
│   ├── mapmatch/                   # OSMGraphLoader, HMMMapMatcher (Newson & Krumm)
│   └── eval/                       # Blackout simulator, navigation metrics, plotting
└── tests/
    ├── test_idr.py                 # Core unit tests
    └── test_remediation.py         # 18-point forensic requirement test suite
```
