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

Evaluated across **100 deterministic blackout intervals** (durations: 15s, 30s, 60s) on held-out test drives `Vfa01` and `Vfa02`:

| Configuration | Scenarios | Median Drift (%) | Mean Drift (%) | P90 Drift (%) | P95 Drift (%) | Mean RMSE (m) | Mean CEP50 (m) | Pass Rate (<10%) |
|---|---|---|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 100 | **29.63%** | 38.55% | 66.53% | 125.26% | 123.98 m | 85.69 m | **14.0%** |
| **Config B: EKF + AI Velocity + NHC** | 100 | **72.05%** | 89.36% | 159.07% | 210.15% | 307.82 m | 206.37 m | **2.0%** |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 100 | **56.80%** | 81.91% | 171.33% | 241.52% | 290.33 m | 204.11 m | **6.0%** |

### Benchmark Definitions:
- **Drift Percentage**: $\text{drift\%} = \frac{\text{final horizontal position error (m)}}{\text{ground-truth distance travelled (m)}} \times 100\%$
- **CEP50**: Circular Error Probable (50th percentile horizontal position error over the outage window).
- **Physical Interpretation**: Unassisted consumer smartphone IMUs suffer from severe low-frequency accelerometer and gyro biases ($>0.2 \text{ m/s}^2$). Over extended 30s–60s outages ($>500\text{ m}$ traveled), double integration accumulates quadratic position error. Applying independent OSM HMM map matching constrains lateral drift back to road centerlines, reducing median drift from 72.05% to 56.80% and mean drift from 89.36% to 81.91%.

### 📍 1 km Outage Performance (`~1 km` Range Metrics)

#### 1. Exact 1 km Scenario from Multi-Scenario Benchmark (`Vfa01_t725_d60s`)
| Configuration | `total_distance_m` | `final_drift_m` | `drift_percent` | RMSE (m) | CEP50 (m) |
|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 1012.54 m | 635.58 m | 62.77% | 324.25 m | 179.70 m |
| **Config B: EKF + AI Velocity + NHC** | 1012.54 m | 1118.14 m | 110.43% | 520.64 m | 252.15 m |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 1012.54 m | 582.76 m | 57.55% | 245.31 m | 68.57 m |

#### 2. Average Across All 1 km Range Scenarios (800 m – 1200 m, 12 Scenarios)
| Configuration | Mean `total_distance_m` | Mean `final_drift_m` | Mean `drift_percent` |
|---|---|---|---|
| **Config A: Raw IMU Baseline** | 935.28 m | 317.88 m | 32.48% |
| **Config B: EKF + AI Velocity + NHC** | 935.28 m | 771.15 m | 80.09% |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 935.28 m | 675.04 m | 69.89% |

#### 3. Single-Trip Continuous 1.15 km Outage (`reports/eval_results.json`)
| Configuration | `total_distance_m` | `final_drift_m` | `drift_percent` | RMSE (m) | CEP50 (m) |
|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 1153.11 m | 27.97 m | 2.43% | 10.69 m | 5.97 m |
| **Config B: EKF + AI Velocity + NHC** | 1153.11 m | 46.32 m | 4.02% | 21.42 m | 13.15 m |
| **Config C: EKF + AI Velocity + NHC + OSM Snap** | 1153.11 m | 29.19 m | 2.53% | 17.08 m | 12.78 m |

---

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
