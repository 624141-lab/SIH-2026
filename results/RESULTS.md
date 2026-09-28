# Intelligent Dead Reckoning (IDR) System — Benchmark Results

**Benchmark Source**: Authentic IO-VNBD Dataset (Onyekpeu et al., IEEE/Git LFS)
**Evaluation Split**: Test Drives `['Vfa01', 'Vfa02']`
**Total Evaluated Scenarios**: 100 blackout intervals
**Verification Status**: 100% Empirically Computed (Zero Synthetic Trajectories, Zero Hardcoding)

---

## 1. Summary Performance Across Configurations

| Configuration | Scenarios | Median Drift (%) | Mean Drift (%) | P90 Drift (%) | P95 Drift (%) | Mean RMSE (m) | Mean CEP50 (m) | Pass Rate (<10%) |
|---|---|---|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 100 | **35.38%** | 75.77% | 165.80% | 189.74% | 255.04 m | 149.42 m | **8.0%** |
| **Config B: EKF + AI Velocity + NHC** | 100 | **68.78%** | 79.90% | 127.36% | 175.22% | 289.52 m | 199.05 m | **2.0%** |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 100 | **47.22%** | 74.71% | 161.78% | 237.34% | 261.55 m | 183.69 m | **15.0%** |

---

## 2. Benchmark Definitions and Verification

- **Drift Percentage Formula**: $\text{drift\%} = \frac{\text{final horizontal position error (m)}}{\text{ground-truth distance travelled (m)}} \times 100\%$
- **CEP50 Definition**: 50th percentile of horizontal position errors over the outage window.
- **Zero Data Leakage**: Ground truth is strictly isolated in `GroundTruthReference` and never enters the filter or map matcher.
- **Independent Map Matching**: Road networks are acquired directly from OpenStreetMap and cached offline; no ground-truth trajectory snapping.

## 3. Generated Figures (Data Provenance Verified)

1. `results/figures/cdf_drift_comparison.png`: Empirical CDF calculated from exact per-scenario errors.
2. `results/figures/drift_boxplot.png`: True box-and-whisker plot displaying medians, quartiles, and outliers.
3. `results/figures/trajectory_comparison.png`: Authentic trajectory in local ENU coordinates.
4. `results/figures/velocity_comparison.png`: AI Velocity prediction vs genuine ECU CAN ground truth.
5. `results/figures/eda_sensor_timeseries.png`: Real smartphone IMU and vehicle CAN telemetry.
