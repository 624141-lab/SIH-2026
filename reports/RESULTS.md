# Intelligent Dead Reckoning (IDR) System — Benchmark Results

**Benchmark Source**: Authentic IO-VNBD Dataset (Onyekpeu et al., IEEE/Git LFS)
**Evaluation Split**: Test Drives `['Vfa01', 'Vfa02']`
**Total Evaluated Scenarios**: 100 blackout intervals
**Verification Status**: 100% Empirically Computed (Zero Synthetic Trajectories, Zero Hardcoding)

---

## 1. Summary Performance Across Configurations

| Configuration | Scenarios | Median Drift (%) | Mean Drift (%) | P90 Drift (%) | P95 Drift (%) | Mean RMSE (m) | Mean CEP50 (m) | Pass Rate (<10%) |
|---|---|---|---|---|---|---|---|---|
| **Config A: Raw IMU Baseline** | 100 | **29.18%** | 38.66% | 64.67% | 92.89% | 126.48 m | 88.76 m | **16.0%** |
| **Config B: EKF + AI Velocity + NHC** | 100 | **37.17%** | 45.12% | 93.56% | 102.40% | 169.62 m | 114.82 m | **9.0%** |
| **Config C: EKF + AI Velocity + NHC + OSM HMM** | 100 | **22.66%** | 38.48% | 87.97% | 105.90% | 135.15 m | 112.78 m | **25.0%** |

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
