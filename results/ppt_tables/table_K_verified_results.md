# TABLE K — MASTER VERIFIED BENCHMARK PERFORMANCE

### Overview
This table provides the definitive, reproducible performance metrics of the final Intelligent Dead Reckoning pipeline (System 6) across the full 50-scenario IO-VNBD evaluation suite. Every metric is computed directly from raw sensor data without post-processing or manual edits.

---

### Master Performance Summary (Current Verified System)

| Performance Metric | Current Verified Value | Context & Evaluation Basis |
| :--- | :---: | :--- |
| **Total Scenarios Evaluated** | **50 scenarios** | 26 on Drive `Vfa01` (Validation) + 24 on Drive `Vfa02` (Untouched Test) |
| **Validation Scenarios (`Vfa01`)** | **26 scenarios** | Mapped highway and urban segments with OpenStreetMap priors |
| **Untouched Test Scenarios (`Vfa02`)**| **24 scenarios** | Unmapped rural segments (evaluating baseline open-loop boundaries) |
| **Median Positional Drift (%)** | **31.06%** | Overall 50-scenario distribution median (50% of runs $\le 31.06\%$) |
| **Mean Positional Drift (%)** | **58.13%** | Arithmetic mean across all 50 scenarios |
| **P25 Positional Drift (%)** | **11.50%** | Lower quartile (top 25% of all scenarios achieve $\le 11.50\%$ drift) |
| **P75 Positional Drift (%)** | **65.20%** | Upper quartile |
| **P90 Positional Drift (%)** | **163.85%** | 90th percentile (heavily skewed by unmapped `Vfa02` rural runs) |
| **Maximum Positional Drift (%)**| **311.19%** | Worst scenario (`Vfa02_t10s_d30s`: unmapped stop-and-go cornering) |
| **SIH 26168 Pass Rate (<10%)** | **22.0% (11/50)** | 11 scenarios strictly beat the $<10\%$ drift threshold |
| **Pass Rate on Mapped Highway (`Vfa01`)**| **34.6% (9/26)** | 9 of 26 mapped scenarios satisfy $<10\%$ drift (mean error: 107.58 m) |
| **Scenarios Within 15% Drift** | **30.0% (15/50)** | 15 scenarios achieve $\le 15.0\%$ drift |
| **Scenarios Within 20% Drift** | **36.0% (18/50)** | 18 scenarios achieve $\le 20.0\%$ drift |
| **Best-Case Positional Drift** | **0.80% (7.80 m)** | Scenario `Vfa02_t110s_d60s` ($974.6\text{ m}$ traveled, $60.0\text{ s}$ outage) |
| **Best-Case Drift (Mapped Corridor)**| **2.58% (30.02 m)**| Scenario `Vfa01_t30s_d60s` ($1,162.5\text{ m}$ traveled, $60.0\text{ s}$ outage) |
| **Proposal Demonstration Scenario**| **7.97% (90.84 m)**| Scenario `Vfa01_t70s_d60s` ($1,139.7\text{ m}$ traveled, $60.0\text{ s}$ outage) |
| **Median-Case Positional Drift**| **31.65% (89.99 m)**| Scenario `Vfa01_t110s_d15s` ($284.3\text{ m}$ traveled, $15.0\text{ s}$ outage) |
| **Worst-Case Positional Drift** | **311.19% (546.11 m)**| Scenario `Vfa02_t10s_d30s` ($175.5\text{ m}$ traveled, $30.0\text{ s}$ outage) |
| **Maximum Outage Distance Evaluated**| **1,162.5 m** | Scenario `Vfa01_t30s_d60s` (exceeds SIH 1 km requirement) |
| **Maximum Outage Duration Evaluated**| **60.0 seconds** | 10 Hz continuous inertial propagation without satellite updates |
