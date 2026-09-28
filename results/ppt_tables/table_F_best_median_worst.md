# TABLE F — REPRESENTATIVE SCENARIOS (BEST, MEDIAN, WORST)

### Overview
This table details the representative scenarios extracted directly from the authentic 50-scenario benchmark. All distances, errors, and driving conditions are verified from current raw sensor and ground-truth execution.

---

### Representative Benchmark Scenarios

| Case Type | Scenario ID | Drive ID | Distance | Outage | Final Error | Positional Drift | Main Driving Condition | Dominant Error Source |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Best (Overall)** | `Vfa02_t110s_d60s` | `Vfa02` | 974.6 m | 60.0 s | **7.80 m** | **0.80%** | Cruising @ 58.5 km/h, single gentle curve | Minimal error; near-perfect velocity tracking |
| **Best (Mapped)** | `Vfa01_t30s_d60s` | `Vfa01` | 1,162.5 m | 60.0 s | **30.02 m** | **2.58%** | Fast highway cruising @ 69.7 km/h | Residual along-track scale error; map keeps cross-track < 3.5 m |
| **Proposal Demo** | `Vfa01_t70s_d60s` | `Vfa01` | 1,139.7 m | 60.0 s | **90.84 m** | **7.97%** | Steady highway cruising @ 68.4 km/h | Minor longitudinal scale lag; strict SIH Pass (<10%) |
| **Median Case** | `Vfa01_t110s_d15s` | `Vfa01` | 284.3 m | 15.0 s | **89.99 m** | **31.65%** | High-speed straight stretch @ 68.2 km/h | Short distance amplifies initial filter settling error |
| **Worst Case** | `Vfa02_t10s_d30s` | `Vfa02` | 175.5 m | 30.0 s | **546.11 m** | **311.19%** | Stop-and-go (8.8s stopped), sharp 55° turn | Gyro drift during stop + unmapped rural road topology |

---

### Key Observations
1. **Long Outages (>1 km) Excel:** On high-speed segments (60 s, 1+ km), the pipeline achieves its strongest results (0.80% to 7.97% drift) because forward velocity is steady and road priors constrain lateral divergence.
2. **Short Segments Show Higher Drift %:** In short 15 s scenarios (284 m), an 89 m error yields 31.65% drift because the distance denominator is small while the EKF initial transient takes 3–5 seconds to settle.
3. **Severe Stops Require ZUPT:** In `Vfa02_t10s_d30s`, the vehicle stopped for 8.8 seconds. Without Zero Velocity Updates (ZUPT) and without map priors, gyro drift integrated into 546 m of open-loop error.
