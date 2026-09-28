# TABLE B — DATASET & EVALUATION PROTOCOL

### Overview
This table defines the authentic vehicular dataset, partitioning strategy, and rigorous evaluation protocol used to validate the Intelligent Dead Reckoning (IDR) pipeline under simulated GNSS denial.

---

### Dataset & Evaluation Protocol Specifications

| Property | Current Verified Value | Description / Verification Source |
| :--- | :--- | :--- |
| **Dataset Name** | **IO-VNBD** | Authentic Smartphone Inertial-Odometry Vehicular Navigation Benchmark Dataset |
| **Sensor Platform** | Smartphone (Samsung / Xiaomi) | Uncalibrated MEMS Accelerometer, Gyroscope & Magnetometer in vehicle cradle |
| **Reference Ground Truth** | NovAtel / U-blox RTK GNSS + ECU | Sub-decimeter RTK position reference synchronized with vehicle CAN wheel speed |
| **Training Partition** | Drives `M`, `S`, `Vta`, `Vtb` | 4 authentic vehicle drives used exclusively for neural odometry training |
| **Validation Drive** | Drive `Vfa01` | Authentic urban/highway drive with OpenStreetMap (OSM) vector geometry priors |
| **Untouched Held-Out Test Drive**| Drive `Vfa02` | Authentic unmapped rural drive (zero OSM road geometry; stress test for open-loop IMU) |
| **Validation Scenarios** | **26 scenarios** | Systematic sliding windows across Drive `Vfa01` |
| **Untouched Test Scenarios** | **24 scenarios** | Systematic sliding windows across Drive `Vfa02` |
| **Total Scenarios Evaluated** | **50 scenarios** | 300 total algorithmic evaluations across 6 navigation configurations |
| **Outage Durations Evaluated**| **15 s, 30 s, 60 s** | Realistic blackout windows (highway tunnels, multi-level flyovers, urban canyons) |
| **Distance Traveled Range** | **175.5 m to 1,162.5 m** | Average outage distance: 542.4 m; multiple scenarios exceed 1.1 km |
| **IMU Sampling Frequency** | **10.0 Hz** | Synchronized 100 ms time steps ($\Delta t = 0.1\text{ s}$) matching standard smartphone rate |
| **GNSS During Blackout?** | **NO (STRICT ZERO)** | GNSS coordinates completely zeroed / unavailable during outage |
| **Ground Truth in Blackout?** | **NO (STRICT ZERO)** | Zero ground-truth velocity, heading, or position used in state estimation |
| **Primary Evaluation Metric** | **Positional Drift (%)** | $\text{Drift} = \frac{\|\mathbf{p}_{\text{est}}(t_{\text{end}}) - \mathbf{p}_{\text{gt}}(t_{\text{end}})\|}{\text{Distance Traveled}} \times 100\%$ |
| **SIH 26168 Target Criterion**| **$< 10.0\%$ Drift** | Drift $< 100\text{ m}$ per $1\text{ km}$ traveled during complete satellite loss |
