# TABLE A — CURRENT PIPELINE ARCHITECTURE

### Pipeline Identification
- **Configuration Name:** System 6: Full Hardened IDR Pipeline
- **Evaluation Entry Point:** [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py) & [`scripts/generate_proposal_figures.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/generate_proposal_figures.py)
- **Model Checkpoint:** [`models/inertial_odom.pt`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/models/inertial_odom.pt) (SHA-256: `1c4c7a769e1b1252...`)
- **Sensors Ingested:** 6-axis Smartphone IMU (Accelerometer, Gyroscope @ 10 Hz)
- **No Vehicle CAN/OBD Required:** Operates strictly on smartphone sensors without physical vehicle wire tapping.

---

### End-to-End Processing Stages

| Stage | Current Algorithm | Source File | Input | Output | Real-Time / Causal? |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **1. Sensor Ingestion** | Synced Stream Loader | `src/idr/io/loader.py` | Smartphone CSV (`S-*.csv`) | 10 Hz calibrated IMU array | **YES (Streaming)** |
| **2. Attitude Alignment** | Static Gravity & Forward Acceleration Aligner | `src/idr/calib/alignment.py` | Raw 6-axis IMU | Rotation matrix $R_{phone \to vehicle}$ | **YES (Pre-drive / Cradle)** |
| **3. Vibration Filter** | Low-Pass Moving Average & Covariance Inflation | `src/idr/filters/vibration_filter.py` | Vehicle-frame IMU | Filtered IMU, Dynamic Covariance Scale $s_{cov}$ | **YES (Causal, 1.0s window)** |
| **4. AI Neural Odometry** | `InertialOdomNet` (1D-CNN + GRU) | `src/idr/models/inertial_odom.py` | 50-step IMU window (5.0s @ 10 Hz) | 2D displacement $(\Delta x, \Delta y)$, Log-variance $\log \sigma^2$ | **YES (0 look-ahead, 0.93ms)** |
| **5. Kinematic Filter** | 7-State Kinematic Extended Kalman Filter | `src/idr/filters/ekf.py` | Forward Accel, Yaw Rate, AI Velocity | State $\mathbf{x} = [p_e, p_n, p_u, v_e, v_n, v_u, \psi]^T$ | **YES (Causal Kalman step)** |
| **6. Adaptive NHC** | Centripetal-Aware Non-Holonomic Constraints | `src/idr/filters/nhc.py` | Filter state, Yaw Rate $\omega_z$, Forward Speed $v$ | Zero lateral & vertical velocity pseudo-measurements | **YES (Instantaneous update)** |
| **7. Road Map Fusion** | Online Causal HMM with Out-of-Corridor Gate | `src/idr/mapmatch/causal_matcher.py` | EKF trajectory, Offline OSM Graph | Road-projected coordinates, Cross-track confinement | **YES (0 look-ahead / causal)** |
| **8. GNSS Reacquisition** | $C^1$ Continuous Cosine-Blend Smoother | `src/idr/eval/transition.py` | Restored GNSS fix, Dead-reckoning state | Smoothly blended trajectory (3.5s duration) | **YES (Post-blackout causal)** |

---

### Key Hyperparameters
- **AI Window Size:** 50 samples ($5.0\text{ s}$ @ $10\text{ Hz}$).
- **EKF Process Noise:** Acceleration variance $Q_a = 0.5\text{ m}^2/\text{s}^4$, Yaw-rate variance $Q_\omega = 0.05\text{ rad}^2/\text{s}^2$.
- **Adaptive NHC Scaling:** $\sigma_{lat}^2 = \sigma_{base}^2 + k_{turn} (v_{fwd} \cdot |\omega_z|)^2$, with $\sigma_{base} = 0.05\text{ m/s}$ and $k_{turn} = 0.5$.
- **Map Out-of-Corridor Gate:** $30.0\text{ m}$ maximum search radius with emission standard deviation $\sigma_z = 8.0\text{ m}$.
