# TABLE G — DRIVING CONDITIONS ANALYSIS (EMPIRICALLY VERIFIED)

### Overview
This table characterizes the exact physical dynamics and kinematics of representative scenarios calculated directly from raw smartphone IMU sensors and vehicle ECU ground truth.

---

### Kinematic & Dynamics Characterization

| Scenario ID | Mean Speed | Max Speed | Speed Range | Max Yaw Rate | Cumulative Heading | Net Turn | Max Lat Accel | Stop Duration | Maneuver Classification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Vfa01_t70s_d60s`** | 68.4 km/h (19.0 m/s) | 76.6 km/h | 56.8–76.6 km/h | 14.4 deg/s | 202.1 deg | 5.5 deg | 6.13 m/s² | 0.0 s (0 stops) | Continuous Highway Cruising |
| **`Vfa01_t30s_d60s`** | 69.7 km/h (19.4 m/s) | 78.0 km/h | 56.8–78.0 km/h | 19.9 deg/s | 209.4 deg | 18.7 deg | 4.58 m/s² | 0.0 s (0 stops) | High-Speed Highway with Curve |
| **`Vfa02_t110s_d60s`** | 58.5 km/h (16.2 m/s) | 73.9 km/h | 28.8–73.9 km/h | 27.1 deg/s | 216.0 deg | 89.3 deg | 6.96 m/s² | 0.0 s (0 stops) | Cruising with 90° Turn |
| **`Vfa01_t110s_d15s`** | 68.2 km/h (19.0 m/s) | 76.2 km/h | 64.8–76.2 km/h | 14.2 deg/s | 56.6 deg | 4.2 deg | 4.32 m/s² | 0.0 s (0 stops) | Short Fast Straight |
| **`Vfa02_t10s_d30s`** | 21.1 km/h (5.8 m/s) | 47.8 km/h | 0.0–47.8 km/h | 19.3 deg/s | 104.8 deg | 55.0 deg | 5.65 m/s² | 8.8 s (88 samples) | Stop-and-Go Urban Cornering |

---

### Empirical Insights
1. **Dynamic Turning Load:** Peak lateral accelerations range between $4.32\text{ m/s}^2$ and $6.96\text{ m/s}^2$, validating the necessity of centripetal-aware adaptive NHC ($\sigma_{\text{lat}}^2$ scaling) to prevent vehicle slip distortion.
2. **Stationary Phase Vulnerability:** When vehicles stop in traffic (`Vfa02_t10s_d30s`), IMU integration without stationary zero-velocity updates (ZUPT) accumulates heading error rapidly.
3. **Speed Stability:** Scenarios with steady cruising speeds (55–75 km/h) demonstrate the highest neural odometry accuracy (speed tracking MAE $\approx 1.14\text{ m/s}$).
