# 02 — END-TO-END WORKFLOW: SAMPLE-BY-SAMPLE LIFECYCLE

---

## 1. Chronological Processing for a Single Timestep ($t_k$)

Here is the exact step-by-step lifecycle of a single sensor observation as it flows through the executable code 
```text
1. Sensor Arrival (10 Hz):
   phone_imu[k] = [ax, ay, az, gx, gy, gz]
   │
   ▼
2. Coordinate Transformation (Phone -> Vehicle):
   acc_v[k], gyro_v[k] = aligner.transform_imu(acc_p, gyro_p)
   │
   ▼
3. Vibration & Gravity Separation:
   s_acc, s_gyro, state, cov_scale = vfilter.process(acc_v[k], gyro_v[k])
   fwd_acc = s_acc[0], yaw_rate = s_gyro[2]
   │
   ▼
4. AI Odometry Inference (Sliding Window):
   window = phone_imu[k-50 : k]  (shape: 6 x 50)
   dx_body, dy_body, log_var_x, log_var_y = odom_model(window)
   v_ai = dx_body / (50 * dt)
   sig_x = exp(0.5 * log_var_x)
   │
   ▼
5. Kinematic Filter State Prediction:
   ekf.predict(fwd_acc, yaw_rate)
   Propagates: x_pred = f(x_{k-1}, u_k),  P_pred = F * P * F^T + Q
   │
   ▼
6. Adaptive NHC Pseudo-Measurement Update:
   apply_adaptive_nhc_update(ekf, yaw_rate, sigma_lat_base, k_turn)
   y = [-v_lat_pred, -v_vert_pred]
   Constrains lateral slip based on centripetal acceleration
   │
   ▼
7. AI Velocity Measurement Update:
   v_ai_tracking = 0.85 * (v_ai_tracking + fwd_acc * dt) + 0.15 * v_ai
   ekf.update_velocity(v_ai_tracking, R_speed = max(0.1, sig_x * sqrt(cov_scale)))
   Adjusts forward velocity and position in ENU state
   │
   ▼
8. Spatial Road Prior (Causal HMM Map Matching):
   match_output = causal_matcher.step(ekf.x[0], ekf.x[1], motion_heading = ekf.x[6])
   Projects onto OSM centerline if confidence >= 0.20 and dist <= 30.0m
   │
   ▼
9. Output Navigation Coordinate:
   p_nav[k] = [match_output.provisional_x, match_output.provisional_y]
```

---

## 2. Sample-by-Sample Data Flow Table

| Step | Operation Name | Source File & Function | Input Variables | Output Variables | Physical Meaning |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | Stream Ingestion | `src/idr/io/loader.py` | CSV row from `S-*.csv` | `seg_imu_raw[i]` (6,) | Raw uncalibrated IMU readings in phone frame |
| **2** | Attitude Alignment | `src/idr/calib/alignment.py`<br>`transform_imu()` | `seg_imu_raw`, `aligner.R_phone_to_vehicle` | `acc_v[i]` (3,)<br>`gyro_v[i]` (3,) | Acceleration and angular rates rotated into vehicle body frame (X=fwd, Y=lat, Z=up) |
| **3** | Vibration & Shock Filter | `src/idr/filters/vibration_filter.py`<br>`process()` | `acc_v[i]`, `gyro_v[i]` | `s_acc` (3,)<br>`s_gyro` (3,)<br>`cov_scale` (float) | Removes gravity via low-pass, isolates linear acceleration, and flags dynamic vibration disturbances |
| **4** | AI Odometry Window | `src/idr/models/inertial_odom.py`<br>`forward()` | 50 historical IMU samples: `windows[i]` (1, 6, 50) | `dx_b` (float)<br>`sig_x` (float) | Neural network predicts body forward displacement over 5.0 s and aleatoric log-variance |
| **5** | EKF Mechanization Predict | `src/idr/filters/ekf.py`<br>`predict()` | `fwd_acc` ($a_{x,v}$), `yaw_rate` ($\omega_{z,v}$) | `ekf.x` (9,)<br>`ekf.P` (9, 9) | Propagates dead-reckoned position and yaw using forward acceleration and yaw rate |
| **6** | Adaptive NHC Update | `src/idr/filters/nhc.py`<br>`apply_adaptive_nhc_update()` | `ekf`, `yaw_rate`, `cov_scale` | `v_lat_after` (float)<br>`passed` (bool) | Injects $v_{\text{lat}} \approx 0, v_{\text{vert}} \approx 0$ as pseudo-measurements; inflates $\sigma_{\text{lat}}$ during turns |
| **7** | AI Velocity Update | `src/idr/filters/ekf.py`<br>`update_velocity()` | `v_ai_tracking` (m/s), `R_speed` | Updated `ekf.x`<br>Updated `ekf.P` | Bounds longitudinal scale error by updating forward velocity states |
| **8** | Causal Map Matching | `src/idr/mapmatch/causal_matcher.py`<br>`step()` | `ekf.x[0]`, `ekf.x[1]`, `ekf.x[6]` | `provisional_x`<br>`provisional_y` | Snaps position to nearest OSM road centerline if within 30 m corridor gate |
| **9** | Reacquisition Smoothing | `src/idr/eval/transition.py`<br>`apply_smoothing()` | `current_dr_enu`, `raw_gnss_enu` | `smoothed_enu` (2,) | Applied after blackout ends: smoothly merges DR position to restored GNSS fix over 3.5 s |

---
