# AI-ML based Intelligent Dead Reckoning system for seamless navigation (SIH26168)

---

## 1. What Problem Is This Project Solving?

Every modern navigation app on your phone (Google Maps, Apple Maps, Waze) relies on **GNSS** (Global Navigation Satellite Systems, like GPS, GLONASS, or Galileo). Satellites orbit thousands of kilometers above Earth, beaming radio signals that tell your phone its exact location.

### The Real-World Problem: The "Blackout"
When a vehicle enters an underground tunnel, a multi-tier expressway flyover, a parking structure, or a dense urban canyon between skyscrapers:
1. Satellite signals are physically blocked (**GNSS outage**).
2. The blue navigation dot on your phone freezes, spins wildly, or jumps across city blocks (**multipath reflection**).
3. The driver misses critical highway exits, turns onto the wrong ramp, or loses turn-by-turn guidance.

### The Standard Industry Solution vs. Our Challenge
* **Automotive OEMs (Tesla, Mercedes):** Tap directly into the vehicle's internal computer (CAN-bus / OBD-II port) to read the car's wheel speed sensors and steering angle.
* **The SIH 26168 Challenge:** **We are NOT allowed to touch the car's wiring or OBD port.** The entire navigation system must run purely on an ordinary smartphone resting in a dashboard cradle, using low-cost internal MEMS sensors (accelerometer and gyroscope) to track the car through complete satellite blackouts exceeding **1 kilometer (>60 seconds)** with **less than 10% drift** (<100 meters error per 1,000 meters traveled).

---

## 2. Why Is This Extremely Difficult? (The "Sensor Trap")

If you have an accelerometer, physics says:
$$\text{acceleration} \xrightarrow{\text{integrate}} \text{velocity} \xrightarrow{\text{integrate}} \text{position}$$

In theory, you can calculate where the car is by integrating acceleration twice. In practice on a $200 smartphone, this fails catastrophically:
1. **Sensor Noise & Bias:** Smartphone sensors cost less than $1. They have tiny manufacturing imperfections called *biases*. Even when the phone is sitting still on a table, the accelerometer thinks it is accelerating slightly ($0.05\text{ m/s}^2$), and the gyroscope thinks it is slowly turning ($0.2^\circ/\text{s}$).
2. **Double Integration Explosion:** When you integrate a constant acceleration error twice:
   $$\text{position error} \propto \frac{1}{2} \cdot \text{bias} \cdot t^2$$
   After just 60 seconds, an uncorrected smartphone accelerometer drifts by over **$200\text{ meters}$ to $1,000\text{ meters}$**!
3. **Pure AI Fails Too:** Training a deep neural network to predict position directly from raw IMU sounds modern, but deep learning models have no concept of physical laws. Within 15 seconds of blackout, pure neural odometry accumulates heading drift and hallucinates the vehicle driving through buildings and rivers.

---

## 3. The Core Innovation: The Hybrid AI-Kinematic Pipeline

Rather than trusting pure physics or pure AI, our **Intelligent Dead Reckoning (IDR)** engine fuses deep learning with classical aerospace estimation and spatial road priors into an 8-stage causal pipeline:

```text
[Raw Smartphone IMU: Accel + Gyro @ 10 Hz]
                     │
                     ▼
       Stage 1: Ingestion & Calibration
                     │
                     ▼
       Stage 2: Phone-to-Vehicle Alignment  ──► Computes 3D Rotation Matrix R_{p→v}
                     │
                     ▼
       Stage 3: Vibration & Shock Filter    ──► Dynamic Noise Scaling s_{cov}
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
Stage 4: InertialOdomNet    Stage 5: 7-State Kinematic EKF
(1D-CNN + GRU Odometry)     (Predicts x, y, z, v_e, v_n, v_u, ψ)
Infers v_fwd + σ²                ▲
         │                       │
         └───── Pseudo-Speed ────┘
                     │
                     ▼
       Stage 6: Adaptive Centripetal NHC   ──► Enforces v_lat ≈ 0 (prevents side slip)
                     │
                     ▼
       Stage 7: Causal HMM Map Matcher     ──► 30m Corridor Gating on OSM Vector Graph
                     │
                     ▼
       Stage 8: GNSS Reacquisition Blending ──► C¹ Cosine Smoother (3.5s transition)
                     │
                     ▼
          [Continuous Meter-Level Trajectory]
```

---

## 3. Chronological Processing for a Single Timestep ($t_k$)

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

## 4. What Each Block Actually Does (In Simple Terms)

1. **Stage 1 (Ingestion):** Reads the phone's 3-axis accelerometer and 3-axis gyroscope at a steady $10\text{ Hz}$ clock ($0.1\text{ s}$ intervals).
2. **Stage 2 (Alignment):** A phone in a car cradle is tilted at an arbitrary angle. This module watches gravity while stationary and detects the first forward acceleration to calculate a 3D rotation matrix ($R_{phone \to vehicle}$), transforming phone measurements into true vehicle coordinates (X=forward, Y=right, Z=up).
3. **Stage 3 (Vibration Filter):** Engine rumble, potholes, and bass from speakers cause high-frequency vibrations that ruin dead reckoning. This filter separates gravity, detects shocks, and automatically inflates filter uncertainty ($s_{\text{cov}}$) so temporary jolts don't corrupt navigation.
4. **Stage 4 (AI Odometry - `InertialOdomNet`):** A custom neural network (1D-CNN + GRU) inspects a rolling 5-second window of IMU vibrations to infer instantaneous forward speed ($v_{\text{fwd}}$) and its confidence ($\sigma^2$), completely without vehicle wheel sensors.
5. **Stage 5 (Kinematic EKF):** An Extended Kalman Filter acts as the "conductor." It runs continuous physical equations of motion, propagating vehicle position and heading, using the AI speed as a measurement correction.
6. **Stage 6 (Adaptive NHC):** Cars have wheels; they drive forward, they don't slide sideways like hockey pucks. Non-Holonomic Constraints (NHC) force lateral and vertical speed to zero. Crucially, our system is *adaptive*: when the car takes a sharp turn, it relaxes this rule using centripetal physics ($\sigma_{\text{lat}}^2 \propto (v \cdot |\omega_z|)^2$) so tyre slip doesn't distort heading.
7. **Stage 7 (Causal Map Matcher):** Vehicles drive on roads, not in lakes. An online Hidden Markov Model (HMM) checks the OpenStreetMap (OSM) vector graph in real time with **zero future look-ahead**. If the vehicle is within $30\text{ meters}$ of a valid corridor, it projects the position along the road centerline. If off-road or unmapped, safety gates reject snapping.
8. **Stage 8 (Reacquisition Smoother):** When emerging from a tunnel, the first satellite fix might be 30 meters away from the dead-reckoned estimate. Instead of the vehicle "teleporting" instantly, a $C^1$ continuous cosine curve smoothly blends the trajectory back to satellites over $3.5\text{ seconds}$.

---
