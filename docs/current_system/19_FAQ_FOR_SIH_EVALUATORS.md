# 19 — FAQ FOR SIH EVALUATORS: 32 TOUGH QUESTIONS ANSWERED

---

## Category 1: System Architecture & Algorithms

### Q1: Why combine AI with an Extended Kalman Filter instead of using an End-to-End Deep Learning network?
**Answer:** Pure neural networks are black boxes with no concept of physical laws. If an end-to-end network encounters a bumpy road or an unseen speed profile, it diverges unboundedly without physical limits. By embedding `InertialOdomNet` as a velocity pseudo-measurement inside an Extended Kalman Filter, we retain kinematic guarantees: the vehicle's position and heading are strictly governed by Newton's equations of motion and vehicle non-holonomic constraints.

### Q2: Why can't we just integrate raw smartphone IMU data?
**Answer:** As proven mathematically, double-integrating accelerometer bias causes position error to grow quadratically ($e \propto \frac{1}{2} b_a t^2$). In our baseline experiments on Drive `Vfa01`, raw IMU mechanization drifted by $210.32\text{ m}$ ($18.45\%$) within just 60 seconds.

### Q3: Why use an Extended Kalman Filter (EKF) instead of an Unscented Kalman Filter (UKF)?
**Answer:** The vehicular kinematic equations in 2D local ENU have straightforward analytical Jacobians ($F$ and $H$), which can be computed in microseconds. An Unscented Kalman Filter requires propagating $2n+1 = 19$ sigma points through non-linear functions at every step, increasing CPU overhead by $5\times\text{--}8\times$ with negligible accuracy benefit on planar vehicle motion.

### Q4: What does Non-Holonomic Constraint (NHC) contribute to the system?
**Answer:** A road vehicle cannot slide sideways without skidding ($v_{\text{lat}} \approx 0$). In our EKF, the measurement Jacobian term $\frac{\partial v_{\text{lat}}}{\partial \psi} = -v_E \cos\psi - v_N \sin\psi$ couples lateral velocity directly to vehicle heading. This makes heading observable during straight-line driving without a compass or satellite updates.

### Q5: Why does NHC need to be "Adaptive"?
**Answer:** Classical rigid NHC assumes zero lateral velocity ($v_{\text{lat}}=0$) at all times. During sharp cornering, tyres deform and vehicle slip angle reaches $2^\circ\text{--}5^\circ$. Enforcing $v_{\text{lat}}=0$ during turns distorts the heading estimate. Our adaptive formulation inflates lateral measurement noise dynamically based on centripetal acceleration: $\sigma_{\text{lat}}^2 = \sigma_{\text{base}}^2 + k_{\text{turn}}(v_{\text{fwd}}|\omega_z|)^2$.

### Q6: What does Map Matching contribute, and why OpenStreetMap (OSM)?
**Answer:** In long satellite blackouts, even a tiny residual gyroscope bias ($0.05^\circ/\text{s}$) causes cross-track position to drift. Map matching binds the trajectory to the true geometric road centerline. We use OpenStreetMap because it is globally open, license-free, supports offline vector graph caching, and covers over 95% of national highway corridors.

---

## Category 2: AI Neural Odometry (`InertialOdomNet`)

### Q7: Exactly what does the AI model predict?
**Answer:** It consumes a 50-step ($5.0\text{ s}$) historical window of 6-axis IMU data and predicts a 4-dimensional vector: $[\Delta x_{\text{body}}, \Delta y_{\text{body}}, \log\sigma_x^2, \log\sigma_y^2]$, representing 2D body displacement over the window and heteroscedastic aleatoric uncertainty.

### Q8: What features does the AI receive as input?
**Answer:** Exactly 6 raw IMU channels: 3-axis accelerometer $[a_x, a_y, a_z]$ and 3-axis gyroscope $[\omega_x, \omega_y, \omega_z]$ sampled at $10\text{ Hz}$ in the phone frame. It receives zero vehicle speed, zero wheel ticks, and zero GNSS signals.

### Q9: Does the AI model use GNSS or ground truth during the outage?
**Answer:** **STRICTLY NO.** During inference, ground-truth speed and GNSS coordinates are completely disconnected. The model infers speed purely from high-frequency vibration and motion harmonics.

### Q10: How was the model trained?
**Answer:** Trained offline on authentic vehicular drives (`M`, `S`, `Vta`, `Vtb`) from the IO-VNBD dataset. Supervised targets were derived from vehicle ECU wheel speeds during the offline training phase only. The model was trained using Gaussian Negative Log-Likelihood (NLL) with AdamW optimization.

### Q11: How well does the model track speed?
**Answer:** Evaluated against ground-truth vehicle ECU telemetry on Drive `Vfa01`, `InertialOdomNet` achieves a Mean Absolute Error (MAE) of **$1.14\text{ m/s}$ ($4.1\text{ km/h}$)** across the entire speed range ($0\text{--}80\text{ km/h}$).

---

## Category 3: Vehicle Navigation & Calibration

### Q12: How do you estimate vehicle heading before the blackout?
**Answer:** Prior to the blackout, when GNSS is healthy, heading is initialized from the GNSS Course Over Ground (velocity vector azimuth $\psi_0 = \text{atan2}(v_N, v_E)$). During the blackout, heading is propagated via gyroscope integration and corrected by adaptive NHC and road map alignment.

### Q13: What happens if the phone is placed crooked in the cradle?
**Answer:** The `PhoneToVehicleAligner` module runs a two-step calibration:
1. Static gravity vector leveling determines Pitch and Roll.
2. Initial forward vehicle acceleration during straight driving determines the forward Yaw axis ($X_v$).
All subsequent IMU readings are rotated via rotation matrix $\mathbf{R}_{pv}$.

### Q14: How do you handle gyroscope bias ($b_\omega$)?
**Answer:** Gyro bias is modeled as an active state element (`x[8]`) in the 9-state EKF. During straight driving, adaptive NHC innovations provide cross-terms that estimate and cancel gyro bias in real time.

---

## Category 4: Map Matching & Causality

### Q15: What happens if the vehicle drives onto an unmapped rural road?
**Answer:** Our Causal HMM matcher incorporates an **Out-of-Corridor safety gate ($30.0\text{ meters}$)** and a **confidence gate ($0.20$)**. If the nearest OSM road is farther than $30\text{ m}$ or has an incompatible heading, map snapping is **REJECTED**, and the system outputs the raw EKF dead-reckoned trajectory.

### Q16: Can map matching make the result worse?
**Answer:** Yes, if digital maps are missing. On Drive `Vfa02` (unmapped rural dirt road), the HMM occasionally snapped dead-reckoned positions to perpendicular secondary roads, increasing mean error from $241\text{ m}$ to $350\text{ m}$. We disclose this limitation openly to justify Phase 2 Visual Odometry.

### Q17: Is your map matcher causal? Does it look into the future?
**Answer:** **It is 100% causal.** We set `use_finalized=False`, meaning the matcher outputs the instantaneous forward trellis match (`provisional_x, y`) with **$0.0\text{ seconds}$ look-ahead**. Retrospective Viterbi backward smoothing was explicitly purged from the codebase.

---

## Category 5: Benchmark & Results

### Q18: What is the SIH 26168 target criterion?
**Answer:** Drift $< 10.0\%$ of total distance traveled during complete satellite blackouts exceeding 1 kilometer.

### Q19: How do you calculate drift percentage?
**Answer:**
$$\text{Drift}(\%) = \frac{\text{Final Position Error (meters)}}{\text{Total Distance Traveled (meters)}} \times 100\%$$

### Q20: What are your verified results on the 1.14 km highway demonstration scenario?
**Answer:** On scenario `Vfa01_t70s_d60s` ($1,139.7\text{ m}$ traveled over $60.0\text{ s}$ blackout):
* Baseline 1 (Raw IMU): $210.32\text{ m}$ ($18.45\%$) $\to$ FAIL.
* Baseline 2 (Standard EKF): $222.67\text{ m}$ ($19.54\%$) $\to$ FAIL.
* Baseline 3 (EKF + NHC): $221.98\text{ m}$ ($19.48\%$) $\to$ FAIL.
* **Proposed IDR Pipeline: $90.84\text{ m}$ ($7.97\%$) $\to$ PASS (<10% SIH Target).**

### Q21: What is the overall pass rate across the full 50-scenario benchmark?
**Answer:** Across all 50 scenarios (including unmapped rural routes), **$22.0\%$ (11/50)** strictly beat $<10\%$ drift, and **$36.0\%$** achieve $\le 20\%$ drift. In mapped highway corridors (`Vfa01`), the pass rate is **$34.6\%$ (9/26)** with a median drift of **$15.28\%$**.

### Q22: What is your best scenario result?
**Answer:** Overall best is `Vfa02_t110s_d60s` with **$0.80\%$ drift ($7.80\text{ m}$ error over $974.6\text{ m}$)**. Best on mapped highway is `Vfa01_t30s_d60s` with **$2.58\%$ drift ($30.02\text{ m}$ error over $1,162.5\text{ m}$)**.

### Q23: What is your worst scenario result, and why did it fail?
**Answer:** Scenario `Vfa02_t10s_d30s` ($311.19\%$ drift, $546.11\text{ m}$ error). The vehicle stopped at an intersection for $8.8\text{ seconds}$ on an unmapped rural road. Accelerometer bias integrated into false creeping speed without ZUPT, and unconstrained gyro drift rotated the trajectory into open terrain.

---

## Category 6: Edge Deployment & Practical Feasibility

### Q24: Can this actually run on a standard smartphone in real time?
**Answer:** **Yes.** The neural odometry model was exported to ONNX format ($1.80\text{ MB}$). Benchmarked on standard mobile CPUs, inference requires only **$0.93\text{ ms}$** per window. With an update rate of $10\text{ Hz}$ ($100\text{ ms}$ time budget), the entire pipeline consumes $<2\%$ CPU utilization.

### Q25: How does the system transition when GNSS returns after a tunnel?
**Answer:** It uses a $C^1$ continuous **Reacquisition Smoother** with a cosine bell curve ($w(t) = \frac{1}{2}(1 - \cos(\pi t / 3.5))$) over $3.5\text{ seconds}$. This eliminates abrupt position leaps or speed spikes, ensuring smooth turn-by-turn map guidance.

### Q26: Does your system require vehicle OBD-II or CAN-bus wiring?
**Answer:** **No.** It is 100% standalone on the smartphone, verified by test script `tests/test_zero_vehicle_connection.py`.

### Q27: How does your system handle traffic vibration and bass from music?
**Answer:** The `VibrationMotionFilter` calculates rolling variance and kurtosis over a 1.0 s window. When vibration exceeds $2.5\text{ m}^2/\text{s}^4$, it dynamically inflates the Kalman covariance scale ($s_{\text{cov}}$ up to 15.0), preventing false speed updates.

### Q28: What is the memory footprint on device?
**Answer:** The ONNX model is $1.80\text{ MB}$. Offline OpenStreetMap vector cache for an entire metropolitan region requires $<25\text{ MB}$. Total runtime RAM is $<40\text{ MB}$.

### Q29: What happens if the driver touches or adjusts the phone in the cradle?
**Answer:** Transient hand motion produces sharp spikes in jerk ($>20\text{ m/s}^3$) and angular rate. The vibration filter flags `PHONE_MOTION`, temporarily freezes the EKF propagation, and re-levels the vertical gravity vector once the phone becomes stationary.

### Q30: What is your Phase 2 development roadmap for the Grand Finale?
**Answer:**
1. Integrate Visual-Inertial Odometry (VIO) using the smartphone rear camera to eliminate open-loop gyro drift in unmapped rural areas.
2. Package the Python/ONNX engine into a native Android Java/Kotlin app with background SensorManager services.
3. Implement Zero Velocity Updates (ZUPT) for stationary traffic intervals.

### Q31: Why did historical project reports mention 3.02 m (0.29%) drift?
**Answer:** That result came from an early diagnostic script (`test_viterbi_backward.py`) running retrospective backward Viterbi smoothing across the completed trip. Because backward smoothing uses future trajectory information, we rejected and purged that number, replacing it with the authentic, verified causal result of **$90.84\text{ m}$ ($7.97\%$)**.

### Q32: Why did scenario `t70s_d60s` evaluate to 44.66 m in the benchmark script but 90.84 m in the proposal figures?
**Answer:** Both results pass the SIH $<10\%$ benchmark ($<114.0\text{ m}$). `evaluate_hardened_pipeline.py` uses pre-drive calibrated alignment ($R_{pv}$ estimated from motion), yielding $44.66\text{ m}$ ($3.92\%$). `generate_proposal_figures.py` sets identity alignment $R=\mathbf{I}$ to simulate a zero-calibration cradle mount, yielding $90.84\text{ m}$ ($7.97\%$). We intentionally chose the more conservative $7.97\%$ figure for our proposal slides to guarantee 100% defensibility.
