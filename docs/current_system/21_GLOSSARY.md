# 21 — GLOSSARY: DEFINITIVE NAVIGATION TERMINOLOGY

---

### Along-Track Error
* **Simple Meaning:** Longitudinal position error (how far ahead or behind the car is relative to reality).
* **Technical Meaning:** The component of the position error vector projected onto the vehicle's heading unit vector: $e_{\text{along}} = |(\mathbf{p}_{\text{est}} - \mathbf{p}_{\text{gt}}) \cdot \mathbf{u}_{\text{along}}|$.
* **In This Project:** Accounts for $99.8\%$ of residual error on mapped highway runs due to slight AI speed scale lag.

### Cross-Track Error
* **Simple Meaning:** Lateral position error (how far sideways the car has drifted off its lane).
* **Technical Meaning:** The component of the position error vector projected perpendicular to the vehicle's heading unit vector: $e_{\text{cross}} = |(\mathbf{p}_{\text{est}} - \mathbf{p}_{\text{gt}}) \cdot \mathbf{u}_{\text{cross}}|$.
* **In This Project:** Bounded to $<4.15\text{ m}$ on mapped corridors by Causal HMM map matching.

### Dead Reckoning (DR)
* **Simple Meaning:** Navigating blind by using past position, speed, and heading to estimate current location.
* **Technical Meaning:** Recursive state propagation integrating self-contained inertial kinematic observations without external reference fixes.
* **In This Project:** Active during satellite blackout intervals ($t_0 \le t \le t_{\text{out}}$).

### EKF (Extended Kalman Filter)
* **Simple Meaning:** A smart mathematical algorithm that continuously estimates where the car is by combining physics equations with noisy sensor measurements.
* **Technical Meaning:** A non-linear recursive Bayesian estimator linearizing state transitions and measurement models via first-order Taylor series Jacobians.
* **In This Project:** Implemented as a 9-state filter in [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py).

### ENU (East-North-Up)
* **Simple Meaning:** A local map coordinate system where X points East, Y points North, and Z points Up into the sky.
* **Technical Meaning:** A Cartesian topocentric reference frame tangent to the WGS-84 reference ellipsoid at a designated local geodetic origin $(\phi_0, \lambda_0)$.
* **In This Project:** All vehicle trajectories are tracked and evaluated in meters ENU.

### GNSS (Global Navigation Satellite System)
* **Simple Meaning:** Satellite navigation systems, including American GPS, European Galileo, and Indian NavIC.
* **Technical Meaning:** Space-based radio-positioning constellations providing absolute 3D position, velocity, and time (PVT).
* **In This Project:** Used for pre-blackout initialization and post-blackout reacquisition.

### HDOP (Horizontal Dilution of Precision)
* **Simple Meaning:** A number indicating how good or bad satellite geometry is in the sky (lower is better).
* **Technical Meaning:** Multiplier representing geometric magnification of satellite ranging errors on the horizontal position plane.
* **In This Project:** In [`src/idr/filters/gnss_monitor.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/gnss_monitor.py), HDOP $> 3.5$ triggers the degraded GNSS state.

### Heteroscedastic Uncertainty
* **Simple Meaning:** The neural network tells you how confident it is with every single prediction.
* **Technical Meaning:** An aleatoric uncertainty formulation where the network outputs both the mean prediction $\hat{y}$ and an input-dependent predicted variance $\sigma^2(x)$.
* **In This Project:** `InertialOdomNet` outputs $\log\sigma_x^2$, used directly to set the EKF measurement covariance $R_{\text{speed}}$.

### HMM (Hidden Markov Model)
* **Simple Meaning:** A statistical model that finds the most likely road you are driving on by balancing distance to roads with heading consistency.
* **Technical Meaning:** A doubly stochastic probabilistic graphical model evaluating emission likelihoods (spatial distance) and transition probabilities (graph network topology).
* **In This Project:** Implemented causally in [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py).

### IMU (Inertial Measurement Unit)
* **Simple Meaning:** A microchip containing an accelerometer and gyroscope.
* **Technical Meaning:** A 6-DOF sensor package measuring 3-axis specific force and 3-axis angular rates.
* **In This Project:** Standard smartphone MEMS IMU sampled at $10.0\text{ Hz}$.

### Jacobian Matrix ($F$, $H$)
* **Simple Meaning:** A matrix of partial derivatives that tells the filter how much an error in one state (like heading) affects another state (like position).
* **Technical Meaning:** First-order Taylor series expansion: $F = \frac{\partial f}{\partial \mathbf{x}}$ and $H = \frac{\partial h}{\partial \mathbf{x}}$.
* **In This Project:** Derived analytically in [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py).

### NHC (Non-Holonomic Constraints)
* **Simple Meaning:** The physical rule that car wheels roll forward but cannot slide sideways.
* **Technical Meaning:** Kinematic differential constraints on wheeled mobile robots enforcing zero lateral and vertical body velocities ($v_{\text{lat}} \approx 0, v_{\text{up}} \approx 0$).
* **In This Project:** Formulated as adaptive EKF pseudo-measurements in [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py).

### NIS (Normalized Innovation Squared)
* **Simple Meaning:** A mathematical test that checks whether a sensor reading is an impossible anomaly.
* **Technical Meaning:** Chi-square test statistic: $\text{NIS} = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$ comparing innovation vector $\mathbf{y}$ against innovation covariance $\mathbf{S}$.
* **In This Project:** Used in NHC ($\text{NIS} \le 9.21$) and GNSS quality monitoring ($\text{NIS} \le 11.34$) to reject corrupting updates.

### OSM (OpenStreetMap)
* **Simple Meaning:** A free, globally editable geographic map of all real-world road networks.
* **Technical Meaning:** An open vector spatial database providing road geometries, node intersections, and roadway classifications.
* **In This Project:** Cached locally as NetworkX directed graphs in `data/osm/`.

### Sensor Bias
* **Simple Meaning:** A permanent offset in the sensor that makes it output a small non-zero number even when resting still.
* **Technical Meaning:** The expected non-zero offset of a sensor output under zero true input: $b = \mathbb{E}[\text{meas}] - \text{truth}$.
* **In This Project:** Actively tracked and estimated in the EKF state vector as $b_a$ and $b_\omega$.

### ZUPT (Zero Velocity Update)
* **Simple Meaning:** Forcing speed to zero when the car stops at a red light.
* **Technical Meaning:** A pseudo-measurement update setting the velocity vector to zero whenever stationarity is detected, arresting open-loop integration drift.
* **In This Project:** Identified as a mandatory Phase 2 upgrade to resolve stop-and-go errors.
