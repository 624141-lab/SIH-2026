# 17 — FAILURE MODES & SYSTEM LIMITATIONS: AN HONEST AUDIT

---

## 1. Proven Failure Modes Matrix

Every navigation system has failure boundaries. Documenting them honestly proves engineering maturity to SIH evaluators.

| Failure Mode | Empirical Evidence in Benchmark | Root Physical Cause | Current Code Mitigation | Status / Phase 2 Upgrade |
| :--- | :--- | :--- | :--- | :--- |
| **1. Unbounded Gyro Drift in Unmapped Areas** | Drive `Vfa02` exhibits $74.97\%$ median drift | Consumer MEMS gyroscope has unmodeled bias ($0.2^\circ/\text{s}$). Without OSM maps, heading drifts open-loop | Adaptive NHC provides partial observability; $30\text{ m}$ corridor gate rejects bad roads | **Unresolved without Camera.** Solved in Phase 2 via Visual-Inertial Odometry (VIO) |
| **2. Stationary False Motion** | Scenario `Vfa02_t10s_d30s` stopped for 8.8s; drifted $546\text{ m}$ | When car is stopped, accelerometer noise integrates into low-speed creeping | Vibration filter flags stationary state, but zero-velocity update (ZUPT) is not fully coupled to EKF | **Partially Resolved.** Phase 2 requires strict ZUPT clamping ($\mathbf{v}=0$) when vehicle stops |
| **3. Along-Track Scale Lag** | Mapped scenarios exhibit 99% along-track error ($90.7\text{ m}$ in `t70s_d60s`) | 1D-CNN underestimates peak highway acceleration / deceleration transients | Blended velocity tracking: $0.85 v_{\text{inertial}} + 0.15 v_{\text{ai}}$ | **Partially Resolved.** Fine-tune neural odometry loss on acceleration derivative |
| **4. Map False Latching** | Scenario `Vfa02_t110s_d15s` error jumped by $+392\text{ m}$ after map matching | Rural dirt road missing in OpenStreetMap; HMM snapped to a perpendicular street | $30\text{ m}$ search radius gate + $0.20$ minimum confidence threshold | **Resolved for distances >30m.** Unmapped roads within 30m of highways still pose hazard |
| **5. Severe Centripetal Slip** | Baselines drift $>200\text{ m}$ on high-speed curved ramps | High lateral acceleration ($>4\text{ m/s}^2$) violates the classical assumption that $v_{\text{lat}} = 0$ | Adaptive centripetal variance inflation: $\sigma_{\text{lat}}^2 \propto (v \cdot \|\omega_z\|)^2$ | **RESOLVED in current code.** Chi-square gating rejects updates if $NIS > 9.21$ |

---

## 2. Operational Design Domain (ODD)

Evaluators will ask:
> *"What are the exact operating conditions where your system is guaranteed to work?"*

### Safe Operational Boundaries:
1. **Device Mounting:** Smartphone must be secured in a rigid dashboard or windshield cradle. (Unconstrained pedestrian pocket motion is outside scope).
2. **Vehicle Modality:** Four-wheeled road automobiles (passenger cars, taxis, commercial vans) obeying non-holonomic kinematics.
3. **Road Environment:** Mapped public roads, expressways, tunnels, and flyovers with standard OpenStreetMap coverage.
4. **Outage Duration:** Designed for satellite blackouts up to **$60.0\text{ seconds}$** (exceeding $1.1\text{ km}$ at highway speeds).

### Out-of-Scope Scenarios (Where System Degrades):
* Off-road desert or forest trails with zero digital map data.
* Boats, aircraft, or hovering drones (violate non-holonomic constraints).
* Prolonged stop-and-go gridlock exceeding 5 minutes without satellite or visual aiding.
