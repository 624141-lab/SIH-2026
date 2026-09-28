# 10 — GNSS FUSION & OUTAGE HANDLER: FSM & REACQUISITION

---

## 1. The GNSS Lifecycle State Machine

An automated dead-reckoning engine must know *when* to trust satellites and *when* to switch to autonomous inertial mode.

The detection engine is implemented in [`src/idr/filters/gnss_monitor.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/gnss_monitor.py) as `GNSSDeficitDetector`, governed by a 4-state Finite State Machine (FSM):

```text
       ┌───────────────────────────────┐
       │         NORMAL_GNSS           │ ◄─────────────────────────┐
       └──────────────┬────────────────┘                           │
                      │ HDOP > 3.5 or NIS > 11.34                  │
                      │ (3 consecutive cycles)                     │
                      ▼                                            │
       ┌───────────────────────────────┐                           │
       │           DEGRADED            │                           │
       └──────────────┬────────────────┘                           │
                      │ Fix age > 0.5s or Satellite lost           │ 5 clean fixes
                      ▼                                            │ (debounced)
       ┌───────────────────────────────┐                           │
       │          GNSS_DENIED          │                           │
       │     (Dead Reckoning Active)   │                           │
       └──────────────┬────────────────┘                           │
                      │ First valid fix arrives                    │
                      ▼                                            │
       ┌───────────────────────────────┐                           │
       │          RECOVERING           │                           │
       │   (C¹ Cosine Blend Active)    ├───────────────────────────┘
       └───────────────────────────────┘
```

---

## 2. Signal Health & Degradation Criteria

The detector checks 4 physical metrics at every step:
1. **Fix Timeout (`max_fix_age_sec = 0.5`):** If no GNSS NMEA sentence arrives within $500\text{ ms}$, the receiver has lost line-of-sight.
2. **Horizontal Dilution of Precision (`hdop_threshold = 3.5`):** HDOP measures satellite geometry. If satellites cluster in a narrow sliver of sky between buildings, HDOP $> 3.5$ indicates high multipath error.
3. **Positional Variance (`pos_sigma_threshold = 8.0 m`):** Reports reported GPS 1-sigma uncertainty.
4. **Normalized Innovation Squared (`nis_gate_threshold = 11.34`):** If a satellite fix suddenly claims the car jumped 50 meters sideways in 100 ms, the Chi-square gate (3-DOF, $p=0.01$) detects spoofing/multipath and rejects the fix.
5. **Hysteresis & Debouncing:** Requires 3 consecutive bad frames to declare an outage, and 5 consecutive clean frames to declare recovery, preventing rapid chatter under trees.

---

## 3. GNSS Usage Matrix Across Blackout Phases

| Information Stream | Before Outage ($t < t_0$) | During Outage ($t_0 \le t \le t_{\text{out}}$) | After Outage ($t > t_{\text{out}}$) | Operational Purpose |
| :--- | :---: | :---: | :---: | :--- |
| **GNSS Position** | **YES** | **STRICT NO** | **YES** | Anchors trip origin $p_0$; blends during reacquisition |
| **GNSS Velocity** | **YES** | **STRICT NO** | **YES** | Calibrates initial velocity and phone forward yaw axis |
| **GNSS Heading / Course** | **YES** | **STRICT NO** | **YES** | Initializes vehicle heading state $\psi_0$ |
| **Ground-Truth Data** | **NO** | **NO** | **NO** | Strictly quarantined in evaluator; 0 state access |

---

## 4. GNSS Reacquisition: The Discontinuity Problem

When a car exits a 60-second highway tunnel, two things happen:
1. Dead reckoning has accumulated an error (e.g. $90.84\text{ meters}$ in `Vfa01_t70s_d60s`).
2. Satellites lock on, providing the true geodetic position.

### The Naive Mistake: Hard Reset
If the navigation app immediately jumps from the dead-reckoned dot to the satellite dot in a single frame ($100\text{ ms}$):
* The velocity state spikes to $90.84\text{ m} / 0.1\text{ s} = 908\text{ m/s}$ ($3,270\text{ km/h}$)!
* The Kalman filter covariance matrix destabilizes, causing numerical instability.

### The Solution: $C^1$ Continuous Cosine-Blend Smoother
Implemented in [`src/idr/eval/transition.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/eval/transition.py) as `ReacquisitionSmoother`:

When satellite reception returns, we blend dead reckoning position $\mathbf{p}_{\text{dr}}(t)$ toward raw satellite fix $\mathbf{p}_{\text{gnss}}(t)$ over $T = 3.5\text{ seconds}$ using a **cosine bell curve**:

$$w(t) = \frac{1}{2}\left(1 - \cos\left(\pi \frac{t}{T}\right)\right), \quad t \in [0, T]$$
$$\mathbf{p}_{\text{smooth}}(t) = (1 - w(t)) \cdot \mathbf{p}_{\text{dr}}(t) + w(t) \cdot \mathbf{p}_{\text{gnss}}(t)$$

```text
Weight w(t)
 1.0 ┌──────────────────────────────────────────────┐ ── 100% GNSS Fix
     │                                     . - ' "  │
 0.5 │                             . - ' "          │
     │                     . - ' "                  │
 0.0 └─── . - ' " ──────────────────────────────────┘ ── 100% Dead Reckoning
     t=0s (Exit tunnel)                          t=3.5s (Full Convergence)
```

### Properties:
* **Zero Position Jump:** At $t=0$, $w=0 \implies \mathbf{p}_{\text{smooth}} = \mathbf{p}_{\text{dr}}$.
* **Zero Velocity Spike ($C^1$ Continuous):** Because $\frac{dw}{dt} = 0$ at $t=0$ and $t=T$, the transition has continuous first derivatives, ensuring smooth turn-by-turn guidance without screen flicker.
