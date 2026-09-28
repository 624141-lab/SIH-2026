# TABLE C — INFORMATION FLOW & CAUSALITY AUDIT

### Overview
This table audits the temporal boundary and data isolation rules enforced across the IDR navigation engine to prove zero data leakage from future samples or ground truth during simulated GNSS outages.

---

### Information Flow Validity Matrix

| Information Stream | Before Outage ($t < t_0$) | During Outage ($t_0 \le t \le t_{out}$) | After Outage ($t > t_{out}$) | Operational Purpose & Verification Proof |
| :--- | :---: | :---: | :---: | :--- |
| **GNSS Position** | **YES** | **NO** | **YES** | Pre-outage fix initialization; reacquisition after signal return |
| **GNSS Velocity** | **YES** | **NO** | **YES** | Pre-outage initial velocity anchor; zeroed during blackout |
| **GNSS Heading / Course** | **YES** | **NO** | **YES** | Initial yaw reference prior to blackout; propagated via IMU during outage |
| **Ground-Truth Position** | **NO** | **NO** | **NO** | Evaluator only; strictly quarantined from filter state vector |
| **Ground-Truth Velocity** | **NO** | **NO** | **NO** | Quarantined; AI model infers velocity strictly from 6-axis IMU |
| **Future Trajectory** | **NO** | **NO** | **NO** | Zero future look-ahead; causal time stepping ($\Delta t = 0.1\text{ s}$) |
| **Future Road Identity** | **NO** | **NO** | **NO** | HMM state propagation depends only on past and current trellis emissions |
| **OSM Road Geometry Prior**| **YES** | **YES** | **YES** | Offline geographic vector map loaded prior to trip; spatial road network bounds drift |
| **Phone IMU Window** | **YES** | **YES** | **YES** | 50-sample historical window $[t-5.0\text{s}, t]$; zero future frames used |
| **Reacquisition Smoother** | **NO** | **NO** | **YES** | $C^1$ continuous cosine blend (3.5 s) prevents discontinuous jump when GNSS locks |

---

### Strict Non-Holonomic Constraint (NHC) Formulation
During the blackout, vehicle non-holonomic constraints are applied causally at every filter step:
$$v_{\text{lat}} \approx 0, \quad v_{\text{up}} \approx 0$$
Adaptive measurement noise scaling dynamically relaxes constraints during cornering to prevent trajectory distortion:
$$\sigma_{\text{lat}}^2 = \sigma_{\text{base}}^2 + k_{\text{turn}} (v_{\text{fwd}} \cdot |\omega_z|)^2$$
where $\sigma_{\text{base}} = 0.05\text{ m/s}$ and $k_{\text{turn}} = 0.5$.
