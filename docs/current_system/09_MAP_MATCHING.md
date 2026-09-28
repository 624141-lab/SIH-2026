# 09 — MAP MATCHING: CAUSAL HMM ROAD NETWORK FUSION

---

## 1. What Is Map Matching From Zero?

Cars do not drive randomly across open fields; they travel along structured civil engineering corridors: paved roads, highways, bridges, and tunnels.

**Map Matching** is the mathematical process of snapping noisy dead-reckoned coordinates onto a digital road network map (OpenStreetMap, OSM):
* If the inertial filter thinks the car is 8 meters to the right in a roadside ditch, map matching projects the vehicle onto the true lane centerline.
* Because roads are continuous lines, map matching **completely eliminates cross-track lateral drift** on mapped corridors.

---

## 2. The Road Graph Representation

In [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py), the local road network is represented as a directed graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
* **Nodes ($\mathcal{V}$):** Intersections, roundabouts, and road ends (with latitude/longitude converted to local ENU coordinates).
* **Edges ($\mathcal{E}$):** Road segments represented as geometric `LineString` polylines with:
  * Length $L_e$
  * Heading azimuth $\theta_e \in [-\pi, \pi]$
  * Spatial indexing via a fast spatial k-d tree (`scipy.spatial.cKDTree`) for $O(\log |\mathcal{E}|)$ candidate search.

---

## 3. The Hidden Markov Model (HMM) Formulation

A simple "snap to nearest road" algorithm fails at highway interchanges where an overpass crosses directly above a local street. To pick the correct road, we use a **Hidden Markov Model (HMM)**.

```text
Time t-1:   [Candidate Road A]        [Candidate Road B]
                    │                         │
                    └───────────┬─────────────┘
                                │ Transition Probability P(Road_t | Road_{t-1})
                                ▼
Time t:     [Candidate Road C]        [Candidate Road D]
                    ▲
                    │ Emission Probability P(z_t | Road_C)
            [Noisy EKF Position z_t]
```

### 1. Emission Probability ($s_{\text{dist}}$)
How close is the estimated EKF position $(x_t, y_t)$ to candidate road segment $r_i$?
$$s_{\text{dist}} = \exp\left(-\frac{d(z_t, r_i)^2}{2\sigma_z^2}\right)$$
where $d(z_t, r_i)$ is the perpendicular distance to the road segment and $\sigma_z = 8.0\text{ meters}$.

### 2. Heading Alignment Probability ($s_{\text{head}}$)
Does the vehicle's motion heading ($\psi_t$) match the direction of the road ($\theta_{r_i}$)?
$$s_{\text{head}} = \max\left(0, \cos(\psi_t - \theta_{r_i})\right)$$
A car cannot be driving backwards down a one-way highway.

### 3. Transition Probability ($s_{\text{trans}}$)
How realistic is moving from candidate road $r_{t-1}$ to candidate road $r_t$?
$$\Delta d = |\|\mathbf{p}_t - \mathbf{p}_{t-1}\| - d_{\text{graph}}(r_{t-1}, r_t)|$$
$$s_{\text{trans}} = \exp\left(-\frac{\Delta d}{\beta}\right)$$
If switching to road $r_t$ requires an impossible 200-meter leap across a concrete barrier in 100 ms, the transition probability plummets to near zero.

### 4. Multi-Hypothesis Confidence Score
In `src/idr/mapmatch/causal_matcher.py` line 358:
$$\text{confidence} = 0.35 \cdot s_{\text{dist}} + 0.25 \cdot s_{\text{head}} + 0.25 \cdot s_{\text{trans}} + 0.15 \cdot s_{\text{cont}}$$

---

## 4. How the System Guarantees Zero Future Look-Ahead (Causality)

### Why Old Retrospective Viterbi Was Purged
Earlier offline research scripts ran full backward Viterbi smoothing across the entire 60-second trip *after* the drive ended. That gave an unrealistically low drift ($3.02\text{ m} / 0.29\%$) because the computer looked into the future to see where the car came out of the tunnel!

### The Current Causal Implementation
In our production pipeline ([`evaluate_hardened_pipeline.py` line 183](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py#L183)):
```python
traj_s6 = causal_matcher.match_trajectory_causal(traj_s5, use_finalized=False)
```
* `use_finalized=False` instructs the matcher to output **`provisional_x`, `provisional_y`**.
* The provisional match is computed **instantaneously at timestep $t$** using strictly forward trellis recursion.
* **Look-ahead is exactly $0.0\text{ seconds}$** (Zero future frames, zero backward smoothing).

---

## 5. Out-of-Corridor Safety Gating (Preventing False Snapping)

What happens if the vehicle drives into an unmapped parking garage, a rural dirt track, or an unmapped farm road?

Without safety gating, the matcher would force-snap the vehicle to the nearest known highway 500 meters away!

### The Two Safety Gates:
```python
# src/idr/mapmatch/causal_matcher.py line 361
if confidence < 0.20 or prov_c.dist > 30.0:
    return CausalMatchOutput(
        provisional_x = est_x,
        provisional_y = est_y,
        is_accepted = False
    )
```
1. **$30.0\text{ m}$ Corridor Gate:** If the nearest OSM road is farther than $30.0\text{ meters}$, snapping is **REJECTED**. The system outputs the raw unmapped EKF coordinate.
2. **Confidence Gate ($0.20$):** If the candidate road has a conflicting heading or disjoint connectivity, snapping is **REJECTED**.

---

## 6. What Happens When OSM Maps Are Missing? (Drive `Vfa02` Analysis)

* On **Drive `Vfa01`** (mapped urban/highway), the road network is fully mapped. The matcher improves **96.2% of scenarios** (25/26), cutting mean error from $241\text{ m}$ to $107\text{ m}$.
* On **Drive `Vfa02`** (untouched rural road), OpenStreetMap lacks vector geometry for small rural paths. The matcher rejects invalid candidates, but cannot bound lateral gyro drift.
* This honest empirical boundary proves why Phase 2 Visual-Inertial Odometry is needed.
