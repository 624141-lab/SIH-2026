# 16 — CURRENT RESULTS INTERPRETATION: PHYSICAL INSIGHTS

---

## 1. Master Performance Summary

The current verified IDR system was evaluated across **50 authentic scenarios** from the IO-VNBD dataset (26 on Drive `Vfa01` [Validation], 24 on Drive `Vfa02` [Untouched Test]):

| Dataset Domain | Scenarios | Median Drift (%) | Mean Drift (%) | P90 Drift (%) | Mean Error (m) | Pass Rate (<10%) | Within 20% Drift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mapped Corridors (`Vfa01`)** | 26 | **15.28%** | **17.82%** | **31.96%** | **107.58 m** | **34.6% (9/26)** | **73.1% (19/26)** |
| **Unmapped Rural (`Vfa02`)** | 24 | **74.97%** | **101.80%** | **216.70%** | **350.89 m** | **8.3% (2/24)** | **12.5% (3/24)** |
| **Combined Full Benchmark** | **50** | **31.06%** | **58.13%** | **163.85%** | **224.37 m** | **22.0% (11/50)**| **36.0% (18/50)** |

---

## 2. Deep Dive Into Representative Scenarios: What Physically Happened?

### 1. Best Case Overall: `Vfa02_t110s_d60s` (0.80% Drift / 7.80 m Error)
* **Outage:** $60.0\text{ s}$, Distance: $974.6\text{ m}$, Mean Speed: $58.5\text{ km/h}$.
* **Physical Conditions:** Steady cruising on a wide rural road with a single gentle $89.3^\circ$ sweeping curve, zero stops.
* **Why it Succeeded:** The neural network tracked the vehicle's steady speed with near-zero residual error, and the gyroscope bias happened to be near zero during this segment, allowing the vehicle to track true trajectory without map assistance.

### 2. Best Mapped Highway Run: `Vfa01_t30s_d60s` (2.58% Drift / 30.02 m Error)
* **Outage:** $60.0\text{ s}$, Distance: $1,162.5\text{ m}$, Mean Speed: $69.7\text{ km/h}$.
* **Physical Conditions:** High-speed highway cruising with gentle curved geometry ($18.7^\circ$ net heading change), zero stops.
* **Why it Succeeded:** The Causal HMM map matcher continuously locked onto the highway centerline. Cross-track lateral error was constrained to $<3.5\text{ meters}$. The only residual error was a minor $29.8\text{ m}$ longitudinal scale lag from the AI speed estimator over $1.16\text{ km}$.

### 3. Proposal Demonstration Run: `Vfa01_t70s_d60s` (7.97% Drift / 90.84 m Error)
* **Outage:** $60.0\text{ s}$, Distance: $1,139.7\text{ m}$, Mean Speed: $68.4\text{ km/h}$.
* **Physical Conditions:** Continuous highway cruising on Drive `Vfa01`, zero stops, uncalibrated cradle alignment.
* **Why it Succeeded:** In raw IMU integration, error explodes to $210.32\text{ m}$ ($18.45\%$). Our full pipeline keeps drift to **$7.97\%$ ($90.84\text{ m}$)**, comfortably below the SIH $10\%$ threshold ($<114.0\text{ m}$).

### 4. Median Case: `Vfa01_t110s_d15s` (31.65% Drift / 89.99 m Error)
* **Outage:** $15.0\text{ s}$, Distance: $284.3\text{ m}$, Mean Speed: $68.2\text{ km/h}$.
* **Physical Conditions:** Fast straight road segment.
* **Why Drift % Looks Higher:** The absolute position error ($89.99\text{ m}$) was comparable to the 60-second run. However, because the blackout lasted only 15 seconds, the distance traveled was short ($284\text{ m}$). A small denominator ($284\text{ m}$) causes the percentage to calculate as $31.65\%$, even though the car was tracked reasonably well during EKF initial settling.

### 5. Worst Case: `Vfa02_t10s_d30s` (311.19% Drift / 546.11 m Error)
* **Outage:** $30.0\text{ s}$, Distance: $175.5\text{ m}$, Mean Speed: $21.1\text{ km/h}$.
* **Physical Conditions:** Urban stop-and-go driving with an unmapped sharp $55.0^\circ$ corner. **The vehicle was stationary for 8.8 seconds (88 samples < 0.5 m/s).**
* **Why it Failed:** When the vehicle stopped at the intersection, the smartphone accelerometer continued integrating noisy bias, creating false forward motion. Simultaneously, open-loop gyroscope drift rotated the heading by $>40^\circ$. Because Drive `Vfa02` lacks OSM road vectors, the map matcher could not save the trajectory, causing $546\text{ meters}$ of open-loop divergence.

---

## 3. Two Critical Technical Questions Evaluators Will Ask

### Question 1: "Why can a system have near-zero cross-track error but large total position error?"
* **Answer:** Total position error is the vector sum: $e_{\text{total}} = \sqrt{e_{\text{along}}^2 + e_{\text{cross}}^2}$.
* In a mapped corridor, map matching constrains cross-track error to the lane width ($e_{\text{cross}} < 4.0\text{ m}$).
* However, map matching cannot constrain **along-track position** (how far along the road you have driven). If the AI speed estimator predicts $60\text{ km/h}$ when the car is actually traveling at $65\text{ km/h}$, you will accumulate an along-track scale error ($e_{\text{along}} = 80\text{ m}$) while remaining perfectly centered in your lane!

### Question 2: "Why can map matching sometimes improve a result and sometimes make it dramatically worse?"
* **Answer:** Map matching acts as an information amplifier.
  * **When the road is present in OSM (`Vfa01`):** It eliminates lateral drift, improving **96.2% of scenarios** (cutting error from 566 m to 44 m in `t50s_d60s`).
  * **When the road is missing from OSM (`Vfa02`):** The vehicle is driving on a new rural road that OpenStreetMap does not know about. If an unmapped vehicle drifts near an adjacent perpendicular street, the HMM snaps the vehicle to the wrong road, pulling it hundreds of meters away from ground truth!
  * **Our Solution:** A strict $30.0\text{ m}$ corridor search gate prevents the worst snaps, but true resilience on unmapped roads requires Phase 2 Visual-Inertial Odometry.
