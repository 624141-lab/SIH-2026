# 04 — IMU & SENSOR PROCESSING: VIBRATION, SHOCK & GRAVITY

---

## 1. Why Raw Smartphone IMU Cannot Be Used Directly

An accelerometer inside a smartphone does not measure "pure vehicle acceleration." It measures **specific force** ($\mathbf{f}$):
$$\mathbf{f} = \mathbf{a}_{\text{dynamic}} - \mathbf{g}$$
where $\mathbf{g}$ is the acceleration due to Earth's gravity ($9.80665\text{ m/s}^2$ pointing straight down toward the Earth's center).

Furthermore, when placed in a moving car, the sensor experiences:
1. **Engine Combustion Rumble:** High-frequency cyclic vibration ($20\text{--}100\text{ Hz}$).
2. **Road Surface Irregularities:** Potholes, expansion joints, and speed bumps creating transient shocks exceeding $20\text{ m/s}^2$ ($>2g$).
3. **In-Cabin Disturbances:** The driver tapping the touchscreen, passengers shifting in seats, or audio subwoofer vibrations.

If fed directly into a Kalman filter, these disturbances will be interpreted as massive changes in vehicle speed, corrupting state estimates.

---

## 2. The Multi-Stage Signal Processor (`VibrationMotionFilter`)

The vibration and shock mitigation logic is implemented in [`src/idr/filters/vibration_filter.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/vibration_filter.py).

### Signal Processing Flowchart

```text
Raw IMU [acc, gyro]
        │
        ▼
[1. Low-Pass Gravity Separation] ──► Isolates true gravity vector (alpha = 0.95)
        │
        ▼
[2. Sliding Window Buffer (N=10)] ──► 1.0 s historical window at 10 Hz
        │
        ▼
[3. Statistical Extraction] ────────► Computes Jerk, Rolling Variance, Kurtosis
        │
        ▼
[4. Motion State Classifier] ───────► STATIONARY, NORMAL, VIBRATION, SHOCK, etc.
        │
        ▼
[5. Dynamic Covariance Inflation] ──► Generates cov_scale (1.0 to 15.0)
```

---

## 3. Detailed Step-by-Step Processing

### Step 1: Low-Pass Gravity Separation
Gravity changes direction very slowly in the phone frame (only when the vehicle tilts on a grade). Vehicle acceleration changes quickly. We isolate gravity using an exponential low-pass filter:

$$\mathbf{g}_{\text{est}}(k) = \alpha \cdot \mathbf{g}_{\text{est}}(k-1) + (1 - \alpha) \cdot \mathbf{a}_{\text{raw}}(k)$$

```python
# src/idr/filters/vibration_filter.py lines 75-76
self.gravity_est = self.alpha_gravity * self.gravity_est + (1.0 - self.alpha_gravity) * acc
dyn_acc = acc - self.gravity_est
```
* **Parameter:** $\alpha = 0.95$ ($0.95$ weight on historical gravity, $0.05$ on current measurement).
* **Physical Meaning:** $\mathbf{g}_{\text{est}}$ tracks the gravity vector. $\mathbf{a}_{\text{dyn}}$ represents dynamic linear acceleration.

### Step 2: Jerk Computation
Jerk is the rate of change of acceleration ($m/s^3$):
$$jerk(k) = \frac{\|\mathbf{a}(k) - \mathbf{a}(k-1)\|}{\Delta t}$$
* **Purpose:** Potholes and road bumps cause sharp spikes in jerk ($jerk > 50\text{ m/s}^3$), whereas normal vehicle acceleration has low jerk ($jerk < 5\text{ m/s}^3$).

### Step 3: Rolling Variance & Kurtosis
Across the 10-sample rolling window ($1.0\text{ second}$), we calculate:
* **Acceleration Variance ($\sigma_a^2$):** Measures continuous chassis vibration.
* **Kurtosis ($\kappa$):** Measures the "tailedness" of the acceleration distribution:
  $$\kappa = \frac{\frac{1}{N}\sum (a_i - \bar{a})^4}{\left(\frac{1}{N}\sum (a_i - \bar{a})^2\right)^2}$$
* **Purpose:** A high kurtosis ($\kappa > 5.0$) signals an isolated transient shock (hitting a pothole) rather than continuous engine vibration.

### Step 4: Motion State Classification & Covariance Inflation
Based on statistical thresholds, the filter classifies the motion into:
* `NORMAL_DRIVING`: $\text{cov\_scale} = 1.0$ (Trust sensor measurements fully).
* `VIBRATION_DISTURBANCE`: $\text{cov\_scale} = 3.0 \cdot \min(5.0, \sigma_a^2 / 2.5)$ (Inflate process noise to discount vibration).
* `SHOCK_POTHOLE`: $\text{cov\_scale} = 15.0$ and clamp dynamic acceleration to $\le 5.0\text{ m/s}^2$ (Prevent filter divergence from vertical impact).

### Step 5: Where `cov_scale` Enters the Navigation Pipeline
In [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py#L166-L175):
```python
# Adaptive NHC measurement noise is inflated by sqrt(cov_scale)
sigma_lat_base = 0.05 * np.sqrt(cov_scale)

# AI velocity measurement noise is inflated by sqrt(cov_scale)
R_speed = max(0.1, sig_x * np.sqrt(cov_scale))
```
When driving over rough pavement, the EKF automatically reduces its reliance on noisy sensor inputs and relies more heavily on kinematic motion continuity.
