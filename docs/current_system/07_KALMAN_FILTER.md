# 07 — KALMAN FILTER: 9-STATE KINEMATIC EKF

---

## 1. Kalman Filter Intuition: Combining Physics with Measurements

Imagine you are driving with your eyes closed:
1. **The Physics Model (Prediction):** You pressed the gas pedal for 3 seconds. Physics says you should be moving at $40\text{ km/h}$. But you are uncertain because the road might have been slightly uphill.
2. **The Sensor Observation (Measurement):** You open your eyes briefly and look at the speedometer. It says $36\text{ km/h}$. But the speedometer needle vibrates slightly.

A **Kalman Filter** is an optimal mathematical balance:
* If your physics model is very accurate and your sensor is noisy, it trusts the physics model.
* If your physics model is drifting (like an IMU double-integral) and your sensor is reliable, it pulls the estimate toward the sensor.
* It tracks not only the estimate ($\mathbf{x}$), but also how uncertain it is ($\mathbf{P}$, the covariance matrix).

---

## 2. The Current Filter Architecture

The filter is implemented in [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) as an **Extended Kalman Filter (EKF)**. An *Extended* Kalman filter is used because vehicular kinematics are non-linear (heading involves sines and cosines).

### Current 9-State Vector ($\mathbf{x} \in \mathbb{R}^9$)

| State Element | Variable | Physical Meaning | Unit | Initial Variance ($P_0$) |
| :---: | :---: | :--- | :---: | :---: |
| `x[0]` | $p_E$ | Position East (World ENU) | $\text{meters}$ | $10.0\text{ m}^2$ |
| `x[1]` | $p_N$ | Position North (World ENU) | $\text{meters}$ | $10.0\text{ m}^2$ |
| `x[2]` | $p_U$ | Position Up (World ENU) | $\text{meters}$ | $10.0\text{ m}^2$ |
| `x[3]` | $v_E$ | Velocity East | $\text{m/s}$ | $1.0\text{ (m/s)}^2$ |
| `x[4]` | $v_N$ | Velocity North | $\text{m/s}$ | $1.0\text{ (m/s)}^2$ |
| `x[5]` | $v_U$ | Velocity Up | $\text{m/s}$ | $1.0\text{ (m/s)}^2$ |
| `x[6]` | $\psi$ | Heading / Yaw angle (from East) | $\text{radians}$ | $(5^\circ)^2 = 7.6 \times 10^{-3}\text{ rad}^2$ |
| `x[7]` | $b_a$ | Forward accelerometer bias | $\text{m/s}^2$ | $0.2^2 = 0.04\text{ (m/s}^2)^2$ |
| `x[8]` | $b_\omega$| Gyroscope yaw rate bias | $\text{rad/s}$ | $0.02^2 = 4 \times 10^{-4}\text{ (rad/s)}^2$ |

---

## 3. Step 1: Kinematic Mechanization (Prediction)

Every $0.1\text{ seconds}$ ($\Delta t = 0.1$), the filter consumes forward acceleration $a_{\text{fwd}}$ and yaw rate $\omega_z$:

### State Transition Model ($\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k)$)
1. **Bias-Corrected Inputs:**
   $$a_{\text{corr}} = a_{\text{fwd}} - b_a$$
   $$\omega_{\text{corr}} = \omega_z - b_\omega$$
2. **Current Forward Speed along Vehicle Heading:**
   $$v_{\text{fwd}} = v_E \cos\psi + v_N \sin\psi$$
3. **World Acceleration with Centripetal Turning Dynamics:**
   When a car turns, centripetal acceleration ($\boldsymbol{\omega} \times \mathbf{v}$) pushes laterally. In world coordinates:
   $$a_E = a_{\text{corr}} \cos\psi - v_{\text{fwd}} \cdot \omega_{\text{corr}} \sin\psi$$
   $$a_N = a_{\text{corr}} \sin\psi + v_{\text{fwd}} \cdot \omega_{\text{corr}} \cos\psi$$
4. **State Propagation:**
   $$p_E(k) = p_E(k-1) + v_E \Delta t + \frac{1}{2} a_E \Delta t^2$$
   $$p_N(k) = p_N(k-1) + v_N \Delta t + \frac{1}{2} a_N \Delta t^2$$
   $$v_E(k) = v_E(k-1) + a_E \Delta t$$
   $$v_N(k) = v_N(k-1) + a_N \Delta t$$
   $$\psi(k) = \text{wrap}_{[-\pi, \pi]}\left(\psi(k-1) + \omega_{\text{corr}} \Delta t\right)$$

### Covariance Propagation
$$\mathbf{P}_k^- = \mathbf{F}_k \mathbf{P}_{k-1} \mathbf{F}_k^T + \mathbf{Q}$$
where $\mathbf{F} = \frac{\partial f}{\partial \mathbf{x}}$ is the Jacobian matrix encoding how heading errors couple into position and velocity drift:
$$F[3, 6] = \frac{\partial \dot{v}_E}{\partial \psi} \Delta t = -(a_{\text{corr}} \sin\psi + v_{\text{fwd}} \omega_{\text{corr}} \cos\psi) \Delta t$$
$$F[4, 6] = \frac{\partial \dot{v}_N}{\partial \psi} \Delta t = (a_{\text{corr}} \cos\psi - v_{\text{fwd}} \omega_{\text{corr}} \sin\psi) \Delta t$$

---

## 4. Step 2: Measurement Updates

During GNSS blackout, two types of measurements are ingested to correct the state:

### 1. AI Forward Velocity Update (`update_velocity()`)
* **Measurement:** Forward speed from neural odometry ($z = v_{\text{ai}}$).
* **Measurement Function:** $h(\mathbf{x}) = v_E \cos\psi + v_N \sin\psi$.
* **Measurement Jacobian ($\mathbf{H}$):**
  $$\mathbf{H} = \begin{bmatrix} 0 & 0 & 0 & \cos\psi & \sin\psi & 0 & v_{\text{lat}} & 0 & 0 \end{bmatrix}$$
  where $v_{\text{lat}} = -v_E \sin\psi + v_N \cos\psi$.
* **Kalman Update Equations:**
  $$\text{Innovation: } y = z - h(\mathbf{x})$$
  $$\text{Innovation Covariance: } S = \mathbf{H} \mathbf{P} \mathbf{H}^T + R_{\text{speed}}$$
  $$\text{Kalman Gain: } \mathbf{K} = \mathbf{P} \mathbf{H}^T S^{-1}$$
  $$\text{State Correction: } \mathbf{x} = \mathbf{x} + \mathbf{K} y$$
  $$\text{Covariance Correction: } \mathbf{P} = (\mathbf{I} - \mathbf{K}\mathbf{H}) \mathbf{P} (\mathbf{I} - \mathbf{K}\mathbf{H})^T + \mathbf{K} R \mathbf{K}^T$$

### 2. Adaptive NHC Update (Described in Document 08)
Enforces zero lateral slip in the body frame ($v_{\text{lat}} \approx 0$).
