# 11 — COMPLETE MATHEMATICAL MODEL: THE FULL DERIVATION

---

## 1. Sensor Measurement Equations

Let $\mathbf{a}_m \in \mathbb{R}^3$ and $\boldsymbol{\omega}_m \in \mathbb{R}^3$ be the raw specific force and angular velocity reported by the smartphone MEMS IMU:

$$\mathbf{a}_m(t) = \mathbf{R}_{pv}^T \left( \mathbf{a}_{\text{dyn}}(t) - \mathbf{g} \right) + \mathbf{b}_a(t) + \mathbf{n}_a(t)$$
$$\boldsymbol{\omega}_m(t) = \mathbf{R}_{pv}^T \boldsymbol{\omega}_{\text{true}}(t) + \mathbf{b}_\omega(t) + \mathbf{n}_\omega(t)$$

* $\mathbf{R}_{pv} \in SO(3)$: Phone-to-Vehicle 3D rotation matrix.
* $\mathbf{g} = [0, 0, -9.80665]^T\text{ m/s}^2$: Gravity vector in vehicle ENU frame.
* $\mathbf{b}_a, \mathbf{b}_\omega$: Sensor biases modeled as random walk processes:
  $$\dot{\mathbf{b}}_a(t) = \mathbf{w}_{ba}(t), \quad \dot{\mathbf{b}}_\omega(t) = \mathbf{w}_{b\omega}(t)$$
* $\mathbf{n}_a, \mathbf{n}_\omega$: Zero-mean white Gaussian measurement noise.

---

## 2. Phone-to-Vehicle Transformation & Gravity Separation

1. **Rotation to Vehicle Frame:**
   $$\mathbf{a}_v(t) = \mathbf{R}_{pv} \cdot \mathbf{a}_m(t)$$
   $$\boldsymbol{\omega}_v(t) = \mathbf{R}_{pv} \cdot \boldsymbol{\omega}_m(t)$$
2. **Exponential Low-Pass Gravity Separation:**
   $$\hat{\mathbf{g}}_v(k) = \alpha \cdot \hat{\mathbf{g}}_v(k-1) + (1 - \alpha) \cdot \mathbf{a}_v(k), \quad \alpha = 0.95$$
   $$\mathbf{a}_{\text{dyn}}(k) = \mathbf{a}_v(k) - \hat{\mathbf{g}}_v(k)$$
   The longitudinal forward acceleration is $a_{\text{fwd}} = \mathbf{a}_{\text{dyn}}[0]$, and yaw rate is $\omega_z = \boldsymbol{\omega}_v[2]$.

---

## 3. Neural Odometry Model (`InertialOdomNet`)

Given a 50-step historical IMU window $\mathbf{W}_k \in \mathbb{R}^{6 \times 50}$:
$$\mathbf{W}_k = [\mathbf{a}_m(k-49:k), \boldsymbol{\omega}_m(k-49:k)]^T$$

The neural network parameterizes a conditional Gaussian distribution:
$$f_\theta(\mathbf{W}_k) = \begin{bmatrix} \Delta x_k \\ \Delta y_k \\ \log \sigma_{x,k}^2 \\ \log \sigma_{y,k}^2 \end{bmatrix}$$

Forward velocity is extracted as:
$$v_{\text{ai}}(k) = \max\left(0, \frac{\Delta x_k}{50 \cdot \Delta t}\right), \quad \sigma_{\text{ai}}(k) = \exp\left(\frac{1}{2} \log \sigma_{x,k}^2\right)$$

---

## 4. 7-State Kinematic EKF Propagation

The state vector is $\mathbf{x} = [p_E, p_N, p_U, v_E, v_N, v_U, \psi, b_a, b_\omega]^T$.

### State Propagation:
$$p_E(k) = p_E(k-1) + v_E \Delta t + \frac{1}{2} a_E \Delta t^2$$
$$p_N(k) = p_N(k-1) + v_N \Delta t + \frac{1}{2} a_N \Delta t^2$$
$$v_E(k) = v_E(k-1) + a_E \Delta t$$
$$v_N(k) = v_N(k-1) + a_N \Delta t$$
$$\psi(k) = \psi(k-1) + (\omega_z - b_\omega) \Delta t$$

where:
$$v_{\text{fwd}} = v_E \cos\psi + v_N \sin\psi$$
$$a_E = (a_{\text{fwd}} - b_a) \cos\psi - v_{\text{fwd}} (\omega_z - b_\omega) \sin\psi$$
$$a_N = (a_{\text{fwd}} - b_a) \sin\psi + v_{\text{fwd}} (\omega_z - b_\omega) \cos\psi$$

### Jacobian Matrix ($\mathbf{F}_k = \partial f / \partial \mathbf{x}$):
$$\mathbf{F}_k = \mathbf{I}_9 + \Delta t \begin{bmatrix}
\mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} & \mathbf{0}_{3 \times 3} \\
\mathbf{0}_{2 \times 3} & \mathbf{0}_{2 \times 3} & \begin{matrix} -(a_{\text{corr}} \sin\psi + v_{\text{fwd}} \omega_{\text{corr}} \cos\psi) \\ (a_{\text{corr}} \cos\psi - v_{\text{fwd}} \omega_{\text{corr}} \sin\psi) \end{matrix} & \begin{matrix} -\cos\psi & 0 \\ -\sin\psi & 0 \end{matrix} \\
\mathbf{0}_{1 \times 3} & \mathbf{0}_{1 \times 3} & 0 & 0 & -1 \\
\mathbf{0}_{2 \times 3} & \mathbf{0}_{2 \times 3} & \mathbf{0}_{2 \times 1} & \mathbf{0}_{2 \times 2}
\end{bmatrix}$$

---

## 5. Dual Measurement Updates

### Update A: AI Speed Measurement
$$z_v = v_{\text{ai}}, \quad h_v(\mathbf{x}) = v_E \cos\psi + v_N \sin\psi$$
$$\mathbf{H}_v = \begin{bmatrix} 0 & 0 & 0 & \cos\psi & \sin\psi & 0 & (-v_E \sin\psi + v_N \cos\psi) & 0 & 0 \end{bmatrix}$$
$$R_v = \max(0.01, \sigma_{\text{ai}}^2 \cdot s_{\text{cov}})$$

### Update B: Adaptive Centripetal NHC Update
$$z_{\text{nhc}} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, \quad h_{\text{nhc}}(\mathbf{x}) = \begin{bmatrix} -v_E \sin\psi + v_N \cos\psi \\ v_U \end{bmatrix}$$
$$\mathbf{H}_{\text{nhc}} = \begin{bmatrix} 0 & 0 & 0 & -\sin\psi & \cos\psi & 0 & (-v_E \cos\psi - v_N \sin\psi) & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \end{bmatrix}$$
$$\mathbf{R}_{\text{nhc}} = \begin{bmatrix} \sigma_{\text{base}}^2 \cdot s_{\text{cov}} + k_{\text{turn}} (v_{\text{fwd}} \cdot |\omega_z|)^2 & 0 \\ 0 & \sigma_{\text{vert}}^2 \end{bmatrix}$$

---

## 6. Spatial Map Likelihood & Cosine Reacquisition

1. **Causal Map Posterior:**
   $$P(\text{road}_i | \mathbf{p}_k) \propto \exp\left(-\frac{\text{dist}(\mathbf{p}_k, \text{segment}_i)^2}{2 \sigma_z^2}\right) \cdot \max(0, \cos(\psi_k - \theta_i))$$
2. **Reacquisition Blending ($t \in [0, 3.5\text{ s}]$):**
   $$w(t) = \frac{1}{2} \left(1 - \cos\left(\pi \frac{t}{3.5}\right)\right)$$
   $$\mathbf{p}_{\text{final}}(t) = (1 - w(t)) \cdot \mathbf{p}_{\text{dr}}(t) + w(t) \cdot \mathbf{p}_{\text{gnss}}(t)$$
