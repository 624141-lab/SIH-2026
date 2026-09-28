# 08 — ADAPTIVE NHC: NON-HOLONOMIC CONSTRAINTS

---

## 1. What Are Non-Holonomic Constraints (NHC)?

In classical robotics and mechanics, a **holonomic** robot (like an omnidirectional warehouse robot or a drone) can move in any direction at any moment: forward, backward, left, right, or up.

A standard road vehicle is **non-holonomic**:
* Its wheels roll forward and backward.
* Under normal driving conditions, the rubber tyres have high lateral friction against asphalt. The car **cannot slide sideways like an air-hockey puck**.
* The car is bound to the road plane; it **cannot fly vertically into the air**.

Therefore, in the vehicle's body coordinate frame:
$$v_{\text{lateral}} \approx 0\text{ m/s}$$
$$v_{\text{vertical}} \approx 0\text{ m/s}$$

---

## 2. Why Is NHC the "Magic Bullet" for Dead Reckoning?

During a GNSS blackout, the computer has no satellites telling it where it is or which way it is heading.

If the gyroscope suffers from a small bias, the estimated heading will drift. But notice what happens mathematically:
* The car is driving East ($v_E = 20\text{ m/s}, v_N = 0$).
* If the filter heading drifts slightly by $\Delta \psi$, the filter calculates a non-zero lateral velocity:
  $$v_{\text{lat}} = -v_E \sin\psi + v_N \cos\psi \ne 0$$
* Because physics dictates $v_{\text{lat}} = 0$, the filter perceives this non-zero value as an "error" (innovation).
* **The Crucial Insight:** By correcting this sideways error, the Kalman filter directly corrects the **heading state $\psi$**!

### The Mathematical Observability Proof
In [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py#L69-L70), look at the measurement Jacobian $H$:
$$H[0, 6] = \frac{\partial v_{\text{lat}}}{\partial \psi} = -\cos\psi \cdot v_E - \sin\psi \cdot v_N$$
Because $H[0, 6]$ is proportional to forward vehicle speed, **whenever the vehicle is moving forward, the lateral velocity constraint directly observes and bounds heading drift**!

---

## 3. The Catch: Why Rigid NHC Fails During Cornering

If NHC is so powerful, why not enforce $v_{\text{lat}} = 0$ strictly with zero tolerance?

### The Physics of Tyre Slip
When a car takes a highway exit ramp at $70\text{ km/h}$, tyres deform under massive centripetal force ($a_c = v \cdot \omega_z$). The car experiences a physical **slip angle** ($\beta = 2^\circ\text{--}5^\circ$).
* If the filter rigidly insists $v_{\text{lat}} = 0$ with high certainty ($\sigma_{\text{lat}} = 0.01\text{ m/s}$), the real-world cornering forces violate the assumption.
* The Kalman filter panics, assuming the gyroscope must be completely broken, and distorts the vehicle heading!
* This is why **Baseline 3 (Rigid NHC)** performs poorly on curved trajectories.

---

## 4. The Solution: Centripetal-Aware Adaptive NHC

Our implementation in [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py#L100-L168) dynamically relaxes the lateral constraint during cornering using vehicle dynamics physics:

$$\sigma_{\text{lat\_adapted}}^2 = \sigma_{\text{base}}^2 + k_{\text{turn}} \cdot a_{\text{centripetal}}^2$$

where:
* $a_{\text{centripetal}} = |v_{\text{fwd}} \cdot \omega_z|$ (Centripetal acceleration in $\text{m/s}^2$).
* $\sigma_{\text{base}} = 0.05\text{ m/s}$ (Baseline lateral uncertainty on straight roads).
* $k_{\text{turn}} = 0.5$ (Cornering inflation scale factor).

### Physical Behavior:
1. **On Straight Corridors ($\omega_z \approx 0$):**
   $a_c \approx 0 \implies \sigma_{\text{lat}} \approx 0.05\text{ m/s}$. The filter tightly constrains heading drift to zero.
2. **During High-Speed Curves ($v = 20\text{ m/s}, \omega_z = 0.2\text{ rad/s} \implies a_c = 4.0\text{ m/s}^2$):**
   $$\sigma_{\text{lat\_adapted}}^2 = (0.05)^2 + 0.5 \cdot (4.0)^2 = 0.0025 + 8.0 = 8.0025$$
   $$\sigma_{\text{lat\_adapted}} \approx 2.83\text{ m/s}$$
   The filter recognizes that tyre slip is occurring and automatically inflates its noise threshold, allowing the vehicle to corner naturally without corrupting the heading estimate!

---

## 5. Mahalanobis Chi-Square Innovation Gating

What happens if the vehicle hits a patch of black ice, hydroplanes, or swerves violently?
Even the adaptive variance might not be enough. To protect the filter, we compute the **Normalized Innovation Squared (NIS)**:

$$\text{NIS} = \mathbf{y}^T \mathbf{S}^{-1} \mathbf{y}$$

where $\mathbf{y} = [-v_{\text{lat\_pred}}, -v_{\text{vert\_pred}}]^T$ and $\mathbf{S} = \mathbf{H} \mathbf{P} \mathbf{H}^T + \mathbf{R}$.
* In 2 degrees of freedom, the 99% confidence threshold of a Chi-square distribution is $\chi_{2, 0.99}^2 = 9.21$.
* **Rejection Rule:**
  ```python
  # src/idr/filters/nhc.py line 155
  if nis > 9.21:
      return False, v_lat_before, v_lat_before, sigma_lat_adapted
  ```
  If $\text{NIS} > 9.21$, the measurement is discarded as an unmodeled dynamic anomaly, preventing filter corruption.
