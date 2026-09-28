# 05 — PHONE-TO-VEHICLE ALIGNMENT: ATTITUDE RECONSTRUCTION

---

## 1. Why Is Alignment Required?

When a driver places their phone in a dashboard cradle, the phone's internal axes do NOT line up with the car's axes:
* The phone might be tilted backwards at a $30^\circ$ pitch angle.
* It might be angled slightly toward the driver with a $15^\circ$ yaw angle.
* It might be tilted sideways in landscape or portrait orientation.

If you don't calibrate this alignment:
1. Earth's gravity ($9.81\text{ m/s}^2$) will leak into the forward acceleration axis, making the computer think the car is continuously accelerating at rocket speeds.
2. A sharp turn around the vehicle's vertical axis will be measured as a mix of roll and pitch, completely distorting the heading estimate.

---

## 2. Mathematical Definition

We need to find a 3D rotation matrix $\mathbf{R}_{pv} \in \mathbb{R}^{3 \times 3}$ (Phone-to-Vehicle) such that any vector measured in the phone frame ($\mathbf{u}_p$) is transformed into the vehicle frame ($\mathbf{u}_v$):

$$\mathbf{u}_v = \mathbf{R}_{pv} \cdot \mathbf{u}_p$$

where:
* $\mathbf{u}_v = [u_{\text{fwd}}, u_{\text{lat}}, u_{\text{up}}]^T$
* $\mathbf{R}_{pv}$ is an orthonormal rotation matrix ($\mathbf{R}_{pv}^T \mathbf{R}_{pv} = \mathbf{I}$, $\det(\mathbf{R}_{pv}) = +1$).

---

## 3. How the Current Implementation Solves Alignment

The alignment engine is implemented in [`src/idr/calib/alignment.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/calib/alignment.py) via the `PhoneToVehicleAligner` class.

Alignment is solved in **two physical phases**:
1. **Vertical Leveling (Roll & Pitch):** Solved while the vehicle is stationary using Earth's gravity vector.
2. **Forward Heading (Yaw):** Solved during initial forward acceleration using motion dynamics or pre-blackout GNSS track.

---

### Phase 1: Stationary Gravity Vector Estimation (Leveling)

#### 1. Automatic Stationary Detection
The function `detect_stationary_windows()` monitors accelerometer and gyroscope signals to find an interval where the vehicle is stopped:
* $|\mathbf{a}_{\text{mag}} - 9.81| < 0.45\text{ m/s}^2$ (Specific force magnitude equals gravity).
* $\|\boldsymbol{\omega}\| < 0.06\text{ rad/s}$ ($<3.4^\circ/\text{s}$, no rotation).
* Rolling variance $\sigma_a^2 < 0.04\text{ m}^2/\text{s}^4$.
* Duration $\ge 1.5\text{ seconds}$ (15 continuous steps @ 10 Hz).

#### 2. Deriving Roll and Pitch via Rodrigues Rotation
While stationary, the only force acting on the phone is the upward normal force balancing gravity:
$$\mathbf{z}_p = \frac{\bar{\mathbf{a}}_{\text{stat}}}{\|\bar{\mathbf{a}}_{\text{stat}}\|}$$
where $\mathbf{z}_p$ is the unit vector pointing straight "up" in the phone's coordinate system.

To align $\mathbf{z}_p$ with the vehicle's vertical axis $\mathbf{z}_v = [0, 0, 1]^T$:
1. Calculate the rotation axis: $\mathbf{k} = \mathbf{z}_p \times \mathbf{z}_v$.
2. Calculate the tilt angle: $\theta = \arccos(\mathbf{z}_p \cdot \mathbf{z}_v)$.
3. Construct the leveling rotation matrix $\mathbf{R}_{\text{level}}$ using Rodrigues' formula:
   $$\mathbf{R}_{\text{level}} = \mathbf{I} + \sin\theta \cdot \mathbf{K} + (1 - \cos\theta) \cdot \mathbf{K}^2$$
   where $\mathbf{K}$ is the skew-symmetric cross-product matrix of the normalized axis $\mathbf{k}$.

*Source Code:* [`src/idr/calib/alignment.py` lines 215–231](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/calib/alignment.py#L215-L231).

---

### Phase 2: Forward Longitudinal Axis Estimation (Yaw)

Once leveled, the phone's horizontal plane matches the vehicle's horizontal plane, but we still do not know which direction points toward the front windshield (Yaw $\psi_{pv}$).

#### Yaw Observability Principle
* When the vehicle is stopped at a red light, **yaw is physically unobservable** (gravity provides zero horizontal information). The aligner reports `ALIGNMENT_UNCERTAIN`.
* As soon as the vehicle accelerates forward, Newton's second law creates a distinct acceleration vector along the vehicle's longitudinal axis $X_v$.

#### Solving Yaw via Dynamic Acceleration
In `update_forward_motion()`:
1. Acceleration is leveled onto the horizontal plane: $\mathbf{a}_{\text{horiz}} = \mathbf{R}_{\text{level}} \cdot \mathbf{a}_p$.
2. When forward acceleration occurs ($|\dot{v}| > 0.25\text{ m/s}^2$ from pre-blackout GNSS velocity):
   $$\mathbf{x}_{\text{dir}} = \sum_{k} \mathbf{a}_{\text{horiz}}(k) \cdot a_{\text{fwd}}(k)$$
3. The forward unit vector is normalized: $\mathbf{x}_v = \mathbf{x}_{\text{dir}} / \|\mathbf{x}_{\text{dir}}\|$.
4. The lateral axis is orthogonalized: $\mathbf{y}_v = \mathbf{z}_v \times \mathbf{x}_v$.
5. The full 3D rotation matrix is assembled:
   $$\mathbf{R}_{pv} = \begin{bmatrix} \mathbf{x}_v^T \\ \mathbf{y}_v^T \\ \mathbf{z}_v^T \end{bmatrix}$$

*Status:* The aligner transitions to `ALIGNED`, and all subsequent IMU samples are rotated using `transform_imu()`.

---

## 4. What Can Go Wrong?

1. **Uncalibrated Cradle Setup:** If the vehicle starts directly in a blackout without pre-drive motion or pre-blackout GNSS track, yaw cannot be resolved, forcing the pipeline to fall back to identity alignment ($\mathbf{R}=\mathbf{I}$).
2. **Phone Dislodged Mid-Drive:** If the phone falls out of its cradle, the rotation matrix becomes invalid. (Mitigated in Phase 2 via continuous gravity tracking).
