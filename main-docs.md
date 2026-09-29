# 01 — PROJECT FROM ZERO: INERTIAL NAVIGATION FUNDAMENTALS

---

## 1. Introduction: The Core Concepts

To defend this project, you need an intuitive understanding of navigation physics. Here are the core building blocks:

### 1. GNSS (Global Navigation Satellite System)
* **What it is:** A constellation of satellites (GPS, GLONASS, Galileo, NavIC) sending timestamped radio signals.
* **How it works:** Your phone calculates the distance to at least 4 satellites by measuring signal travel time, triangulating your latitude, longitude, and altitude.
* **Limitation:** Requires a direct line-of-sight to space. Blocked by concrete, rock, steel, and dense buildings.

### 2. IMU (Inertial Measurement Unit)
* **What it is:** A tiny microchip inside your phone containing Micro-Electro-Mechanical Systems (MEMS).
* **Two primary sensors:**
  1. **Accelerometer:** Measures linear acceleration ($m/s^2$) along 3 orthogonal axes ($X, Y, Z$). *Important:* It also continuously measures Earth's gravity ($9.81\text{ m/s}^2$).
  2. **Gyroscope:** Measures angular velocity / rate of rotation ($\text{rad/s}$ or $\text{deg/s}$) around 3 orthogonal axes.

### 3. Kinematic Quantities (From Acceleration to Position)
* **Acceleration ($a$):** How quickly velocity is changing (e.g. stepping on the gas pedal: $2.0\text{ m/s}^2$).
* **Velocity ($v$):** Speed with direction (e.g. traveling East at $20\text{ m/s} = 72\text{ km/h}$).
* **Displacement ($\Delta p$):** How far you moved during a specific time step $\Delta t$.
* **Position ($p$):** Where you are in the world (e.g., East = $450\text{ m}$, North = $1,200\text{ m}$).
* **Heading / Yaw ($\psi$):** The compass direction the vehicle's nose is pointing (e.g., $0^\circ$ = East, $90^\circ$ = North).

---

## 2. Dead Reckoning: Navigating Without Satellites

### The Concept
Imagine walking in a completely pitch-black room with your eyes closed. If you know:
1. Where you started ($p_0 = [0, 0]$),
2. How fast you are walking ($1.5\text{ m/s}$), and
3. What direction your nose is pointing ($45^\circ$ North-East),

you can calculate where you are after 10 seconds:
$$\text{Distance} = 1.5\text{ m/s} \times 10\text{ s} = 15\text{ meters}$$
This process of estimating current position based on past position, speed, and heading is called **Dead Reckoning (DR)**.

---

## 3. Why Does Classical Dead Reckoning Fail on Smartphones?

In pure physics, you integrate acceleration to get velocity, and integrate velocity to get position:

$$v(t) = v(0) + \int_{0}^{t} a(\tau) \, d\tau$$
$$p(t) = p(0) + \int_{0}^{t} v(\tau) \, d\tau$$

### The Enemy: Sensor Bias and Quadratic Error Growth
Smartphone MEMS sensors suffer from **bias** ($b$), an internal calibration error that makes the sensor output a non-zero value even when completely stationary.

Suppose the accelerometer has a tiny bias $b_a = 0.1\text{ m/s}^2$ (roughly $1\%$ of gravity). If the vehicle is moving at a constant speed, the true acceleration is $a_{\text{true}} = 0$, but the sensor reports $a_{\text{meas}} = 0.1\text{ m/s}^2$.

Let's integrate that error over time:
1. **Velocity Error:**
   $$e_v(t) = \int_{0}^{t} b_a \, d\tau = b_a \cdot t$$
   After 60 seconds: $e_v = 0.1 \times 60 = 6.0\text{ m/s}$ ($21.6\text{ km/h}$ error).
2. **Position Error (Double Integration):**
   $$e_p(t) = \int_{0}^{t} e_v(\tau) \, d\tau = \frac{1}{2} b_a \cdot t^2$$
   After 60 seconds:
   $$e_p(60) = \frac{1}{2} \times 0.1 \times (60)^2 = \frac{1}{2} \times 0.1 \times 3600 = \mathbf{180\text{ meters!}}$$

Even with a tiny $0.1\text{ m/s}^2$ error, position drifts by $180\text{ meters}$ in just one minute. On real consumer phones with vibration and temperature shifts, bias can exceed $0.3\text{ m/s}^2$, generating over **$500\text{ meters}$ of drift**.

### Gyroscope Drift: Rotating the Vector into Ruin
The gyroscope has a bias too ($b_\omega \approx 0.2^\circ/\text{s}$). If the heading estimate drifts by just $10^\circ$:
* The vehicle is driving straight forward.
* The computer thinks the vehicle is driving $10^\circ$ off to the right.
* At $72\text{ km/h}$ ($20\text{ m/s}$), the sideways cross-track error grows by:
  $$\Delta y = 20\text{ m/s} \times \sin(10^\circ) \times 60\text{ s} \approx 208\text{ meters!}$$

This is why **Baseline 1 (Raw IMU)** and **Baseline 2 (Standard EKF)** fail with $98\%\text{--}146\%$ drift in our benchmark.

---

## 4. Coordinate Reference Frames

Navigation is impossible unless you clearly define which coordinate system you are talking about. Our system uses three distinct frames:

```text
       Phone Frame (p)                 Vehicle Body Frame (v)               World ENU Frame (w)
      [Tilting in cradle]              [Fixed to car chassis]              [Tangent plane on Earth]
            +Z_p                                 +Z_v (Up)                              +North
             ▲                                    ▲                                       ▲
             │                                    │                                       │
             │   ▲ +Y_p                           │   ▲ +X_v (Forward)                    │
             │  /                                 │  /                                    │
             │ /                                  │ /                                     └────────► +East
   ──────────┼────────► +X_p            ──────────┼────────► +Y_v (Right)                /
            /                                    /                                      ▼ +Up
```

### 1. Phone Frame ($p$): $\{X_p, Y_p, Z_p\}$
* Attached to the physical smartphone hardware.
* When a driver clips the phone into a magnetic air vent mount, the phone screen might face slightly up and left.
* Measurements from the accelerometer and gyroscope come out in this arbitrary phone frame.

### 2. Vehicle Body Frame ($v$): $\{X_v, Y_v, Z_v\}$
* Fixed rigidly to the automobile chassis:
  * **$X_v$ (Longitudinal):** Points forward through the front windshield.
  * **$Y_v$ (Lateral):** Points out the passenger window (to the right).
  * **$Z_v$ (Vertical):** Points vertically through the car roof.
* Cars can only accelerate forward/backward along $X_v$. They cannot slide sideways along $Y_v$ without skidding.

### 3. World Navigation Frame ($w$): Local ENU (East-North-Up)
* Fixed to the surface of the Earth at the location where the GNSS blackout started.
  * **East ($X_w$):** Tangent to Earth pointing East.
  * **North ($Y_w$):** Tangent to Earth pointing North.
  * **Up ($Z_w$):** Normal to Earth's ellipsoid pointing into the sky.
* We express estimated vehicle position $(p_E, p_N)$ in meters relative to the start of the trip.

---

## 5. Summary: What the Entire Project Does in Plain English

1. Take messy IMU signals from a crooked phone in a vibrating car.
2. Rotate them so the computer knows which way is "car forward" and which way is "Earth down."
3. Filter out engine buzz, potholes, and music rumble.
4. Use an AI model to guess forward speed from vibrations so we never have to double-integrate acceleration.
5. Feed that speed into a Kalman filter that enforces real car physics (wheels don't slide sideways).
6. Snap the estimated route to the nearest real street on OpenStreetMap.
7. Smoothly merge back onto satellites when the sky opens up.
