# 03 — DATASET & DATA LOADING: IO-VNBD INGESTION

---

## 1. What is the IO-VNBD Dataset?

The **IO-VNBD** (Inertial-Odometry Vehicular Navigation Benchmark Dataset) is an authentic benchmark dataset recorded in moving road vehicles equipped with:
1. **Smartphone in Cradle:** Commercial Android devices (Samsung, Xiaomi) recording raw uncalibrated MEMS accelerometer, gyroscope, and magnetometer data.
2. **Reference Ground Truth Platform:** High-precision RTK GNSS (NovAtel / U-blox receiver) synchronized with the vehicle's onboard computer (CAN-bus wheel speeds and steering angles).

The dataset is partitioned into pairs of CSV files for each drive:
* `S-<drive_id>.csv`: Smartphone sensor stream.
* `V-<drive_id>.csv`: Vehicle ground truth stream.

---

## 2. Sensor Mapping & Field Isolation

To guarantee scientific authenticity, we strictly separate **navigation inputs** (sensors available on an ordinary phone) from **ground truth** (used solely by the evaluator to calculate error metrics).

### Data Field Breakdown

| Concept | File | Column Name | Unit | Category | Used During Blackout? | Description & Role |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Phone Acceleration** | `S-*.csv` | `acc_x`, `acc_y`, `acc_z` | $\text{m/s}^2$ | **Input** | **YES** | Raw 3-axis specific force in phone frame |
| **Phone Angular Rate** | `S-*.csv` | `gyro_x`, `gyro_y`, `gyro_z` | $\text{rad/s}$ | **Input** | **YES** | Raw 3-axis rotational velocity in phone frame |
| **Phone Magnetometer** | `S-*.csv` | `mag_x`, `mag_y`, `mag_z` | $\mu\text{T}$ | **Input** | **NO** | Not used in final pipeline due to in-cabin electromagnetic distortion |
| **Pre-Blackout GNSS Fix**| `S-*.csv` | `phone_lat`, `phone_lon` | $\text{deg}$ | **Input** | **NO (Pre-only)**| Used ONLY prior to blackout to set $p_0$ and initial heading |
| **Vehicle Wheel Speed** | `V-*.csv` | `wheel_speed` | $\text{m/s}$ | **Ground Truth** | **NO (STRICT ZERO)**| Vehicle ECU speed; strictly quarantined from navigation loop |
| **Vehicle Heading** | `V-*.csv` | `heading` | $\text{rad}$ | **Ground Truth** | **NO (STRICT ZERO)**| High-grade reference heading used only to score estimator error |
| **RTK Reference Latitude**| `V-*.csv` | `lat` | $\text{deg}$ | **Ground Truth** | **NO (STRICT ZERO)**| Used to generate local ENU coordinates $p_{gt}(t)$ for error scoring |
| **RTK Reference Longitude**|`V-*.csv` | `lon` | $\text{deg}$ | **Ground Truth** | **NO (STRICT ZERO)**| Used to generate local ENU coordinates $p_{gt}(t)$ for error scoring |

---

## 3. Data Loading & Synchronization Pipeline

The data ingestion engine is implemented in [`src/idr/io/loader.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/io/loader.py).

### Schema Detection (`detect_schema()`)
Because different Android devices log sensor columns under varying naming conventions (`acc_x` vs `accelerometer_x_mps2`), `detect_schema()` inspects the raw CSV header and maps diverse schemas into a standardized internal representation:
* Phone columns $\to$ `['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'phone_lat', 'phone_lon', ...]`
* Vehicle columns $\to$ `['timestamp', 'lat', 'lon', 'wheel_speed', 'heading', ...]`

### Temporal Grid Alignment (`load_drive_pair()`)
1. **Clock Discrepancies:** Smartphone IMUs and vehicle RTK systems run on separate hardware clocks with minor sampling jitter.
2. **Resampling to Uniform Grid:**
   The loader identifies the phone timestamp vector $t_s$ ($10\text{ Hz}$, $\Delta t = 0.1\text{ s}$).
   Vehicle numerical streams (RTK position, ECU speed) are interpolated onto $t_s$ using linear 1D interpolation:
   ```python
   # src/idr/io/loader.py lines 183-186
   v_aligned[col] = np.interp(t_s, t_v, vehicle_df[col].values)
   ```
3. **Array Packaging:** The synchronized drive is stored in an `IOVNBDrive` dataclass with identical row lengths, ensuring sample $k$ in the phone stream corresponds to ground truth timestamp $k$.

---

## 4. Local ENU Coordinate Conversion

Geodetic GPS coordinates (Latitude $\phi$, Longitude $\lambda$) must be converted into planar meters (East $e$, North $n$) for Cartesian Kalman filtering:

$$R_{\text{earth}} = 6,378,137.0\text{ meters}$$
$$e(t) = (\lambda(t) - \lambda_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}} \cdot \cos\left(\phi_0 \cdot \frac{\pi}{180}\right)$$
$$n(t) = (\phi(t) - \phi_0) \cdot \frac{\pi}{180} \cdot R_{\text{earth}}$$

where $(\phi_0, \lambda_0)$ is the reference coordinate at the start of the drive segment.
All navigation states and errors are subsequently evaluated in meters.
