# 06 — AI ODOMETRY: `InertialOdomNet` DEEP SPEED INFERENCE

---

## 1. Why Do We Need Neural Odometry?

In classical dead reckoning, you integrate accelerometer readings to calculate velocity:
$$v(t) = \int a(t) \, dt$$
As proven in Document 01, sensor bias causes this integral to drift linearly, causing position to drift quadratically ($t^2$).

### The AI Breakthrough: Direct Speed Inference from Vibrations
When a car drives at $80\text{ km/h}$, the engine spins at $2,500\text{ RPM}$, the tyres vibrate across road texture, and the suspension oscillates at characteristic frequencies. When the car slows to $20\text{ km/h}$, those vibration patterns change completely.

A deep neural network can learn this mapping:
$$\text{IMU Vibration Pattern (5 seconds)} \xrightarrow{\text{InertialOdomNet}} \text{Forward Velocity } v_{\text{fwd}} \pm \sigma^2$$
Because the network directly predicts *velocity* rather than integrating acceleration, **bias does not accumulate over time**. If the network has an error at second 10, that error disappears at second 11 instead of compounding forever!

---

## 2. Neural Network Architecture Explained From Zero

The model is defined in [`src/idr/models/inertial_odom.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/models/inertial_odom.py) as `InertialOdomNet`.

```text
Input Tensor: (Batch, 6 channels, 50 timesteps)
       │
       ▼
[Stem: Conv1D (k=7, c=64) + BatchNorm + GELU]
       │
       ▼
[ResBlock 1: Conv1D (k=5, c=64, dilation=1) + Residual]
       │
       ▼
[ResBlock 2: Conv1D (k=5, c=128, dilation=2) + Residual]
       │
       ▼
[ResBlock 3: Conv1D (k=5, c=128, dilation=4) + Residual]
       │
       ▼
[Temporal Aggregator: Causal Forward GRU (hidden=128, num_layers=1)]
       │
       ▼
[Regression Head: Linear(128 -> 64) -> GELU -> Dropout(0.1) -> Linear(64 -> 4)]
       │
       ▼
Output Tensor: [dx_body, dy_body, log_var_x, log_var_y]
```

### Architectural Components Explained:

#### 1. 1D Temporal Convolution (Conv1D)
* **What it is:** A mathematical sliding filter (kernel) that slides along the time dimension of the IMU stream.
* **Why 1D instead of 2D:** Audio and IMU signals are 1D time-series, unlike 2D image grids.
* **Kernel Size ($k=5$):** Looks at 5 consecutive samples ($0.5\text{ seconds}$) simultaneously to detect high-frequency vibration rhythms.
* **Channels (64, 128):** Independent feature detectors (e.g. one channel might detect engine RPM peaks, another detects chassis sway).

#### 2. Dilated Residual Blocks (`ResBlock1D`)
* **Dilation:** Spacing between kernel elements.
  * Dilation=1: looks at samples $[t, t-1, t-2, t-3, t-4]$.
  * Dilation=4: looks at samples $[t, t-4, t-8, t-12, t-16]$.
* **Why Dilation Matters:** It expands the network's receptive field to see multiple seconds of history without increasing the number of weights or slowing down inference.
* **Skip Connection (Residual):** Adds the input directly to the output ($x + f(x)$), preventing vanishing gradients during backpropagation.

#### 3. Causal Gated Recurrent Unit (GRU)
* **What it is:** A recurrent neural network with internal memory gates that summarize temporal evolution.
* **Causal Constraint:** Unlike a bidirectional GRU that cheats by reading future frames, our GRU processes strictly forward in time (hidden state $h_t$ depends only on $h_{t-1}$ and current input $x_t$).

#### 4. Heteroscedastic Output Head
Instead of predicting just speed, the network outputs **4 values**:
* `dx_body`: Forward displacement along vehicle $X_v$ over the 5.0 s window (meters).
* `dy_body`: Lateral displacement along vehicle $Y_v$ over the 5.0 s window (meters).
* `log_var_x`: Predicted log-variance $\log(\sigma_x^2)$ representing uncertainty in forward speed.
* `log_var_y`: Predicted log-variance $\log(\sigma_y^2)$ representing uncertainty in lateral speed.

---

## 3. Loss Function: Negative Log-Likelihood (NLL)

The model is trained using Gaussian Negative Log-Likelihood (NLL) with an MSE regularizer:

$$\mathcal{L}_{\text{NLL}} = \frac{1}{2} \exp(-\log \sigma^2) \|\mathbf{y}_{\text{pred}} - \mathbf{y}_{\text{gt}}\|^2 + \frac{1}{2} \log \sigma^2$$

### Why Predict Uncertainty ($\log \sigma^2$)?
* If the car is driving smoothly on a straight highway, the network predicts low uncertainty ($\sigma^2 \approx 0.1$). The Kalman filter trusts the AI speed heavily.
* If the car is skidding on gravel or the driver is shaking the phone, the network outputs high uncertainty ($\sigma^2 \approx 5.0$). The Kalman filter automatically discounts the AI speed and trusts other sensors.

---

## 4. How AI Prediction Enters the Kalman Filter

In [`scripts/evaluate_hardened_pipeline.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py#L171-L175):

1. **Converting Window Displacement to Velocity:**
   The network predicts displacement $\Delta x$ over the 50-step ($5.0\text{ s}$) window:
   $$v_{\text{ai}} = \max\left(0.0, \frac{\Delta x}{50 \cdot \Delta t}\right)$$
2. **Converting Log-Variance to Measurement Standard Deviation:**
   $$\sigma_x = \exp(0.5 \cdot \text{log\_var}_x)$$
3. **Low-Pass Velocity Blending:**
   To eliminate high-frequency jitter, the raw AI speed is blended with inertial integration:
   $$v_{\text{tracking}} = 0.85 \cdot (v_{\text{tracking}} + a_{\text{fwd}} \cdot \Delta t) + 0.15 \cdot v_{\text{ai}}$$
4. **EKF Velocity Update:**
   ```python
   ekf.update_velocity(v_ai_tracking, R_speed = max(0.1, sig_x * np.sqrt(cov_scale)))
   ```
   This updates the filter velocity state $[v_e, v_n]$ along the current heading $\psi$.

---

## 5. Model Specs & Edge Performance

* **Parameters:** **463,042** (Lightweight; 1.77 MB PyTorch checkpoint, 1.80 MB ONNX).
* **Inference Latency:** **$0.93\text{ ms}$** per window on standard CPU ($>100\times$ faster than the $100\text{ ms}$ budget).
* **Speed Accuracy:** Forward speed Mean Absolute Error (MAE) = **$1.14\text{ m/s}$ ($4.1\text{ km/h}$)** on authentic IO-VNBD test drives.
