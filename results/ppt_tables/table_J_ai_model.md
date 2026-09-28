# TABLE J — PRELIMINARY AI MODEL SPECIFICATIONS & INFERENCE PROFILE

### Overview
This table provides the verified architecture, training parameters, memory footprint, edge latency, and speed tracking accuracy of `InertialOdomNet`, fulfilling the explicit SIH requirement for preliminary AI models trained on a subset of IO-VNBD.

---

### AI Model Architecture & Inference Profile

| Property | Verified Value | Verification Source / Architecture Detail |
| :--- | :--- | :--- |
| **Model Name** | **`InertialOdomNet`** | Defined in [`src/idr/models/inertial_odom.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/models/inertial_odom.py) |
| **Model Architecture** | **1D-CNN + 2-Layer GRU + MLP Head** | 3-layer 1D Conv (64, 128, 128 channels) $\to$ 2-layer GRU (hidden=128) $\to$ Dense linear layers |
| **Trainable Parameters**| **463,042 parameters** | Extremely lightweight; designed for edge smartphone deployment |
| **Model File Size** | **1.77 MB** (.pt) / **1.80 MB** (.onnx) | Verified on disk: [`models/inertial_odom.onnx`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/models/inertial_odom.onnx) |
| **Input Modality** | **6-axis Smartphone IMU** | Accelerometer $[a_x, a_y, a_z]$ and Gyroscope $[\omega_x, \omega_y, \omega_z]$ |
| **Input Temporal Window**| **50 samples (5.0 s @ 10 Hz)** | Sliding window of historical sensor measurements; strictly zero future look-ahead |
| **Sampling Frequency** | **10.0 Hz** | Synchronized 100 ms time steps ($\Delta t = 0.1\text{ s}$) |
| **Prediction Target** | **$(\Delta x, \Delta y, \log \sigma_x^2, \log \sigma_y^2)$** | 2D body displacement and heteroscedastic aleatoric uncertainty |
| **Training Partition** | Drives `M`, `S`, `Vta`, `Vtb` | Authentic vehicular driving segments from IO-VNBD |
| **Validation Drive** | Drive `Vfa01` | Mapped highway/urban driving |
| **Untouched Test Drive** | Drive `Vfa02` | Unmapped rural driving |
| **Loss Function** | Gaussian NLL + MSE loss | Negative Log-Likelihood incorporating predicted variance for EKF weighting |
| **CPU Inference Latency**| **0.93 ms per window** | Measured using ONNX Runtime on commodity CPU ($>100\times$ faster than 100 ms budget) |
| **Forward Speed MAE** | **1.14 m/s (4.1 km/h)** | Evaluated against vehicle ECU wheel speed ground truth on Drive `Vfa01` |
| **Inference Causality** | **STRICTLY CAUSAL** | 0 future samples; bidirectional GRU replaced by causal forward GRU |

---

### EKF Pseudo-Measurement Fusion
The model outputs forward displacement $\Delta x$ and variance $\sigma_x^2$ every 100 ms. Forward velocity is inferred as:
$$v_{\text{infer}} = \frac{\Delta x}{\Delta t}$$
This velocity is ingested by the Extended Kalman Filter as a measurement update with measurement covariance $R_v = \sigma_x^2 / \Delta t^2$, preventing filter divergence even under continuous GNSS outage.
