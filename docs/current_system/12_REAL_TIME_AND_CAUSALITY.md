# 12 — REAL-TIME EXECUTION & CAUSALITY AUDIT

---

## 1. What Is Causality in Navigation?

In real-world driving, a navigation system operates in the present:
* It knows what happened in the past ($t \le t_{\text{now}}$).
* It receives sensor readings arriving right now ($t = t_{\text{now}}$).
* **It CANNOT know what will happen in the future ($t > t_{\text{now}}$).**

A system is **strictly causal** if its position estimate at time $t$ depends *only* on measurements up to time $t$. If an algorithm uses future frames or runs backward smoothing across a completed trip, it is **non-causal (cheating)** and cannot be deployed in a live vehicle.

---

## 2. Component-by-Component Causality Audit

| Component | Uses Past Data? | Uses Current Sample? | Uses Future Data? | Look-Ahead Horizon | Causal Verdict | Current Code Proof |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Attitude Aligner** | YES | YES | **NO** | $0.0\text{ seconds}$ | **CAUSAL** | Pre-drive static window + initial forward acceleration |
| **2. Vibration Filter** | YES (1.0 s window) | YES | **NO** | $0.0\text{ seconds}$ | **CAUSAL** | Causal moving average and rolling kurtosis (`vfilter.process()`) |
| **3. AI Odometry** | YES (5.0 s window) | YES | **NO** | $0.0\text{ seconds}$ | **CAUSAL** | 50 historical steps $[t-50:t]$; forward GRU without future tokens |
| **4. Kinematic EKF** | YES (Markovian $x_{k-1}$) | YES | **NO** | $0.0\text{ seconds}$ | **CAUSAL** | Standard Kalman recursive step $\mathbf{x}_k = f(\mathbf{x}_{k-1}, \mathbf{u}_k)$ |
| **5. Adaptive NHC** | NO | YES | **NO** | $0.0\text{ seconds}$ | **CAUSAL** | Instantaneous body lateral velocity zeroing |
| **6. Map Matching** | YES (Trellis history) | YES | **NO** | $0.0\text{ seconds}$ | **CAUSAL** | `use_finalized=False` outputs instantaneous `provisional_x, y` |
| **7. Reacquisition** | NO | YES | **NO** | $0.0\text{ seconds}$ | **CAUSAL** | Cosine blend initiates only after satellite signal re-locks |

---

## 3. The Sliding Window vs. Future Look-Ahead

Evaluators often confuse **sliding windows** with **future look-ahead**:

### What Our AI Model Does (Causal History Buffer):
```text
Time t=50:  Takes IMU samples [0, 1, 2, ..., 49, 50]  ──► Predicts speed at t=50
Time t=51:  Takes IMU samples [1, 2, 3, ..., 50, 51]  ──► Predicts speed at t=51
```
At every moment, the window looks **backward into the past**. It never touches sample $t+1$.

### What Non-Causal Cheating Would Look Like:
```text
Time t=50:  Takes IMU samples [25, 26, ..., 50, ..., 74, 75] ──► Uses 25 future samples!
```
Our codebase explicitly forbids centered or forward-looking windows. In [`scripts/evaluate_hardened_pipeline.py` line 144](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/evaluate_hardened_pipeline.py#L144):
```python
w_start = max(0, glob_i - 50)
win = phone_imu[w_start:glob_i].T  # Strictly backward-looking
```

---

## 4. Causal Execution vs. Offline Evaluation

It is vital to distinguish the **navigation algorithm** from the **evaluation metric logger**:

1. **Inside the Navigation Loop:**
   * Only receives `phone_imu` and offline OSM vector road maps.
   * Has zero access to `wheel_speed`, `heading`, `lat`, or `lon`.
2. **Outside the Navigation Loop (Evaluator):**
   * Takes the final predicted trajectory array $\mathbf{p}_{\text{est}}[0:N]$ and the RTK reference array $\mathbf{p}_{\text{gt}}[0:N]$.
   * Computes Euclidean error: $e(t) = \|\mathbf{p}_{\text{est}}(t) - \mathbf{p}_{\text{gt}}(t)\|$.
   * Computes drift percentage: $\text{Drift} = (e_{\text{final}} / \text{Distance}) \times 100\%$.

The navigation loop is 100% blind to ground truth. Ground truth is only read after the simulation ends to grade performance.
