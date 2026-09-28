# 20 — BEGINNER LEARNING PATH: 10-LEVEL MASTERY ROADMAP

---

## Overview
This roadmap is designed to take you from knowing nothing about inertial navigation to being able to explain, demonstrate, and defend the entire IDR codebase in front of SIH evaluators.

---

### Level 1: GNSS & Inertial Sensing (The Physics of Motion)
* **Goal:** Understand what an accelerometer and gyroscope actually measure.
* **Concepts to Learn:** Specific force, gravity ($9.81\text{ m/s}^2$), angular rate ($\text{rad/s}$), sample rate ($10\text{ Hz}$), sensor bias, and Gaussian noise.
* **Equation:** $\mathbf{a}_{\text{meas}} = \mathbf{a}_{\text{dyn}} - \mathbf{g} + \mathbf{b}_a + \mathbf{n}_a$.
* **Source File to Read:** [`src/idr/filters/vibration_filter.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/vibration_filter.py) lines 1–60.
* **Test Question:** *If a phone rests flat on a table, what does its accelerometer read along the vertical axis?* (Answer: $+9.81\text{ m/s}^2$).

---

### Level 2: Coordinate Reference Frames & Rotations
* **Goal:** Understand why a tilted phone in a cradle cannot be used directly.
* **Concepts to Learn:** Phone Frame ($\{X_p, Y_p, Z_p\}$), Vehicle Body Frame ($\{X_v, Y_v, Z_v\}$), and Local World ENU ($\{E, N, U\}$).
* **Equation:** $\mathbf{u}_v = \mathbf{R}_{pv} \cdot \mathbf{u}_p$ with $\mathbf{R}_{pv}^T \mathbf{R}_{pv} = \mathbf{I}$.
* **Source File to Read:** [`src/idr/calib/alignment.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/calib/alignment.py) lines 30–70 and 190–240.
* **Test Question:** *Why can't you calculate forward vehicle acceleration if you don't know the phone's pitch angle?* (Answer: Gravity will leak into the forward axis).

---

### Level 3: Dead Reckoning & Integration Explosion
* **Goal:** Understand why naive integration fails within 60 seconds.
* **Concepts to Learn:** First integration ($v = \int a \, dt$), second integration ($p = \iint a \, dt^2$), and quadratic error growth ($e = \frac{1}{2} b_a t^2$).
* **Equation:** $e_p(60) = \frac{1}{2} (0.1) (60)^2 = 180\text{ meters}$.
* **Document to Read:** [`docs/current_system/01_PROJECT_FROM_ZERO.md`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/docs/current_system/01_PROJECT_FROM_ZERO.md).
* **Test Question:** *If you have an accelerometer bias of $0.2\text{ m/s}^2$, what is your position error after 30 seconds?* (Answer: $\frac{1}{2} \times 0.2 \times 900 = 90\text{ meters}$).

---

### Level 4: The Kalman Filter Concept
* **Goal:** Master the balance between physical prediction and sensor measurements.
* **Concepts to Learn:** State vector ($\mathbf{x}$), state covariance ($\mathbf{P}$), process noise ($\mathbf{Q}$), measurement noise ($\mathbf{R}$), Kalman gain ($\mathbf{K}$), and innovation ($y = z - h(\mathbf{x})$).
* **Equation:** $\mathbf{K} = \mathbf{P}^- \mathbf{H}^T (\mathbf{H} \mathbf{P}^- \mathbf{H}^T + \mathbf{R})^{-1}$.
* **Source File to Read:** [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) lines 80–100.
* **Test Question:** *What happens to the Kalman gain $\mathbf{K}$ when measurement noise $\mathbf{R} \to \infty$?* (Answer: $\mathbf{K} \to 0$; the filter ignores the noisy measurement).

---

### Level 5: The 9-State Kinematic Extended Kalman Filter
* **Goal:** Understand the exact state vector and mechanization model used in this repository.
* **Concepts to Learn:** 9-state ENU vector, non-linear heading propagation, centripetal acceleration ($\boldsymbol{\omega} \times \mathbf{v}$), and process Jacobian ($\mathbf{F}$).
* **Equation:** $\mathbf{x} = [p_E, p_N, p_U, v_E, v_N, v_U, \psi, b_a, b_\omega]^T$.
* **Source File to Read:** [`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py) lines 37–81.
* **Test Question:** *Why does the state vector include $b_a$ and $b_\omega$?* (Answer: So the filter can continuously estimate and subtract sensor biases).

---

### Level 6: Vehicle Kinematics & Adaptive NHC
* **Goal:** Understand why non-holonomic constraints keep the car on track.
* **Concepts to Learn:** $v_{\text{lat}} \approx 0$, heading observability Jacobian ($\frac{\partial v_{\text{lat}}}{\partial \psi}$), centripetal variance inflation, and Chi-square gating ($NIS \le 9.21$).
* **Equation:** $\sigma_{\text{lat\_adapted}}^2 = \sigma_{\text{base}}^2 + k_{\text{turn}} (v_{\text{fwd}} \cdot |\omega_z|)^2$.
* **Source File to Read:** [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py) lines 100–168.
* **Test Question:** *Why does rigid NHC fail during high-speed cornering?* (Answer: Real tyre slip violates $v_{\text{lat}}=0$, distorting heading).

---

### Level 7: Deep Inertial Odometry (`InertialOdomNet`)
* **Goal:** Understand how AI infers vehicle speed without CAN-bus wiring.
* **Concepts to Learn:** 1D dilated convolutions, causal GRU, receptive field, 50-step window ($5.0\text{ s}$), and heteroscedastic uncertainty ($\log\sigma^2$).
* **Equation:** $v_{\text{ai}} = \max\left(0, \frac{\Delta x}{50 \cdot \Delta t}\right), \quad R_{\text{speed}} = \sigma_{\text{ai}}^2 \cdot s_{\text{cov}}$.
* **Source File to Read:** [`src/idr/models/inertial_odom.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/models/inertial_odom.py).
* **Test Question:** *Why does the AI predict velocity instead of position?* (Answer: Velocity errors do not compound over time; position errors compound quadratically).

---

### Level 8: Digital Road Networks & Causal HMM Map Matching
* **Goal:** Master real-time spatial projection onto OpenStreetMap.
* **Concepts to Learn:** NetworkX road graph, KD-Tree candidate search, emission probability, transition probability, $30\text{ m}$ corridor gate, and zero look-ahead causality.
* **Equation:** $\text{confidence} = 0.35 s_{\text{dist}} + 0.25 s_{\text{head}} + 0.25 s_{\text{trans}} + 0.15 s_{\text{cont}}$.
* **Source File to Read:** [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py) lines 255–425.
* **Test Question:** *What happens if the vehicle drives on an unmapped road farther than 30 meters from any known highway?* (Answer: Map snapping is rejected; raw EKF is output).

---

### Level 9: GNSS Outage Management & Reacquisition Smoothing
* **Goal:** Understand how the system handles satellite blackouts and restores guidance.
* **Concepts to Learn:** 4-state FSM (`NORMAL_GNSS`, `DEGRADED`, `GNSS_DENIED`, `RECOVERING`), HDOP threshold, and $C^1$ continuous cosine-bell blending.
* **Equation:** $w(t) = \frac{1}{2}\left(1 - \cos\left(\pi \frac{t}{3.5}\right)\right)$.
* **Source Files to Read:** [`src/idr/filters/gnss_monitor.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/gnss_monitor.py) and [`src/idr/eval/transition.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/eval/transition.py).
* **Test Question:** *Why shouldn't you instantly snap the position dot to the satellite coordinate when exiting a tunnel?* (Answer: It causes an instantaneous velocity spike and map UI jump).

---

### Level 10: Full Multi-Modal System Defense
* **Goal:** Confidently walk evaluators through the live code, figures, and benchmark tables.
* **Documents to Read:**
  1. [`docs/current_system/13_CURRENT_PIPELINE_CODE_TRACE.md`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/docs/current_system/13_CURRENT_PIPELINE_CODE_TRACE.md)
  2. [`docs/current_system/19_FAQ_FOR_SIH_EVALUATORS.md`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/docs/current_system/19_FAQ_FOR_SIH_EVALUATORS.md)
  3. [`results/ppt_tables/FINAL_PPT_TABLES.md`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/ppt_tables/FINAL_PPT_TABLES.md)
* **Final Milestone:** You can run [`scripts/generate_proposal_figures.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/generate_proposal_figures.py) and explain every single curve, number, and equation without looking at notes.
