# DO NOT CLAIM: UNSUPPORTED CLAIMS & HIGH-RISK STATEMENTS

This document lists claims, phrases, and metrics that **MUST BE AVOIDED** or **STRICTLY QUALIFIED** in the SIH 26168 proposal PPT. Overstating capabilities or presenting condition-specific numbers as universal truths will invite immediate evaluator skepticism and damage proposal credibility.

---

## 1. High-Risk Red-Flag Claims Matrix

| Prohibited Claim / Phrasing | Why It Is Dangerous / Unsupported | Verified Reality in Repository | Safe PPT Replacement Phrasing |
| :--- | :--- | :--- | :--- |
| **"Achieves <10% drift universally across all drives and conditions"** | False generalization. Drive `Vfa02` (untouched test set) experiences 44% to 75% median drift due to unmapped road network and open-loop MEMS gyro drift. | `<10%` drift is verified on evaluated highway/corridor blackout scenarios (e.g. `Vfa01_t70s_d60s` at $7.97\%$, `Vfa01_t45s_d60s` at $3.08\%$ and `Vfa01_t30s_d60s` at $2.58\%$). | *"Preliminary prototype achieves <10% drift (7.97% / 90.84 m over 1.14 km) on evaluated highway corridor scenarios in IO-VNBD."* |
| **"Sub-meter / Lane-level accuracy everywhere"** | Unachievable with consumer smartphone IMUs alone in unmapped or complex urban scenarios without RTK or high-resolution cameras. | Position error reaches tens of meters during extended outages without road constraints. | *"Sub-lane lateral confinement within verified OSM road corridors via causal map matching."* |
| **"Fully tested and deployed mobile Android application"** | No live Android APK or real-time JNI pipeline has been compiled or field-tested. The prototype is desktop Python with ONNX export. | `InertialOdomNet` has been exported to ONNX ($1.77\text{ MB}$, $0.93\text{ ms}$ latency), verifying edge computational feasibility. | *"Edge-optimized ONNX model architecture (<2 MB, <1 ms latency) architected specifically for real-time mobile deployment in Phase 2."* |
| **"200 Hz FOG hardware integration completed"** | No physical Fiber Optic Gyroscope hardware was connected or validated. | Only a modular sensor interface abstraction exists in software. | *"Modular sensor abstraction architecture designed to support external high-frequency IMU/FOG sensors."* |
| **"Works with arbitrary phone mounting and movement"** | Dynamic motion in a user's hand during sharp turns will corrupt the gravity vector without vision. | Phone alignment was calibrated under vehicle dashboard cradle mounting assumptions. | *"Automated static gravity and forward-acceleration alignment for rigid vehicle mounting (dashboard/cradle)."* |
| **"0.29% drift (3.02 m) across the board"** | This specific number was derived under an offline retrospective batch Viterbi traceback and pre-calibrated body frame alignment. | Under strictly causal online filtering, `Vfa01_t70s_d60s` achieves $7.97\%$ ($90.84\text{ m}$), `Vfa01_t45s_d60s` achieves $3.08\%$ ($35.71\text{ m}$), and `Vfa01_t30s_d60s` achieves $2.58\%$ ($30.02\text{ m}$). | Cite the verified causal numbers: **$7.97\%$ drift ($90.84\text{ m}$ error over $1,139.7\text{ m}$)**, **$3.08\%$ drift ($35.71\text{ m}$ over $1,159.2\text{ m}$)**, or **$2.58\%$ drift ($30.02\text{ m}$ over $1,162.5\text{ m}$)**. |
| **"AI alone replaces all inertial navigation principles"** | Pure AI odometry without kinematic constraints diverges rapidly ($>200\text{ m}$ cross-track error) under cornering slip. | AI odometry provides forward velocity aiding; EKF + NHC provides physical lateral constraint. | *"Hybrid AI-Kinematic fusion: Deep neural odometry synergized with Extended Kalman Filtering and Non-Holonomic Constraints."* |
| **"Zero gyro drift"** | Consumer MEMS gyros physically drift at $0.1^\circ - 0.5^\circ/\text{s}$, accumulating degrees of heading error over 60 seconds. | In unmapped areas (`Vfa02`), open-loop gyro drift is the dominant error source. | *"Adaptive covariance scaling and road-heading fusion mitigate MEMS gyro drift during mapped corridor blackouts."* |

---

## 2. Benchmark Presentation Rules

1. **Always State the Scenario Parameters**: Never state "3.08% drift" in isolation. Always state: *"3.08% drift (35.7 m final position error over 1,159.2 m traveled during a 60-second GNSS blackout at ~70 km/h on authentic IO-VNBD Drive Vfa01)"*.
2. **Disclose Baseline Comparisons**: Always contrast the proposed system against the Raw IMU baseline ($>300\%$ drift) to show that the prototype's algorithms are directly responsible for the accuracy.
3. **Frame Drive `Vfa02` as Future Work / Scientific Justification**: If evaluators ask about generalizability across all drives, do NOT pretend that Vfa02 passed with $<10\%$. Frankly explain that Drive `Vfa02` traverses rural roads outside the OSM map corridor, isolating the exact boundary where visual odometry (VIO) is required—which forms the core rationale for Phase 2 development.
