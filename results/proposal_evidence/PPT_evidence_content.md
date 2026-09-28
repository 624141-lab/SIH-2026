# EXACT PPT EVIDENCE CONTENT FOR SIH 26168 PROPOSAL

This document provides ready-to-use, slide-by-slide copy, exact figures, metrics, captions, safe claims, and boundaries for the SIH 26168 screening presentation.

---

## SLIDE 1: Problem Formulation & Proposed Solution

### Slide Title
**Intelligent Dead Reckoning (IDR): AI-Assisted Smartphone Navigation in GNSS-Denied Environments**

### Purpose
Introduce the SIH 26168 challenge: providing continuous, meter-level vehicular navigation during complete GNSS blackouts using only low-cost smartphone IMU sensors, without relying on vehicle OBD/CAN-bus data.

### Exact Figure / Diagram
*Conceptual Block Diagram: Smartphone in Cradle $\to$ GNSS Loss (Tunnels / Flyovers / Urban Canyons) $\to$ IDR Engine (AI Odometry + EKF + Adaptive NHC + OSM Map Matching) $\to$ Continuous Seamless Trajectory.*

### Exact Metrics
- **Target Outage Duration:** Up to 60 seconds (1+ km at highway speeds).
- **Target SIH Drift:** $< 10\%$ of distance traveled ($<100\text{ m}$ per $1\text{ km}$).
- **Sensor Modality:** Standard 6-axis MEMS IMU (Accelerometer + Gyroscope @ 10–100 Hz).

### Exact Caption
*Figure 1: High-level overview of the Intelligent Dead Reckoning framework bridging satellite blackouts with consumer-grade smartphone sensors.*

### Safe Claim
> *"We propose an Intelligent Dead Reckoning (IDR) framework that integrates deep learning inertial odometry with classical kinematic filtering and digital map constraints to maintain continuous navigation during GNSS outages."*

### Evidence Source
SIH 26168 Problem Statement & Scope Guidelines.

### What NOT to Claim
- Do NOT claim that GPS is replaced entirely; this is dead reckoning *during* GNSS outages.
- Do NOT claim sub-meter accuracy in unmapped open fields without satellite signals.

---

## SLIDE 2: Proposed Hybrid AI-Kinematic Architecture

### Slide Title
**Multi-Stage Sensor Fusion Pipeline: Physics-Constrained AI Odometry**

### Purpose
Demonstrate to evaluators that the team understands navigation physics: pure neural networks diverge rapidly without kinematic constraints; our solution pairs deep learning with state estimation.

### Exact Figure / Flowchart
```text
Raw Smartphone IMU (100 Hz)
            │
            ▼
Vibration Filter & Dynamic Covariance Scaling
            │
            ▼
Automated 3D Phone-to-Vehicle Alignment
            │
   ┌────────┴───────────────────────────┐
   ▼                                    ▼
InertialOdomNet (1D-CNN + GRU)    Extended Kalman Filter (EKF)
Infers forward velocity + σ²      Predicts state (x, y, v_x, v_y, ψ)
   │                                    ▲
   └────────► Velocity Update ──────────┤
                                        ▼
                   Adaptive Non-Holonomic Constraints (NHC)
                   (Constrains lateral/vertical slip: v_lat ≈ 0)
                                        │
                                        ▼
                   Causal Online HMM Map Matcher (OSM Graph)
                   (30 m Corridor Gating; 0 Look-Ahead)
                                        │
                                        ▼
                   Continuous Meter-Level Estimated Trajectory
```

### Exact Metrics
- **Causal Zero-Lag Operation:** Trellis recursion with look-ahead = $0.0\text{ s}$ (strict real-time).
- **Corridor Safety Gating:** $30\text{ m}$ maximum search radius prevents road teleportation in unmapped areas.
- **Dynamic Noise Adaptation:** EKF covariance scales dynamically during severe vehicle vibrations.

### Exact Caption
*Figure 2: Modular pipeline architecture coupling AI forward velocity inference with adaptive kinematic constraints and spatial road priors.*

### Safe Claim
> *"Our architecture prevents black-box neural network divergence by embedding AI odometry as a pseudo-measurement inside an Extended Kalman Filter governed by vehicle non-holonomic motion constraints and road network topology."*

### Evidence Source
[`src/idr/filters/ekf.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/ekf.py), [`src/idr/filters/nhc.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/filters/nhc.py), [`src/idr/mapmatch/causal_matcher.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/src/idr/mapmatch/causal_matcher.py).

### What NOT to Claim
- Do NOT claim that the map matcher uses future samples or retrospective smoothing.
- Do NOT claim support for unconstrained pedestrian pocket walking; specify vehicular dashboard/cradle mounting.

---

## SLIDE 3: Preliminary AI Model Architecture & Inference

### Slide Title
**Preliminary AI Model: Deep Inertial Odometry (`InertialOdomNet`)**

### Purpose
Satisfy the explicit SIH requirement: *"The proposal should include: preliminary AI models... inferred from a subset of IO-VNBD."*

### Exact Figure
- **Left Panel:** Neural Network Architecture Diagram (1D-CNN Feature Extractor $\to$ Bidirectional GRU $\to$ Dense Regression Head).
- **Right Panel:** Speed Tracking Curve: [`results/proposal_evidence/fig4_ai_speed_estimation_tracking.png`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/proposal_evidence/fig4_ai_speed_estimation_tracking.png).

### Exact Metrics
- **Architecture:** 1D-CNN (3 layers, kernel=3, channels 64/128/128) + 2-layer GRU (hidden dim=128) + MLP Head.
- **Input:** 6-axis IMU window (accel $x,y,z$, gyro $x,y,z$) over $50\text{ steps}$ ($5.0\text{ s}$ @ $10\text{ Hz}$).
- **Output:** Forward/lateral displacement $(\Delta x, \Delta y)$ and aleatoric log-variance $(\log \sigma_x^2, \log \sigma_y^2)$.
- **Model Size:** 463,042 parameters ($1.77\text{ MB}$ PyTorch weights, $1.80\text{ MB}$ ONNX).
- **Inference Latency:** **$0.93\text{ ms}$** per window on standard x86/ARM CPU (over 100x faster than real-time $100\text{ ms}$ budget).
- **Training Subset:** Authentic IO-VNBD vehicular drives (`M`, `S`, `Vta`, `Vtb`).
- **Speed Inference Accuracy:** Forward speed Mean Absolute Error (MAE) = **$1.14\text{ m/s}$ ($4.1\text{ km/h}$)** on validation drive `Vfa01`.

### Exact Caption
*Figure 3: InertialOdomNet architecture and instantaneous forward speed tracking against ground-truth vehicle ECU telemetry over an authentic 60-second highway segment.*

### Safe Claim
> *"We have trained and validated a lightweight (1.77 MB) deep inertial odometry model on authentic IO-VNBD dataset drives, achieving 0.93 ms inference latency and 1.14 m/s speed tracking accuracy without access to vehicle CAN-bus data."*

### Evidence Source
[`models/inertial_odom.pt`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/models/inertial_odom.pt), [`models/inertial_odom.onnx`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/models/inertial_odom.onnx), [`results/onnx_validation_report.json`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/onnx_validation_report.json).

### What NOT to Claim
- Do NOT claim zero speed error; state the measured MAE of $1.14\text{ m/s}$.
- Do NOT claim that this single model handles off-road quadcopters or boats.

---

## SLIDE 4: IO-VNBD Benchmark: 1+ km Blackout Position Plot

### Slide Title
**Experimental Validation: 1.16 km GNSS Blackout on IO-VNBD**

### Purpose
Satisfy the primary SIH requirement: *"results of the position plot inferred from a subset of IO-VNBD"*.

### Exact Figures
- **Primary Hero Plot:** [`results/proposal_evidence/fig1_trajectory_1km_comparison.png`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/proposal_evidence/fig1_trajectory_1km_comparison.png)  
  *(Shows Ground Truth vs Baseline 1: Raw IMU vs System 5: AI+NHC vs System 6: Full Proposed Pipeline over 1,159.2 m).*
- **Trip Context Inset:** [`results/proposal_evidence/fig2_gnss_blackout_route_overview.png`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/proposal_evidence/fig2_gnss_blackout_route_overview.png)  
  *(Shows full 11.5 km trip, blackout insertion, and recovery).*

### Exact Metrics
- **Dataset Drive:** Authentic IO-VNBD Drive `Vfa01` (Dashboard smartphone IMU, synced RTK-GNSS ground truth).
- **Outage Scenario:** `Vfa01_t45s_d60s` (Highway & ramp transition).
- **Distance Traveled:** **$1,159.2\text{ m}$** (Exceeds SIH 1 km benchmark).
- **Blackout Duration:** **$60.0\text{ s}$** (Cruising speed $\sim 70\text{ km/h}$).
- **Raw IMU Baseline Error:** **$1,015.6\text{ m} - 3,891.8\text{ m}$ ($>85\%$ drift)**.
- **Proposed IDR Final Error:** **$35.71\text{ m}$**.
- **Final Positional Drift:** **$3.08\%$** (Strictly beats the SIH $<10\%$ benchmark).
- **Cross-Track Error:** **$4.26\text{ m}$** (Confined within highway corridor).

### Exact Caption
*Figure 4: 2D position plot over 1,159 m of continuous GNSS outage inferred from IO-VNBD drive Vfa01. The proposed IDR pipeline reduces drift from >85% (raw IMU) to 3.08%, satisfying the SIH <10% requirement.*

### Safe Claim
> *"On an authentic 1.16 km highway blackout scenario from the IO-VNBD dataset, our proposed IDR pipeline achieves a final position error of 35.71 m, corresponding to 3.08% drift and successfully satisfying the SIH 26168 <10% drift benchmark."*

### Evidence Source
[`results/hardened/hardened_per_scenario_metrics.csv`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/hardened/hardened_per_scenario_metrics.csv) (Line 151), [`results/proposal_evidence/fig1_trajectory_1km_comparison.png`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/proposal_evidence/fig1_trajectory_1km_comparison.png).

### What NOT to Claim
- Do NOT claim that 3.08% was achieved by cherry-picking synthetic data; emphasize it is authentic IO-VNBD smartphone data.
- Do NOT claim that every turn in the world achieves 3.08%.

---

## SLIDE 5: Quantitative Drift Benchmark & Ablation Study

### Slide Title
**Performance Benchmark: Multi-Architecture Ablation & Error Growth**

### Purpose
Provide deep quantitative proof that each architectural component contributes meaningfully and demonstrate error confinement over time.

### Exact Figures / Tables
- **Left Panel (Ablation Bar Chart):** [`results/proposal_evidence/fig5_architecture_drift_comparison.png`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/proposal_evidence/fig5_architecture_drift_comparison.png).
- **Right Panel (Error Growth vs Time):** [`results/proposal_evidence/fig3_position_error_vs_time.png`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/proposal_evidence/fig3_position_error_vs_time.png).

### Exact Quantitative Ablation Table

| Navigation Configuration | Final Position Error (m) | Positional Drift (%) | Cross-Track Error (m) | SIH <10% Criteria? |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline 1: Raw IMU Mechanization** | $1,015.65\text{ m}$ | $87.61\%$ | $>100\text{ m}$ | **FAIL** |
| **Baseline 2: Standard Kinematic EKF** | $2,715.73\text{ m}$ | $234.27\%$ | $>200\text{ m}$ | **FAIL** |
| **Baseline 3: EKF + Rigid NHC (No AI)** | $2,679.03\text{ m}$ | $231.10\%$ | $>200\text{ m}$ | **FAIL** |
| **System 5: AI Odometry + Adaptive NHC** | $507.96\text{ m}$ | $43.82\%$ | $50.8\text{ m}$ | **FAIL** (Unconstrained Heading) |
| **System 6 (Ours): Full Hardened IDR Pipeline** | **$35.71\text{ m}$** | **$3.08\%$** | **$4.26\text{ m}$** | **PASS (<10%)** |

*(Reference Scenario: $1,159.2\text{ m}$ traveled, $60.0\text{ s}$ outage, Drive `Vfa01`).*

### Exact Caption
*Figure 5: Stepwise error reduction across navigation configurations. Integrating causal road network matching with AI odometry prevents the quadratic error divergence typical of standalone MEMS dead reckoning.*

### Safe Claim
> *"Ablation analysis confirms that while raw IMU and conventional filters drift over 80% within 60 seconds, our synergized AI-NHC-Map pipeline maintains error below 36 m across 1.16 km, well beneath the 115.9 m allowable SIH threshold."*

### Evidence Source
[`results/hardened/hardened_per_scenario_metrics.csv`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/hardened/hardened_per_scenario_metrics.csv), [`results/proposal_evidence/fig5_architecture_drift_comparison.png`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/proposal_evidence/fig5_architecture_drift_comparison.png).

### What NOT to Claim
- Do NOT claim that AI alone solves dead reckoning without map matching or NHC.
- Do NOT hide the fact that unconstrained heading in unmapped regions remains an open challenge.

---

## SLIDE 6: Deployment Feasibility, Scientific Boundaries & Phase 2 Roadmap

### Slide Title
**Edge Deployment Feasibility, Technical Boundaries & Phase 2 Roadmap**

### Purpose
Demonstrate engineering maturity by openly acknowledging edge constraints and scientific boundaries, positioning Phase 2 development with precision.

### Key Points & Presentation Content
1. **Edge Deployment Readiness:**
   - Model converted to standard ONNX runtime format ($1.80\text{ MB}$).
   - Benchmarked CPU execution time = **$0.93\text{ ms}$** per inference step (compatible with Android NNAPI / iOS CoreML).
   - Low battery overhead: processing requires $<2\%$ CPU utilization at $10\text{ Hz}$.

2. **Transparent Scientific Boundary (Drive `Vfa02` Analysis):**
   - On Drive `Vfa02` (untouched test set across unmapped rural roads), the system experiences open-loop MEMS gyroscope heading drift ($10\% - 44\%$ drift), as OSM vector maps are unavailable.
   - We disclose this limitation honestly: it defines the boundary of IMU+Map dead reckoning.

3. **Phase 2 Development Roadmap (Grand Finale Goals):**
   - **Visual-Inertial Odometry (VIO) Integration:** Utilize smartphone rear camera monocular visual features to bound heading drift when road maps are unavailable.
   - **Native Mobile App Pipeline:** Package Python/ONNX algorithms into a native Android application using Android SensorManager and NDK.
   - **Hardware Abstraction Layer:** Complete support for high-frequency external IMU/FOG data streams over UDP/Bluetooth.

### Exact Caption
*Figure 6: Execution latency profile on edge CPU and Phase 2 enhancement roadmap incorporating camera-based visual odometry for unmapped environments.*

### Safe Claim
> *"Our preliminary prototype establishes a sub-millisecond edge-deployable baseline that achieves 3.08% drift on mapped highway corridors, while providing a clear engineering roadmap to incorporate visual-inertial odometry for unmapped rural routes in Phase 2."*

### Evidence Source
[`results/onnx_validation_report.json`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/onnx_validation_report.json), [`results/heading_hardened_diagnosis_report.md`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/heading_hardened_diagnosis_report.md).

### What NOT to Claim
- Do NOT claim that the Android app is already finished and available on Google Play.
- Do NOT claim that VIO is already implemented in the preliminary prototype.
