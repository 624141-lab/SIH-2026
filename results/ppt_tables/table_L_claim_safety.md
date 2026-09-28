# TABLE L — REQUIREMENT & CLAIM SAFETY AUDIT

### Overview
This table provides a strict classification of all technical statements, performance claims, and figures considered for the SIH 26168 screening presentation. Every claim must have verifiable code/data backing before appearing on a presentation slide.

---

### Claim Safety Classification Matrix

| Proposed PPT Statement / Claim | Evidence & Source File | Classification Status | Permitted in PPT? | Safe Phrasing / Mandatory Constraint |
| :--- | :--- | :---: | :---: | :--- |
| **"System achieves 7.97% drift over 1.14 km outage"** | [`scripts/generate_proposal_figures.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/generate_proposal_figures.py) & `fig1` | **VERIFIED** | **YES** | Safe as representative 1.14 km highway demonstration on Drive `Vfa01`. |
| **"InertialOdomNet runs in 0.93 ms on standard CPU"** | [`results/onnx_validation_report.json`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/results/onnx_validation_report.json) | **VERIFIED** | **YES** | Confirms sub-millisecond edge feasibility (>100x real-time). |
| **"AI odometry reduces mean baseline error by 79%"** | Baseline 3 (1,157 m) $\to$ System 5 (241 m) in Table D | **DERIVED** | **YES** | Valid mathematically from 50-scenario benchmark. |
| **"System achieves 0.29% drift (3.02 m) universally"** | Old retrospective Viterbi HMM script | **INVALID** | **NO (PURGED)** | Relied on future look-ahead; contradicted by causal execution. |
| **"System achieves 1.96% drift (20.36 m)"** | Scenario `t220s` from older intermediate alignment | **HISTORICAL** | **NO** | Replaced by canonical `Vfa01_t70s_d60s` (90.84 m / 7.97%). |
| **"SIH <10% drift satisfied universally in all drives"** | Full 50-scenario benchmark pass rate is 22.0% | **INVALID** | **NO** | Must state: *"Satisfied in 34.6% of mapped scenarios, reaching 2.58% drift."* |
| **"Real-time causal zero look-ahead operation"** | `CausalHMMMapMatcher(lag_steps=15, lookahead=0)` | **VERIFIED** | **YES** | Trellis propagation is causal with zero future data access. |
| **"Operates without vehicle OBD-II / CAN-bus wiring"** | Ingests purely smartphone 6-axis IMU CSV | **VERIFIED** | **YES** | True consumer smartphone application without vehicle wiring. |
| **"Fully robust across unmapped rural roads"** | Drive `Vfa02` exhibits 74.97% median drift | **CONDITION-SPECIFIC** | **QUALIFIED ONLY** | State that unmapped rural driving defines Phase 2 VIO roadmap. |
| **"Native Android app ready on Google Play Store"** | Preliminary PyTorch/ONNX algorithms only | **UNVERIFIED** | **NO** | Frame mobile app development as Phase 2 Grand Finale goal. |

---

### Prohibited Words & Guardrails
- **DO NOT USE:** *"Zero error"*, *"Universally <10% drift"*, *"Replaces GPS permanently"*, *"Fully solved on all roads"*.
- **MANDATORY DISCLOSURES:** Always state that results are demonstrated on authentic IO-VNBD smartphone data, and acknowledge that unmapped rural roads experience open-loop gyro drift, motivating the Phase 2 Visual-Inertial Odometry upgrade.
