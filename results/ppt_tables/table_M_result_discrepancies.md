# TABLE M — HISTORICAL RESULT DISCREPANCY AUDIT

### Overview
This table documents historical metrics from previous diagnostic scripts or markdown drafts that conflict with the current verified pipeline, detailing the exact technical reason for each discrepancy.

---

### Audit of Conflicting Historical Claims

| Historical Claim | Claimed Value | Current Verified Result | Discrepancy Cause & Forensic Audit Finding | Current Status |
| :--- | :---: | :---: | :--- | :---: |
| **"Hero Outage Drift"** | **3.02 m (0.29%)** | **90.84 m (7.97%)** | The 3.02 m claim was produced by an early offline script (`test_viterbi_backward.py`) using full-trajectory retrospective Viterbi smoothing with future map look-ahead. Contradicted by causal zero-lag execution. | **PURGED & REJECTED** |
| **"Scenario t220s Hero Drift"** | **20.36 m (1.96%)** | **44.66 m / 90.84 m** | Scenario `t220s` was an isolated curve segment evaluated with pre-fitted gyro bias parameters. The canonical SIH demonstration scenario is `Vfa01_t70s_d60s` (1,139.7 m). | **REPLACED BY CANONICAL RUN** |
| **"Scenario t45s Drift"** | **35.71 m (3.08%)** | **35.71 m (3.08%)** | Scenario `Vfa01_t45s_d60s` achieves 3.08% drift when map matching succeeds across multiple gentle highway bends. Fully reproducible in current codebase. | **VALID REFERENCE RUN** |
| **"Scenario t70s Evaluation: Script vs Figure"** | **44.66 m (3.92%)** | **90.84 m (7.97%)** | `evaluate_hardened_pipeline.py` uses dynamic pre-drive attitude alignment, yielding 44.66 m (3.92%). `generate_proposal_figures.py` sets identity alignment $R=\mathbf{I}$ (simulating uncalibrated cradle setup), yielding 90.84 m (7.97%). Both strictly pass SIH <10%. | **BOTH VERIFIED (CONSERVATIVE 7.97% USED)** |
| **"Universal <10% Claim"** | **100% Pass** | **22.0% Overall (34.6% Mapped)** | Early draft assumed mapped highway performance applied to unmapped rural routes. Empirical evaluation on unmapped `Vfa02` proved low-cost MEMS gyro drift accumulates without map priors. | **QUALIFIED & DISCLOSED** |

---

### Conclusion on Claim Integrity
By adopting the conservative **90.84 m (7.97% drift)** result for the primary proposal demonstration (Figure 1–5), the presentation remains completely defensible against hostile evaluator probing. Every number presented can be reproduced live by running [`scripts/generate_proposal_figures.py`](file:///c:/Users/varshith/Downloads/AI_gemini_pro/scripts/generate_proposal_figures.py).
