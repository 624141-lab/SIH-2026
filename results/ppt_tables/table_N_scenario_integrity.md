# TABLE N — SCENARIO INTEGRITY & OVERLAP AUDIT

### Overview
This table audits the 50 evaluated scenarios to establish whether they represent genuinely distinct evaluations, checking for repeated scenario IDs, duplicate windows, segment overlap, and output collisions.

---

### Scenario Integrity Verification Matrix

| Integrity Check | Verification Result | Forensic Audit Details & Evidence |
| :--- | :---: | :--- |
| **Total Scenario IDs in Benchmark** | **50 unique IDs** | Verified in `results/hardened/hardened_per_scenario_metrics.csv` |
| **Duplicate Scenario IDs** | **0 duplicates** | All 50 IDs are strictly unique strings |
| **Total Evaluation Runs** | **300 runs** | Exactly 50 scenarios evaluated across 6 navigation configurations |
| **Duplicate Input Segments** | **0 identical segments** | No two scenarios share identical `[start_idx, end_idx]` pairs |
| **Temporal Window Overlap** | **Documented Partial Overlap** | Scenarios use systematic temporal strides ($\Delta t_{\text{stride}} = 20\text{ s}$). Different duration tests (15 s, 30 s, 60 s) from the same drive share underlying route history (e.g., `t10s_d15s` is a sub-window of `t10s_d30s`), representing independent operational blackout lengths |
| **Reused Output Trajectories** | **0 collisions** | Every predicted trajectory array has a distinct numerical hash |
| **Drive Diversity** | **2 distinct drives** | Drive `Vfa01` (26 scenarios, urban/highway) and Drive `Vfa02` (24 scenarios, rural) |
| **Distance Diversity** | **175.5 m to 1,162.5 m** | Continuous distribution across short, medium, and long-range outages |
| **Independent Algorithmic State**| **100% Independent** | Estimator state is fully re-initialized at the start of each scenario; no state leakage between runs |

---

### Audit Conclusion
The 50 scenarios constitute a legitimate, mathematically diverse benchmark suite testing short (15 s), medium (30 s), and long (60 s) satellite outages across varied road curvatures, vehicle speeds, and map coverage conditions.
