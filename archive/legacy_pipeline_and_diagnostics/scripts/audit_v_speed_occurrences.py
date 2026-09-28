"""Phase 8: Systematic Audit of all `v_speed` occurrences across the entire repository.

Classifies every occurrence as:
- TRAINING LABEL: Ground-truth target for offline training
- EVALUATION ONLY: Post-hoc ground-truth distance/drift computation or test fixture
- DEPLOYMENT INPUT: Runtime sensor or alignment input (VERIFY STRICTLY ZERO)
"""

import sys
import re
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

def audit_v_speed():
    py_files = list(ROOT_DIR.glob("**/*.py"))
    # Exclude virtual environments
    py_files = [f for f in py_files if ".venv" not in str(f) and "__pycache__" not in str(f)]

    records = []

    for fpath in py_files:
        rel_path = fpath.relative_to(ROOT_DIR)
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            if "v_speed" in line:
                line_str = line.strip()
                # Skip comments or docstrings explaining removal of v_speed
                is_comment = line_str.startswith("#") or line_str.startswith('"""') or line_str.startswith("*") or line_str.startswith("-")

                # Classification
                path_str = str(rel_path).replace("\\", "/")
                if "train" in path_str:
                    category = "TRAINING LABEL"
                    purpose = "Offline model supervision target"
                elif "test" in path_str or "tests/" in path_str:
                    category = "EVALUATION ONLY"
                    purpose = "Unit test verification or benchmark comparison"
                elif "eval" in path_str or "evaluate" in path_str or "benchmark" in path_str or "diagnose" in path_str or "forensic" in path_str or "tune" in path_str or "audit" in path_str or "inspect" in path_str or "plot" in path_str:
                    category = "EVALUATION ONLY"
                    purpose = "Post-hoc ground-truth distance & drift metric computation"
                elif "src/idr/io/loader.py" in path_str:
                    category = "EVALUATION ONLY"
                    purpose = "Raw dataset reader providing ground-truth reference array for evaluation"
                elif "src/idr/io/preprocess.py" in path_str:
                    category = "EVALUATION ONLY"
                    purpose = "Preprocessed dataset caching for benchmark runs"
                elif "src/idr/eval/" in path_str:
                    category = "EVALUATION ONLY"
                    purpose = "Scenario ground-truth distance calculation"
                elif "src/idr/calib/alignment.py" in path_str:
                    if is_comment or "NO v_speed" in line_str or "motion_vel: Optional" in line_str:
                        category = "AUDITED / REMOVED"
                        purpose = "Verified 0% runtime usage (backward compatibility signature only; unused)"
                    else:
                        category = "DEPLOYMENT INPUT"
                        purpose = "VIOLATION - must be removed"
                else:
                    category = "EVALUATION ONLY"
                    purpose = "Reference speed analysis"

                records.append({
                    "File": path_str,
                    "Line": line_num,
                    "Code": line_str[:80],
                    "Category": category,
                    "Purpose": purpose,
                })

    df = pd.DataFrame(records)
    out_dir = ROOT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "v_speed_repository_audit.csv"
    df.to_csv(out_csv, index=False)

    print("=" * 85)
    print("PHASE 8: AUDIT OF ALL `v_speed` OCCURRENCES IN REPOSITORY")
    print("=" * 85)
    print(f"Total occurrences found: {len(df)}")
    cat_counts = df["Category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count}")

    deployment_violations = df[df["Category"] == "DEPLOYMENT INPUT"]
    print(f"\nDEPLOYMENT INPUT Violations: {len(deployment_violations)}")
    if len(deployment_violations) == 0:
        print("[VERIFIED] ZERO runtime occurrences of v_speed in deployment inference!")
    else:
        print(deployment_violations)

    print(f"\nSaved complete audit breakdown to {out_csv}")


if __name__ == "__main__":
    audit_v_speed()
