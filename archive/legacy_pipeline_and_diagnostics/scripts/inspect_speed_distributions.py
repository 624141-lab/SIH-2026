"""Phase 3 Inspection: Speed distribution of training drives (M, S, Vta, Vtb) vs Vfa01 vs Vfa02.

Categorizes samples into speed bins:
0-5 m/s, 5-10 m/s, 10-15 m/s, 15-20 m/s, 20-25 m/s, 25-30+ m/s.
Outputs table and summary CSV.
"""

import sys
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from src.idr.io.loader import load_drive_pair

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "results" / "tuning"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def analyze_speed_distributions():
    logger.info("Collecting speed data across training drives, Vfa01, and Vfa02...")

    # Training drive identifiers
    train_ids = [
        "M",
        "S1", "S2", "S3a", "S3b", "S3c", "S4",
        "Vta1a", "Vta1b", "Vta2", "Vta3", "Vta4", "Vta5", "Vta6", "Vta7", "Vta8", "Vta9", "Vta10",
        "Vta11", "Vta12", "Vta13", "Vta14", "Vta15", "Vta16", "Vta17", "Vta19", "Vta20",
        "Vta21", "Vta22", "Vta23", "Vta24", "Vta25", "Vta26", "Vta27", "Vta28", "Vta29", "Vta30",
        "Vtb1", "Vtb2", "Vtb3", "Vtb4", "Vtb5", "Vtb6", "Vtb7", "Vtb8", "Vtb9", "Vtb10", "Vtb11", "Vtb12",
    ]

    train_speeds = []
    loaded_train_count = 0

    for d_id in train_ids:
        try:
            drive = load_drive_pair(DATA_DIR, d_id)
            _, _, v_spd, _ = drive.get_synced_data()
            train_speeds.append(v_spd)
            loaded_train_count += 1
        except Exception as e:
            logger.warning(f"Could not load train drive {d_id}: {e}")

    all_train_speeds = np.concatenate(train_speeds) if train_speeds else np.array([])
    logger.info(f"Loaded {loaded_train_count} training drives: {len(all_train_speeds):,} samples.")

    # Validation drive Vfa01
    drive_vfa01 = load_drive_pair(DATA_DIR, "Vfa01")
    _, _, vfa01_speeds, _ = drive_vfa01.get_synced_data()

    # Test drive Vfa02 (strictly read-only distribution inspection)
    drive_vfa02 = load_drive_pair(DATA_DIR, "Vfa02")
    _, _, vfa02_speeds, _ = drive_vfa02.get_synced_data()

    bins = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 1000.0]
    bin_labels = [
        "0–5 m/s",
        "5–10 m/s",
        "10–15 m/s",
        "15–20 m/s",
        "20–25 m/s",
        "25–30+ m/s",
    ]

    rows = []
    for i in range(len(bin_labels)):
        low = bins[i]
        high = bins[i + 1]

        if i == len(bin_labels) - 1:
            tr_count = int(np.sum(all_train_speeds >= low))
            v1_count = int(np.sum(vfa01_speeds >= low))
            v2_count = int(np.sum(vfa02_speeds >= low))
        else:
            tr_count = int(np.sum((all_train_speeds >= low) & (all_train_speeds < high)))
            v1_count = int(np.sum((vfa01_speeds >= low) & (vfa01_speeds < high)))
            v2_count = int(np.sum((vfa02_speeds >= low) & (vfa02_speeds < high)))

        tr_pct = (tr_count / len(all_train_speeds)) * 100.0 if len(all_train_speeds) > 0 else 0.0
        v1_pct = (v1_count / len(vfa01_speeds)) * 100.0
        v2_pct = (v2_count / len(vfa02_speeds)) * 100.0

        rows.append({
            "Speed Range": bin_labels[i],
            "Training Samples": f"{tr_count:,} ({tr_pct:.1f}%)",
            "Vfa01 Samples": f"{v1_count:,} ({v1_pct:.1f}%)",
            "Vfa02 Samples": f"{v2_count:,} ({v2_pct:.1f}%)",
            "tr_count": tr_count,
            "v1_count": v1_count,
            "v2_count": v2_count,
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "speed_distribution_breakdown.csv"
    df.to_csv(out_csv, index=False)
    logger.info(f"[+] Saved speed distribution table to {out_csv}")

    print("\n" + "=" * 80)
    print("SPEED DISTRIBUTION COMPARISON: TRAIN (M, S, Vta, Vtb) vs Vfa01 vs Vfa02")
    print("=" * 80)
    print(df[["Speed Range", "Training Samples", "Vfa01 Samples", "Vfa02 Samples"]].to_string(index=False))

    print("\nSummary Statistics:")
    print(f"  Train: Mean = {np.mean(all_train_speeds):.2f} m/s, Median = {np.median(all_train_speeds):.2f} m/s, Max = {np.max(all_train_speeds):.2f} m/s")
    print(f"  Vfa01: Mean = {np.mean(vfa01_speeds):.2f} m/s, Median = {np.median(vfa01_speeds):.2f} m/s, Max = {np.max(vfa01_speeds):.2f} m/s")
    print(f"  Vfa02: Mean = {np.mean(vfa02_speeds):.2f} m/s, Median = {np.median(vfa02_speeds):.2f} m/s, Max = {np.max(vfa02_speeds):.2f} m/s")


if __name__ == "__main__":
    analyze_speed_distributions()
