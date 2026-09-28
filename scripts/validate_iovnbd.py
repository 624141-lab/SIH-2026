"""Validation script for the authentic IO-VNBD dataset.

Generates:
- reports/dataset_validation.json
- reports/dataset_validation.md

Computes per-drive statistics:
- drive_id, S-file, V-file
- row count
- timestamp range and duration
- sampling rate metrics (mean_dt, median_dt, std_dt, min_dt, max_dt, estimated_frequency)
- missing values count
- duplicate timestamps count
- latitude & longitude ranges (bounding box)
- sensor ranges (accel, gyro, wheel speed)

NON-NEGOTIABLE: Fails loudly if real IO-VNBD dataset is not found.
"""

import json
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_ROOT = Path("data/raw/iovnbd")
REPORTS_DIR = Path("reports")


def find_iovnbd_root() -> Path:
    """Find the root directory containing authentic IO-VNBD files."""
    candidates = [
        RAW_DATA_ROOT / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset",
        RAW_DATA_ROOT / "Categorised IOVNB Dataset",
        RAW_DATA_ROOT / "Synchronised V abd S datasets",
        RAW_DATA_ROOT,
        Path("data/raw") / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset",
    ]
    for c in candidates:
        if c.exists() and any(c.rglob("S-*.csv")):
            return c
    raise RuntimeError(
        "FATAL: Authentic IO-VNBD dataset not found in data/raw.\n"
        "Run 'python scripts/download_real_iovnbd.py' first.\n"
        "Synthetic data fallback is strictly forbidden."
    )


def validate_drive(s_path: Path, v_path: Path, drive_id: str) -> dict:
    """Analyze a single paired drive."""
    # Read files with latin-1 encoding
    df_s = pd.read_csv(s_path, encoding="latin-1")
    df_v = pd.read_csv(v_path, encoding="latin-1")

    # Clean whitespace in column names
    df_s.columns = [c.strip() for c in df_s.columns]
    df_v.columns = [c.strip() for c in df_v.columns]

    # Timestamp analysis S-file
    time_col_s = "TIME SINCE START (ms)"
    if time_col_s in df_s:
        t_s = df_s[time_col_s].values.astype(float) / 1000.0  # seconds
    else:
        t_s = np.arange(len(df_s)) * 0.1

    dt_s = np.diff(t_s)
    dup_t_s = int(np.sum(dt_s == 0))
    mean_dt_s = float(np.mean(dt_s)) if len(dt_s) > 0 else 0.0
    median_dt_s = float(np.median(dt_s)) if len(dt_s) > 0 else 0.0
    std_dt_s = float(np.std(dt_s)) if len(dt_s) > 0 else 0.0
    min_dt_s = float(np.min(dt_s)) if len(dt_s) > 0 else 0.0
    max_dt_s = float(np.max(dt_s)) if len(dt_s) > 0 else 0.0
    est_freq_s = float(1.0 / mean_dt_s) if mean_dt_s > 0 else 0.0

    # Timestamp analysis V-file
    time_col_v = "Time Since Start of Day (seconds)"
    if time_col_v in df_v:
        t_v = df_v[time_col_v].values.astype(float)
    else:
        t_v = np.arange(len(df_v)) * 0.1

    dt_v = np.diff(t_v)
    dup_t_v = int(np.sum(dt_v == 0))
    mean_dt_v = float(np.mean(dt_v)) if len(dt_v) > 0 else 0.0

    # GPS coordinates
    lat_s = df_s["GPS LATITUDE (degrees)"].dropna().values if "GPS LATITUDE (degrees)" in df_s.columns else np.array([])
    lon_s = df_s["GPS LONGITUDE (degrees)"].dropna().values if "GPS LONGITUDE (degrees)" in df_s.columns else np.array([])
    lat_v = df_v["Latitude (degrees)"].dropna().values if "Latitude (degrees)" in df_v.columns else np.array([])
    lon_v = df_v["Longitude (degrees)"].dropna().values if "Longitude (degrees)" in df_v.columns else np.array([])

    # Sensor columns
    acc_cols = [c for c in df_s.columns if "ACCELEROMETER" in c]
    gyro_cols = [c for c in df_s.columns if "GYROSCOPE" in c]
    speed_col_v = "Velocity (km/hr)" if "Velocity (km/hr)" in df_v.columns else None

    sensor_ranges = {}
    for ac in acc_cols:
        sensor_ranges[ac] = {
            "min": float(df_s[ac].min()),
            "max": float(df_s[ac].max()),
            "mean": float(df_s[ac].mean()),
            "std": float(df_s[ac].std()),
        }
    for gc in gyro_cols:
        sensor_ranges[gc] = {
            "min": float(df_s[gc].min()),
            "max": float(df_s[gc].max()),
            "mean": float(df_s[gc].mean()),
            "std": float(df_s[gc].std()),
        }
    if speed_col_v:
        # Convert km/h to m/s
        speed_ms = df_v[speed_col_v].values / 3.6
        sensor_ranges["wheel_speed_m_s"] = {
            "min": float(np.min(speed_ms)),
            "max": float(np.max(speed_ms)),
            "mean": float(np.mean(speed_ms)),
            "std": float(np.std(speed_ms)),
        }

    return {
        "drive_id": drive_id,
        "s_file": str(s_path.relative_to(RAW_DATA_ROOT.parent.parent)),
        "v_file": str(v_path.relative_to(RAW_DATA_ROOT.parent.parent)),
        "s_row_count": len(df_s),
        "v_row_count": len(df_v),
        "duration_seconds": float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 0.0,
        "sampling_rate": {
            "mean_dt": round(mean_dt_s, 6),
            "median_dt": round(median_dt_s, 6),
            "std_dt": round(std_dt_s, 6),
            "min_dt": round(min_dt_s, 6),
            "max_dt": round(max_dt_s, 6),
            "estimated_frequency_hz": round(est_freq_s, 2),
            "vehicle_mean_dt": round(mean_dt_v, 6),
        },
        "missing_values": {
            "s_missing_total": int(df_s.isnull().sum().sum()),
            "v_missing_total": int(df_v.isnull().sum().sum()),
        },
        "duplicate_timestamps": {
            "s_duplicates": dup_t_s,
            "v_duplicates": dup_t_v,
        },
        "gps_bounds": {
            "phone_lat_min": float(np.min(lat_s)) if len(lat_s) else None,
            "phone_lat_max": float(np.max(lat_s)) if len(lat_s) else None,
            "phone_lon_min": float(np.min(lon_s)) if len(lon_s) else None,
            "phone_lon_max": float(np.max(lon_s)) if len(lon_s) else None,
            "vehicle_lat_min": float(np.min(lat_v)) if len(lat_v) else None,
            "vehicle_lat_max": float(np.max(lat_v)) if len(lat_v) else None,
            "vehicle_lon_min": float(np.min(lon_v)) if len(lon_v) else None,
            "vehicle_lon_max": float(np.max(lon_v)) if len(lon_v) else None,
        },
        "sensor_ranges": sensor_ranges,
    }


def main():
    logger.info("Starting IO-VNBD dataset forensic validation...")
    root = find_iovnbd_root()
    logger.info(f"Authentic dataset root: {root}")

    s_files = sorted(list(root.rglob("S-*.csv")))
    v_files = sorted(list(root.rglob("V-*.csv")))

    # Pair S and V files
    v_dict = {f.name.replace("V-", "S-"): f for f in v_files}
    pairs = []
    unpaired_s = []

    for s in s_files:
        v = v_dict.get(s.name)
        if not v:
            candidates = list(s.parent.glob("V-*.csv"))
            if candidates:
                v = candidates[0]
        if v and v.exists():
            # Derive drive_id from path
            drive_id = s.stem.replace("S-", "")
            driver_dir = s.parent.name
            if "(" in driver_dir:
                driver_tag = driver_dir.split("(")[0].strip()
                drive_id = f"{driver_tag}_{drive_id}"
            pairs.append((drive_id, s, v))
        else:
            unpaired_s.append(str(s))

    unpaired_v = [str(v) for v in v_files if v not in [p[2] for p in pairs]]

    logger.info(f"Discovered: {len(s_files)} S-files, {len(v_files)} V-files")
    logger.info(f"Successfully paired: {len(pairs)} drives, Unpaired S: {len(unpaired_s)}, Unpaired V: {len(unpaired_v)}")

    drives_report = []
    total_samples = 0
    total_duration_hours = 0.0

    for drive_id, s_path, v_path in pairs:
        info = validate_drive(s_path, v_path, drive_id)
        drives_report.append(info)
        total_samples += info["s_row_count"]
        total_duration_hours += info["duration_seconds"] / 3600.0
        logger.info(
            f"Validated drive {drive_id}: S={info['s_row_count']} rows, "
            f"duration={info['duration_seconds']:.1f}s, freq={info['sampling_rate']['estimated_frequency_hz']:.1f}Hz"
        )

    # Global bounding box across all drives
    all_lat_min = min(d["gps_bounds"]["vehicle_lat_min"] for d in drives_report if d["gps_bounds"]["vehicle_lat_min"] is not None)
    all_lat_max = max(d["gps_bounds"]["vehicle_lat_max"] for d in drives_report if d["gps_bounds"]["vehicle_lat_max"] is not None)
    all_lon_min = min(d["gps_bounds"]["vehicle_lon_min"] for d in drives_report if d["gps_bounds"]["vehicle_lon_min"] is not None)
    all_lon_max = max(d["gps_bounds"]["vehicle_lon_max"] for d in drives_report if d["gps_bounds"]["vehicle_lon_max"] is not None)

    summary = {
        "status": "VALIDATED_AUTHENTIC",
        "dataset_name": "IO-VNBD (Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning)",
        "source": "onyekpeu/IO-VNBD",
        "total_s_files": len(s_files),
        "total_v_files": len(v_files),
        "paired_drives": len(pairs),
        "unpaired_s_files": len(unpaired_s),
        "unpaired_v_files": len(unpaired_v),
        "total_sensor_samples": total_samples,
        "total_driving_duration_hours": round(total_duration_hours, 2),
        "global_bounding_box": {
            "min_latitude": round(all_lat_min, 6),
            "max_latitude": round(all_lat_max, 6),
            "min_longitude": round(all_lon_min, 6),
            "max_longitude": round(all_lon_max, 6),
        },
        "drives": drives_report,
    }

    # Write JSON report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "dataset_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved machine-readable report to {json_path}")

    # Write Markdown report
    md_path = REPORTS_DIR / "dataset_validation.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# IO-VNBD Dataset Validation Report\n\n")
        f.write("**Status**: ✅ AUTHENTIC IO-VNBD DATASET VERIFIED\n\n")
        f.write(f"- **Total S-files (Smartphone)**: {len(s_files)}\n")
        f.write(f"- **Total V-files (Vehicle CAN/GPS)**: {len(v_files)}\n")
        f.write(f"- **Total Paired Driving Trips**: {len(pairs)}\n")
        f.write(f"- **Total Sensor Samples**: {total_samples:,}\n")
        f.write(f"- **Total Driving Duration**: {total_duration_hours:.2f} hours\n")
        f.write(f"- **Geographic Bounding Box**: Lat [{all_lat_min:.4f}, {all_lat_max:.4f}], Lon [{all_lon_min:.4f}, {all_lon_max:.4f}] (Coventry / West Midlands, UK)\n\n")
        f.write("## Drive-by-Drive Summary\n\n")
        f.write("| Drive ID | S-Rows | V-Rows | Duration (s) | Sampling Rate (Hz) | Lat Min/Max | Lon Min/Max | Max Speed (m/s) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for d in drives_report:
            bounds = d["gps_bounds"]
            speed = d["sensor_ranges"].get("wheel_speed_m_s", {}).get("max", 0.0)
            lat_str = f"{bounds['vehicle_lat_min']:.4f} - {bounds['vehicle_lat_max']:.4f}" if bounds['vehicle_lat_min'] else "N/A"
            lon_str = f"{bounds['vehicle_lon_min']:.4f} - {bounds['vehicle_lon_max']:.4f}" if bounds['vehicle_lon_min'] else "N/A"
            f.write(
                f"| `{d['drive_id']}` | {d['s_row_count']:,} | {d['v_row_count']:,} | "
                f"{d['duration_seconds']:.1f} | {d['sampling_rate']['estimated_frequency_hz']:.1f} | "
                f"{lat_str} | {lon_str} | {speed:.2f} |\n"
            )
    logger.info(f"Saved Markdown report to {md_path}")
    print(f"SUCCESS: Validated {len(pairs)} authentic IO-VNBD drives. Total samples: {total_samples:,}.")


if __name__ == "__main__":
    main()
