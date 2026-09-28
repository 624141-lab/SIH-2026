"""Loader for authentic IO-VNBD synchronized drive pairs."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import numpy as np
import pandas as pd
from .schema import detect_schema, SchemaMap

logger = logging.getLogger(__name__)


@dataclass
class IOVNBDrive:
    """Encapsulates a synchronized authentic IO-VNBD drive recording."""
    drive_id: str
    phone_df: pd.DataFrame
    vehicle_df: pd.DataFrame
    phone_schema: SchemaMap
    vehicle_schema: SchemaMap

    def get_synced_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract synchronized arrays:
        - phone_imu: (N, 6) [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] in smartphone body frame
        - gt_coords: (N, 3) [lat, lon, alt] from high-accuracy vehicle GPS
        - vehicle_speed: (N,) forward reference speed from vehicle CAN ECU (m/s)
        - timestamps: (N,) synchronized relative seconds starting at 0.0
        """
        N = len(self.phone_df)
        
        # Phone IMU (6 channels)
        phone_imu = self.phone_df[["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]].to_numpy(dtype=np.float32)
        
        # Ground-truth GPS coordinates from vehicle ECU
        gt_cols = ["lat", "lon", "alt"]
        for c in gt_cols:
            if c not in self.vehicle_df.columns:
                if c == "alt":
                    self.vehicle_df[c] = 0.0
                elif c in ["lat", "lon"]:
                    # Fallback to phone GPS if vehicle GPS missing
                    phone_c = f"phone_{c}"
                    if phone_c in self.phone_df.columns:
                        self.vehicle_df[c] = self.phone_df[phone_c]
                    else:
                        raise ValueError(f"Drive {self.drive_id} missing GPS coordinate {c}")
        
        gt_coords = self.vehicle_df[gt_cols].to_numpy(dtype=np.float64)[:N]
        if len(gt_coords) < N:
            gt_coords = np.pad(gt_coords, ((0, N - len(gt_coords)), (0, 0)), mode="edge")

        # Vehicle reference speed (CAN wheel speed, m/s)
        if "wheel_speed" in self.vehicle_df.columns:
            v_speed = self.vehicle_df["wheel_speed"].to_numpy(dtype=np.float32)[:N]
            if len(v_speed) < N:
                v_speed = np.pad(v_speed, (0, N - len(v_speed)), mode="edge")
        else:
            v_speed = np.zeros(N, dtype=np.float32)

        t = self.phone_df["timestamp"].to_numpy(dtype=np.float64)
        return phone_imu, gt_coords, v_speed, t


def standardize_phone_dataframe(df: pd.DataFrame, schema: SchemaMap) -> pd.DataFrame:
    """Standardize phone dataframe into canonical units and columns."""
    inv_map = {col: field for field, col in schema.field_to_col.items()}
    df_renamed = df.rename(columns=inv_map)
    cols_to_keep = [f for f in schema.field_to_col.keys() if f in df_renamed.columns]
    std = df_renamed[cols_to_keep].copy()

    # Normalize timestamp: TIME SINCE START (ms) -> relative seconds
    if "timestamp" in std.columns:
        t_raw = std["timestamp"].values.astype(float)
        # If timestamp is in ms, convert to seconds
        if np.nanmedian(np.diff(t_raw)) > 1.0:
            t = (t_raw - t_raw[0]) / 1000.0
        else:
            t = t_raw - t_raw[0]
        std["timestamp"] = t
    else:
        std["timestamp"] = np.arange(len(std)) * 0.1

    # Phone speed km/h -> m/s if present
    if "phone_speed" in std.columns:
        std["phone_speed"] = std["phone_speed"].astype(float) / 3.6

    std = std.interpolate(method="linear").bfill().ffill()
    return std


def standardize_vehicle_dataframe(df: pd.DataFrame, schema: SchemaMap) -> pd.DataFrame:
    """Standardize vehicle dataframe into canonical units and columns."""
    inv_map = {col: field for field, col in schema.field_to_col.items()}
    df_renamed = df.rename(columns=inv_map)
    cols_to_keep = [f for f in schema.field_to_col.keys() if f in df_renamed.columns]
    std = df_renamed[cols_to_keep].copy()

    # Normalize timestamp
    if "timestamp" in std.columns:
        t_raw = std["timestamp"].values.astype(float)
        t = t_raw - t_raw[0]
        std["timestamp"] = t
    else:
        std["timestamp"] = np.arange(len(std)) * 0.1

    # Speed km/h -> m/s
    if "wheel_speed" in std.columns:
        std["wheel_speed"] = std["wheel_speed"].astype(float) / 3.6

    # Altitude km -> m
    if "alt" in std.columns:
        std["alt"] = std["alt"].astype(float) * 1000.0

    # Yaw rate deg/s -> rad/s
    if "yaw_rate" in std.columns:
        std["yaw_rate"] = np.deg2rad(std["yaw_rate"].astype(float))

    std = std.interpolate(method="linear").bfill().ffill()
    return std


def find_drive_files(raw_dir: Path, drive_id: str) -> Tuple[Path, Path]:
    """Locate authentic S-*.csv and V-*.csv files for a given drive identifier."""
    raw_dir = Path(raw_dir)
    
    # Handle drive_id formatting (e.g., M_M, S1, Vfa01, Vta1a)
    stem_target = drive_id
    if "_" in drive_id:
        stem_target = drive_id.split("_")[-1]  # M_M -> M

    # Search for matching S file
    s_candidates = list(raw_dir.rglob(f"S-{stem_target}.csv"))
    if not s_candidates:
        # Search case-insensitive or partial
        for p in raw_dir.rglob("S-*.csv"):
            if stem_target.lower() in p.stem.lower():
                s_candidates.append(p)
                break

    if not s_candidates:
        raise FileNotFoundError(f"FATAL: Authentic IO-VNBD file S-{drive_id}.csv not found in {raw_dir}")

    s_path = s_candidates[0]
    folder = s_path.parent
    
    # Search for matching V file in the same folder
    v_candidates = list(folder.glob("V-*.csv"))
    if not v_candidates:
        # Search globally
        v_candidates = list(raw_dir.rglob(f"V-{stem_target}.csv"))
    
    if not v_candidates:
        raise FileNotFoundError(f"FATAL: Authentic IO-VNBD file V-{drive_id}.csv not found in {raw_dir}")

    v_path = v_candidates[0]
    return s_path, v_path


def load_drive_pair(raw_dir: Path, drive_id: str) -> IOVNBDrive:
    """Load and synchronize an authentic IO-VNBD drive pair."""
    s_path, v_path = find_drive_files(raw_dir, drive_id)

    raw_phone = pd.read_csv(s_path, encoding="latin-1")
    raw_vehicle = pd.read_csv(v_path, encoding="latin-1")

    phone_schema = detect_schema(raw_phone, file_type_hint="smartphone")
    vehicle_schema = detect_schema(raw_vehicle, file_type_hint="vehicle")

    phone_df = standardize_phone_dataframe(raw_phone, phone_schema)
    vehicle_df = standardize_vehicle_dataframe(raw_vehicle, vehicle_schema)

    # Synchronize lengths if needed
    N_s = len(phone_df)
    N_v = len(vehicle_df)

    if N_s != N_v:
        # Align vehicle dataframe onto phone timestamp grid
        t_s = phone_df["timestamp"].values
        t_v = vehicle_df["timestamp"].values
        
        # Resample vehicle numerical columns onto t_s
        v_aligned = pd.DataFrame({"timestamp": t_s})
        for col in vehicle_df.columns:
            if col != "timestamp" and np.issubdtype(vehicle_df[col].dtype, np.number):
                v_aligned[col] = np.interp(t_s, t_v, vehicle_df[col].values)
            elif col != "timestamp":
                # Forward fill categorical
                v_aligned[col] = vehicle_df[col].iloc[:N_s].values
        vehicle_df = v_aligned

    return IOVNBDrive(
        drive_id=drive_id,
        phone_df=phone_df,
        vehicle_df=vehicle_df,
        phone_schema=phone_schema,
        vehicle_schema=vehicle_schema,
    )
