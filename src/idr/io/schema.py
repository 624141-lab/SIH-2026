"""Schema mapping and normalization for authentic IO-VNBD dataset files."""

from dataclasses import dataclass
import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical fields defined for IDR system
CANONICAL_FIELDS_SMARTPHONE = [
    "timestamp",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "phone_lat", "phone_lon", "phone_alt",
    "phone_speed", "phone_heading",
]

CANONICAL_FIELDS_VEHICLE = [
    "timestamp",
    "lat", "lon", "alt",
    "wheel_speed", "heading",
    "yaw_rate", "steering_angle",
    "acc_long", "acc_lat",
]


@dataclass
class SchemaMap:
    """Mapping from canonical field names to actual CSV column names."""
    field_to_col: Dict[str, str]
    file_type: str  # 'smartphone' or 'vehicle'

    def has_required_imu(self) -> bool:
        required = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
        return all(f in self.field_to_col for f in required)

    def has_required_gps(self) -> bool:
        if self.file_type == "smartphone":
            return "phone_lat" in self.field_to_col and "phone_lon" in self.field_to_col
        return "lat" in self.field_to_col and "lon" in self.field_to_col

    def has_required_reference(self) -> bool:
        if self.file_type == "vehicle":
            return "wheel_speed" in self.field_to_col
        return True


def clean_col(c: str) -> str:
    """Strip whitespace and normalize column name."""
    return c.strip()


def detect_schema(csv_path_or_df, file_type_hint: Optional[str] = None) -> SchemaMap:
    """Detect and validate column mapping for authentic IO-VNBD CSV files."""
    if isinstance(csv_path_or_df, pd.DataFrame):
        raw_columns = list(csv_path_or_df.columns)
    else:
        sample = pd.read_csv(csv_path_or_df, nrows=3, encoding="latin-1")
        raw_columns = list(sample.columns)

    cols = {clean_col(c): c for c in raw_columns}
    field_to_col: Dict[str, str] = {}

    # Check whether this is a smartphone (S) or vehicle (V) file
    is_smartphone = any("ACCELEROMETER" in c for c in cols) or (file_type_hint == "smartphone")
    is_vehicle = any("Velocity (km/hr)" in c or "Wheel Speed" in c for c in cols) or (file_type_hint == "vehicle")

    if is_smartphone and not (file_type_hint == "vehicle"):
        file_type = "smartphone"
        # Exact mappings for authentic IO-VNBD smartphone files
        for c, orig in cols.items():
            if "TIME SINCE START" in c:
                field_to_col["timestamp"] = orig
            elif "ACCELEROMETER X" in c:
                field_to_col["acc_x"] = orig
            elif "ACCELEROMETER Y" in c:
                field_to_col["acc_y"] = orig
            elif "ACCELEROMETER Z" in c:
                field_to_col["acc_z"] = orig
            elif "GYROSCOPE Roll" in c:
                field_to_col["gyro_x"] = orig  # Roll is x-rotation
            elif "GYROSCOPE Pitch" in c:
                field_to_col["gyro_y"] = orig  # Pitch is y-rotation
            elif "GYROSCOPE Yaw" in c:
                field_to_col["gyro_z"] = orig  # Yaw is z-rotation
            elif "MAGNETIC FIELD X" in c:
                field_to_col["mag_x"] = orig
            elif "MAGNETIC FIELD Y" in c:
                field_to_col["mag_y"] = orig
            elif "MAGNETIC FIELD Z" in c:
                field_to_col["mag_z"] = orig
            elif "GPS LATITUDE" in c:
                field_to_col["phone_lat"] = orig
            elif "GPS LONGITUDE" in c:
                field_to_col["phone_lon"] = orig
            elif "GPS ALTITUDE" in c:
                field_to_col["phone_alt"] = orig
            elif "GPS SPEED" in c:
                field_to_col["phone_speed"] = orig
            elif "GPS ORIENTATION" in c:
                field_to_col["phone_heading"] = orig
    else:
        file_type = "vehicle"
        # Exact mappings for authentic IO-VNBD vehicle files
        for c, orig in cols.items():
            if "Time Since Start of Day" in c:
                field_to_col["timestamp"] = orig
            elif c == "Latitude (degrees)":
                field_to_col["lat"] = orig
            elif c == "Longitude (degrees)":
                field_to_col["lon"] = orig
            elif c == "Height (km)":
                field_to_col["alt"] = orig
            elif c == "Velocity (km/hr)":
                field_to_col["wheel_speed"] = orig
            elif c == "Heading (degrees)":
                field_to_col["heading"] = orig
            elif c == "Yaw Rate (deg/sec)":
                field_to_col["yaw_rate"] = orig
            elif c == "Steering Angle (degrees)":
                field_to_col["steering_angle"] = orig
            elif "Indicated Longitudinal Acceleration" in c:
                field_to_col["acc_long"] = orig
            elif "Indicated Lateral Acceleration" in c:
                field_to_col["acc_lat"] = orig

    schema = SchemaMap(field_to_col=field_to_col, file_type=file_type)

    if file_type == "smartphone" and not schema.has_required_imu():
        missing = [f for f in ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"] if f not in field_to_col]
        raise ValueError(f"FATAL: Authentic smartphone IMU fields missing: {missing}. Available: {list(cols.keys())}")

    if file_type == "vehicle" and not schema.has_required_reference():
        raise ValueError(f"FATAL: Authentic vehicle reference speed missing. Available: {list(cols.keys())}")

    return schema
