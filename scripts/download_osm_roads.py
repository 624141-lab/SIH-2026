"""Acquire independent OpenStreetMap (OSM) road networks for evaluation drives.

RULE 15, 16, 17, 64:
- NEVER construct road graphs from ground-truth trajectories.
- The road network must be acquired from an independent OpenStreetMap source.
- Downloaded ONCE and serialized into offline road graph cache.
- Runtime evaluation executes 100% offline from the local cached graph.
"""

import json
import logging
from pathlib import Path
import urllib.parse
import urllib.request
import numpy as np

from idr.config import OSM_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
import ssl

logger = logging.getLogger(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]


def fetch_osm_corridor(
    min_lat: float, max_lat: float, min_lon: float, max_lon: float, output_path: Path, margin: float = 0.02
) -> Path:
    """Fetch road segments from OpenStreetMap via Overpass API for a given bounding box."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 1000:
        logger.info(f"Offline OSM road graph already exists: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        return output_path

    bbox = f"{min_lat - margin:.4f},{min_lon - margin:.4f},{max_lat + margin:.4f},{max_lon + margin:.4f}"
    logger.info(f"Fetching independent OSM roads for bounding box [{bbox}]...")

    ssl_ctx = ssl._create_unverified_context()
    data = None
    last_err = None

    for highway_filter in [
        "motorway|trunk|primary",
        "motorway|trunk|primary|secondary|tertiary",
    ]:
        query = f"""
[out:json][timeout:60];
(
  way["highway"~"{highway_filter}"]({bbox});
);
out geom;
"""
        for ep in OVERPASS_ENDPOINTS:
            try:
                logger.info(f"Attempting Overpass endpoint: {ep} (filter: {highway_filter})...")
                encoded_url = ep + "?data=" + urllib.parse.quote(query)
                req = urllib.request.Request(encoded_url, headers={"User-Agent": "IDR-Research-Benchmark/1.0"})
                with urllib.request.urlopen(req, timeout=65, context=ssl_ctx) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    elements = data.get("elements", [])
                    if len(elements) > 0:
                        logger.info(f"Received {len(elements)} raw road elements from {ep}.")
                        break
            except Exception as e:
                last_err = e
                logger.warning(f"Endpoint {ep} failed: {e}")
        if data and len(data.get("elements", [])) > 0:
            break
            last_err = e

    if data is None:
        raise RuntimeError(
            f"FATAL: Unable to acquire independent OSM road graph for {bbox}: {last_err}\n"
            "Ground-truth trajectory fallback is strictly forbidden by Rule 15 & 64."
        )

    elements = data.get("elements", [])
    try:
            
            # Extract road segments with geometry
            road_network = {
                "source": "OpenStreetMap via Overpass API (Independent Ground Truth)",
                "bbox": {
                    "min_lat": min_lat - margin,
                    "max_lat": max_lat + margin,
                    "min_lon": min_lon - margin,
                    "max_lon": max_lon + margin,
                },
                "num_ways": len(elements),
                "ways": [],
            }
            
            for el in elements:
                geom = el.get("geometry", [])
                if len(geom) >= 2:
                    coords = [[pt["lat"], pt["lon"]] for pt in geom]
                    tags = el.get("tags", {})
                    road_network["ways"].append({
                        "id": el["id"],
                        "highway": tags.get("highway", "unknown"),
                        "name": tags.get("name", ""),
                        "maxspeed": tags.get("maxspeed", ""),
                        "oneway": tags.get("oneway", "no"),
                        "coordinates": coords,
                    })

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(road_network, f)
            logger.info(f"Successfully cached independent OSM road graph to {output_path} ({len(road_network['ways'])} valid road segments).")
            return output_path

    except Exception as e:
        logger.error(f"Failed to acquire OSM data from Overpass: {e}")
        raise RuntimeError(
            f"FATAL: Unable to acquire independent OSM road graph for {bbox}: {e}\n"
            "Ground-truth trajectory fallback is strictly forbidden by Rule 15 & 64."
        ) from e


def acquire_osm_for_test_drives():
    """Acquire offline OSM graphs for test evaluation drives (Vfa01, Vfa02)."""
    # Load dataset validation to obtain test drive bounding boxes
    val_json = REPORTS_DIR / "dataset_validation.json"
    if not val_json.exists():
        raise FileNotFoundError("Run 'py -3.11 scripts/validate_iovnbd.py' first to determine drive bounding boxes.")

    with open(val_json, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    for drive in val_data["drives"]:
        drive_id = drive["drive_id"]
        if drive_id in ["Vfa01", "Vfa02"]:
            bounds = drive["gps_bounds"]
            out_file = OSM_DIR / f"{drive_id}_osm_roads.json"
            fetch_osm_corridor(
                min_lat=bounds["vehicle_lat_min"],
                max_lat=bounds["vehicle_lat_max"],
                min_lon=bounds["vehicle_lon_min"],
                max_lon=bounds["vehicle_lon_max"],
                output_path=out_file,
                margin=0.015,
            )


if __name__ == "__main__":
    acquire_osm_for_test_drives()
