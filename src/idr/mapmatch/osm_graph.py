"""Independent OpenStreetMap Road Graph Loader with offline caching.

RULES 15, 16, 17, 64:
- Road graphs are loaded from independent OpenStreetMap data.
- NEVER construct road graphs from ground-truth trajectories.
- The evaluation operates 100% offline from cached road geometries.
"""

from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json
import logging
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point

logger = logging.getLogger(__name__)


def geodetic_to_enu(lat: float, lon: float, ref_lat: float, ref_lon: float) -> Tuple[float, float]:
    """Convert geodetic lat/lon (degrees) to local East-North (meters) relative to a reference."""
    R_earth = 6378137.0  # WGS84 equatorial radius
    d_lat = np.deg2rad(lat - ref_lat)
    d_lon = np.deg2rad(lon - ref_lon)
    ref_lat_rad = np.deg2rad(ref_lat)
    
    north = d_lat * R_earth
    east = d_lon * R_earth * np.cos(ref_lat_rad)
    return float(east), float(north)


class OSMGraphLoader:
    """Manages offline loading and conversion of OpenStreetMap road networks into local ENU road graphs."""

    def __init__(self, cache_dir: Path = Path("data/osm")):
        self.cache_dir = Path(cache_dir)
        self.graph: Optional[nx.MultiDiGraph] = None

    def load_offline_graph(self, drive_id: str, ref_lat: float, ref_lon: float) -> nx.MultiDiGraph:
        """Load pre-cached independent OSM road network and transform to local ENU coordinates.
        
        Args:
            drive_id: Drive identifier (e.g. 'Vfa01' or 'Vfa02')
            ref_lat, ref_lon: Origin for local East-North coordinate projection
        """
        # Look for drive-specific cached OSM road file
        candidates = [
            self.cache_dir / f"{drive_id}_osm_roads.json",
            self.cache_dir / f"{drive_id.split('_')[-1]}_osm_roads.json",
        ]
        
        osm_json_path = None
        for c in candidates:
            if c.exists():
                osm_json_path = c
                break
                
        if osm_json_path is None:
            # Fallback to any available cached OSM graph
            available = list(self.cache_dir.glob("*_osm_roads.json"))
            if available:
                osm_json_path = available[0]
            else:
                raise FileNotFoundError(
                    f"FATAL: Offline OSM road network not found for drive '{drive_id}' in {self.cache_dir}.\n"
                    "Run 'py -3.11 scripts/download_osm_roads.py' to acquire independent OSM data.\n"
                    "Ground truth road fallback is strictly forbidden by Rule 15 & 64."
                )

        logger.info(f"Loading independent OSM road network from {osm_json_path} (offline mode)")
        with open(osm_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ways = data.get("ways", [])
        logger.info(f"Loaded {len(ways)} independent OSM ways from {osm_json_path.name}")

        G = nx.MultiDiGraph()
        node_id = 0

        for way in ways:
            coords = way.get("coordinates", [])
            if len(coords) < 2:
                continue

            # Convert geodetic lat/lon coordinates to local ENU meters
            enu_pts = []
            for lat, lon in coords:
                e, n = geodetic_to_enu(lat, lon, ref_lat, ref_lon)
                enu_pts.append((e, n))

            # Add way segments to graph
            for i in range(len(enu_pts) - 1):
                p1 = enu_pts[i]
                p2 = enu_pts[i + 1]
                
                # Check segment length to avoid zero-length edges
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                seg_len = np.hypot(dx, dy)
                if seg_len < 0.5:
                    continue

                u = node_id
                v = node_id + 1
                node_id += 2

                G.add_node(u, x=p1[0], y=p1[1])
                G.add_node(v, x=p2[0], y=p2[1])

                geom = LineString([p1, p2])
                azimuth = np.arctan2(dy, dx)
                G.add_edge(
                    u, v, 0,
                    geometry=geom,
                    length=seg_len,
                    azimuth=azimuth,
                    highway=way.get("highway", "primary"),
                    name=way.get("name", ""),
                )
                
                # Two-way roads allow traversal in opposite direction
                if way.get("oneway", "no") != "yes":
                    G.add_edge(
                        v, u, 0,
                        geometry=LineString([p2, p1]),
                        length=seg_len,
                        azimuth=np.arctan2(-dy, -dx),
                        highway=way.get("highway", "primary"),
                        name=way.get("name", ""),
                    )

        self.graph = G
        logger.info(f"Constructed independent road graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} directed edges.")
        return G
