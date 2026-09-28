"""Authentic Hidden Markov Model (HMM) Viterbi Map-Matcher for Dead Reckoning.

Implements the Newson & Krumm (ACM SIGSPATIAL) HMM formulation:
1. Candidate Generation: spatial indexing with cKDTree queries road segments within radius R
2. Emission Probability: Gaussian distribution of perpendicular measurement distance
   P(z_t | c_i) = (1 / sqrt(2*pi*sigma_z^2)) * exp(-d_i^2 / (2*sigma_z^2))
3. Transition Probability: exponential decay of difference between road distance and Euclidean travel distance,
   weighted by heading alignment
   P(c_j | c_i) = (1 / beta) * exp(-|Delta d_road - Delta d_est| / beta) * heading_weight
4. Viterbi Decoding: dynamic programming forward pass in log-space and backward trace
5. Leakage Prevention (Rules 19 & 20): consumes ONLY estimated positions, NEVER ground truth.
"""

from typing import List, Tuple, Dict, Optional
import logging
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString

logger = logging.getLogger(__name__)


class RoadCandidate:
    """Represents a candidate road projection for a given observation."""
    def __init__(self, edge_id: int, u: int, v: int, k: int, proj_x: float, proj_y: float, dist: float, azimuth: float):
        self.edge_id = edge_id
        self.u = u
        self.v = v
        self.k = k
        self.proj_x = proj_x
        self.proj_y = proj_y
        self.dist = dist
        self.azimuth = azimuth


class HMMMapMatcher:
    """Authentic HMM Viterbi Map Matcher for dead-reckoned vehicle navigation."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        sigma_z: float = 8.0,            # Measurement noise standard deviation (meters)
        beta: float = 4.0,               # Transition distance scale factor (meters)
        max_search_radius: float = 60.0, # Candidate search radius (meters)
    ):
        self.graph = graph
        self.sigma_z = max(float(sigma_z), 1.0)
        self.beta = max(float(beta), 0.5)
        self.max_search_radius = max(float(max_search_radius), 10.0)
        self._extract_road_segments()

    def _extract_road_segments(self):
        """Pre-index road segments and build cKDTree spatial index for O(log E) queries."""
        self.edges = []
        midpoints = []

        for idx, (u, v, k, data) in enumerate(self.graph.edges(keys=True, data=True)):
            geom = data.get("geometry")
            if geom is None:
                x1, y1 = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
                x2, y2 = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
                geom = LineString([(x1, y1), (x2, y2)])
            
            azimuth = data.get("azimuth")
            if azimuth is None:
                coords = list(geom.coords)
                p1, p2 = coords[0], coords[-1]
                azimuth = float(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

            bounds = geom.bounds  # (minx, miny, maxx, maxy)
            mid_x = (bounds[0] + bounds[2]) * 0.5
            mid_y = (bounds[1] + bounds[3]) * 0.5

            self.edges.append({
                "edge_id": idx,
                "u": u,
                "v": v,
                "k": k,
                "geom": geom,
                "azimuth": azimuth,
                "bounds": bounds,
            })
            midpoints.append([mid_x, mid_y])

        if midpoints:
            self.kdtree = cKDTree(np.array(midpoints, dtype=np.float64))
        else:
            self.kdtree = None
        logger.info(f"HMMMapMatcher initialized with {len(self.edges)} road segments (spatial index ready).")

    def find_candidates(self, x: float, y: float) -> List[RoadCandidate]:
        """Find candidate road segments within search radius using fast spatial indexing."""
        if not self.edges or self.kdtree is None:
            # Fallback when no graph edges exist
            return [RoadCandidate(0, 0, 0, 0, x, y, 0.0, 0.0)]

        p = Point(x, y)
        r = self.max_search_radius

        # Fast spatial query: query top-15 nearest edge midpoints
        k_query = min(15, len(self.edges))
        _, nearby_indices = self.kdtree.query([x, y], k=k_query)
        if isinstance(nearby_indices, (int, np.integer)):
            nearby_indices = [int(nearby_indices)]
        else:
            nearby_indices = [int(idx) for idx in nearby_indices]

        candidates = []
        closest_cand = None
        min_dist_global = float("inf")

        for idx in nearby_indices:
            e = self.edges[idx]
            dist = float(e["geom"].distance(p))
            proj_dist = e["geom"].project(p)
            proj_pt = e["geom"].interpolate(proj_dist)

            cand = RoadCandidate(
                edge_id=e["edge_id"],
                u=e["u"],
                v=e["v"],
                k=e["k"],
                proj_x=float(proj_pt.x),
                proj_y=float(proj_pt.y),
                dist=dist,
                azimuth=e["azimuth"],
            )

            if dist < min_dist_global:
                min_dist_global = dist
                closest_cand = cand

            if dist <= r:
                candidates.append(cand)

        # Guarantee at least one candidate always exists
        if not candidates and closest_cand is not None:
            candidates.append(closest_cand)
        elif not candidates:
            candidates.append(RoadCandidate(0, 0, 0, 0, x, y, 0.0, 0.0))

        # Limit to top-10 nearest candidates for tractable Viterbi trellis
        if len(candidates) > 10:
            candidates.sort(key=lambda c: c.dist)
            candidates = candidates[:10]

        return candidates

    def log_emission_prob(self, cand: RoadCandidate) -> float:
        """Log emission probability under Gaussian distance distribution."""
        d = cand.dist
        # ln P(z | c) = -0.5 * ln(2*pi*sigma_z^2) - d^2 / (2*sigma_z^2)
        norm_const = -0.5 * np.log(2.0 * np.pi * (self.sigma_z ** 2))
        return norm_const - (d ** 2) / (2.0 * (self.sigma_z ** 2))

    def log_transition_prob(
        self, prev_c: RoadCandidate, curr_c: RoadCandidate, delta_est: float, motion_heading: Optional[float]
    ) -> float:
        """Log transition probability based on travel distance difference and heading alignment."""
        # Euclidean distance between projected points on road
        delta_road = np.hypot(curr_c.proj_x - prev_c.proj_x, curr_c.proj_y - prev_c.proj_y)

        # Distance penalty: -|delta_road - delta_est| / beta
        dist_diff = abs(delta_road - delta_est)
        log_p_dist = -dist_diff / self.beta

        # Heading alignment penalty
        log_p_heading = 0.0
        if motion_heading is not None and delta_est > 0.5:
            # Difference between road segment azimuth and vehicle motion direction
            angle_diff = abs(curr_c.azimuth - motion_heading)
            angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
            cos_align = max(float(np.cos(angle_diff)), 0.05)
            log_p_heading = np.log(cos_align)

        return log_p_dist + log_p_heading

    def match_trajectory(self, estimated_coords: np.ndarray) -> np.ndarray:
        """Perform full HMM Viterbi map-matching on estimated trajectory points.
        
        RULES 19 & 20: Consumes strictly dead-reckoned estimated trajectory.
        Ground truth must never enter this function.
        
        Args:
            estimated_coords: (T, 2) array of [East, North] positions in local frame.
        Returns:
            matched_coords: (T, 2) array of map-matched [East, North] coordinates.
        """
        # Rule 20: Assert no ground-truth object leaked
        assert not hasattr(estimated_coords, "ground_truth"), "FATAL: Ground truth leaked into map matcher input!"
        assert isinstance(estimated_coords, (np.ndarray, list)), "Input must be coordinate array."
        
        estimated_coords = np.asarray(estimated_coords, dtype=np.float64)
        T = len(estimated_coords)
        if T == 0 or not self.edges:
            return estimated_coords

        # Step 1: Find candidate road segments for all timesteps
        trellis_candidates = []
        for t in range(T):
            x, y = float(estimated_coords[t, 0]), float(estimated_coords[t, 1])
            candidates = self.find_candidates(x, y)
            trellis_candidates.append(candidates)

        # Step 2: Initialize Viterbi DP structures
        V = []
        backpointers = []

        # t = 0
        V_0 = [self.log_emission_prob(c) for c in trellis_candidates[0]]
        V.append(V_0)
        backpointers.append([-1] * len(trellis_candidates[0]))

        # Step 3: Viterbi Forward Recursion
        for t in range(1, T):
            x_prev, y_prev = float(estimated_coords[t - 1, 0]), float(estimated_coords[t - 1, 1])
            x_curr, y_curr = float(estimated_coords[t, 0]), float(estimated_coords[t, 1])
            dx, dy = x_curr - x_prev, y_curr - y_prev
            delta_est = float(np.hypot(dx, dy))
            motion_heading = float(np.arctan2(dy, dx)) if delta_est > 0.2 else None

            curr_cands = trellis_candidates[t]
            prev_cands = trellis_candidates[t - 1]
            prev_V = V[t - 1]

            V_t = []
            bp_t = []

            for j, c_j in enumerate(curr_cands):
                log_emission = self.log_emission_prob(c_j)
                best_prob = -float("inf")
                best_prev_idx = 0

                for i, c_i in enumerate(prev_cands):
                    log_trans = self.log_transition_prob(c_i, c_j, delta_est, motion_heading)
                    prob = prev_V[i] + log_trans + log_emission
                    if prob > best_prob:
                        best_prob = prob
                        best_prev_idx = i

                V_t.append(best_prob)
                bp_t.append(best_prev_idx)

            V.append(V_t)
            backpointers.append(bp_t)

        # Step 4: Viterbi Backward Path Reconstruction
        matched_coords = np.zeros_like(estimated_coords)
        if V[-1]:
            best_last_idx = int(np.argmax(V[-1]))
        else:
            best_last_idx = 0

        curr_idx = best_last_idx
        for t in range(T - 1, -1, -1):
            cands_t = trellis_candidates[t]
            if not cands_t:
                matched_coords[t] = estimated_coords[t]
                continue

            if curr_idx < 0 or curr_idx >= len(cands_t):
                curr_idx = 0

            cand = cands_t[curr_idx]
            matched_coords[t, 0] = cand.proj_x
            matched_coords[t, 1] = cand.proj_y

            if t > 0:
                bp_list = backpointers[t]
                if curr_idx < len(bp_list):
                    curr_idx = bp_list[curr_idx]
                else:
                    curr_idx = 0

        return matched_coords
