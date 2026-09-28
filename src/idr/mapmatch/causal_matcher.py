"""Real-Time Online Causal HMM Map Matcher with Out-of-Corridor Gating.

Implements:
1. Strict Online Forward Trellis Recursion: Look-ahead = 0.0 seconds.
2. Out-of-Corridor Distance & Confidence Gating: Rejects map snapping when nearest road > max_distance_gate.
3. Multi-Hypothesis Confidence Scoring: Combines emission likelihood and heading alignment.
4. Optional Bounded Fixed-Lag Finalization: Traces back from live state by L steps (1.5s latency).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree


@dataclass
class CausalRoadCandidate:
    """Road candidate projection for a single observation."""
    edge_id: int
    u: int
    v: int
    k: int
    proj_x: float
    proj_y: float
    dist: float
    azimuth: float


@dataclass
class CausalMatchOutput:
    """Output of the causal map matcher at timestamp t."""
    step: int
    time_s: float
    # Real-time provisional match (0 latency, 0 look-ahead)
    provisional_x: float
    provisional_y: float
    provisional_road_id: int
    # Finalized match (latency = lag_steps * dt)
    finalized_x: float
    finalized_y: float
    finalized_road_id: int
    emission_prob: float
    confidence: float
    is_accepted: bool = True
    d_min: float = 0.0


class CausalHMMMapMatcher:
    """Real-Time Online Causal HMM Map Matcher with bounded latency and out-of-corridor rejection."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        sigma_z: float = 8.0,              # Measurement noise standard deviation (meters)
        beta: float = 4.0,                 # Transition scale factor (meters)
        max_search_radius: float = 60.0,   # Candidate search radius (meters)
        lag_steps: int = 15,               # Fixed-lag smoothing horizon (15 steps = 1.5s at 10Hz)
        dt: float = 0.1,                   # Sampling period (seconds)
        max_distance_gate: float = 30.0,   # Out-of-corridor distance threshold (meters)
        min_road_confidence: float = 0.20, # Minimum confidence threshold to accept map snapping
    ):
        self.graph = graph
        self.sigma_z = max(float(sigma_z), 1.0)
        self.beta = max(float(beta), 0.5)
        self.max_search_radius = max(float(max_search_radius), 10.0)
        self.lag_steps = max(int(lag_steps), 0)
        self.dt = dt
        self.max_distance_gate = max(float(max_distance_gate), 5.0)
        self.min_road_confidence = max(float(min_road_confidence), 0.05)

        self._extract_road_segments()
        self.reset()

    def _extract_road_segments(self):
        """Pre-index road segments and build cKDTree for O(log E) nearest queries."""
        self.edges = []
        midpoints = []

        for idx, (u, v, k, data) in enumerate(self.graph.edges(keys=True, data=True)):
            geom = data.get("geometry")
            if geom is None:
                u_data = self.graph.nodes[u]
                v_data = self.graph.nodes[v]
                geom = LineString([(u_data["x"], u_data["y"]), (v_data["x"], v_data["y"])])

            coords = list(geom.coords)
            if len(coords) < 2:
                continue

            dx = coords[-1][0] - coords[0][0]
            dy = coords[-1][1] - coords[0][1]
            azimuth = float(np.arctan2(dy, dx))

            mid_pt = geom.interpolate(0.5, normalized=True)
            midpoints.append([mid_pt.x, mid_pt.y])

            self.edges.append({
                "edge_id": idx,
                "u": u,
                "v": v,
                "k": k,
                "geom": geom,
                "azimuth": azimuth,
                "length": float(geom.length),
            })

        if midpoints:
            self.kdtree = cKDTree(np.array(midpoints, dtype=np.float64))
        else:
            self.kdtree = None

    def reset(self):
        """Reset internal online trellis state for a new navigation session."""
        self.step_count = 0
        self.prev_candidates: List[CausalRoadCandidate] = []
        self.prev_V: List[float] = []
        self.prev_pos: Optional[Tuple[float, float]] = None

        self.history_candidates: List[List[CausalRoadCandidate]] = []
        self.history_backpointers: List[List[int]] = []
        self.history_provisional: List[Tuple[float, float, int]] = []

    def find_candidates(self, x: float, y: float, search_radius: Optional[float] = None) -> Tuple[List[CausalRoadCandidate], float]:
        """Find candidate road segments within search radius using fast spatial indexing.
        
        Returns (candidates, min_dist_global).
        Rejects candidates that exceed max_distance_gate.
        """
        if not self.edges or self.kdtree is None:
            return [], 999.0

        p = Point(x, y)
        r = search_radius if search_radius is not None else self.max_search_radius
        effective_r = min(r, self.max_distance_gate)

        k_query = min(25, len(self.edges))
        _, nearby_indices = self.kdtree.query([x, y], k=k_query)
        if isinstance(nearby_indices, (int, np.integer)):
            nearby_indices = [int(nearby_indices)]
        else:
            nearby_indices = [int(idx) for idx in nearby_indices]

        candidates = []
        min_dist_global = float("inf")

        for idx in nearby_indices:
            e = self.edges[idx]
            dist = float(e["geom"].distance(p))
            if dist < min_dist_global:
                min_dist_global = dist

            if dist <= effective_r:
                proj_dist = e["geom"].project(p)
                proj_pt = e["geom"].interpolate(proj_dist)
                candidates.append(CausalRoadCandidate(
                    edge_id=e["edge_id"],
                    u=e["u"],
                    v=e["v"],
                    k=e["k"],
                    proj_x=float(proj_pt.x),
                    proj_y=float(proj_pt.y),
                    dist=dist,
                    azimuth=e["azimuth"],
                ))

        if len(candidates) > 8:
            candidates.sort(key=lambda c: c.dist)
            candidates = candidates[:8]

        return candidates, min_dist_global

    def log_emission_prob(self, cand: CausalRoadCandidate) -> float:
        """Log emission probability under Gaussian distance: ln P(z | c) = -0.5 * (d / sigma_z)^2."""
        d = cand.dist
        norm_const = -0.5 * np.log(2.0 * np.pi * (self.sigma_z ** 2))
        return float(norm_const - (d ** 2) / (2.0 * (self.sigma_z ** 2)))

    def log_transition_prob(
        self,
        prev_c: CausalRoadCandidate,
        curr_c: CausalRoadCandidate,
        delta_est: float,
        motion_heading: Optional[float],
    ) -> float:
        """Log transition probability: exponential travel distance penalty + heading alignment."""
        delta_road = float(np.hypot(curr_c.proj_x - prev_c.proj_x, curr_c.proj_y - prev_c.proj_y))
        dist_diff = abs(delta_road - delta_est)
        log_p_dist = -dist_diff / self.beta

        log_p_heading = 0.0
        if motion_heading is not None and delta_est > 0.5:
            angle_diff = abs(curr_c.azimuth - motion_heading)
            angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi
            cos_align = max(float(np.cos(angle_diff)), 0.05)
            log_p_heading = float(np.log(cos_align))

        return float(log_p_dist + log_p_heading)

    def step(
        self,
        est_x: float,
        est_y: float,
        curr_t: Optional[float] = None,
        motion_heading: Optional[float] = None,
    ) -> CausalMatchOutput:
        """Execute a single online causal map-matching step at timestamp t."""
        if curr_t is None:
            curr_t = self.step_count * self.dt

        candidates, d_min = self.find_candidates(est_x, est_y)

        # -------------------------------------------------------------
        # OUT-OF-CORRIDOR REJECTION
        # -------------------------------------------------------------
        if not candidates or d_min > self.max_distance_gate:
            self.prev_pos = (est_x, est_y)
            self.prev_candidates = []
            self.prev_V = []
            self.step_count += 1
            return CausalMatchOutput(
                step=self.step_count - 1,
                time_s=curr_t,
                provisional_x=est_x,
                provisional_y=est_y,
                provisional_road_id=-1,
                finalized_x=est_x,
                finalized_y=est_y,
                finalized_road_id=-1,
                emission_prob=0.0,
                confidence=0.0,
                is_accepted=False,
                d_min=d_min,
            )

        # -------------------------------------------------------------
        # INITIAL STEP (t = 0 or resuming from off-corridor)
        # -------------------------------------------------------------
        if self.step_count == 0 or not self.prev_candidates or self.prev_pos is None:
            V_0 = [self.log_emission_prob(c) for c in candidates]
            best_idx = int(np.argmax(V_0))
            best_c = candidates[best_idx]

            p_emit = float(np.exp(-0.5 * (best_c.dist / self.sigma_z)**2))
            s_head = 1.0
            if motion_heading is not None:
                h_diff = (best_c.azimuth - motion_heading + np.pi) % (2.0 * np.pi) - np.pi
                s_head = max(0.0, float(np.cos(h_diff)))
            confidence = float(0.6 * p_emit + 0.4 * s_head)

            # Confidence gate check
            if confidence < self.min_road_confidence or best_c.dist > self.max_distance_gate:
                self.prev_pos = (est_x, est_y)
                self.prev_candidates = []
                self.prev_V = []
                self.step_count += 1
                return CausalMatchOutput(
                    step=self.step_count - 1,
                    time_s=curr_t,
                    provisional_x=est_x,
                    provisional_y=est_y,
                    provisional_road_id=-1,
                    finalized_x=est_x,
                    finalized_y=est_y,
                    finalized_road_id=-1,
                    emission_prob=p_emit,
                    confidence=confidence,
                    is_accepted=False,
                    d_min=d_min,
                )

            self.prev_candidates = candidates
            self.prev_V = V_0
            self.prev_pos = (est_x, est_y)
            self.history_candidates.append(candidates)
            self.history_backpointers.append([-1] * len(candidates))
            self.history_provisional.append((best_c.proj_x, best_c.proj_y, best_c.edge_id))
            self.step_count += 1

            return CausalMatchOutput(
                step=self.step_count - 1,
                time_s=curr_t,
                provisional_x=best_c.proj_x,
                provisional_y=best_c.proj_y,
                provisional_road_id=best_c.edge_id,
                finalized_x=best_c.proj_x,
                finalized_y=best_c.proj_y,
                finalized_road_id=best_c.edge_id,
                emission_prob=p_emit,
                confidence=confidence,
                is_accepted=True,
                d_min=best_c.dist,
            )

        # -------------------------------------------------------------
        # FORWARD TRELLIS DYNAMIC PROGRAMMING RECURSION
        # -------------------------------------------------------------
        dx = est_x - self.prev_pos[0]
        dy = est_y - self.prev_pos[1]
        delta_est = float(np.hypot(dx, dy))
        if motion_heading is None and delta_est > 0.2:
            motion_heading = float(np.arctan2(dy, dx))

        V_t = []
        bp_t = []

        for j, c_j in enumerate(candidates):
            log_emit = self.log_emission_prob(c_j)
            best_prob = -float("inf")
            best_prev = 0

            for i, c_i in enumerate(self.prev_candidates):
                log_trans = self.log_transition_prob(c_i, c_j, delta_est, motion_heading)
                prob = self.prev_V[i] + log_trans + log_emit
                if prob > best_prob:
                    best_prob = prob
                    best_prev = i

            V_t.append(best_prob)
            bp_t.append(best_prev)

        best_curr_idx = int(np.argmax(V_t))
        prov_c = candidates[best_curr_idx]

        # -------------------------------------------------------------
        # 4-FACTOR ROAD CONFIDENCE SCORE
        # 1. Candidate distance: Gaussian emission decay
        # 2. Heading consistency: Cosine alignment with motion heading
        # 3. Transition consistency: Distance traveled on road vs filter displacement
        # 4. Road continuity: Edge continuity with previous match
        # -------------------------------------------------------------
        s_dist = float(np.exp(-0.5 * (prov_c.dist / self.sigma_z) ** 2))
        
        s_head = 1.0
        if motion_heading is not None:
            h_diff = (prov_c.azimuth - motion_heading + np.pi) % (2.0 * np.pi) - np.pi
            s_head = max(0.0, float(np.cos(h_diff)))
            
        s_trans = 1.0
        s_cont = 0.5
        if self.prev_candidates and bp_t:
            prev_best_c = self.prev_candidates[bp_t[best_curr_idx]]
            delta_road = float(np.hypot(prov_c.proj_x - prev_best_c.proj_x, prov_c.proj_y - prev_best_c.proj_y))
            dist_diff = abs(delta_road - delta_est)
            s_trans = float(np.exp(-dist_diff / self.beta))
            
            # Road continuity
            if prov_c.edge_id == prev_best_c.edge_id:
                s_cont = 1.0
            elif prov_c.u == prev_best_c.v or prov_c.v == prev_best_c.u:
                s_cont = 0.85
            elif self.graph.has_edge(prev_best_c.u, prov_c.v) or self.graph.has_edge(prev_best_c.v, prov_c.u):
                s_cont = 0.70
            else:
                s_cont = 0.30

        confidence = float(0.35 * s_dist + 0.25 * s_head + 0.25 * s_trans + 0.15 * s_cont)

        # Rejection check on best candidate
        if confidence < self.min_road_confidence or prov_c.dist > self.max_distance_gate:
            self.prev_pos = (est_x, est_y)
            self.step_count += 1
            return CausalMatchOutput(
                step=self.step_count - 1,
                time_s=curr_t,
                provisional_x=est_x,
                provisional_y=est_y,
                provisional_road_id=-1,
                finalized_x=est_x,
                finalized_y=est_y,
                finalized_road_id=-1,
                emission_prob=s_dist,
                confidence=confidence,
                is_accepted=False,
                d_min=prov_c.dist,
            )

        # Update trellis running state
        self.prev_candidates = candidates
        self.prev_V = V_t
        self.prev_pos = (est_x, est_y)
        self.history_candidates.append(candidates)
        self.history_backpointers.append(bp_t)
        self.history_provisional.append((prov_c.proj_x, prov_c.proj_y, prov_c.edge_id))

        # Fixed-lag finalized match
        if len(self.history_backpointers) > self.lag_steps and self.lag_steps > 0:
            curr_idx = best_curr_idx
            for s in range(len(self.history_backpointers) - 1, len(self.history_backpointers) - 1 - self.lag_steps, -1):
                if 0 <= curr_idx < len(self.history_backpointers[s]):
                    curr_idx = self.history_backpointers[s][curr_idx]
                else:
                    curr_idx = 0
            target_hist_idx = len(self.history_candidates) - 1 - self.lag_steps
            if 0 <= target_hist_idx < len(self.history_candidates) and 0 <= curr_idx < len(self.history_candidates[target_hist_idx]):
                final_c = self.history_candidates[target_hist_idx][curr_idx]
                final_x = final_c.proj_x
                final_y = final_c.proj_y
                final_road = final_c.edge_id
            else:
                final_x = prov_c.proj_x
                final_y = prov_c.proj_y
                final_road = prov_c.edge_id
        else:
            final_x = prov_c.proj_x
            final_y = prov_c.proj_y
            final_road = prov_c.edge_id

        self.step_count += 1

        return CausalMatchOutput(
            step=self.step_count - 1,
            time_s=curr_t,
            provisional_x=prov_c.proj_x,
            provisional_y=prov_c.proj_y,
            provisional_road_id=prov_c.edge_id,
            finalized_x=final_x,
            finalized_y=final_y,
            finalized_road_id=final_road,
            emission_prob=s_dist,
            confidence=confidence,
            is_accepted=True,
            d_min=prov_c.dist,
        )

    def match_trajectory_causal(
        self,
        trajectory: np.ndarray,
        headings: Optional[np.ndarray] = None,
        use_finalized: bool = False,
    ) -> np.ndarray:
        """Process an entire trajectory strictly sequentially / causally."""
        self.reset()
        T = len(trajectory)
        out = np.zeros((T, 2), dtype=np.float64)

        for t in range(T):
            h = float(headings[t]) if headings is not None else None
            res = self.step(float(trajectory[t, 0]), float(trajectory[t, 1]), motion_heading=h)
            if use_finalized:
                out[t, 0] = res.finalized_x
                out[t, 1] = res.finalized_y
            else:
                out[t, 0] = res.provisional_x
                out[t, 1] = res.provisional_y

        return out

    def match_trajectory_causal_with_stats(
        self,
        trajectory: np.ndarray,
        headings: Optional[np.ndarray] = None,
        use_finalized: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process trajectory causally and return both matched coordinates and gating statistics."""
        self.reset()
        T = len(trajectory)
        out = np.zeros((T, 2), dtype=np.float64)
        accepted_count = 0
        d_mins = []

        for t in range(T):
            h = float(headings[t]) if headings is not None else None
            res = self.step(float(trajectory[t, 0]), float(trajectory[t, 1]), motion_heading=h)
            if use_finalized:
                out[t, 0] = res.finalized_x
                out[t, 1] = res.finalized_y
            else:
                out[t, 0] = res.provisional_x
                out[t, 1] = res.provisional_y

            if res.is_accepted:
                accepted_count += 1
            d_mins.append(res.d_min)

        stats = {
            "total_steps": T,
            "accepted_steps": accepted_count,
            "rejected_steps": T - accepted_count,
            "rejection_rate_pct": (T - accepted_count) / T * 100.0 if T > 0 else 0.0,
            "mean_d_min": float(np.mean(d_mins)) if d_mins else 0.0,
            "max_d_min": float(np.max(d_mins)) if d_mins else 0.0,
        }
        return out, stats
