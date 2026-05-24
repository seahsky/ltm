"""
Frontier-based subgoal proposer.

Maintains a 2-D top-down occupancy grid that is updated each step from depth
+ agent pose, then extracts frontier cells (free cells adjacent to unknown
cells) and clusters them into K subgoal candidates per decision step.

Decision cadence: every N steps (default 10), or when the agent has been
"stuck" (low position delta over the last few steps). Between decisions the
runner just keeps executing the previously-chosen subgoal via a greedy step
controller (move toward the candidate; turn if blocked).

Out of scope for this slice:
- learned planner / VLM planner
- proper SLAM (we just splat depth as cones in the agent frame)
- collision-aware path planning (we use straight-line bearing)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Action ids must match the order in habitat_env._ACTION_NAMES.
ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_TURN_LEFT = 2
ACTION_TURN_RIGHT = 3


@dataclass
class FrontierCandidate:
    """One candidate subgoal proposed to the planner."""
    candidate_id: int
    world_xy: np.ndarray         # (2,) target xy in world frame, meters
    grid_rc: Tuple[int, int]     # row/col on the occupancy grid
    distance_m: float            # straight-line distance from agent
    bearing_rad: float           # heading delta (relative to agent yaw)
    cluster_size: int            # how many frontier cells voted for this
    raw_score: float             # planner's intrinsic score (higher = better)
    source: str = "planner"      # "planner" (frontier cluster) or "memory" (LTM-injected)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Occupancy grid
# ----------------------------------------------------------------------


# Cell states. Using uint8 for compactness.
CELL_UNKNOWN = 0
CELL_FREE = 1
CELL_OCCUPIED = 2


@dataclass
class OccupancyGrid:
    """Top-down occupancy grid in the world XZ plane (Habitat y is up)."""

    resolution_m: float = 0.1   # meters per cell
    size_m: float = 20.0        # square side in meters
    origin_xy: Tuple[float, float] = (0.0, 0.0)  # world coord of grid (0,0)
    grid: np.ndarray = field(init=False)

    def __post_init__(self):
        n = int(self.size_m / self.resolution_m)
        self.grid = np.full((n, n), CELL_UNKNOWN, dtype=np.uint8)

    @property
    def n(self) -> int:
        return self.grid.shape[0]

    def world_to_grid(self, x: float, z: float) -> Tuple[int, int]:
        ox, oz = self.origin_xy
        c = int((x - ox) / self.resolution_m)
        r = int((z - oz) / self.resolution_m)
        return r, c

    def grid_to_world(self, r: int, c: int) -> Tuple[float, float]:
        ox, oz = self.origin_xy
        x = ox + (c + 0.5) * self.resolution_m
        z = oz + (r + 0.5) * self.resolution_m
        return x, z

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.n and 0 <= c < self.n

    def mark(self, r: int, c: int, state: int):
        if self.in_bounds(r, c):
            self.grid[r, c] = state


# ----------------------------------------------------------------------
# Frontier planner
# ----------------------------------------------------------------------


class FrontierPlanner:
    """Generate K frontier subgoal candidates per decision step.

    Args:
        decision_period: emit a fresh candidate set every N steps.
        n_candidates: K, max candidates per decision.
        stuck_radius_m: position delta below which we consider the agent stuck.
        stuck_window: how many steps we look back to detect stuck.
        forward_fov_deg: depth splat field of view (Habitat default ~79 hfov).
        max_depth_m: max range of depth ray-casting.
        grid_size_m: side length of the occupancy grid.
        grid_res_m: cell size in meters.
    """

    def __init__(
        self,
        decision_period: int = 10,
        n_candidates: int = 4,
        stuck_radius_m: float = 0.1,
        stuck_window: int = 8,
        forward_fov_deg: float = 79.0,
        max_depth_m: float = 5.0,
        grid_size_m: float = 20.0,
        grid_res_m: float = 0.1,
    ):
        self.decision_period = decision_period
        self.n_candidates = n_candidates
        self.stuck_radius_m = stuck_radius_m
        self.stuck_window = stuck_window
        self.forward_fov_deg = forward_fov_deg
        self.max_depth_m = max_depth_m

        self.grid = OccupancyGrid(
            resolution_m=grid_res_m,
            size_m=grid_size_m,
            origin_xy=(-grid_size_m / 2.0, -grid_size_m / 2.0),
        )

        self._step_count = 0
        self._pos_history: List[np.ndarray] = []
        self._candidate_counter = 0
        self._last_action: Optional[int] = None
        self._escape_toggle: bool = False
        self._force_replan: bool = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def reset(self):
        n = self.grid.n
        self.grid.grid = np.full((n, n), CELL_UNKNOWN, dtype=np.uint8)
        self._step_count = 0
        self._pos_history = []
        self._candidate_counter = 0
        self._last_action = None
        self._escape_toggle = False
        self._force_replan = False

    def update(self, depth: np.ndarray, agent_pos: np.ndarray, agent_yaw: float):
        """Splat depth into the grid. Cheap raycast: for each column take the
        closest depth, mark cells along the ray as FREE up to it, and the
        endpoint as OCCUPIED."""
        self._step_count += 1
        self._pos_history.append(np.asarray(agent_pos, dtype=np.float32))
        if len(self._pos_history) > max(self.stuck_window * 2, 32):
            self._pos_history.pop(0)

        if depth is None or depth.size == 0:
            return

        # Use a single horizontal scanline (middle row) to keep this CPU-cheap.
        h, w = depth.shape[-2], depth.shape[-1]
        scan = depth[h // 2] if depth.ndim == 2 else depth[h // 2, :]
        if scan.ndim == 2:  # safety, in case depth was (H, W, 1)
            scan = scan[:, 0]

        # Map column index → bearing in [-fov/2, fov/2] relative to agent yaw.
        fov = math.radians(self.forward_fov_deg)
        bearings = np.linspace(-fov / 2.0, fov / 2.0, num=w)
        # Habitat yaw=0 looks along -z by default; +x is to the right.
        # We model agent forward as the unit vector (sin(yaw), cos(yaw)) in (x, z).
        # Each ray bearing rotates forward by `b`.
        ax, az = float(agent_pos[0]), float(agent_pos[2])
        for col in range(0, w, max(1, w // 64)):  # subsample columns
            d = float(scan[col])
            if not np.isfinite(d) or d <= 0.05:
                continue
            d = min(d, self.max_depth_m)
            theta = agent_yaw + bearings[col]
            # March along the ray, marking FREE up to (d - res), OCCUPIED at d.
            steps = int(d / self.grid.resolution_m)
            for s in range(steps):
                rr = (s + 0.5) * self.grid.resolution_m
                x = ax + math.sin(theta) * rr
                z = az + math.cos(theta) * rr
                r, c = self.grid.world_to_grid(x, z)
                if self.grid.in_bounds(r, c) and self.grid.grid[r, c] != CELL_OCCUPIED:
                    self.grid.mark(r, c, CELL_FREE)
            # Endpoint: occupied.
            x = ax + math.sin(theta) * d
            z = az + math.cos(theta) * d
            r, c = self.grid.world_to_grid(x, z)
            if self.grid.in_bounds(r, c):
                self.grid.mark(r, c, CELL_OCCUPIED)

    def is_decision_step(self) -> bool:
        if self._step_count == 0:
            return True
        if self._force_replan:
            self._force_replan = False
            return True
        if self._step_count % self.decision_period == 0:
            return True
        return self._is_stuck()

    def _is_stuck(self) -> bool:
        if len(self._pos_history) < self.stuck_window:
            return False
        recent = np.stack(self._pos_history[-self.stuck_window:], axis=0)
        delta = np.linalg.norm(recent.max(axis=0) - recent.min(axis=0))
        return float(delta) < self.stuck_radius_m

    def propose(self, agent_pos: np.ndarray, agent_yaw: float) -> List[FrontierCandidate]:
        """Return up to K frontier candidates."""
        cells = self._extract_frontier_cells()
        if len(cells) == 0:
            # Fallback: emit a random-walk candidate ahead of the agent.
            return [self._random_walk_candidate(agent_pos, agent_yaw)]

        clusters = self._cluster_cells(cells, max_clusters=self.n_candidates)
        candidates: List[FrontierCandidate] = []
        ax, az = float(agent_pos[0]), float(agent_pos[2])
        for cluster_cells in clusters:
            arr = np.asarray(cluster_cells, dtype=np.float32)  # (M, 2) (r, c)
            r_med, c_med = arr.mean(axis=0)
            x, z = self.grid.grid_to_world(int(r_med), int(c_med))
            dx, dz = x - ax, z - az
            dist = math.hypot(dx, dz)
            bearing_world = math.atan2(dx, dz)
            bearing_rel = _wrap_pi(bearing_world - agent_yaw)
            # Score: prefer larger clusters and moderate distances (1-4 m).
            size_score = math.tanh(len(cluster_cells) / 10.0)
            dist_score = math.exp(-((dist - 2.5) ** 2) / 4.0)
            raw_score = 0.6 * size_score + 0.4 * dist_score
            self._candidate_counter += 1
            candidates.append(
                FrontierCandidate(
                    candidate_id=self._candidate_counter,
                    world_xy=np.array([x, z], dtype=np.float32),
                    grid_rc=(int(r_med), int(c_med)),
                    distance_m=dist,
                    bearing_rad=bearing_rel,
                    cluster_size=len(cluster_cells),
                    raw_score=raw_score,
                )
            )
        # Sort by intrinsic score desc.
        candidates.sort(key=lambda c: c.raw_score, reverse=True)
        return candidates[: self.n_candidates]

    def step_controller(self, candidate: FrontierCandidate, agent_yaw: float) -> int:
        """Convert a chosen candidate into a single discrete action.

        Tiny greedy controller: turn toward the candidate's bearing if we're
        more than ~15° off, else move forward. The runner re-plans every
        ``decision_period`` steps regardless, so this only needs to be
        roughly correct.

        Collision-escape: if we just told the agent to move FORWARD and the
        last 3 logged positions barely moved (<0.1 m bbox diagonal), the
        agent is stalled against geometry. Override with an alternating
        TURN so the next planner tick sees a different bearing.
        """
        bearing = candidate.bearing_rad
        deg15 = math.radians(15.0)
        if bearing > deg15:
            action = ACTION_TURN_LEFT
        elif bearing < -deg15:
            action = ACTION_TURN_RIGHT
        else:
            action = ACTION_FORWARD

        if (
            action == ACTION_FORWARD
            and self._last_action == ACTION_FORWARD
            and len(self._pos_history) >= 3
        ):
            recent = np.stack(self._pos_history[-3:], axis=0)
            bbox_diag = float(
                np.linalg.norm(recent.max(axis=0) - recent.min(axis=0))
            )
            if bbox_diag < 0.1:
                action = ACTION_TURN_LEFT if self._escape_toggle else ACTION_TURN_RIGHT
                self._escape_toggle = not self._escape_toggle
                # Force a fresh LLM proposal on the next iteration so the
                # runner's bearing-recompute can't immediately re-align the
                # candidate and undo this TURN. Without this, the agent
                # oscillates ±30° forever (see Phase 3 smoke trace).
                self._force_replan = True

        self._last_action = action
        return action

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _extract_frontier_cells(self) -> List[Tuple[int, int]]:
        g = self.grid.grid
        n = self.grid.n
        # Frontier = FREE cell with at least one UNKNOWN 4-neighbor.
        free_mask = g == CELL_FREE
        unk_up = np.zeros_like(free_mask)
        unk_up[1:, :] = g[:-1, :] == CELL_UNKNOWN
        unk_dn = np.zeros_like(free_mask)
        unk_dn[:-1, :] = g[1:, :] == CELL_UNKNOWN
        unk_lt = np.zeros_like(free_mask)
        unk_lt[:, 1:] = g[:, :-1] == CELL_UNKNOWN
        unk_rt = np.zeros_like(free_mask)
        unk_rt[:, :-1] = g[:, 1:] == CELL_UNKNOWN
        frontier = free_mask & (unk_up | unk_dn | unk_lt | unk_rt)
        rs, cs = np.where(frontier)
        return list(zip(rs.tolist(), cs.tolist()))

    @staticmethod
    def _cluster_cells(cells: List[Tuple[int, int]], max_clusters: int) -> List[List[Tuple[int, int]]]:
        """Greedy spatial clustering — group cells within a small radius."""
        if not cells:
            return []
        unvisited = set(cells)
        clusters: List[List[Tuple[int, int]]] = []
        radius_sq = 9  # ~3 cells

        while unvisited and len(clusters) < max_clusters * 4:
            seed = next(iter(unvisited))
            unvisited.remove(seed)
            stack = [seed]
            comp: List[Tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                # Find unvisited neighbours within radius.
                hits = [
                    (rr, cc) for (rr, cc) in list(unvisited)
                    if (rr - r) ** 2 + (cc - c) ** 2 <= radius_sq
                ]
                for h in hits:
                    unvisited.discard(h)
                    stack.append(h)
            clusters.append(comp)

        clusters.sort(key=len, reverse=True)
        return clusters[:max_clusters]

    def _random_walk_candidate(self, agent_pos: np.ndarray, agent_yaw: float) -> FrontierCandidate:
        """Last-resort candidate when no frontier is visible."""
        ax, az = float(agent_pos[0]), float(agent_pos[2])
        target_dist = 1.5
        x = ax + math.sin(agent_yaw) * target_dist
        z = az + math.cos(agent_yaw) * target_dist
        r, c = self.grid.world_to_grid(x, z)
        self._candidate_counter += 1
        return FrontierCandidate(
            candidate_id=self._candidate_counter,
            world_xy=np.array([x, z], dtype=np.float32),
            grid_rc=(r, c),
            distance_m=target_dist,
            bearing_rad=0.0,
            cluster_size=0,
            raw_score=0.1,
            metadata={"fallback": "random_walk"},
        )

    # ------------------------------------------------------------------
    # diversity-aware propose (Run 4)
    # ------------------------------------------------------------------

    def propose_diverse(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        k: int = 3,
    ) -> List[FrontierCandidate]:
        """Return up to k directionally-diverse candidates.

        Built for the Run-4 injection path in ``EpisodeRunner``: a single
        "1.5 m forward" candidate gets killed by the 0.5 m de-dup against
        the LLM's matching forward pick, leaving zero frontier candidates
        in the pool. When the occupancy grid has no real frontier cells,
        emit a compass fan around the agent so at least the side picks
        survive de-dup. When real cells exist but fewer than k, top up
        with compass picks for the missing slots.
        """
        cands: List[FrontierCandidate] = list(self.propose(agent_pos, agent_yaw))
        # If propose() only had the random-walk fallback to offer, swap it
        # wholesale for a compass fan — the single forward pick is exactly
        # what we want to de-dup against.
        if len(cands) == 1 and cands[0].metadata.get("fallback") == "random_walk":
            return self._compass_fallback(agent_pos, agent_yaw, k=k)
        # Don't mix real-cluster candidates with compass top-ups: the
        # FrontierPhysicsScorer's bearing-alignment term would let
        # compass candidates (raw=0.7) outscore lower-rated real clusters
        # at off-axis bearings. Pure separation keeps the semantics
        # straightforward — real frontier info beats fallback compass.
        return cands[:k]

    def _compass_fallback(
        self,
        agent_pos: np.ndarray,
        agent_yaw: float,
        k: int = 3,
    ) -> List[FrontierCandidate]:
        """Emit k candidates at evenly-spaced angles around the agent at a
        fixed 1.5 m distance. Offset 0 is forward; for k=3 the offsets are
        ``[0, 2π/3, 4π/3]`` so two of the three picks lie behind the agent
        — useful for escaping start-wall stalls."""
        if k <= 0:
            return []
        out: List[FrontierCandidate] = []
        target_dist = 1.5
        ax, az = float(agent_pos[0]), float(agent_pos[2])
        offsets = [i * (2.0 * math.pi / max(1, k)) for i in range(k)]
        for off in offsets:
            theta = agent_yaw + off
            x = ax + math.sin(theta) * target_dist
            z = az + math.cos(theta) * target_dist
            r, c = self.grid.world_to_grid(x, z)
            self._candidate_counter += 1
            out.append(
                FrontierCandidate(
                    candidate_id=self._candidate_counter,
                    world_xy=np.array([x, z], dtype=np.float32),
                    grid_rc=(r, c),
                    distance_m=target_dist,
                    bearing_rad=_wrap_pi(off),
                    cluster_size=0,
                    # 0.7 chosen against FrontierPhysicsScorer's planner
                    # path (`0.5·raw + 0.3·bearing + 0.2·dist`): coherence
                    # ≈ 0.625 at bearing=2π/3 vs LLM-stub forward's 0.525
                    # (wins) and real-LLM forward's ~0.725 (loses).
                    # Compass beats a stuck stub-forward LLM but yields to
                    # a confident, parseable LLM ANSWER.
                    raw_score=0.7,
                    metadata={"fallback": "compass", "offset_rad": float(off)},
                )
            )
        return out


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
