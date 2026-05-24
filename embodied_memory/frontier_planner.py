"""
Frontier-based subgoal proposer.

Maintains a 2-D top-down occupancy grid that is updated each step from depth
+ agent pose, then extracts frontier cells (free cells adjacent to unknown
cells) and clusters them into K subgoal candidates per decision step.

Decision cadence: every N steps (default 10), or when the agent has been
"stuck" (low position delta over the last few steps). Between decisions the
runner keeps executing the previously-chosen subgoal via the step
controller, which runs A* over the occupancy grid to the subgoal and steers
toward a short-lookahead waypoint on that path (Run-6 collision-aware
control), so the agent routes around obstacles instead of wedging on the
straight-line bearing.

Out of scope for this slice:
- learned planner / VLM planner
- proper SLAM (we just splat depth as cones in the agent frame)
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
# A* grid path planning (Run-6: collision-aware step controller)
# ----------------------------------------------------------------------


def _inflate_occupied(grid_array: np.ndarray, radius_cells: int = 1) -> np.ndarray:
    """Boolean mask of OCCUPIED cells dilated by ``radius_cells`` (8-conn).

    The agent has physical extent (~0.18 m radius in Habitat) while the grid is
    0.1 m/cell, so an A* path that hugs an OCCUPIED cell still collides.
    Inflating the obstacle mask by one cell keeps the path ~0.1 m off walls.
    Pure-numpy dilation (OR of 8 shifted copies) so the faiss/habitat-free test
    harness loads this module without scipy.
    """
    occ = grid_array == CELL_OCCUPIED
    if radius_cells <= 0:
        return occ
    out = occ.copy()
    for _ in range(radius_cells):
        cur = out
        nxt = cur.copy()
        nxt[1:, :] |= cur[:-1, :]
        nxt[:-1, :] |= cur[1:, :]
        nxt[:, 1:] |= cur[:, :-1]
        nxt[:, :-1] |= cur[:, 1:]
        nxt[1:, 1:] |= cur[:-1, :-1]
        nxt[1:, :-1] |= cur[:-1, 1:]
        nxt[:-1, 1:] |= cur[1:, :-1]
        nxt[:-1, :-1] |= cur[1:, 1:]
        out = nxt
    return out


def _snap_to_free(
    blocked_mask: np.ndarray,
    rc: Tuple[int, int],
    max_radius: int = 5,
) -> Optional[Tuple[int, int]]:
    """Nearest non-blocked cell to ``rc`` within ``max_radius`` cells, or None.

    Frontier medians (and LTM-injected picks) can land on or next to a wall;
    after inflation that cell reads blocked and A* would refuse it as a goal.
    Snap to the closest passable cell (BFS rings, nearest-by-Euclidean within a
    ring) so the planner has a reachable target.
    """
    n_rows, n_cols = blocked_mask.shape
    r0, c0 = int(rc[0]), int(rc[1])
    if 0 <= r0 < n_rows and 0 <= c0 < n_cols and not blocked_mask[r0, c0]:
        return (r0, c0)
    for rad in range(1, max_radius + 1):
        best: Optional[Tuple[int, Tuple[int, int]]] = None
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                if max(abs(dr), abs(dc)) != rad:  # ring perimeter only
                    continue
                rr, cc = r0 + dr, c0 + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols and not blocked_mask[rr, cc]:
                    d2 = dr * dr + dc * dc
                    if best is None or d2 < best[0]:
                        best = (d2, (rr, cc))
        if best is not None:
            return best[1]
    return None


def _reachable_mask(
    grid_array: np.ndarray,
    start_rc: Tuple[int, int],
    inflate_radius_cells: int = 1,
) -> np.ndarray:
    """Boolean mask of cells reachable from ``start_rc`` over FREE+UNKNOWN with
    OCCUPIED inflated-blocked and the start force-cleared — identical
    traversability and no-corner-cut rules to :func:`astar` (Run-6.1).

    The Run-6 census showed A* returns no-path ~92% of steps: the chosen
    frontier sits outside the agent's connected free/unknown component (1-cell
    inflation over a noisy depth grid fragments the space). This flood lets the
    planner (a) keep only reachable frontiers and (b) pick a reachable sub-goal
    toward an unreachable frontier instead of straight-lining into the wall.
    """
    from collections import deque

    n_rows, n_cols = grid_array.shape
    reach = np.zeros((n_rows, n_cols), dtype=bool)
    sr, sc = int(start_rc[0]), int(start_rc[1])
    if not (0 <= sr < n_rows and 0 <= sc < n_cols):
        return reach
    blocked = _inflate_occupied(grid_array, inflate_radius_cells)
    blocked[sr, sc] = False
    reach[sr, sc] = True
    dq = deque([(sr, sc)])
    ortho = ((-1, 0), (1, 0), (0, -1), (0, 1))
    diag = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ortho:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                dq.append((nr, nc))
        for dr, dc in diag:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols) or reach[nr, nc] or blocked[nr, nc]:
                continue
            if blocked[r, nc] or blocked[nr, c]:
                continue  # no diagonal corner-cutting (matches astar)
            reach[nr, nc] = True
            dq.append((nr, nc))
    return reach


def _nearest_reachable(
    reach: np.ndarray, target_rc: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    """Reachable cell closest (Euclidean) to ``target_rc``, or None if the mask
    is empty. Used to pick a reachable sub-goal toward an unreachable frontier."""
    rs, cs = np.nonzero(reach)
    if rs.size == 0:
        return None
    d2 = (rs - int(target_rc[0])) ** 2 + (cs - int(target_rc[1])) ** 2
    i = int(np.argmin(d2))
    return (int(rs[i]), int(cs[i]))


def astar(
    grid_array: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    *,
    inflate_radius_cells: int = 1,
    unknown_cost: float = 1.5,
    allow_diagonal: bool = True,
    max_expansions: Optional[int] = None,
) -> Optional[List[Tuple[int, int]]]:
    """A* over the occupancy array; inclusive cell path ``[start, ..., goal]``.

    FREE + UNKNOWN cells are traversable, OCCUPIED (inflated) cells are blocked.
    8-connected with a no-corner-cutting guard (a diagonal step is legal only
    when both shared orthogonal neighbours are unblocked). Octile heuristic
    (admissible/consistent for unit/√2 step costs).

    UNKNOWN cells are passable but cost ``unknown_cost``× a FREE step, so the
    search prefers observed-free corridors yet still crosses unobserved space
    when that is the only route. Such a path self-corrects: a wrongly-optimistic
    UNKNOWN cell flips OCCUPIED on collision and the next per-step replan routes
    around it.

    The agent's own (start) cell is force-cleared in the blocked mask, so
    standing next to a wall — where inflation would otherwise seal the start —
    never strands the planner. Returns None if no path (including a fully
    blocked goal, or if ``max_expansions`` is reached first — bounds the rare
    case of a passable-but-trapped goal forcing a full-grid exhaustion).
    """
    import heapq

    n_rows, n_cols = grid_array.shape
    sr, sc = int(start_rc[0]), int(start_rc[1])
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    if not (0 <= sr < n_rows and 0 <= sc < n_cols):
        return None
    if not (0 <= gr < n_rows and 0 <= gc < n_cols):
        return None

    blocked = _inflate_occupied(grid_array, inflate_radius_cells)
    blocked[sr, sc] = False  # never self-block the agent's own cell
    if blocked[gr, gc]:
        return None
    if (sr, sc) == (gr, gc):
        return [(sr, sc)]

    sqrt2 = math.sqrt(2.0)
    if allow_diagonal:
        # (dr, dc, base_cost, orthogonal-neighbour guards for corner-cut)
        moves = [
            (-1, 0, 1.0, ()),
            (1, 0, 1.0, ()),
            (0, -1, 1.0, ()),
            (0, 1, 1.0, ()),
            (-1, -1, sqrt2, ((-1, 0), (0, -1))),
            (-1, 1, sqrt2, ((-1, 0), (0, 1))),
            (1, -1, sqrt2, ((1, 0), (0, -1))),
            (1, 1, sqrt2, ((1, 0), (0, 1))),
        ]
    else:
        moves = [(-1, 0, 1.0, ()), (1, 0, 1.0, ()), (0, -1, 1.0, ()), (0, 1, 1.0, ())]

    def _h(r: int, c: int) -> float:
        dr, dc = abs(r - gr), abs(c - gc)
        return (dr + dc) + (sqrt2 - 2.0) * min(dr, dc)

    def _cell_cost(r: int, c: int) -> float:
        return unknown_cost if grid_array[r, c] == CELL_UNKNOWN else 1.0

    open_heap: List[Tuple[float, int, Tuple[int, int]]] = []
    counter = 0
    heapq.heappush(open_heap, (_h(sr, sc), counter, (sr, sc)))
    g_score: Dict[Tuple[int, int], float] = {(sr, sc): 0.0}
    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
    closed = set()

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)
        if cur == (gr, gc):
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        if cur in closed:
            continue
        closed.add(cur)
        if max_expansions is not None and len(closed) > max_expansions:
            return None
        cr, cc = cur
        for dr, dc, base, guards in moves:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < n_rows and 0 <= nc < n_cols):
                continue
            if blocked[nr, nc]:
                continue
            if guards and any(blocked[cr + g[0], cc + g[1]] for g in guards):
                continue  # no diagonal corner-cutting
            tentative = g_score[cur] + base * _cell_cost(nr, nc)
            if tentative < g_score.get((nr, nc), float("inf")):
                came_from[(nr, nc)] = cur
                g_score[(nr, nc)] = tentative
                counter += 1
                heapq.heappush(open_heap, (tentative + _h(nr, nc), counter, (nr, nc)))
    return None


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
        camera_height_m: float = 0.88,
        obstacle_min_h: float = 0.3,
        lookahead_m: float = 0.4,
        inflate_radius_cells: int = 1,
        unknown_cost: float = 1.5,
        astar_max_expansions: int = 20000,
    ):
        self.decision_period = decision_period
        self.n_candidates = n_candidates
        self.stuck_radius_m = stuck_radius_m
        self.stuck_window = stuck_window
        self.forward_fov_deg = forward_fov_deg
        self.max_depth_m = max_depth_m
        # A* step controller (Run-6). lookahead_m: how far along the A* path the
        # steering waypoint sits (smooths bearing vs the jittery next cell).
        # inflate_radius_cells: obstacle dilation for agent radius (1 cell ≈
        # 0.1 m clearance). unknown_cost: UNKNOWN-cell penalty (>1 prefers
        # observed-free routes while keeping unexplored space passable).
        self.lookahead_m = lookahead_m
        self.inflate_radius_cells = inflate_radius_cells
        self.unknown_cost = unknown_cost
        # Bound A* worst case (a passable-but-trapped goal can otherwise force a
        # full-grid exhaustion ~190 ms); on the cap, fall back to straight-line.
        self.astar_max_expansions = astar_max_expansions
        # Height gate for the depth splat (Run-5 densification): a back-
        # projected endpoint counts as an obstacle only if it rises more than
        # ``obstacle_min_h`` above the floor. ``camera_height_m`` is the agent
        # eye height above the floor; ``_floor_y`` is fixed at episode start in
        # ``reset()`` (agent_y - camera_height_m).
        self.camera_height_m = camera_height_m
        self.obstacle_min_h = obstacle_min_h
        self._floor_y = -camera_height_m

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
        # Run-6 diagnostic counters (instrumentation only; never gate behavior).
        # Separate the replan-every-step driver (force-replan vs stuck vs
        # scheduled) and the A* outcome so a stable-target-but-stuck episode is
        # distinguishable from target-flipping. Zeroed per-episode in reset().
        self._stats = self._zero_stats()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_stats() -> Dict[str, int]:
        return {
            "replan_scheduled": 0,   # is_decision_step via step % decision_period
            "replan_forced": 0,      # is_decision_step via _force_replan
            "replan_stuck": 0,       # is_decision_step via _is_stuck()
            "astar_path": 0,              # A* reached the chosen frontier
            "astar_reachable_fallback": 0,  # frontier unreachable → A* to nearest reachable cell
            "astar_fallback": 0,          # boxed in → last-resort straight-line bearing
            "collision_escape": 0,   # FORWARD-stall override (alternating TURN)
        }

    def controller_stats(self) -> Dict[str, int]:
        """Per-episode controller census (Run-6 instrumentation). Distinguishes
        a force-replan loop (astar_fallback high) from a stuck loop
        (replan_stuck high) from genuine geometry stalls (collision_escape
        high). Action mix is counted in the runner, not here."""
        return dict(self._stats)

    def reset(self, agent_pos: Optional[np.ndarray] = None):
        """Clear the grid and per-episode state.

        If ``agent_pos`` is provided, re-center the grid origin so the
        agent starts at the grid's geometric center. This fixes a
        long-standing bug where the grid was hard-centered at world
        origin (0, 0) with a 20 m span, while HM3D agent starts are
        frequently 15–20 m away (e.g. z=-17.77 in scene wcojb4TFT35).
        Without recentering, every grid lookup is out-of-bounds and the
        occupancy data is silently lost.
        """
        n = self.grid.n
        self.grid.grid = np.full((n, n), CELL_UNKNOWN, dtype=np.uint8)
        self._step_count = 0
        self._pos_history = []
        self._candidate_counter = 0
        self._last_action = None
        self._escape_toggle = False
        self._force_replan = False
        self._stats = self._zero_stats()
        if agent_pos is not None:
            ax, az = float(agent_pos[0]), float(agent_pos[2])
            half = self.grid.size_m / 2.0
            self.grid.origin_xy = (ax - half, az - half)
            # Fix the floor reference for the height gate at the agent's start.
            self._floor_y = float(agent_pos[1]) - self.camera_height_m

    def update(self, depth: np.ndarray, agent_pos: np.ndarray, agent_yaw: float):
        """Splat depth into the occupancy grid via a multi-row per-pixel
        projection with a height gate (Run-5 densification).

        The previous implementation splatted a single middle-row scanline
        subsampled to 64 columns. At eye height that scanline mostly hit
        walls/furniture and missed floor openings/doorways, so too few FREE
        cells were carved and frontiers clustered against walls — the agent
        had no navigable subgoal and barely moved. We now back-project a
        subsampled grid of pixels (both axes), march FREE along each ray's
        ground range, and gate the endpoint on world height: low endpoints
        are floor (FREE, walkable — this is what fills doorways/openings),
        high endpoints are obstacles (OCCUPIED).

        Geometry assumes a square sensor (so vfov == hfov), planar Habitat
        depth, and pitch == 0 (the planner only emits yaw actions 0–3), so
        the pinhole intrinsics reduce to ``f_px = (w/2)/tan(hfov/2)`` and the
        world height of a pixel is ``world_h ≈ agent_y + Yc``.
        """
        self._step_count += 1
        self._pos_history.append(np.asarray(agent_pos, dtype=np.float32))
        if len(self._pos_history) > max(self.stuck_window * 2, 32):
            self._pos_history.pop(0)

        if depth is None or depth.size == 0:
            return
        if depth.ndim == 3 and depth.shape[-1] == 1:  # (H, W, 1) → (H, W)
            depth = depth[..., 0]
        h, w = depth.shape[-2], depth.shape[-1]

        # Pinhole intrinsics from horizontal FOV on a square sensor.
        hfov = math.radians(self.forward_fov_deg)
        f_px = (w / 2.0) / math.tan(hfov / 2.0)
        cx = w / 2.0
        cy = h / 2.0

        # Habitat yaw=0 looks along +z under our (sin θ, cos θ) forward model;
        # +x is to the right. agent_y feeds the height gate, _floor_y is fixed.
        ax, ay, az = float(agent_pos[0]), float(agent_pos[1]), float(agent_pos[2])
        floor_y = self._floor_y
        res = self.grid.resolution_m

        # Subsample both axes to ≤~1k pixels/step. Coarsen the steps if the
        # per-step time regresses (see Run-5 plan, Change 2).
        row_step = max(1, h // 28)
        col_step = max(1, w // 28)

        for row in range(0, h, row_step):
            depth_row = depth[row]
            yc_term = (cy - row) / f_px
            for col in range(0, w, col_step):
                d = float(depth_row[col])
                if not math.isfinite(d) or d <= 0.05:
                    continue
                d = min(d, self.max_depth_m)
                Xc = d * (col - cx) / f_px          # right offset (camera frame)
                Yc = d * yc_term                    # up offset (camera frame)
                g = math.hypot(Xc, d)               # ground range (Zc == d)
                if g < res:
                    continue
                theta = agent_yaw + math.atan2(Xc, d)
                sin_t = math.sin(theta)
                cos_t = math.cos(theta)
                # March FREE up to (g - res); occupied cells stay sticky.
                n_free = int((g - res) / res)
                for s in range(n_free):
                    rr = (s + 0.5) * res
                    x = ax + sin_t * rr
                    z = az + cos_t * rr
                    r, c = self.grid.world_to_grid(x, z)
                    if self.grid.in_bounds(r, c) and self.grid.grid[r, c] != CELL_OCCUPIED:
                        self.grid.mark(r, c, CELL_FREE)
                # Endpoint: gate on world height. Obstacles win over FREE so a
                # wall is never erased by a floor ray that overshoots it.
                world_h = ay + Yc
                x = ax + sin_t * g
                z = az + cos_t * g
                r, c = self.grid.world_to_grid(x, z)
                if not self.grid.in_bounds(r, c):
                    continue
                if (world_h - floor_y) > self.obstacle_min_h:
                    self.grid.mark(r, c, CELL_OCCUPIED)
                elif self.grid.grid[r, c] != CELL_OCCUPIED:
                    self.grid.mark(r, c, CELL_FREE)

    def is_decision_step(self) -> bool:
        if self._step_count == 0:
            return True
        if self._force_replan:
            self._force_replan = False
            self._stats["replan_forced"] += 1
            return True
        if self._step_count % self.decision_period == 0:
            self._stats["replan_scheduled"] += 1
            return True
        stuck = self._is_stuck()
        if stuck:
            self._stats["replan_stuck"] += 1
        return stuck

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
        # Reachability filter (Run-6.1): drop frontiers A* can't route to from
        # here so the rerank chooses among reachable options. One flood over the
        # same traversable grid A* uses. Keep all if none are reachable (don't
        # strand the rerank) — the step controller's reachable-fallback then
        # still steers toward the nearest reachable cell.
        start_rc = self.grid.world_to_grid(ax, az)
        reach = _reachable_mask(self.grid.grid, start_rc, self.inflate_radius_cells)
        blocked = _inflate_occupied(self.grid.grid, self.inflate_radius_cells)
        reachable = []
        for c in candidates:
            g = _snap_to_free(blocked, c.grid_rc, max_radius=5)
            if g is not None and reach[g[0], g[1]]:
                reachable.append(c)
        if reachable:
            candidates = reachable

        # Sort by intrinsic score desc.
        candidates.sort(key=lambda c: c.raw_score, reverse=True)
        return candidates[: self.n_candidates]

    def step_controller(
        self,
        candidate: FrontierCandidate,
        agent_pos: np.ndarray,
        agent_yaw: float,
    ) -> int:
        """Convert a chosen candidate into a single discrete action via
        collision-aware A* over the occupancy grid (Run-6).

        Earlier slices steered by straight-line bearing to the candidate; on
        HM3D the straight line to a reachable frontier routinely crosses
        geometry, so ``move_forward`` collided and the agent wedged at the
        start wall (Phase-2 Runs 1–5, oracle proved the env navigable). We now
        run A* from the agent's cell to the candidate's cell over the grid
        (FREE + UNKNOWN traversable, OCCUPIED inflated-and-blocked) and steer
        toward a short-lookahead waypoint on that path — the agent routes
        *around* obstacles, like the navmesh oracle does.

        Falls back to the straight-line bearing (and forces a replan) when A*
        finds no path, so a transiently unreachable candidate degrades to the
        old behaviour rather than freezing.

        Collision-escape is kept as a safety net: if we just told the agent to
        move FORWARD and the last 3 logged positions barely moved (<0.1 m bbox
        diagonal), the agent is stalled against geometry the grid hasn't
        captured (sub-cell, or grid-vs-navmesh disagreement). Override with an
        alternating TURN and force a replan.
        """
        ax, az = float(agent_pos[0]), float(agent_pos[2])
        action = self._astar_action(candidate, ax, az, agent_yaw)

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
                # Force a fresh proposal next tick so the runner's bearing-
                # recompute can't re-align the candidate and undo this TURN.
                self._force_replan = True
                self._stats["collision_escape"] += 1

        self._last_action = action
        return action

    def _bearing_to_action(self, bearing_rel: float) -> int:
        """Greedy ±15° rule: turn toward the bearing if off-axis, else FORWARD."""
        deg15 = math.radians(15.0)
        if bearing_rel > deg15:
            return ACTION_TURN_LEFT
        if bearing_rel < -deg15:
            return ACTION_TURN_RIGHT
        return ACTION_FORWARD

    def _astar_action(
        self,
        candidate: FrontierCandidate,
        ax: float,
        az: float,
        agent_yaw: float,
    ) -> int:
        """Action toward the next A* waypoint.

        Routes from the agent cell to the candidate cell over the occupancy
        grid and steers toward a waypoint ~``lookahead_m`` along that path. When
        the chosen frontier is unreachable on the inflated grid (no goal-snap or
        A* no-path — the Run-6 census found this on ~92% of steps), route
        instead to the nearest *reachable* cell toward it (Run-6.1): collision-
        aware progress into reachable space, rather than straight-lining into
        the wall the frontier sits behind. Only when the agent is genuinely
        boxed into a single cell does it degrade to the old straight-line
        bearing + force-replan.
        """
        start_rc = self.grid.world_to_grid(ax, az)
        # Snap the goal off any wall it landed on so A* has a reachable target.
        blocked = _inflate_occupied(self.grid.grid, self.inflate_radius_cells)
        goal_rc = _snap_to_free(blocked, candidate.grid_rc, max_radius=5)
        path = None
        if goal_rc is not None:
            path = astar(
                self.grid.grid,
                start_rc,
                goal_rc,
                inflate_radius_cells=self.inflate_radius_cells,
                unknown_cost=self.unknown_cost,
                max_expansions=self.astar_max_expansions,
            )
        if path and len(path) >= 2:
            self._stats["astar_path"] += 1
            return self._steer_along(path, ax, az, agent_yaw)
        # Chosen frontier unreachable → head to the nearest reachable cell.
        return self._reachable_fallback_action(candidate, start_rc, ax, az, agent_yaw)

    def _steer_along(
        self, path: List[Tuple[int, int]], ax: float, az: float, agent_yaw: float
    ) -> int:
        """Steer toward a ~``lookahead_m`` waypoint on an A* path (clamped to
        the path end), via the greedy ±15° rule."""
        n_ahead = max(1, int(round(self.lookahead_m / self.grid.resolution_m)))
        wp = path[min(n_ahead, len(path) - 1)]
        wx, wz = self.grid.grid_to_world(wp[0], wp[1])
        bearing_rel = _wrap_pi(math.atan2(wx - ax, wz - az) - agent_yaw)
        return self._bearing_to_action(bearing_rel)

    def _reachable_fallback_action(
        self,
        candidate: FrontierCandidate,
        start_rc: Tuple[int, int],
        ax: float,
        az: float,
        agent_yaw: float,
    ) -> int:
        """Chosen frontier is unreachable: flood the reachable component and A*
        to the reachable cell nearest the frontier, steering there (Run-6.1).
        We do NOT force a replan — the sub-goal recedes toward the frontier as
        the agent advances and re-observes, opening the pocket — so the runner
        commits to this heading instead of re-reranking every step. Degrades to
        the old straight-line bearing only if the agent is boxed into one cell."""
        reach = _reachable_mask(self.grid.grid, start_rc, self.inflate_radius_cells)
        sub_rc = _nearest_reachable(reach, candidate.grid_rc)
        if sub_rc is not None and sub_rc != (int(start_rc[0]), int(start_rc[1])):
            path = astar(
                self.grid.grid,
                start_rc,
                sub_rc,
                inflate_radius_cells=self.inflate_radius_cells,
                unknown_cost=self.unknown_cost,
                max_expansions=self.astar_max_expansions,
            )
            if path and len(path) >= 2:
                self._stats["astar_reachable_fallback"] += 1
                return self._steer_along(path, ax, az, agent_yaw)
        return self._straight_line_fallback(candidate)

    def _straight_line_fallback(self, candidate: FrontierCandidate) -> int:
        """Last-resort straight-line bearing controller, used only when the
        agent is boxed into a single reachable cell. Forces a replan so the
        runner re-picks a candidate next tick instead of re-driving into the
        same wall."""
        self._force_replan = True
        self._stats["astar_fallback"] += 1
        return self._bearing_to_action(candidate.bearing_rad)

    def grid_stats(self) -> Dict[str, int]:
        """Occupancy-grid census for run instrumentation (Run-5).

        Returns four int counts: FREE / OCCUPIED / UNKNOWN cells (which sum to
        ``n * n``) plus ``frontier_cells`` (FREE cells with an UNKNOWN
        4-neighbor; a subset of FREE, not part of the sum). Makes a smoke run
        interpretable — if ``cells_free`` is still tiny the densification
        didn't take.
        """
        g = self.grid.grid
        free = int(np.count_nonzero(g == CELL_FREE))
        occupied = int(np.count_nonzero(g == CELL_OCCUPIED))
        unknown = int(np.count_nonzero(g == CELL_UNKNOWN))
        frontier = len(self._extract_frontier_cells())
        return {
            "cells_free": free,
            "cells_occupied": occupied,
            "cells_unknown": unknown,
            "frontier_cells": frontier,
        }

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
        — useful for escaping start-wall stalls.

        Run-5 smoke 6 fix: the raw_score is now occupancy-aware. Each
        compass ray samples the agent's 2 m FOV in the grid and scores
        on the FREE-vs-OCCUPIED ratio:

            raw_score = clip(0.7 + 0.3·frac_free − 0.5·frac_occupied, 0, 1)

        - All FREE: 1.0 (clearly winning over LLM real at 0.725)
        - All UNKNOWN: 0.7 (matches Run-4 smoke-3 baseline)
        - All OCCUPIED: 0.2 (loses to LLM stub at 0.1 by coherence)
        - Mix biased toward FREE → above baseline; biased toward
          OCCUPIED → below baseline. The rerank's stable-sort tie-break
          stops dominating when frac differs across ray directions.
        """
        if k <= 0:
            return []
        out: List[FrontierCandidate] = []
        target_dist = 1.5
        # Scan a bit past the waypoint so cells just beyond the target also
        # count — gives a stronger signal than scanning only to the target.
        scan_dist = target_dist + 0.5
        ax, az = float(agent_pos[0]), float(agent_pos[2])
        offsets = [i * (2.0 * math.pi / max(1, k)) for i in range(k)]
        for off in offsets:
            theta = agent_yaw + off
            x = ax + math.sin(theta) * target_dist
            z = az + math.cos(theta) * target_dist
            r, c = self.grid.world_to_grid(x, z)
            frac_free, frac_occ = self._ray_occupancy_fractions(
                ax, az, theta, scan_dist
            )
            raw_score = float(
                np.clip(0.7 + 0.3 * frac_free - 0.5 * frac_occ, 0.0, 1.0)
            )
            self._candidate_counter += 1
            out.append(
                FrontierCandidate(
                    candidate_id=self._candidate_counter,
                    world_xy=np.array([x, z], dtype=np.float32),
                    grid_rc=(r, c),
                    distance_m=target_dist,
                    bearing_rad=_wrap_pi(off),
                    cluster_size=0,
                    raw_score=raw_score,
                    metadata={
                        "fallback": "compass",
                        "offset_rad": float(off),
                        "frac_free": float(frac_free),
                        "frac_occupied": float(frac_occ),
                    },
                )
            )
        return out

    def _ray_occupancy_fractions(
        self,
        ax: float,
        az: float,
        theta: float,
        max_dist: float,
    ) -> Tuple[float, float]:
        """Sample grid cells along the ray from (ax, az) at angle theta
        out to ``max_dist`` meters. Returns (frac_free, frac_occupied)
        across in-bounds samples. Skips cell 0 (agent's own position).

        Uses the same yaw convention as ``OccupancyGrid.update`` —
        ``(sin θ, cos θ)`` as the forward direction in (x, z) — so the
        scan is consistent with what's been splatted into the grid.
        """
        n_samples = max(1, int(max_dist / self.grid.resolution_m))
        n_free = 0
        n_occupied = 0
        n_in_bounds = 0
        for i in range(1, n_samples + 1):  # skip i=0 = agent's own cell
            rr = i * self.grid.resolution_m
            x = ax + math.sin(theta) * rr
            z = az + math.cos(theta) * rr
            r, c = self.grid.world_to_grid(x, z)
            if not self.grid.in_bounds(r, c):
                continue
            n_in_bounds += 1
            state = int(self.grid.grid[r, c])
            if state == CELL_FREE:
                n_free += 1
            elif state == CELL_OCCUPIED:
                n_occupied += 1
        if n_in_bounds == 0:
            return 0.0, 0.0
        return n_free / n_in_bounds, n_occupied / n_in_bounds


def _wrap_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
