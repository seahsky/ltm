"""The geometric frontier proposer: the four pieces of the old 1129 lines that survive.

ADR-0008 keeps depth-to-occupancy integration (``occupancy.py``), frontier-cell
extraction and clustering, geometric candidate scoring, and the compass fallback for
when no frontier exists. Everything else in ``frontier_planner.py`` is gone: grid A\\*
and the three steering fallbacks it needed (dead on the live path — the navmesh follower
does the steering), the semantic value head (ADR-0006's documented negative, four
independent non-lifts), and the grid-flood reachability filter, which
``reachability.py`` replaces with the navmesh. Same algorithm, same geometric scoring,
no capability added or removed.

**One thing does not carry: ``_random_walk_candidate``.** It emitted a single waypoint
1.5 m straight ahead when the grid had no frontier cells, and the live path never used
it — ``propose_diverse`` swapped it wholesale for the compass fan
(``frontier_planner.py:1014``), because a lone forward pick is exactly what the
de-duplication against another proposer's forward pick removes, leaving the pool empty.
With the LLM planner dropped there is no second proposer to collide with, but the fan is
still strictly better: it divides the whole circle, so some of its directions lie behind
the agent, which is what escapes a start-wall stall. So the no-frontier branch is the fan,
and the two-step dance collapses.

**Frontier candidates and compass candidates are never mixed**, carried deliberately
from ``propose_diverse``: the scorer's bearing term would let a fan pick at a convenient
bearing outscore a real cluster at an awkward one, and real frontier information should
beat a fallback. Whichever kind exists, the pool is all of one kind.

The proposer's *logic* is module-level and pure — extraction, clustering, scoring, the
occupancy ray scan. ``FrontierProposer`` is a thin holder for the state that genuinely
accumulates across an episode (the grid, the position history, the candidate counter),
so what a test wants to pin does not need an object at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from earshot.agent.config import PlannerConfig
from earshot.agent.occupancy import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    OccupancyGrid,
    bearing_rel,
    cell_counts,
    forward_xz,
    integrate_depth,
    new_grid,
    wrap_pi,
)
from earshot.types import Pose, Xyz

__all__ = [
    "SOURCE_FRONTIER",
    "SOURCE_COMPASS",
    "SOURCE_INVESTIGATE",
    "Candidate",
    "frontier_cells",
    "cluster_cells",
    "frontier_score",
    "compass_score",
    "ray_occupancy_fractions",
    "FrontierProposer",
]

# Where a candidate came from. The scorer dispatches on this, and it is the seam the
# follow-on memory effort adds to (ADR-0008: "memory plugs in here later") — a
# `SOURCE_MEMORY` proposer would emit the same type into the same pool. Strings rather
# than an enum because they are written to the audit record, where a `<Source.FRONTIER>`
# repr is not a value anyone can read back.
SOURCE_FRONTIER = "frontier"
SOURCE_COMPASS = "compass"
SOURCE_INVESTIGATE = "investigate"


@dataclass(frozen=True)
class Candidate:
    """One waypoint offered to the scorer.

    ``position`` carries the **agent's own y**, not a height of its own: a frontier is a
    grid cell over ``(x, z)`` and has no elevation, and the navmesh snap in
    ``reachability.py`` needs a 3D point. Taking the agent's y makes every candidate
    same-floor by construction, which is what stops ``snap_point`` landing on the storey
    below — and it is the same policy ADR-0010 states for the anomaly source.

    Frozen: the old ``FrontierCandidate`` was mutated in flight by the runner
    (``episode_runner.py:2377`` rewrote ``bearing_rad`` and ``distance_m`` on the
    candidate it was already steering to), so "the candidate the scorer chose" and "the
    candidate in hand" were the same object with different contents.
    """

    candidate_id: int
    position: Xyz
    source: str
    distance_m: float
    bearing_rad: float
    raw_score: float
    cluster_size: int = 0
    grid_rc: Optional[Tuple[int, int]] = None
    # Filled by `reachability.reachable_pool` once the navmesh has answered. `None`
    # means "not yet asked", which is a different state from "unreachable" — an
    # unreachable candidate is not in the pool at all.
    geodesic_m: Optional[float] = None


# ----------------------------------------------------------------------
# extraction, clustering, scoring — all pure
# ----------------------------------------------------------------------


def frontier_cells(grid: OccupancyGrid) -> List[Tuple[int, int]]:
    """Free cells with at least one unknown 4-neighbour: the edge of the known map."""
    cells = grid.cells
    free = cells == CELL_FREE
    unknown_neighbour = np.zeros_like(free)
    unknown_neighbour[1:, :] |= cells[:-1, :] == CELL_UNKNOWN
    unknown_neighbour[:-1, :] |= cells[1:, :] == CELL_UNKNOWN
    unknown_neighbour[:, 1:] |= cells[:, :-1] == CELL_UNKNOWN
    unknown_neighbour[:, :-1] |= cells[:, 1:] == CELL_UNKNOWN
    rows, cols = np.where(free & unknown_neighbour)
    return list(zip(rows.tolist(), cols.tolist()))


def cluster_cells(
    cells: Sequence[Tuple[int, int]], *, max_clusters: int, radius_cells: int
) -> List[List[Tuple[int, int]]]:
    """Greedy single-link clustering: group cells within ``radius_cells`` of each other.

    Returned largest-first and truncated to ``max_clusters``. The search itself is
    allowed to find more than that (``max_clusters * 4``) before stopping, because
    truncating the *search* would return whichever clusters happened to be seeded first
    rather than the biggest ones.
    """
    remaining = set(cells)
    radius_sq = int(radius_cells) ** 2
    clusters: List[List[Tuple[int, int]]] = []
    while remaining and len(clusters) < max(1, int(max_clusters)) * 4:
        stack = [remaining.pop()]
        component: List[Tuple[int, int]] = []
        while stack:
            row, col = stack.pop()
            component.append((row, col))
            near = [
                (r, c)
                for (r, c) in remaining
                if (r - row) ** 2 + (c - col) ** 2 <= radius_sq
            ]
            for cell in near:
                remaining.discard(cell)
                stack.append(cell)
        clusters.append(component)
    clusters.sort(key=len, reverse=True)
    return clusters[: max(1, int(max_clusters))]


def frontier_score(cluster_size: int, distance_m: float) -> float:
    """Prefer big clusters at moderate range. Carried verbatim.

    ``0.6 * tanh(size / 10) + 0.4 * exp(-(d - 2.5)^2 / 4)``: a wide opening is worth more
    than a sliver, and 2.5 m is far enough to be progress but near enough that the map
    in between is already observed.
    """
    size_term = math.tanh(float(cluster_size) / 10.0)
    dist_term = math.exp(-((float(distance_m) - 2.5) ** 2) / 4.0)
    return 0.6 * size_term + 0.4 * dist_term


def compass_score(frac_free: float, frac_occupied: float) -> float:
    """Occupancy-aware score for a fan direction, in ``[0, 1]``. Carried verbatim.

    ``clip(0.7 + 0.3 * frac_free - 0.5 * frac_occupied, 0, 1)``: all free is 1.0, all
    unknown is the 0.7 baseline, all occupied is 0.2. Without the occupancy term every
    fan direction scored identically and the pick fell to the sort's tie-break, which
    walked the agent into walls.
    """
    return float(
        np.clip(0.7 + 0.3 * float(frac_free) - 0.5 * float(frac_occupied), 0.0, 1.0)
    )


def ray_occupancy_fractions(
    grid: OccupancyGrid, origin: Xyz, yaw_rad: float, max_dist_m: float
) -> Tuple[float, float]:
    """``(frac_free, frac_occupied)`` of the in-bounds cells along one ray.

    Samples every cell width from just past the agent's own cell out to ``max_dist_m``,
    along :func:`occupancy.forward_xz` — the same direction convention the splat wrote
    the grid with, which is the only way the scan reads back what was observed.
    """
    n_samples = max(1, int(float(max_dist_m) / grid.resolution_m))
    dir_x, dir_z = forward_xz(yaw_rad)
    n_free = n_occupied = n_in_bounds = 0
    for i in range(1, n_samples + 1):
        reach = i * grid.resolution_m
        row, col = grid.world_to_grid(origin.x + dir_x * reach, origin.z + dir_z * reach)
        if not grid.in_bounds(row, col):
            continue
        n_in_bounds += 1
        state = int(grid.cells[row, col])
        if state == CELL_FREE:
            n_free += 1
        elif state == CELL_OCCUPIED:
            n_occupied += 1
    if n_in_bounds == 0:
        return (0.0, 0.0)
    return (n_free / n_in_bounds, n_occupied / n_in_bounds)


# ----------------------------------------------------------------------
# the state holder
# ----------------------------------------------------------------------


@dataclass
class FrontierProposer:
    """Per-episode exploration state: the map, the recent poses, the candidate counter.

    ``reset(pose)`` at every episode start — it re-centres the grid, which is not
    optional (see ``occupancy.new_grid``).
    """

    cfg: PlannerConfig = field(default_factory=PlannerConfig)
    grid: Optional[OccupancyGrid] = None
    n_steps: int = 0
    n_candidates_emitted: int = 0
    _positions: List[Tuple[float, float, float]] = field(default_factory=list)
    _replan_requested: bool = False
    _n_replan_scheduled: int = 0
    _n_replan_requested: int = 0
    _n_replan_stuck: int = 0

    # -- lifecycle -------------------------------------------------------

    def reset(self, pose: Pose) -> None:
        self.grid = new_grid(pose.position, self.cfg)
        self.n_steps = 0
        self.n_candidates_emitted = 0
        self._positions = []
        self._replan_requested = False
        self._n_replan_scheduled = 0
        self._n_replan_requested = 0
        self._n_replan_stuck = 0

    def _require_grid(self) -> OccupancyGrid:
        if self.grid is None:
            raise RuntimeError(
                "the proposer has no map: reset(pose) was never called, so the grid was "
                "never centred on the agent. A grid centred anywhere else reads out of "
                "bounds for every HM3D start and silently discards the whole episode's "
                "occupancy"
            )
        return self.grid

    # -- observing -------------------------------------------------------

    def observe(self, depth: object, pose: Pose) -> None:
        """Fold one step's depth frame and pose into the map."""
        grid = self._require_grid()
        self.n_steps += 1
        self._positions.append(pose.position.as_tuple())
        window = max(int(self.cfg.stuck_window) * 2, 32)
        if len(self._positions) > window:
            del self._positions[: len(self._positions) - window]
        if depth is not None:
            self.grid = integrate_depth(grid, depth, pose, self.cfg)

    # -- cadence ---------------------------------------------------------

    def request_replan(self) -> None:
        """Ask for a fresh pool on the next cadence check.

        The runner calls this when the waypoint it is steering to stopped being
        answerable — the follower found no route, or the controller changed the task. It
        is a request rather than a direct ``propose`` so every re-proposal is counted in
        one place; a force-replan loop and a stuck loop look identical in a trajectory
        and are different bugs (Run 6).
        """
        self._replan_requested = True

    def is_decision_step(self) -> bool:
        """Time for a fresh pool? Consumes a pending replan request."""
        if self.n_steps == 0:
            return True
        if self._replan_requested:
            self._replan_requested = False
            self._n_replan_requested += 1
            return True
        if self.n_steps % int(self.cfg.decision_period) == 0:
            self._n_replan_scheduled += 1
            return True
        if self.is_stuck():
            self._n_replan_stuck += 1
            return True
        return False

    def is_stuck(self) -> bool:
        """Has the agent moved less than ``stuck_radius_m`` over the recent window?"""
        window = int(self.cfg.stuck_window)
        if len(self._positions) < window:
            return False
        recent = np.asarray(self._positions[-window:], dtype=np.float64)
        spread = float(np.linalg.norm(recent.max(axis=0) - recent.min(axis=0)))
        return spread < float(self.cfg.stuck_radius_m)

    # -- proposing -------------------------------------------------------

    def propose(self, pose: Pose) -> List[Candidate]:
        """Up to ``n_candidates`` waypoints, highest intrinsic score first.

        Never empty: with no frontier cells the compass fan stands in, and the fan is
        unconditional. Reachability is **not** decided here — that is
        ``reachability.py``'s navmesh filter, and keeping the two apart is what makes
        ADR-0008's invariant checkable rather than assumed.
        """
        grid = self._require_grid()
        cells = frontier_cells(grid)
        if not cells:
            return self.compass_fan(pose)
        clusters = cluster_cells(
            cells,
            max_clusters=int(self.cfg.n_candidates),
            radius_cells=int(self.cfg.cluster_radius_cells),
        )
        candidates = [self._candidate_from_cluster(cluster, pose) for cluster in clusters]
        candidates.sort(key=lambda c: (-c.raw_score, c.candidate_id))
        # `max(1, ...)` for the same reason `compass_fan` has it: a config with
        # `n_candidates <= 0` would otherwise truncate a real pool to empty here and
        # break the invariant one layer before `reachability` gets to assert it.
        return candidates[: max(1, int(self.cfg.n_candidates))]

    def _candidate_from_cluster(
        self, cluster: Sequence[Tuple[int, int]], pose: Pose
    ) -> Candidate:
        grid = self._require_grid()
        rows_cols = np.asarray(cluster, dtype=np.float64)
        row, col = (int(v) for v in rows_cols.mean(axis=0))
        x, z = grid.grid_to_world(row, col)
        dx, dz = x - pose.position.x, z - pose.position.z
        distance = math.hypot(dx, dz)
        return self._emit(
            position=Xyz(x, pose.position.y, z),
            source=SOURCE_FRONTIER,
            distance_m=distance,
            bearing_rad=bearing_rel(pose.yaw_rad, dx, dz),
            raw_score=frontier_score(len(cluster), distance),
            cluster_size=len(cluster),
            grid_rc=(row, col),
        )

    def compass_fan(self, pose: Pose) -> List[Candidate]:
        """``n_candidates`` picks at evenly-spaced headings, at ``compass_dist_m``.

        Offset 0 is straight ahead and the rest divide the circle evenly, so at least one
        pick lies behind the agent (two of three at ``n_candidates=3``, one of four at the
        default) — which is the point, since the case this exists for is a map with no
        visible frontier, and that usually means the agent is facing a wall.

        **Public because the reachability filter needs it.** ``propose`` reaches for the
        fan when the map has no frontier *cells*, but the other empty case is a pool of
        real frontier candidates that the navmesh rejects wholesale — the
        occupancy-versus-navmesh disagreement ADR-0008's invariant exists for. The runner
        answers that by re-proposing the fan and filtering again, so the fan has to be
        callable on its own.
        """
        grid = self._require_grid()
        k = max(1, int(self.cfg.n_candidates))
        reach = float(self.cfg.compass_dist_m)
        scan = reach + float(self.cfg.compass_scan_extra_m)
        out: List[Candidate] = []
        for i in range(k):
            offset = i * (2.0 * math.pi / k)
            ray_yaw = pose.yaw_rad + offset
            dir_x, dir_z = forward_xz(ray_yaw)
            frac_free, frac_occupied = ray_occupancy_fractions(grid, pose.position, ray_yaw, scan)
            x = pose.position.x + dir_x * reach
            z = pose.position.z + dir_z * reach
            out.append(
                self._emit(
                    position=Xyz(x, pose.position.y, z),
                    source=SOURCE_COMPASS,
                    distance_m=reach,
                    bearing_rad=wrap_pi(offset),
                    raw_score=compass_score(frac_free, frac_occupied),
                    grid_rc=grid.world_to_grid(x, z),
                )
            )
        out.sort(key=lambda c: (-c.raw_score, c.candidate_id))
        return out

    def _emit(
        self,
        *,
        position: Xyz,
        source: str,
        distance_m: float,
        bearing_rad: float,
        raw_score: float,
        cluster_size: int = 0,
        grid_rc: Optional[Tuple[int, int]] = None,
    ) -> Candidate:
        """The one place a ``candidate_id`` is issued, so ids are unique per episode."""
        self.n_candidates_emitted += 1
        return Candidate(
            candidate_id=self.n_candidates_emitted,
            position=position,
            source=source,
            distance_m=distance_m,
            bearing_rad=bearing_rad,
            raw_score=raw_score,
            cluster_size=cluster_size,
            grid_rc=grid_rc,
        )

    # -- instrumentation -------------------------------------------------

    def stats(self) -> Dict[str, int]:
        """Per-episode census. Instrumentation only; nothing here gates behaviour.

        The three replan counters separate a force-replan loop from a stuck loop from a
        healthy scheduled cadence — indistinguishable in a trajectory, and different
        bugs (Run 6). The grid counts say whether the densification took at all.
        """
        counts = cell_counts(self._require_grid())
        counts.update(
            {
                "frontier_cells": len(frontier_cells(self._require_grid())),
                "replan_scheduled": self._n_replan_scheduled,
                "replan_requested": self._n_replan_requested,
                "replan_stuck": self._n_replan_stuck,
                "candidates_emitted": self.n_candidates_emitted,
            }
        )
        return counts
