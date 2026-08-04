"""ADR-0008's invariant: the pool is never empty, and every candidate in it is
navmesh-reachable and snapped.

Unconditional, and no longer behind ``REMEMBR_ANTITHRASH_SINGLEGOAL`` — the clean room
carries the behaviour the flag gated, not the flag, and the runs the default-OFF setting
existed to keep byte-identical are being deleted. What it protects is measured: the
depth-derived occupancy grid and the navmesh disagree, so an ungated proposer emits
waypoints the follower can never route to (``n_waypoint_unreachable`` 60-99 per episode,
``min_d2g`` stuck around 8 m).

**The navmesh decides, not a grid flood.** The old filter re-implemented traversability
over the inflated occupancy grid (``_reachable_mask`` / ``_snap_to_free``) — the same
disagreeing map, asked twice. Here ``snap_point`` and the geodesic query are the
simulator's own, injected as callables so this module has no idea habitat-sim exists
(ADR-0013) and the whole filter unit-tests on a Mac against two lambdas.

**There is no horizontal snap cap, and that is deliberate.** ``snap_point`` returns the
*nearest* navigable point, so a correction of a metre or two is the filter doing its job
on a frontier cell that the occupancy splat placed slightly inside geometry. The one snap
this refuses is a **cross-floor** one, at ADR-0010's ``|dy| < 1.0 m``
(``PlannerConfig.same_floor_m``): the candidate's y is the agent's own, so a snap that
lands a storey down is a different room reached by stairs, not a correction. Contrast the
detector's 0.5 m gate
(``DetectorConfig.max_snap_m``), which answers a different question — a detection claims
to *be an object*, so a large correction means the claim was wrong, while a waypoint only
claims to be somewhere to go.

**An empty pool raises.** The agent stands on the navmesh, so a fan of directions around
it that yields nothing snappable and reachable means a navmesh island or an off-navmesh
seat: a broken episode, not a degraded one. It is ticket 12's discipline applied to the
planner, and the counters in ``PoolReport`` are there so the raise says *which* stage ate
the candidates rather than only that they are gone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Sequence

from earshot.agent.config import PlannerConfig
from earshot.agent.occupancy import bearing_rel
from earshot.agent.proposers import Candidate
from earshot.types import Pose, Xyz

__all__ = [
    "SnapPoint",
    "Geodesic",
    "EmptyPoolError",
    "PoolReport",
    "reachable_pool",
    "assert_pool",
]

# The two injected callables, named so the runner's adapters have something to satisfy.
# `World.geodesic_distance` takes a *sequence* of ends (it wraps habitat-sim's
# `MultiGoalShortestPath`), so the runner passes `lambda a, b: world.geodesic_distance(a,
# [b])`. Both return `None` rather than NaN or inf for failure — ticket 21 converted
# those at the boundary precisely so a failure is in the type here.
SnapPoint = Callable[[Xyz], Optional[Xyz]]
Geodesic = Callable[[Xyz, Xyz], Optional[float]]


class EmptyPoolError(RuntimeError):
    """Nothing proposed survived the navmesh. ADR-0008's invariant, violated.

    Fatal on purpose: with no waypoint there is no action, and an agent that silently
    stops choosing looks in the trajectory exactly like an agent that chose to stand
    still. The old tree's version of this was a straight-line fallback into the wall the
    frontier sat behind, which is how "a waypoint was chosen" and "the agent got there"
    came apart with nothing in the code marking where.
    """


@dataclass(frozen=True)
class PoolReport:
    """What the filter did, per decision step. Instrumentation, and the raise's evidence."""

    candidates: List[Candidate]
    n_proposed: int
    n_off_navmesh: int
    n_wrong_floor: int
    n_unreachable: int

    @property
    def n_kept(self) -> int:
        return len(self.candidates)

    def counters(self) -> Dict[str, int]:
        """Flat counts for the audit record."""
        return {
            "pool_proposed": self.n_proposed,
            "pool_off_navmesh": self.n_off_navmesh,
            "pool_wrong_floor": self.n_wrong_floor,
            "pool_unreachable": self.n_unreachable,
            "pool_kept": self.n_kept,
        }

    def diagnosis(self) -> str:
        return (
            "{} proposed, {} off-navmesh (snap_point found nothing), {} snapped to "
            "another floor, {} unreachable (no navmesh route from the agent)".format(
                self.n_proposed, self.n_off_navmesh, self.n_wrong_floor, self.n_unreachable
            )
        )


def reachable_pool(
    candidates: Sequence[Candidate],
    pose: Pose,
    *,
    snap_point: SnapPoint,
    geodesic: Geodesic,
    cfg: Optional[PlannerConfig] = None,
) -> PoolReport:
    """Snap every candidate to the navmesh and keep the ones the agent can route to.

    Each survivor is returned **at its snapped position**, with ``distance_m`` and
    ``bearing_rad`` recomputed there and the geodesic distance recorded. Keeping the
    pre-snap geometry would describe a point the agent is not going to, and the scorer's
    distance and bearing terms would then rank the pool on a waypoint that does not
    exist.

    Does not raise on an empty result: a pool of real frontier candidates that the
    navmesh rejects wholesale is the case the compass fan answers, so the caller gets to
    try again before :func:`assert_pool` makes it fatal.

    The floor tolerance comes from ``cfg.same_floor_m`` and has **no local default**, so
    the value in force is the one the run record carries. An earlier revision had a
    literal here as well as in ``PlannerConfig``, which is two homes that can drift and an
    effective value invisible in ``env_report.json`` — the exact failure ``config.py``
    argues ``DetectorConfig`` exists to prevent.
    """
    planner = cfg or PlannerConfig()
    kept: List[Candidate] = []
    n_off_navmesh = n_wrong_floor = n_unreachable = 0
    for candidate in candidates:
        snapped = snap_point(candidate.position)
        if snapped is None:
            n_off_navmesh += 1
            continue
        if abs(snapped.height_difference_to(candidate.position)) >= float(planner.same_floor_m):
            n_wrong_floor += 1
            continue
        distance = geodesic(pose.position, snapped)
        if distance is None:
            n_unreachable += 1
            continue
        dx, dz = snapped.x - pose.position.x, snapped.z - pose.position.z
        kept.append(
            replace(
                candidate,
                position=snapped,
                distance_m=math.hypot(dx, dz),
                bearing_rad=bearing_rel(pose.yaw_rad, dx, dz),
                geodesic_m=float(distance),
            )
        )
    return PoolReport(
        candidates=kept,
        n_proposed=len(candidates),
        n_off_navmesh=n_off_navmesh,
        n_wrong_floor=n_wrong_floor,
        n_unreachable=n_unreachable,
    )


def assert_pool(report: PoolReport, *, stage: str) -> List[Candidate]:
    """Return the pool, or raise :class:`EmptyPoolError` naming what ate it.

    ``stage`` is the caller's description of what it had already tried — "frontier", or
    "compass fan after the frontier pool emptied". Without it the message says only that
    the pool is empty, which does not distinguish a broken navmesh from an agent wedged
    in a one-cell island.
    """
    if not report.candidates:
        raise EmptyPoolError(
            "the candidate pool is empty at stage {!r}: {}. ADR-0008's invariant is that "
            "the pool is never empty and every member is navmesh-reachable and snapped; "
            "with no waypoint there is no action, and an agent that stops choosing is "
            "indistinguishable in a trajectory from one that chose to stand still".format(
                stage, report.diagnosis()
            )
        )
    return report.candidates
