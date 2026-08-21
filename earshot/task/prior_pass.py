"""The prior pass: a scripted tour of a scene's anchor rooms, with the source sounding.

ADR-0018 leaves the prior pass undesigned and names it as open. This is the answer taken on
2026-08-21: **a scripted navmesh tour, not an agent-driven episode**.

**Why scripted.** The matrix already carries a measured 16.2% per-episode flip rate and roughly
90 episodes a cell, so it cannot afford another variance source. An agent-driven prior pass
would give each seen cell whatever coverage that episode happened to achieve, and a null would
then be unreadable: memory failed, or the prior pass never entered the room. A fixed route makes
coverage identical across scenes by construction, so a cell's memory quality is confounded with
nothing.

**Why not an oracle write.** Writing the associations straight into the store would cost no
episodes at all and be perfectly controlled. CLAUDE.md forbids it: a capability is exercised,
never proxied. The tour runs the real audio sensor and the real encoder, so what the store
receives is what the agent could actually have perceived.

**One stop per ROOM, not per object.** The semantic store learns at the room level (ADR-0018,
2026-08-20), so a second sofa adds nothing it can learn and does add scene-specific tour length.
One stop per room keeps the route comparable between a four-room house and a nine-room one.

**The cost is not a doubling.** ADR-0018 says the matrix "is a two-visit design and episode cost
roughly doubles". That over-counts. The episodic store is scene-keyed, so ONE prior pass serves
every test episode in that scene; the semantic store is scene-agnostic, so ONE pass hearing a
class serves every test episode of that class anywhere. Ten seen scenes and six heard classes is
tens of prior episodes against a test set in the hundreds.

**Nothing here writes to a memory store.** The store is new code that ADR-0018 commits to and
that does not exist yet. This module produces the RECORD a store will consume, and stops there,
so the tour can be exercised and measured before anything depends on it.

Layer note (ADR-0013): `plan_tour` and everything above `walk_tour` is pure and imports no
simulator, so it tests on a Mac. `walk_tour` takes a `World` and therefore lives in `task/`,
which is the layer allowed to hold one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.types import Xyz

__all__ = [
    "TourStop",
    "TourPlan",
    "LegOutcome",
    "TourRecord",
    "candidate_stops",
    "plan_tour",
    "walk_tour",
]

# A leg that has not arrived in this many steps is ABANDONED and said so. Without a budget a
# follower that oscillates between two navmesh polygons hangs the whole sweep, and the run
# reports nothing rather than reporting a bad tour.
DEFAULT_LEG_BUDGET = 200


@dataclass(frozen=True)
class TourStop:
    """One waypoint: a navigable point, the room it is in, and the object it belongs to.

    `room` is what the semantic store learns. `category` is kept for the audit record only:
    two scenes both reaching `living_room` via different objects is a fact worth being able
    to read back, and it is never what the store is keyed on.
    """

    room: str
    category: str
    point: Xyz


@dataclass(frozen=True)
class TourPlan:
    """An ordered route, and every candidate that could not join it.

    `unreachable` is not a warning to be skimmed. 23 of 365 episodes in the anomaly-response
    sweep had no navmesh route to their source and that went unnoticed until a tool counted
    them, so a dropped stop is a first-class field with a reason attached.
    """

    stops: Tuple[TourStop, ...]
    unreachable: Tuple[Tuple[TourStop, str], ...]

    @property
    def rooms(self) -> Tuple[str, ...]:
        return tuple(stop.room for stop in self.stops)

    def as_dict(self) -> Dict[str, object]:
        return {
            "stops": [
                {"room": s.room, "category": s.category, "point": list(s.point.as_tuple())}
                for s in self.stops
            ],
            "unreachable": [
                {"room": s.room, "category": s.category, "reason": reason}
                for s, reason in self.unreachable
            ],
        }


@dataclass(frozen=True)
class LegOutcome:
    """What happened walking to one stop. Abandoned legs are recorded, never dropped."""

    stop: TourStop
    reached: bool
    steps: int
    final_gap_m: Optional[float]
    reason: str


@dataclass(frozen=True)
class TourRecord:
    """The prior pass as it actually ran. This is what a memory store will consume."""

    scene: str
    legs: Tuple[LegOutcome, ...]
    observations: Tuple[Mapping[str, object], ...]

    @property
    def rooms_reached(self) -> Tuple[str, ...]:
        return tuple(leg.stop.room for leg in self.legs if leg.reached)

    @property
    def complete(self) -> bool:
        """Every planned leg arrived.

        An INCOMPLETE prior pass must not be treated as a seen scene. That is the whole
        reason the tour is scripted: a partial tour reintroduces exactly the coverage
        variance the design chose against, and silently accepting one would hide it.
        """
        return bool(self.legs) and all(leg.reached for leg in self.legs)

    def as_dict(self) -> Dict[str, object]:
        return {
            "scene": self.scene,
            "complete": self.complete,
            "rooms_reached": list(self.rooms_reached),
            "legs": [
                {
                    "room": leg.stop.room,
                    "category": leg.stop.category,
                    "reached": leg.reached,
                    "steps": leg.steps,
                    "final_gap_m": leg.final_gap_m,
                    "reason": leg.reason,
                }
                for leg in self.legs
            ],
            "n_observations": len(self.observations),
        }


def candidate_stops(
    points_by_category: Mapping[str, Sequence[Xyz]],
    room_of_category: Mapping[str, str],
) -> Tuple[TourStop, ...]:
    """Every navigable anchor point in the scene, tagged with its room.

    A category the room map does not carry is SKIPPED rather than raising: HM3D scenes
    publish `plant` goals and `plant` deliberately has no room (ADR-0018), so raising here
    would make every scene with a houseplant unplannable.
    """
    stops: List[TourStop] = []
    for category in sorted(points_by_category):
        room = room_of_category.get(category)
        if room is None:
            continue
        for point in points_by_category[category]:
            stops.append(TourStop(room=room, category=category, point=point))
    return tuple(stops)


def plan_tour(
    candidates: Sequence[TourStop],
    start: Xyz,
    geodesic: Callable[[Xyz, Xyz], Optional[float]],
) -> TourPlan:
    """One stop per room, ordered nearest-first by geodesic distance from the walk so far.

    Greedy rather than optimal on purpose. Tour length is not the object of study, and an
    exact solver would be a second thing that can be wrong for no gain. Greedy is
    deterministic, which is what the design needs.

    `geodesic` returns None where the navmesh has no route. Those candidates are recorded in
    `unreachable` with the reason, and a room whose every candidate is unroutable simply does
    not appear in the plan. `TourRecord.complete` is then the guard: a scene that cannot be
    fully toured is not a seen scene.

    Ties break on `(category, point)` so two runs of the same scene produce the same route.
    """
    unreachable: List[Tuple[TourStop, str]] = []
    reachable: List[Tuple[TourStop, float]] = []
    for stop in candidates:
        distance = geodesic(start, stop.point)
        if distance is None:
            unreachable.append((stop, "no navmesh route from the tour start"))
            continue
        reachable.append((stop, float(distance)))

    # One per room: the nearest reachable candidate. A second sofa teaches the semantic store
    # nothing it did not learn at the first, and it makes the route length a property of the
    # house rather than of the design.
    best_per_room: Dict[str, Tuple[TourStop, float]] = {}
    for stop, distance in reachable:
        held = best_per_room.get(stop.room)
        if held is None or (distance, stop.category, stop.point.as_tuple()) < (
            held[1],
            held[0].category,
            held[0].point.as_tuple(),
        ):
            if held is not None:
                unreachable.append((held[0], "a nearer candidate covers {}".format(stop.room)))
            best_per_room[stop.room] = (stop, distance)
        else:
            unreachable.append((stop, "a nearer candidate covers {}".format(stop.room)))

    remaining = [stop for stop, _d in best_per_room.values()]
    ordered: List[TourStop] = []
    here = start
    while remaining:
        scored: List[Tuple[float, str, Tuple[float, float, float], TourStop]] = []
        for stop in remaining:
            distance = geodesic(here, stop.point)
            if distance is None:
                continue
            scored.append((float(distance), stop.category, stop.point.as_tuple(), stop))
        if not scored:
            # Reachable from the start but not from where the tour now stands. Recording it
            # rather than looping is the point: an unorderable remainder is a finding.
            for stop in remaining:
                unreachable.append((stop, "no navmesh route from the previous stop"))
            break
        scored.sort(key=lambda item: item[:3])
        chosen = scored[0][3]
        ordered.append(chosen)
        remaining = [stop for stop in remaining if stop is not chosen]
        here = chosen.point

    return TourPlan(stops=tuple(ordered), unreachable=tuple(unreachable))


def walk_tour(
    world: object,
    plan: TourPlan,
    *,
    scene: str,
    observe: Optional[Callable[[TourStop], Mapping[str, object]]] = None,
    leg_budget: int = DEFAULT_LEG_BUDGET,
    goal_radius: float = 1.0,
) -> TourRecord:
    """Drive `plan` with the navmesh follower and record what happened on every leg.

    `world` is a `sim.World`, taken as `object` so this module's signature carries no
    simulator type: ADR-0013's injection rule. It uses `follower`, `pose` and `step`.

    `observe` is called once on arrival at each stop and whatever it returns is appended to
    the record. That is the seam the memory store will attach to; it is a parameter rather
    than an import so this module never grows a dependency on a store that does not exist.

    A leg that exhausts `leg_budget` is recorded as abandoned WITH its final gap, and the
    tour carries on to the next stop. Stopping the whole pass would throw away the rooms that
    did work; hiding the abandonment would let a partial tour read as a seen scene. Neither
    is acceptable, so it is recorded and `TourRecord.complete` goes False.
    """
    next_action = world.follower(goal_radius)  # type: ignore[attr-defined]
    legs: List[LegOutcome] = []
    observations: List[Mapping[str, object]] = []

    for stop in plan.stops:
        steps = 0
        reached = False
        reason = "budget of {} steps exhausted".format(leg_budget)
        while steps < leg_budget:
            try:
                action = next_action(stop.point)
            except Exception as exc:  # NoRouteError, and anything the follower raises
                reason = "follower refused the target: {}".format(exc)
                break
            if action is None:
                reached = True
                reason = "arrived"
                break
            world.step(action)  # type: ignore[attr-defined]
            steps += 1

        position = world.pose().position  # type: ignore[attr-defined]
        gap = world.geodesic_distance(position, [stop.point])  # type: ignore[attr-defined]
        legs.append(
            LegOutcome(
                stop=stop,
                reached=reached,
                steps=steps,
                final_gap_m=None if gap is None else float(gap),
                reason=reason,
            )
        )
        if reached and observe is not None:
            observations.append(observe(stop))

    return TourRecord(scene=scene, legs=tuple(legs), observations=tuple(observations))
