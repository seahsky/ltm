#!/usr/bin/env python3
"""Does the scripted prior pass actually tour a real HM3D scene? V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

**Two things no Mac can answer**, and the whole design rests on both.

*Can a real scene be toured at all.* `tests/mac/test_prior_pass.py` plans against a Euclidean
stand-in and walks a `FakeWorld` that arrives on command. A green suite there licenses nothing
about the navmesh: ADR-0014's rule that a capability is exercised, never proxied. The greedy
follower is also the component that livelocked against a wall in ticket 26, so "the plan says
three stops" and "the agent reaches three stops" are different claims.

*How many of the three anchor rooms a scene can offer.* HM3D publishes ObjectNav goals per
scene and not every scene has a toilet. A scene missing a room cannot host that room's
episodes, so this is the same yield question `yield_sweep.sh` asks about sources, one level up.
**It is printed, because that number decides whether the 2x2 is buildable at the sizes the
power question assumes.**

**This test prints its measurements** (ADR-0014). Steps per leg is what sizes the prior phase.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import os
import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time.
import earshot  # noqa: F401
from earshot.audio.vocabulary import ROOMS, ROOM_OF_ANCHOR
from earshot.task.dataset import goal_table
from earshot.task.episodes import (
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    load_scene,
)
from earshot.task.prior_pass import candidate_stops, plan_tour, walk_tour

SPLIT = os.environ.get("SS2_SPLIT", "val")
PLACEMENT_SEED = 20260821
# Generous on purpose. This test asks whether a leg CAN be walked, not whether it is walked
# efficiently; a tight budget here would report a navmesh failure that is really a budget.
LEG_BUDGET = 400
GOAL_RADIUS = 1.0

_WORLD = None
_SCENE = None
_LABEL = None


def setUpModule():
    global _WORLD, _SCENE, _LABEL
    from earshot.sim.world import World, camera_sensor_specs

    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    override = os.environ.get("SS2_SCENE_LABEL")
    labels = [override] if override else list(available_scenes(split_dir))
    for label in labels:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _SCENE, _LABEL = dataset, label
            break
    if _SCENE is None:
        raise unittest.SkipTest("no HM3D scene mesh on this box")
    print("\n  scene: {}".format(_SCENE.scene_path), flush=True)
    # Cameras only. The tour's route is a navmesh question; the audio sensor costs the
    # guard's arming render and answers nothing this module asks.
    _WORLD = World(_SCENE.scene_path, camera_sensor_specs(width=256, height=256))
    _WORLD.seed_navmesh(PLACEMENT_SEED)


def tearDownModule():
    if _WORLD is not None:
        _WORLD.close()


def _points_by_category():
    """``{category: [view point]}`` from the scene's own published goals.

    The first view point per goal, matching `dataset._first_view_point`'s reason: the
    highest-IoU entry can be malformed and entry zero is always real.
    """
    out = {}
    for category, goals in goal_table(_SCENE).items():
        points = [g.view_points[0].position for g in goals if g.view_points]
        if points:
            out[category] = points
    return out


def _plan():
    start = _WORLD.random_navigable_point()
    _WORLD.set_pose(start)
    candidates = candidate_stops(_points_by_category(), ROOM_OF_ANCHOR)
    return start, candidates, plan_tour(candidates, start, _geodesic)


def _geodesic(a, b):
    return _WORLD.geodesic_distance(a, [b])


class TestTheSceneCanBeToured(unittest.TestCase):
    def test_the_anchor_room_yield_of_this_scene(self):
        """How many of the three rooms this scene can offer. A count, not a pass/fail.

        A scene short of a room cannot host that room's episodes, which shrinks the cell
        sizes the power question is arguing about. Failing here would be wrong: a house
        with no toilet goal is a property of HM3D, not a defect in the tour.
        """
        by_category = _points_by_category()
        anchors = {c: len(p) for c, p in by_category.items() if c in ROOM_OF_ANCHOR}
        rooms = {ROOM_OF_ANCHOR[c] for c in anchors}
        print("  anchor categories published: {}".format(anchors or "NONE"), flush=True)
        print(
            "  rooms tourable: {}/{}  {}".format(
                len(rooms), len(ROOMS), sorted(rooms) or "NONE"
            ),
            flush=True,
        )
        missing = sorted(set(ROOMS) - rooms)
        if missing:
            print("  rooms this scene CANNOT host: {}".format(missing), flush=True)
        self.assertGreaterEqual(len(by_category), 1, "scene published no goals at all")

    def test_a_real_tour_plans_and_completes(self):
        """The healthy arm: plan on the real navmesh, walk it with the real follower."""
        start, candidates, plan = _plan()
        print("  start: {}".format(start), flush=True)
        print(
            "  candidates {} -> planned stops {} ({})".format(
                len(candidates), len(plan.stops), ", ".join(plan.rooms) or "NONE"
            ),
            flush=True,
        )
        for stop, reason in plan.unreachable:
            print("    dropped {:12s} {}: {}".format(stop.room, stop.category, reason))
        if not plan.stops:
            self.skipTest(
                "no anchor room is reachable from this start, which is a scene fact "
                "rather than a tour failure -- see the yield count above"
            )

        record = walk_tour(
            _WORLD, plan, scene=_LABEL, leg_budget=LEG_BUDGET, goal_radius=GOAL_RADIUS
        )
        for leg in record.legs:
            print(
                "    {:12s} {:14s} reached={} steps={:4d} gap={} ({})".format(
                    leg.stop.room,
                    leg.stop.category,
                    leg.reached,
                    leg.steps,
                    "-" if leg.final_gap_m is None else "{:.2f} m".format(leg.final_gap_m),
                    leg.reason,
                ),
                flush=True,
            )
        total = sum(leg.steps for leg in record.legs)
        print(
            "  TOUR: complete={} rooms={} total steps={}".format(
                record.complete, list(record.rooms_reached), total
            ),
            flush=True,
        )
        self.assertTrue(
            record.complete,
            "the tour did not reach every planned stop within {} steps a leg. Every stop "
            "was geodesic-reachable at plan time, so this is the follower failing to walk "
            "a route the navmesh says exists -- ticket 26's livelock, not a budget.".format(
                LEG_BUDGET
            ),
        )

    def test_an_impossible_budget_is_abandoned_and_recorded(self):
        """The forced-failure arm (ADR-0014): the guard fires on the real follower too.

        One step a leg cannot reach anything. If this came back `complete`, the completeness
        flag would be decorative and a partial tour could pass as a seen scene.
        """
        start, _candidates, plan = _plan()
        if not plan.stops:
            self.skipTest("no reachable anchor room from this start")
        record = walk_tour(_WORLD, plan, scene=_LABEL, leg_budget=1, goal_radius=GOAL_RADIUS)
        gaps = [leg.final_gap_m for leg in record.legs]
        print(
            "  starved tour: complete={} legs={} gaps={}".format(
                record.complete,
                len(record.legs),
                ["-" if g is None else "{:.2f}".format(g) for g in gaps],
            ),
            flush=True,
        )
        self.assertFalse(record.complete)
        self.assertEqual(len(record.legs), len(plan.stops), "an abandoned leg was dropped")
        self.assertTrue(
            all(leg.steps <= 1 for leg in record.legs), "the budget was not enforced"
        )
        self.assertTrue(
            any(not leg.reached for leg in record.legs),
            "one step a leg reached everything, so this scene's stops are on top of the "
            "start and the arm proves nothing",
        )

    def test_the_plan_is_the_same_twice_on_the_real_navmesh(self):
        """Determinism is the entire argument for scripting over an agent-driven pass.

        The Mac test proves the planner is deterministic given a deterministic geodesic.
        This proves the geodesic is one: a navmesh query that varied between calls would
        make the route vary and nothing above would notice.
        """
        start = _WORLD.random_navigable_point()
        _WORLD.set_pose(start)
        candidates = candidate_stops(_points_by_category(), ROOM_OF_ANCHOR)
        first = plan_tour(candidates, start, _geodesic)
        second = plan_tour(candidates, start, _geodesic)
        print("  route: {}".format(list(first.rooms) or "NONE"), flush=True)
        self.assertEqual(first.as_dict(), second.as_dict())


if __name__ == "__main__":
    unittest.main()
