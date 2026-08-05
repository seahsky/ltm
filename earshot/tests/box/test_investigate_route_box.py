#!/usr/bin/env python3
"""Does the realizable detour's probe actually route around an obstacle? V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

**The one assertion that settles ticket 26's structural fix, and no Mac can run it.**

The realizable arm used to apply ``realizable_investigate_step``'s action straight to the
simulator. That gave the whole detour no planner and no map: ``move_forward`` was its only
translation and the energy gradient decided where forward pointed, so a blocked line to the
source was a measured **livelock** — pressed flat against the wall, zero lateral movement,
unchanged by tripling the step budget, in every geometry tried. The first box episode's
"never line-of-sight, ``min_d2source`` 3.19 m" is that failure.

The fix routes the detour through the candidate pool like the oracle arm: the climb names a
probe point (``realizable_investigate_probe``) and the navmesh follower gets there however it
can. Whether it *can* is a navmesh property. ``tests/mac/`` fakes a straight-line follower
(``_task_fakes.FakeWorld.follower``), so a green suite there licenses nothing at all about
routing — ADR-0014's rule that a capability is exercised, never proxied.

**This test prints its measurements** (ADR-0014), because the numbers are what the next
ticket reads: how far the detour got, how many of its forwards were walls, and whether the
route it took was longer than the straight line it could not walk.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import math
import os
import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time.
import earshot  # noqa: F401
from earshot.agent.config import ControllerConfig
from earshot.agent.controller import (
    ACT_FORWARD,
    realizable_investigate_probe,
)
from earshot.task.episodes import (
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    load_scene,
)

SPLIT = os.environ.get("SS2_SPLIT", "val")

CFG = ControllerConfig()

# A pair of navmesh points is "walled" when the geodesic route between them is this much
# longer than the straight line. Below it the agent could have walked straight there and
# the test would prove nothing about routing.
DETOUR_RATIO = 1.4
PAIR_DRAW_TRIES = 400
PAIR_MIN_M = 2.0
PAIR_MAX_M = 8.0
FOLLOW_MAX_STEPS = 300
PLACEMENT_SEED = 20260805

_WORLD = None
_SCENE = None


def setUpModule():
    global _WORLD, _SCENE
    from earshot.sim.world import World, camera_sensor_specs

    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    override = os.environ.get("SS2_SCENE_LABEL")
    labels = [override] if override else list(available_scenes(split_dir))
    for label in labels:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _SCENE = dataset
            break
    if _SCENE is None:
        raise unittest.SkipTest("no HM3D scene mesh on this box")
    print("\n  scene: {}".format(_SCENE.scene_path), flush=True)
    # Cameras only. This module asks two questions — does the follower route, and is the
    # probe where `move_forward` actually goes — and neither renders a sound, so an audio
    # sensor here would buy nothing and cost the guard's arming render every episode.
    _WORLD = World(_SCENE.scene_path, camera_sensor_specs(width=256, height=256))
    _WORLD.seed_navmesh(PLACEMENT_SEED)


def tearDownModule():
    if _WORLD is not None:
        _WORLD.close()


def _walled_pair():
    """Two navigable points the navmesh can only connect the long way round.

    Drawn rather than hand-placed: a hard-coded coordinate pair is a claim about one
    scene's furniture, and the box picks whichever scene it has. Returns
    ``(start, target, straight_m, geodesic_m)``.
    """
    for _ in range(PAIR_DRAW_TRIES):
        start = _WORLD.random_navigable_point()
        target = _WORLD.random_navigable_point()
        straight = start.horizontal_distance_to(target)
        if not (PAIR_MIN_M <= straight <= PAIR_MAX_M):
            continue
        if abs(start.y - target.y) > 0.5:  # another storey, not another room
            continue
        geodesic = _WORLD.geodesic_distance(start, [target])
        if geodesic is None or math.isinf(geodesic):
            continue
        if geodesic >= straight * DETOUR_RATIO:
            return start, target, straight, geodesic
    return None


class TestTheFollowerRoutesAroundAnObstacle(unittest.TestCase):
    """The capability the fix depends on, exercised on the real navmesh."""

    def test_a_probe_behind_a_wall_is_reached_the_long_way(self):
        pair = _walled_pair()
        if pair is None:
            self.skipTest(
                "no walled pair found in {} draws — this scene's navmesh may be one "
                "open room, which is not a failure of the follower".format(
                    PAIR_DRAW_TRIES
                )
            )
        start, target, straight, geodesic = pair
        print(
            "  walled pair: straight {:.2f} m, geodesic {:.2f} m (ratio {:.2f})".format(
                straight, geodesic, geodesic / straight
            ),
            flush=True,
        )

        _WORLD.set_pose(start)
        follow = _WORLD.follower(goal_radius=0.25)
        collisions, steps = 0, 0
        for steps in range(1, FOLLOW_MAX_STEPS + 1):
            action = follow(target)
            if action is None:
                break
            collisions += int(_WORLD.step(action))
        reached = _WORLD.pose().position.horizontal_distance_to(target)
        print(
            "  reached {:.2f} m from the probe in {} steps, {} collisions".format(
                reached, steps, collisions
            ),
            flush=True,
        )
        self.assertLess(
            reached,
            1.0,
            "the follower did not reach a probe it had a {:.2f} m route to — the "
            "detour's fix rests on this and the realizable arm has no other way "
            "around an obstacle".format(geodesic),
        )

    def test_the_straight_line_the_old_arm_would_have_walked_hits_the_wall(self):
        """The control: what stepping ``move_forward`` at the bearing actually buys.

        Without it a green above could mean the pair was not really walled. This walks
        the old arm's move — face the target, push forward — and asserts it does *not*
        arrive, which is the livelock the probe replaced.
        """
        pair = _walled_pair()
        if pair is None:
            self.skipTest("no walled pair found; see the test above")
        start, target, straight, geodesic = pair

        _WORLD.set_pose(start)
        collisions = 0
        for _ in range(FOLLOW_MAX_STEPS):
            pose = _WORLD.pose()
            dx = target.x - pose.position.x
            dz = target.z - pose.position.z
            # Face it, then push: the old arm's whole repertoire.
            bearing = math.atan2(-dx, -dz) - pose.yaw_rad
            bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi
            if abs(bearing) > math.radians(15.0):
                collisions += int(
                    _WORLD.step("turn_left" if bearing > 0 else "turn_right")
                )
                continue
            collisions += int(_WORLD.step(ACT_FORWARD))
            if _WORLD.pose().position.horizontal_distance_to(target) < 1.0:
                break
        reached = _WORLD.pose().position.horizontal_distance_to(target)
        print(
            "  straight-line walk stopped {:.2f} m away, {} collisions "
            "(straight {:.2f} m, geodesic {:.2f} m)".format(
                reached, collisions, straight, geodesic
            ),
            flush=True,
        )
        self.assertGreater(
            collisions,
            0,
            "the straight line was walkable, so this pair does not demonstrate what "
            "the probe is for",
        )


class TestTheProbeIsWhereTheFrameSaysItIs(unittest.TestCase):
    """The probe's geometry against the simulator's own idea of forward.

    ``tests/mac/test_agent_controller.py`` pins this against ``agent/occupancy``'s frame,
    which is the tree agreeing with itself. Ticket 23 found the old tree agreeing with
    itself while being 180 degrees out from habitat, so the frame is checked here too, by
    acting.
    """

    def test_a_forward_probe_lands_where_move_forward_goes(self):
        start = _WORLD.random_navigable_point()
        _WORLD.set_pose(start)
        pose = _WORLD.pose()
        probe = realizable_investigate_probe(ACT_FORWARD, pose, CFG)

        before = _WORLD.pose().position
        for _ in range(int(round(CFG.investigate_probe_m / 0.25))):
            if _WORLD.step(ACT_FORWARD):
                self.skipTest("walked into something; this test needs clear floor")
        after = _WORLD.pose().position
        walked = after.horizontal_distance_to(before)
        error = after.horizontal_distance_to(probe)
        print(
            "  walked {:.2f} m; probe is {:.2f} m from where that landed".format(
                walked, error
            ),
            flush=True,
        )
        self.assertLess(
            error,
            0.4,
            "the forward probe is not where move_forward actually goes — the probe "
            "heading and habitat's forward disagree, which is ticket 23's defect in a "
            "new place",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
