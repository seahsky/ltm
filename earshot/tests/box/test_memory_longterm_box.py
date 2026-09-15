#!/usr/bin/env python3
"""`h^traj` on REAL navmesh routes, because its components could carry nothing.

    conda activate ss2
    bash earshot/tools/box_gate.sh

**THE QUESTION THE MAC SUITE CANNOT ASK.** `tests/mac/test_memory_longterm.py` proves
`trajectory_descriptor` separates an L-shaped walk from a straight one, using fixtures
built to differ. It cannot say whether real routes differ. If every walk the
`ShortestPathFollower` produces comes out at straightness 0.99 with no turns, then
`straightness` and `mean_abs_turn_rad` are two constants, `h^traj` is one number wearing
four, and `rho_k.trajectory` (eq. 17) summarises nothing.

That is the same class of failure PR #110's box arm was written for, and the same one this
repo keeps finding after the fact: grid-A* found no path on ~92% of steps; the S1+
captioner never lifted; the Step-4 affordance head proposed and was never chosen.

**No encoders and no audio here on purpose.** This file needs the NAVMESH and nothing
else, so it stays cheap and its failure can only mean one thing. `h^av` is measured in
`test_memory_consolidate_box.py`, which already pays for CLIP and CLAP.

**These tests print their measurements** (ADR-0014). The four components per route, and
the spread of each across routes, are what say whether `h^traj` is worth four floats.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from earshot.memory.longterm import (
    TRAJECTORY_COMPONENTS,
    ExperienceEntry,
    ExperienceStore,
    Outcome,
    abstract,
    trajectory_descriptor,
)
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene

SPLIT = "val"
PLACEMENT_SEED = 20260821
ROUTES = 8
MIN_GEODESIC_M = 3.0
MAX_STEPS = 400
GOAL_RADIUS = 0.3

# A walk of 3 m or more around real furniture is not a straight line. If every route comes
# out within this much of every other, `straightness` is a constant and carries nothing.
MIN_STRAIGHTNESS_SPREAD = 0.05
# And at least one route must actually turn. Radians per step, averaged over the route.
MIN_TURN_RAD = 0.05

_DATASET = None


def setUpModule():
    global _DATASET
    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    override = os.environ.get("SS2_SCENE_LABEL")
    for label in ([override] if override else list(available_scenes(split_dir))):
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _DATASET = dataset
            break
    if _DATASET is None:
        raise unittest.SkipTest("no ObjectNav {} scene has its mesh on this box".format(SPLIT))
    print("\n  scene: {}".format(_DATASET.scene_label), flush=True)


class TestTheTrajectoryDescriptorOnRealRoutes(unittest.TestCase):
    """Several real navmesh routes, walked and described."""

    @classmethod
    def setUpClass(cls):
        from earshot.sim.world import World, camera_sensor_specs
        from earshot.types import NoRouteError

        cls.world = World(_DATASET.scene_path, camera_sensor_specs(width=64, height=64))
        cls.world.seed_navmesh(PLACEMENT_SEED)
        cls.walks = []
        attempts = 0
        while len(cls.walks) < ROUTES and attempts < ROUTES * 20:
            attempts += 1
            start = cls.world.random_navigable_point()
            target = cls.world.random_navigable_point()
            geodesic = cls.world.geodesic_distance(start, [target])
            if geodesic is None or geodesic < MIN_GEODESIC_M:
                continue
            cls.world.set_pose(start)
            follow = cls.world.follower(goal_radius=GOAL_RADIUS)
            positions = [cls.world.pose().position]
            try:
                for _ in range(MAX_STEPS):
                    action = follow(target)
                    if action is None:
                        break
                    cls.world.step(action)
                    positions.append(cls.world.pose().position)
            except NoRouteError:
                continue
            if len(positions) < 3:
                continue
            cls.walks.append((geodesic, positions))
        print("  {} route(s) of >= {:.1f} m geodesic, from {} attempt(s)".format(
            len(cls.walks), MIN_GEODESIC_M, attempts), flush=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "world", None) is not None:
            cls.world.close()

    def test_enough_routes_were_walked_to_measure_a_spread(self):
        """The control. Two routes cannot show that four components carry anything, and a
        skip here would make every measurement below vacuous rather than green."""
        self.assertGreaterEqual(
            len(self.walks), 4,
            "only {} route(s) of at least {:.1f} m were walked; the spread below is "
            "measured over too few".format(len(self.walks), MIN_GEODESIC_M),
        )

    def test_the_four_components_are_not_four_constants(self):
        """**THE MEASUREMENT THIS FILE EXISTS FOR.** Prints every component of every real
        route. Red means `h^traj` is one number wearing four and `rho_k.trajectory`
        summarises nothing."""
        descriptors = [trajectory_descriptor(positions) for _geo, positions in self.walks]
        print("  {:>4}  {:>7}  {:>9}  {:>12}  {:>12}  {:>17}".format(
            "n", "steps", "geodesic", *TRAJECTORY_COMPONENTS[:3]))
        for index, ((geodesic, positions), descriptor) in enumerate(
            zip(self.walks, descriptors)
        ):
            print("  {:>4}  {:>7}  {:>9.2f}  {:>12.2f}  {:>12.2f}  {:>17.4f}   turn "
                  "{:.4f}".format(
                      index, len(positions), geodesic, float(descriptor[0]),
                      float(descriptor[1]), float(descriptor[2]), float(descriptor[3])))
        stacked = np.stack(descriptors)
        for axis, name in enumerate(TRAJECTORY_COMPONENTS):
            column = stacked[:, axis]
            print("  {:<22} min {:8.4f}  mean {:8.4f}  max {:8.4f}  spread {:8.4f}".format(
                name, float(column.min()), float(column.mean()), float(column.max()),
                float(column.max() - column.min())))
        self.assertTrue(
            bool(np.isfinite(stacked).all()),
            "a real route produced a non-finite descriptor component",
        )
        straightness = stacked[:, 2]
        self.assertGreater(
            float(straightness.max() - straightness.min()), MIN_STRAIGHTNESS_SPREAD,
            "every real route had the same straightness to within {}, so the component "
            "that separates walking-to-it from wandering-into-it is a constant".format(
                MIN_STRAIGHTNESS_SPREAD),
        )
        self.assertGreater(
            float(stacked[:, 3].max()), MIN_TURN_RAD,
            "no real route turned by an average of {} rad per step, so "
            "mean_abs_turn_rad carries nothing either".format(MIN_TURN_RAD),
        )

    def test_a_real_walk_is_never_straighter_than_straight(self):
        """The bound `straightness` claims, checked against routes rather than fixtures:
        net displacement can never exceed path length, so the ratio is in [0, 1]."""
        for _geodesic, positions in self.walks:
            descriptor = trajectory_descriptor(positions)
            self.assertLessEqual(float(descriptor[1]), float(descriptor[0]) + 1e-4)
            self.assertGreaterEqual(float(descriptor[2]), 0.0)
            self.assertLessEqual(float(descriptor[2]), 1.0 + 1e-6)

    def test_the_follower_does_not_walk_the_geodesic_exactly(self):
        """What `path_length_m` costs above the shortest route. Printed rather than
        asserted tightly: it is the number that says whether a pattern's mean path length
        describes the agent's behaviour or the navmesh's."""
        ratios = [
            float(trajectory_descriptor(positions)[0]) / geodesic
            for geodesic, positions in self.walks
            if geodesic > 0.0
        ]
        print("  walked / geodesic: min {:.3f}  mean {:.3f}  max {:.3f}".format(
            min(ratios), sum(ratios) / len(ratios), max(ratios)))
        self.assertGreater(
            min(ratios), 0.5,
            "a route walked less than half its own geodesic, so the positions and the "
            "distance are not describing the same walk",
        )

    def test_g_summarises_real_walks_into_a_pattern(self):
        """`G` (eq. 16) over experiences whose `h^traj` are these real descriptors. The
        mean the pattern reports must be the mean of the walks that went into it, which is
        the only claim `rho_k.trajectory` makes."""
        half = len(self.walks) // 2
        groups = (("toilet_flush", "toilet", self.walks[:half]),
                  ("snoring", "bed", self.walks[half:]))
        entries = []
        for sound, obj, walks in groups:
            for index, (_geodesic, positions) in enumerate(walks):
                entries.append(ExperienceEntry(
                    context=np.asarray(
                        [1.0, 0.0] if sound == "toilet_flush" else [0.0, 1.0],
                        dtype=np.float32,
                    ),
                    trajectory=trajectory_descriptor(positions),
                    target_concept=obj,
                    outcome=Outcome(reached=True, final_gap_m=0.1 * (index + 1)),
                    sound_concept=sound,
                    room_concept=None,
                ))
        patterns = abstract(ExperienceStore().extend(entries), min_support=2)
        print("  G over {} real walks -> {} pattern(s)".format(len(entries), len(patterns)))
        for pattern in patterns.patterns:
            print("    {:<14} -> {:<8} support {}  mean path {:.2f} m  mean "
                  "straightness {:.4f}  mean gap {:.2f} m".format(
                      pattern.sound_concept, pattern.object_concept,
                      pattern.strategy.support, float(pattern.strategy.trajectory[0]),
                      float(pattern.strategy.trajectory[2]),
                      pattern.strategy.mean_final_gap_m))
        self.assertEqual(len(patterns), 2)
        by_sound = {pattern.sound_concept: pattern for pattern in patterns.patterns}
        for sound, _obj, walks in groups:
            expected = float(np.mean([
                trajectory_descriptor(positions)[0] for _geodesic, positions in walks
            ]))
            self.assertAlmostEqual(
                float(by_sound[sound].strategy.trajectory[0]), expected, places=3,
                msg="the {} pattern did not report the mean path length of the walks "
                    "that produced it".format(sound),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
