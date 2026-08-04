#!/usr/bin/env python3
"""The agent frame, measured against the simulator. V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

``agent/occupancy.py`` states that habitat's forward is ``(-sin yaw, -cos yaw)`` over
``(x, z)`` and that a positive yaw change is a left turn. Everything in ``agent/`` rests
on it: the depth splat writes the map along it, the candidate bearing is measured from it,
and the detector back-projects through it. The Mac can only check that the tree's two frame
consumers **agree with each other** (``tests/mac/test_agent_frame.py``) — whether they
agree with the *simulator* is behaviour we did not write, which is ADR-0014's definition of
a box-only assertion.

**This is not a hypothetical risk.** The old tree used ``(+sin, +cos)``, 180 degrees out,
and it was internally consistent: the splat, the bearing and the detector's back-projection
all agreed with each other and all disagreed with habitat. The predicted symptoms are the
recorded ones — no A* path on roughly 92% of steps, ``n_waypoint_unreachable`` 60-99 per
episode, ``n_detector_localized`` 0 across a matrix. A test of this shape is what would
have caught it, and it is cheap: act, and measure where the agent went.

**These tests print their measurements** (ADR-0014), because the numbers are what make a
frame claim checkable by the next reader rather than trusted.

Four measurements: ``move_forward``'s direction at four yaws, ``turn_left``'s sign, the
free-cell half-planes on a real render, and a back-projected pixel's forward sign and
height. The inverted frame fails all four; the 0.88 m camera-origin error fails the last.

**One assumption these do NOT settle, stated so it is not read as covered.** The splat and
the back-projection both take habitat's depth to be *planar* — distance along the optical
axis, not euclidean range — which is what the old tree's docstring asserted and what
``x_cam = d * (u - cx) / f`` requires. The centre-pixel check below cannot discriminate the
two, because on the optical axis they are the same number. Discriminating needs a
fronto-parallel surface filling the frame, which no HM3D pose guarantees. If it is wrong,
the symptom is a cone that widens with range rather than an inverted map.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import math
import os
import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which habitat-sim
# reads at import time.
import earshot  # noqa: F401
from earshot.agent.config import PlannerConfig
from earshot.agent.detector import back_project_pinhole
from earshot.agent.occupancy import (
    CELL_FREE,
    camera_to_world,
    forward_xz,
    integrate_depth,
    intrinsics_from_hfov,
    new_grid,
    right_xz,
    wrap_pi,
)
from earshot.task.episodes import (
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    load_scene,
)

SPLIT = os.environ.get("SS2_SPLIT", "val")

PLACEMENT_SEED = 20260804
POSE_TRIES = 64
# A forward step is 0.25 m (habitat-lab's ObjectNav HM3D config). Anything under this
# moved too little to give a direction; anything at all over it means the step landed.
MIN_DISPLACEMENT_M = 0.05
# The turn is 30 degrees. Generous tolerance: this pins the SIGN and the axis, not the
# actuation's precision.
TURN_TOLERANCE_DEG = 5.0
# The angle between the commanded heading and the measured displacement. A forward step
# that grazes geometry slides along it, so this is a bound on "went forward", not on
# "went exactly there".
HEADING_TOLERANCE_DEG = 20.0

_DATASET = None


def setUpModule():
    global _DATASET
    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    label_override = os.environ.get("SS2_SCENE_LABEL")
    candidates = [label_override] if label_override else list(available_scenes(split_dir))
    for label in candidates:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _DATASET = dataset
            break
    if _DATASET is None:
        raise unittest.SkipTest(
            "no ObjectNav {} scene has its mesh on this box (looked under {})".format(
                SPLIT, scenes_dir
            )
        )
    print("\n  scene: {}".format(_DATASET.scene_label), flush=True)


def _yaw_quaternion(yaw_rad):
    """``[x, y, z, w]`` for a rotation of ``yaw_rad`` about ``+y`` — the dataset's order."""
    return [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)]


def _new_world():
    from earshot.sim.world import World, camera_sensor_specs

    world = World(_DATASET.scene_path, camera_sensor_specs(width=256, height=256))
    world.seed_navmesh(PLACEMENT_SEED)
    return world


def _seat_where_forward_is_clear(world, yaw_rad):
    """Find a navigable pose from which one forward step actually moves the agent.

    Returns ``(pose_before, pose_after)``. Skips rather than fails if every sampled pose
    is against geometry: this test is about the direction of a step that happened, and a
    scene where none happens is a scene problem, not a frame result.
    """
    for attempt in range(POSE_TRIES):
        world.set_pose(world.random_navigable_point(), _yaw_quaternion(yaw_rad))
        before = world.pose()
        world.step("move_forward")
        after = world.pose()
        moved = before.position.horizontal_distance_to(after.position)
        if moved >= MIN_DISPLACEMENT_M:
            print("    seated after {} tries, moved {:.4f} m".format(attempt + 1, moved), flush=True)
            return before, after
    raise unittest.SkipTest(
        "no navigable pose in {} tries left room for a forward step at yaw {:.3f}".format(
            POSE_TRIES, yaw_rad
        )
    )


class TestTheAgentFrame(unittest.TestCase):
    """The decisive measurement: act, then look at where the agent went."""

    @classmethod
    def setUpClass(cls):
        cls.world = _new_world()

    @classmethod
    def tearDownClass(cls):
        cls.world.close()

    def test_the_seated_yaw_round_trips(self):
        """``set_pose`` then ``pose()`` must agree, or every number below is about a
        different heading than the one requested."""
        print("\n  --- yaw round-trip ---", flush=True)
        for yaw_deg in (0.0, 30.0, 90.0, 175.0, -60.0):
            yaw = math.radians(yaw_deg)
            self.world.set_pose(self.world.random_navigable_point(), _yaw_quaternion(yaw))
            read = self.world.pose().yaw_rad
            print("    set {:>7.2f} deg -> read {:>7.2f} deg".format(
                yaw_deg, math.degrees(read)), flush=True)
            self.assertAlmostEqual(wrap_pi(read - yaw), 0.0, places=4)

    def test_move_forward_goes_where_forward_xz_says(self):
        """The whole finding, in one assertion. The old convention predicts 180 degrees off."""
        print("\n  --- forward direction ---", flush=True)
        for yaw_deg in (0.0, 90.0, 180.0, -90.0):
            yaw = math.radians(yaw_deg)
            before, after = self._step_at(yaw)
            dx = after.position.x - before.position.x
            dz = after.position.z - before.position.z
            expected_x, expected_z = forward_xz(before.yaw_rad)
            travelled = math.hypot(dx, dz)
            cosine = (dx * expected_x + dz * expected_z) / travelled
            error_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
            print(
                "    yaw {:>7.2f} deg: moved ({:+.4f}, {:+.4f}) = {:.4f} m, "
                "predicted ({:+.4f}, {:+.4f}), error {:.2f} deg".format(
                    yaw_deg, dx, dz, travelled, expected_x, expected_z, error_deg
                ),
                flush=True,
            )
            self.assertLess(
                error_deg,
                HEADING_TOLERANCE_DEG,
                "at yaw {:.2f} deg the agent moved {:.1f} degrees off the heading "
                "agent/occupancy.forward_xz predicts. Near 180 means the frame is "
                "inverted, which is the defect the old tree carried: the map, the "
                "candidate bearing and the detector's back-projection would all be "
                "point-reflected through the agent and all agree with each other."
                .format(yaw_deg, error_deg),
            )

    def test_turn_left_increases_the_yaw(self):
        """Which is why ``bearing_rel`` is positive-is-left and the stall rule turns as it does."""
        print("\n  --- turn sense ---", flush=True)
        self.world.set_pose(self.world.random_navigable_point(), _yaw_quaternion(0.0))
        before = self.world.pose().yaw_rad
        self.world.step("turn_left")
        left_delta = math.degrees(wrap_pi(self.world.pose().yaw_rad - before))
        self.world.step("turn_right")
        self.world.step("turn_right")
        right_delta = math.degrees(wrap_pi(self.world.pose().yaw_rad - before))
        print("    turn_left  {:+.2f} deg".format(left_delta), flush=True)
        print("    turn_right {:+.2f} deg (net, from the same start)".format(right_delta), flush=True)
        self.assertGreater(left_delta, 0.0, "turn_left must be a positive rotation about +y")
        self.assertLess(right_delta, 0.0, "turn_right must be a negative rotation about +y")
        self.assertAlmostEqual(abs(left_delta), 30.0, delta=TURN_TOLERANCE_DEG)

    def test_the_depth_splat_carves_in_front_of_the_agent(self):
        """The map, on a real render. A mirrored frame carves the half-plane behind."""
        print("\n  --- occupancy half-planes ---", flush=True)
        cfg = PlannerConfig()
        self.world.set_pose(self.world.random_navigable_point(), _yaw_quaternion(0.0))
        pose = self.world.pose()
        depth = self.world.observe()["depth"]
        grid = integrate_depth(new_grid(pose.position, cfg), depth, pose, cfg)

        ahead = behind = 0
        forward_x, forward_z = forward_xz(pose.yaw_rad)
        for row in range(grid.n):
            for col in range(grid.n):
                if int(grid.cells[row, col]) != CELL_FREE:
                    continue
                x, z = grid.grid_to_world(row, col)
                along = (x - pose.position.x) * forward_x + (z - pose.position.z) * forward_z
                if along > 0.0:
                    ahead += 1
                elif along < 0.0:
                    behind += 1
        print("    free cells ahead {}, behind {}".format(ahead, behind), flush=True)
        self.assertGreater(ahead, 0, "the splat carved nothing at all — check the depth frame")
        self.assertGreater(
            ahead,
            4 * max(1, behind),
            "the camera sees only forward, so free space must be overwhelmingly ahead. "
            "A dominant BEHIND count is the inverted frame the old tree carried",
        )

    def test_a_back_projected_pixel_lands_in_front_and_between_floor_and_sensor(self):
        """The detector's half of the same frame, which had two errors rather than one.

        **Two of the four numbers printed are internal consistency, not measurement**, and
        saying so is the point: the centre pixel sits exactly on the principal point
        (``cx = width / 2``), so ``along`` is the depth and ``lateral`` is zero for *any*
        frame convention — computed from the same ``forward_xz`` the transform used. They
        are kept because they would catch a future refactor desynchronising the two, and
        the Mac suite already pins that.

        The two with box content, one for each of the old file's errors:

        - ``along > 0`` — the sign of the optical axis. The old ``+z`` convention puts the
          point behind the agent.
        - the **bottom-centre** pixel's height sits between the floor and the sensor. It
          looks downward and so overwhelmingly hits the floor or something on it, and the
          old transform — built from the agent's base position — placed it 0.88 m *below*
          the navmesh. That is measured against the simulator's own sensor placement,
          because the depth and the pose both come from it.
        """
        print("\n  --- back-projection ---", flush=True)
        cfg = PlannerConfig()
        self.world.set_pose(self.world.random_navigable_point(), _yaw_quaternion(0.0))
        pose = self.world.pose()
        depth = self.world.observe()["depth"]
        frame = depth[..., 0] if getattr(depth, "ndim", 2) == 3 else depth
        height, width = int(frame.shape[0]), int(frame.shape[1])
        intrinsics = intrinsics_from_hfov(width, height, cfg.forward_fov_deg)
        transform = camera_to_world(pose, cfg.eye_height_m)

        u, v = width // 2, height // 2
        centre_depth = float(frame[v, u])
        if not (centre_depth > 0.0) or not math.isfinite(centre_depth):
            self.skipTest("the centre pixel has no valid depth at this pose")
        point = back_project_pinhole(u, v, centre_depth, intrinsics, transform)
        forward_x, forward_z = forward_xz(pose.yaw_rad)
        right_x, right_z = right_xz(pose.yaw_rad)
        dx, dz = point.x - pose.position.x, point.z - pose.position.z
        along = dx * forward_x + dz * forward_z
        lateral = dx * right_x + dz * right_z
        print(
            "    centre pixel: depth {:.4f} m -> along {:+.4f} m, lateral {:+.4f} m "
            "(the last two are internal consistency)".format(centre_depth, along, lateral),
            flush=True,
        )
        self.assertGreater(along, 0.0, "the back-projected point is BEHIND the agent")
        self.assertAlmostEqual(along, centre_depth, delta=0.2 + 0.05 * centre_depth)
        self.assertLess(abs(lateral), 0.2 + 0.05 * centre_depth)

        floor_v = height - 1
        floor_depth = float(frame[floor_v, u])
        if not (floor_depth > 0.0) or not math.isfinite(floor_depth):
            self.skipTest("the bottom-centre pixel has no valid depth at this pose")
        floor_point = back_project_pinhole(u, floor_v, floor_depth, intrinsics, transform)
        above_agent = floor_point.y - pose.position.y
        print(
            "    bottom-centre pixel: depth {:.4f} m -> {:+.4f} m above the agent "
            "(sensor is at {:+.4f})".format(floor_depth, above_agent, cfg.eye_height_m),
            flush=True,
        )
        self.assertGreater(
            above_agent,
            -0.5,
            "a downward-looking pixel back-projected BELOW the floor the agent stands on. "
            "The old transform used the agent's base as the camera origin, which put every "
            "detection {:.2f} m low".format(cfg.eye_height_m),
        )
        self.assertLess(
            above_agent,
            cfg.eye_height_m,
            "a downward-looking pixel cannot land at or above the sensor",
        )

    # -- helpers ---------------------------------------------------------

    def _step_at(self, yaw_rad):
        return _seat_where_forward_is_clear(self.world, yaw_rad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
