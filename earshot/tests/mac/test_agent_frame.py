"""The frame convention, asserted by comparing the tree's two independent consumers.

Ticket 09's finding was that the lateral sign inverted from world frame to agent frame
under live rendering **with no code change** — a convention written down in two places and
checked in none. This is the missing check, in the only form available: ``agent/`` may not
import ``audio/`` (ADR-0013), so nothing inside the package can compare them, but a test
may import both.

What it pins is not "the arithmetic is what we typed". It is that ``agent/occupancy.py``'s
right axis, the bearing sign its greedy turn rule reads, and the camera transform its
detector back-projects through all describe the **same** agent as
``audio/lateral.bearing_lateral_sign`` does. The old tree's occupancy splat and candidate
bearing were 180 degrees out from habitat's heading (``occupancy.py``'s docstring has the
derivation and the recorded symptoms), and a test of this shape is what would have caught
it: the audio side and the planner side disagreed, and both looked plausible alone.

``bearing_lateral_sign`` is analyst-only for the *controller* (§3.3) because it reads the
ground-truth source position. A test is not the controller, and it is the geometry oracle
here rather than a signal.
"""

import ast
import math
import unittest

import _tree
from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.config import PlannerConfig
from earshot.agent.occupancy import (
    bearing_rel,
    camera_to_world,
    forward_xz,
    heading_to,
    intrinsics_from_hfov,
    right_xz,
    wrap_pi,
)
from earshot.audio.lateral import (
    ILD_DEAD_ZONE,
    LATERAL_LEFT,
    LATERAL_RIGHT,
    bearing_lateral_sign,
)
from earshot.types import Pose, Xyz

YAWS = [0.0, 0.4, math.pi / 2, 2.0, math.pi, -0.7, -math.pi / 2, 3.0]


def pose_at(yaw, x=0.0, z=0.0, y=0.0):
    return Pose(position=Xyz(x, y, z), yaw_rad=yaw)


class TestTheTwoFrameConsumersAgree(unittest.TestCase):
    def test_the_right_axis_matches_the_audio_layers(self):
        """``occupancy.right_xz`` and ``lateral.bearing_lateral_sign`` name one axis."""
        for yaw in YAWS:
            rx, rz = right_xz(yaw)
            for offset in (Xyz(1.0, 0.0, 0.0), Xyz(0.0, 0.0, 1.5), Xyz(-2.0, 0.0, 0.7)):
                pose = pose_at(yaw)
                lateral = offset.x * rx + offset.z * rz
                # Three-way, because a source dead ahead is neither side and both
                # modules abstain on it — comparing only the two signs would fail on
                # agreement rather than on disagreement.
                expected = 0
                if abs(lateral) >= ILD_DEAD_ZONE:
                    expected = LATERAL_RIGHT if lateral > 0 else LATERAL_LEFT
                self.assertEqual(
                    bearing_lateral_sign(pose, offset),
                    expected,
                    "yaw {:.3f}, offset {}: the planner's right axis and the audio "
                    "layer's disagree".format(yaw, offset),
                )

    def test_a_target_to_the_right_has_a_negative_relative_bearing(self):
        """The two sign conventions have to compose, not merely each be self-consistent.

        ``bearing_rel`` is positive-is-left because habitat's ``turn_left`` is a positive
        rotation about ``+y``; ``lateral_sign`` is positive-is-right because that is what
        the ear order gives. So a source the audio layer calls ``+1`` must be one the turn
        rule steers *right* toward, and the controller's stall branch reads exactly that
        pair.
        """
        for yaw in YAWS:
            rx, rz = right_xz(yaw)
            target = Xyz(rx * 3.0, 0.0, rz * 3.0)
            self.assertEqual(bearing_lateral_sign(pose_at(yaw), target), LATERAL_RIGHT)
            self.assertLess(bearing_rel(yaw, target.x, target.z), 0.0)

    def test_straight_ahead_is_zero_bearing_and_zero_lateral(self):
        for yaw in YAWS:
            fx, fz = forward_xz(yaw)
            self.assertAlmostEqual(bearing_rel(yaw, fx * 4.0, fz * 4.0), 0.0, places=9)
            self.assertEqual(bearing_lateral_sign(pose_at(yaw), Xyz(fx, 0.0, fz)), 0)


class TestTheFrameItself(unittest.TestCase):
    def test_forward_is_minus_z_and_right_is_plus_x_at_zero_yaw(self):
        """Habitat's convention, and the half-turn the old tree had wrong."""
        self.assertAlmostEqual(forward_xz(0.0)[0], 0.0, places=12)
        self.assertAlmostEqual(forward_xz(0.0)[1], -1.0, places=12)
        self.assertAlmostEqual(right_xz(0.0)[0], 1.0, places=12)
        self.assertAlmostEqual(right_xz(0.0)[1], 0.0, places=12)

    def test_forward_and_right_are_unit_and_perpendicular(self):
        for yaw in YAWS:
            fx, fz = forward_xz(yaw)
            rx, rz = right_xz(yaw)
            self.assertAlmostEqual(math.hypot(fx, fz), 1.0, places=12)
            self.assertAlmostEqual(math.hypot(rx, rz), 1.0, places=12)
            self.assertAlmostEqual(fx * rx + fz * rz, 0.0, places=12)

    def test_heading_to_inverts_forward(self):
        for yaw in YAWS:
            fx, fz = forward_xz(yaw)
            self.assertAlmostEqual(wrap_pi(heading_to(fx, fz) - yaw), 0.0, places=9)

    def test_increasing_yaw_turns_left(self):
        """Which is why ``bearing_rel`` is positive-is-left, and why ``turn_left`` is +yaw.

        A target on the agent's left must need a positive yaw change to face.
        """
        rx, rz = right_xz(0.0)
        left = Xyz(-rx * 2.0, 0.0, -rz * 2.0)
        self.assertGreater(bearing_rel(0.0, left.x, left.z), 0.0)
        self.assertEqual(bearing_lateral_sign(pose_at(0.0), left), LATERAL_LEFT)


class TestTheCameraTransform(unittest.TestCase):
    def test_the_transform_columns_are_the_frames_axes(self):
        """A detection back-projected through it lands where the map says forward is."""
        for yaw in YAWS:
            transform = camera_to_world(pose_at(yaw), 0.88)
            fx, fz = forward_xz(yaw)
            rx, rz = right_xz(yaw)
            self.assertAlmostEqual(transform[0, 0], rx, places=12)
            self.assertAlmostEqual(transform[2, 0], rz, places=12)
            # The camera's +z is BACKWARD: the optical axis is -z.
            self.assertAlmostEqual(transform[0, 2], -fx, places=12)
            self.assertAlmostEqual(transform[2, 2], -fz, places=12)

    def test_the_origin_is_the_sensor_not_the_agents_feet(self):
        """The 0.88 m the old ``_agent_pose_matrix`` left out."""
        transform = camera_to_world(pose_at(0.3, x=2.0, z=-5.0, y=0.1), 0.88)
        self.assertAlmostEqual(transform[0, 3], 2.0, places=12)
        self.assertAlmostEqual(transform[1, 3], 0.98, places=12)
        self.assertAlmostEqual(transform[2, 3], -5.0, places=12)

    def test_the_bottom_row_is_affine(self):
        transform = camera_to_world(pose_at(1.1), 0.88)
        self.assertEqual(list(transform[3, :]), [0.0, 0.0, 0.0, 1.0])


class TestTheCameraNumbersAgreeWithTheSensorSpec(unittest.TestCase):
    """``PlannerConfig`` and ``sim/world.camera_sensor_specs`` hold the same two numbers.

    Four independent literals, and ``config.py`` says a disagreement "silently scales the
    map" — with no runtime check, because ``agent/`` may not import ``sim`` (ADR-0013). But
    a **test** sits outside the layer graph, and a static one sits outside the Mac/box split
    too: ``sim/world.py`` cannot be imported on this machine, so the defaults are read out
    of its ``ast``. Same mechanism as ``test_layering.py``, for the same reason — the thing
    being checked is the real source rather than a copy of it.

    Failure scenario this closes: someone widens the sensor to ``hfov=90`` for a better
    field of view, the splat keeps back-projecting at 79 degrees, every ray lands at the
    wrong angle, the map skews, and every test stays green.
    """

    def _camera_spec_defaults(self):
        tree = _tree.parse(_tree.PACKAGE_ROOT / "sim" / "world.py")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "camera_sensor_specs":
                names = [arg.arg for arg in node.args.kwonlyargs]
                values = [
                    default.value if isinstance(default, ast.Constant) else None
                    for default in node.args.kw_defaults
                ]
                return dict(zip(names, values))
        self.fail("sim/world.py has no camera_sensor_specs — the check below is vacuous")

    def test_the_field_of_view_matches(self):
        self.assertEqual(
            PlannerConfig().forward_fov_deg, self._camera_spec_defaults()["hfov"]
        )

    def test_the_eye_height_matches(self):
        self.assertEqual(
            PlannerConfig().eye_height_m, self._camera_spec_defaults()["eye_height"]
        )

    def test_both_names_were_actually_found(self):
        """A renamed keyword would make the two tests above pass on a ``None``."""
        defaults = self._camera_spec_defaults()
        for name in ("hfov", "eye_height"):
            self.assertIn(name, defaults)
            self.assertIsNotNone(defaults[name])


class TestIntrinsics(unittest.TestCase):
    def test_focal_length_from_hfov_and_square_pixels(self):
        intr = intrinsics_from_hfov(640, 480, 90.0)
        self.assertAlmostEqual(intr.fx, 320.0, places=9)
        self.assertAlmostEqual(intr.fy, 320.0, places=9)
        self.assertEqual((intr.cx, intr.cy), (320.0, 240.0))

    def test_a_non_square_sensor_is_fine(self):
        """The old comment claimed a square sensor was assumed; square *pixels* suffice."""
        intr = intrinsics_from_hfov(640, 480, 79.0)
        self.assertAlmostEqual(intr.fx, intr.fy, places=12)
        self.assertGreater(intr.fx, 0.0)

    def test_a_degenerate_image_raises(self):
        with self.assertRaises(ValueError):
            intrinsics_from_hfov(0, 480, 79.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
