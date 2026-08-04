"""The detector seam: both arms, the carried helpers, and each rejection reason.

``OracleDetector`` is what ticket 23 asks to be green, and it is small because the
simulator's geometry arrives as one injected callable. ``CaptionDetector`` ships **live
but untested until R2** in the sense that matters — no run has ever exercised it against a
real VLM — but every gate it applies is pure and is pinned here, which is the difference
between "untested" and "unwritten".

The back-projection cases are the ones that carry weight, because two frame errors in the
old file put every detection behind the agent and 0.88 m too low (``occupancy.py`` has the
derivation). A fake grounder licenses nothing about what Qwen2-VL actually emits — the
parse cases are built from the two formats observed on RACE — but it licenses everything
about what happens to a box once it exists.
"""

import math
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.config import DetectorConfig
from earshot.agent.detector import (
    CaptionDetector,
    Detection,
    GoalDetector,
    OracleDetector,
    back_project_pinhole,
    parse_qwen_bbox,
    robust_depth_at_pixel,
)
from earshot.agent.occupancy import camera_to_world, forward_xz, intrinsics_from_hfov
from earshot.types import Pose, Xyz

POSE = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)


class StubGrounder:
    """Returns a canned decode. The structural interface ``vlm.py`` will satisfy."""

    def __init__(self, text=""):
        self.text = text
        self.calls = []

    def ground(self, image, obj):
        self.calls.append(obj)
        return self.text


def box_tokens(x1, y1, x2, y2, paren=False):
    inner = "({},{}),({},{})" if paren else "{},{},{},{}"
    return "<|box_start|>" + inner.format(x1, y1, x2, y2) + "<|box_end|>"


class TestBboxParsing(unittest.TestCase):
    def test_the_flat_format(self):
        self.assertEqual(
            parse_qwen_bbox(box_tokens(10, 20, 30, 40), image_hw=(64, 64)),
            [(10, 20, 30, 40)],
        )

    def test_the_documented_paren_format(self):
        self.assertEqual(
            parse_qwen_bbox(box_tokens(10, 20, 30, 40, paren=True), image_hw=(64, 64)),
            [(10, 20, 30, 40)],
        )

    def test_normalized_coordinates_are_detected_and_scaled(self):
        """The RACE c4 observation: a 256x256 input producing (452,414),(586,586)."""
        parsed = parse_qwen_bbox(
            box_tokens(452, 414, 586, 586, paren=True), image_hw=(256, 256)
        )
        self.assertEqual(parsed, [(116, 106, 150, 150)])

    def test_the_interpretation_can_be_forced(self):
        """The image must NOT be 1000x1000, or the scale factor is 1.0 and the parameter
        makes no difference — the first version of this test was vacuous for that reason."""
        text = box_tokens(100, 100, 200, 200)
        self.assertEqual(
            parse_qwen_bbox(text, image_hw=(500, 500), normalized=True),
            [(50, 50, 100, 100)],
        )
        self.assertEqual(
            parse_qwen_bbox(text, image_hw=(500, 500), normalized=False),
            [(100, 100, 200, 200)],
        )

    def test_prose_with_no_box_parses_to_nothing(self):
        self.assertEqual(parse_qwen_bbox("I can see a chair.", image_hw=(64, 64)), [])

    def test_an_empty_or_none_decode_parses_to_nothing(self):
        self.assertEqual(parse_qwen_bbox("", image_hw=(64, 64)), [])
        self.assertEqual(parse_qwen_bbox(None, image_hw=(64, 64)), [])

    def test_a_zero_area_box_is_dropped(self):
        self.assertEqual(parse_qwen_bbox(box_tokens(30, 30, 30, 40), image_hw=(64, 64)), [])

    def test_a_pixel_space_box_overhanging_the_frame_is_dropped(self):
        """``normalized=False`` is load-bearing: with auto-detect, a box whose coordinates
        exceed the image is read as normalized and scaled to zero area, so the *bounds*
        branch never runs and deleting it would leave the suite green. An overhanging box
        that survived would reach ``robust_depth_at_pixel``, whose edge clipping silently
        samples the frame border instead of rejecting."""
        self.assertEqual(
            parse_qwen_bbox(box_tokens(10, 10, 20, 20), image_hw=(15, 15), normalized=False),
            [],
        )
        self.assertEqual(
            parse_qwen_bbox(box_tokens(2, 2, 12, 12), image_hw=(15, 15), normalized=False),
            [(2, 2, 12, 12)],
        )

    def test_a_format_hint_with_placeholder_coordinates_parses_to_nothing(self):
        """The decode includes the prompt echo, and the prompt inlines the box tokens.

        The old path was safe only because the hint spells the coordinates as
        ``(x1,y1),(x2,y2)`` and the pattern requires digits. A hint written with real
        numbers would make a phantom detection out of the prompt — see ``Grounder``.
        """
        prompt_echo = (
            "Locate the chair in this image. Output the bounding box as "
            "<|object_ref_start|>chair<|object_ref_end|>"
            "<|box_start|>(x1,y1),(x2,y2)<|box_end|>."
        )
        self.assertEqual(parse_qwen_bbox(prompt_echo, image_hw=(64, 64)), [])

    def test_several_boxes_all_parse(self):
        text = box_tokens(1, 1, 5, 5) + " and " + box_tokens(10, 10, 20, 20)
        self.assertEqual(len(parse_qwen_bbox(text, image_hw=(64, 64))), 2)


class TestRobustDepth(unittest.TestCase):
    def test_the_median_of_the_window_is_returned(self):
        depth = np.arange(25, dtype=np.float32).reshape(5, 5) + 1.0
        self.assertEqual(robust_depth_at_pixel(depth, u=2, v=2, patch=5), 13.0)

    def test_habitats_zero_no_return_is_rejected(self):
        depth = np.zeros((5, 5), dtype=np.float32)
        depth[2, 2] = 4.0
        self.assertEqual(robust_depth_at_pixel(depth, u=2, v=2, patch=5), 4.0)

    def test_nan_and_inf_are_rejected(self):
        depth = np.full((5, 5), np.nan, dtype=np.float32)
        depth[0, 0] = np.inf
        depth[2, 2] = 2.0
        self.assertEqual(robust_depth_at_pixel(depth, u=2, v=2, patch=5), 2.0)

    def test_an_entirely_invalid_window_is_none(self):
        self.assertIsNone(robust_depth_at_pixel(np.zeros((5, 5)), u=2, v=2))

    def test_the_window_is_clipped_at_the_image_edge(self):
        depth = np.full((5, 5), 3.0, dtype=np.float32)
        self.assertEqual(robust_depth_at_pixel(depth, u=0, v=0, patch=5), 3.0)

    def test_an_hw1_frame_is_accepted(self):
        depth = np.full((5, 5, 1), 3.0, dtype=np.float32)
        self.assertEqual(robust_depth_at_pixel(depth, u=2, v=2), 3.0)


class TestBackProjection(unittest.TestCase):
    def setUp(self):
        self.intr = intrinsics_from_hfov(64, 64, 90.0)

    def test_the_principal_point_projects_straight_ahead(self):
        """In FRONT of the agent. The old file's ``+z`` put it behind."""
        point = back_project_pinhole(
            32, 32, 3.0, self.intr, camera_to_world(POSE, 0.88)
        )
        fx, fz = forward_xz(POSE.yaw_rad)
        self.assertAlmostEqual(point.x, fx * 3.0, places=6)
        self.assertAlmostEqual(point.z, fz * 3.0, places=6)

    def test_the_height_is_measured_from_the_sensor(self):
        point = back_project_pinhole(32, 32, 3.0, self.intr, camera_to_world(POSE, 0.88))
        self.assertAlmostEqual(point.y, 0.88, places=6)

    def test_a_pixel_to_the_right_lands_on_the_agents_right(self):
        point = back_project_pinhole(60, 32, 3.0, self.intr, camera_to_world(POSE, 0.88))
        self.assertGreater(point.x, 0.0)

    def test_a_pixel_above_centre_lands_higher(self):
        point = back_project_pinhole(32, 5, 3.0, self.intr, camera_to_world(POSE, 0.88))
        self.assertGreater(point.y, 0.88)

    def test_the_projection_follows_the_yaw(self):
        for yaw in (0.0, math.pi / 2, -1.3, math.pi):
            pose = Pose(position=Xyz(1.0, 0.0, -2.0), yaw_rad=yaw)
            point = back_project_pinhole(32, 32, 4.0, self.intr, camera_to_world(pose, 0.88))
            fx, fz = forward_xz(yaw)
            self.assertAlmostEqual(point.x, 1.0 + fx * 4.0, places=5)
            self.assertAlmostEqual(point.z, -2.0 + fz * 4.0, places=5)

    def test_an_invalid_depth_never_becomes_a_coordinate(self):
        transform = camera_to_world(POSE, 0.88)
        for depth in (0.0, -1.0, float("nan"), float("inf"), None):
            self.assertIsNone(back_project_pinhole(32, 32, depth, self.intr, transform))


class TestOracleDetector(unittest.TestCase):
    def test_it_satisfies_the_seam(self):
        self.assertIsInstance(OracleDetector(lambda obj: 1.0), GoalDetector)

    def test_within_the_radius_detects(self):
        detector = OracleDetector(lambda obj: 0.4)
        self.assertTrue(detector.detects("chair"))
        self.assertEqual(detector.n_detections, 1)

    def test_outside_the_radius_does_not(self):
        detector = OracleDetector(lambda obj: 4.0)
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_distance_m, 4.0)

    def test_the_boundary_counts_as_detected(self):
        cfg = DetectorConfig(oracle_radius_m=1.0)
        self.assertTrue(OracleDetector(lambda obj: 1.0, cfg).detects("chair"))

    def test_an_unanswerable_query_is_not_a_detection(self):
        """Treating it as one would STOP on a missing goal and score it a success."""
        detector = OracleDetector(lambda obj: None)
        self.assertFalse(detector.detects("chair"))
        self.assertIsNone(detector.last_distance_m)

    def test_observe_is_a_no_op_rather_than_an_error(self):
        detector = OracleDetector(lambda obj: 0.2)
        detector.observe(rgb=None, depth=None, pose=POSE)
        self.assertTrue(detector.detects("chair"))

    def test_the_query_carries_the_object_name(self):
        seen = []
        detector = OracleDetector(lambda obj: seen.append(obj) or 0.1)
        detector.detects("toilet")
        self.assertEqual(seen, ["toilet"])


class TestCaptionDetector(unittest.TestCase):
    def _detector(self, text, snap=lambda p: p, cfg=None):
        return CaptionDetector(
            StubGrounder(text),
            snap,
            cfg=cfg or DetectorConfig(),
            hfov_deg=90.0,
            eye_height_m=0.88,
        )

    def _frame(self, depth_value=0.6, size=64):
        return {
            "rgb": np.zeros((size, size, 3), dtype=np.uint8),
            "depth": np.full((size, size), float(depth_value), dtype=np.float32),
            "pose": POSE,
        }

    def test_it_satisfies_the_seam(self):
        self.assertIsInstance(self._detector(""), GoalDetector)

    def test_a_believable_detection_within_reach(self):
        detector = self._detector(box_tokens(28, 28, 36, 36))
        detector.observe(**self._frame(depth_value=0.6))
        self.assertTrue(detector.detects("chair"))
        self.assertIsInstance(detector.last_detection, Detection)
        self.assertEqual(detector.last_detection.object_name, "chair")
        self.assertAlmostEqual(detector.last_detection.depth_m, 0.6, places=6)

    def test_no_frame_yet_is_a_rejection_not_a_crash(self):
        detector = self._detector(box_tokens(28, 28, 36, 36))
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_rejection, "no_frame")

    def test_prose_only_is_an_empty_parse(self):
        detector = self._detector("there is no chair here")
        detector.observe(**self._frame())
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_rejection, "empty_parse")

    def test_an_all_invalid_depth_window_is_rejected(self):
        detector = self._detector(box_tokens(28, 28, 36, 36))
        frame = self._frame()
        frame["depth"] = np.zeros((64, 64), dtype=np.float32)
        detector.observe(**frame)
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_rejection, "all_depths_invalid")

    def test_an_off_navmesh_point_is_rejected(self):
        detector = self._detector(box_tokens(28, 28, 36, 36), snap=lambda p: None)
        detector.observe(**self._frame())
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_rejection, "off_navmesh")

    def test_a_depth_overshoot_below_the_floor_is_rejected_first(self):
        """The L3 case: 0.76 m below the navmesh with only 0.21 m of horizontal offset.

        The floor pre-filter has to run before the floor-plane gate, or a small horizontal
        offset rescues a point that is underground.
        """
        detector = self._detector(
            box_tokens(28, 28, 36, 36), snap=lambda p: Xyz(p.x + 0.21, p.y + 0.76, p.z)
        )
        detector.observe(**self._frame())
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_rejection, "below_floor")

    def test_an_elevated_object_is_not_rejected_for_being_elevated(self):
        """The bug ``3307f19`` fixed: a chair seat snaps DOWN to the navmesh, and the 3D
        gate rejected exactly those. The gate is on the floor plane."""
        detector = self._detector(
            box_tokens(28, 28, 36, 36), snap=lambda p: Xyz(p.x, p.y - 0.85, p.z)
        )
        detector.observe(**self._frame())
        self.assertTrue(detector.detects("chair"))

    def test_a_far_horizontal_snap_is_rejected(self):
        detector = self._detector(
            box_tokens(28, 28, 36, 36), snap=lambda p: Xyz(p.x + 2.0, p.y, p.z)
        )
        detector.observe(**self._frame())
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_rejection, "snap_too_far")

    def test_an_object_seen_but_out_of_reach_is_not_here(self):
        """The piece that makes ``locate()`` fit ``detects()``: the seam asks "here"."""
        detector = self._detector(box_tokens(28, 28, 36, 36))
        detector.observe(**self._frame(depth_value=4.0))
        self.assertFalse(detector.detects("chair"))
        self.assertEqual(detector.last_rejection, "out_of_reach")

    def test_the_closest_box_wins(self):
        text = box_tokens(28, 28, 36, 36) + box_tokens(2, 2, 10, 10)
        detector = self._detector(text)
        frame = self._frame(depth_value=4.0)
        frame["depth"][2:12, 2:12] = 0.5  # the corner box is much nearer
        detector.observe(**frame)
        self.assertTrue(detector.detects("chair"))
        self.assertAlmostEqual(detector.last_detection.depth_m, 0.5, places=6)

    def test_rejections_are_counted_by_reason(self):
        """The old failure log wrote JSON lines; ``report/artifacts.py`` is the only writer."""
        detector = self._detector("no box here")
        detector.observe(**self._frame())
        detector.detects("chair")
        detector.detects("bed")
        self.assertEqual(detector.rejections["empty_parse"], 2)
        self.assertEqual(detector.n_queries, 2)
        self.assertEqual(detector.n_detections, 0)

    def test_a_failed_query_clears_the_previous_detection(self):
        """Otherwise a stale ``last_detection`` reads as this step's visual confirm."""
        detector = self._detector(box_tokens(28, 28, 36, 36))
        detector.observe(**self._frame(depth_value=0.6))
        self.assertTrue(detector.detects("chair"))
        detector.observe(**self._frame(depth_value=4.0))
        self.assertFalse(detector.detects("chair"))
        self.assertIsNone(detector.last_detection)

    def test_the_object_name_reaches_the_grounder(self):
        grounder = StubGrounder(box_tokens(28, 28, 36, 36))
        detector = CaptionDetector(grounder, lambda p: p, hfov_deg=90.0)
        detector.observe(**self._frame(depth_value=0.6))
        detector.detects("plant")
        self.assertEqual(grounder.calls, ["plant"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
