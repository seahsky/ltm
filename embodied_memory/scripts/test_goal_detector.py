"""
Layer-1 sanity tests for goal_detector — pure parser + geometry helpers.
Stdlib + numpy only; no Qwen-VL, no Habitat.

Invoke with::

    python embodied_memory/scripts/test_goal_detector.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import goal_detector as gd  # noqa: E402


def case_parse_qwen_bbox_well_formed():
    s = "Sure, the chair is at <|box_start|>100,200,300,400<|box_end|>."
    out = gd.parse_qwen_bbox(s, image_hw=(480, 640))
    assert out == [(100, 200, 300, 400)], out
    print("  case_parse_qwen_bbox_well_formed: OK")


def case_parse_qwen_bbox_normalized():
    s = "<|box_start|>500,500,750,750<|box_end|>"
    out = gd.parse_qwen_bbox(s, image_hw=(400, 800), normalized=True)
    assert out == [(400, 200, 600, 300)], out
    print("  case_parse_qwen_bbox_normalized: OK")


def case_parse_no_bbox_returns_empty():
    assert gd.parse_qwen_bbox("no box here", image_hw=(480, 640)) == []
    print("  case_parse_no_bbox_returns_empty: OK")


def case_parse_malformed_bbox_returns_empty():
    s = "<|box_start|>100,200,abc<|box_end|>"
    assert gd.parse_qwen_bbox(s, image_hw=(480, 640)) == []
    print("  case_parse_malformed_bbox_returns_empty: OK")


def case_parse_multi_bbox_returns_all():
    s = "<|box_start|>10,20,30,40<|box_end|> and <|box_start|>50,60,70,80<|box_end|>"
    out = gd.parse_qwen_bbox(s, image_hw=(480, 640))
    assert out == [(10, 20, 30, 40), (50, 60, 70, 80)], out
    print("  case_parse_multi_bbox_returns_all: OK")


def case_robust_depth_returns_median():
    depth = np.array([
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)
    d = gd.robust_depth_at_pixel(depth, u=2, v=2, patch=5)
    assert d == 2.0, d
    print("  case_robust_depth_returns_median: OK")


def case_robust_depth_ignores_nan_zero_inf():
    depth = np.array([
        [np.nan, 0.0, np.inf],
        [1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5],
    ], dtype=np.float32)
    d = gd.robust_depth_at_pixel(depth, u=1, v=1, patch=3)
    assert d == 1.5, d
    print("  case_robust_depth_ignores_nan_zero_inf: OK")


def case_robust_depth_returns_none_if_all_invalid():
    depth = np.array([
        [np.nan, 0.0],
        [0.0, np.nan],
    ], dtype=np.float32)
    d = gd.robust_depth_at_pixel(depth, u=0, v=0, patch=2)
    assert d is None, d
    print("  case_robust_depth_returns_none_if_all_invalid: OK")


def case_back_project_pinhole_identity_pose():
    intrinsics = {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}
    agent_pose = np.eye(4, dtype=np.float32)
    pt = gd.back_project_pinhole(u=128, v=128, depth=2.0,
                                  intrinsics=intrinsics, agent_pose=agent_pose)
    assert pt is not None
    assert np.allclose(pt, np.array([0.0, 0.0, -2.0], dtype=np.float32), atol=1e-5), pt
    print("  case_back_project_pinhole_identity_pose: OK")


def case_back_project_pinhole_translated_pose():
    intrinsics = {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}
    agent_pose = np.eye(4, dtype=np.float32)
    agent_pose[:3, 3] = np.array([5.0, 1.5, 3.0], dtype=np.float32)
    pt = gd.back_project_pinhole(u=128, v=128, depth=2.0,
                                  intrinsics=intrinsics, agent_pose=agent_pose)
    assert np.allclose(pt, np.array([5.0, 1.5, 1.0], dtype=np.float32), atol=1e-5), pt
    print("  case_back_project_pinhole_translated_pose: OK")


def case_back_project_pinhole_invalid_depth():
    intrinsics = {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}
    agent_pose = np.eye(4, dtype=np.float32)
    assert gd.back_project_pinhole(u=10, v=10, depth=float("nan"),
                                    intrinsics=intrinsics, agent_pose=agent_pose) is None
    assert gd.back_project_pinhole(u=10, v=10, depth=0.0,
                                    intrinsics=intrinsics, agent_pose=agent_pose) is None
    assert gd.back_project_pinhole(u=10, v=10, depth=float("inf"),
                                    intrinsics=intrinsics, agent_pose=agent_pose) is None
    print("  case_back_project_pinhole_invalid_depth: OK")


# ----------------------------------------------------------------------
# GoalDetector class (uses mock model + processor + pathfinder)
# ----------------------------------------------------------------------


class _MockModel:
    """Mock Qwen2-VL model — only role is to be passed through to
    _MockProcessor.batch_decode by the locate() pipeline."""
    def generate(self, **kwargs):
        return [[0]]   # token ids — content irrelevant, _MockProcessor decodes


class _MockProcessor:
    """Mock processor returning a pre-set decoded text on batch_decode."""
    def __init__(self, decoded_text: str):
        self.decoded_text = decoded_text
        self.eos_token_id = 0
    def apply_chat_template(self, messages, **kwargs):
        return "prompt"
    def __call__(self, **kwargs):
        class _Inputs:
            input_ids = np.array([[0]])
            def to(self, device):
                return self
        return _Inputs()
    def batch_decode(self, *args, **kwargs):
        return [self.decoded_text]


class _MockPathfinder:
    """Mock Habitat PathFinder — snap_point returns a configured target."""
    def __init__(self, snap_target):
        self.snap_target = np.asarray(snap_target, dtype=np.float32)
    def snap_point(self, pt):
        return self.snap_target


def _intrinsics():
    return {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}


def case_locate_returns_point_when_bbox_and_snap_ok():
    proc = _MockProcessor("the chair is at <|box_start|>120,120,160,160<|box_end|>")
    # back-project of center (140,140) at depth 2.0 with identity pose gives
    # roughly (0.094, -0.094, -2.0); we set the mock snap target near that.
    snap = np.array([0.1, 0.0, -2.0], dtype=np.float32)
    pathfinder = _MockPathfinder(snap_target=snap)
    det = gd.GoalDetector(_MockModel(), proc, pathfinder, max_snap_dist=0.5)
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.full((256, 256), 2.0, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    out = det.locate(rgb=rgb, depth=depth, goal_category="chair",
                     agent_pose=pose, intrinsics=_intrinsics())
    assert out is not None
    assert np.allclose(out, snap, atol=1e-5), out
    print("  case_locate_returns_point_when_bbox_and_snap_ok: OK")


def case_locate_returns_none_when_no_bbox_in_output():
    proc = _MockProcessor("I don't see a chair here.")
    pathfinder = _MockPathfinder(snap_target=np.zeros(3, dtype=np.float32))
    det = gd.GoalDetector(_MockModel(), proc, pathfinder)
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                     depth=np.full((256, 256), 2.0, dtype=np.float32),
                     goal_category="chair",
                     agent_pose=np.eye(4, dtype=np.float32),
                     intrinsics=_intrinsics())
    assert out is None
    print("  case_locate_returns_none_when_no_bbox_in_output: OK")


def case_locate_returns_none_when_depth_invalid_at_center():
    proc = _MockProcessor("<|box_start|>120,120,160,160<|box_end|>")
    pathfinder = _MockPathfinder(snap_target=np.zeros(3, dtype=np.float32))
    det = gd.GoalDetector(_MockModel(), proc, pathfinder)
    depth = np.zeros((256, 256), dtype=np.float32)   # all zero -> invalid
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                     depth=depth, goal_category="chair",
                     agent_pose=np.eye(4, dtype=np.float32),
                     intrinsics=_intrinsics())
    assert out is None
    print("  case_locate_returns_none_when_depth_invalid_at_center: OK")


def case_locate_returns_none_when_snap_too_far():
    proc = _MockProcessor("<|box_start|>120,120,160,160<|box_end|>")
    # snap_point returns a far-away point -> distance > max_snap_dist -> None
    far_snap = np.array([100.0, 0.0, 100.0], dtype=np.float32)
    pathfinder = _MockPathfinder(snap_target=far_snap)
    det = gd.GoalDetector(_MockModel(), proc, pathfinder, max_snap_dist=0.5)
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                     depth=np.full((256, 256), 2.0, dtype=np.float32),
                     goal_category="chair",
                     agent_pose=np.eye(4, dtype=np.float32),
                     intrinsics=_intrinsics())
    assert out is None
    print("  case_locate_returns_none_when_snap_too_far: OK")


def case_locate_picks_lowest_depth_bbox_among_multiple():
    # Two bboxes: bbox A at center (90,90), bbox B at center (140,140).
    # Depth: 3.0 everywhere except a small patch at (140,140) which is 1.0.
    # The detector should pick bbox B (lower depth -> closer surface) and
    # return a non-None snapped point.
    proc = _MockProcessor(
        "two chairs: <|box_start|>80,80,100,100<|box_end|> "
        "and <|box_start|>130,130,150,150<|box_end|>"
    )
    snap = np.array([0.5, 0.0, -1.0], dtype=np.float32)
    pathfinder = _MockPathfinder(snap_target=snap)
    det = gd.GoalDetector(_MockModel(), proc, pathfinder, max_snap_dist=10.0)
    depth = np.full((256, 256), 3.0, dtype=np.float32)
    depth[130:151, 130:151] = 1.0   # SECOND bbox is closer (lower depth)
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                     depth=depth, goal_category="chair",
                     agent_pose=np.eye(4, dtype=np.float32),
                     intrinsics=_intrinsics())
    assert out is not None
    assert np.allclose(out, snap, atol=1e-5)
    print("  case_locate_picks_lowest_depth_bbox_among_multiple: OK")


def main() -> int:
    print("goal_detector Layer-1 sanity tests (parser + geometry)")
    case_parse_qwen_bbox_well_formed()
    case_parse_qwen_bbox_normalized()
    case_parse_no_bbox_returns_empty()
    case_parse_malformed_bbox_returns_empty()
    case_parse_multi_bbox_returns_all()
    case_robust_depth_returns_median()
    case_robust_depth_ignores_nan_zero_inf()
    case_robust_depth_returns_none_if_all_invalid()
    case_back_project_pinhole_identity_pose()
    case_back_project_pinhole_translated_pose()
    case_back_project_pinhole_invalid_depth()
    case_locate_returns_point_when_bbox_and_snap_ok()
    case_locate_returns_none_when_no_bbox_in_output()
    case_locate_returns_none_when_depth_invalid_at_center()
    case_locate_returns_none_when_snap_too_far()
    case_locate_picks_lowest_depth_bbox_among_multiple()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
