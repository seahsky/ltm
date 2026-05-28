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
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
