"""
goal_detector — precise final-approach localization for ObjectNav.

When the captioner's keyword-STOP fires (the goal category is mentioned in
the current caption), this module asks Qwen2-VL to *locate* the goal in the
image, back-projects the bbox center through the depth sensor, and snaps the
result to the Habitat navmesh. The output is a 3D waypoint that the existing
``EpisodeRunner._waypoint_action`` follower navigates to before STOPping —
collapsing the 0.5-to-2 m caption-detection range into the 0.1 m success
ring.

This module exposes:
  - parse_qwen_bbox          — pure-string parsing of Qwen-VL bbox tokens
  - robust_depth_at_pixel    — 5x5-patch median, rejects NaN/0/inf
  - back_project_pinhole     — (u, v, depth) + pose -> 3D world point
  - GoalDetector             — the class that ties them together with the
                                already-loaded Qwen2-VL model (added in
                                Task 2)

Layer-1 tests (test_goal_detector.py) exercise these helpers without any
model or simulator. Layer-2 tests (test_episode_runner_detector.py) cover
the EpisodeRunner integration via mocks.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple, Dict

import numpy as np


# ----------------------------------------------------------------------
# Qwen-VL bbox parsing
# ----------------------------------------------------------------------


_BBOX_RE = re.compile(
    r"<\|box_start\|>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*<\|box_end\|>"
)


def parse_qwen_bbox(
    text: str,
    image_hw: Tuple[int, int],
    normalized: bool = False,
) -> List[Tuple[int, int, int, int]]:
    """Parse Qwen2-VL grounding bboxes from a text output.

    Qwen2-VL emits bboxes wrapped in ``<|box_start|>x1,y1,x2,y2<|box_end|>``.
    Coordinates are usually in pixel space matching the input image; some
    fine-tunes emit them in [0, 1000] normalized space (set ``normalized=True``
    to scale to pixels using ``image_hw``).

    Returns a list of ``(x1, y1, x2, y2)`` tuples (possibly empty). Malformed
    tokens are silently dropped (regex match failures); we never raise.
    """
    out: List[Tuple[int, int, int, int]] = []
    H, W = image_hw
    for m in _BBOX_RE.finditer(text):
        try:
            x1, y1, x2, y2 = (int(g) for g in m.groups())
        except ValueError:
            continue
        if normalized:
            x1 = int(round(x1 * W / 1000.0))
            x2 = int(round(x2 * W / 1000.0))
            y1 = int(round(y1 * H / 1000.0))
            y2 = int(round(y2 * H / 1000.0))
        # basic sanity: positive area, inside image
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            continue
        out.append((x1, y1, x2, y2))
    return out


# ----------------------------------------------------------------------
# Depth helpers
# ----------------------------------------------------------------------


def robust_depth_at_pixel(
    depth: np.ndarray,
    u: int,
    v: int,
    patch: int = 5,
) -> Optional[float]:
    """Median depth in a (patch x patch) window around (u, v).

    Rejects NaN, 0.0, and +/- inf (Habitat returns 0 for "no return" and inf
    can appear near the far-clip plane). Returns None if every pixel in the
    window is invalid.
    """
    h, w = depth.shape[:2]
    half = patch // 2
    u0, u1 = max(u - half, 0), min(u + half + 1, w)
    v0, v1 = max(v - half, 0), min(v + half + 1, h)
    window = np.asarray(depth[v0:v1, u0:u1], dtype=np.float64).ravel()
    finite = np.isfinite(window) & (window > 0.0)
    if not finite.any():
        return None
    return float(np.median(window[finite]))


# ----------------------------------------------------------------------
# Back-projection
# ----------------------------------------------------------------------


def back_project_pinhole(
    u: int,
    v: int,
    depth: float,
    intrinsics: Dict[str, float],
    agent_pose: np.ndarray,
) -> Optional[np.ndarray]:
    """Back-project (u, v, depth) through the pinhole camera and agent pose
    into a 3D world point.

    Camera convention matches Habitat: x-right, y-up, looking along -z. A
    depth of D at the principal point in the identity pose gives world
    point (0, 0, -D).
    """
    if depth is None or not np.isfinite(depth) or depth <= 0.0:
        return None
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    # Camera-frame point. Habitat camera looks down -z.
    x_cam = (float(u) - cx) * depth / fx
    y_cam = -(float(v) - cy) * depth / fy   # image y points down; world y up
    z_cam = -depth
    pt_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
    pt_world = (agent_pose @ pt_cam)[:3]
    return pt_world.astype(np.float32)
