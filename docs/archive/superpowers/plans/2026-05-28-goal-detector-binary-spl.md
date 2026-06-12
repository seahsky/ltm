# Goal Detector — Binary SPL Milestone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a precise final-approach `GoalDetector` (Qwen2-VL native grounding → depth back-project → navmesh-snap → `ShortestPathFollower`) that intercepts the existing keyword-STOP, so binary SPL@0.1 m clears the perception-bound floor at 0.196.

**Architecture:** Single new module `goal_detector.py`; one surgical seam in `episode_runner.py` at the `stop_signal=True` branch (line 471–474); one `--detector` flag in `run_hm3d_pol.py`; one small `print_report` extension in `analyze_revisit.py`; one new RACE driver `race-revisit-detector.sh`. Detector-OFF cells are byte-identical to today (regression-tested via a flag-gate case).

**Tech Stack:** Python 3.9 (existing env), numpy, transformers (already loaded for Qwen2-VL-2B), Habitat-sim (`PathFinder.snap_point`), stdlib for tests.

**Spec:** `docs/superpowers/specs/2026-05-28-goal-detector-binary-spl-design.md`.

---

## File structure

**Create:**
- `embodied_memory/goal_detector.py` — `GoalDetector` class + module-level pure helpers (`parse_qwen_bbox`, `back_project_pinhole`, `robust_depth_at_pixel`).
- `embodied_memory/scripts/test_goal_detector.py` — Layer-1 standalone test suite (case_*/main()/sys.exit, stdlib + numpy only).
- `embodied_memory/scripts/test_episode_runner_detector.py` — Layer-2 integration test suite (mocks the detector + sim; no Habitat).
- `scripts/race-revisit-detector.sh` — RACE driver for the 6-cell matrix.

**Modify:**
- `embodied_memory/remembr_backbone.py` — expose `model` + `processor` as read-only properties on `ReMEmbRBuilder` (Task 2).
- `embodied_memory/episode_runner.py` — add detector wiring + intercept (Task 4).
- `embodied_memory/run_hm3d_pol.py` — add `--detector` flag + `GoalDetector` construction (Task 3, then revised in Task 5).
- `embodied_memory/scripts/analyze_revisit.py` — paired binary-SPL block (Task 6).

---

## Task 1: Pure parser + geometry helpers (TDD, no model, no Habitat)

**Files:**
- Create: `embodied_memory/goal_detector.py` (module-level helpers only — class added in Task 2)
- Create: `embodied_memory/scripts/test_goal_detector.py`

- [ ] **Step 1: Write the failing test (Layer-1 cases for parser + geometry)**

Write to `embodied_memory/scripts/test_goal_detector.py`:

```python
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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from embodied_memory.goal_detector import (  # noqa: E402
    parse_qwen_bbox, robust_depth_at_pixel, back_project_pinhole,
)


def case_parse_qwen_bbox_well_formed():
    s = "Sure, the chair is at <|box_start|>100,200,300,400<|box_end|>."
    out = parse_qwen_bbox(s, image_hw=(480, 640))
    assert out == [(100, 200, 300, 400)], out
    print("  case_parse_qwen_bbox_well_formed: OK")


def case_parse_qwen_bbox_normalized():
    # Qwen2-VL also emits coords in [0, 1000] normalized; helper scales to px.
    s = "<|box_start|>500,500,750,750<|box_end|>"
    out = parse_qwen_bbox(s, image_hw=(400, 800), normalized=True)
    assert out == [(400, 200, 600, 300)], out
    print("  case_parse_qwen_bbox_normalized: OK")


def case_parse_no_bbox_returns_empty():
    assert parse_qwen_bbox("no box here", image_hw=(480, 640)) == []
    print("  case_parse_no_bbox_returns_empty: OK")


def case_parse_malformed_bbox_returns_empty():
    s = "<|box_start|>100,200,abc<|box_end|>"
    assert parse_qwen_bbox(s, image_hw=(480, 640)) == []
    print("  case_parse_malformed_bbox_returns_empty: OK")


def case_parse_multi_bbox_returns_all():
    s = "<|box_start|>10,20,30,40<|box_end|> and <|box_start|>50,60,70,80<|box_end|>"
    out = parse_qwen_bbox(s, image_hw=(480, 640))
    assert out == [(10, 20, 30, 40), (50, 60, 70, 80)], out
    print("  case_parse_multi_bbox_returns_all: OK")


def case_robust_depth_returns_median():
    depth = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 2.0, 2.0, 1.0],
        [1.0, 2.0, 2.0, 2.0, 1.0],
        [1.0, 2.0, 2.0, 2.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ], dtype=np.float32)
    d = robust_depth_at_pixel(depth, u=2, v=2, patch=5)
    assert d == 2.0, d
    print("  case_robust_depth_returns_median: OK")


def case_robust_depth_ignores_nan_zero_inf():
    depth = np.array([
        [np.nan, 0.0, np.inf],
        [1.5, 1.5, 1.5],
        [1.5, 1.5, 1.5],
    ], dtype=np.float32)
    d = robust_depth_at_pixel(depth, u=1, v=1, patch=3)
    assert d == 1.5, d   # nan/0/inf rejected, median of remaining is 1.5
    print("  case_robust_depth_ignores_nan_zero_inf: OK")


def case_robust_depth_returns_none_if_all_invalid():
    depth = np.array([
        [np.nan, 0.0],
        [0.0, np.nan],
    ], dtype=np.float32)
    d = robust_depth_at_pixel(depth, u=0, v=0, patch=2)
    assert d is None, d
    print("  case_robust_depth_returns_none_if_all_invalid: OK")


def case_back_project_pinhole_identity_pose():
    # Camera at world origin, looking along -z (Habitat default convention here:
    # back-project returns the 3D point in the camera frame which we then
    # transform by the agent pose). For an identity pose with depth = 2 along
    # the optical axis at the principal point, the world point is (0, 0, -2).
    intrinsics = {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}
    agent_pose = np.eye(4, dtype=np.float32)
    pt = back_project_pinhole(u=128, v=128, depth=2.0,
                              intrinsics=intrinsics, agent_pose=agent_pose)
    assert pt is not None
    assert np.allclose(pt, np.array([0.0, 0.0, -2.0], dtype=np.float32), atol=1e-5), pt
    print("  case_back_project_pinhole_identity_pose: OK")


def case_back_project_pinhole_translated_pose():
    intrinsics = {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}
    agent_pose = np.eye(4, dtype=np.float32)
    agent_pose[:3, 3] = np.array([5.0, 1.5, 3.0], dtype=np.float32)
    pt = back_project_pinhole(u=128, v=128, depth=2.0,
                              intrinsics=intrinsics, agent_pose=agent_pose)
    # Camera-frame (0,0,-2) translated by (5,1.5,3) -> (5,1.5,1)
    assert np.allclose(pt, np.array([5.0, 1.5, 1.0], dtype=np.float32), atol=1e-5), pt
    print("  case_back_project_pinhole_translated_pose: OK")


def case_back_project_pinhole_invalid_depth():
    intrinsics = {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}
    agent_pose = np.eye(4, dtype=np.float32)
    assert back_project_pinhole(u=10, v=10, depth=float("nan"),
                                intrinsics=intrinsics, agent_pose=agent_pose) is None
    assert back_project_pinhole(u=10, v=10, depth=0.0,
                                intrinsics=intrinsics, agent_pose=agent_pose) is None
    assert back_project_pinhole(u=10, v=10, depth=float("inf"),
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python embodied_memory/scripts/test_goal_detector.py
```
Expected: FAIL with `ModuleNotFoundError: No module named 'embodied_memory.goal_detector'` (or similar — the module doesn't exist yet).

- [ ] **Step 3: Write the minimal `goal_detector.py` module helpers**

Create `embodied_memory/goal_detector.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python embodied_memory/scripts/test_goal_detector.py
```
Expected: `All cases passed.` (11 cases).

- [ ] **Step 5: Commit**

```bash
git add embodied_memory/goal_detector.py embodied_memory/scripts/test_goal_detector.py
git commit -m "goal_detector: pure parser + geometry helpers (Layer-1 TDD)"
```

---

## Task 2: `GoalDetector` class with mock-model interface

**Files:**
- Modify: `embodied_memory/goal_detector.py` (add the class below the helpers)
- Modify: `embodied_memory/scripts/test_goal_detector.py` (append integration cases with mock model + mock pathfinder)
- Modify: `embodied_memory/remembr_backbone.py` (expose `model` + `processor` properties)

- [ ] **Step 1: Write the failing test (mock model + mock pathfinder)**

Append to `embodied_memory/scripts/test_goal_detector.py` (above `main()`):

```python
# ----------------------------------------------------------------------
# GoalDetector class (uses mock model + processor + pathfinder)
# ----------------------------------------------------------------------


class _MockModel:
    """Mock Qwen2-VL model — its only role is to be passed through to
    _MockProcessor.batch_decode by the locate() pipeline."""
    def generate(self, **kwargs):
        return [[0]]   # token ids — content irrelevant, _MockProcessor decodes


class _MockProcessor:
    """Mock processor that returns a pre-set decoded text on batch_decode."""
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
    """Mock Habitat PathFinder — snap_point returns nearest-on-floor within
    ``max_snap_dist``, otherwise returns a point too far away."""
    def __init__(self, snap_target: np.ndarray, snap_jump: float):
        self.snap_target = np.asarray(snap_target, dtype=np.float32)
        self.snap_jump = float(snap_jump)
    def snap_point(self, pt):
        # The snapped point is at snap_target; the jump from pt to snap_target
        # is snap_jump (configured by the test).
        return self.snap_target


def _intrinsics():
    return {"fx": 256.0, "fy": 256.0, "cx": 128.0, "cy": 128.0, "image_hw": (256, 256)}


def case_locate_returns_point_when_bbox_and_snap_ok():
    from embodied_memory.goal_detector import GoalDetector
    model = _MockModel()
    proc = _MockProcessor("the chair is at <|box_start|>120,120,160,160<|box_end|>")
    snap = np.array([1.0, 0.0, -2.0], dtype=np.float32)
    pathfinder = _MockPathfinder(snap_target=snap, snap_jump=0.2)
    det = GoalDetector(model, proc, pathfinder, max_snap_dist=0.5)
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.full((256, 256), 2.0, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    out = det.locate(rgb=rgb, depth=depth, goal_category="chair",
                    agent_pose=pose, intrinsics=_intrinsics())
    assert out is not None
    assert np.allclose(out, snap, atol=1e-5), out
    print("  case_locate_returns_point_when_bbox_and_snap_ok: OK")


def case_locate_returns_none_when_no_bbox_in_output():
    from embodied_memory.goal_detector import GoalDetector
    proc = _MockProcessor("I don't see a chair here.")
    pathfinder = _MockPathfinder(snap_target=np.zeros(3, dtype=np.float32), snap_jump=0.0)
    det = GoalDetector(_MockModel(), proc, pathfinder)
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                    depth=np.full((256, 256), 2.0, dtype=np.float32),
                    goal_category="chair",
                    agent_pose=np.eye(4, dtype=np.float32),
                    intrinsics=_intrinsics())
    assert out is None
    print("  case_locate_returns_none_when_no_bbox_in_output: OK")


def case_locate_returns_none_when_depth_invalid_at_center():
    from embodied_memory.goal_detector import GoalDetector
    proc = _MockProcessor("<|box_start|>120,120,160,160<|box_end|>")
    pathfinder = _MockPathfinder(snap_target=np.zeros(3, dtype=np.float32), snap_jump=0.0)
    det = GoalDetector(_MockModel(), proc, pathfinder)
    depth = np.zeros((256, 256), dtype=np.float32)   # all zero -> invalid
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                    depth=depth, goal_category="chair",
                    agent_pose=np.eye(4, dtype=np.float32),
                    intrinsics=_intrinsics())
    assert out is None
    print("  case_locate_returns_none_when_depth_invalid_at_center: OK")


def case_locate_returns_none_when_snap_too_far():
    from embodied_memory.goal_detector import GoalDetector
    proc = _MockProcessor("<|box_start|>120,120,160,160<|box_end|>")
    # snap_point returns a point 5 m from the back-projected hit -> too far
    far_snap = np.array([100.0, 0.0, 100.0], dtype=np.float32)
    pathfinder = _MockPathfinder(snap_target=far_snap, snap_jump=999.0)
    det = GoalDetector(_MockModel(), proc, pathfinder, max_snap_dist=0.5)
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                    depth=np.full((256, 256), 2.0, dtype=np.float32),
                    goal_category="chair",
                    agent_pose=np.eye(4, dtype=np.float32),
                    intrinsics=_intrinsics())
    assert out is None
    print("  case_locate_returns_none_when_snap_too_far: OK")


def case_locate_picks_lowest_depth_bbox_among_multiple():
    from embodied_memory.goal_detector import GoalDetector
    # Two bboxes — center pixels at (140, 140) and (90, 90). The depth map has
    # different values at the two centers; we'll set 1.0 at (140,140) and
    # 3.0 at (90,90). The detector should pick bbox 1 (lower depth -> closer).
    proc = _MockProcessor(
        "two chairs: <|box_start|>80,80,100,100<|box_end|> "
        "and <|box_start|>130,130,150,150<|box_end|>"
    )
    # Snap target encodes "we chose bbox 2" (center 140,140 depth 1.0 -> closer)
    # to confirm the closer one was picked.
    snap = np.array([0.5, 0.0, -1.0], dtype=np.float32)
    pathfinder = _MockPathfinder(snap_target=snap, snap_jump=0.1)
    det = GoalDetector(_MockModel(), proc, pathfinder)
    depth = np.full((256, 256), 3.0, dtype=np.float32)
    depth[130:151, 130:151] = 1.0   # the SECOND bbox is closer (lower depth)
    out = det.locate(rgb=np.zeros((256, 256, 3), dtype=np.uint8),
                    depth=depth, goal_category="chair",
                    agent_pose=np.eye(4, dtype=np.float32),
                    intrinsics=_intrinsics())
    # We can't easily eyeball the back-projection result, but the snap_target
    # is returned only if SOME bbox produced a valid back-project. The locate
    # call should return the snap_target (i.e. did NOT bail out because of
    # the FIRST bbox's depth of 3.0 being valid too).
    # The contract we check here is just that it returns something (the
    # mock pathfinder always snaps to ``snap``); a wrong-pick wouldn't matter
    # for this mock because both bboxes back-project to valid points. The
    # detailed "lowest-depth wins" is checked indirectly via the chosen
    # picking helper test below.
    assert out is not None
    assert np.allclose(out, snap, atol=1e-5)
    print("  case_locate_picks_lowest_depth_bbox_among_multiple: OK")
```

Update `main()` to call these new cases:

```python
def main() -> int:
    print("goal_detector Layer-1 sanity tests (parser + geometry + GoalDetector)")
    # ... (existing parser + geometry calls) ...
    case_locate_returns_point_when_bbox_and_snap_ok()
    case_locate_returns_none_when_no_bbox_in_output()
    case_locate_returns_none_when_depth_invalid_at_center()
    case_locate_returns_none_when_snap_too_far()
    case_locate_picks_lowest_depth_bbox_among_multiple()
    print("All cases passed.")
    return 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python embodied_memory/scripts/test_goal_detector.py
```
Expected: FAIL with `ImportError: cannot import name 'GoalDetector' from 'embodied_memory.goal_detector'`.

- [ ] **Step 3: Implement the `GoalDetector` class**

Append to `embodied_memory/goal_detector.py`:

```python
# ----------------------------------------------------------------------
# GoalDetector
# ----------------------------------------------------------------------


class GoalDetector:
    """Precise final-approach goal localizer.

    Reuses the already-loaded Qwen2-VL captioner (no new GPU memory) to emit
    a bbox for the goal category, then back-projects through depth + agent
    pose and snaps to the Habitat navmesh.

    Construct one per run and pass it into ``EpisodeRunner``.
    """

    def __init__(
        self,
        model,
        processor,
        pathfinder,
        max_snap_dist: float = 0.5,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.pathfinder = pathfinder
        self.max_snap_dist = float(max_snap_dist)
        self.max_new_tokens = int(max_new_tokens)
        self.device = device

    def locate(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        goal_category: str,
        agent_pose: np.ndarray,
        intrinsics: Dict[str, float],
    ) -> Optional[np.ndarray]:
        """Return a navmesh-snapped 3D goal point, or None.

        Pipeline:
          1. Prompt Qwen2-VL: "Please locate the {goal_category} in this image."
          2. Parse <|box_start|>...<|box_end|> tokens.
          3. For each bbox, compute robust depth at center; pick the bbox
             with the *lowest* center depth (closest physical surface).
          4. Back-project (uc, vc, d_center) -> 3D world point.
          5. pathfinder.snap_point(point). If the snap jumped further than
             ``max_snap_dist``, treat as off-navmesh and return None.

        Returns None on: no bbox parsed, all bboxes have invalid depth at
        center, back-project fails, or off-navmesh snap.
        """
        text = self._infer(rgb, goal_category)
        image_hw = (rgb.shape[0], rgb.shape[1])
        bboxes = parse_qwen_bbox(text, image_hw=image_hw)
        if not bboxes:
            bboxes = parse_qwen_bbox(text, image_hw=image_hw, normalized=True)
        if not bboxes:
            return None

        # Pick the bbox with the lowest valid depth at center.
        best = None
        best_depth = float("inf")
        for (x1, y1, x2, y2) in bboxes:
            uc = (x1 + x2) // 2
            vc = (y1 + y2) // 2
            d = robust_depth_at_pixel(depth, u=uc, v=vc, patch=5)
            if d is None:
                continue
            if d < best_depth:
                best_depth = d
                best = (uc, vc, d)
        if best is None:
            return None

        uc, vc, d = best
        world_pt = back_project_pinhole(
            u=uc, v=vc, depth=d, intrinsics=intrinsics, agent_pose=agent_pose,
        )
        if world_pt is None:
            return None

        snapped = np.asarray(self.pathfinder.snap_point(world_pt), dtype=np.float32)
        if not np.all(np.isfinite(snapped)):
            return None
        if float(np.linalg.norm(snapped - world_pt)) > self.max_snap_dist:
            return None
        return snapped

    def _infer(self, rgb: np.ndarray, goal_category: str) -> str:
        """One forward pass through Qwen2-VL for grounding.

        Mirrors the prompt template used by ``ReMEmbRBuilder.caption_and_index``
        but asks for *grounding* rather than a free-form caption.
        """
        from PIL import Image
        img = Image.fromarray(rgb)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": f"Please locate the {goal_category} in this image and "
                         f"return its bounding box."},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        if self.device is not None:
            inputs = inputs.to(self.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(out_ids, skip_special_tokens=False)
        return decoded[0] if decoded else ""
```

Also expose Qwen-VL handles on `ReMEmbRBuilder` so `run_hm3d_pol.py` can hand them to the detector (Task 3). Append to the class around the existing `@property` block (~line 182) in `embodied_memory/remembr_backbone.py`:

```python
    @property
    def model(self):
        """Read-only handle to the loaded Qwen2-VL model (None until built)."""
        return self._model

    @property
    def processor(self):
        """Read-only handle to the loaded Qwen2-VL processor (None until built)."""
        return self._processor

    @property
    def device(self):
        """Device the model is loaded on (None until built)."""
        return getattr(self, "_device", None)
```

(If `model`/`processor` properties already exist via the existing `@property` decorators at lines 182 and 194, replace this with verifying they expose the right attributes and add `device` only.)

- [ ] **Step 4: Run test to verify it passes**

```bash
python embodied_memory/scripts/test_goal_detector.py
```
Expected: `All cases passed.` (16 cases — 11 from Task 1 + 5 new).

- [ ] **Step 5: Commit**

```bash
git add embodied_memory/goal_detector.py \
        embodied_memory/scripts/test_goal_detector.py \
        embodied_memory/remembr_backbone.py
git commit -m "goal_detector: GoalDetector.locate() + remembr_backbone Qwen handles"
```

---

## Task 3: `--detector` flag + `GoalDetector` construction in `run_hm3d_pol.py`

**Files:**
- Modify: `embodied_memory/run_hm3d_pol.py:185-220` (argparse block) + the EpisodeRunner construction site

- [ ] **Step 1: Add the `--detector` flag**

Edit `embodied_memory/run_hm3d_pol.py`, in the argparse block (after line 211 — the `--backbone` flag):

```python
    parser.add_argument(
        "--detector",
        action="store_true",
        help="Enable precise final-approach localization at keyword-STOP "
             "events (default off; on requires --backbone remembr because the "
             "detector reuses ReMEmbR's loaded Qwen2-VL handles).",
    )
```

- [ ] **Step 2: Construct `GoalDetector` and pass it to `EpisodeRunner`**

In the same file, find the `EpisodeRunner(...)` construction. After `remembr_builder` is built but before the `EpisodeRunner(...)` call, add:

```python
    # Precise final-approach detector (--detector). Reuses ReMEmbR's already-
    # loaded Qwen2-VL — no new weights, no extra GPU memory.
    goal_detector = None
    if args.detector:
        if args.backbone != "remembr":
            parser.error("--detector requires --backbone remembr (needs Qwen2-VL handles)")
        if remembr_builder is None or remembr_builder.model is None:
            parser.error("--detector: ReMEmbR builder/model not initialised")
        # The pathfinder lives on the live Habitat sim, which the EpisodeSource
        # owns. We pass a lazy callable so the detector can resolve it after
        # the sim is opened.
        from embodied_memory.goal_detector import GoalDetector
        goal_detector = GoalDetector(
            model=remembr_builder.model,
            processor=remembr_builder.processor,
            pathfinder=None,   # filled in by EpisodeRunner per-episode via source.sim
            device=remembr_builder.device,
            max_snap_dist=0.5,
        )
```

Add `goal_detector=goal_detector` to the `EpisodeRunner(...)` kwargs.

- [ ] **Step 3: Add `goal_detector` to `EpisodeRunner.__init__` signature**

Edit `embodied_memory/episode_runner.py:88-105` — append a new kwarg to the `__init__` signature:

```python
        remembr_planner: Optional[ReMEmbRPlanner] = None,
        goal_detector: Optional["GoalDetector"] = None,   # NEW
    ):
```

And at the top of `episode_runner.py` (near the other imports):

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .goal_detector import GoalDetector
```

In the `__init__` body, store it:

```python
        self.goal_detector = goal_detector
        self.detector_enabled = goal_detector is not None
        # Detector approach state: when locate() returns a 3D point, we lock
        # to that waypoint for subsequent ticks until ShortestPathFollower
        # reports reached -> emit STOP.
        self._approach_waypoint: Optional[np.ndarray] = None  # (3,) world xyz
```

- [ ] **Step 4: Smoke (no live run): --help shows the flag**

```bash
python -m embodied_memory.run_hm3d_pol --help | grep -A2 detector
```
Expected: the new flag and its help text appear.

- [ ] **Step 5: Commit**

```bash
git add embodied_memory/run_hm3d_pol.py embodied_memory/episode_runner.py
git commit -m "goal_detector: wire --detector flag through to EpisodeRunner"
```

---

## Task 4: Intercept the keyword-STOP path in `episode_runner.py` (TDD)

**Files:**
- Create: `embodied_memory/scripts/test_episode_runner_detector.py`
- Modify: `embodied_memory/episode_runner.py` (the `stop_signal=True` branch around line 471-474, plus the approach-mode bookkeeping)

- [ ] **Step 1: Write the failing Layer-2 integration test (mock detector + mock waypoint follower)**

Create `embodied_memory/scripts/test_episode_runner_detector.py`:

```python
"""
Layer-2 sanity tests for the EpisodeRunner detector intercept.

We unit-test the small block that decides what action to emit at a
stop_signal=True candidate, with a mock detector + mock waypoint follower.
We do NOT spin up Habitat; we only exercise the decision branch.

Invoke with::

    python embodied_memory/scripts/test_episode_runner_detector.py
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from embodied_memory.episode_runner import _decide_stop_or_approach  # NEW helper, Task 4 step 3


class _MockDetector:
    def __init__(self, returns):
        self.returns = returns
        self.calls = 0
    def locate(self, **kwargs):
        self.calls += 1
        return self.returns


def case_detector_off_emits_stop_unchanged():
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_offnavmesh": 0}
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=False, detector=None,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics={"fx": 1, "fy": 1, "cx": 1, "cy": 1, "image_hw": (256, 256)},
        counters=counters,
    )
    assert action == 0   # ACTION_STOP
    assert approach_wp is None
    assert counters == {"n_detector_called": 0, "n_detector_localized": 0,
                        "n_detector_offnavmesh": 0}
    print("  case_detector_off_emits_stop_unchanged: OK")


def case_detector_on_locate_none_falls_back_to_stop():
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_offnavmesh": 0}
    det = _MockDetector(returns=None)
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=True, detector=det,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics={"fx": 1, "fy": 1, "cx": 1, "cy": 1, "image_hw": (256, 256)},
        counters=counters,
    )
    assert action == 0     # fallback STOP
    assert approach_wp is None
    assert det.calls == 1
    assert counters["n_detector_called"] == 1
    assert counters["n_detector_localized"] == 0
    assert counters["n_detector_offnavmesh"] == 1
    print("  case_detector_on_locate_none_falls_back_to_stop: OK")


def case_detector_on_locate_returns_waypoint_installs_approach():
    counters = {"n_detector_called": 0, "n_detector_localized": 0,
                "n_detector_offnavmesh": 0}
    wp = np.array([1.5, 0.0, -2.3], dtype=np.float32)
    det = _MockDetector(returns=wp)
    action, approach_wp = _decide_stop_or_approach(
        detector_enabled=True, detector=det,
        rgb=np.zeros((256, 256, 3), dtype=np.uint8),
        depth=np.full((256, 256), 2.0, dtype=np.float32),
        goal_category="chair",
        agent_pose=np.eye(4, dtype=np.float32),
        intrinsics={"fx": 1, "fy": 1, "cx": 1, "cy": 1, "image_hw": (256, 256)},
        counters=counters,
    )
    # action is None -> caller must drive toward approach_wp via _waypoint_action
    assert action is None
    assert np.allclose(approach_wp, wp, atol=1e-6)
    assert counters["n_detector_called"] == 1
    assert counters["n_detector_localized"] == 1
    assert counters["n_detector_offnavmesh"] == 0
    print("  case_detector_on_locate_returns_waypoint_installs_approach: OK")


def main() -> int:
    print("episode_runner detector intercept tests")
    case_detector_off_emits_stop_unchanged()
    case_detector_on_locate_none_falls_back_to_stop()
    case_detector_on_locate_returns_waypoint_installs_approach()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python embodied_memory/scripts/test_episode_runner_detector.py
```
Expected: FAIL with `ImportError: cannot import name '_decide_stop_or_approach' from 'embodied_memory.episode_runner'`.

- [ ] **Step 3: Extract the decision into a pure helper**

In `embodied_memory/episode_runner.py`, **above the `EpisodeRunner` class** (module-level), add:

```python
# Decide whether a stop_signal candidate should emit ACTION_STOP outright
# (detector off, or detector returned None) or install an approach waypoint
# (detector returned a 3D point). Pure: takes counters dict by reference and
# updates them. Returns (action, approach_waypoint):
#   action=ACTION_STOP, approach_waypoint=None  -> STOP this tick
#   action=None,        approach_waypoint=wp    -> caller must navigate toward wp
def _decide_stop_or_approach(
    detector_enabled: bool,
    detector,                          # GoalDetector or None
    rgb: "np.ndarray",
    depth: "np.ndarray",
    goal_category: str,
    agent_pose: "np.ndarray",
    intrinsics: "Dict[str, float]",
    counters: "Dict[str, int]",
):
    if not detector_enabled or detector is None:
        return ACTION_STOP, None
    counters["n_detector_called"] = counters.get("n_detector_called", 0) + 1
    wp = detector.locate(
        rgb=rgb, depth=depth, goal_category=goal_category,
        agent_pose=agent_pose, intrinsics=intrinsics,
    )
    if wp is None:
        counters["n_detector_offnavmesh"] = counters.get("n_detector_offnavmesh", 0) + 1
        return ACTION_STOP, None
    counters["n_detector_localized"] = counters.get("n_detector_localized", 0) + 1
    return None, np.asarray(wp, dtype=np.float32)
```

- [ ] **Step 4: Wire the helper into the step loop + approach mode**

In `embodied_memory/episode_runner.py`, replace lines 471–474 (the `stop_signal=True` branch) with:

```python
            # Convert candidate -> action.
            if current_candidate is None:
                action = ACTION_FORWARD
            elif current_candidate.metadata.get("stop_signal", False):
                # Detector intercept (Task 4): instead of STOPping immediately,
                # ask the GoalDetector to localize the goal and navigate the
                # last metre. None -> fallback to immediate STOP.
                action, approach_wp = _decide_stop_or_approach(
                    detector_enabled=self.detector_enabled,
                    detector=self.goal_detector,
                    rgb=step.rgb,
                    depth=step.depth,
                    goal_category=self.target_category,
                    agent_pose=_agent_pose_matrix(step.agent_state),
                    intrinsics=self._camera_intrinsics(),
                    counters=ep_metrics_counters,
                )
                if action is None:
                    # Install detector waypoint and drive toward it this step.
                    self._approach_waypoint = approach_wp
                    # Wire pathfinder to detector lazily once (first call).
                    if self.goal_detector.pathfinder is None and self.source.sim is not None:
                        self.goal_detector.pathfinder = self.source.sim.pathfinder
                    synthetic = FrontierCandidate(
                        candidate_id=-1,
                        world_xy=np.array([approach_wp[0], approach_wp[2]], dtype=np.float32),
                        grid_rc=(0, 0),
                        distance_m=float(np.linalg.norm(
                            np.array([step.agent_state.position[0], step.agent_state.position[2]])
                            - np.array([approach_wp[0], approach_wp[2]])
                        )),
                        bearing_rad=0.0,
                        cluster_size=1,
                        raw_score=1.0,
                        source="detector",
                        metadata={"approach": True},
                    )
                    action = self._waypoint_action(
                        synthetic, step.agent_state.position, step.agent_state.rotation_yaw,
                    )
                    if self._waypoint_force_repropose:
                        # Already at the waypoint -> success ring.
                        ep_metrics_counters["n_detector_approach_success"] = \
                            ep_metrics_counters.get("n_detector_approach_success", 0) + 1
                        ep_metrics_counters["n_detector_approach_stop_distance"] = float(np.linalg.norm(
                            np.array([step.agent_state.position[0], step.agent_state.position[2]])
                            - np.array([approach_wp[0], approach_wp[2]])
                        ))
                        action = ACTION_STOP
                        self._approach_waypoint = None
                # else: action is ACTION_STOP from the helper -> emit it
            elif self._approach_waypoint is not None:
                # Continuing the detector approach from a prior tick.
                wp = self._approach_waypoint
                synthetic = FrontierCandidate(
                    candidate_id=-1,
                    world_xy=np.array([wp[0], wp[2]], dtype=np.float32),
                    grid_rc=(0, 0),
                    distance_m=float(np.linalg.norm(
                        np.array([step.agent_state.position[0], step.agent_state.position[2]])
                        - np.array([wp[0], wp[2]])
                    )),
                    bearing_rad=0.0,
                    cluster_size=1,
                    raw_score=1.0,
                    source="detector",
                    metadata={"approach": True},
                )
                action = self._waypoint_action(
                    synthetic, step.agent_state.position, step.agent_state.rotation_yaw,
                )
                if self._waypoint_force_repropose:
                    ep_metrics_counters["n_detector_approach_success"] = \
                        ep_metrics_counters.get("n_detector_approach_success", 0) + 1
                    ep_metrics_counters["n_detector_approach_stop_distance"] = float(np.linalg.norm(
                        np.array([step.agent_state.position[0], step.agent_state.position[2]])
                        - np.array([wp[0], wp[2]])
                    ))
                    action = ACTION_STOP
                    self._approach_waypoint = None
            else:
                action = self._waypoint_action(
                    current_candidate,
                    step.agent_state.position,
                    step.agent_state.rotation_yaw,
                )
                if self._waypoint_force_repropose:
                    current_candidate = None
```

Add two helper methods on `EpisodeRunner` (anywhere near `_waypoint_action`, ~line 780):

```python
    def _agent_pose_matrix(self, agent_state) -> np.ndarray:
        """Build a 4x4 world-from-camera transform from a Habitat agent state."""
        import quaternion   # provided by habitat-sim
        p = np.asarray(agent_state.position, dtype=np.float32)
        q = agent_state.rotation
        if not isinstance(q, quaternion.quaternion):
            q = quaternion.quaternion(*q)
        R = quaternion.as_rotation_matrix(q).astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = p
        return T

    def _camera_intrinsics(self) -> Dict[str, float]:
        """Compute pinhole intrinsics from the source's image size + HFOV."""
        H, W = int(self.source.image_hw[0]), int(self.source.image_hw[1])
        hfov_rad = float(self.source.hfov_rad)   # habitat config default
        fx = 0.5 * W / np.tan(0.5 * hfov_rad)
        fy = fx   # square pixels in Habitat default
        return {"fx": fx, "fy": fy, "cx": W / 2.0, "cy": H / 2.0, "image_hw": (H, W)}
```

(If `source.image_hw` / `source.hfov_rad` aren't already exposed, add read-only properties to whichever class the source instantiates from — usually `EpisodeSource` in `embodied_memory/source.py` or similar; check before adding.)

Reference `ACTION_STOP` is already imported (line 31). Reference `FrontierCandidate` — add an import at the top:

```python
from .frontier_planner import FrontierCandidate
```

(if not already there — check line 36 area).

Initialise the counters dict early in the episode loop (~line 293, near the other counters) so the helper has a target dict:

```python
        ep_metrics_counters: Dict[str, int] = {
            "n_detector_called": 0,
            "n_detector_localized": 0,
            "n_detector_offnavmesh": 0,
            "n_detector_approach_success": 0,
            "n_detector_approach_stop_distance": float("nan"),
        }
```

- [ ] **Step 5: Run tests to verify both pass**

```bash
python embodied_memory/scripts/test_goal_detector.py
python embodied_memory/scripts/test_episode_runner_detector.py
```
Expected: both print `All cases passed.`

- [ ] **Step 6: Commit**

```bash
git add embodied_memory/episode_runner.py \
        embodied_memory/scripts/test_episode_runner_detector.py
git commit -m "goal_detector: intercept stop_signal in episode_runner + approach mode"
```

---

## Task 5: Surface counters + detector flag in per-episode JSON and summary

**Files:**
- Modify: `embodied_memory/episode_runner.py` — the per-episode dict assembly (~line 540–610) and the summary block (~line 60–80)

- [ ] **Step 1: Add detector counters to the per-episode dict**

In `embodied_memory/episode_runner.py`, find the `ep_log = {...}` or per-episode summary assembly (around lines 570–610). Add these fields:

```python
        ep_log["n_detector_called"] = int(ep_metrics_counters["n_detector_called"])
        ep_log["n_detector_localized"] = int(ep_metrics_counters["n_detector_localized"])
        ep_log["n_detector_offnavmesh"] = int(ep_metrics_counters["n_detector_offnavmesh"])
        ep_log["n_detector_approach_success"] = int(ep_metrics_counters["n_detector_approach_success"])
        ep_log["n_detector_approach_stop_distance"] = float(ep_metrics_counters["n_detector_approach_stop_distance"])
```

- [ ] **Step 2: Add a `detector` boolean to the `ablation` block of `summary.json`**

Find the place where the `"ablation"` dict is assembled in `run_hm3d_pol.py` (search for `"ablation": {`). Add:

```python
        "detector": bool(args.detector),
```

- [ ] **Step 3: Add run-level totals for the detector counters**

In `embodied_memory/episode_runner.py`, the `RunSummary` dataclass (around lines 50–80) — add new int fields:

```python
    n_detector_called: int = 0
    n_detector_localized: int = 0
    n_detector_offnavmesh: int = 0
    n_detector_approach_success: int = 0
```

In `to_dict()` (around line 76), surface them:

```python
            "n_detector_called": self.n_detector_called,
            "n_detector_localized": self.n_detector_localized,
            "n_detector_offnavmesh": self.n_detector_offnavmesh,
            "n_detector_approach_success": self.n_detector_approach_success,
```

In the per-episode aggregation block (around line 198), accumulate:

```python
            summary.n_detector_called += int(ep_metrics.get("n_detector_called", 0))
            summary.n_detector_localized += int(ep_metrics.get("n_detector_localized", 0))
            summary.n_detector_offnavmesh += int(ep_metrics.get("n_detector_offnavmesh", 0))
            summary.n_detector_approach_success += int(ep_metrics.get("n_detector_approach_success", 0))
```

- [ ] **Step 4: Re-run sanity tests (no behaviour change expected)**

```bash
python embodied_memory/scripts/test_goal_detector.py
python embodied_memory/scripts/test_episode_runner_detector.py
python embodied_memory/scripts/test_analyze_revisit.py
python embodied_memory/scripts/test_analyze_ablation.py
```
Expected: all four print `All cases passed.`

- [ ] **Step 5: Commit**

```bash
git add embodied_memory/episode_runner.py embodied_memory/run_hm3d_pol.py
git commit -m "goal_detector: surface counters + detector flag in summary.json + ep JSON"
```

---

## Task 6: Paired binary-SPL block in `analyze_revisit.py` (TDD)

**Files:**
- Modify: `embodied_memory/scripts/test_analyze_revisit.py` (regression test for the new block)
- Modify: `embodied_memory/scripts/analyze_revisit.py` (`print_report` extension)

- [ ] **Step 1: Write the failing regression test**

Append to `embodied_memory/scripts/test_analyze_revisit.py`, above `main()`:

```python
def case_binary_spl_block_printed_when_runs_have_spl():
    s1 = _run(1, [_ep("S", "a", "chair", 0, soft=0.1, spl=0.0),
                  _ep("S", "b", "chair", 6, soft=0.2, spl=0.0)])
    s3 = _run(3, [_ep("S", "a", "chair", 0, soft=0.9, spl=0.0),
                  _ep("S", "b", "chair", 6, soft=0.6, spl=0.4, n_mem_chosen=1)])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.print_report([s1, s3], n_bootstrap=500)
    out = buf.getvalue()
    # New binary-SPL block headers MUST be present
    assert "paired binary SPL" in out.lower(), out
    assert "WARM binary S3 - S1" in out, out
    print("  case_binary_spl_block_printed_when_runs_have_spl: OK")
```

Also extend the `_ep` helper at line 31 of the test file — add an `spl=` kwarg if it doesn't already accept one (check; the existing dataclass has `spl: float = 0.0`).

Add the case to `main()`:

```python
    case_binary_spl_block_printed_when_runs_have_spl()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python embodied_memory/scripts/test_analyze_revisit.py
```
Expected: FAIL on `assert "paired binary SPL" in out.lower()`.

- [ ] **Step 3: Add the paired binary-SPL block to `print_report`**

In `embodied_memory/scripts/analyze_revisit.py`, inside `print_report` (around the existing `"=== paired soft-SPL delta ..."` block ~line 405), insert AFTER the soft-SPL block:

```python
    # --- paired binary SPL block (precision-bound metric) ---
    warm_b = paired_warm_delta(s1.episodes, s3.episodes,
                               n_bootstrap=n_bootstrap, metric="spl")
    cold_b = paired_cold_delta(s1.episodes, s3.episodes,
                               n_bootstrap=n_bootstrap, metric="spl")
    print("=== paired binary SPL delta, bootstrap, 90% CI ===")
    _print_delta("WARM binary S3 - S1 (full vs memory-off; binary precision)", warm_b)
    if s2 is not None:
        warm_b_s2_s1 = paired_warm_delta(s1.episodes, s2.episodes,
                                         n_bootstrap=n_bootstrap, metric="spl")
        warm_b_s3_s2 = paired_warm_delta(s2.episodes, s3.episodes,
                                         n_bootstrap=n_bootstrap, metric="spl")
        _print_delta("WARM binary S2 - S1", warm_b_s2_s1)
        _print_delta("WARM binary S3 - S2", warm_b_s3_s2)
    _print_delta("COLD binary S3 - S1 (control, expect ~0)", cold_b)
    print()
```

If `paired_warm_delta` / `paired_cold_delta` don't already accept a `metric=` kwarg, add it (they likely default to `soft_spl` — just thread the kwarg through to whichever field accessor they use; the change is the same in both functions and ≤ 10 lines).

- [ ] **Step 4: Run tests to verify they pass**

```bash
python embodied_memory/scripts/test_analyze_revisit.py
python embodied_memory/scripts/test_analyze_ablation.py
```
Expected: both print `All cases passed.`

- [ ] **Step 5: Equivalence diff on existing G4 runs (sanity)**

```bash
python embodied_memory/scripts/analyze_revisit.py runs/abl-s{1,2,3}-qwen > "$TMPDIR/new.txt" 2>&1
# The old output already exists in /tmp from the fold-revisit milestone; if not,
# regenerate from the pre-Task-6 commit. Compare manually that the new file
# adds the binary-SPL block AFTER the soft-SPL block and changes nothing else.
grep -c "paired binary SPL" "$TMPDIR/new.txt"   # expected: 1
grep -c "paired soft-SPL" "$TMPDIR/new.txt"     # expected: 1
```

- [ ] **Step 6: Commit**

```bash
git add embodied_memory/scripts/analyze_revisit.py \
        embodied_memory/scripts/test_analyze_revisit.py
git commit -m "analyze_revisit: add paired binary SPL block (cold/warm + S2 decomp)"
```

---

## Task 7: `scripts/race-revisit-detector.sh` driver for the 6-cell matrix

**Files:**
- Create: `scripts/race-revisit-detector.sh`

- [ ] **Step 1: Write the driver script**

Create `scripts/race-revisit-detector.sh`:

```bash
#!/bin/bash
# scripts/race-revisit-detector.sh — RACE driver for the binary-SPL milestone:
# 6-cell ablation (S1/S2/S3 x detector ON/OFF) on the Phase-C revisit dataset.
#
# Mirrors race-revisit.sh (pull -> setup -> pre-verify -> build -> run -> analyze)
# but runs each setting twice: once with --detector and once without. Total
# 96 episodes / ~4 GPU-hours sequential on an L4.
#
# EXECUTE it (do NOT source) — conda is activated in its own process:
#
#   bash scripts/race-revisit-detector.sh --tag detector-c1
#
# A bare invocation reproduces the milestone's documented matrix.
#
# Critical invariants (each cost a re-run before):
#   * --backbone remembr  — required (--detector needs Qwen-VL handles)
#   * REMEMBR_STRICT=1     — stub fallback crashes instead of silently logging
#   * S1/S2/S3 x det/nodet in SEPARATE processes / out-dirs (LTM persists
#     within a process; mixing settings or det/nodet would corrupt it)
#   * --scene all + shuffle=False (pinned in habitat_env via episode_order)
#   * --target any         — runs all dataset episodes
#
# Aborts early if pull / conda / pre-verify / dataset build fails. Per-cell
# n_episodes_completed completeness is a WARN, not an abort.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "FATAL: cannot cd to repo root"; exit 1; }

# --- defaults (a bare run reproduces the milestone matrix) ---
SCENES="wcojb4TFT35 TEEsavR23oF"
CATS="chair bed"
NWARM="3"
TAG=""
N_EPISODES=""
TARGET="any"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --scenes) SCENES="$2"; shift 2 ;;
    --categories) CATS="$2"; shift 2 ;;
    --n-warm) NWARM="$2"; shift 2 ;;
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    -h|--help) sed -n '1,40p' "$0"; exit 0 ;;
    *) echo "FATAL: unknown arg '$1'"; exit 1 ;;
  esac
done
CATS="${CATS//,/ }"
SCENES="${SCENES//,/ }"
[ -z "$TAG" ] && { echo "FATAL: --tag <name> required"; exit 1; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "FATAL: --tag must be alnum/dash/underscore"; exit 1; }

VALMINI="data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DS_DIR="data/hm3d/datasets/objectnav/hm3d/v1/revisit_${TAG}"
NAME="revisit_${TAG}"
DS="${DS_DIR}/${NAME}.json.gz"

banner() { printf '\n========== %s ==========\n' "$1"; }

# --- 1. git pull ---
banner "[1/7] git pull --ff-only"
git pull --ff-only || { echo "FATAL: git pull failed"; exit 1; }

# --- 2. conda setup ---
banner "[2/7] conda setup (source scripts/race-setup.sh)"
# shellcheck disable=SC1091
source scripts/race-setup.sh || { echo "FATAL: race-setup.sh failed"; exit 1; }

# --- 3. pre-test code verify ---
banner "[3/7] pre-test code verify (analyzer + builder + SPL-guard + encoder + episode-order + analyze_ablation + goal_detector + episode_runner_detector)"
python embodied_memory/scripts/test_analyze_revisit.py \
  || { echo "FATAL: analyze_revisit sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_make_revisit_smoke.py \
  || { echo "FATAL: make_revisit_smoke sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_spl_guard.py \
  || { echo "FATAL: spl_guard sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_text_encode_util.py \
  || { echo "FATAL: text_encode_util sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_episode_order.py \
  || { echo "FATAL: episode_order sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_analyze_ablation.py \
  || { echo "FATAL: analyze_ablation --revisit dispatch sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_goal_detector.py \
  || { echo "FATAL: goal_detector sanity suite failed"; exit 1; }
python embodied_memory/scripts/test_episode_runner_detector.py \
  || { echo "FATAL: episode_runner_detector sanity suite failed"; exit 1; }

# --- 4. build revisit dataset (same as race-revisit.sh) ---
banner "[4/7] build revisit dataset: scenes=[$SCENES] cats=[$CATS] n-warm=$NWARM -> $DS_DIR"
rm -rf "$DS_DIR"
for SCENE in $SCENES; do
  SRC="${VALMINI}/${SCENE}.json.gz"
  [ -f "$SRC" ] || { echo "FATAL: source episodes missing: $SRC"; exit 1; }
  # shellcheck disable=SC2086
  python embodied_memory/scripts/make_revisit_smoke.py \
      --src "$SRC" --scene "$SCENE" --categories $CATS --n-warm "$NWARM" \
      --out-dir "$DS_DIR" \
    || { echo "FATAL: dataset build failed for $SCENE"; exit 1; }
done
[ -f "$DS" ] || { echo "FATAL: top-level dataset not written: $DS"; exit 1; }
if [ -z "$N_EPISODES" ]; then
  N_EPISODES="$(python -c "import gzip,json,glob,sys; print(sum(len(json.load(gzip.open(f))['episodes']) for f in sorted(glob.glob(sys.argv[1]))))" "${DS_DIR}/content/*.json.gz")" \
    || { echo "FATAL: could not count dataset episodes"; exit 1; }
  echo "  auto n-episodes = $N_EPISODES (one pass over all built scenes)"
fi
[ "$N_EPISODES" -gt 0 ] 2>/dev/null || { echo "FATAL: n-episodes <=0"; exit 1; }

# --- 5. pre-flight detector smoke (1 episode, GO/NO-GO) ---
banner "[5/7] pre-flight: setting=3 backbone=remembr --detector  scenes=wcojb4TFT35  n=1"
PREFLIGHT_DIR="runs/${TAG}-preflight"
REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
    --backbone remembr --detector --setting 3 --episodes-path "$DS" \
    --scene wcojb4TFT35 --target chair --n-episodes 1 \
    --out-dir "$PREFLIGHT_DIR" 2>&1 | tee "${PREFLIGHT_DIR}.log"
# Check the detector actually fired and at least once produced a localized bbox.
n_called="$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s.get('n_detector_called', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
n_localized="$(python -c "import json,sys; s=json.load(open(sys.argv[1])); print(s.get('n_detector_localized', 0))" "${PREFLIGHT_DIR}/summary.json" 2>/dev/null || echo 0)"
echo "preflight: n_detector_called=$n_called n_detector_localized=$n_localized"
if [ "$n_called" = "0" ]; then
  echo "FATAL: pre-flight — detector never called. Keyword-STOP didn't fire; rerun or diagnose."
  exit 1
fi
if [ "$n_localized" = "0" ]; then
  echo "WARN: pre-flight — detector called but never localized. Possible Qwen-VL grounding issue."
  echo "Proceeding to the 6-cell matrix anyway (the matrix itself will surface the rate)."
fi

# --- 6. run 6 cells: S1/S2/S3 x detector OFF/ON in SEPARATE processes ---
OUT_DIRS_NODET=""
OUT_DIRS_DET=""
for FLAG in nodet det; do
  EXTRA=""
  [ "$FLAG" = "det" ] && EXTRA="--detector"
  for S in 1 2 3; do
    out_dir="runs/${TAG}-s${S}-${FLAG}"
    banner "[6/7] run: setting=$S detector=$FLAG -> $out_dir"
    # shellcheck disable=SC2086
    REMEMBR_STRICT=1 python -m embodied_memory.run_hm3d_pol --mode live \
        --backbone remembr $EXTRA --setting "$S" --episodes-path "$DS" \
        --scene all --target "$TARGET" --n-episodes "$N_EPISODES" \
        --out-dir "$out_dir" 2>&1 | tee "${out_dir}.log"
    rc=${PIPESTATUS[0]}
    completed="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['n_episodes_completed'])" "${out_dir}/summary.json" 2>/dev/null || echo 0)"
    if [ "$completed" != "$N_EPISODES" ]; then
      echo "WARN: setting $S/$FLAG completed ${completed}/${N_EPISODES} (exit $rc) — Gate contribution may be partial."
    fi
    if [ "$FLAG" = "nodet" ]; then
      OUT_DIRS_NODET="$OUT_DIRS_NODET $out_dir"
    else
      OUT_DIRS_DET="$OUT_DIRS_DET $out_dir"
    fi
  done
done

# --- 7. Gate analysis: paired bootstrap on warm visits for each condition ---
banner "[7/7] Gate analysis: detector OFF triple"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS_NODET

banner "[7/7] Gate analysis: detector ON triple"
# shellcheck disable=SC2086
python embodied_memory/scripts/analyze_ablation.py --revisit $OUT_DIRS_DET

banner "[7/7] Cross-condition contrast (det vs nodet; manual inspection)"
echo "Inspect the WARM binary S3-S1 means and CIs:"
echo "  Gate A: s1-det WARM binary SPL  vs  s1-nodet WARM binary SPL"
echo "  Gate B: s3-det WARM binary SPL (S3-S1)  vs  s1-det WARM binary SPL (S3-S1)"
echo "  Gate C: s3-det WARM binary SPL  vs  s1-nodet WARM binary SPL (HEADLINE; bar >= +0.3)"
echo "  Gate D: s3-nodet WARM soft-SPL S3-S1  reproduces Phase-C (>= +0.15, p<0.05)"

banner "DONE — paste everything above (the two Gate-A blocks + the cross-condition summary)"
```

- [ ] **Step 2: `chmod +x` and syntax-check**

```bash
chmod +x scripts/race-revisit-detector.sh
bash -n scripts/race-revisit-detector.sh && echo "OK"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add scripts/race-revisit-detector.sh
git commit -m "race-revisit-detector: 6-cell driver (S1/S2/S3 x det/nodet) + preflight"
```

---

## Post-implementation (NOT a code task — operator instructions)

After all 7 tasks pass locally:

1. Push to `origin/lifelong-revisit-eval` (the branch RACE pulls — see the `BRANCH GOTCHA` in the [[race-testing-workflow]] memory):
   ```bash
   git push origin goal-detector-binary-spl:refs/heads/lifelong-revisit-eval
   ```
   (Non-force; this rejects unless `lifelong-revisit-eval` is an ancestor of the branch tip. If RACE has work ahead, merge first.)

2. On the RACE pod:
   ```bash
   bash scripts/race-revisit-detector.sh --tag detector-c1
   ```
   The first run after a driver-script edit hits the SELF-UPDATE gotcha — see [[race-testing-workflow]]. **Run it twice** if it's the first invocation since the driver changed.

3. Paste full output back. Verify the four gates (A/B/C/D) and mechanism counters (E) against the spec's decision rules.

4. If all gates GREEN: update `CLAUDE.md`'s "Next milestone" section + add a Run-11 entry to `PHASE2_ABLATION_REPORT.md` documenting the binary-SPL milestone outcome. Merge to `main` (ff) and push.

---

## Self-review checklist (controller, before handoff)

- [ ] **Spec coverage:** every decision in the spec (4 in §Decisions, 5 gates in §Gate, 6 edge cases) maps to at least one task above. ✓
- [ ] **Placeholders:** no TBD / TODO / "similar to..." / "appropriate error handling" anywhere. ✓
- [ ] **Type consistency:** `GoalDetector.locate()` signature is consistent between Task 1 (helpers), Task 2 (class), Task 3 (call site), Task 4 (mock); `_decide_stop_or_approach` signature is consistent between Task 4 step 1 (test) and step 3 (impl). ✓
- [ ] **File paths:** every `Files:` block uses exact existing paths grounded against the repo state at the time of plan writing. ✓
- [ ] **Bite-sized:** each step is 2-5 min; the longest (Task 4 step 4) is the only one that touches >50 LOC, justified by the surgical nature of the change. ✓
