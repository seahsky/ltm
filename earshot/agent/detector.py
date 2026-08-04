"""``GoalDetector.detects(obj)`` — one seam, two arms, serving two questions.

ADR-0008's detector seam answers "is object X here", and the same call serves both the
**primary-task STOP** and the anomaly response's **visual confirm**. That is not a
convenience: §4.1's arrival rule is peak-or-plateau *plus* visual confirm, so if the two
were different components the realizable arm's arrival criterion and the primary task's
success criterion could disagree about the same object in the same frame.

Two implementations ship. ``OracleDetector`` is what the smoke runs, and the disclosure
rides with it: an oracle STOP deletes the ``stop_miss`` half of the 0.031 benchmark-SPL
decomposition outright, so smoke find numbers are not capability numbers.
``CaptionDetector`` is the R2 arm and ships **live but untested until R2** — ticket 10's
disclosed cost, chosen over letting the seam ship with one side.

**The old class exposed ``locate(...)``, not ``detects(...)``**, so this is a reshaping
rather than a port. What carries is everything that decides whether a localization is
believable: ``parse_qwen_bbox``, ``robust_depth_at_pixel``, ``back_project_pinhole``, and
the L3 gates — the below-floor pre-filter and the floor-plane (xz) snap test. OWLv2 is
dropped: base and large both sat in the noise floor on HM3D sim renders (max box score
0.031 / 0.058), and the arc closed as an honest negative.

**Two corrections to what the old file did, both frame errors, both fixed here.**
``back_project_pinhole`` documented "camera +Z is forward", which is 180 degrees out from
habitat — the optical axis is ``-z`` — and it was handed a transform built from the
agent's *base* position, so every detection landed behind the agent and 0.88 m too low.
See ``occupancy.py`` for the derivation and for the recorded symptoms those two predict
(``n_detector_localized`` 0 across a matrix; a back-projection 0.76 m below the navmesh).
The clean room derives the transform from ``occupancy.camera_to_world``, so the detector
and the map cannot drift apart.

**The failure log does not carry.** The old ``_debug_log`` appended a JSON line per
failure to a path the detector held, and ADR-0013 makes ``report/artifacts.py`` the only
module in the tree that writes anything. The reasons are counters and a ``last_rejection``
string instead, which the audit record carries — the same information, in the artefact
that already exists to hold it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from earshot.agent.config import DetectorConfig
from earshot.agent.occupancy import Intrinsics, camera_to_world, intrinsics_from_hfov
from earshot.types import Pose, Xyz

__all__ = [
    "Detection",
    "GoalDetector",
    "Grounder",
    "OracleDetector",
    "CaptionDetector",
    "parse_qwen_bbox",
    "robust_depth_at_pixel",
    "back_project_pinhole",
]


# Both Qwen2-VL grounding formats: `<|box_start|>x1,y1,x2,y2<|box_end|>` and the
# documented native-grounding paren form `<|box_start|>(x1,y1),(x2,y2)<|box_end|>`.
# Carried verbatim.
_BBOX_RE = re.compile(
    r"<\|box_start\|>"
    r"\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?"
    r"\s*,\s*"
    r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*"
    r"<\|box_end\|>"
)


@dataclass(frozen=True)
class Detection:
    """A believed localization: what, where, and the evidence it rests on."""

    object_name: str
    position: Xyz
    distance_m: float
    depth_m: float
    bbox: Tuple[int, int, int, int]


@runtime_checkable
class GoalDetector(Protocol):
    """The seam. ``observe`` takes the current frame, ``detects`` answers about an object.

    Split in two because the oracle answers from the simulator's geometry and the caption
    arm answers from pixels, and the runner's loop must not have to know which it holds.
    ``observe`` is therefore part of the protocol even though the oracle ignores it.
    """

    def observe(self, *, rgb: object, depth: object, pose: Pose) -> None:
        ...  # pragma: no cover - protocol

    def detects(self, obj: str) -> bool:
        ...  # pragma: no cover - protocol


@runtime_checkable
class Grounder(Protocol):
    """Anything that emits grounding tokens for an object in an image.

    The structural interface of ``vlm.py``'s Qwen2-VL-2B connector, declared at the
    consumer in the shape ``audio/normality.py``'s ``Captioner`` set: the concrete model
    drags torch and transformers, and this layer's whole Mac surface depends on importing
    neither.

    **``ground()`` may return the whole decode, prompt echo included** — that is what the
    old path did (``goal_detector._infer`` returned
    ``batch_decode(..., skip_special_tokens=False)`` over the full sequence and handed it
    straight to the parser; the assistant-span extraction was only ever used in its debug
    log). It was safe for a reason nothing wrote down: the prompt inlines the box tokens as
    a *format hint* with placeholder coordinates, ``(x1,y1),(x2,y2)``, and ``_BBOX_RE``
    requires digits. A future prompt that spelled the hint with real numbers would produce
    a phantom detection from the prompt itself, so the contract is stated here and
    ``tests/mac/test_agent_detector.py`` pins the placeholder case.
    """

    def ground(self, image: object, obj: str) -> str:
        ...  # pragma: no cover - protocol


# ----------------------------------------------------------------------
# the pure helpers — carried
# ----------------------------------------------------------------------


def parse_qwen_bbox(
    text: str,
    image_hw: Tuple[int, int],
    normalized: Optional[bool] = None,
) -> List[Tuple[int, int, int, int]]:
    """Parse Qwen2-VL grounding bboxes out of a decode. Pixel-space tuples, possibly empty.

    Coordinates may be pixels matching the input image or Qwen's documented ``[0, 1000]``
    normalized space. With ``normalized=None`` this auto-detects: if any coordinate across
    all matched boxes exceeds the larger image dimension, every box is treated as
    normalized and scaled. Pixel output is bounded by the image, so the signal is
    unambiguous in the common case.

    Malformed tokens are dropped rather than raised on — a model emitting prose instead of
    a box is a detection failure, not a program error, and the caller's counter records it.
    """
    height, width = image_hw
    raw: List[Tuple[int, int, int, int]] = []
    for match in _BBOX_RE.finditer(text or ""):
        try:
            x1, y1, x2, y2 = (int(g) for g in match.groups())
        except ValueError:
            continue
        raw.append((x1, y1, x2, y2))
    if not raw:
        return []

    if normalized is None:
        normalized = max(max(box) for box in raw) > max(height, width)

    out: List[Tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in raw:
        if normalized:
            x1 = int(round(x1 * width / 1000.0))
            x2 = int(round(x2 * width / 1000.0))
            y1 = int(round(y1 * height / 1000.0))
            y2 = int(round(y2 * height / 1000.0))
        if x2 <= x1 or y2 <= y1:
            continue
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            continue
        out.append((x1, y1, x2, y2))
    return out


def robust_depth_at_pixel(
    depth: object, u: int, v: int, patch: int = 5
) -> Optional[float]:
    """Median depth in a ``patch x patch`` window, or ``None`` if it is all invalid.

    Rejects NaN, 0.0 and infinities: habitat returns 0 for "no return" and inf can appear
    near the far clip plane, and either would back-project to a point that is not a
    surface.
    """
    frame = np.asarray(depth, dtype=np.float64)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    height, width = frame.shape[:2]
    half = int(patch) // 2
    u0, u1 = max(int(u) - half, 0), min(int(u) + half + 1, width)
    v0, v1 = max(int(v) - half, 0), min(int(v) + half + 1, height)
    window = frame[v0:v1, u0:u1].ravel()
    valid = np.isfinite(window) & (window > 0.0)
    if not valid.any():
        return None
    return float(np.median(window[valid]))


def back_project_pinhole(
    u: int,
    v: int,
    depth: float,
    intrinsics: Intrinsics,
    world_from_camera: np.ndarray,
) -> Optional[Xyz]:
    """``(u, v, depth)`` through the pinhole and the camera transform, to a world point.

    ``world_from_camera`` is the 4x4 ``occupancy.camera_to_world`` builds.

    Habitat's camera convention: ``x`` right, ``y`` up, and the optical axis along
    ``-z``, so a pixel at the principal point with depth ``d`` is ``(0, 0, -d)`` in camera
    coordinates. **The old file used ``+d``**, which put every detection behind the agent;
    the transform's own two errors are corrected in ``occupancy.camera_to_world``.

    ``None`` for a depth that is not a positive finite number, so an invalid reading
    cannot become a coordinate.
    """
    if depth is None or not np.isfinite(depth) or float(depth) <= 0.0:
        return None
    d = float(depth)
    point_camera = np.array(
        [
            (float(u) - intrinsics.cx) * d / intrinsics.fx,
            (intrinsics.cy - float(v)) * d / intrinsics.fy,
            -d,
            1.0,
        ],
        dtype=np.float64,
    )
    return Xyz.from_sequence((world_from_camera @ point_camera)[:3])


# ----------------------------------------------------------------------
# the two arms
# ----------------------------------------------------------------------


class OracleDetector:
    """"Is object X here" answered from the simulator's geometry. What the smoke runs.

    The distance function is injected (ADR-0013): the runner passes something like
    ``lambda obj: world.geodesic_distance(world.pose().position, view_points[obj])``, so
    this class holds no simulator handle and unit-tests against a dict.

    ``None`` from the distance function means "the simulator could not answer" — no such
    object in this episode, or no navmesh route to it — and reads as *not detected*. The
    alternative, treating an unanswerable query as a detection, would STOP the episode on
    a missing goal and score it as a success.

    **Required disclosure.** This is an oracle STOP: it does not exercise goal detection
    at all. ``diagnose_spin`` decomposed the 0.031 benchmark SPL as stop_miss ~50% +
    explore_timeout ~45% + success ~5%, so this deletes the stop_miss half. Smoke find
    numbers will look far better for a reason that must be disclosed rather than enjoyed.
    """

    def __init__(
        self,
        distance_to: Callable[[str], Optional[float]],
        cfg: Optional[DetectorConfig] = None,
    ) -> None:
        self._distance_to = distance_to
        self.cfg = cfg or DetectorConfig()
        self.last_distance_m: Optional[float] = None
        self.n_queries = 0
        self.n_detections = 0

    def observe(self, *, rgb: object, depth: object, pose: Pose) -> None:
        """A no-op, and deliberately not an error.

        The oracle reads geometry, not pixels. It is in the protocol so the runner's loop
        is the same shape whichever arm is configured — ``Detector.ORACLE`` and
        ``Detector.CAPTION`` are an experimental arm, not two code paths.
        """
        del rgb, depth, pose

    def detects(self, obj: str) -> bool:
        self.n_queries += 1
        distance = self._distance_to(obj)
        self.last_distance_m = None if distance is None else float(distance)
        if distance is None:
            return False
        found = float(distance) <= float(self.cfg.oracle_radius_m)
        if found:
            self.n_detections += 1
        return found


class CaptionDetector:
    """"Is object X here" answered from a grounding VLM, depth, and the navmesh.

    The R2 arm. A detection is believed only if it survives, in order: a parsed box, a
    valid depth at its centre, a back-projection, the below-floor pre-filter, the
    floor-plane snap gate, and finally being *within reach* — the question the seam asks
    is "here", not "somewhere".

    **The reach test is the piece that is new**, because the old ``locate()`` returned a
    waypoint for the runner to approach and this returns a verdict. Everything before it
    is carried unchanged, and the radius is ``DetectorConfig.here_radius_m``, set at
    Find-SR's 1.0 m primary ring so both arms answer the same question.

    Rejections are counted by reason and the most recent one is readable, which is what
    made the old arc diagnosable at all: ``n_detector_localized`` was 0 across a whole
    matrix, and the only way to see why without re-running it was a per-failure record.
    """

    REJECTIONS = (
        "no_frame",
        "empty_parse",
        "all_depths_invalid",
        "back_project_failed",
        "off_navmesh",
        "below_floor",
        "snap_too_far",
        "out_of_reach",
    )

    def __init__(
        self,
        grounder: Grounder,
        snap_point: Callable[[Xyz], Optional[Xyz]],
        *,
        cfg: Optional[DetectorConfig] = None,
        hfov_deg: float = 79.0,
        eye_height_m: float = 0.88,
    ) -> None:
        self._grounder = grounder
        self._snap_point = snap_point
        self.cfg = cfg or DetectorConfig()
        self.hfov_deg = float(hfov_deg)
        self.eye_height_m = float(eye_height_m)
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._pose: Optional[Pose] = None
        self.last_detection: Optional[Detection] = None
        self.last_rejection: Optional[str] = None
        self.n_queries = 0
        self.n_detections = 0
        self.rejections: Dict[str, int] = {reason: 0 for reason in self.REJECTIONS}

    def observe(self, *, rgb: object, depth: object, pose: Pose) -> None:
        """Hold this step's frame. RGB and depth come from the same render (ticket 21)."""
        self._rgb = None if rgb is None else np.asarray(rgb)
        self._depth = None if depth is None else np.asarray(depth)
        self._pose = pose

    def detects(self, obj: str) -> bool:
        self.n_queries += 1
        self.last_detection = None
        self.last_rejection = None
        detection = self._locate(obj)
        if detection is None:
            return False
        self.last_detection = detection
        self.n_detections += 1
        return True

    # -- internals -------------------------------------------------------

    def _reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1
        self.last_rejection = reason

    def _locate(self, obj: str) -> Optional[Detection]:
        if self._rgb is None or self._depth is None or self._pose is None:
            self._reject("no_frame")
            return None
        rgb, depth, pose = self._rgb, self._depth, self._pose
        image_hw = (int(rgb.shape[0]), int(rgb.shape[1]))

        boxes = parse_qwen_bbox(self._grounder.ground(rgb, obj), image_hw=image_hw)
        if not boxes:
            self._reject("empty_parse")
            return None

        # The closest surface wins. A goal category can match several boxes and the
        # nearest one is the instance the agent could actually be standing at.
        best: Optional[Tuple[int, int, float, Tuple[int, int, int, int]]] = None
        for (x1, y1, x2, y2) in boxes:
            u, v = (x1 + x2) // 2, (y1 + y2) // 2
            d = robust_depth_at_pixel(depth, u, v, patch=int(self.cfg.depth_patch_px))
            if d is None:
                continue
            if best is None or d < best[2]:
                best = (u, v, d, (x1, y1, x2, y2))
        if best is None:
            self._reject("all_depths_invalid")
            return None
        u, v, d, bbox = best

        intrinsics = intrinsics_from_hfov(image_hw[1], image_hw[0], self.hfov_deg)
        world = back_project_pinhole(
            u, v, d, intrinsics, camera_to_world(pose, self.eye_height_m)
        )
        if world is None:
            self._reject("back_project_failed")
            return None

        snapped = self._snap_point(world)
        if snapped is None:
            self._reject("off_navmesh")
            return None
        # Below-floor FIRST: a point back-projected under the navmesh is depth overshoot,
        # not a surface, and the floor-plane gate below would happily rescue one that
        # overshot straight down.
        if world.y < snapped.y - float(self.cfg.snap_floor_eps_m):
            self._reject("below_floor")
            return None
        if snapped.horizontal_distance_to(world) > float(self.cfg.max_snap_m):
            self._reject("snap_too_far")
            return None

        distance = pose.position.horizontal_distance_to(snapped)
        if distance > float(self.cfg.here_radius_m):
            self._reject("out_of_reach")
            return None
        return Detection(
            object_name=obj,
            position=snapped,
            distance_m=distance,
            depth_m=float(d),
            bbox=bbox,
        )
