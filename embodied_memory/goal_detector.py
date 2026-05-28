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
    # Accepts both Qwen2-VL formats:
    #   <|box_start|>x1,y1,x2,y2<|box_end|>          (flat)
    #   <|box_start|>(x1,y1),(x2,y2)<|box_end|>      (paren, the documented
    #                                                 native-grounding output)
    # Optional parens around each (x,y) pair tolerate either.
    r"<\|box_start\|>"
    r"\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?"
    r"\s*,\s*"
    r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*"
    r"<\|box_end\|>"
)


def _extract_assistant_output(text: str, max_chars: int = 800) -> str:
    """Pull the model's generated tokens out of a chat-template-formatted decode.

    Qwen2-VL's processor.batch_decode(..., skip_special_tokens=False) returns
    the full prompt + response. The prompt portion is large (system prompt +
    hundreds of <|image_pad|> tokens for a 256x256 image); the model's actual
    generation comes after ``<|im_start|>assistant\\n``. We slice that out so
    the failure log shows the diagnostic signal (was a bbox emitted? prose
    only? empty?) without being drowned in scaffolding.

    Falls back to the last ``max_chars`` chars of the input if the assistant
    marker isn't found (defensive: keeps the log useful even if the chat
    template format shifts).
    """
    if not text:
        return ""
    marker = "<|im_start|>assistant"
    idx = text.find(marker)
    if idx >= 0:
        tail = text[idx + len(marker):]
        # Strip the leading newline the chat template emits.
        tail = tail.lstrip("\n")
        return tail[:max_chars] + ("...[truncated]" if len(tail) > max_chars else "")
    # Marker not found: surface the literal tail so the format is still
    # diagnosable.
    tail = text[-max_chars:]
    return f"[no-assistant-marker]...{tail}"


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

    Camera convention matches the codebase (frontier_planner.py OccupancyGrid):
    x-right, y-up, camera +Z is forward (depth = +Zc). At yaw=0, agent faces
    +z in world; a pixel at the principal point with depth=D back-projects to
    world point (0, 0, +D).
    """
    if depth is None or not np.isfinite(depth) or depth <= 0.0:
        return None
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    # Camera-frame point. Codebase convention: +Z is forward (depth = +Zc).
    x_cam = (float(u) - cx) * depth / fx
    y_cam = -(float(v) - cy) * depth / fy   # image y points down; world y up
    z_cam = depth
    pt_cam = np.array([x_cam, y_cam, z_cam, 1.0], dtype=np.float32)
    pt_world = (agent_pose @ pt_cam)[:3]
    return pt_world.astype(np.float32)


# ----------------------------------------------------------------------
# GoalDetector
# ----------------------------------------------------------------------


class GoalDetector:
    """Precise final-approach goal localizer.

    Reuses the already-loaded Qwen2-VL captioner (no new GPU memory) to emit
    a bbox for the goal category, then back-projects through depth + agent
    pose and snaps to the Habitat navmesh.
    """

    def __init__(
        self,
        model,
        processor,
        pathfinder,
        max_snap_dist: float = 0.5,
        max_new_tokens: int = 64,
        device: Optional[str] = None,
        debug_log_path: Optional[str] = None,
    ):
        self.model = model
        self.processor = processor
        self.pathfinder = pathfinder
        self.max_snap_dist = float(max_snap_dist)
        self.max_new_tokens = int(max_new_tokens)
        self.device = device
        # When set, every locate() failure appends one JSON line capturing the
        # reason + (truncated) decoded Qwen-VL output. Detector-c1 found
        # n_detector_localized=0 across the whole matrix; this is the only way
        # to see what the model is actually emitting without re-running the
        # full ablation.
        self.debug_log_path = debug_log_path

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
          1. Prompt Qwen2-VL: "Please locate the {goal_category} ..."
          2. Parse <|box_start|>...<|box_end|> tokens.
          3. For each bbox, compute robust depth at center; pick the bbox
             with the *lowest* center depth (closest physical surface).
          4. Back-project (uc, vc, d_center) -> 3D world point.
          5. pathfinder.snap_point(point). If the snap jumped further than
             ``max_snap_dist``, return None (off-navmesh).
        """
        text = self._infer(rgb, goal_category)
        image_hw = (rgb.shape[0], rgb.shape[1])
        bboxes = parse_qwen_bbox(text, image_hw=image_hw)
        # We do NOT fall back to normalized=True: Qwen2-VL-2B-Instruct emits
        # pixel-space coordinates by default. Re-interpreting slightly-out-of-
        # bounds pixel coords as normalized [0,1000] would produce a spurious
        # bbox at a different location. If the pre-flight smoke surfaces
        # normalized output instead, flip this default.
        if not bboxes:
            self._debug_log("empty_parse", decoded=text, goal_category=goal_category)
            return None

        best = None
        best_depth = float("inf")
        depths_seen: List[Optional[float]] = []
        for (x1, y1, x2, y2) in bboxes:
            uc = (x1 + x2) // 2
            vc = (y1 + y2) // 2
            d = robust_depth_at_pixel(depth, u=uc, v=vc, patch=5)
            depths_seen.append(d)
            if d is None:
                continue
            if d < best_depth:
                best_depth = d
                best = (uc, vc, d)
        if best is None:
            self._debug_log(
                "all_depths_invalid",
                decoded=text, goal_category=goal_category,
                n_bboxes=len(bboxes),
                depths=[float(d) if d is not None else None for d in depths_seen],
            )
            return None

        uc, vc, d = best
        world_pt = back_project_pinhole(
            u=uc, v=vc, depth=d, intrinsics=intrinsics, agent_pose=agent_pose,
        )
        if world_pt is None:
            self._debug_log(
                "back_project_failed",
                decoded=text, goal_category=goal_category,
                uc=int(uc), vc=int(vc), depth=float(d),
            )
            return None

        snapped = np.asarray(self.pathfinder.snap_point(world_pt), dtype=np.float32)
        if not np.all(np.isfinite(snapped)):
            self._debug_log(
                "snap_not_finite",
                decoded=text, goal_category=goal_category,
                world_pt=[float(x) for x in world_pt],
            )
            return None
        snap_dist = float(np.linalg.norm(snapped - world_pt))
        if snap_dist > self.max_snap_dist:
            self._debug_log(
                "snap_too_far",
                decoded=text, goal_category=goal_category,
                world_pt=[float(x) for x in world_pt],
                snapped=[float(x) for x in snapped],
                snap_dist=snap_dist, max_snap_dist=self.max_snap_dist,
            )
            return None
        return snapped

    def _debug_log(self, reason: str, decoded: str = "", goal_category: str = "", **extra) -> None:
        """Append one JSON line per locate() failure to self.debug_log_path.

        No-op when debug_log_path is None. Wrapped in try/except so a logging
        failure can never crash the agent loop.
        """
        if not self.debug_log_path:
            return
        try:
            import json, os, time
            os.makedirs(os.path.dirname(self.debug_log_path) or ".", exist_ok=True)
            text = decoded or ""
            entry = {
                "ts": time.time(),
                "reason": reason,
                "goal_category": goal_category,
                "decoded_len": len(text),
                # The decoded text begins with the chat-template echo (system
                # prompt + hundreds of <|image_pad|> vision tokens). Naive
                # head-only truncation hides the model's actual generated
                # output, which lives AFTER the <|im_start|>assistant marker.
                # Extract it explicitly so the log surfaces the diagnostic
                # signal we actually need (whether the model emitted box
                # tokens, prose, or nothing). c2 wasted a 24-min matrix on
                # this — never again.
                "assistant_output": _extract_assistant_output(text, max_chars=800),
                # Keep the head as a sanity check (confirms the prompt
                # structure looks right) and the tail in case our marker is
                # missing for some reason.
                "decoded_head": text[:200],
                "decoded_tail": text[-300:] if len(text) > 200 else "",
                **extra,
            }
            with open(self.debug_log_path, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Never let logging crash the agent.

    def _infer(self, rgb: np.ndarray, goal_category: str) -> str:
        """One forward pass through Qwen2-VL for grounding.

        Returns the decoded text (which contains <|box_start|>...<|box_end|>
        tokens on a successful detection).
        """
        from PIL import Image
        img = Image.fromarray(rgb)
        # Prompt history (each cost a RACE iteration):
        #   c1/c2/c3: "Please locate the {cat} in this image and return its
        #     bounding box." -> Qwen2-VL-2B-Instruct refused as "I'm sorry,
        #     but as an AI language model, I don't have the ability to see
        #     images or locate objects within them." (16/16 + 1 preflight =
        #     17/17 failures). The polite-VQA phrasing triggered the RLHF
        #     safety refusal despite a real image being attached.
        # Fix: use Qwen2-VL's grounding-task vocabulary (imperative "Locate",
        # no "please"), and *inline the expected output format* so the model
        # can't claim it doesn't know how to respond. The exact-token format
        # hint also biases sampling toward the <|box_start|> tokens we parse.
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text",
                 "text": (
                     f"Locate the {goal_category} in this image. "
                     f"Output the bounding box as "
                     f"<|object_ref_start|>{goal_category}<|object_ref_end|>"
                     f"<|box_start|>(x1,y1),(x2,y2)<|box_end|>."
                 )},
            ],
        }]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=img, return_tensors="pt")
        if self.device is not None:
            inputs = inputs.to(self.device)
        # BatchEncoding (transformers) is a dict subclass; mock objects may not be.
        generate_kwargs: Dict = dict(inputs) if isinstance(inputs, dict) else {"input_ids": inputs.input_ids}
        out_ids = self.model.generate(**generate_kwargs, max_new_tokens=self.max_new_tokens)
        decoded = self.processor.batch_decode(out_ids, skip_special_tokens=False)
        return decoded[0] if decoded else ""
