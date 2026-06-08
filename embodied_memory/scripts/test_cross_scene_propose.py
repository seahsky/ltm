"""
Sanity tests for the cross-environment seam in
``EmbodiedMemoryBridge.propose_memory_candidates`` (step 2 of the diagnose-first
program).

The action-path retrieval hard-filters fine-layer hits to the current scene
(``memory_bridge.py:829``), so a sighting accumulated in scene A can never
contribute while the agent is in scene B — cross-environment reuse (the
proposal's actual thesis) is structurally impossible today. This adds an
env-gated seam, ``LTM_CROSS_SCENE``:

  * OFF (default): byte-identical behaviour — a scene-mismatched hit is dropped.
  * ON: a scene-mismatched hit that clears the cosine bar is COUNTED (the agent
    genuinely recalled the goal category from another scene) but still NOT
    injected as a waypoint — its stored ``agent_position`` is in the *other*
    scene's coordinate frame, so navigating there is geometrically meaningless.
    The counter (``stats()["n_cross_scene_recall"]``) makes the cross-env recall
    measurable; producing a *positive* cross-env waypoint needs a region /
    affordance mechanism (step 4), not raw fine-layer geometry.

Runs on the REAL bridge (faiss + dialogue_memory). SKIP-prints (exit 0) when the
heavy deps are unavailable, mirroring the other RACE-capable suites.

Invoke with::

    python embodied_memory/scripts/test_cross_scene_propose.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from embodied_memory.memory_bridge import EmbodiedMemoryBridge
except ImportError as e:  # faiss / torch / transformers missing locally
    print(f"SKIP test_cross_scene_propose: heavy deps unavailable ({e})")
    sys.exit(0)


_DIM = 8


def _encode(text: str) -> np.ndarray:
    """Deterministic category-keyed encoder so cosines are exact."""
    t = text.lower()
    v = np.zeros(_DIM, dtype=np.float32)
    if "chair" in t:
        v[0] = 1.0
    elif "bed" in t:
        v[1] = 1.0
    else:
        v[2] = 1.0
    return v


def _mk_bridge() -> EmbodiedMemoryBridge:
    # Default toggles = S3 (full system): STM + consolidation + LTM + rerank.
    return EmbodiedMemoryBridge(
        text_embed_dim=_DIM,
        visual_embed_dim=_DIM,
        text_encode_fn=_encode,
    )


def _insert_chair(bridge, scene_id: str, pos=(1.0, 0.0, 2.0)) -> None:
    bridge.ltm.insert(
        level="fine",
        embedding=_encode("a wooden chair"),
        content="a wooden chair by the desk",
        metadata={
            "scene_id": scene_id,
            "agent_position": list(pos),
            "step_idx": 3,
            "episode_id": f"ep-{scene_id}",
        },
    )


def _propose(bridge):
    return bridge.propose_memory_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        agent_yaw=0.0,
        target_category="chair",
    )


def _without_flag(fn):
    prev = os.environ.pop("LTM_CROSS_SCENE", None)
    try:
        return fn()
    finally:
        if prev is not None:
            os.environ["LTM_CROSS_SCENE"] = prev


def _with_flag(fn):
    prev = os.environ.get("LTM_CROSS_SCENE")
    os.environ["LTM_CROSS_SCENE"] = "1"
    try:
        return fn()
    finally:
        if prev is None:
            os.environ.pop("LTM_CROSS_SCENE", None)
        else:
            os.environ["LTM_CROSS_SCENE"] = prev


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------


def case_cross_scene_off_drops_mismatch():
    """Default (flag off): a scene-A sighting is dropped in scene B, and the
    cross-scene recall counter stays 0 (current behaviour preserved)."""
    bridge = _mk_bridge()
    _insert_chair(bridge, scene_id="A")
    bridge.begin_episode("ep-B", scene_id="B")
    out = _without_flag(lambda: _propose(bridge))
    assert out == [], f"scene-mismatched hit must be dropped when flag off, got {out}"
    assert bridge.stats()["n_cross_scene_recall"] == 0, bridge.stats()
    print("  case cross_scene_off_drops_mismatch: OK")


def case_cross_scene_on_counts_not_injects():
    """Flag on: the scene-A sighting is RECALLED (counter += 1) but NOT injected
    as a waypoint — its position is in scene A's frame, invalid in scene B."""
    bridge = _mk_bridge()
    _insert_chair(bridge, scene_id="A")
    bridge.begin_episode("ep-B", scene_id="B")
    out = _with_flag(lambda: _propose(bridge))
    assert out == [], f"cross-scene hit must NOT be injected (geometry guard), got {out}"
    assert bridge.stats()["n_cross_scene_recall"] == 1, bridge.stats()
    print("  case cross_scene_on_counts_not_injects: OK")


def case_same_scene_injected_regardless_of_flag():
    """A same-scene sighting is injected as a memory candidate whether or not the
    cross-scene flag is set, and never counts as a cross-scene recall."""
    bridge = _mk_bridge()
    _insert_chair(bridge, scene_id="B")
    bridge.begin_episode("ep-B", scene_id="B")
    out = _with_flag(lambda: _propose(bridge))
    assert len(out) == 1 and out[0].source == "memory", \
        f"same-scene hit must inject one memory candidate, got {out}"
    assert bridge.stats()["n_cross_scene_recall"] == 0, bridge.stats()
    print("  case same_scene_injected_regardless_of_flag: OK")


def main() -> int:
    print("cross-environment propose seam sanity tests")
    case_cross_scene_off_drops_mismatch()
    case_cross_scene_on_counts_not_injects()
    case_same_scene_injected_regardless_of_flag()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
