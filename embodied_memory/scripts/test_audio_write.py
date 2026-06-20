"""
TDD for Step 2 — MemoryBridge.write_audio_event (persist a heard audio anomaly as
a recallable fine-layer LTM item). Pure CPU, deterministic category-keyed encoder
(same pattern as test_cross_scene_propose.py) so cosines are exact; no SBERT/CLAP.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_audio_write.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from embodied_memory.memory_bridge import EmbodiedMemoryBridge  # noqa: E402

_DIM = 8


def _encode(text: str) -> np.ndarray:
    """Deterministic category-keyed encoder so the goal query and the audio-event
    caption (which share the '{cat}' token) land on the same basis axis."""
    t = text.lower()
    v = np.zeros(_DIM, dtype=np.float32)
    if "chair" in t:
        v[0] = 1.0
    elif "bed" in t:
        v[1] = 1.0
    else:
        v[2] = 1.0
    return v


def _mk_bridge(disable_ltm: bool = False) -> EmbodiedMemoryBridge:
    return EmbodiedMemoryBridge(
        text_embed_dim=_DIM, visual_embed_dim=_DIM,
        text_encode_fn=_encode, disable_ltm=disable_ltm,
    )


def _propose(bridge, cat="bed", agent=(5.0, 0.0, 5.0), planner_xys=None):
    return bridge.propose_memory_candidates(
        agent_pos=np.array(agent, dtype=np.float32),
        agent_yaw=0.0,
        target_category=cat,
        planner_world_xys=planner_xys or [],
    )


def case_write_inserts_retrievable_fine_item():
    b = _mk_bridge()
    b.begin_episode("ep-S", scene_id="S")
    eid = b.write_audio_event("bed", [1.0, 0.0, 2.0], step_idx=30, anomaly_class="alarm")
    assert eid is not None
    assert len(b.ltm.fine) == 1, len(b.ltm.fine)
    entry = b.ltm.fine.entries[0]
    assert entry.content == "there is a bed; heard alarm here", entry.content
    assert entry.metadata["type"] == "audio_event"
    assert entry.metadata["scene_id"] == "S"
    assert entry.metadata["agent_position"] == [1.0, 0.0, 2.0]
    assert b.modules_invoked.get("ltm_audio_write") is True
    print("  case write_inserts_retrievable_fine_item: OK")


def case_write_passes_three_gates_and_emits_waypoint():
    b = _mk_bridge()
    b.begin_episode("ep-S", scene_id="S")
    b.write_audio_event("bed", [1.0, 0.0, 2.0], step_idx=30, anomaly_class="alarm")
    out = _propose(b, cat="bed")
    assert len(out) == 1 and out[0].source == "memory", out
    # routes to the SOURCE xz, not the agent pose
    assert np.allclose(out[0].world_xy, [1.0, 2.0], atol=1e-4), out[0].world_xy
    print("  case write_passes_three_gates_and_emits_waypoint: OK")


def case_routes_to_source_not_agent_pose():
    # the whole experiment: the recalled waypoint is the sound location, so an
    # agent far from the source still gets a waypoint pointing AT it.
    b = _mk_bridge()
    b.begin_episode("ep-S", scene_id="S")
    b.write_audio_event("bed", [9.0, 0.0, -3.0], step_idx=12, anomaly_class="glass_break")
    out = _propose(b, cat="bed", agent=(0.0, 0.0, 0.0))
    assert len(out) == 1 and np.allclose(out[0].world_xy, [9.0, -3.0], atol=1e-4), out
    print("  case routes_to_source_not_agent_pose: OK")


def case_scene_mismatch_gate():
    b = _mk_bridge()
    b.begin_episode("ep-S", scene_id="S")
    b.write_audio_event("bed", [1.0, 0.0, 2.0], step_idx=30, anomaly_class="alarm")
    b.begin_episode("ep-OTHER", scene_id="OTHER")  # different scene at query time
    out = _propose(b, cat="bed")
    assert out == [], f"scene-mismatched audio write must not inject, got {out}"
    print("  case scene_mismatch_gate: OK")


def case_self_dedup_avoided_but_planner_dedup_holds():
    # source far from the agent's own frontier -> emitted; frontier ON the source -> deduped
    b = _mk_bridge()
    b.begin_episode("ep-S", scene_id="S")
    b.write_audio_event("bed", [10.0, 0.0, 10.0], step_idx=5, anomaly_class="alarm")
    out_far = _propose(b, cat="bed", agent=(0.0, 0.0, 0.0), planner_xys=[np.array([0.0, 0.0])])
    assert len(out_far) == 1, out_far
    out_dedup = _propose(b, cat="bed", agent=(0.0, 0.0, 0.0),
                         planner_xys=[np.array([10.0, 10.0])])
    assert out_dedup == [], f"frontier at the source must dedup the audio waypoint, got {out_dedup}"
    print("  case self_dedup_avoided_but_planner_dedup_holds: OK")


def case_disable_ltm_returns_none():
    b = _mk_bridge(disable_ltm=True)
    b.begin_episode("ep-S", scene_id="S")
    eid = b.write_audio_event("bed", [1.0, 0.0, 2.0], step_idx=30, anomaly_class="alarm")
    assert eid is None and len(b.ltm.fine) == 0
    print("  case disable_ltm_returns_none: OK")


def case_bad_inputs_return_none():
    b = _mk_bridge()
    b.begin_episode("ep-S", scene_id="S")
    assert b.write_audio_event("", [1.0, 0.0, 2.0], 1) is None          # no category
    assert b.write_audio_event("bed", None, 1) is None                  # no source
    assert b.write_audio_event("bed", [1.0, 0.0], 1) is None            # source < 3 dims
    assert len(b.ltm.fine) == 0
    print("  case bad_inputs_return_none: OK")


def case_caption_matches_object_not_class():
    # a write for object 'bed' must be recalled by the 'bed' query and NOT by 'chair'
    b = _mk_bridge()
    b.begin_episode("ep-S", scene_id="S")
    b.write_audio_event("bed", [1.0, 0.0, 2.0], 30, anomaly_class="alarm")
    assert len(_propose(b, cat="bed")) == 1
    assert _propose(b, cat="chair") == [], "object-category caption must not match a different goal"
    print("  case caption_matches_object_not_class: OK")


def main() -> int:
    cases = [
        case_write_inserts_retrievable_fine_item,
        case_write_passes_three_gates_and_emits_waypoint,
        case_routes_to_source_not_agent_pose,
        case_scene_mismatch_gate,
        case_self_dedup_avoided_but_planner_dedup_holds,
        case_disable_ltm_returns_none,
        case_bad_inputs_return_none,
        case_caption_matches_object_not_class,
    ]
    print(f"running {len(cases)} write_audio_event cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
