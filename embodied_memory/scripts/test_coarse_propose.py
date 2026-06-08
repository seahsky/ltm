"""
Sanity tests for the coarse-affordance proposer (step 4, Stage 2):
``EmbodiedMemoryBridge.propose_coarse_candidates`` + the ``FrontierPhysicsScorer``
``source=="coarse"`` branch.

The coarse head grounds a position-free ``category -> preferred_room`` prior to the
CURRENT scene: given room_anchors (position, caption) from the live scene, it emits
ONE FrontierCandidate(source="coarse") at the nearest anchor whose caption resolves
to the goal category's affordant room-type. This is the cross-env mechanism — it
needs no stored scene position, so it fires in a brand-new scene where the fine
layer (scene-filtered) has nothing.

Runs on the REAL bridge (faiss + dialogue_memory). SKIP-prints (exit 0) when the
heavy deps are unavailable.

Invoke with::

    python embodied_memory/scripts/test_coarse_propose.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from embodied_memory.memory_bridge import EmbodiedMemoryBridge, FrontierPhysicsScorer
    from embodied_memory.frontier_planner import FrontierCandidate
except ImportError as e:  # faiss / torch / transformers missing locally
    print(f"SKIP test_coarse_propose: heavy deps unavailable ({e})")
    sys.exit(0)


def _mk_bridge() -> EmbodiedMemoryBridge:
    return EmbodiedMemoryBridge(
        text_embed_dim=8, visual_embed_dim=8,
        text_encode_fn=lambda s: np.zeros(8, dtype=np.float32),
    )


def _propose(bridge, cat, anchors, **kw):
    bridge.begin_episode("ep-coarse", scene_id="SCENE_B")
    return bridge.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        agent_yaw=0.0, target_category=cat, room_anchors=anchors, **kw,
    )


# ----------------------------------------------------------------------
# propose_coarse_candidates — grounding the prior to current-scene anchors
# ----------------------------------------------------------------------


def case_grounds_to_matching_room():
    # chair -> living_room; one living-room anchor, one bedroom anchor -> emit the
    # living-room one as a source="coarse" candidate (no stored scene position used).
    anchors = [([1.0, 0.0, 2.0], "a cozy living room with a sofa and a window"),
               ([5.0, 0.0, 5.0], "a spacious bedroom with a bed and a lamp")]
    out = _propose(_mk_bridge(), "chair", anchors)
    assert len(out) == 1, out
    c = out[0]
    assert c.source == "coarse", c.source
    assert list(c.world_xy) == [1.0, 2.0], c.world_xy
    assert c.metadata["preferred_room"] == "living_room", c.metadata
    assert c.raw_score > 0.0, c.raw_score
    print("  case grounds_to_matching_room: OK")


def case_no_matching_room_returns_empty():
    anchors = [([5.0, 0.0, 5.0], "a spacious bedroom with a bed")]   # chair wants living_room
    assert _propose(_mk_bridge(), "chair", anchors) == []
    print("  case no_matching_room_returns_empty: OK")


def case_abstaining_anchor_ignored():
    # a caption with no room word (resolve_room -> None) is never a match
    anchors = [([1.0, 0.0, 2.0], "a wooden chair tucked under a table")]
    assert _propose(_mk_bridge(), "chair", anchors) == []
    print("  case abstaining_anchor_ignored: OK")


def case_unknown_category_returns_empty():
    anchors = [([1.0, 0.0, 2.0], "a cozy living room")]
    assert _propose(_mk_bridge(), "no_such_category", anchors) == []
    print("  case unknown_category_returns_empty: OK")


def case_nearest_matching_anchor_wins():
    # two living-room anchors; the NEARER one is emitted (top_k=1)
    anchors = [([10.0, 0.0, 0.0], "a large living room with a fireplace"),
               ([2.0, 0.0, 0.0], "a cozy living room with a couch")]
    out = _propose(_mk_bridge(), "chair", anchors)
    assert len(out) == 1 and list(out[0].world_xy) == [2.0, 0.0], out
    print("  case nearest_matching_anchor_wins: OK")


def case_dedup_against_planner_xys():
    anchors = [([1.0, 0.0, 2.0], "a cozy living room")]
    out = _propose(_mk_bridge(), "chair", anchors,
                   planner_world_xys=[np.array([1.1, 2.1], dtype=np.float32)])
    assert out == [], out      # within dedup radius of a planner candidate
    print("  case dedup_against_planner_xys: OK")


# ----------------------------------------------------------------------
# FrontierPhysicsScorer source=="coarse" branch
# ----------------------------------------------------------------------


def case_defaults_to_stm_pending():
    # with no explicit room_anchors, the proposer grounds on the current episode's
    # STM buffer (observe_keyframe -> _pending: caption + agent_position).
    from types import SimpleNamespace
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")

    def _kf(cap, pos):
        return SimpleNamespace(
            step_idx=0, caption=cap,
            text_embedding=np.zeros(8, dtype=np.float32),
            visual_embedding=np.zeros(8, dtype=np.float32),
            agent_position=np.array(pos, dtype=np.float32), agent_yaw=0.0)

    b.observe_keyframe(_kf("a cozy living room with a sofa and a window", [3.0, 0.0, 0.0]),
                       action=1, reward=0.0)
    b.observe_keyframe(_kf("a spacious bedroom with a bed and a lamp", [9.0, 0.0, 0.0]),
                       action=1, reward=0.0)
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        agent_yaw=0.0, target_category="chair")    # no room_anchors -> uses _pending
    assert len(out) == 1 and list(out[0].world_xy) == [3.0, 0.0], out
    print("  case defaults_to_stm_pending: OK")


def case_scorer_coarse_branch_drops_bearing():
    # raw_score dominant, bearing IGNORED (heading to the affordant room is worth a
    # turn) — so a coarse candidate facing backwards still scores well.
    cand = FrontierCandidate(
        candidate_id=1, world_xy=np.array([0.0, 2.0], dtype=np.float32),
        grid_rc=(-1, -1), distance_m=2.0, bearing_rad=3.0,  # ~backwards
        cluster_size=0, raw_score=0.7, source="coarse", metadata={},
    )
    s = FrontierPhysicsScorer().score("go", np.zeros(8, dtype=np.float32),
                                      {"frontier_candidate": cand})
    # (1-0.2)*0.7 + 0.2*dist_score(dist=2 -> 1.0) = 0.56 + 0.2 = 0.76
    assert abs(s - 0.76) < 1e-6, s
    # contrast: scored as a PLANNER candidate the backwards bearing would tank it
    planner = FrontierCandidate(
        candidate_id=2, world_xy=np.array([0.0, 2.0], dtype=np.float32),
        grid_rc=(-1, -1), distance_m=2.0, bearing_rad=3.0,
        cluster_size=0, raw_score=0.7, source="planner", metadata={},
    )
    sp = FrontierPhysicsScorer().score("go", np.zeros(8, dtype=np.float32),
                                       {"frontier_candidate": planner})
    assert s > sp, (s, sp)   # coarse branch beats the bearing-penalised planner score
    print("  case scorer_coarse_branch_drops_bearing: OK")


def main() -> int:
    print("coarse-affordance proposer sanity tests")
    case_grounds_to_matching_room()
    case_no_matching_room_returns_empty()
    case_abstaining_anchor_ignored()
    case_unknown_category_returns_empty()
    case_nearest_matching_anchor_wins()
    case_dedup_against_planner_xys()
    case_defaults_to_stm_pending()
    case_scorer_coarse_branch_drops_bearing()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
