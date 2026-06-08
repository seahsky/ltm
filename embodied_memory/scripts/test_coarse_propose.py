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
# CLIP room_classifier plumbing (Stage 5): CLIP-first, caption-fallback. The
# binding constraint is room PERCEPTION — captions name the affordant room too
# rarely. A CLIP zero-shot room classifier (on each anchor's visual embedding)
# gives a DENSE room signal; resolve_room(caption) is the fallback when CLIP
# abstains or no embedding is present.
# ----------------------------------------------------------------------


def _emb(marker: float) -> np.ndarray:
    v = np.zeros(8, dtype=np.float32)
    v[0] = float(marker)
    return v


def _clf(mapping: dict):
    """Synthetic CLIP room classifier: emb[0] (rounded int) -> room or None."""
    def clf(emb):
        if emb is None:
            return None
        return mapping.get(int(round(float(emb[0]))))
    return clf


def case_clip_overrides_sparse_caption():
    # caption names NO room (resolve_room abstains) but CLIP says living_room ->
    # the anchor IS tagged living_room and the chair-goal coarse head fires. THE
    # WHOLE POINT: dense room signal where the caption keyword is silent.
    anchors = [([1.0, 0.0, 2.0], "a wooden chair tucked under a table", _emb(1))]
    out = _propose(_mk_bridge(), "chair", anchors, room_classifier=_clf({1: "living_room"}))
    assert len(out) == 1, out
    assert out[0].metadata["preferred_room"] == "living_room", out[0].metadata
    assert list(out[0].world_xy) == [1.0, 2.0], out[0].world_xy
    print("  case clip_overrides_sparse_caption: OK")


def case_clip_abstain_falls_back_to_caption():
    # CLIP abstains (emb marker not in map -> None) but the caption DOES name the
    # room -> caption fallback still tags it. CLIP augments, never removes coverage.
    anchors = [([1.0, 0.0, 2.0], "a cozy living room with a sofa", _emb(9))]
    out = _propose(_mk_bridge(), "chair", anchors, room_classifier=_clf({}))  # 9 -> None
    assert len(out) == 1 and out[0].metadata["preferred_room"] == "living_room", out
    print("  case clip_abstain_falls_back_to_caption: OK")


def case_clip_confident_label_overrides_caption_keyword():
    # caption keyword says living_room but CLIP is CONFIDENT it's a bedroom ->
    # CLIP-first trusts the perceptual label -> chair(wants living) finds no match.
    anchors = [([1.0, 0.0, 2.0], "a cozy living room", _emb(2))]
    out = _propose(_mk_bridge(), "chair", anchors, room_classifier=_clf({2: "bedroom"}))
    assert out == [], out
    print("  case clip_confident_label_overrides_caption_keyword: OK")


def case_no_classifier_is_caption_only_backcompat():
    # without a classifier the proposer is caption-only (unchanged contract): a
    # no-room caption abstains -> empty. 3-tuple anchors with a None embedding too.
    anchors = [([1.0, 0.0, 2.0], "a wooden chair under a table", _emb(1))]
    assert _propose(_mk_bridge(), "chair", anchors) == []                   # no classifier
    anchors_none = [([1.0, 0.0, 2.0], "a cozy living room", None)]          # 3-tuple, None emb
    out = _propose(_mk_bridge(), "chair", anchors_none, room_classifier=_clf({1: "bedroom"}))
    assert len(out) == 1 and out[0].metadata["preferred_room"] == "living_room", out  # caption used
    print("  case no_classifier_is_caption_only_backcompat: OK")


def case_diag_clip_tag_recorded():
    # SA-1/2/4: a CLIP-tagged match populates _last_coarse_diag with the clip count,
    # the room histogram, the match count, and the grounding kind.
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    anchors = [([1.0, 0.0, 2.0], "a wooden chair under a table", _emb(1))]  # caption abstains
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", room_anchors=anchors, room_classifier=_clf({1: "living_room"}))
    d = b._last_coarse_diag
    assert len(out) == 1
    assert d["pref"] == "living_room", d
    assert d["n_tagged_clip"] == 1 and d["n_tagged_caption"] == 0, d
    assert d["n_abstained"] == 0, d
    assert d["room_hist"].get("living_room") == 1, d
    assert d["n_room_match"] == 1, d
    assert d["grounding"] == "stm", d   # no frontier_cands -> stm grounding
    print("  case diag_clip_tag_recorded: OK")


def case_diag_caption_and_abstain_recorded():
    # SA-2: caption-fallback tag + an abstaining anchor are counted distinctly.
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    anchors = [
        ([1.0, 0.0, 2.0], "a cozy living room with a sofa", _emb(9)),  # CLIP abstains(9->None)->caption
        ([5.0, 0.0, 5.0], "a wooden chair under a table", _emb(9)),    # CLIP abstains + caption abstains
    ]
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", room_anchors=anchors, room_classifier=_clf({}))
    d = b._last_coarse_diag
    assert len(out) == 1, out
    assert d["n_tagged_clip"] == 0, d
    assert d["n_tagged_caption"] == 1, d
    assert d["n_abstained"] == 1, d
    assert d["n_anchors"] == 2, d
    print("  case diag_caption_and_abstain_recorded: OK")


def case_diag_stashed_on_abort():
    # SA-1: a propose that returns [] (no room matches the goal) STILL stashes the
    # diag so abstain/zero-fire stats are recoverable (the prior runs could not tell
    # 'never proposed' from 'proposed-but-not-chosen').
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    anchors = [([5.0, 0.0, 5.0], "a spacious bedroom with a bed", _emb(9))]  # chair wants living
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", room_anchors=anchors, room_classifier=_clf({}))
    d = b._last_coarse_diag
    assert out == [], out
    assert d["n_anchors"] == 1, d
    assert d["n_tagged_caption"] == 1, d          # bedroom tagged (just not a match)
    assert d["n_room_match"] == 0, d              # but no anchor matched living_room
    assert d["room_hist"].get("bedroom") == 1, d
    print("  case diag_stashed_on_abort: OK")


def case_diag_top_cos_from_cos_fn():
    # SA-3: when a room_cos_fn is supplied the diag records the running max top cosine.
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    anchors = [([1.0, 0.0, 2.0], "a wooden chair under a table", _emb(1))]
    b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", room_anchors=anchors, room_classifier=_clf({1: "living_room"}),
        room_cos_fn=lambda emb: 0.37)
    d = b._last_coarse_diag
    assert abs(d["top_cos_max"] - 0.37) < 1e-6, d
    print("  case diag_top_cos_from_cos_fn: OK")


def case_stm_default_uses_visual_embedding_for_clip():
    # default anchors from _pending must carry each record's visual_embedding so
    # the CLIP classifier can tag a no-room-caption keyframe.
    from types import SimpleNamespace
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    kf = SimpleNamespace(
        step_idx=0, caption="a wooden chair under a table",   # caption abstains
        text_embedding=np.zeros(8, dtype=np.float32),
        visual_embedding=_emb(1), agent_position=np.array([3.0, 0.0, 0.0], dtype=np.float32),
        agent_yaw=0.0)
    b.observe_keyframe(kf, action=1, reward=0.0)
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", room_classifier=_clf({1: "living_room"}))
    assert len(out) == 1 and list(out[0].world_xy) == [3.0, 0.0], out
    print("  case stm_default_uses_visual_embedding_for_clip: OK")


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


def case_frontier_grounding_steers_to_affordant_region():
    # FRONTIER-grounding (the real exploration value): each unexplored frontier is
    # room-tagged by its NEAREST captioned STM keyframe. For chair (-> living_room),
    # the frontier near the living-room keyframe is emitted; the one near the bedroom
    # keyframe is not. The emitted waypoint is the FRONTIER xy (unexplored), not the
    # visited keyframe.
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    stm = [([3.0, 0.0, 0.0], "a cozy living room with a sofa"),     # tags nearby frontier living
           ([19.0, 0.0, 0.0], "a spacious bedroom with a bed")]      # tags nearby frontier bedroom
    frontier = [
        FrontierCandidate(candidate_id=1, world_xy=np.array([4.0, 0.0], dtype=np.float32),
                          grid_rc=(-1, -1), distance_m=4.0, bearing_rad=0.0, cluster_size=3,
                          raw_score=0.8, source="frontier", metadata={}),   # near living kf
        FrontierCandidate(candidate_id=2, world_xy=np.array([20.0, 0.0], dtype=np.float32),
                          grid_rc=(-1, -1), distance_m=20.0, bearing_rad=0.0, cluster_size=3,
                          raw_score=0.8, source="frontier", metadata={}),   # near bedroom kf
    ]
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", frontier_cands=frontier, room_anchors=stm)
    assert len(out) == 1, out
    assert out[0].source == "coarse", out[0].source
    assert list(out[0].world_xy) == [4.0, 0.0], out[0].world_xy   # the living-tagged FRONTIER
    assert out[0].metadata.get("grounded") == "frontier", out[0].metadata
    print("  case frontier_grounding_steers_to_affordant_region: OK")


def case_frontier_grounding_survives_planner_dedup():
    # REGRESSION (clip1 RACE run, 2026-06-09): the real call site passes
    # planner_world_xys = ALL candidate xys INCLUDING the frontier candidates
    # (episode_runner.py: planner_world_xys=[c.world_xy for c in all_cands]). A
    # frontier-grounded coarse target sits EXACTLY on a frontier's xy, so deduping
    # it against the planner pool removed it at distance 0 -> 100% of the time ->
    # the head emitted 0 candidates DESPITE a room match (n_coarse_room_matched>0,
    # n_coarse_candidates=0 in the run). A frontier-grounded coarse candidate is
    # MEANT to ride/boost an existing frontier, so it must SURVIVE the planner dedup.
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    stm = [([3.0, 0.0, 0.0], "a cozy living room with a sofa")]
    frontier = [FrontierCandidate(candidate_id=1, world_xy=np.array([4.0, 0.0], dtype=np.float32),
                                  grid_rc=(-1, -1), distance_m=4.0, bearing_rad=0.0, cluster_size=3,
                                  raw_score=0.8, source="frontier", metadata={})]
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", frontier_cands=frontier, room_anchors=stm,
        planner_world_xys=[np.array([4.0, 0.0], dtype=np.float32)])  # the frontier IS in the pool
    assert len(out) == 1, ("frontier-grounded coarse must survive planner dedup", out)
    assert list(out[0].world_xy) == [4.0, 0.0], out[0].world_xy
    assert out[0].metadata.get("grounded") == "frontier", out[0].metadata
    # diag must still record the match
    assert b._last_coarse_diag["n_room_match"] >= 1, b._last_coarse_diag
    print("  case frontier_grounding_survives_planner_dedup: OK")


def case_frontier_falls_back_to_stm_when_none_match():
    # no frontier's nearest room matches -> fall back to the visited affordant-room
    # anchor position (STM grounding), so the head still fires.
    b = _mk_bridge()
    b.begin_episode("ep-coarse", scene_id="SCENE_B")
    stm = [([3.0, 0.0, 0.0], "a cozy living room with a sofa")]
    frontier = [FrontierCandidate(candidate_id=1, world_xy=np.array([50.0, 0.0], dtype=np.float32),
                                  grid_rc=(-1, -1), distance_m=50.0, bearing_rad=0.0, cluster_size=3,
                                  raw_score=0.8, source="frontier", metadata={})]  # nearest=living, but...
    # the lone frontier IS nearest to the living anchor, so it DOES match -> frontier path.
    # Force a no-frontier-match by giving a frontier whose nearest tagged anchor is bedroom:
    stm2 = [([3.0, 0.0, 0.0], "a cozy living room"), ([49.0, 0.0, 0.0], "a spacious bedroom")]
    out = b.propose_coarse_candidates(
        agent_pos=np.array([0.0, 0.0, 0.0], dtype=np.float32), agent_yaw=0.0,
        target_category="chair", frontier_cands=frontier, room_anchors=stm2)
    # frontier [50,0] nearest anchor is bedroom[49] -> no frontier match -> STM fallback to living[3,0]
    assert len(out) == 1 and list(out[0].world_xy) == [3.0, 0.0], out
    assert out[0].metadata.get("grounded") == "stm", out[0].metadata
    print("  case frontier_falls_back_to_stm_when_none_match: OK")


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
    case_frontier_grounding_steers_to_affordant_region()
    case_frontier_grounding_survives_planner_dedup()
    case_frontier_falls_back_to_stm_when_none_match()
    case_clip_overrides_sparse_caption()
    case_clip_abstain_falls_back_to_caption()
    case_clip_confident_label_overrides_caption_keyword()
    case_no_classifier_is_caption_only_backcompat()
    case_diag_clip_tag_recorded()
    case_diag_caption_and_abstain_recorded()
    case_diag_stashed_on_abort()
    case_diag_top_cos_from_cos_fn()
    case_stm_default_uses_visual_embedding_for_clip()
    case_scorer_coarse_branch_drops_bearing()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
