"""
TDD for diagnose_goal_anchored_recall — the free offline re-score (#2) that fixes
the two anchor bugs in diagnose_audio_doa_calib:

  (a) it measures recall presence to the nearest COLD-SIGHTED-instance VIEW_POINT
      (a stored memory candidate IS a caption-time viewing pose), not to the
      object CENTER of an arbitrary goals[0];
  (b) it re-keys "the correct instance" to pick_cold_instance, not goals[0]
      (correcting the multi-instance lower bound), and
  (c) reports the view_point->object-center offset distribution (binary-SPL@0.1m
      storage headroom).

Pure stdlib; tested with synthetic content + episode dicts (the real m3q logs +
content live on RACE — only the RUN needs them).

    python embodied_memory/scripts/test_diagnose_goal_anchored_recall.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose_goal_anchored_recall as gr  # noqa: E402


def _vp(pos, iou):
    return {"agent_state": {"position": list(pos), "rotation": [0, 0, 0, 1]}, "iou": iou}


def _inst(object_id, center, vps):
    return {"object_id": object_id, "position": list(center), "view_points": list(vps)}


def _content(scene, category, instances):
    return {"goals_by_category": {f"{scene}.basis.glb_{category}": instances}, "episodes": []}


def _mem(world_xy):
    return {"id": 1, "world_xy": list(world_xy), "source": "memory", "raw_score": 0.4,
            "distance_m": 1.0, "bearing_rad": 0.0, "cluster_size": 1}


def _decision(mem_world_xys):
    return {"step_idx": 0,
            "candidates": [_mem(xy) for xy in mem_world_xys]
            + [{"id": 99, "world_xy": [9.0, 9.0], "source": "frontier"}]}


def _ep(scene, category, decisions, target_position):
    # target_position = the logged goals[0] center (the LEGACY anchor we correct away from)
    return {"scene_id": scene, "target_category": category, "decisions": decisions,
            "steps": [], "target_position": list(target_position), "start_position": None}


# Canonical fixture: cold instance A (high iou) is FAR from goals[0]=B (list-order 0).
#   A: center (10,0,10), one view_point at (10,0,11.1) -> vp 1.1m from A's center
#   B: center (0,0,0)   (this is goals[0] / the logged target_position)
def _fixture_content():
    return _content("S", "chair", [
        _inst("B", (0, 0, 0), [_vp((0, 0, 1.0), iou=0.3)]),     # list-order 0 -> goals[0]
        _inst("A", (10, 0, 10), [_vp((10, 0, 11.1), iou=0.9)]),  # highest iou -> cold instance
    ])


def case_cold_instance_is_highest_iou_not_list0():
    c = _fixture_content()
    cold = gr.cold_instance(c, "chair")
    assert cold["object_id"] == "A", cold       # NOT list-order-0 (B)
    assert gr.instance_center(cold) == [10, 0, 10], gr.instance_center(cold)
    assert gr.instance_view_points(cold) == [[10, 0, 11.1]], gr.instance_view_points(cold)
    print("  case_cold_instance_is_highest_iou_not_list0: OK")


def case_nearest_instance_crosscheck_pathB():
    c = _fixture_content()
    # a pose near A's view_point resolves to A (Path B agrees with Path A)
    assert gr.nearest_instance(c, "chair", [10, 0, 11.0])["object_id"] == "A"
    assert gr.nearest_instance(c, "chair", [0, 0, 1.0])["object_id"] == "B"
    print("  case_nearest_instance_crosscheck_pathB: OK")


def case_view_point_anchor_vs_center_offset():
    # candidate sits AT A's view_point (10,0,11.1): ~0 from the vp, 1.1m from A's center.
    c = _fixture_content()
    ep = _ep("S", "chair", [_decision([(10.0, 11.1)])], target_position=[0, 0, 0])
    a = gr.analyze_episode(ep, c)
    assert a["fire_decisions"] == 1, a
    # present@view_point: True even at the tight 0.5m ring
    assert a["present_vp"][0.5] == 1, a["present_vp"]
    # present@center(cold A): False at 0.5m (offset 1.1m), True at 1.5m  -> the artifact
    assert a["present_center"][0.5] == 0, a["present_center"]
    assert a["present_center"][1.5] == 1, a["present_center"]
    print("  case_view_point_anchor_vs_center_offset: OK")


def case_rekey_flips_goals0_false_positive():
    # candidate near goals[0]=B (0,0,1) but FAR from the cold instance A.
    # LEGACY anchor (goals[0]=B) would call it present; the cold-instance re-key must NOT.
    c = _fixture_content()
    ep = _ep("S", "chair", [_decision([(0.0, 1.0)])], target_position=[0, 0, 0])
    a = gr.analyze_episode(ep, c)
    assert a["present_legacy"][1.0] == 1, a["present_legacy"]   # near goals[0] -> legacy present
    assert a["present_vp"][1.0] == 0, a["present_vp"]           # but NOT near the cold instance
    assert a["present_center"][1.0] == 0, a["present_center"]
    print("  case_rekey_flips_goals0_false_positive: OK")


def case_offset_distribution_storage_headroom():
    c = _fixture_content()
    offs = gr.offset_vp_to_center(gr.cold_instance(c, "chair"))
    assert len(offs) == 1 and abs(offs[0] - 1.1) < 1e-6, offs   # single vp, 1.1m off center
    stats = gr.offset_stats([gr.cold_instance(c, "chair")])
    assert abs(stats["median"] - 1.1) < 1e-6, stats
    # 1.1m median >> 0.1m ring -> storage headroom NONE at 0.1m
    assert stats["headroom_0p1"] is False, stats
    print("  case_offset_distribution_storage_headroom: OK")


def case_verdict_anchor_artifact():
    # legacy(goals[0]) presence LOW but cold-instance view_point presence HIGH
    # -> the old RECALL-GAP was an ANCHOR-ARTIFACT.
    agg = {"fire_decisions": 10,
           "present_vp": {1.0: 8}, "present_center": {1.0: 4}, "present_legacy": {1.0: 1}}
    v, _ = gr.recommend(agg)
    assert v == "ANCHOR-ARTIFACT", v
    print("  case_verdict_anchor_artifact: OK")


def case_verdict_recall_gap_confirmed():
    # low presence even at the correct view_point anchor -> genuine gap
    agg = {"fire_decisions": 10,
           "present_vp": {1.0: 2}, "present_center": {1.0: 1}, "present_legacy": {1.0: 1}}
    v, _ = gr.recommend(agg)
    assert v == "RECALL-GAP-CONFIRMED", v
    print("  case_verdict_recall_gap_confirmed: OK")


def case_verdict_recall_ok():
    # high presence at view_point AND legacy already high -> no anchor issue, recall fine
    agg = {"fire_decisions": 10,
           "present_vp": {1.0: 9}, "present_center": {1.0: 8}, "present_legacy": {1.0: 8}}
    v, _ = gr.recommend(agg)
    assert v == "GOAL-ANCHORED-RECALL-OK", v
    print("  case_verdict_recall_ok: OK")


def case_insufficient_data():
    v, _ = gr.recommend({"fire_decisions": 0, "present_vp": {}, "present_center": {},
                         "present_legacy": {}})
    assert v == "INSUFFICIENT-DATA", v
    print("  case_insufficient_data: OK")


def main() -> int:
    print("diagnose_goal_anchored_recall tests")
    case_cold_instance_is_highest_iou_not_list0()
    case_nearest_instance_crosscheck_pathB()
    case_view_point_anchor_vs_center_offset()
    case_rekey_flips_goals0_false_positive()
    case_offset_distribution_storage_headroom()
    case_verdict_anchor_artifact()
    case_verdict_recall_gap_confirmed()
    case_verdict_recall_ok()
    case_insufficient_data()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
