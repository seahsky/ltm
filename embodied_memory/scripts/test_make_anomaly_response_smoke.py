"""
TDD for make_anomaly_response_smoke — the N3 dataset builder for the anomaly-
RESPONSE task (primary find-goal + an anomaly-sound INTERRUPT from a source
DECOUPLED from the goal). The current make_audiogoal_smoke pins the source
~0.5 m from the goal view_point (source==goal, LOS) → degenerate detour + loud-
bed step-0 false-fire. N3 places the source at a DIFFERENT real object, audible
but far from the goal, so the interrupt→investigate→resume→report loop is real.

Only the PURE builder logic is unit-tested here (RIR render / navmesh geodesic /
captioner are RACE/sim-bound):
  * pick_anomaly_source — decoupled source = a real goal view_point of a
    different object, >= min_sep from the primary goal, nearest-first.
  * anomaly_response_construction_issues — the $0 gate that PERMITS
    anomaly_object != object_category (the decoupling) and FAILs a co-located
    source / wrong t_anom polarity.
  * build_dataset — decoupled stamping + the byte-identical anomaly_class=None
    delegation to make_revisit_smoke.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_make_anomaly_response_smoke.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_anomaly_response_smoke as n3  # noqa: E402
import make_revisit_smoke as mk  # noqa: E402


# ----------------------------------------------------------------------
# fixtures — HM3D-shaped goals_by_category (keys end in _<category>) +
# per-category source episodes with validated (finite-geodesic) starts.
# ----------------------------------------------------------------------
def _inst(object_id, vp_positions, iou0=0.9):
    return {
        "object_id": object_id,
        "view_points": [
            {"iou": iou0 - 0.01 * i,
             "agent_state": {"position": list(p), "rotation": [0.0, 0.0, 0.0, 1.0]}}
            for i, p in enumerate(vp_positions)
        ],
    }


def _src_content(goals, episodes):
    return {
        "category_to_task_category_id": {c: i for i, c in enumerate(
            sorted({k.split("_", 1)[1] if "_" in k else k for k in goals}))},
        "category_to_scene_annotation_category_id": {},
        "goals_by_category": goals,
        "episodes": episodes,
    }


def _ep(cat, start, gd=3.0):
    return {"object_category": cat, "scene_id": "S.basis.glb",
            "start_position": list(start), "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "info": {"geodesic_distance": gd}}


# bed at origin (primary goal), chair ~5 m away, a 2nd bed ~6 m away.
_GOALS = {
    "0_bed": [_inst("bed_0", [[0.0, 0.0, 0.0]]), _inst("bed_1", [[6.0, 0.0, 0.0]], iou0=0.5)],
    "0_chair": [_inst("chair_0", [[5.0, 0.0, 0.0]])],
}
_EPS = [_ep("bed", [1.0, 0.0, 3.0]), _ep("bed", [2.0, 0.0, 4.0]), _ep("chair", [5.0, 0.0, 3.0])]


# ----------------------------------------------------------------------
# pick_anomaly_source
# ----------------------------------------------------------------------
def case_source_prefers_other_category():
    # primary goal = bed@origin; the chair (different category, 5 m) beats the
    # 2nd bed (6 m) because a DIFFERENT category is preferred over distance.
    src = n3.pick_anomaly_source(_GOALS, ["bed", "chair"], "bed", [0.0, 0.0, 0.0], min_sep_m=3.0)
    assert src["anomaly_object"] == "chair", src
    assert src["object_id"] == "chair_0", src
    assert abs(src["position"][0] - 5.0) < 1e-6, src


def case_source_rejects_too_close():
    # raise min_sep above the chair (5 m) → chair rejected, the 2nd bed (6 m) wins.
    src = n3.pick_anomaly_source(_GOALS, ["bed", "chair"], "bed", [0.0, 0.0, 0.0], min_sep_m=5.5)
    assert src["anomaly_object"] == "bed" and src["object_id"] == "bed_1", src


def case_source_falls_back_to_other_instance_same_category():
    # only bed instances present → decouple by a DIFFERENT instance of bed.
    goals = {"0_bed": [_inst("bed_0", [[0.0, 0.0, 0.0]]), _inst("bed_1", [[6.0, 0.0, 0.0]])]}
    src = n3.pick_anomaly_source(goals, ["bed"], "bed", [0.0, 0.0, 0.0], min_sep_m=3.0)
    assert src["anomaly_object"] == "bed" and src["object_id"] == "bed_1", src


def case_source_nearest_among_qualifying_other_category():
    # two other-category objects both qualify; the NEAREST (>= min_sep) wins
    # (reachability discipline — farthest lands on disconnected islands).
    goals = {
        "0_bed": [_inst("bed_0", [[0.0, 0.0, 0.0]])],
        "0_chair": [_inst("chair_near", [[4.0, 0.0, 0.0]]), _inst("chair_far", [[9.0, 0.0, 0.0]])],
    }
    src = n3.pick_anomaly_source(goals, ["bed", "chair"], "bed", [0.0, 0.0, 0.0], min_sep_m=3.0)
    assert src["object_id"] == "chair_near", src


def case_source_uses_valid_viewpoint_when_top_iou_lacks_position():
    # an instance whose HIGHEST-iou view_point has an empty position must still be
    # usable via a lower-iou vp that HAS one (not crash / skip the whole category).
    goals = {
        "0_bed": [_inst("bed_0", [[0.0, 0.0, 0.0]])],
        "0_chair": [{"object_id": "chair_x", "view_points": [
            {"iou": 0.99, "agent_state": {"position": []}},              # top-iou, no pos
            {"iou": 0.50, "agent_state": {"position": [5.0, 0.0, 0.0]}},  # lower-iou, has pos
        ]}],
    }
    src = n3.pick_anomaly_source(goals, ["bed", "chair"], "bed", [0.0, 0.0, 0.0], min_sep_m=3.0)
    assert src["object_id"] == "chair_x" and abs(src["position"][0] - 5.0) < 1e-6, src


def case_source_raises_when_no_decoupled_candidate():
    # single object only → cannot decouple.
    goals = {"0_bed": [_inst("bed_0", [[0.0, 0.0, 0.0]])]}
    try:
        n3.pick_anomaly_source(goals, ["bed"], "bed", [0.0, 0.0, 0.0], min_sep_m=3.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError when no object >= min_sep")


# ----------------------------------------------------------------------
# anomaly_response_construction_issues
# ----------------------------------------------------------------------
def _built():
    return n3.build_dataset(_src_content(_GOALS, _EPS), ["bed"], n_warm=2,
                            anomaly_class="alarm", min_source_sep_m=3.0)


def case_issues_clean_ok():
    assert n3.anomaly_response_construction_issues(_built(), min_source_sep_m=3.0) == []


def case_issues_permit_anomaly_object_differs():
    # the decoupled build stamps anomaly_object='chair' != object_category='bed';
    # that MUST NOT be flagged (the key relaxation vs lifelong_construction_issues).
    content = _built()
    assert any((e.get("info") or {}).get("anomaly_object") != e.get("object_category")
               for e in content["episodes"]), "fixture should have a decoupled object"
    issues = n3.anomaly_response_construction_issues(content, min_source_sep_m=3.0)
    assert not any("anomaly_object" in s for s in issues), issues


def case_issues_fail_source_co_located_with_goal():
    content = _built()
    for e in content["episodes"]:
        if "-cold-" in e["episode_id"]:
            e["info"]["source_position"] = list(e["start_position"])  # source == goal vp
    issues = n3.anomaly_response_construction_issues(content, min_source_sep_m=3.0)
    assert any("co-located" in s or "< " in s for s in issues), issues


def case_issues_fail_source_not_3d():
    content = _built()
    content["episodes"][0]["info"]["source_position"] = [1.0, 2.0]
    issues = n3.anomaly_response_construction_issues(content, min_source_sep_m=3.0)
    assert any("not 3D" in s for s in issues), issues


def case_issues_fail_warm_never_fires():
    content = _built()
    for e in content["episodes"]:
        if "-warm-" in e["episode_id"]:
            e["info"]["t_anom"] = 10000   # silent → never interrupts the search
    issues = n3.anomaly_response_construction_issues(content, min_source_sep_m=3.0)
    assert any("warm" in s and "FIRE" in s.upper() for s in issues), issues


def case_issues_fail_cold_not_silent():
    content = _built()
    for e in content["episodes"]:
        if "-cold-" in e["episode_id"]:
            e["info"]["t_anom"] = 3   # fires during the mapping pass → not silent
    issues = n3.anomaly_response_construction_issues(content, min_source_sep_m=3.0)
    assert any("seed" in s and "silent" in s.lower() for s in issues), issues


# ----------------------------------------------------------------------
# build_dataset
# ----------------------------------------------------------------------
def case_build_none_class_byte_identical_to_revisit():
    src = _src_content(_GOALS, _EPS)
    got = n3.build_dataset(src, ["bed"], n_warm=2, anomaly_class=None)
    exp = mk.build_dataset(src, ["bed"], n_warm=2)
    assert got == exp, "anomaly_class=None must equal make_revisit_smoke.build_dataset"


def case_build_decoupled_stamps_source_and_object():
    content = _built()
    assert content["episodes"], "should build episodes"
    for e in content["episodes"]:
        info = e["info"]
        assert info["anomaly_class"] == "alarm"
        assert info["anomaly_object"] == "chair", info          # decoupled object
        assert e["object_category"] == "bed"                    # primary find-target unchanged
        assert len(info["source_position"]) == 3
        # M3 polarity: cold silent (high), warm fires (low)
        if "-cold-" in e["episode_id"]:
            assert info["t_anom"] > 100, info
        else:
            assert info["t_anom"] <= 100, info


def case_build_source_decoupled_from_goal():
    # the stamped source is >= min_sep from the goal view_point → passes the gate.
    content = n3.build_dataset(_src_content(_GOALS, _EPS), ["bed"], n_warm=1,
                               anomaly_class="alarm", min_source_sep_m=3.0)
    cold = next(e for e in content["episodes"] if "-cold-" in e["episode_id"])
    dx = cold["info"]["source_position"][0] - cold["start_position"][0]
    dz = cold["info"]["source_position"][2] - cold["start_position"][2]
    assert (dx * dx + dz * dz) ** 0.5 >= 3.0, cold


def main() -> int:
    cases = [
        case_source_prefers_other_category,
        case_source_rejects_too_close,
        case_source_falls_back_to_other_instance_same_category,
        case_source_nearest_among_qualifying_other_category,
        case_source_uses_valid_viewpoint_when_top_iou_lacks_position,
        case_source_raises_when_no_decoupled_candidate,
        case_issues_clean_ok,
        case_issues_permit_anomaly_object_differs,
        case_issues_fail_source_co_located_with_goal,
        case_issues_fail_source_not_3d,
        case_issues_fail_warm_never_fires,
        case_issues_fail_cold_not_silent,
        case_build_none_class_byte_identical_to_revisit,
        case_build_decoupled_stamps_source_and_object,
        case_build_source_decoupled_from_goal,
    ]
    print(f"running {len(cases)} make_anomaly_response_smoke cases…")
    for c in cases:
        c()
        print(f"  case {c.__name__}: OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
