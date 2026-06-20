"""
Sanity test for ``make_audiogoal_smoke`` — the M2 warm-episode dataset builder
for the AudioGoal task, a fork of ``make_revisit_smoke`` that ALSO writes the
anomaly config (class / object / source_position / per-episode t_anom) into each
episode's ``info`` dict. Pure dict manipulation — no Habitat / sim / captioner.

Cold episodes get a high ``t_anom`` (silent mapping pass); warm episodes get a
low one (the anomaly fires). The pose/source selection reuses make_revisit_smoke
verbatim; only the episode-assembly + source layer is new.

    python embodied_memory/scripts/test_make_audiogoal_smoke.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_audiogoal_smoke as mk2  # noqa: E402
import make_revisit_smoke as mk  # noqa: E402


def _vp(pos, rot=(0.0, 0.0, 0.0, 1.0), iou=1.0):
    return {"agent_state": {"position": list(pos), "rotation": list(rot)}, "iou": iou}


def _goal(pos, vps):
    return {"position": list(pos), "view_points": vps}


def _template(cat="bed", eid="100"):
    return {
        "episode_id": eid,
        "scene_id": "hm3d/val/00800-TEEsavR23oF/TEEsavR23oF.basis.glb",
        "object_category": cat,
        "start_position": [9.0, 0.0, 9.0],
        "start_rotation": [0.0, 1.0, 0.0, 0.0],
        "goals": [{"position": [1.0, 0.0, 1.0]}],
        "info": {"geodesic_distance": 5.0},
        "shortest_paths": [],
    }


def _src_content(cat="bed"):
    """Minimal source content: one category with a goal instance + a few starts."""
    goals = {f"00800-TEEsavR23oF_{cat}": [
        _goal([1.0, 0.0, 1.0], [_vp([2.0, 0.0, 2.0], iou=0.9)])]}
    eps = [
        {**_template(cat, "100"), "start_position": [9.0, 0.0, 9.0]},
        {**_template(cat, "101"), "start_position": [8.0, 0.0, 8.0]},
        {**_template(cat, "102"), "start_position": [7.5, 0.0, 7.5]},
    ]
    return {"goals_by_category": goals, "episodes": eps,
            "category_to_task_category_id": {}, "category_to_scene_annotation_category_id": {}}


def _cold_warm():
    cold = {"position": [2.0, 0.0, 2.0], "rotation": [0, 0, 0, 1]}
    warm = [{"position": [9.0, 0.0, 9.0], "rotation": [0, 0, 0, 1]},
            {"position": [8.0, 0.0, 8.0], "rotation": [0, 0, 0, 1]}]
    return cold, warm


# ---- A. reuse passthrough -------------------------------------------------

def case_reuses_revisit_pure_fns():
    assert mk2.pick_cold_pose is mk.pick_cold_pose
    assert mk2.pick_warm_poses is mk.pick_warm_poses
    assert mk2._goals_key is mk._goals_key
    print("  case reuses_revisit_pure_fns: OK")


# ---- B. build_category_episodes audio info --------------------------------

def case_episode_info_has_audio_keys():
    cold, warm = _cold_warm()
    eps = mk2.build_category_episodes(_template(), cold, warm, "bed",
                                      anomaly_class="alarm",
                                      source_position=[2.5, 0.0, 2.0])
    for ep in eps:
        info = ep["info"]
        assert info["anomaly_class"] == "alarm", info
        assert info["anomaly_object"] == "bed", info          # defaults to category
        assert info["source_position"] == [2.5, 0.0, 2.0], info
        assert "t_anom" in info
    print("  case episode_info_has_audio_keys: OK")


def case_cold_silent_warm_fires():
    cold, warm = _cold_warm()
    eps = mk2.build_category_episodes(_template(), cold, warm, "bed",
                                      anomaly_class="alarm", source_position=[0, 0, 0],
                                      t_anom_cold=10000, t_anom_warm=30)
    assert eps[0]["info"]["t_anom"] == 10000, "cold must be silent (high t_anom)"
    for ep in eps[1:]:
        assert ep["info"]["t_anom"] == 30, "warm must fire (low t_anom)"
    print("  case cold_silent_warm_fires: OK")


def case_episode_id_class_qualified():
    cold, warm = _cold_warm()
    eps = mk2.build_category_episodes(_template(), cold, warm, "bed",
                                      anomaly_class="alarm", source_position=[0, 0, 0])
    ids = [ep["episode_id"] for ep in eps]
    assert ids[0] == "bed-alarm-cold-0", ids
    assert ids[1] == "bed-alarm-warm-1" and ids[2] == "bed-alarm-warm-2", ids
    print("  case episode_id_class_qualified: OK")


def case_anomaly_object_override():
    cold, warm = _cold_warm()
    eps = mk2.build_category_episodes(_template(), cold, warm, "bed",
                                      anomaly_class="alarm", anomaly_object="sofa",
                                      source_position=[0, 0, 0])
    for ep in eps:
        assert ep["info"]["anomaly_object"] == "sofa", ep["info"]
        assert ep["object_category"] == "bed", "goal category unchanged"
    print("  case anomaly_object_override: OK")


def case_info_preserves_existing_keys():
    cold, warm = _cold_warm()
    eps = mk2.build_category_episodes(_template(), cold, warm, "bed",
                                      anomaly_class="alarm", source_position=[0, 0, 0])
    for ep in eps:
        assert ep["info"]["geodesic_distance"] == 5.0, "must not clobber existing info"
    print("  case info_preserves_existing_keys: OK")


# ---- C. back-compat (objectnav/revisit byte-identity) ---------------------

def case_no_anomaly_class_is_revisit_identical():
    src = _src_content("bed")
    out2 = mk2.build_dataset(src, ["bed"], n_warm=2, anomaly_class=None)
    out1 = mk.build_dataset(src, ["bed"], n_warm=2)
    assert out2["episodes"] == out1["episodes"], "anomaly_class=None must match make_revisit_smoke"
    for ep in out2["episodes"]:
        assert "anomaly_class" not in ep.get("info", {}), "no audio keys when class is None"
    print("  case no_anomaly_class_is_revisit_identical: OK")


# ---- D. source selection --------------------------------------------------

def case_pick_source_position_offsets_x():
    assert mk2.pick_source_position([1.0, 2.0, 3.0], offset_m=0.5) == [1.5, 2.0, 3.0]
    print("  case pick_source_position_offsets_x: OK")


def case_source_default_from_cold_pose():
    src = _src_content("bed")
    out = mk2.build_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm")
    # cold pose is the goal view_point [2,0,2]; default source = +0.5 x
    for ep in out["episodes"]:
        assert ep["info"]["source_position"] == [2.5, 0.0, 2.0], ep["info"]
    print("  case source_default_from_cold_pose: OK")


def case_source_override_applies_all():
    src = _src_content("bed")
    out = mk2.build_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm",
                            source_position=[5.0, 0.0, 6.0])
    for ep in out["episodes"]:
        assert ep["info"]["source_position"] == [5.0, 0.0, 6.0], ep["info"]
    print("  case source_override_applies_all: OK")


# ---- E. manifest ----------------------------------------------------------

def case_collect_source_manifest():
    src = _src_content("bed")
    out = mk2.build_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm")
    manifest = mk2.collect_source_manifest(out)
    assert len(manifest) == 1, f"cold+warm of one (scene,cat,class) → 1 entry, got {manifest}"
    m = manifest[0]
    assert m["anomaly_class"] == "alarm" and m["object_category"] == "bed"
    assert m["source_position"] == [2.5, 0.0, 2.0]
    assert "TEEsavR23oF" in m["scene_id"]
    print("  case collect_source_manifest: OK")


# ---- F. IO round-trip -----------------------------------------------------

def case_write_dataset_roundtrip_preserves_audio_info():
    import tempfile
    src = _src_content("bed")
    out = mk2.build_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm")
    with tempfile.TemporaryDirectory() as td:
        top = mk2.write_dataset(td, "TEEsavR23oF", out, src)
        content = mk._load_gz(os.path.join(td, "content", "TEEsavR23oF.json.gz"))
        eps = content["episodes"]
        assert eps and all(e["info"]["anomaly_class"] == "alarm" for e in eps)
        assert mk._load_gz(top)["episodes"] == [], "top json carries empty episodes"
    print("  case write_dataset_roundtrip_preserves_audio_info: OK")


def _seed(out):
    return [e for e in out["episodes"] if "-cold-" in e["episode_id"]]


def _recall(out):
    return [e for e in out["episodes"] if "-warm-" in e["episode_id"]]


def case_lifelong_polarity_inverted():
    src = _src_content()
    out = mk2.build_lifelong_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm")
    assert _seed(out)[0]["info"]["t_anom"] == mk2._T_ANOM_SEED_DEFAULT == 1, "seed must FIRE"
    for r in _recall(out):
        assert r["info"]["t_anom"] == mk2._T_ANOM_RECALL_DEFAULT == 10000, "recall must be SILENT"
    print("  case lifelong_polarity_inverted: OK")


def case_lifelong_recall_far_from_source():
    src = _src_content()
    out = mk2.build_lifelong_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm",
                                     min_dist=4.0)
    src_xyz = _recall(out)[0]["info"]["source_position"]
    for r in _recall(out):
        d = mk2._xz_dist(r["start_position"], src_xyz)
        assert d is not None and d >= 4.0, f"recall start only {d}m from source"
    print("  case lifelong_recall_far_from_source: OK")


def case_lifelong_construction_flags_redundancy_not_fail():
    # the default build seeds at the goal view_point (~0.5m from source) -> the
    # checker must WARN (redundancy-risk) but NOT FAIL.
    src = _src_content()
    out = mk2.build_lifelong_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm")
    issues = mk2.lifelong_construction_issues(out)
    assert any(i.startswith("REDUNDANCY-RISK") for i in issues), issues
    assert not any(i.startswith("FAIL") for i in issues), issues
    print("  case lifelong_construction_flags_redundancy_not_fail: OK")


def case_lifelong_construction_fails_on_close_recall():
    src = _src_content()
    out = mk2.build_lifelong_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm")
    # move a recall start ONTO the source -> must FAIL (deduped / re-mapped).
    src_xyz = _recall(out)[0]["info"]["source_position"]
    _recall(out)[0]["start_position"] = list(src_xyz)
    issues = mk2.lifelong_construction_issues(out)
    assert any(i.startswith("FAIL") and "recall" in i for i in issues), issues
    print("  case lifelong_construction_fails_on_close_recall: OK")


def case_lifelong_construction_fails_on_silent_seed():
    src = _src_content()
    # seed that does NOT fire (high t_anom) -> FAIL
    out = mk2.build_lifelong_dataset(src, ["bed"], n_warm=2, anomaly_class="alarm",
                                     t_anom_seed=10000)
    issues = mk2.lifelong_construction_issues(out)
    assert any(i.startswith("FAIL") and "seed" in i and "FIRES" in i for i in issues), issues
    print("  case lifelong_construction_fails_on_silent_seed: OK")


def main() -> int:
    cases = [
        case_reuses_revisit_pure_fns,
        case_episode_info_has_audio_keys,
        case_cold_silent_warm_fires,
        case_episode_id_class_qualified,
        case_anomaly_object_override,
        case_info_preserves_existing_keys,
        case_no_anomaly_class_is_revisit_identical,
        case_pick_source_position_offsets_x,
        case_source_default_from_cold_pose,
        case_source_override_applies_all,
        case_collect_source_manifest,
        case_write_dataset_roundtrip_preserves_audio_info,
        case_lifelong_polarity_inverted,
        case_lifelong_recall_far_from_source,
        case_lifelong_construction_flags_redundancy_not_fail,
        case_lifelong_construction_fails_on_close_recall,
        case_lifelong_construction_fails_on_silent_seed,
    ]
    print(f"running {len(cases)} make_audiogoal_smoke cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
