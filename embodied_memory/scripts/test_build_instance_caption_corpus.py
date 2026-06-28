"""
Sanity test for build_instance_caption_corpus pure logic (no habitat / no
transformers): instance enumeration, view_point sampling, pose extraction, the
render-job plan, and the captioner-spec parsing.

    python embodied_memory/scripts/test_build_instance_caption_corpus.py
"""
from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import build_instance_caption_corpus as B  # noqa: E402


def _vp(iou, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return {"iou": iou, "agent_state": {"position": list(pos), "rotation": list(rot)}}


def _content(goals):  # goals: {full_key: [instance dicts]}
    return {"goals_by_category": goals}


# ----------------------------------------------------------------------
# find_goal_instances — suffix match, multi-word categories
# ----------------------------------------------------------------------
def case_find_goal_instances_matches_glb_suffix():
    c = _content({
        "wcojb4TFT35.basis.glb_chair": [{"object_id": 1}, {"object_id": 2}],
        "wcojb4TFT35.basis.glb_tv_monitor": [{"object_id": 9}],
    })
    assert [i["object_id"] for i in B.find_goal_instances(c, "chair")] == [1, 2]
    # multi-word category must not collide with a shorter one
    assert [i["object_id"] for i in B.find_goal_instances(c, "tv_monitor")] == [9]
    assert B.find_goal_instances(c, "sofa") == []
    # 'monitor' must NOT match 'tv_monitor'
    assert B.find_goal_instances(c, "monitor") == []


# ----------------------------------------------------------------------
# sample_viewpoints — varied, deterministic, iou-gated
# ----------------------------------------------------------------------
def case_sample_viewpoints_count_and_determinism():
    vps = [_vp(i / 100.0, pos=(i, 0, 0)) for i in range(1, 51)]  # 50 vps, iou 0.01..0.50
    a = B.sample_viewpoints(vps, 6)
    b = B.sample_viewpoints(vps, 6)
    assert len(a) == 6
    assert [v["agent_state"]["position"] for v in a] == [v["agent_state"]["position"] for v in b]


def case_sample_viewpoints_drops_zero_iou_and_undersized():
    vps = [_vp(0.0), _vp(0.0), _vp(0.4), _vp(0.3)]  # only 2 with iou>0
    got = B.sample_viewpoints(vps, 6)
    assert len(got) == 2  # fewer than n -> return all the good ones
    assert all(v["iou"] > 0 for v in got)


def case_sample_viewpoints_prefers_high_iou_varied():
    vps = [_vp(i / 100.0, pos=(i, 0, 0)) for i in range(1, 51)]
    got = B.sample_viewpoints(vps, 4)
    iarr = [v["iou"] for v in got]
    assert iarr[0] == max(v["iou"] for v in vps)  # best-visibility first
    assert len(set(iarr)) == len(iarr)            # varied, not duplicates


# ----------------------------------------------------------------------
# viewpoint_pose
# ----------------------------------------------------------------------
def case_viewpoint_pose_scalar_last_quaternion():
    pos, rot = B.viewpoint_pose(_vp(0.5, pos=(-1.36, 2.79, -10.4), rot=(0.0, -0.012, 0.0, 0.9999)))
    assert pos == [-1.36, 2.79, -10.4]
    assert rot == [0.0, -0.012, 0.0, 0.9999]  # [x,y,z,w], w last


# ----------------------------------------------------------------------
# plan_corpus — only >=2-instance categories, every job has a pose
# ----------------------------------------------------------------------
def case_plan_corpus_skips_singleton_categories():
    c = _content({
        "s.basis.glb_chair": [
            {"object_id": 1, "view_points": [_vp(0.4), _vp(0.3)]},
            {"object_id": 2, "view_points": [_vp(0.5), _vp(0.2)]},
        ],
        "s.basis.glb_bed": [{"object_id": 7, "view_points": [_vp(0.6), _vp(0.5)]}],  # 1 inst -> skip
    })
    jobs = B.plan_corpus(c, "s", ["chair", "bed", "sofa"], n_viewpoints=2)
    cats = {j["category"] for j in jobs}
    assert cats == {"chair"}, cats
    assert {j["object_id"] for j in jobs} == {1, 2}
    assert all("position" in j and "rotation" in j for j in jobs)
    assert len(jobs) == 2 * 2  # 2 instances x 2 viewpoints


def case_plan_corpus_caps_viewpoints_per_instance():
    c = _content({
        "s.basis.glb_chair": [
            {"object_id": 1, "view_points": [_vp(i / 100.0) for i in range(1, 31)]},
            {"object_id": 2, "view_points": [_vp(i / 100.0) for i in range(1, 31)]},
        ],
    })
    jobs = B.plan_corpus(c, "s", ["chair"], n_viewpoints=5)
    assert len(jobs) == 2 * 5
    assert all(j["viewpoint_idx"] < 5 for j in jobs)


# ----------------------------------------------------------------------
# parse_captioners
# ----------------------------------------------------------------------
def case_parse_captioners_default_and_pairs():
    d = B.parse_captioners([])
    assert d["qwen2-vl-2b"] == "Qwen/Qwen2-VL-2B-Instruct" and d["caprl-3b"] == "internlm/CapRL-3B"
    p = B.parse_captioners(["a=org/A", "b=org/B"])
    assert p == {"a": "org/A", "b": "org/B"}
    try:
        B.parse_captioners(["noequals"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on a malformed --captioners entry")


# ----------------------------------------------------------------------
# real val_mini content (if present) — enumeration sanity end-to-end
# ----------------------------------------------------------------------
def case_real_valmini_enumerates_chair_instances():
    path = "data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/wcojb4TFT35.json.gz"
    if not os.path.isfile(path):
        print("  case real_valmini_enumerates_chair_instances: SKIP (no data)")
        return
    c = B.load_content(path)
    insts = B.find_goal_instances(c, "chair")
    assert len(insts) >= 2, len(insts)
    jobs = B.plan_corpus(c, "wcojb4TFT35", ["chair"], n_viewpoints=4)
    assert len(jobs) == len(insts) * 4
    assert all(len(j["rotation"]) == 4 and len(j["position"]) == 3 for j in jobs)
    print(f"  case real_valmini_enumerates_chair_instances: OK ({len(insts)} chair instances)")


def main() -> int:
    cases = [
        case_find_goal_instances_matches_glb_suffix,
        case_sample_viewpoints_count_and_determinism,
        case_sample_viewpoints_drops_zero_iou_and_undersized,
        case_sample_viewpoints_prefers_high_iou_varied,
        case_viewpoint_pose_scalar_last_quaternion,
        case_plan_corpus_skips_singleton_categories,
        case_plan_corpus_caps_viewpoints_per_instance,
        case_parse_captioners_default_and_pairs,
        case_real_valmini_enumerates_chair_instances,
    ]
    print(f"running {len(cases)} build_instance_caption_corpus cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
