"""
Sanity test for ``make_revisit_smoke`` — the Phase-B1 controlled-start
revisit dataset builder.

The builder emits a tiny HM3D-ObjectNav dataset (no Habitat / sim needed) in
which, per category, episode 0 is a "cold" visit that starts **at a high-iou
goal view_point** (so the agent provably captions the goal and deposits it in
the LTM) and episodes 1..N are "warm" revisits that start **far from every
goal** (so reaching the goal benefits from recalling the cold sighting). The
cold episode is ordered first so its LTM entry persists to the warm visits
when run in a single process.

It reuses the source scene's ``goals_by_category`` (valid view_points →
success still computes) and clones a real episode as the template (valid
``goals`` / ``info`` / ``scene_id``), overriding only the start pose +
episode_id. This test exercises the pure builders on synthetic dicts plus a
gzip round-trip.

Invoke with::

    python embodied_memory/scripts/test_make_revisit_smoke.py
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_revisit_smoke as mk  # noqa: E402


def _vp(pos, rot=(0.0, 0.0, 0.0, 1.0), iou=1.0):
    return {"agent_state": {"position": list(pos), "rotation": list(rot)}, "iou": iou}


def _goal(pos, vps):
    return {"position": list(pos), "view_points": vps}


def _template(cat="chair", eid="100"):
    return {
        "episode_id": eid,
        "scene_id": "hm3d/val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb",
        "object_category": cat,
        "start_position": [9.0, 0.0, 9.0],
        "start_rotation": [0.0, 1.0, 0.0, 0.0],
        "goals": [{"position": [1.0, 0.0, 1.0]}],
        "info": {"geodesic_distance": 5.0},
        "shortest_paths": [],
    }


# ----------------------------------------------------------------------
# pick_cold_pose — highest-iou view_point across instances
# ----------------------------------------------------------------------


def case_cold_pose_picks_max_iou_viewpoint():
    goals = [
        _goal([0, 0, 0], [_vp([1, 0, 1], iou=0.3), _vp([2, 0, 2], iou=0.9)]),
        _goal([5, 0, 5], [_vp([4, 0, 4], iou=1.7)]),  # best
    ]
    pose = mk.pick_cold_pose(goals)
    assert pose["position"] == [4, 0, 4], pose
    print("  case cold_pose_picks_max_iou_viewpoint: OK")


def case_cold_pose_carries_rotation():
    goals = [_goal([0, 0, 0], [_vp([1, 0, 1], rot=[0, 0.7, 0, 0.7], iou=2.0)])]
    pose = mk.pick_cold_pose(goals)
    assert pose["rotation"] == [0, 0.7, 0, 0.7], pose
    print("  case cold_pose_carries_rotation: OK")


# ----------------------------------------------------------------------
# pick_warm_poses — farthest navigable candidates from the goals
# ----------------------------------------------------------------------


def case_warm_poses_ranked_far_from_goals():
    # goal view_points cluster near origin
    goal_vp_positions = [[0, 0, 0], [1, 0, 0]]
    candidates = [
        {"position": [0.5, 0, 0], "rotation": [0, 0, 0, 1]},   # near goal
        {"position": [10, 0, 0], "rotation": [0, 0, 0, 1]},    # far (best)
        {"position": [3, 0, 0], "rotation": [0, 0, 0, 1]},     # mid
    ]
    warm = mk.pick_warm_poses(candidates, goal_vp_positions, n=2, min_dist=2.0)
    assert len(warm) == 2, warm
    assert warm[0]["position"] == [10, 0, 0], warm
    assert warm[1]["position"] == [3, 0, 0], warm
    print("  case warm_poses_ranked_far_from_goals: OK")


def case_warm_poses_drops_too_close():
    goal_vp_positions = [[0, 0, 0]]
    candidates = [
        {"position": [0.5, 0, 0], "rotation": [0, 0, 0, 1]},  # < min_dist
        {"position": [9, 0, 0], "rotation": [0, 0, 0, 1]},
    ]
    warm = mk.pick_warm_poses(candidates, goal_vp_positions, n=5, min_dist=2.0)
    assert len(warm) == 1 and warm[0]["position"] == [9, 0, 0], warm
    print("  case warm_poses_drops_too_close: OK")


# ----------------------------------------------------------------------
# build_category_episodes — clone + override + order cold first
# ----------------------------------------------------------------------


def case_build_category_episodes_cold_first():
    tmpl = _template("chair")
    cold = {"position": [4, 0, 4], "rotation": [0, 0.7, 0, 0.7]}
    warm = [{"position": [10, 0, 0], "rotation": [0, 0, 0, 1]},
            {"position": [12, 0, 1], "rotation": [0, 0, 0, 1]}]
    eps = mk.build_category_episodes(tmpl, cold, warm, "chair")
    assert len(eps) == 3, len(eps)
    # cold first, start at the cold pose, goals preserved from the template
    assert eps[0]["start_position"] == [4, 0, 4]
    assert eps[0]["start_rotation"] == [0, 0.7, 0, 0.7]
    assert eps[0]["goals"] == tmpl["goals"]
    assert eps[0]["object_category"] == "chair"
    # warm visits follow, at the far poses
    assert eps[1]["start_position"] == [10, 0, 0]
    assert eps[2]["start_position"] == [12, 0, 1]
    # episode_ids are unique
    ids = [e["episode_id"] for e in eps]
    assert len(set(ids)) == 3, ids
    # builder must not mutate the template
    assert tmpl["start_position"] == [9.0, 0.0, 9.0]
    print("  case build_category_episodes_cold_first: OK")


# ----------------------------------------------------------------------
# build_dataset — assemble content dict over categories
# ----------------------------------------------------------------------


def _src_content():
    glb = "wcojb4TFT35.basis.glb"
    return {
        "category_to_task_category_id": {"chair": 0, "bed": 1},
        "category_to_scene_annotation_category_id": {"chair": 3, "bed": 4},
        "goals_by_category": {
            f"{glb}_chair": [_goal([0, 0, 0], [_vp([1, 0, 1], iou=1.5)])],
            f"{glb}_bed": [_goal([20, 0, 20], [_vp([19, 0, 19], iou=1.2)])],
        },
        "episodes": [
            {**_template("chair", "1"), "start_position": [8, 0, 8]},
            {**_template("bed", "2"), "start_position": [25, 0, 25]},
            {**_template("chair", "3"), "start_position": [15, 0, 0]},
        ],
    }


def case_build_dataset_two_categories():
    src = _src_content()
    content = mk.build_dataset(src, categories=["chair", "bed"], n_warm=1)
    # goals_by_category preserved verbatim
    assert content["goals_by_category"] == src["goals_by_category"]
    cats = [e["object_category"] for e in content["episodes"]]
    # chair block (cold+1 warm) then bed block (cold+1 warm)
    assert cats == ["chair", "chair", "bed", "bed"], cats
    # within each category the cold visit is first (its start == a goal view_point)
    chair_cold = content["episodes"][0]
    assert chair_cold["start_position"] == [1, 0, 1], chair_cold["start_position"]
    print("  case build_dataset_two_categories: OK")


def case_build_dataset_warm_starts_same_category():
    # The bed episode start [25,0,25] is Euclidean-farthest from the chair goal,
    # but a non-chair start may be navmesh-unreachable to the chair goal (other
    # island/floor) -> Infinity geodesic -> NaN soft_SPL. Warm starts must come
    # only from the SAME category's source episodes (validated reachable to a
    # goal of that category), so chair's warm start is the chair episode [15,0,0].
    src = _src_content()
    content = mk.build_dataset(src, categories=["chair"], n_warm=1)
    chair_eps = [e for e in content["episodes"] if e["object_category"] == "chair"]
    assert len(chair_eps) == 2, chair_eps           # cold + 1 warm
    warm_starts = [e["start_position"] for e in chair_eps[1:]]
    assert warm_starts == [[15, 0, 0]], warm_starts
    assert [25, 0, 25] not in warm_starts, warm_starts  # never the bed start
    print("  case build_dataset_warm_starts_same_category: OK")


def case_build_dataset_skips_missing_category():
    src = _src_content()
    content = mk.build_dataset(src, categories=["chair", "sofa"], n_warm=1)
    cats = {e["object_category"] for e in content["episodes"]}
    assert cats == {"chair"}, cats  # sofa absent → silently skipped
    print("  case build_dataset_skips_missing_category: OK")


# ----------------------------------------------------------------------
# write_dataset — gzip round-trip in habitat layout
# ----------------------------------------------------------------------


def case_write_dataset_roundtrip():
    src = _src_content()
    content = mk.build_dataset(src, categories=["chair"], n_warm=2)
    with tempfile.TemporaryDirectory() as d:
        top = mk.write_dataset(
            out_dir=d, scene="wcojb4TFT35", content=content,
            category_maps=src,
        )
        # top-level json.gz exists, has empty episodes + category maps
        assert os.path.isfile(top), top
        tj = json.load(gzip.open(top))
        assert tj["episodes"] == []
        assert "category_to_task_category_id" in tj
        # content/<scene>.json.gz exists with the built episodes + goals
        cpath = os.path.join(d, "content", "wcojb4TFT35.json.gz")
        assert os.path.isfile(cpath), cpath
        cj = json.load(gzip.open(cpath))
        assert "goals_by_category" in cj
        assert len(cj["episodes"]) == 3  # 1 cold + 2 warm
        assert cj["episodes"][0]["start_position"] == [1, 0, 1]  # cold at view_point
    print("  case write_dataset_roundtrip: OK")


def case_two_builds_into_one_dir_are_additive():
    # Phase C builds each scene into ONE shared out-dir; the per-scene
    # content/<scene>.json.gz writes must be additive (the 2nd build must not
    # clobber the 1st), and the rewritten top-level must re-load with empty
    # episodes + a category map.
    src_a = _src_content()
    src_b = _src_content()
    content_a = mk.build_dataset(src_a, categories=["chair"], n_warm=1)
    content_b = mk.build_dataset(src_b, categories=["bed"], n_warm=1)
    with tempfile.TemporaryDirectory() as d:
        mk.write_dataset(out_dir=d, scene="sceneA", content=content_a, category_maps=src_a)
        top = mk.write_dataset(out_dir=d, scene="sceneB", content=content_b, category_maps=src_b)
        assert os.path.isfile(os.path.join(d, "content", "sceneA.json.gz")), "1st build clobbered"
        assert os.path.isfile(os.path.join(d, "content", "sceneB.json.gz")), "2nd build missing"
        tj = json.load(gzip.open(top))
        assert tj["episodes"] == []
        assert "category_to_task_category_id" in tj
    print("  case two_builds_into_one_dir_are_additive: OK")


def main() -> int:
    print("Phase-B1 controlled-start dataset builder sanity tests")
    case_cold_pose_picks_max_iou_viewpoint()
    case_cold_pose_carries_rotation()
    case_warm_poses_ranked_far_from_goals()
    case_warm_poses_drops_too_close()
    case_build_category_episodes_cold_first()
    case_build_dataset_two_categories()
    case_build_dataset_warm_starts_same_category()
    case_build_dataset_skips_missing_category()
    case_write_dataset_roundtrip()
    case_two_builds_into_one_dir_are_additive()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
