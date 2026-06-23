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
# pick_cold_instance — the goal INSTANCE owning the highest-iou view_point
# (instance-keyed mode needs the instance, not just the pose)
# ----------------------------------------------------------------------


def case_pick_cold_instance_returns_max_iou_instance():
    goals = [
        _goal([0, 0, 0], [_vp([1, 0, 1], iou=0.3), _vp([2, 0, 2], iou=0.9)]),
        _goal([5, 0, 5], [_vp([4, 0, 4], iou=1.7)]),  # owns the global-best vp
    ]
    inst = mk.pick_cold_instance(goals)
    assert inst["position"] == [5, 0, 5], inst
    # consistency: the cold pose is the chosen instance's best view_point
    assert mk.pick_cold_pose([inst])["position"] == [4, 0, 4]
    print("  case pick_cold_instance_returns_max_iou_instance: OK")


def case_pick_cold_instance_raises_when_no_viewpoints():
    try:
        mk.pick_cold_instance([_goal([0, 0, 0], [])])
    except ValueError:
        print("  case pick_cold_instance_raises_when_no_viewpoints: OK")
        return
    raise AssertionError("expected ValueError on empty view_points")


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
# build_dataset(instance_keyed=True) — restrict the goal set to the single
# cold-sighted (highest-iou) instance so reaching a DIFFERENT same-category
# instance no longer counts; this gives instance-discrimination a metric target.
# ----------------------------------------------------------------------


def _multi_instance_src(glb="wcojb4TFT35.basis.glb"):
    """One category ("chair") with TWO instances: a high-iou TARGET near the
    origin and a low-iou DISTRACTOR near [20,0,20]. The category's source
    episode starts sit one near each instance, so the two build modes pick
    different warm starts (category-level drops the start near the distractor;
    instance-keyed keeps it because it is far from the TARGET)."""
    return {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {
            f"{glb}_chair": [
                _goal([0, 0, 0], [_vp([1, 0, 1], iou=1.8)]),      # TARGET (best vp)
                _goal([20, 0, 20], [_vp([19, 0, 19], iou=0.5)]),  # DISTRACTOR
            ],
        },
        "episodes": [
            {**_template("chair", "1"), "start_position": [19, 0, 19]},  # far from target
            {**_template("chair", "2"), "start_position": [1, 0, 1]},    # on the target vp
        ],
    }


def case_build_dataset_default_preserves_all_instances():
    # Regression guard: default (instance_keyed=False) leaves goals untouched.
    src = _multi_instance_src()
    glb = "wcojb4TFT35.basis.glb"
    content = mk.build_dataset(src, categories=["chair"], n_warm=1)
    assert content["goals_by_category"] == src["goals_by_category"]
    assert len(content["goals_by_category"][f"{glb}_chair"]) == 2
    print("  case build_dataset_default_preserves_all_instances: OK")


def case_build_dataset_instance_keyed_restricts_goals():
    src = _multi_instance_src()
    glb = "wcojb4TFT35.basis.glb"
    gkey = f"{glb}_chair"
    content = mk.build_dataset(src, categories=["chair"], n_warm=1, instance_keyed=True)
    goals = content["goals_by_category"][gkey]
    # goal set restricted to exactly the TARGET (max-iou) instance
    assert len(goals) == 1, goals
    assert goals[0]["position"] == [0, 0, 0], goals
    # object_category unchanged ("chair") so the "there is a chair" query is intact
    assert all(e["object_category"] == "chair" for e in content["episodes"])
    # cold episode still starts at the target instance's best view_point
    assert content["episodes"][0]["start_position"] == [1, 0, 1]
    # the builder must not mutate the source goals_by_category
    assert len(src["goals_by_category"][gkey]) == 2
    print("  case build_dataset_instance_keyed_restricts_goals: OK")


def case_build_dataset_instance_keyed_warm_far_from_target_instance():
    # Warm starts are filtered by distance to the TARGET instance's view_points
    # only — a start sitting on the DISTRACTOR (far from the target) is a valid
    # warm start, whereas the category-level build would drop it as "too close
    # to a goal" because it is on the distractor's view_point.
    src = _multi_instance_src()
    inst = mk.build_dataset(src, categories=["chair"], n_warm=2, instance_keyed=True)
    warm = [e["start_position"] for e in inst["episodes"][1:]]
    assert warm == [[19, 0, 19]], warm  # far from target, kept; [1,0,1] dropped (on target)

    # contrast: category-level drops [19,0,19] (on the distractor's view_point)
    cat = mk.build_dataset(src, categories=["chair"], n_warm=2, instance_keyed=False)
    cat_warm = [e["start_position"] for e in cat["episodes"][1:]]
    assert [19, 0, 19] not in cat_warm, cat_warm
    print("  case build_dataset_instance_keyed_warm_far_from_target_instance: OK")


def case_build_dataset_instance_keyed_roundtrip_survives_gzip():
    src = _multi_instance_src()
    glb = "wcojb4TFT35.basis.glb"
    content = mk.build_dataset(src, categories=["chair"], n_warm=1, instance_keyed=True)
    with tempfile.TemporaryDirectory() as d:
        mk.write_dataset(out_dir=d, scene="wcojb4TFT35", content=content, category_maps=src)
        cj = json.load(gzip.open(os.path.join(d, "content", "wcojb4TFT35.json.gz")))
        assert len(cj["goals_by_category"][f"{glb}_chair"]) == 1  # restriction persisted
    print("  case build_dataset_instance_keyed_roundtrip_survives_gzip: OK")


def case_build_dataset_instance_keyed_records_distractor_labels():
    # Part B: every instance-keyed episode carries offline disambiguation labels
    # (the TARGET centroid + the DISTRACTOR centroids), read by the analyzer's
    # wrong-instance-recall readout. Query stays category-level.
    src = _multi_instance_src()
    ik = mk.build_dataset(src, categories=["chair"], n_warm=1, instance_keyed=True)
    for ep in ik["episodes"]:
        lab = ep["info"]["instance_labels"]
        assert lab["target_center"] == [1.0, 0.0, 1.0], lab          # target vp centroid
        assert lab["distractor_centers"] == [[19.0, 0.0, 19.0]], lab  # the other instance
        assert "target_object_id" in lab, lab
    # category-level path adds NO labels (byte-identical regression)
    cat = mk.build_dataset(src, categories=["chair"], n_warm=1, instance_keyed=False)
    assert all("instance_labels" not in (ep.get("info") or {}) for ep in cat["episodes"])
    print("  case build_dataset_instance_keyed_records_distractor_labels: OK")


def case_pick_distractor_instances_nearest_first_capped():
    glb = "wcojb4TFT35.basis.glb"
    insts = [
        _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),     # TARGET
        _goal([2, 0, 0], [_vp([2, 0, 0], iou=0.6)]),     # nearest distractor
        _goal([9, 0, 0], [_vp([9, 0, 0], iou=0.5)]),     # mid distractor
        _goal([30, 0, 0], [_vp([30, 0, 0], iou=0.4)]),   # far distractor
    ]
    target = mk.pick_cold_instance(insts)
    assert target["position"] == [0, 0, 0], target
    # cap=2 -> the two NEAREST distractors, nearest first
    d2 = mk.pick_distractor_instances(target, insts, n=2)
    assert [d["position"] for d in d2] == [[2, 0, 0], [9, 0, 0]], d2
    # n=None -> all distractors (still nearest-first)
    all_d = mk.pick_distractor_instances(target, insts, n=None)
    assert [d["position"] for d in all_d] == [[2, 0, 0], [9, 0, 0], [30, 0, 0]], all_d
    # single-instance category -> no distractors
    assert mk.pick_distractor_instances(insts[0], [insts[0]]) == []
    print("  case pick_distractor_instances_nearest_first_capped: OK")


def case_build_category_episodes_seed_between_cold_and_warm():
    tmpl = _template("chair")
    cold = {"position": [0, 0, 0], "rotation": [0, 0, 0, 1]}
    warm = [{"position": [10, 0, 0], "rotation": [0, 0, 0, 1]}]
    seeds = [{"position": [2, 0, 0], "rotation": [0, 0, 0, 1]},
             {"position": [3, 0, 0], "rotation": [0, 0, 0, 1]}]
    eps = mk.build_category_episodes(tmpl, cold, warm, "chair", seed_poses=seeds)
    # order: cold, seed-0, seed-1, warm-1  (seeds BETWEEN cold and warm)
    assert [e["episode_id"] for e in eps] == \
        ["chair-cold-0", "chair-seed-0", "chair-seed-1", "chair-warm-1"], eps
    assert [e["start_position"] for e in eps] == \
        [[0, 0, 0], [2, 0, 0], [3, 0, 0], [10, 0, 0]]
    # only the seed episodes carry info['seed_only']=True
    assert eps[0]["info"].get("seed_only") is None
    assert eps[1]["info"]["seed_only"] is True
    assert eps[2]["info"]["seed_only"] is True
    assert eps[3]["info"].get("seed_only") is None
    # no seed_poses -> byte-identical to the old signature (regression guard)
    plain = mk.build_category_episodes(tmpl, cold, warm, "chair")
    assert [e["episode_id"] for e in plain] == ["chair-cold-0", "chair-warm-1"]
    assert all("seed_only" not in (e.get("info") or {}) for e in plain)
    print("  case build_category_episodes_seed_between_cold_and_warm: OK")


def case_build_dataset_seed_distractors_emits_seed_episodes():
    # End-to-end: instance-keyed + seed_distractors emits, per category,
    # cold-0 + N seed-{k} (seed_only=True, target-keyed goals) + warm-*.
    src = _multi_instance_src()  # chair: 1 target + 1 distractor
    glb = "wcojb4TFT35.basis.glb"
    gkey = f"{glb}_chair"
    content = mk.build_dataset(src, categories=["chair"], n_warm=1,
                               instance_keyed=True, seed_distractors=True)
    ids = [e["episode_id"] for e in content["episodes"]]
    # exactly one distractor in this fixture -> one seed episode, between cold/warm
    assert ids == ["chair-cold-0", "chair-seed-0", "chair-warm-1"], ids
    seed = next(e for e in content["episodes"] if "seed" in e["episode_id"])
    # seed starts at the DISTRACTOR view_point [19,0,19] (it gets captioned+seeded)
    assert seed["start_position"] == [19, 0, 19], seed["start_position"]
    assert seed["info"]["seed_only"] is True
    # success stays keyed to the TARGET only (a recalled distractor mis-routes)
    assert len(content["goals_by_category"][gkey]) == 1
    assert content["goals_by_category"][gkey][0]["position"] == [0, 0, 0]
    # every episode keeps instance_labels (the analyzer's wrong-instance readout)
    assert all("instance_labels" in e["info"] for e in content["episodes"])
    print("  case build_dataset_seed_distractors_emits_seed_episodes: OK")


def case_build_dataset_seed_distractors_off_is_byte_identical():
    # Default (seed_distractors=False) is byte-identical to the plain instance-keyed
    # build — the new path is fully opt-in.
    src = _multi_instance_src()
    base = mk.build_dataset(src, categories=["chair"], n_warm=2, instance_keyed=True)
    same = mk.build_dataset(src, categories=["chair"], n_warm=2, instance_keyed=True,
                            seed_distractors=False)
    assert base == same, "seed_distractors=False must not change the dataset"
    assert all("seed-" not in e["episode_id"] for e in base["episodes"])
    assert all("seed_only" not in e["info"] for e in base["episodes"])
    print("  case build_dataset_seed_distractors_off_is_byte_identical: OK")


def case_build_dataset_seed_distractors_caps_count():
    # n_distractors caps the seed-episode count (chair has many instances).
    glb = "wcojb4TFT35.basis.glb"
    src = {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {f"{glb}_chair": [
            _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),     # TARGET
            _goal([2, 0, 0], [_vp([2, 0, 0], iou=0.6)]),
            _goal([5, 0, 0], [_vp([5, 0, 0], iou=0.5)]),
            _goal([9, 0, 0], [_vp([9, 0, 0], iou=0.4)]),
        ]},
        "episodes": [{**_template("chair", "1"), "start_position": [20, 0, 0]}],
    }
    content = mk.build_dataset(src, categories=["chair"], n_warm=1,
                               instance_keyed=True, seed_distractors=True,
                               n_distractors=2)
    seeds = [e for e in content["episodes"] if "seed" in e["episode_id"]]
    assert len(seeds) == 2, seeds                       # capped at 2 of 3 distractors
    # nearest distractors seeded ([2,0,0] then [5,0,0]); farthest [9,0,0] dropped
    assert [s["start_position"] for s in seeds] == [[2, 0, 0], [5, 0, 0]], seeds
    print("  case build_dataset_seed_distractors_caps_count: OK")


def case_build_dataset_instance_keyed_warm_starts_reachability_biased():
    # With two valid (>= min_dist from target) warm candidates, the instance-keyed
    # build now orders NEAREST-to-target first (reachability bias, caveat-A fix),
    # the opposite of the farthest-first category path.
    glb = "wcojb4TFT35.basis.glb"
    src = {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {f"{glb}_chair": [
            _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),       # TARGET at origin
            _goal([30, 0, 30], [_vp([30, 0, 30], iou=0.5)]),   # distractor far away
        ]},
        "episodes": [
            {**_template("chair", "1"), "start_position": [5, 0, 5]},    # ~7.07m from target
            {**_template("chair", "2"), "start_position": [12, 0, 12]},  # ~16.97m from target
        ],
    }
    ik = mk.build_dataset(src, categories=["chair"], n_warm=2, instance_keyed=True)
    warm = [e["start_position"] for e in ik["episodes"][1:]]
    assert warm == [[5, 0, 5], [12, 0, 12]], warm   # nearest-to-target first
    print("  case build_dataset_instance_keyed_warm_starts_reachability_biased: OK")


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


# ----------------------------------------------------------------------
# build_cross_env_dataset — cold sighting in scene A, warm visit in scene B
# ----------------------------------------------------------------------


def _scene_src(glb: str, cat: str = "chair"):
    """One-category source content for scene <glb>: a goal viewpoint + 2 starts."""
    scene_id = f"hm3d/val/0000-{glb}/{glb}.basis.glb"

    def tmpl(eid, start):
        return {
            "episode_id": eid, "scene_id": scene_id, "object_category": cat,
            "start_position": list(start), "start_rotation": [0, 0, 0, 1],
            "goals": [{"position": [1.0, 0.0, 1.0]}],
            "info": {"geodesic_distance": 5.0}, "shortest_paths": [],
        }

    return {
        "category_to_task_category_id": {cat: 0},
        "category_to_scene_annotation_category_id": {cat: 3},
        "goals_by_category": {f"{glb}_{cat}": [_goal([0, 0, 0], [_vp([1, 0, 1], iou=1.5)])]},
        "episodes": [tmpl("h1", [8, 0, 8]), tmpl("h2", [15, 0, 0])],
    }


def case_cross_env_cold_home_warm_away():
    home = _scene_src("AAA")
    away = _scene_src("BBB")
    out = mk.build_cross_env_dataset(("AAA", home), ("BBB", away), ["chair"], n_warm=2)
    assert set(out.keys()) == {"AAA", "BBB"}, out.keys()

    home_eps = out["AAA"]["episodes"]
    away_eps = out["BBB"]["episodes"]
    # exactly one COLD sighting in the home scene, starting at its goal viewpoint
    assert len(home_eps) == 1, home_eps
    assert home_eps[0]["start_position"] == [1, 0, 1], home_eps[0]["start_position"]
    assert home_eps[0]["scene_id"].endswith("AAA.basis.glb"), home_eps[0]["scene_id"]
    assert "cold" in home_eps[0]["episode_id"], home_eps[0]["episode_id"]
    # two WARM revisits in the away scene, from the away scene's own starts
    assert len(away_eps) == 2, away_eps
    assert away_eps[0]["scene_id"].endswith("BBB.basis.glb"), away_eps[0]["scene_id"]
    assert all("warm" in e["episode_id"] for e in away_eps), away_eps
    # episode ids unique across the whole dataset
    ids = [e["episode_id"] for e in home_eps + away_eps]
    assert len(set(ids)) == len(ids), ids
    # each scene keeps its OWN goals_by_category (so success computes per scene)
    assert out["AAA"]["goals_by_category"] == home["goals_by_category"]
    assert out["BBB"]["goals_by_category"] == away["goals_by_category"]
    print("  case cross_env_cold_home_warm_away: OK")


def case_cross_env_warm_starts_from_away_scene():
    home = _scene_src("AAA")
    away = _scene_src("BBB")
    out = mk.build_cross_env_dataset(("AAA", home), ("BBB", away), ["chair"], n_warm=2)
    warm_starts = [e["start_position"] for e in out["BBB"]["episodes"]]
    # both away starts are far from the away goal viewpoint [1,0,1]; farthest first
    # ([15,0,0] is 14.0 m away, [8,0,8] is 9.9 m → [15,0,0] leads)
    assert warm_starts == [[15, 0, 0], [8, 0, 8]], warm_starts
    print("  case cross_env_warm_starts_from_away_scene: OK")


def case_cross_env_skips_category_absent_in_either_scene():
    # category must exist in BOTH scenes; chair only in home, bed only in away
    home = _scene_src("AAA", cat="chair")
    away = _scene_src("BBB", cat="bed")
    out = mk.build_cross_env_dataset(("AAA", home), ("BBB", away), ["chair", "bed"], n_warm=1)
    # chair: no away match; bed: no home match -> nothing buildable
    total = sum(len(c["episodes"]) for c in out.values())
    assert total == 0, out
    print("  case cross_env_skips_category_absent_in_either_scene: OK")


# ----------------------------------------------------------------------
# build_changed_world_dataset — the goal MOVES between the cold mapping pass
# and the warm visit: cold starts AT instance A (seeds it), warm success is
# keyed to a DIFFERENT instance B, so the cold sighting of A is now STALE. The
# regime the M4 temporal-context head was designed for (recency≈reliability).
# ----------------------------------------------------------------------


def _single_instance_src(glb="wcojb4TFT35.basis.glb"):
    """One category ("chair") with a SINGLE instance — changed-world needs two,
    so this category must be SKIPPED."""
    return {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {f"{glb}_chair": [_goal([0, 0, 0], [_vp([1, 0, 1], iou=1.8)])]},
        "episodes": [{**_template("chair", "1"), "start_position": [1, 0, 1]}],
    }


def _three_instance_src(glb="wcojb4TFT35.basis.glb"):
    """chair with THREE instances: cold target A (best iou) at origin, a NEAR
    other B_near, and a FAR other B_far. The moved-to goal must be B_near (a
    genuine move, but nearest → most likely same navmesh component → reachable;
    the farthest is the one that produced Infinity geodesic / NaN soft_SPL)."""
    return {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {
            f"{glb}_chair": [
                _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),     # A (cold target)
                _goal([5, 0, 5], [_vp([5, 0, 5], iou=0.6)]),     # B_near
                _goal([40, 0, 40], [_vp([40, 0, 40], iou=0.5)]),  # B_far
            ],
        },
        "episodes": [{**_template("chair", "1"), "start_position": [3, 0, 3]}],
    }


def case_pick_warm_instance_returns_the_other_instance():
    # 2-instance fixture: nearest==farthest==the only other instance, so the
    # result is unchanged ([20,0,20]) — but the semantics are now "the one valid
    # move", not "farthest".
    src = _multi_instance_src()
    insts = src["goals_by_category"]["wcojb4TFT35.basis.glb_chair"]
    cold = mk.pick_cold_instance(insts)          # TARGET at [0,0,0]
    warm = mk.pick_warm_instance(insts, cold)
    assert warm is not None and warm["position"] == [20, 0, 20], warm
    print("  case pick_warm_instance_returns_the_other_instance: OK")


def case_pick_warm_instance_prefers_nearest_genuine_move():
    # 3 instances: the moved-to goal must be the NEAREST other (B_near at [5,0,5]),
    # NOT the farthest ([40,0,40]) — reachability over distance.
    insts = _three_instance_src()["goals_by_category"]["wcojb4TFT35.basis.glb_chair"]
    cold = mk.pick_cold_instance(insts)          # A (iou 1.8) at [0,0,0]
    warm = mk.pick_warm_instance(insts, cold)
    assert warm["position"] == [5, 0, 5], warm
    print("  case pick_warm_instance_prefers_nearest_genuine_move: OK")


def case_pick_warm_instance_respects_min_move():
    # An other instance within min_move (0.5 m < 1.5) is NOT a genuine move and
    # is skipped in favour of the next-nearest above the floor.
    glb = "wcojb4TFT35.basis.glb"
    insts = [
        _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),     # A
        _goal([0.5, 0, 0], [_vp([0.5, 0, 0], iou=0.6)]),  # too close (< min_move)
        _goal([6, 0, 6], [_vp([6, 0, 6], iou=0.5)]),     # genuine move
    ]
    cold = mk.pick_cold_instance(insts)
    warm = mk.pick_warm_instance(insts, cold, min_move=1.5)
    assert warm["position"] == [6, 0, 6], warm
    print("  case pick_warm_instance_respects_min_move: OK")


def case_pick_warm_instance_falls_back_when_all_below_min_move():
    # Only one other instance and it is below min_move → still return it (never
    # silently drop a >=2-instance category); the NaN-guard is the backstop.
    insts = [
        _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),
        _goal([0.3, 0, 0], [_vp([0.3, 0, 0], iou=0.6)]),
    ]
    cold = mk.pick_cold_instance(insts)
    warm = mk.pick_warm_instance(insts, cold, min_move=1.5)
    assert warm is not None and warm["position"] == [0.3, 0, 0], warm
    print("  case pick_warm_instance_falls_back_when_all_below_min_move: OK")


def case_pick_warm_instance_none_when_single():
    insts = _single_instance_src()["goals_by_category"]["wcojb4TFT35.basis.glb_chair"]
    assert mk.pick_warm_instance(insts, insts[0]) is None
    print("  case pick_warm_instance_none_when_single: OK")


def case_changed_world_keys_goal_to_nearest_move():
    # End-to-end: the 3-instance src keys success to B_near ([5,0,5]) and the cold
    # episode still starts at A's best view_point (seeds the now-stale A).
    src = _three_instance_src()
    gkey = "wcojb4TFT35.basis.glb_chair"
    content = mk.build_changed_world_dataset(src, categories=["chair"], n_warm=1)
    goals = content["goals_by_category"][gkey]
    assert len(goals) == 1 and goals[0]["position"] == [5, 0, 5], goals
    assert content["episodes"][0]["start_position"] == [0, 0, 0], content["episodes"][0]
    print("  case changed_world_keys_goal_to_nearest_move: OK")


def case_changed_world_keys_goal_to_different_instance():
    src = _multi_instance_src()
    glb = "wcojb4TFT35.basis.glb"
    gkey = f"{glb}_chair"
    content = mk.build_changed_world_dataset(src, categories=["chair"], n_warm=1)
    goals = content["goals_by_category"][gkey]
    # success is keyed to the MOVED-TO instance B (the distractor at [20,0,20]),
    # NOT the cold-seeded A — so reaching the stale sighting no longer counts.
    assert len(goals) == 1, goals
    assert goals[0]["position"] == [20, 0, 20], goals
    # the cold episode still STARTS at instance A's best view_point, so A is
    # captioned and seeded into the (now-stale) LTM.
    assert content["episodes"][0]["start_position"] == [1, 0, 1], content["episodes"][0]
    # category unchanged → "there is a chair" query intact; source not mutated.
    assert all(e["object_category"] == "chair" for e in content["episodes"])
    assert len(src["goals_by_category"][gkey]) == 2
    print("  case changed_world_keys_goal_to_different_instance: OK")


def case_changed_world_marks_every_episode():
    src = _multi_instance_src()
    content = mk.build_changed_world_dataset(src, categories=["chair"], n_warm=2)
    assert content["episodes"], "expected cold+warm episodes"
    assert all(e.get("info", {}).get("goal_changed") is True for e in content["episodes"]), \
        [e.get("info") for e in content["episodes"]]
    print("  case changed_world_marks_every_episode: OK")


def case_changed_world_skips_single_instance_category():
    src = _single_instance_src()
    content = mk.build_changed_world_dataset(src, categories=["chair"], n_warm=1)
    # only one instance → no genuine move → category skipped (no episodes).
    assert content["episodes"] == [], content["episodes"]
    print("  case changed_world_skips_single_instance_category: OK")


# ----------------------------------------------------------------------
# changed-world warm starts are REACHABILITY-biased toward B (RACE cw-2 bug):
# the farthest-first selection grabbed a start on a navmesh component
# DISCONNECTED from B (→ Infinity start→B geodesic → NaN soft_SPL on every
# wcojb chair warm episode), while the cold episode — starting at an A
# view_point — reached B fine. So warm starts must be drawn from the
# PROVEN-reachable region (A's view_points + the NEAREST-to-B source starts),
# NOT the farthest island. The regular revisit path stays farthest-first.
# ----------------------------------------------------------------------


def _disconnected_island_src(glb="wcojb4TFT35.basis.glb"):
    """Cold target A at the origin; the moved-to goal B is the NEAREST genuine
    move at [6,0,0]; there is also a 3rd instance so the category is buildable.
    The category's source episode starts are a NEAR-B reachable start [10,0,0]
    (4 m from B) and a FAR ISLAND start [100,0,100] (≈140 m from B → a different
    navmesh component → Infinity geodesic). Farthest-first grabs the island;
    reachability-biased selection must avoid it and prefer the near start and
    A's own (proven-reachable) view_point."""
    return {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {
            f"{glb}_chair": [
                _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),      # A (cold target)
                _goal([6, 0, 0], [_vp([6, 0, 0], iou=0.6)]),      # B (nearest move)
                _goal([50, 0, 50], [_vp([50, 0, 50], iou=0.5)]),  # 3rd instance
            ],
        },
        "episodes": [
            {**_template("chair", "1"), "start_position": [10, 0, 0]},      # near B (reachable)
            {**_template("chair", "2"), "start_position": [100, 0, 100]},   # far island (unreachable)
        ],
    }


def case_changed_world_warm_avoids_disconnected_island():
    # n_warm=2 → the two reachable poses (near-B source [10,0,0] at 4 m, then A's
    # own view_point [0,0,0] at 6 m) are chosen; the farthest island [100,0,100]
    # is NEVER selected even though farthest-first would grab it first.
    src = _disconnected_island_src()
    content = mk.build_changed_world_dataset(src, categories=["chair"], n_warm=2)
    warm = [e["start_position"] for e in content["episodes"][1:]]
    assert [100, 0, 100] not in warm, warm                  # island avoided
    assert warm == [[10, 0, 0], [0, 0, 0]], warm            # near source, then A view_point
    # success is keyed to B (the nearest move), cold still starts at A.
    gkey = "wcojb4TFT35.basis.glb_chair"
    assert content["goals_by_category"][gkey][0]["position"] == [6, 0, 0]
    assert content["episodes"][0]["start_position"] == [0, 0, 0]
    print("  case changed_world_warm_avoids_disconnected_island: OK")


def case_changed_world_warm_drops_too_close_to_b():
    # A reachable pose closer than min_dist to B must be dropped (a real path to
    # B has to remain — the warm agent cannot start on top of B).
    glb = "wcojb4TFT35.basis.glb"
    src = {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {
            f"{glb}_chair": [
                _goal([0, 0, 0], [_vp([0, 0, 0], iou=1.8)]),   # A
                _goal([6, 0, 0], [_vp([6, 0, 0], iou=0.6)]),   # B (nearest move)
                _goal([50, 0, 50], [_vp([50, 0, 50], iou=0.5)]),
            ],
        },
        "episodes": [
            {**_template("chair", "1"), "start_position": [6.5, 0, 0]},  # 0.5 m from B (< min_dist)
            {**_template("chair", "2"), "start_position": [10, 0, 0]},   # 4 m from B (kept)
        ],
    }
    content = mk.build_changed_world_dataset(src, categories=["chair"], n_warm=3, min_dist=2.0)
    warm = [e["start_position"] for e in content["episodes"][1:]]
    assert [6.5, 0, 0] not in warm, warm                    # too close → dropped
    assert [10, 0, 0] in warm, warm
    assert [0, 0, 0] in warm, warm                          # A view_point (6 m, kept)
    print("  case changed_world_warm_drops_too_close_to_b: OK")


def case_pick_warm_poses_changed_world_nearest_first():
    # Unit test of the new selector: NEAREST-to-B first (NOT farthest), with the
    # proven-reachable A poses winning ties against equidistant source starts.
    goal_b = [[0, 0, 0]]
    source = [
        {"position": [20, 0, 0], "rotation": [0, 0, 0, 1]},  # far island
        {"position": [3, 0, 0], "rotation": [0, 0, 0, 1]},   # nearer
    ]
    reachable = [{"position": [5, 0, 0], "rotation": [0, 0, 0, 1]}]  # A view_point
    warm = mk.pick_warm_poses_changed_world(source, reachable, goal_b, n=3, min_dist=2.0)
    # nearest-first: source [3] (3 m), A vp [5] (5 m), source island [20] (20 m)
    assert [p["position"] for p in warm] == [[3, 0, 0], [5, 0, 0], [20, 0, 0]], warm
    print("  case pick_warm_poses_changed_world_nearest_first: OK")


def case_pick_warm_poses_changed_world_prefers_reachable_on_tie():
    # At equal distance to B, the proven-reachable A pose out-ranks the source
    # start (priority tiebreak → reachability bias).
    goal_b = [[0, 0, 0]]
    source = [{"position": [5, 0, 0], "rotation": [0, 0, 0, 1]}]
    reachable = [{"position": [0, 0, 5], "rotation": [0, 0, 0, 1]}]  # also 5 m from B
    warm = mk.pick_warm_poses_changed_world(source, reachable, goal_b, n=1, min_dist=2.0)
    assert warm[0]["position"] == [0, 0, 5], warm  # A view_point wins the tie
    print("  case pick_warm_poses_changed_world_prefers_reachable_on_tie: OK")


def case_build_dataset_regular_path_still_farthest_first():
    # Regression guard: the REGULAR revisit build (build_dataset) MUST stay
    # farthest-first and is NOT changed by the changed-world reachability fix.
    # On the same island fixture, the regular path grabs the farthest source
    # start ([100,0,100]) first — the exact behaviour the revisit eval depends on.
    src = _disconnected_island_src()
    content = mk.build_dataset(src, categories=["chair"], n_warm=2)
    warm = [e["start_position"] for e in content["episodes"][1:]]
    assert warm[0] == [100, 0, 100], warm  # farthest from all goals → first (unchanged)
    # and pick_warm_poses itself is byte-identical farthest-first
    fw = mk.pick_warm_poses(
        [{"position": [10, 0, 0], "rotation": [0, 0, 0, 1]},
         {"position": [100, 0, 100], "rotation": [0, 0, 0, 1]}],
        [[0, 0, 0]], n=2, min_dist=2.0)
    assert [p["position"] for p in fw] == [[100, 0, 100], [10, 0, 0]], fw
    print("  case build_dataset_regular_path_still_farthest_first: OK")


# ----------------------------------------------------------------------
# --seed warm-start RESAMPLE — a genuinely independent SECOND SAMPLE of the
# warm-revisit headline. The pipeline is otherwise fully deterministic; the
# ONLY independent sample is a different (seeded) n-subset of the SAME eligible
# warm-start pool. The cold pose + instance choice MUST stay deterministic so
# success-keying is unchanged across seeds. (test_* names → pytest collects.)
# ----------------------------------------------------------------------


def _seed_pool_src(glb="wcojb4TFT35.basis.glb"):
    """One category ("chair") with ONE goal instance near the origin and FIVE
    same-category source-episode starts ALL beyond min_dist from the goal — an
    eligible pool of 5 with n=3, so resampling can pick a different valid
    3-subset of the SAME pool. The goal sits at [0,0,0]; the starts are spread
    10..14 m away so none is dropped by the min_dist filter."""
    return {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {
            f"{glb}_chair": [_goal([0, 0, 0], [_vp([0, 0, 0], iou=1.5)])],
        },
        "episodes": [
            {**_template("chair", "1"), "start_position": [10, 0, 0]},
            {**_template("chair", "2"), "start_position": [11, 0, 0]},
            {**_template("chair", "3"), "start_position": [12, 0, 0]},
            {**_template("chair", "4"), "start_position": [13, 0, 0]},
            {**_template("chair", "5"), "start_position": [14, 0, 0]},
        ],
    }


def _eligible_warm_positions(src, glb="wcojb4TFT35.basis.glb", min_dist=2.0):
    """The eligible warm-start pool computed EXACTLY as the builder would (same
    category filter + min_dist filter), as a set of position tuples."""
    goal_vps = mk._goal_view_point_positions(src["goals_by_category"][f"{glb}_chair"])
    pool = set()
    for ep in src["episodes"]:
        if ep.get("object_category") != "chair":
            continue
        pos = ep["start_position"]
        d = min(mk._dist(pos, g) for g in goal_vps) if goal_vps else float("inf")
        if d >= min_dist:
            pool.add(tuple(pos))
    return pool


def test_seed_none_warm_selection_byte_identical():
    # (a) seed=None reproduces the pre-change deterministic farthest-first top-n
    # EXACTLY on a fixture with pool>n. The unit selector and the full build path
    # must both match the no-arg (legacy) call.
    src = _seed_pool_src()
    glb = "wcojb4TFT35.basis.glb"
    goal_vps = mk._goal_view_point_positions(src["goals_by_category"][f"{glb}_chair"])
    candidates = [
        {"position": ep["start_position"], "rotation": ep["start_rotation"]}
        for ep in src["episodes"]
    ]
    legacy = mk.pick_warm_poses(candidates, goal_vps, n=3, min_dist=2.0)
    seeded_none = mk.pick_warm_poses(candidates, goal_vps, n=3, min_dist=2.0, seed=None)
    assert legacy == seeded_none, (legacy, seeded_none)
    # farthest-first top-3 → [14], [13], [12]
    assert [p["position"] for p in seeded_none] == [[14, 0, 0], [13, 0, 0], [12, 0, 0]]
    # end-to-end build is byte-identical with seed absent vs seed=None
    base = mk.build_dataset(src, categories=["chair"], n_warm=3)
    same = mk.build_dataset(src, categories=["chair"], n_warm=3, seed=None)
    assert base == same, "seed=None must be byte-identical to no-seed build"
    print("  case seed_none_warm_selection_byte_identical: OK")


def test_seed_changed_world_none_byte_identical():
    # seed=None on the changed-world selector reproduces the nearest-first pick.
    goal_b = [[0, 0, 0]]
    source = [
        {"position": [20, 0, 0], "rotation": [0, 0, 0, 1]},
        {"position": [3, 0, 0], "rotation": [0, 0, 0, 1]},
    ]
    reachable = [{"position": [5, 0, 0], "rotation": [0, 0, 0, 1]}]
    legacy = mk.pick_warm_poses_changed_world(source, reachable, goal_b, n=3, min_dist=2.0)
    seeded_none = mk.pick_warm_poses_changed_world(
        source, reachable, goal_b, n=3, min_dist=2.0, seed=None)
    assert legacy == seeded_none, (legacy, seeded_none)
    assert [p["position"] for p in seeded_none] == [[3, 0, 0], [5, 0, 0], [20, 0, 0]]
    print("  case seed_changed_world_none_byte_identical: OK")


def test_seed_reproducible():
    # (b) seed=K is REPRODUCIBLE — two calls give the same warm set.
    src = _seed_pool_src()
    glb = "wcojb4TFT35.basis.glb"
    goal_vps = mk._goal_view_point_positions(src["goals_by_category"][f"{glb}_chair"])
    candidates = [
        {"position": ep["start_position"], "rotation": ep["start_rotation"]}
        for ep in src["episodes"]
    ]
    a = mk.pick_warm_poses(candidates, goal_vps, n=3, min_dist=2.0, seed=7)
    b = mk.pick_warm_poses(candidates, goal_vps, n=3, min_dist=2.0, seed=7)
    assert a == b, (a, b)
    # reproducible through the full build path too
    da = mk.build_dataset(src, categories=["chair"], n_warm=3, seed=7)
    db = mk.build_dataset(src, categories=["chair"], n_warm=3, seed=7)
    assert da == db, "seeded build must be reproducible"
    # changed-world selector reproducible too
    goal_b = [[0, 0, 0]]
    source = [{"position": [i, 0, 0], "rotation": [0, 0, 0, 1]} for i in (3, 6, 9, 12, 20)]
    reachable = [{"position": [5, 0, 0], "rotation": [0, 0, 0, 1]}]
    c = mk.pick_warm_poses_changed_world(source, reachable, goal_b, n=3, min_dist=2.0, seed=7)
    d = mk.pick_warm_poses_changed_world(source, reachable, goal_b, n=3, min_dist=2.0, seed=7)
    assert c == d, (c, d)
    print("  case seed_reproducible: OK")


def test_seed_distinct_when_pool_gt_n_equal_when_pool_le_n():
    # (c) seed=1 != seed=2 give different warm sets when pool>n; equal when
    # pool<=n (sample takes all → the same full set regardless of seed).
    src = _seed_pool_src()  # pool of 5
    glb = "wcojb4TFT35.basis.glb"
    goal_vps = mk._goal_view_point_positions(src["goals_by_category"][f"{glb}_chair"])
    candidates = [
        {"position": ep["start_position"], "rotation": ep["start_rotation"]}
        for ep in src["episodes"]
    ]
    s1 = mk.pick_warm_poses(candidates, goal_vps, n=3, min_dist=2.0, seed=1)
    s2 = mk.pick_warm_poses(candidates, goal_vps, n=3, min_dist=2.0, seed=2)
    set1 = {tuple(p["position"]) for p in s1}
    set2 = {tuple(p["position"]) for p in s2}
    assert set1 != set2, (set1, set2)  # different 3-subsets of the 5-pool
    # pool <= n → sample takes ALL → identical set across seeds (order may differ,
    # but the SET is the whole eligible pool either way)
    full_pool = _eligible_warm_positions(src)
    f1 = mk.pick_warm_poses(candidates, goal_vps, n=5, min_dist=2.0, seed=1)
    f2 = mk.pick_warm_poses(candidates, goal_vps, n=5, min_dist=2.0, seed=2)
    assert {tuple(p["position"]) for p in f1} == full_pool
    assert {tuple(p["position"]) for p in f2} == full_pool
    # changed-world: distinct seeds differ when pool>n
    goal_b = [[0, 0, 0]]
    source = [{"position": [i, 0, 0], "rotation": [0, 0, 0, 1]} for i in (3, 6, 9, 12, 20)]
    reachable = []
    cw1 = mk.pick_warm_poses_changed_world(source, reachable, goal_b, n=3, min_dist=2.0, seed=1)
    cw2 = mk.pick_warm_poses_changed_world(source, reachable, goal_b, n=3, min_dist=2.0, seed=2)
    assert {tuple(p["position"]) for p in cw1} != {tuple(p["position"]) for p in cw2}, (cw1, cw2)
    print("  case seed_distinct_when_pool_gt_n_equal_when_pool_le_n: OK")


def test_seed_resample_members_satisfy_eligibility():
    # (d) EVERY resampled warm start is a member of the eligible pool — none
    # violates the min_dist / category filter. Mix eligible + ineligible starts;
    # the resample must NEVER surface an ineligible one regardless of seed.
    glb = "wcojb4TFT35.basis.glb"
    src = {
        "category_to_task_category_id": {"chair": 0},
        "category_to_scene_annotation_category_id": {"chair": 3},
        "goals_by_category": {
            f"{glb}_chair": [_goal([0, 0, 0], [_vp([0, 0, 0], iou=1.5)])],
        },
        "episodes": [
            {**_template("chair", "1"), "start_position": [0.5, 0, 0]},  # < min_dist (drop)
            {**_template("chair", "2"), "start_position": [1.0, 0, 0]},  # < min_dist (drop)
            {**_template("chair", "3"), "start_position": [10, 0, 0]},
            {**_template("chair", "4"), "start_position": [11, 0, 0]},
            {**_template("chair", "5"), "start_position": [12, 0, 0]},
            {**_template("chair", "6"), "start_position": [13, 0, 0]},
        ],
    }
    eligible = _eligible_warm_positions(src)
    assert eligible == {(10, 0, 0), (11, 0, 0), (12, 0, 0), (13, 0, 0)}, eligible
    goal_vps = mk._goal_view_point_positions(src["goals_by_category"][f"{glb}_chair"])
    candidates = [
        {"position": ep["start_position"], "rotation": ep["start_rotation"]}
        for ep in src["episodes"]
    ]
    for sd in (1, 2, 3, 42, 1000):
        warm = mk.pick_warm_poses(candidates, goal_vps, n=3, min_dist=2.0, seed=sd)
        for p in warm:
            assert tuple(p["position"]) in eligible, (sd, p["position"])
        assert (0.5, 0, 0) not in {tuple(p["position"]) for p in warm}
        assert (1.0, 0, 0) not in {tuple(p["position"]) for p in warm}
    # through the full build path, every warm episode start is eligible too
    for sd in (1, 2, 3):
        content = mk.build_dataset(src, categories=["chair"], n_warm=3, seed=sd)
        warm_starts = [tuple(e["start_position"]) for e in content["episodes"][1:]]
        for ws in warm_starts:
            assert ws in eligible, (sd, ws)
    print("  case seed_resample_members_satisfy_eligibility: OK")


def test_seed_cold_pose_and_instance_choice_invariant():
    # (e) the cold pose AND the instance choice are IDENTICAL across
    # seed=None/1/2 — only the warm starts resample. Use the instance-keyed path
    # (which makes the instance choice explicit in goals_by_category) plus the
    # regular path's cold pose.
    src = _multi_instance_src()
    glb = "wcojb4TFT35.basis.glb"
    gkey = f"{glb}_chair"

    cold_positions = []
    keyed_goals = []
    for sd in (None, 1, 2):
        # regular path: cold pose invariant
        reg = mk.build_dataset(src, categories=["chair"], n_warm=1, seed=sd)
        cold_positions.append(reg["episodes"][0]["start_position"])
        # instance-keyed path: the chosen TARGET instance invariant
        ik = mk.build_dataset(src, categories=["chair"], n_warm=1,
                              instance_keyed=True, seed=sd)
        keyed_goals.append(ik["goals_by_category"][gkey])
    assert cold_positions[0] == cold_positions[1] == cold_positions[2], cold_positions
    assert keyed_goals[0] == keyed_goals[1] == keyed_goals[2], keyed_goals
    # and the keyed goal is the deterministic argmax-iou TARGET at [0,0,0]
    assert keyed_goals[0][0]["position"] == [0, 0, 0], keyed_goals[0]
    print("  case seed_cold_pose_and_instance_choice_invariant: OK")


def test_seed_stamped_into_dataset_provenance():
    # (4) provenance: seed is stamped into the written dataset; default None →
    # field absent (no-seed dataset byte-identical).
    src = _seed_pool_src()
    base = mk.build_dataset(src, categories=["chair"], n_warm=3)
    assert "revisit_seed" not in base, base.keys()  # default → absent
    seeded = mk.build_dataset(src, categories=["chair"], n_warm=3, seed=11)
    assert seeded.get("revisit_seed") == 11, seeded.get("revisit_seed")
    # seed=None explicitly is still absent (byte-identical to no-seed)
    none_seed = mk.build_dataset(src, categories=["chair"], n_warm=3, seed=None)
    assert "revisit_seed" not in none_seed, none_seed.keys()
    print("  case seed_stamped_into_dataset_provenance: OK")


def main() -> int:
    print("Phase-B1 controlled-start dataset builder sanity tests")
    case_cold_pose_picks_max_iou_viewpoint()
    case_cold_pose_carries_rotation()
    case_pick_cold_instance_returns_max_iou_instance()
    case_pick_cold_instance_raises_when_no_viewpoints()
    case_warm_poses_ranked_far_from_goals()
    case_warm_poses_drops_too_close()
    case_build_category_episodes_cold_first()
    case_build_dataset_two_categories()
    case_build_dataset_warm_starts_same_category()
    case_build_dataset_skips_missing_category()
    case_build_dataset_default_preserves_all_instances()
    case_build_dataset_instance_keyed_restricts_goals()
    case_build_dataset_instance_keyed_warm_far_from_target_instance()
    case_build_dataset_instance_keyed_roundtrip_survives_gzip()
    case_build_dataset_instance_keyed_records_distractor_labels()
    case_pick_distractor_instances_nearest_first_capped()
    case_build_category_episodes_seed_between_cold_and_warm()
    case_build_dataset_seed_distractors_emits_seed_episodes()
    case_build_dataset_seed_distractors_off_is_byte_identical()
    case_build_dataset_seed_distractors_caps_count()
    case_build_dataset_instance_keyed_warm_starts_reachability_biased()
    case_write_dataset_roundtrip()
    case_two_builds_into_one_dir_are_additive()
    case_cross_env_cold_home_warm_away()
    case_cross_env_warm_starts_from_away_scene()
    case_cross_env_skips_category_absent_in_either_scene()
    case_pick_warm_instance_returns_the_other_instance()
    case_pick_warm_instance_prefers_nearest_genuine_move()
    case_pick_warm_instance_respects_min_move()
    case_pick_warm_instance_falls_back_when_all_below_min_move()
    case_pick_warm_instance_none_when_single()
    case_changed_world_keys_goal_to_nearest_move()
    case_changed_world_keys_goal_to_different_instance()
    case_changed_world_marks_every_episode()
    case_changed_world_skips_single_instance_category()
    case_changed_world_warm_avoids_disconnected_island()
    case_changed_world_warm_drops_too_close_to_b()
    case_pick_warm_poses_changed_world_nearest_first()
    case_pick_warm_poses_changed_world_prefers_reachable_on_tie()
    case_build_dataset_regular_path_still_farthest_first()
    test_seed_none_warm_selection_byte_identical()
    test_seed_changed_world_none_byte_identical()
    test_seed_reproducible()
    test_seed_distinct_when_pool_gt_n_equal_when_pool_le_n()
    test_seed_resample_members_satisfy_eligibility()
    test_seed_cold_pose_and_instance_choice_invariant()
    test_seed_stamped_into_dataset_provenance()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
