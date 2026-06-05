"""
Sanity test for ``make_multion_smoke`` — the MultiON (sequential semantic
ObjectNav) dataset builder.

Flavour B (semantic categories): each episode chains K existing HM3D
categories that co-occur in one scene. The episode record stays a valid
single-goal ObjectNav episode (``object_category == c1`` so Habitat's native
goals/metrics and the category filter work unchanged) and carries the full
ordered chain in ``info["object_categories"]`` — habitat's NavigationEpisode
``info`` dict tolerates arbitrary keys, and the loader already preserves it.

Pure-data tests on synthetic dicts plus a gzip round-trip (pattern of
test_make_revisit_smoke). No Habitat / sim needed.

Invoke with::

    python embodied_memory/scripts/test_make_multion_smoke.py
"""

from __future__ import annotations

import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_multion_smoke as mm  # noqa: E402
import make_revisit_smoke as mk  # noqa: E402


# ----------------------------------------------------------------------
# fixtures (mirror test_make_revisit_smoke)
# ----------------------------------------------------------------------


def _vp(pos, rot=(0.0, 0.0, 0.0, 1.0), iou=1.0):
    return {"agent_state": {"position": list(pos), "rotation": list(rot)}, "iou": iou}


def _goal(pos, vps):
    return {"position": list(pos), "view_points": vps}


def _ep(cat, eid, start=(9.0, 0.0, 9.0)):
    return {
        "episode_id": eid,
        "scene_id": "hm3d/val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb",
        "object_category": cat,
        "start_position": list(start),
        "start_rotation": [0.0, 1.0, 0.0, 0.0],
        "goals": [{"position": [1.0, 0.0, 1.0]}],
        "info": {"geodesic_distance": 5.0},
        "shortest_paths": [],
    }


def _content():
    """Scene with 3 co-occurring categories (chair, bed, toilet), one category
    with goals but NO view_points (sofa -> unusable), and one category with
    episodes but no goals entry (plant -> unusable)."""
    return {
        "category_to_task_category_id": {"chair": 0, "bed": 1, "toilet": 2},
        "category_to_scene_annotation_category_id": {},
        "goals_by_category": {
            "wcojb4TFT35.basis.glb_chair": [_goal([0, 0, 0], [_vp([1, 0, 1])])],
            "wcojb4TFT35.basis.glb_bed": [_goal([5, 0, 5], [_vp([4, 0, 4])])],
            "wcojb4TFT35.basis.glb_toilet": [_goal([8, 0, 2], [_vp([7, 0, 2])])],
            "wcojb4TFT35.basis.glb_sofa": [_goal([3, 0, 3], [])],  # no view_points
        },
        "episodes": [
            _ep("chair", "100", start=(9.0, 0.0, 9.0)),
            _ep("chair", "101", start=(12.0, 0.0, 12.0)),
            _ep("bed", "200", start=(0.5, 0.0, 8.0)),
            _ep("toilet", "300", start=(2.0, 0.0, 9.0)),
            _ep("plant", "400"),  # no goals_by_category entry
        ],
    }


# ----------------------------------------------------------------------
# co_occurring_categories
# ----------------------------------------------------------------------


def case_co_occurring_categories():
    cats = mm.co_occurring_categories(_content())
    assert cats == ["bed", "chair", "toilet"], cats  # sorted, deterministic
    # sofa (no view_points) and plant (no goals entry) must be excluded
    assert "sofa" not in cats and "plant" not in cats
    print("  case co_occurring_categories (usable cats only, sorted): OK")


# ----------------------------------------------------------------------
# sample_orderings
# ----------------------------------------------------------------------


def case_sample_orderings_len_and_distinct():
    rng = random.Random(7)
    orderings = mm.sample_orderings(["bed", "chair", "toilet"], k=3, n=4, rng=rng)
    assert 1 <= len(orderings) <= 4
    for o in orderings:
        assert len(o) == 3, o
        assert len(set(o)) == 3, f"categories must be distinct within an ordering: {o}"
        assert set(o) <= {"bed", "chair", "toilet"}, o
    # orderings themselves are pairwise distinct
    assert len({tuple(o) for o in orderings}) == len(orderings)
    print("  case sample_orderings_len_and_distinct: OK")


def case_sample_orderings_seeded_reproducible():
    a = mm.sample_orderings(["bed", "chair", "toilet"], k=3, n=3, rng=random.Random(42))
    b = mm.sample_orderings(["bed", "chair", "toilet"], k=3, n=3, rng=random.Random(42))
    assert a == b, (a, b)
    print("  case sample_orderings_seeded_reproducible: OK")


def case_sample_orderings_k_too_large_raises():
    try:
        mm.sample_orderings(["bed", "chair"], k=3, n=1, rng=random.Random(0))
    except ValueError:
        print("  case sample_orderings_k_too_large_raises: OK")
        return
    raise AssertionError("expected ValueError for k > len(categories)")


# ----------------------------------------------------------------------
# build_multion_episodes
# ----------------------------------------------------------------------


def case_build_sets_c1_and_info_chain():
    content = _content()
    orderings = [["chair", "bed", "toilet"], ["bed", "toilet", "chair"]]
    eps = mm.build_multion_episodes(content, orderings, min_dist=2.0)
    assert len(eps) == 2, len(eps)
    for ep, ordering in zip(eps, orderings):
        # native single-goal fields keyed to c1 -> Habitat metrics/filter unchanged
        assert ep["object_category"] == ordering[0], ep["object_category"]
        # the full ordered chain rides in info (preserved by the loaders)
        assert ep["info"]["object_categories"] == ordering
        # template fields survive the clone
        assert ep["scene_id"].endswith("wcojb4TFT35.basis.glb")
        assert ep["start_position"] and ep["start_rotation"]
    # episode ids unique
    assert len({ep["episode_id"] for ep in eps}) == 2
    print("  case build_sets_c1_and_info_chain: OK")


def case_build_start_not_goal_adjacent():
    content = _content()
    eps = mm.build_multion_episodes(content, [["chair", "bed", "toilet"]],
                                    min_dist=2.0)
    (ep,) = eps
    # chair view_point is at (1,0,1); the start must be >= 2.0m from it
    sx, _, sz = ep["start_position"]
    d = ((sx - 1.0) ** 2 + (sz - 1.0) ** 2) ** 0.5
    assert d >= 2.0, f"start {ep['start_position']} too close to chair view_point"
    print("  case build_start_not_goal_adjacent: OK")


def case_build_skips_unusable_c1():
    # plant has no goals entry -> ordering starting at plant is skipped
    content = _content()
    eps = mm.build_multion_episodes(content, [["plant", "chair", "bed"]],
                                    min_dist=2.0)
    assert eps == [], eps
    print("  case build_skips_unusable_c1: OK")


# ----------------------------------------------------------------------
# build_dataset + round-trip
# ----------------------------------------------------------------------


def case_dataset_preserves_goals_for_all_k():
    content = _content()
    ds = mm.build_dataset(content, k=3, n_episodes=2, seed=11, min_dist=2.0)
    assert ds["episodes"], "no episodes built"
    gbc = ds["goals_by_category"]
    for ep in ds["episodes"]:
        for cat in ep["info"]["object_categories"]:
            assert mk._goals_key(gbc, cat) is not None, \
                f"goals_by_category must cover every chained category: {cat}"
    print("  case dataset_preserves_goals_for_all_k: OK")


def case_dataset_seeded_reproducible():
    a = mm.build_dataset(_content(), k=3, n_episodes=3, seed=5, min_dist=2.0)
    b = mm.build_dataset(_content(), k=3, n_episodes=3, seed=5, min_dist=2.0)
    assert a["episodes"] == b["episodes"]
    print("  case dataset_seeded_reproducible: OK")


def case_dataset_category_filter():
    ds = mm.build_dataset(_content(), k=2, n_episodes=2, seed=1,
                          categories=["chair", "bed"], min_dist=2.0)
    for ep in ds["episodes"]:
        assert set(ep["info"]["object_categories"]) <= {"chair", "bed"}
    print("  case dataset_category_filter: OK")


def case_roundtrip_via_write_dataset():
    content = _content()
    ds = mm.build_dataset(content, k=3, n_episodes=2, seed=3, min_dist=2.0)
    with tempfile.TemporaryDirectory() as d:
        out_dir = os.path.join(d, "multion_wcojb4TFT35")
        top = mk.write_dataset(out_dir, "wcojb4TFT35", ds, content)
        re_top = mk._load_gz(top)
        assert re_top["episodes"] == [], "top-level must have empty episodes"
        cj = mk._load_gz(os.path.join(out_dir, "content", "wcojb4TFT35.json.gz"))
        assert cj["episodes"] == ds["episodes"]
        assert "goals_by_category" in cj
        # the info chain survives the gzip round-trip
        for ep in cj["episodes"]:
            assert len(ep["info"]["object_categories"]) == 3
    print("  case roundtrip_via_write_dataset: OK")


# ----------------------------------------------------------------------
# habitat_env._category_viewpoints_from_content (pure parser for the
# per-category distance seam; loaded with stubbed deps, no habitat)
# ----------------------------------------------------------------------


def _load_habitat_env_module():
    import importlib.util
    import types
    from pathlib import Path

    emb_dir = Path(__file__).resolve().parent.parent
    if "embodied_memory" not in sys.modules:
        pkg = types.ModuleType("embodied_memory")
        pkg.__path__ = [str(emb_dir)]
        sys.modules["embodied_memory"] = pkg
    src = types.ModuleType("embodied_memory.episode_source")
    for a in ("AgentState", "Episode", "EpisodeSource", "Step"):
        setattr(src, a, type(a, (), {}))
    sys.modules["embodied_memory.episode_source"] = src
    spec = importlib.util.spec_from_file_location(
        "embodied_memory.habitat_env", str(emb_dir / "habitat_env.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embodied_memory.habitat_env"] = mod
    spec.loader.exec_module(mod)
    return mod


def case_category_viewpoints_suffix_match():
    he = _load_habitat_env_module()
    content = {
        "goals_by_category": {
            "scene.basis.glb_tv_monitor": [
                _goal([0, 0, 0], [_vp([1, 0, 1]), _vp([2, 0, 2])])],
            "scene.basis.glb_monitor_stand": [_goal([9, 0, 9], [_vp([8, 0, 8])])],
            "scene.basis.glb_chair": [_goal([5, 0, 5], [_vp([4, 0, 4])])],
        },
    }
    # multi-token category resolves via suffix match (NOT rsplit on '_').
    # Suffix matching mirrors make_revisit_smoke._goals_key, which the whole
    # revisit pipeline already uses for these keys; the real HM3D category
    # set (chair/bed/sofa/toilet/plant/tv_monitor) has no suffix collisions.
    vps = he._category_viewpoints_from_content(content, "tv_monitor")
    assert vps == [[1, 0, 1], [2, 0, 2]], vps
    vps_chair = he._category_viewpoints_from_content(content, "chair")
    assert vps_chair == [[4, 0, 4]], vps_chair
    assert he._category_viewpoints_from_content(content, "plant") == []
    assert he._category_viewpoints_from_content({}, "chair") == []
    print("  case category_viewpoints_suffix_match: OK")


def main() -> int:
    print("make_multion_smoke sanity tests")
    case_co_occurring_categories()
    case_sample_orderings_len_and_distinct()
    case_sample_orderings_seeded_reproducible()
    case_sample_orderings_k_too_large_raises()
    case_build_sets_c1_and_info_chain()
    case_build_start_not_goal_adjacent()
    case_build_skips_unusable_c1()
    case_dataset_preserves_goals_for_all_k()
    case_dataset_seeded_reproducible()
    case_roundtrip_via_write_dataset()
    case_dataset_category_filter()
    case_category_viewpoints_suffix_match()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
