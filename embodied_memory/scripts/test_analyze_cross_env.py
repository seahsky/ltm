"""
Sanity tests for ``analyze_cross_env`` — the cross-ENVIRONMENT transfer analyzer.

The same-scene ``analyze_revisit`` labels cold/warm by visit-order within
``(scene, category)``, which is structurally WRONG for cross-env: there the away
visit is the FIRST visit to scene B yet should count as the "query" (it has a
scene-A sighting in memory). This analyzer relabels by scene ROLE — away-scene
episodes = the query (warm), home-scene episodes = the cold source/control — and
reuses ``analyze_revisit``'s paired bootstrap. The PRIMARY evidence is the
cross-scene recall counter (does the home sighting get recalled in scene B?),
because the seam is counted-not-injected so the soft-SPL delta cannot be
attributed to cross-env transfer.

Invoke with::

    python embodied_memory/scripts/test_analyze_cross_env.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_cross_env as ce  # noqa: E402
import analyze_revisit as ar  # noqa: E402


def _ep(scene, eid, cat, idx, soft_spl=0.0, spl=0.0):
    return ar.RevisitEpisode(
        scene_id=scene, episode_id=eid, target_category=cat, episode_idx=idx,
        soft_spl=soft_spl, spl=spl, success=False, n_steps=10, min_d2g=1.0,
        success_1m=False,
    )


# ----------------------------------------------------------------------
# label_by_scene_role
# ----------------------------------------------------------------------


def case_label_marks_away_warm_home_cold():
    eps = [_ep("HOME", "0", "chair", 0), _ep("AWAY", "1", "chair", 1)]
    ce.label_by_scene_role(eps, away_scene="AWAY")
    home = next(e for e in eps if e.scene_id == "HOME")
    away = next(e for e in eps if e.scene_id == "AWAY")
    assert home.is_cold and not home.is_warm, home.visit_order
    assert away.is_warm and not away.is_cold, away.visit_order
    print("  case label_marks_away_warm_home_cold: OK")


def case_label_handles_full_scene_paths():
    # log scene_id may be a bare name while --away-scene is bare too; match on the
    # bare token even if one side is a glb path.
    eps = [_ep("hm3d/val/00800-AWAY/AWAY.basis.glb", "1", "chair", 1)]
    ce.label_by_scene_role(eps, away_scene="AWAY")
    assert eps[0].is_warm, eps[0].visit_order
    print("  case label_handles_full_scene_paths: OK")


# ----------------------------------------------------------------------
# infer_away_scene — home runs first (lower episode_idx)
# ----------------------------------------------------------------------


def case_infer_away_is_later_scene():
    eps = [_ep("HOME", "0", "chair", 0), _ep("HOME", "1", "bed", 1),
           _ep("AWAY", "2", "chair", 2), _ep("AWAY", "3", "bed", 3)]
    assert ce.infer_away_scene(eps) == "AWAY"
    print("  case infer_away_is_later_scene: OK")


def case_infer_away_none_when_not_two_scenes():
    eps = [_ep("ONLY", "0", "chair", 0)]
    assert ce.infer_away_scene(eps) is None
    print("  case infer_away_none_when_not_two_scenes: OK")


# ----------------------------------------------------------------------
# away_recall_total — max cumulative n_cross_scene_recall over AWAY episodes
# ----------------------------------------------------------------------


def case_away_recall_filters_scene_and_takes_max():
    with tempfile.TemporaryDirectory() as d:
        rows = [
            ("HOME", 0),   # home episode — ignored even if it had a count
            ("AWAY", 4),
            ("AWAY", 9),   # cumulative/monotonic -> max is the total
            ("AWAY", 7),
        ]
        for i, (scene, n) in enumerate(rows):
            json.dump(
                {"scene_id": scene, "episode_id": str(i),
                 "bridge_stats_after": {"n_cross_scene_recall": n}},
                open(os.path.join(d, f"episode_{i:03d}.json"), "w"),
            )
        total = ce.away_recall_total(d, away_scene="AWAY")
        assert total == 9, total
    print("  case away_recall_filters_scene_and_takes_max: OK")


def case_away_recall_zero_when_no_away_or_no_counter():
    with tempfile.TemporaryDirectory() as d:
        json.dump({"scene_id": "HOME", "episode_id": "0",
                   "bridge_stats_after": {"n_cross_scene_recall": 5}},
                  open(os.path.join(d, "episode_000.json"), "w"))
        assert ce.away_recall_total(d, away_scene="AWAY") == 0
    print("  case away_recall_zero_when_no_away_or_no_counter: OK")


# ----------------------------------------------------------------------
# cross_env_verdict — recall counter is PRIMARY; soft-SPL delta is secondary
# ----------------------------------------------------------------------


def case_verdict_recall_zero_inconclusive():
    v = ce.cross_env_verdict(recall_total=0, away_mean=0.0, away_p=0.5)
    assert v.startswith("INCONCLUSIVE"), v
    assert "did not fire" in v.lower() or "never" in v.lower(), v
    print("  case verdict_recall_zero_inconclusive: OK")


def case_verdict_recall_fires_states_no_fine_transfer():
    # recall fired -> the load-bearing fact; the delta CANNOT be cross-env
    # transfer (the seam injects no cross-scene WAYPOINT), so the verdict must say
    # so, attribute the delta to same-(away-)scene memory, and point to step 4.
    v = ce.cross_env_verdict(recall_total=12, away_mean=0.03, away_p=0.4)
    assert v.startswith("RECALL FIRES"), v
    assert "counted-not-injected" in v.lower() or "not inject" in v.lower(), v
    assert "waypoint" in v.lower(), v                       # waypoint-vs-read distinction
    assert "same-scene" in v.lower() or "same-(away-)scene" in v.lower(), v  # corrected class
    assert "within-episode" not in v.lower(), v             # corrected: not the within-episode overclaim
    assert "step 4" in v.lower() or "coarse-affordance" in v.lower(), v
    print("  case verdict_recall_fires_states_no_fine_transfer: OK")


# ----------------------------------------------------------------------
# integration: relabel + reuse paired_warm_delta on the AWAY episodes
# ----------------------------------------------------------------------


def case_paired_away_delta_reuses_revisit_bootstrap():
    # one away episode per category; S3 beats S1 on both -> positive away delta.
    s1 = [_ep("HOME", "0", "chair", 0, soft_spl=0.5),
          _ep("AWAY", "2", "chair", 2, soft_spl=0.10),
          _ep("AWAY", "3", "bed", 3, soft_spl=0.20)]
    s3 = [_ep("HOME", "0", "chair", 0, soft_spl=0.5),
          _ep("AWAY", "2", "chair", 2, soft_spl=0.30),
          _ep("AWAY", "3", "bed", 3, soft_spl=0.50)]
    ce.label_by_scene_role(s1, "AWAY")
    ce.label_by_scene_role(s3, "AWAY")
    d = ar.paired_warm_delta(s1, s3, n_bootstrap=200)
    assert d["n"] == 2, d            # only the 2 AWAY episodes paired
    assert abs(d["mean"] - 0.25) < 1e-6, d   # (0.20 + 0.30)/2
    # the HOME episode is the cold control and is identical -> ~0
    c = ar.paired_cold_delta(s1, s3, n_bootstrap=200)
    assert c["n"] == 1 and abs(c["mean"]) < 1e-6, c
    print("  case paired_away_delta_reuses_revisit_bootstrap: OK")


def main() -> int:
    print("cross-env transfer analyzer sanity tests")
    case_label_marks_away_warm_home_cold()
    case_label_handles_full_scene_paths()
    case_infer_away_is_later_scene()
    case_infer_away_none_when_not_two_scenes()
    case_away_recall_filters_scene_and_takes_max()
    case_away_recall_zero_when_no_away_or_no_counter()
    case_verdict_recall_zero_inconclusive()
    case_verdict_recall_fires_states_no_fine_transfer()
    case_paired_away_delta_reuses_revisit_bootstrap()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
