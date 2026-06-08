"""
Cross-ENVIRONMENT transfer analyzer (diagnose-first step 2, redesigned).

The same-scene ``analyze_revisit`` labels cold/warm by visit-order within
``(scene, category)``. That is structurally WRONG for cross-environment recall:
the away-scene visit is the FIRST visit to scene B, yet it is the "query" we
care about (it runs with a scene-A sighting in the persistent LTM). Applying
visit-order labelling to a multi-visit away scene instead measures *within-away*
same-scene revisit — the confound that made the first ``crossenv-1`` run look
positive (+0.1675) when no cross-scene transfer occurred.

This analyzer relabels by scene ROLE — **away-scene episodes = the query (warm),
home-scene episodes = the cold source/control** — and then reuses
``analyze_revisit``'s paired bootstrap (``paired_warm_delta`` / ``paired_cold_delta``)
unchanged. It pairs the away episodes S3-vs-S1 by ``(scene_id, episode_id)``
(Habitat renumbers episode_id to numeric, but identically across S1/S3, so the
pairing is stable).

PRIMARY evidence is the **cross-scene recall counter**
(``bridge_stats_after.n_cross_scene_recall`` over the away episodes, max of the
process-cumulative snapshot). It is read by *scene_id*, NOT by an
episode_id substring — the first ``crossenv-1`` recall readout reported 0 only
because it filtered on a ``"warm-away"`` id that Habitat had stripped. Because
the ``LTM_CROSS_SCENE`` seam is **counted-not-injected** (a scene-A position is
invalid in scene B), the soft-SPL delta cannot be attributed to cross-env
transfer; the recall counter (>0 vs 0) is the load-bearing number.

Usage::

    python embodied_memory/scripts/analyze_cross_env.py \
        runs/crossenv-2-s1 runs/crossenv-2-s3 --away-scene wcojb4TFT35
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_revisit as ar  # noqa: E402


def _bare(scene: Optional[str]) -> str:
    """Bare scene token from a bare name or a full glb path."""
    if not scene:
        return ""
    base = os.path.basename(str(scene))
    return base.split(".")[0]


def label_by_scene_role(episodes: List["ar.RevisitEpisode"], away_scene: str) -> List["ar.RevisitEpisode"]:
    """Set ``visit_order`` by scene ROLE: away-scene episodes = 1 (warm / query),
    every other (home) episode = 0 (cold / source). Mutates in place."""
    aw = _bare(away_scene)
    for e in episodes:
        e.visit_order = 1 if _bare(e.scene_id) == aw else 0
    return episodes


def infer_away_scene(episodes: List["ar.RevisitEpisode"]) -> Optional[str]:
    """Infer the away scene as the one whose episodes run LATER (home is ordered
    first by ``group_by_scene`` + the cold-first builder). Returns the bare scene
    name, or None unless there are exactly two scenes."""
    by_scene: Dict[str, List[int]] = {}
    for e in episodes:
        by_scene.setdefault(_bare(e.scene_id), []).append(e.episode_idx)
    if len(by_scene) != 2:
        return None
    return max(by_scene, key=lambda s: min(by_scene[s]))


def away_recall_total(run_dir: str, away_scene: str) -> int:
    """Max cumulative ``n_cross_scene_recall`` across the AWAY-scene episode logs.

    The counter is process-cumulative (never reset per episode), so the max over
    the away episodes is the total recall-event count. Filtered by scene_id —
    robust to Habitat renumbering the episode_id.
    """
    aw = _bare(away_scene)
    best = 0
    for f in sorted(glob.glob(os.path.join(run_dir, "episode_*.json"))):
        if f.endswith("_error.json"):
            continue
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        if _bare(d.get("scene_id")) != aw:
            continue
        n = (d.get("bridge_stats_after") or {}).get("n_cross_scene_recall", 0)
        try:
            best = max(best, int(n or 0))
        except (TypeError, ValueError):
            continue
    return best


def cross_env_verdict(recall_total: int, away_mean: float, away_p: float) -> str:
    """Recall counter is PRIMARY (the seam is counted-not-injected, so the
    soft-SPL delta is NOT cross-env transfer)."""
    if recall_total <= 0:
        return (
            "INCONCLUSIVE: the cross-scene recall counter did not fire (counter=0). "
            "Either the home sighting was not deposited/persisted into the LTM, or it "
            "did not clear the retrieval bar in the away scene. Diagnose (persistence / "
            "caption cosine / fetch_k) before drawing any cross-env conclusion."
        )
    return (
        f"RECALL FIRES (cross-scene recall counter = {recall_total} > 0): the LTM DOES "
        "recall the home-scene sighting while the agent is in the away scene. But the "
        "LTM_CROSS_SCENE seam is counted-not-injected (the home position is invalid in "
        f"the away frame), so this recall yields no waypoint — the away S3-S1 soft-SPL "
        f"({away_mean:+.4f}, p={away_p:.3f}) is same-(away-)scene memory (within- OR "
        "cross-episode accumulation across the away episodes), NOT cross-env transfer. The "
        "fine layer cannot inject a cross-scene WAYPOINT (positions are scene-filtered); the "
        "only cross-scene READ is a goal-irrelevant rerank score perturbation that cannot "
        "steer toward the goal. Positive cross-env transfer needs a position-free mechanism "
        "(step 4 coarse-affordance)."
    )


def print_cross_env_report(
    s1: List["ar.RevisitEpisode"],
    s3: List["ar.RevisitEpisode"],
    away_scene: str,
    recall_total: int,
    n_bootstrap: int = 5000,
) -> Dict[str, Any]:
    label_by_scene_role(s1, away_scene)
    label_by_scene_role(s3, away_scene)
    away = ar.paired_warm_delta(s1, s3, n_bootstrap=n_bootstrap)   # away = warm/query
    home = ar.paired_cold_delta(s1, s3, n_bootstrap=n_bootstrap)   # home = cold/control

    print(f"\n=== cross-env transfer (away scene = {_bare(away_scene)}) ===")
    print(f"  AWAY  S3-S1 soft-SPL (query; cross-env effect): n={away['n']}  "
          f"mean={away['mean']:+.4f}  90% CI=[{away['lo']:+.4f}, {away['hi']:+.4f}]  "
          f"one-sided p(<=0)={away['p_le_zero']:.3f}")
    print(f"  HOME  S3-S1 soft-SPL (source/control, expect ~0): n={home['n']}  "
          f"mean={home['mean']:+.4f}  90% CI=[{home['lo']:+.4f}, {home['hi']:+.4f}]")
    print(f"\n=== cross-scene recall (PRIMARY evidence) ===")
    print(f"  away-scene cumulative cross-scene recall events = {recall_total}")
    print(f"\n=== verdict ===")
    print(f"  {cross_env_verdict(recall_total, away['mean'], away['p_le_zero'])}")
    return {"away": away, "home": home, "recall_total": recall_total}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Cross-environment transfer analysis")
    parser.add_argument("run_dirs", nargs="+", help="Two run dirs: the S1 and S3 runs.")
    parser.add_argument("--away-scene", default=None,
                        help="Bare away (warm/query) scene name. Inferred if omitted.")
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args(argv)

    runs = [ar.load_revisit_run(p) for p in args.run_dirs]
    by_setting = {r.setting: r for r in runs}
    if 1 not in by_setting or 3 not in by_setting:
        print(f"ERROR: need one setting=1 and one setting=3 run; got settings "
              f"{[r.setting for r in runs]}.", file=sys.stderr)
        return 1
    s1, s3 = by_setting[1], by_setting[3]

    away = args.away_scene or infer_away_scene(s3.episodes)
    if not away:
        print("ERROR: could not infer the away scene (need exactly two scenes); "
              "pass --away-scene.", file=sys.stderr)
        return 1

    print("=== runs ===")
    for r in (s1, s3):
        print(f"  {r.name}: setting={r.setting} n_episodes={len(r.episodes)}")
    recall_total = away_recall_total(s3.path, away)
    print_cross_env_report(s1.episodes, s3.episodes, away, recall_total, args.bootstrap)
    return 0


if __name__ == "__main__":
    sys.exit(main())
