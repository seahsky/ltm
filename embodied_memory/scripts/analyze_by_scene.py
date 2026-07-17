#!/usr/bin/env python
"""Per-scene, per-category breakdown of a run's episodes.

analyze_ablation reports one aggregate over all episodes, which hides whether a
low mean is uniform or driven by one pathological scene. R1's val_mini aggregate
(succ@1m 0.167, floor-level) could be a single multi-floor scene (wcojb4TFT35)
spinning on cross-floor routing while a single-floor scene (TEEsavR23oF) is fine
-- a distinction that decides whether full-val R1 is worth ~12 days per arm.

This groups a run_hm3d_pol summary.json by scene_id (and optionally category) and
prints the reach/termination signals: succ@1m, soft_spl, steps, per-episode
waypoint-unreachable count (the spin signature), and min distance to goal.

Pure-python over summary.json; no GPU, no deps beyond the stdlib.

    python embodied_memory/scripts/analyze_by_scene.py runs/r1vm-s1 runs/r1vm-s1plus
    python embodied_memory/scripts/analyze_by_scene.py --by-category runs/r1vm-s1
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _load_episodes(run: str) -> List[Dict[str, Any]]:
    """Episodes from a run dir or a direct summary.json path."""
    path = run if run.endswith(".json") else os.path.join(run, "summary.json")
    with open(path) as f:
        return json.load(f).get("episodes") or []


def _mean(eps: List[Dict[str, Any]], key: str, default: float = 0.0) -> float:
    if not eps:
        return 0.0
    return sum(float(e.get(key, default) or 0.0) for e in eps) / len(eps)


def _rate(eps: List[Dict[str, Any]], key: str) -> float:
    if not eps:
        return 0.0
    return sum(1 for e in eps if e.get(key)) / len(eps)


def group_stats(
    eps: List[Dict[str, Any]], by_category: bool = False
) -> List[Tuple[str, Dict[str, float]]]:
    """(group_label -> stats) sorted by label. Group by scene_id, or by
    (scene_id, target_category) when by_category."""
    buckets: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for e in eps:
        scene = str(e.get("scene_id", "?"))
        label = f"{scene}/{e.get('target_category', '?')}" if by_category else scene
        buckets[label].append(e)
    out = []
    for label, group in sorted(buckets.items()):
        out.append((label, {
            "n": len(group),
            "succ@1m": _rate(group, "success_1m"),
            "success": _rate(group, "success"),
            "soft_spl": _mean(group, "soft_spl"),
            "spl": _mean(group, "spl"),
            "steps": _mean(group, "n_steps"),
            "unreach_per_ep": _mean(group, "n_waypoint_unreachable"),
            "min_d2g": _mean(group, "min_distance_to_goal"),
        }))
    return out


def _print_run(run: str, by_category: bool) -> None:
    eps = _load_episodes(run)
    print(f"\n=== {run}  (n={len(eps)}) ===")
    if not eps:
        print("  (no episodes)")
        return
    header = "category" if by_category else "scene"
    print(f"  {header:24s}  n  succ@1m success soft_spl   spl  steps unreach/ep min_d2g")
    for label, s in group_stats(eps, by_category=by_category):
        print(f"  {label:24s} {int(s['n']):2d}   "
              f"{s['succ@1m']:.2f}    {s['success']:.2f}   {s['soft_spl']:.3f} "
              f"{s['spl']:.3f} {s['steps']:5.0f}    {s['unreach_per_ep']:5.0f}  {s['min_d2g']:6.2f}")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-scene/-category breakdown of a run's episodes.")
    ap.add_argument("runs", nargs="+", help="Run dirs (or summary.json paths).")
    ap.add_argument("--by-category", action="store_true",
                    help="Group by (scene, target_category) instead of scene only.")
    args = ap.parse_args(argv)
    for run in args.runs:
        _print_run(run, by_category=args.by_category)
    return 0


if __name__ == "__main__":
    sys.exit(main())
