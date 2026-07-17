#!/usr/bin/env python
"""Are the ObjectNav episodes physically solvable on the loaded navmesh?

R1 (memory-off, stock ObjectNav) showed the agent averaging min_d2g ~8 m from the
goal with n_waypoint_unreachable 60-99/episode and astar_path=0 everywhere. Two
very different roots produce that:

  (controller) the goal IS navmesh-reachable from the start, but the searcher/
                follower fails to route to it -> a controller/frontier bug to fix.
  (setup)       the goal is NOT navmesh-reachable from the start on the loaded
                scene (disconnected navmesh island, wrong navmesh, multi-floor
                split) -> the episodes are unsolvable and min_d2g is floor-bound
                by geometry, not by the controller.

This splits them with the sim's own geodesic, no LLM / no models: for each
episode it resets, then asks sim.geodesic_distance(start -> nearest goal
view_point). A FINITE geodesic => solvable (controller is on trial). INF/None =>
the goal is unreachable from the start (setup is on trial). It also prints the
Euclidean distance so a finite-but-huge geodesic (detour) is visible.

    python embodied_memory/scripts/diagnose_episode_reachability.py --split val_mini
    python embodied_memory/scripts/diagnose_episode_reachability.py --split val_mini --scene TEEsavR23oF
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional

import numpy as np

# Runnable standalone (bare `python embodied_memory/scripts/diagnose_*.py`) with
# no pre-set PYTHONPATH, matching the test files' self-insert.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embodied_memory.habitat_env import HabitatObjectNavSource  # noqa: E402
from embodied_memory.run_hm3d_pol import (
    _resolve_episodes_path_for_split,
    _resolve_scene_list,
)


def _euclid(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a - b))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Per-episode navmesh solvability (geodesic start->goal).")
    ap.add_argument("--split", default="val_mini", choices=["val_mini", "val", "train"])
    ap.add_argument("--scene", default="all",
                    help="scene id / 'all' (auto-discover from the split's content dir).")
    ap.add_argument("--episodes-path", default=None,
                    help="override the split's episodes .json.gz.")
    ap.add_argument("--scene-dataset-path", default=None)
    ap.add_argument("--n-episodes", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=250)
    args = ap.parse_args(argv)

    ep_path = args.episodes_path or _resolve_episodes_path_for_split(args.split)
    if not ep_path:
        print(f"FATAL: no episodes for split={args.split}")
        return 1
    scenes = _resolve_scene_list(args.scene, ep_path)

    source = HabitatObjectNavSource(
        scene_id=scenes if len(scenes) > 1 else scenes[0],
        scene_dataset_path=args.scene_dataset_path,
        episodes_path=ep_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        target_category=None,   # 'any' — every episode, no category filter
    )

    n = source.num_episodes()   # Habitat source counts via num_episodes(), not __len__
    print(f"split={args.split} scenes={scenes} episodes={n}\n")
    print(f"  {'idx':>3} {'scene':14s} {'category':11s} {'geodesic':>9} {'euclid':>7} {'detour':>7}  verdict")

    solvable = 0
    unreachable = 0
    per_scene = {}
    for i in range(n):
        try:
            step, ep = source.reset(i)
        except Exception as e:
            print(f"  {i:>3} reset failed: {type(e).__name__}: {e}")
            continue
        scene = str(getattr(ep, "scene_id", "?")).split("-")[-1]
        cat = str(getattr(ep, "target_category", "?"))
        start = np.asarray(step.agent_state.position, dtype=np.float32)

        geo = source.distance_to_category(start, cat)          # geodesic to nearest view_point, or None
        near = source.nearest_category_viewpoint(start, cat)   # (geo, vp) or None
        euc = _euclid(start, near[1]) if near else float("nan")

        ok = geo is not None and math.isfinite(geo)
        detour = (geo / euc) if (ok and euc and math.isfinite(euc) and euc > 1e-6) else float("nan")
        verdict = "SOLVABLE" if ok else "UNREACHABLE(geodesic=inf/None)"
        rec = per_scene.setdefault(scene, [0, 0])
        if ok:
            solvable += 1; rec[0] += 1
        else:
            unreachable += 1; rec[1] += 1
        gs = f"{geo:9.2f}" if ok else f"{'inf':>9}"
        print(f"  {i:>3} {scene:14s} {cat:11s} {gs} {euc:7.2f} {detour:7.2f}  {verdict}")

    print(f"\n  SOLVABLE   {solvable}/{n}")
    print(f"  UNREACHABLE {unreachable}/{n}")
    for sc, (s, u) in sorted(per_scene.items()):
        print(f"    {sc:14s} solvable={s} unreachable={u}")
    print()
    if unreachable > solvable:
        print("  => VERDICT: SETUP. Most episodes have no navmesh path start->goal; the")
        print("     goal is geometrically unreachable, so min_d2g is floor-bound by the")
        print("     loaded scene/navmesh, NOT the controller. Fix scene/navmesh loading first.")
    else:
        print("  => VERDICT: CONTROLLER. Episodes are navmesh-solvable, so the spin/")
        print("     min_d2g is the searcher/follower failing to route to a reachable goal.")
        print("     Next probe: instrument the follower's None returns vs pathfinder.find_path.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
