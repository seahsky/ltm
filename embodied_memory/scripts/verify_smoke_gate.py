#!/usr/bin/env python3
"""
Verify a Run-4 smoke-gate run.

Reads ``<run_dir>/episode_000.json`` and evaluates the gating conditions
from the Run-4 plan's smoke gate table:

  - crash-free        (JSON parses, top-level keys present)
  - n_steps           > 50
  - path_traveled     ≥ 4 m
  - n_frontier_chosen ≥ 1

``dist_to_goal`` and ``n_stop_signals`` are reported as info only, per
the Run-4 plan (informative but not gating).

The per-step JSON doesn't carry ``distance_to_goal`` (the serializer
drops ``step.info``), so the plan's "dist_to_goal < starting − 2 m"
check is reported as a derived ``displacement`` (start→end straight-line)
without a hard pass/fail — ``path_traveled ≥ 4 m`` is the canonical
movement gate in its place.

Usage::

    python embodied_memory/scripts/verify_smoke_gate.py [run_dir]

Defaults ``run_dir`` to ``runs/remembr-smoke-frontier``. Exits 0 if all
gating conditions pass, 1 otherwise — safe to chain after the smoke
command::

    python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr ... \\
        && python embodied_memory/scripts/verify_smoke_gate.py
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Any, Dict, List, Tuple


GATES: List[Tuple[str, str]] = [
    # (key, label) — order matters for output stability.
    ("crash_free", "crash-free"),
    ("n_steps", "n_steps > 50"),
    ("path_traveled", "path_traveled ≥ 4 m"),
    ("n_frontier_chosen", "n_frontier_chosen ≥ 1"),
]


def _path_length(steps: List[Dict[str, Any]]) -> float:
    """Sum of straight-line segments between consecutive agent_pos
    samples. Habitat positions are (x, y, z); ground-plane is (x, z)."""
    if not steps:
        return 0.0
    pts = []
    for s in steps:
        ap = s.get("agent_pos")
        if not ap or len(ap) < 3:
            continue
        pts.append((float(ap[0]), float(ap[2])))
    return sum(
        math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        for i in range(len(pts) - 1)
    )


def _displacement(steps: List[Dict[str, Any]]) -> float:
    """Straight-line distance from the agent's first to last logged xz."""
    pts = []
    for s in steps:
        ap = s.get("agent_pos")
        if not ap or len(ap) < 3:
            continue
        pts.append((float(ap[0]), float(ap[2])))
    if len(pts) < 2:
        return 0.0
    return math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])


def _evaluate(ep: Dict[str, Any]) -> Dict[str, Any]:
    steps = ep.get("steps") or []
    decisions = ep.get("decisions") or []
    path_traveled = _path_length(steps)
    displacement = _displacement(steps)

    n_steps = int(ep.get("n_steps", 0))
    n_frontier_chosen = int(ep.get("n_frontier_chosen", 0))
    n_frontier_total = sum(
        int(d.get("n_frontier_candidates", 0)) for d in decisions
    )

    return {
        # gating
        "crash_free": True,  # if we got here, JSON parsed and keys read
        "n_steps_val": n_steps,
        "n_steps": n_steps > 50,
        "path_traveled_val": path_traveled,
        "path_traveled": path_traveled >= 4.0,
        "n_frontier_chosen_val": n_frontier_chosen,
        "n_frontier_chosen": n_frontier_chosen >= 1,
        # info-only
        "displacement_val": displacement,
        "dist_to_goal_val": ep.get("distance_to_goal"),
        "n_stop_signals_val": int(ep.get("n_stop_signals", 0)),
        "success_val": bool(ep.get("success", False)),
        "soft_spl_val": float(ep.get("soft_spl", 0.0)),
        "spl_val": float(ep.get("spl", 0.0)),
        "n_frontier_total_val": n_frontier_total,
        "n_decisions_val": len(decisions),
        # metadata
        "scene_id": ep.get("scene_id"),
        "target_category": ep.get("target_category"),
        "episode_id": ep.get("episode_id"),
    }


def _fmt_passfail(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _print_report(run_dir: str, r: Dict[str, Any]) -> bool:
    print(f"=== Run-4 smoke gate verify ===")
    print(f"  run_dir:        {run_dir}")
    print(f"  scene_id:       {r.get('scene_id')}")
    print(f"  episode_id:     {r.get('episode_id')}")
    print(f"  target:         {r.get('target_category')}")
    print()
    print("  --- gating ---")
    print(f"  [{_fmt_passfail(r['crash_free'])}] crash-free            "
          f"(JSON parsed, n_decisions={r['n_decisions_val']})")
    print(f"  [{_fmt_passfail(r['n_steps'])}] n_steps > 50           "
          f"(actual: {r['n_steps_val']})")
    print(f"  [{_fmt_passfail(r['path_traveled'])}] path_traveled ≥ 4 m   "
          f"(actual: {r['path_traveled_val']:.2f} m)")
    print(f"  [{_fmt_passfail(r['n_frontier_chosen'])}] n_frontier_chosen ≥ 1  "
          f"(actual: {r['n_frontier_chosen_val']}, "
          f"of {r['n_frontier_total_val']} proposed)")
    print()
    print("  --- info-only ---")
    d2g = r['dist_to_goal_val']
    d2g_s = f"{d2g:.2f} m" if isinstance(d2g, (int, float)) else str(d2g)
    print(f"  displacement (start→end):  {r['displacement_val']:.2f} m")
    print(f"  dist_to_goal (final):      {d2g_s}")
    print(f"  n_stop_signals:            {r['n_stop_signals_val']}")
    print(f"  success / spl / soft_spl:  {r['success_val']} / "
          f"{r['spl_val']:.3f} / {r['soft_spl_val']:.3f}")
    print()

    gates_passed = all(r[k] for k, _ in GATES)
    print(f"  gate: {_fmt_passfail(gates_passed)}")
    if not gates_passed:
        print()
        print("  Diagnostic hints (from Run-4 plan):")
        if not r["n_steps"]:
            print("    - n_steps low → controller-stall regressed; check that "
                  "frontier_planner.py still has the collision-escape "
                  "(commit 117028d) and force-replan (commit 6265870) patches.")
        if not r["path_traveled"]:
            print("    - path_traveled low → agent spinning at start. Confirm "
                  "n_frontier_chosen > 0; if 0, frontier candidates aren't "
                  "entering the pool (merge logic broken or all de-duped).")
        if not r["n_frontier_chosen"]:
            print("    - n_frontier_chosen = 0 → merge logic wrong, all "
                  "frontier picks de-duped against LLM, or rerank scoring "
                  "floor never picks frontier. Inspect decisions[*]"
                  ".n_frontier_candidates in the JSON.")
    return gates_passed


def _load_episode(path: str):
    if not os.path.isfile(path):
        return None, f"not found: {path}"
    try:
        with open(path) as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"parse failed: {e}"


def _print_multi_summary(run_dir: str, reports: List[Tuple[str, Dict[str, Any]]]) -> bool:
    """Multi-episode summary table + roll-up gate.

    Per-episode gating mirrors single-episode mode. Overall gate semantics:
      - Architectural pass: total n_frontier_chosen across all episodes >= 1.
      - Navigation pass:    at least 1 episode satisfies n_steps > 50 AND
                            path_traveled >= 4 m.
      - Overall PASS if both architectural and navigation pass.
    """
    print(f"=== Run-4 smoke gate verify (multi-episode) ===")
    print(f"  run_dir: {run_dir}")
    print(f"  episodes: {len(reports)}")
    print()
    hdr = (f"  {'ep':>3} {'scene':<18} {'tgt':<10} {'steps':>5} {'path_m':>7} "
           f"{'d2g_m':>7} {'front_chosen':>13} {'stops':>5} {'crash':>5}  arch  nav")
    print(hdr)
    arch_total = 0
    nav_any = False
    for ep_path, r in reports:
        ep_idx = os.path.basename(ep_path).replace("episode_", "").replace(".json", "")
        n_steps = r["n_steps_val"]
        path_v = r["path_traveled_val"]
        d2g = r["dist_to_goal_val"]
        d2g_s = f"{d2g:6.2f}" if isinstance(d2g, (int, float)) else "   n/a"
        n_fc = r["n_frontier_chosen_val"]
        n_stop = r["n_stop_signals_val"]
        crash = "OK" if r["crash_free"] else "FAIL"
        arch_ep = r["n_frontier_chosen"]
        nav_ep = r["n_steps"] and r["path_traveled"]
        arch_total += n_fc
        if nav_ep:
            nav_any = True
        scene = (r.get("scene_id") or "?")[:18]
        tgt = (r.get("target_category") or "?")[:10]
        print(
            f"  {ep_idx:>3} {scene:<18} {tgt:<10} {n_steps:>5d} {path_v:>7.2f} "
            f"{d2g_s} {n_fc:>13d} {n_stop:>5d} {crash:>5}  "
            f"{_fmt_passfail(arch_ep)} {_fmt_passfail(nav_ep)}"
        )
    print()
    print("  --- roll-up gate ---")
    arch_pass = arch_total >= 1
    print(f"  architectural (any-episode n_frontier_chosen >= 1): "
          f"{_fmt_passfail(arch_pass)} (total chosen across run: {arch_total})")
    print(f"  navigation (any-episode n_steps>50 AND path>=4m):   "
          f"{_fmt_passfail(nav_any)}")
    overall = arch_pass and nav_any
    print(f"  gate: {_fmt_passfail(overall)}")
    if not overall:
        print()
        print("  Diagnostic hints:")
        if not arch_pass:
            print("    - architectural fail: no frontier candidates chosen in any "
                  "episode. Merge logic broken or all picks de-duped against LLM. "
                  "Inspect a decisions[*] dump (chosen_source distribution).")
        if not nav_any:
            print("    - navigation fail: no episode satisfies n_steps>50 AND "
                  "path>=4m. If architectural passes, this is downstream of "
                  "(a) false STOP firing early — bump REMEMBR_STOP_MIN_STEP, "
                  "(b) scene corner-stall — try a different scene/episode, or "
                  "(c) LLM degeneracy — stub firing in place of real ANSWERs.")
    return overall


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "run_dir",
        nargs="?",
        default="runs/remembr-smoke-frontier",
        help="smoke run directory (default: runs/remembr-smoke-frontier)",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="single episode index (default: scan all episode_*.json in run_dir)",
    )
    args = parser.parse_args(argv)

    if args.episode is not None:
        ep_path = os.path.join(args.run_dir, f"episode_{args.episode:03d}.json")
        ep, err = _load_episode(ep_path)
        if ep is None:
            print(f"ERROR: {err}")
            return 2
        report = _evaluate(ep)
        ok = _print_report(args.run_dir, report)
        return 0 if ok else 1

    # multi-episode: scan all
    ep_paths = sorted(glob.glob(os.path.join(args.run_dir, "episode_*.json")))
    ep_paths = [p for p in ep_paths if "_error" not in os.path.basename(p)]
    if not ep_paths:
        print(f"ERROR: no episode_*.json files in {args.run_dir}")
        return 2

    reports: List[Tuple[str, Dict[str, Any]]] = []
    for p in ep_paths:
        ep, err = _load_episode(p)
        if ep is None:
            print(f"WARN: skipping {p}: {err}")
            continue
        reports.append((p, _evaluate(ep)))

    if len(reports) == 1:
        # Single-episode dir — defer to the legacy single-episode report.
        ok = _print_report(args.run_dir, reports[0][1])
        return 0 if ok else 1

    ok = _print_multi_summary(args.run_dir, reports)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
