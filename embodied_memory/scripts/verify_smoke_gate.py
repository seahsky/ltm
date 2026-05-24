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
        default=0,
        help="episode index to read (default: 0)",
    )
    args = parser.parse_args(argv)

    ep_path = os.path.join(args.run_dir, f"episode_{args.episode:03d}.json")
    if not os.path.isfile(ep_path):
        print(f"ERROR: {ep_path} not found.")
        return 2
    try:
        with open(ep_path) as f:
            ep = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: {ep_path} failed to parse: {e}")
        return 2

    report = _evaluate(ep)
    ok = _print_report(args.run_dir, report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
