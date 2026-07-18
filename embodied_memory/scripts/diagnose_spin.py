"""Bucket ObjectNav episodes by failure mode to locate the absolute-SPL bottleneck.

R1's r1smoke returned native SPL ~0.03 (vs VLFM 0.304), and the run tail shows a
spin signature: n_steps pinned at the 250 cap, action_forward ~2, action_turn
~240, replan_stuck 200+, n_waypoint_unreachable 200+, action_stop 0 — the agent
turns in place against unreachable waypoints and never moves or stops.

That is a *locomotion/controller* cap, not a frontier-quality (S1+) one, and it
caps BOTH arms — so it matters more for "44% looks weak" than the S1-vs-S1+ lever.
This buckets every episode so we know how much of the 0.03 is spin vs
localization (reached the region, never STOP'd within 0.1 m) vs genuine miss.

Pure stdlib over summary.json — no GPU. Runs on any host with the run dir.

    python embodied_memory/scripts/diagnose_spin.py runs/r1smoke-s1/summary.json [more...]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Tuple

# An episode is a spin-timeout if it burned (almost) the whole budget without
# calling STOP and barely translated — lots of turning, few forward steps.
_TIMEOUT_FRAC = 0.95      # n_steps >= this * max_steps counts as "hit the cap"
_SPIN_FWD_MAX = 10        # forward steps at/below this = "did not translate"


def classify(ep: Dict, max_steps: int) -> str:
    n_steps = int(ep.get("n_steps", 0))
    fwd = int(ep.get("action_forward", 0))
    stop = int(ep.get("action_stop", 0))
    success = bool(ep.get("success", False))
    at_cap = n_steps >= _TIMEOUT_FRAC * max_steps
    if success:
        return "success"                       # STOP within 0.1 m
    if stop >= 1:
        return "stop_miss"                     # called STOP, outside 0.1 m (localization-bound)
    if at_cap and fwd <= _SPIN_FWD_MAX:
        return "spin_timeout"                  # never moved, never stopped — controller spin
    if at_cap:
        return "explore_timeout"               # kept moving, ran out of budget
    return "other"


def summarize(episodes: List[Dict], max_steps: int) -> Tuple[Dict[str, int], Dict[str, float], Dict[str, float]]:
    counts: Dict[str, int] = {}
    soft: Dict[str, float] = {}
    reach1m: Dict[str, int] = {}
    for ep in episodes:
        b = classify(ep, max_steps)
        counts[b] = counts.get(b, 0) + 1
        soft[b] = soft.get(b, 0.0) + float(ep.get("soft_spl", 0.0))
        mind = ep.get("min_distance_to_goal")
        reached = bool(ep.get("success_1m", False)) or (mind is not None and float(mind) < 1.0)
        reach1m[b] = reach1m.get(b, 0) + int(reached)
    mean_soft = {b: soft[b] / counts[b] for b in counts}
    reach_rate = {b: reach1m[b] / counts[b] for b in counts}
    return counts, mean_soft, reach_rate


def _load_episodes(path: str) -> Tuple[List[Dict], int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    eps = data.get("episodes", [])
    max_steps = int((data.get("ablation") or {}).get("max_steps", 0)) or max(
        (int(e.get("n_steps", 0)) for e in eps), default=250)
    return eps, max_steps


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summaries", nargs="+", help="run summary.json path(s)")
    ap.add_argument("--max-steps", type=int, default=0, help="override step cap (else inferred)")
    args = ap.parse_args(argv)

    _ORDER = ["success", "stop_miss", "explore_timeout", "spin_timeout", "other"]
    for path in args.summaries:
        eps, inferred = _load_episodes(path)
        max_steps = args.max_steps or inferred
        counts, mean_soft, reach = summarize(eps, max_steps)
        n = sum(counts.values()) or 1
        print(f"\n== {path}  (n={n}, max_steps={max_steps}) ==")
        print(f"{'bucket':<16}{'n':>4}{'frac':>8}{'mean_soft_spl':>15}{'reach@1m':>10}")
        for b in _ORDER + [k for k in counts if k not in _ORDER]:
            if b not in counts:
                continue
            print(f"{b:<16}{counts[b]:>4}{counts[b]/n:>8.2f}{mean_soft[b]:>15.4f}{reach[b]:>10.2f}")
        spin = counts.get("spin_timeout", 0)
        stop_miss = counts.get("stop_miss", 0)
        print(
            f"  bottleneck: spin_timeout={spin/n:.0%} (controller/locomotion), "
            f"stop_miss={stop_miss/n:.0%} (localization). "
            + ("SPIN dominates → controller is the cap, lifts both arms."
               if spin >= stop_miss and spin > 0 else
               "STOP-localization dominates → detector/arrival, not frontier."
               if stop_miss > spin else "mixed/other.")
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
