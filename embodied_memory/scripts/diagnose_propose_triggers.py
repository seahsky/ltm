"""
Read-only diagnosis of the propose/rerank cadence — why does an episode
re-propose, and how often?

Built for the multion-full1 post-mortem: 3 of 8 S3 episodes re-ranked on
~90% of ticks (ep2 670/749, ep3 600/749, ep7 715/749) while every visible
trigger counter said otherwise — n_propose_reached ≈ 0 (reached-thrash is
fixed), due_to_propose caps at ~n_steps/propose_period ≈ 75, and the
candidate-drop paths' counters (n_waypoint_reached/unreachable,
n_arrival_stop, n_stop_signals) were all 0. A THIRD absorbing mode, not yet
captured. This analyzer mines the per-decision ``decisions[]`` log that every
episode JSON already carries (step_idx, chosen id/source/world_xy, candidate
distances), so it runs on EXISTING run dirs with no GPU and no re-run:

  * decision-gap histogram (gap==1 -> per-tick re-proposes; gap>=period ->
    scheduled cadence);
  * trigger breakdown when the runner recorded it (the ``trigger`` field on
    each decision — new runs only; reconstructed-from-gaps otherwise);
  * chosen-candidate churn: top repeated chosen positions (ping-pong tell),
    same-position re-choose rate, chosen-source tally, near-chosen count;
  * cross-check against the episode counters.

Usage (read-only; point at any run dir(s))::

    python embodied_memory/scripts/diagnose_propose_triggers.py runs/multion-full1-s3
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------


def propose_gap_stats(decisions: List[Dict[str, Any]],
                      propose_period: int = 10) -> Dict[str, Any]:
    """Histogram of step gaps between consecutive decisions. ``n_gap1`` is the
    per-tick re-propose count (the absorbing-mode tell: due_to_propose alone
    can never produce gap==1 runs because last_propose_step refreshes on every
    propose). Pure."""
    steps = [int(d.get("step_idx", -1)) for d in decisions]
    gaps = [b - a for a, b in zip(steps, steps[1:])]
    gaps_sorted = sorted(gaps)
    return {
        "n_decisions": len(steps),
        "n_gap1": sum(1 for g in gaps if g == 1),
        "n_gap_lt_period": sum(1 for g in gaps if g < propose_period),
        "n_gap_ge_period": sum(1 for g in gaps if g >= propose_period),
        "median_gap": (gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else None),
        "max_gap1_run": _max_run(gaps, 1),
    }


def _max_run(values: List[int], target: int) -> int:
    best = cur = 0
    for v in values:
        cur = cur + 1 if v == target else 0
        best = max(best, cur)
    return best


def trigger_breakdown(decisions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Tally of the runner-recorded ``trigger`` field (new runs). Decisions
    from runs predating the field land in ``unrecorded``. Pure."""
    out: Dict[str, int] = {"no_candidate": 0, "scheduled": 0, "reached": 0,
                           "unrecorded": 0}
    for d in decisions:
        t = d.get("trigger")
        if t in out:
            out[t] += 1
        else:
            out["unrecorded"] += 1
    return out


def chosen_churn_stats(decisions: List[Dict[str, Any]],
                       near_m: float = 0.5,
                       round_to: float = 0.5) -> Dict[str, Any]:
    """Churn profile of the CHOSEN candidate across decisions: source tally,
    how often the choice is (a re-quantized) repeat of the previous position
    (ping-pong / re-pick tell), top repeated positions, and how many chosen
    candidates were already within ``near_m`` of the agent at choose time
    (read from the logged per-candidate distance_m). Pure."""
    sources: Counter = Counter()
    positions: Counter = Counter()
    n_same_as_prev = 0
    n_near_chosen = 0
    prev_key: Optional[tuple] = None
    for d in decisions:
        sources[str(d.get("chosen_source"))] += 1
        xy = d.get("chosen_world_xy") or [float("nan"), float("nan")]
        key = (round(float(xy[0]) / round_to) * round_to,
               round(float(xy[1]) / round_to) * round_to)
        positions[key] += 1
        if prev_key is not None and key == prev_key:
            n_same_as_prev += 1
        prev_key = key
        cid = d.get("chosen_id")
        for cand in d.get("candidates") or []:
            if cand.get("id") == cid:
                if float(cand.get("distance_m", 1e9)) < near_m:
                    n_near_chosen += 1
                break
    return {
        "sources": dict(sources),
        "n_same_as_prev": n_same_as_prev,
        "n_near_chosen": n_near_chosen,
        "top_positions": positions.most_common(3),
    }


def classify_mode(gap_stats: Dict[str, Any],
                  churn: Dict[str, Any],
                  n_steps: int) -> str:
    """One-line heuristic verdict per episode. Pure."""
    n = gap_stats["n_decisions"]
    if n == 0:
        return "no decisions logged"
    if n <= max(3, n_steps // 8):
        return "healthy cadence (committed waypoints)"
    if gap_stats["n_gap1"] > n // 2:
        if churn["n_same_as_prev"] > n // 2:
            return ("ABSORBING: per-tick re-propose, SAME pick re-chosen "
                    "(candidate not retained between ticks?)")
        return "ABSORBING: per-tick re-propose, pick ping-pongs (target churn)"
    return "mixed cadence (some bursts; inspect gaps)"


# ----------------------------------------------------------------------
# IO + report
# ----------------------------------------------------------------------


def load_episodes(run_dir: str) -> List[Dict[str, Any]]:
    eps = []
    for p in sorted(glob.glob(os.path.join(run_dir, "episode_*.json"))):
        if p.endswith("_error.json"):
            continue
        try:
            with open(p) as f:
                eps.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue
    return eps


def print_report(run_dir: str, propose_period: int) -> None:
    eps = load_episodes(run_dir)
    print(f"=== {run_dir} ({len(eps)} episodes; propose_period={propose_period}) ===")
    for e in eps:
        decisions = e.get("decisions") or []
        gaps = propose_gap_stats(decisions, propose_period)
        trig = trigger_breakdown(decisions)
        churn = chosen_churn_stats(decisions)
        verdict = classify_mode(gaps, churn, int(e.get("n_steps", 0)))
        cats = ",".join(e.get("target_categories")
                        or [str(e.get("target_category"))])
        print(f"\n  ep{e.get('episode_idx')} [{cats}] "
              f"rerank={e.get('rerank_calls')}/{e.get('n_steps')} "
              f"reached_ctr={e.get('n_propose_reached')} "
              f"filt_near={e.get('n_candidates_filtered_near')} "
              f"wp_reach/unreach={e.get('n_waypoint_reached')}/"
              f"{e.get('n_waypoint_unreachable')}")
        print(f"    gaps: gap1={gaps['n_gap1']} (max run {gaps['max_gap1_run']}) "
              f"<period={gaps['n_gap_lt_period']} >=period={gaps['n_gap_ge_period']} "
              f"median={gaps['median_gap']}")
        print(f"    triggers: no_candidate={trig['no_candidate']} "
              f"scheduled={trig['scheduled']} reached={trig['reached']} "
              f"unrecorded={trig['unrecorded']}")
        print(f"    chosen: sources={churn['sources']} "
              f"same_as_prev={churn['n_same_as_prev']} "
              f"near_chosen={churn['n_near_chosen']}")
        print(f"    top picks: "
              + "; ".join(f"({x:+.1f},{z:+.1f})x{c}"
                          for (x, z), c in churn["top_positions"]))
        print(f"    -> {verdict}")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propose/rerank cadence diagnosis (read-only)")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--propose-period", type=int, default=10,
                        help="REMEMBR_PROPOSE_PERIOD the run used (default 10)")
    args = parser.parse_args(argv)
    for d in args.run_dirs:
        print_report(d, args.propose_period)
    return 0


if __name__ == "__main__":
    sys.exit(main())
