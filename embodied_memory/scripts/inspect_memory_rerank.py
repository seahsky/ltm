"""
Inspect why the rerank picks frontier vs memory candidates — and test
``FrontierPhysicsScorer`` calibration against real run data.

Reads per-episode decision logs (``episode_*.json``) from one or more run
directories and reports:

  1. the empirical distribution of memory-candidate CLIP cosines (carried as
     ``raw_score`` on memory candidates) and of frontier-candidate raw_scores,
     across every decision;
  2. for each decision where a memory candidate competed, the head-to-head —
     which source was chosen and each candidate's ``(score, distance)``;
  3. a calibration readout: replaying the exact ``FrontierPhysicsScorer`` +
     rerank-weight math on the observed memory candidates, it counts how many
     would out-score a baseline frontier candidate under the CURRENT
     ``_MEM_COS_FULL`` and under a PROPOSED value (``--mem-cos-full``). This is
     the knob to turn so memory can compete.

Background: ``n_memory_chosen=0`` in the ``remembr-run62`` smoke traced to memory
cosines (capped ~0.27) sitting just below the crossover vs ``propose_diverse``
frontier candidates (raw ~0.7+), because the dominant 0.70-weight source-aware
physics scorer saturates memory at ``_MEM_COS_FULL=0.32``. Lowering that toward
the real cap lets in-category memories win selection. See ``memory_bridge.py``
``FrontierPhysicsScorer`` (constants) and the rerank weights in ``__init__``.

Usage:
    # default: the validation-smoke runs
    python embodied_memory/scripts/inspect_memory_rerank.py
    # an ablation setting, testing a lower saturation point
    python embodied_memory/scripts/inspect_memory_rerank.py runs/abl-s3-qwen --mem-cos-full 0.27

Plain-text stdout, stdlib only.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# FrontierPhysicsScorer math — MUST stay in sync with memory_bridge.py.
# Mirrored here (not imported) so this tool has no faiss/torch dependency.
#   constants:  FrontierPhysicsScorer._MEM_* (memory_bridge.py:144-146)
#   weights:    EmbodiedMemoryBridge.__init__ default (memory_bridge.py:267)
#               {"history": 0.0, "memory": 0.30, "coherence": 0.70}
#   S_sim is near-uniform across candidates (geometric "go to (x,y)" texts),
#   so for ordering we hold it constant; the comparison below cancels it out.
# ----------------------------------------------------------------------
_MEM_COS_NULL = 0.15
_MEM_DIST_WEIGHT = 0.20
_W_MEMORY = 0.30   # S_sim weight (uniform → cancels in head-to-head)
_W_PHYS = 0.70     # source-aware FrontierPhysicsScorer weight (decides ordering)


def _dist_score(dist: float) -> float:
    if dist <= 0.0:
        return 0.0
    return max(0.0, 1.0 - abs(dist - 2.0) / 4.0)


def _phys_memory(cos: float, dist: float, mem_cos_full: float) -> float:
    span = mem_cos_full - _MEM_COS_NULL
    cos_norm = (cos - _MEM_COS_NULL) / span if span > 0 else 0.0
    cos_norm = min(1.0, max(0.0, cos_norm))
    return min(1.0, max(0.0,
        (1.0 - _MEM_DIST_WEIGHT) * cos_norm + _MEM_DIST_WEIGHT * _dist_score(dist)))


def _phys_frontier(raw: float, bearing: float, dist: float) -> float:
    import math
    bearing_score = max(0.0, 1.0 - abs(bearing) / math.pi)
    return min(1.0, max(0.0,
        0.5 * raw + 0.3 * bearing_score + 0.2 * _dist_score(dist)))


def _final(s_phys: float, s_sim: float = 0.5) -> float:
    # history weight is 0.0; total weight = _W_MEMORY + _W_PHYS = 1.0
    return _W_MEMORY * s_sim + _W_PHYS * s_phys


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


def _iter_decisions(run_dirs: List[str]):
    """Yield (run_label, decision_dict) for every decision in every episode."""
    paths: List[str] = []
    for rd in run_dirs:
        # accept a directory, or a glob pattern
        if any(ch in rd for ch in "*?["):
            for hit in sorted(glob.glob(rd)):
                paths.extend(sorted(glob.glob(os.path.join(hit, "episode_*.json"))))
        else:
            paths.extend(sorted(glob.glob(os.path.join(rd, "episode_*.json"))))
    for p in paths:
        if p.endswith("_error.json"):
            continue
        try:
            d = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  (skipped {p}: {e})", file=sys.stderr)
            continue
        label = os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)
        for dec in d.get("decisions", []):
            yield label, dec


def _summ(xs: List[float], name: str) -> None:
    if not xs:
        print(f"  {name:32} (none)")
        return
    s = sorted(xs)

    def pct(q: float) -> float:
        return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]

    print(f"  {name:32} n={len(s):<5} min={s[0]:.3f} p25={pct(.25):.3f} "
          f"p50={st.median(s):.3f} p75={pct(.75):.3f} p90={pct(.90):.3f} max={s[-1]:.3f}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect rerank frontier-vs-memory selection + test physics calibration.")
    parser.add_argument(
        "run_dirs", nargs="*", default=["runs/remembr-run62-*"],
        help="Run directories or globs (default: runs/remembr-run62-*).")
    parser.add_argument(
        "--mem-cos-full", type=float, default=0.27,
        help="Proposed _MEM_COS_FULL to test against the data (current code = 0.32).")
    parser.add_argument(
        "--frontier-raw", type=float, default=0.70,
        help="Baseline frontier raw_score to compare memory against (propose_diverse "
             "unknown-cell baseline = 0.70).")
    parser.add_argument(
        "--frontier-bearing", type=float, default=0.0,
        help="Baseline frontier bearing (rad); 0 = perfectly aligned (strongest).")
    parser.add_argument(
        "--frontier-dist", type=float, default=1.5,
        help="Baseline frontier distance (m).")
    parser.add_argument(
        "--max-rows", type=int, default=60,
        help="Cap on head-to-head rows printed (0 = all).")
    args = parser.parse_args(argv)

    mem_cos_all: List[float] = []
    mem_dist_all: List[float] = []
    fr_raw_all: List[float] = []
    head2head: List[Tuple[str, int, str, list, list]] = []
    chosen_counts: Dict[str, int] = {}

    for label, dec in _iter_decisions(args.run_dirs):
        cands = dec.get("candidates", [])
        mem = [c for c in cands if c.get("source") == "memory"]
        fr = [c for c in cands if c.get("source") == "frontier"]
        mem_cos_all += [float(c["raw_score"]) for c in mem]
        mem_dist_all += [float(c["distance_m"]) for c in mem]
        fr_raw_all += [float(c["raw_score"]) for c in fr]
        chosen = str(dec.get("chosen_source"))
        chosen_counts[chosen] = chosen_counts.get(chosen, 0) + 1
        if mem:
            head2head.append((
                label, int(dec.get("step_idx", -1)), chosen,
                [(round(float(c["raw_score"]), 3), round(float(c["distance_m"]), 1)) for c in mem],
                [(round(float(c["raw_score"]), 3), round(float(c["distance_m"]), 1)) for c in fr],
            ))

    if not mem_cos_all and not fr_raw_all:
        print("No decisions found. Check the run dir paths.", file=sys.stderr)
        return 1

    print("=" * 78)
    print(f"runs: {args.run_dirs}")
    print(f"chosen_source tally across all decisions: {chosen_counts}")
    print()
    print("=== distributions across ALL decisions ===")
    _summ(mem_cos_all, "memory cosine (raw_score)")
    _summ(mem_dist_all, "memory distance_m")
    _summ(fr_raw_all, "frontier raw_score")

    # ---- calibration readout -------------------------------------------------
    base_fr_final = _final(_phys_frontier(args.frontier_raw, args.frontier_bearing, args.frontier_dist))
    print()
    print("=== calibration readout ===")
    print(f"  baseline frontier: raw={args.frontier_raw} bearing={args.frontier_bearing} "
          f"dist={args.frontier_dist}m  ->  final={base_fr_final:.3f}")

    def win_count(mem_cos_full: float) -> Tuple[int, int]:
        wins = 0
        for cos, dist in zip(mem_cos_all, mem_dist_all):
            if _final(_phys_memory(cos, dist, mem_cos_full)) > base_fr_final:
                wins += 1
        return wins, len(mem_cos_all)

    for tag, mcf in (("CURRENT  0.32", 0.32), (f"PROPOSED {args.mem_cos_full:.2f}", args.mem_cos_full)):
        w, n = win_count(mcf)
        share = (100.0 * w / n) if n else 0.0
        print(f"  _MEM_COS_FULL={tag}: {w}/{n} observed memory candidates "
              f"beat the baseline frontier ({share:.0f}%)")
    print("  (a memory candidate is 'chosen' only if it also out-scores the OTHER live")
    print("   candidates that decision — this is an upper bound on n_memory_chosen.)")

    # ---- head-to-head --------------------------------------------------------
    print()
    print(f"=== decisions where memory competed: {len(head2head)} ===")
    rows = head2head if args.max_rows <= 0 else head2head[: args.max_rows]
    for label, step, chosen, mem, fr in rows:
        print(f"  {label} step{step:>3} chosen={chosen:8} "
              f"mem(cos,dist)={mem} fr(raw,dist)={fr}")
    if args.max_rows > 0 and len(head2head) > args.max_rows:
        print(f"  ... ({len(head2head) - args.max_rows} more; --max-rows 0 for all)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
