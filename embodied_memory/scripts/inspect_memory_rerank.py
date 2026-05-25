"""
Inspect why the rerank picks frontier vs memory vs remembr candidates — and
test ``FrontierPhysicsScorer`` calibration against real run data.

Reads per-episode decision logs (``episode_*.json``) from one or more run
directories and reports:

  1. per-SOURCE distribution of candidate ``raw_score`` (remembr = LLM pick,
     frontier = propose_diverse injection, memory = LTM-injected) and of memory
     ``distance_m``, across every decision;
  2. a FAITHFUL per-decision replay: re-scores the full logged candidate pool
     with the exact ``FrontierPhysicsScorer`` math and predicts which SOURCE
     wins each decision, under the CURRENT ``_MEM_COS_FULL=0.32`` and a sweep of
     proposed values — i.e. predicted ``n_memory_chosen`` if you retune the knob;
  3. the head-to-head on every decision where a memory candidate competed —
     ALL sources shown, with each candidate's ``(raw_score, distance)``.

Background: ``n_memory_chosen=0`` in the ``remembr-run62`` smoke. The dominant
0.70-weight source-aware physics scorer scores memory by CLIP cosine (capped
low) and everything else (remembr LLM pick + frontier) by ``0.5·raw+0.3·bearing
+0.2·dist``. The occupancy-aware frontier injection (raw≈0.875+) out-scores both
memory AND the LLM's own waypoint. See ``memory_bridge.py`` ``FrontierPhysicsScorer``.

The replay assumes the 0.30-weight similarity scorer (S_sim) is ~uniform across
candidates (its texts are geometric "go to (x,y)" strings — see memory_bridge.py
comment) so the argmax is decided by S_phys; this is the documented design.

Usage:
    python embodied_memory/scripts/inspect_memory_rerank.py
    python embodied_memory/scripts/inspect_memory_rerank.py runs/abl-s3-qwen
    python embodied_memory/scripts/inspect_memory_rerank.py --sweep 0.32 0.28 0.25 0.23 0.20

Plain-text stdout, stdlib only (no faiss/torch).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# FrontierPhysicsScorer math — MUST stay in sync with memory_bridge.py.
# Mirrored here (not imported) so this tool has no faiss/torch dependency.
#   constants:  FrontierPhysicsScorer._MEM_* (memory_bridge.py:144-146)
#   weights:    EmbodiedMemoryBridge.__init__ default (memory_bridge.py:267)
#               {"history": 0.0, "memory": 0.30, "coherence": 0.70}
# Memory candidates get the cosine branch; ANY other source (remembr LLM pick,
# frontier injection, legacy "planner") gets the geometric branch.
# ----------------------------------------------------------------------
_MEM_COS_NULL = 0.15
_MEM_DIST_WEIGHT = 0.20


def _dist_score(dist: float) -> float:
    if dist <= 0.0:
        return 0.0
    return max(0.0, 1.0 - abs(dist - 2.0) / 4.0)


def _phys(source: str, raw: float, dist: float, bearing: float, mem_cos_full: float) -> float:
    """Exact FrontierPhysicsScorer.score, parameterised by _MEM_COS_FULL."""
    if source == "memory":
        span = mem_cos_full - _MEM_COS_NULL
        cos_norm = (raw - _MEM_COS_NULL) / span if span > 0 else 0.0
        cos_norm = min(1.0, max(0.0, cos_norm))
        s = (1.0 - _MEM_DIST_WEIGHT) * cos_norm + _MEM_DIST_WEIGHT * _dist_score(dist)
    else:
        bearing_score = max(0.0, 1.0 - abs(bearing) / math.pi)
        s = 0.5 * raw + 0.3 * bearing_score + 0.2 * _dist_score(dist)
    return min(1.0, max(0.0, s))


def _winning_source(cands: List[dict], mem_cos_full: float) -> str:
    """Replay the rerank argmax over a decision's full candidate pool.
    S_sim (0.30 weight) is ~uniform across candidates → argmax is decided by
    S_phys (0.70 weight), so we rank by S_phys directly."""
    best_src, best_score = None, -1.0
    for c in cands:
        s = _phys(c["source"], float(c["raw_score"]), float(c["distance_m"]),
                  float(c.get("bearing_rad", 0.0)), mem_cos_full)
        if s > best_score:
            best_score, best_src = s, c["source"]
    return best_src or "?"


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


def _iter_decisions(run_dirs: List[str]):
    paths: List[str] = []
    for rd in run_dirs:
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
        print(f"  {name:30} (none)")
        return
    s = sorted(xs)

    def pct(q: float) -> float:
        return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]

    print(f"  {name:30} n={len(s):<5} min={s[0]:.3f} p25={pct(.25):.3f} "
          f"p50={st.median(s):.3f} p75={pct(.75):.3f} p90={pct(.90):.3f} max={s[-1]:.3f}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect rerank source selection + test physics calibration on real data.")
    parser.add_argument(
        "run_dirs", nargs="*", default=["runs/remembr-run62-*"],
        help="Run directories or globs (default: runs/remembr-run62-*).")
    parser.add_argument(
        "--sweep", type=float, nargs="+", default=[0.32, 0.28, 0.25, 0.23, 0.20],
        help="_MEM_COS_FULL values to test (first should be the current code value 0.32).")
    parser.add_argument(
        "--max-rows", type=int, default=60, help="Cap on head-to-head rows (0 = all).")
    args = parser.parse_args(argv)

    raw_by_source: Dict[str, List[float]] = {}
    mem_dist_all: List[float] = []
    actual_chosen: Dict[str, int] = {}
    # decisions where a memory candidate competed: keep the full pool for replay
    mem_decisions: List[Tuple[str, int, str, List[dict]]] = []

    for label, dec in _iter_decisions(args.run_dirs):
        cands = dec.get("candidates", [])
        for c in cands:
            raw_by_source.setdefault(c.get("source", "?"), []).append(float(c["raw_score"]))
            if c.get("source") == "memory":
                mem_dist_all.append(float(c["distance_m"]))
        chosen = str(dec.get("chosen_source"))
        actual_chosen[chosen] = actual_chosen.get(chosen, 0) + 1
        if any(c.get("source") == "memory" for c in cands):
            mem_decisions.append((label, int(dec.get("step_idx", -1)), chosen, cands))

    if not raw_by_source:
        print("No decisions found. Check the run dir paths.", file=sys.stderr)
        return 1

    print("=" * 80)
    print(f"runs: {args.run_dirs}")
    print(f"ACTUAL chosen_source tally (from logs): {actual_chosen}")
    print(f"decisions where memory was in the pool: {len(mem_decisions)}")
    print()
    print("=== raw_score distribution per source (all decisions) ===")
    for src in sorted(raw_by_source):
        _summ(raw_by_source[src], f"{src} raw_score")
    _summ(mem_dist_all, "memory distance_m")

    # ---- faithful replay: predicted winning source under each _MEM_COS_FULL ----
    print()
    print("=== predicted winner over memory-present decisions (faithful replay) ===")
    print(f"  (replaying {len(mem_decisions)} decisions; argmax of source-aware S_phys)")
    for mcf in args.sweep:
        tally: Dict[str, int] = {}
        for _label, _step, _chosen, cands in mem_decisions:
            w = _winning_source(cands, mcf)
            tally[w] = tally.get(w, 0) + 1
        mem_wins = tally.get("memory", 0)
        tag = "  (current)" if abs(mcf - 0.32) < 1e-9 else ""
        print(f"  _MEM_COS_FULL={mcf:.2f}: memory wins {mem_wins}/{len(mem_decisions)} "
              f"| full tally {tally}{tag}")
    print("  NOTE: memory must beat the BEST frontier/remembr candidate that decision,")
    print("        not a fixed baseline. If memory never wins even at low saturation,")
    print("        the cap is not the only lever (see frontier raw_score distribution).")

    # ---- head-to-head (all sources) ----
    print()
    print(f"=== memory-present decisions: head-to-head (all sources) ===")
    rows = mem_decisions if args.max_rows <= 0 else mem_decisions[: args.max_rows]
    for label, step, chosen, cands in rows:
        by_src: Dict[str, list] = {}
        for c in cands:
            by_src.setdefault(c["source"], []).append(
                (round(float(c["raw_score"]), 3), round(float(c["distance_m"]), 1)))
        parts = " ".join(f"{s}={by_src[s]}" for s in sorted(by_src))
        print(f"  {label} step{step:>3} chosen={chosen:8} {parts}")
    if args.max_rows > 0 and len(mem_decisions) > args.max_rows:
        print(f"  ... ({len(mem_decisions) - args.max_rows} more; --max-rows 0 for all)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
