"""Semantic-frontier vacuous-arm verdict over a run's summary.json.

Used by the R1 driver (Table 1: S1 geometric frontier vs S1+ BLIP-2 ITM
frontier): after the S1+ arm runs, certify it was a REAL arm before quoting the
A/B. The `semantic_frontier=True` flag on every candidate cannot show this —
it is set whenever the weight is on, regardless of what the scorer produced. Two
silent-vacuous modes it misses:

  (a) NEVER SCORED  — BLIP-2 failed to load or the weight was off: no frontier
                      got a semantic blend (n_semantic_scored == 0).
  (b) SCORED, FLAT  — every frontier read the same semantic value, so raw_score
                      is a uniform rescale of geom_score and the ranking is
                      identical to S1 (semantic_spread_max ~ 0). This is the CLIP
                      flatness measured at 0.020 three times.

Either makes S1+ byte-equivalent to S1 while the run exits 0, so the driver must
FATAL on both. Pure-python over summary.json so the rule is unit-testable without
a GPU (see test_check_semantic_arm.py). Exit 0 = GREEN, 1 = RED.

    python embodied_memory/scripts/check_semantic_arm.py runs/<tag>-s1plus/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple

_MIN_SCORED_DEFAULT = 1
_MIN_SPREAD_DEFAULT = 0.01


def evaluate_semantic_arm(
    summary: Dict[str, Any],
    *,
    min_scored: int = _MIN_SCORED_DEFAULT,
    min_spread: float = _MIN_SPREAD_DEFAULT,
) -> Tuple[bool, List[str]]:
    """Return (green, report_lines). GREEN iff the S1+ arm is a real arm:
    the run completed, the semantic blend scored frontiers, and the scores had
    enough spread to reorder them."""
    attempted = int(summary.get("n_episodes_attempted", 0) or 0)
    completed = int(summary.get("n_episodes_completed", 0) or 0)
    scored = int(summary.get("n_semantic_scored", 0) or 0)
    spread = float(summary.get("semantic_spread_max", 0.0) or 0.0)

    # Partial completion is expected on the full-val split (the runner catches
    # per-episode crashes and continues; analyze_ablation pairs only on episodes
    # present in both arms). So the completion gate only fails when the process
    # produced NOTHING — the real OOM/crash signature.
    complete_ok = completed > 0
    scored_ok = scored >= min_scored
    spread_ok = spread >= min_spread
    green = complete_ok and scored_ok and spread_ok

    lines = [
        f"  COMPLETE  {'ok ' if complete_ok else 'FAIL'} — "
        f"{completed}/{attempted} episodes completed"
        + ("" if complete_ok else " — NO episode completed (immediate crash/OOM)"),
        f"  SCORED    {'ok ' if scored_ok else 'FAIL'} — "
        f"n_semantic_scored={scored} (>= {min_scored}); "
        f"{'the blend ran' if scored_ok else 'VACUOUS: BLIP-2 never scored a frontier (load failed / weight off)'}",
        f"  SPREAD    {'ok ' if spread_ok else 'FAIL'} — "
        f"semantic_spread_max={spread:.4f} (>= {min_spread}); "
        f"{'the signal can reorder' if spread_ok else 'FLAT: uniform value => S1+ ranks frontiers exactly like S1 (the CLIP 0.020 failure)'}",
        f"  ==> {'GREEN: S1+ is a real arm.' if green else 'RED: S1+ is vacuous — do not quote the A/B.'}",
    ]
    return green, lines


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Semantic-frontier vacuous-arm verdict over a summary.json")
    ap.add_argument("summary", help="Path to the S1+ arm's run_hm3d_pol summary.json")
    ap.add_argument("--min-scored", type=int, default=_MIN_SCORED_DEFAULT,
                    help="Min frontiers that must get a semantic blend (default 1).")
    ap.add_argument("--min-spread", type=float, default=_MIN_SPREAD_DEFAULT,
                    help="Min semantic_spread_max to count as non-flat (default 0.01).")
    args = ap.parse_args(argv)

    with open(args.summary) as f:
        summary = json.load(f)
    green, lines = evaluate_semantic_arm(
        summary, min_scored=args.min_scored, min_spread=args.min_spread)
    print("semantic-frontier arm verdict")
    for l in lines:
        print(l)
    print(f"SEMANTIC_ARM_RESULT={'GREEN' if green else 'RED'}")
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
