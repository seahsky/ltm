"""Planner/host fit-smoke verdict over a run's summary.json.

Used by scripts/race-planner-fit-smoke.sh: after a 1-episode cold+warm setting-3
run on the real backbone, certify that the chosen planner/GPU config is VIABLE
before spending on a multi-hour matrix. Four criteria, all must hold (GREEN):

  (a) FIT      — no CUDA OOM / crash; every attempted episode completed.
  (b) NAVIGATE — the warm episode cleared the stall floor (n_steps > --min-steps;
                 the Phase-2 Run-2 Qwen2.5-3B regurgitation stalled at ~9 steps).
  (c) LTM FIRES— the warm visit retrieved AND chose a memory candidate
                 (n_memory_candidates>0 AND n_memory_chosen>0). This path is SBERT
                 cosine + stored position — planner-INDEPENDENT — so it is the
                 control that should pass for any viable host.
  (d) PARSEABLE— the planner emitted a parseable ANSWER (n_planner_goto +
                 n_planner_explore > 0). A too-small planner that breaks the
                 ANSWER protocol shows all-zero here and silently nulls the eval.

Pure-python over summary.json so the verdict logic is unit-testable without a GPU
(see test_check_planner_fit.py). Exit 0 = GREEN, 1 = RED.

    python embodied_memory/scripts/check_planner_fit.py runs/<tag>/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Tuple


def _warm_episodes(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Warm episodes from a revisit run. Prefer the make_revisit_smoke id label
    ('…-warm-N'); if ids are unlabeled (Habitat renumbers to '0','1',…), fall back
    to 'every episode after the first' (the builder always emits cold first)."""
    eps = summary.get("episodes") or []
    labelled = [e for e in eps if "warm" in str(e.get("episode_id", "")).lower()]
    if labelled:
        return labelled
    return eps[1:] if len(eps) > 1 else []


def _i(ep: Dict[str, Any], key: str) -> int:
    v = ep.get(key, 0)
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def evaluate_fit(summary: Dict[str, Any], min_steps: int = 20) -> Tuple[bool, List[str]]:
    """Return (green, report_lines). GREEN iff all four criteria hold for a warm
    episode. ``summary`` is a parsed run_hm3d_pol summary.json dict."""
    lines: List[str] = []

    def mark(ok: bool, name: str, detail: str) -> bool:
        lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return ok

    # (a) FIT — no OOM / crash, full completion.
    notes = summary.get("notes") or []
    notes_l = " ".join(str(n).lower() for n in notes)
    oom = "out of memory" in notes_l
    crashed = "crash" in notes_l or "traceback" in notes_l
    attempted = _i(summary, "n_episodes_attempted")
    completed = _i(summary, "n_episodes_completed")
    fit_detail = (
        f"completed {completed}/{attempted}"
        + (" — CUDA OUT OF MEMORY in notes" if oom else "")
        + ("" if not crashed else " — crash/traceback in notes")
    )
    fit_ok = mark(
        (not oom) and (not crashed) and attempted > 0 and completed == attempted,
        "fit (no OOM/crash, full completion)", fit_detail or "no episode data")

    warm = _warm_episodes(summary)
    if not warm:
        mark(False, "warm episode present", "no warm episode ran (cold-only or empty)")
        lines.append("\n  ==> RED: cannot certify — no completed warm episode to inspect.")
        return False, lines
    # Inspect the first warm episode (the smoke runs exactly one).
    w = warm[0]
    steps = _i(w, "n_steps")
    n_cand = _i(w, "n_memory_candidates")
    n_chosen = _i(w, "n_memory_chosen")
    n_goto = _i(w, "n_planner_goto")
    n_explore = _i(w, "n_planner_explore")

    nav_ok = mark(steps > min_steps, "navigate (cleared the stall floor)",
                  f"warm n_steps={steps} (need > {min_steps})")
    ltm_ok = mark(n_cand > 0 and n_chosen > 0, "LTM seam fires on warm visit",
                  f"n_memory_candidates={n_cand}, n_memory_chosen={n_chosen} (need both > 0)")
    parse_ok = mark((n_goto + n_explore) > 0, "planner emits parseable ANSWER",
                    f"n_planner_goto={n_goto} + n_planner_explore={n_explore} "
                    f"(=0 => ANSWER protocol broken / planner too small)")

    green = fit_ok and nav_ok and ltm_ok and parse_ok
    lines.append(
        f"\n  ==> {'GREEN' if green else 'RED'}: "
        + ("config is viable — proceed to the matrix."
           if green else "do NOT spend on the matrix — fix the FAIL(s) above first."))
    return green, lines


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Planner/host fit-smoke verdict over a summary.json")
    ap.add_argument("summary", help="Path to a run_hm3d_pol summary.json")
    ap.add_argument("--min-steps", type=int, default=20,
                    help="Warm episode must exceed this many steps (stall floor; default 20).")
    args = ap.parse_args(argv)

    with open(args.summary) as f:
        summary = json.load(f)
    green, lines = evaluate_fit(summary, min_steps=args.min_steps)
    print("planner/host fit-smoke verdict")
    for l in lines:
        print(l)
    return 0 if green else 1


if __name__ == "__main__":
    sys.exit(main())
