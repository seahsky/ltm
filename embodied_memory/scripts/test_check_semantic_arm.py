"""
TDD for the R1 semantic-frontier vacuous-arm verdict (check_semantic_arm).

R1 (Table 1) A/Bs S1 (geometric frontier) vs S1+ (BLIP-2 ITM frontier), both
memory-off, on the full HM3D val split. The S1+ arm is only a real arm if the
semantic signal actually reordered frontiers. Two ways it silently does not:

  * NEVER SCORED — BLIP-2 failed to load or the weight was off, so no frontier
    got a semantic blend. n_semantic_scored == 0.
  * SCORED BUT FLAT — every frontier read the same semantic value, so raw_score
    is a uniform rescale of geom_score and the ranking is identical to S1. This
    is the CLIP flatness measured at 0.020 three times; semantic_spread_max ~ 0.

Either makes S1+ byte-equivalent to S1 while the run still exits 0, so the driver
must FATAL on both — the same discipline the anomaly driver already applies to a
vacuous query-expansion arm. This verdict is the testable core the driver calls,
so the FATAL rule is not buried in bash. GREEN iff the run completed AND the arm
scored AND the scores had spread.

Run: PYTHONPATH=. /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_check_semantic_arm.py
"""
from __future__ import annotations

import sys

from embodied_memory.scripts.check_semantic_arm import evaluate_semantic_arm


def _live_summary(**over):
    s = {
        "n_episodes_attempted": 200,
        "n_episodes_completed": 200,
        "n_semantic_scored": 4096,
        "semantic_spread_max": 0.21,
    }
    s.update(over)
    return s


def case_green_when_completed_scored_and_spread():
    ok, lines = evaluate_semantic_arm(_live_summary())
    text = "\n".join(lines)
    assert ok is True, text
    assert "GREEN" in text, text
    print("  case_green_when_completed_scored_and_spread: OK")


def case_never_scored_is_red():
    # BLIP-2 never loaded / weight off — the arm did not fire at all.
    ok, lines = evaluate_semantic_arm(_live_summary(n_semantic_scored=0))
    assert ok is False, lines
    assert any("scor" in l.lower() and ("fail" in l.lower() or "vacuous" in l.lower())
               for l in lines), lines
    print("  case_never_scored_is_red: OK")


def case_scored_but_flat_is_red():
    # The CLIP-flatness signature: many frontiers scored, zero spread => a uniform
    # rescale => S1+ ranks frontiers exactly like S1 => vacuous A/B.
    ok, lines = evaluate_semantic_arm(_live_summary(semantic_spread_max=0.0))
    assert ok is False, lines
    assert any("spread" in l.lower() and ("fail" in l.lower() or "flat" in l.lower())
               for l in lines), lines
    print("  case_scored_but_flat_is_red: OK")


def case_partial_completion_is_still_green():
    # Full-val runs (20 scenes, ~thousands of episodes) lose a handful to
    # off-navmesh crashes; the runner catches them and continues. Partial
    # completion is normal and NOT the semantic arm's concern — analyze_ablation
    # pairs only on episodes present in both arms, so a dropped episode drops its
    # pair. A non-vacuous arm with 180/200 completed is still a real arm.
    ok, lines = evaluate_semantic_arm(_live_summary(n_episodes_completed=180))
    assert ok is True, lines
    print("  case_partial_completion_is_still_green: OK")


def case_zero_completed_is_red():
    # A process that produced no episode at all (immediate OOM / crash) is the
    # real completion failure.
    ok, lines = evaluate_semantic_arm(_live_summary(n_episodes_completed=0))
    assert ok is False, lines
    assert any("complet" in l.lower() or "no episode" in l.lower() for l in lines), lines
    print("  case_zero_completed_is_red: OK")


def case_spread_threshold_is_configurable():
    # A tiny nonzero spread below the floor still reads as flat.
    ok, _ = evaluate_semantic_arm(_live_summary(semantic_spread_max=0.001),
                                  min_spread=0.01)
    assert ok is False
    ok2, _ = evaluate_semantic_arm(_live_summary(semantic_spread_max=0.05),
                                   min_spread=0.01)
    assert ok2 is True
    print("  case_spread_threshold_is_configurable: OK")


def main() -> int:
    print("running check_semantic_arm verdict tests…")
    case_green_when_completed_scored_and_spread()
    case_never_scored_is_red()
    case_scored_but_flat_is_red()
    case_partial_completion_is_still_green()
    case_zero_completed_is_red()
    case_spread_threshold_is_configurable()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
