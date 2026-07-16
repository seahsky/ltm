"""
TDD for the semantic-frontier vacuous-arm counters.

R1 (Table 1) A/Bs S1 (geometric frontier) against S1+ (BLIP-2 ITM frontier). The
A/B is only meaningful if the semantic signal actually REORDERED frontiers in the
S1+ arm, and the existing ``semantic_frontier=True`` flag cannot show that: it is
set on every candidate whenever the weight is on, regardless of what the scorer
produced. When the value map is unobserved every frontier reads
``semantic_value=0.0``, so ``raw_score = (1-w)*geom_score`` — a uniform rescale
that preserves the geometric ranking exactly, making S1+ behave identically to S1
while every counter reads green.

Absence is not the only inert case. A CONSTANT nonzero value is equally inert,
since only spread reorders — and that is not hypothetical, it is the CLIP flatness
measured at 0.020 separation three times. Whether BLIP-2 has spread where CLIP had
none is the question R1 exists to answer, so the guard and the result are the same
measurement: without spread in ``summary.json`` the driver cannot distinguish
"BLIP-2 is flat" from "BLIP-2 never loaded", and would publish the first as the
second.

``FrontierPlanner._last_semantic_diag`` carries the per-propose spread. These
tests pin that ``RunSummary`` carries and serializes the run-level counters
(mirroring ``n_query_expanded``) so the driver's post-run guard can read them from
``summary.json``.

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_summary_semantic_frontier.py
"""
from __future__ import annotations

import sys

from embodied_memory.episode_runner import RunSummary


def case_defaults_zero_and_serialized():
    s = RunSummary()
    assert s.n_semantic_scored == 0, s.n_semantic_scored
    assert s.semantic_spread_max == 0.0, s.semantic_spread_max
    d = s.to_dict()
    assert "n_semantic_scored" in d, sorted(d.keys())
    assert "semantic_spread_max" in d, sorted(d.keys())
    assert d["n_semantic_scored"] == 0, d["n_semantic_scored"]
    assert d["semantic_spread_max"] == 0.0, d["semantic_spread_max"]
    print("  case_defaults_zero_and_serialized: OK")


def case_set_values_serialize():
    s = RunSummary()
    s.n_semantic_scored = 42
    s.semantic_spread_max = 0.31
    d = s.to_dict()
    assert d["n_semantic_scored"] == 42, d["n_semantic_scored"]
    assert abs(d["semantic_spread_max"] - 0.31) < 1e-9, d["semantic_spread_max"]
    print("  case_set_values_serialize: OK")


def case_scored_without_spread_is_the_vacuous_arm():
    # The signature the driver must FATAL on: the branch ran on many frontiers
    # (n_semantic_scored high) yet never once reordered them (spread 0.0). Both
    # fields are needed — n_scored alone reports green on an inert arm, and
    # spread alone cannot tell an inert arm from one that never proposed.
    s = RunSummary()
    s.n_semantic_scored = 128
    s.semantic_spread_max = 0.0
    d = s.to_dict()
    assert d["n_semantic_scored"] > 0 and d["semantic_spread_max"] == 0.0, d
    print("  case_scored_without_spread_is_the_vacuous_arm: OK")


def main() -> int:
    print("running semantic-frontier summary tests…")
    case_defaults_zero_and_serialized()
    case_set_values_serialize()
    case_scored_without_spread_is_the_vacuous_arm()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
