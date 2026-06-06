"""
Sanity tests for ``diagnose_propose_triggers`` — the read-only propose/rerank
cadence analyzer built for the multion-full1 third-absorbing-mode post-mortem.

Pure helpers on synthetic decisions + a report smoke on a temp run dir.

Invoke with::

    python embodied_memory/scripts/test_diagnose_propose_triggers.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose_propose_triggers as dpt  # noqa: E402


def _dec(step, xy=(5.0, 5.0), src="frontier", cid=1, dist=5.0, trigger=None):
    d = {"step_idx": step, "chosen_id": cid, "chosen_world_xy": list(xy),
         "chosen_source": src,
         "candidates": [{"id": cid, "distance_m": dist}]}
    if trigger is not None:
        d["trigger"] = trigger
    return d


def case_gap_stats_per_tick_run():
    # 6 consecutive-tick decisions then one scheduled gap.
    decs = [_dec(s) for s in (1, 2, 3, 4, 5, 6, 16)]
    g = dpt.propose_gap_stats(decs, propose_period=10)
    assert g["n_decisions"] == 7
    assert g["n_gap1"] == 5, g
    assert g["max_gap1_run"] == 5, g
    assert g["n_gap_ge_period"] == 1, g
    assert g["median_gap"] == 1, g
    print("  case_gap_stats_per_tick_run: OK")


def case_gap_stats_healthy_cadence():
    decs = [_dec(s) for s in (1, 11, 21, 31)]
    g = dpt.propose_gap_stats(decs, propose_period=10)
    assert g["n_gap1"] == 0 and g["n_gap_ge_period"] == 3, g
    print("  case_gap_stats_healthy_cadence: OK")


def case_trigger_breakdown_mixed_and_unrecorded():
    decs = [_dec(1, trigger="no_candidate"), _dec(2, trigger="reached"),
            _dec(12, trigger="scheduled"), _dec(22)]  # last: legacy run
    t = dpt.trigger_breakdown(decs)
    assert t == {"no_candidate": 1, "scheduled": 1, "reached": 1,
                 "unrecorded": 1}, t
    print("  case_trigger_breakdown_mixed_and_unrecorded: OK")


def case_churn_same_pick_and_near():
    decs = [_dec(1, xy=(2.0, 2.0), dist=0.3), _dec(2, xy=(2.1, 2.0), dist=0.3),
            _dec(3, xy=(8.0, 8.0), dist=6.0)]
    c = dpt.chosen_churn_stats(decs, near_m=0.5, round_to=0.5)
    # first two quantize to the same 0.5-grid cell -> 1 same-as-prev repeat
    assert c["n_same_as_prev"] == 1, c
    assert c["n_near_chosen"] == 2, c
    assert c["sources"] == {"frontier": 3}, c
    print("  case_churn_same_pick_and_near: OK")


def case_classify_modes():
    healthy = dpt.propose_gap_stats([_dec(s) for s in (1, 11, 21)], 10)
    churn0 = dpt.chosen_churn_stats([])
    assert "healthy" in dpt.classify_mode(healthy, churn0, 749)

    perticks = [_dec(s, xy=(2.0, 2.0)) for s in range(1, 400)]
    g = dpt.propose_gap_stats(perticks, 10)
    c = dpt.chosen_churn_stats(perticks)
    v = dpt.classify_mode(g, c, 749)
    assert v.startswith("ABSORBING") and "SAME pick" in v, v

    pingpong = [_dec(s, xy=((2.0, 2.0) if s % 2 else (8.0, 8.0)))
                for s in range(1, 400)]
    g = dpt.propose_gap_stats(pingpong, 10)
    c = dpt.chosen_churn_stats(pingpong)
    v = dpt.classify_mode(g, c, 749)
    assert v.startswith("ABSORBING") and "ping-pong" in v, v
    print("  case_classify_modes: OK")


def case_report_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        ep = {"episode_idx": 0, "n_steps": 749, "rerank_calls": 3,
              "target_categories": ["bed", "chair", "toilet"],
              "n_propose_reached": 0, "n_candidates_filtered_near": 0,
              "n_waypoint_reached": 0, "n_waypoint_unreachable": 0,
              "decisions": [_dec(1, trigger="no_candidate"),
                            _dec(11, trigger="scheduled"),
                            _dec(21, trigger="scheduled")]}
        with open(os.path.join(tmp, "episode_000.json"), "w") as f:
            json.dump(ep, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dpt.main([tmp])
        out = buf.getvalue()
    assert rc == 0
    assert "gap1=" in out and "triggers:" in out and "->" in out, out
    assert "no_candidate=1" in out and "scheduled=2" in out, out
    print("  case_report_smoke: OK")


def main() -> int:
    print("diagnose_propose_triggers sanity tests")
    case_gap_stats_per_tick_run()
    case_gap_stats_healthy_cadence()
    case_trigger_breakdown_mixed_and_unrecorded()
    case_churn_same_pick_and_near()
    case_classify_modes()
    case_report_smoke()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
