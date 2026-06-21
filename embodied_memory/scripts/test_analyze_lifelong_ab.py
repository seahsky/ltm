"""
TDD for analyze_lifelong_ab pure helpers (no filesystem). Loaded standalone.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_analyze_lifelong_ab.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ll = _load("_lifelong_ab_under_test", _THIS / "analyze_lifelong_ab.py")


def _ep(idx, onset, soft, succ, writes=0, recalled=0, reason=None):
    return {"episode_idx": idx, "n_audio_onset_fired": onset, "soft_spl": soft,
            "success_1m": succ, "n_audio_writes": writes,
            "n_audio_event_recalled": recalled, "audio_write_skip_reason": reason}


def case_split_seed_recall():
    eps = [_ep(0, 1, 0.2, False), _ep(1, 0, 0.4, True), _ep(2, 0, 0.1, False)]
    seed, recall = ll.split_seed_recall(eps)
    assert len(seed) == 1 and seed[0]["episode_idx"] == 0
    assert len(recall) == 2
    print("  case split_seed_recall: OK")


def case_arm_summary_rolls_up_seed_and_recall():
    # seed wrote once + recalled twice across recall episodes
    eps = [_ep(0, 1, 0.0, False, writes=1, reason="ok"),
           _ep(1, 0, 0.5, True, recalled=1), _ep(2, 0, 0.3, False, recalled=1)]
    s = ll.arm_summary({"episodes": eps})
    assert s["seed_writes"] == 1 and s["seed_skip_reason"] == "ok"
    assert s["n_recall"] == 2 and s["recall_recalled"] == 2
    assert abs(s["recall_soft_spl"] - 0.4) < 1e-9
    assert abs(s["recall_succ1m"] - 0.5) < 1e-9
    print("  case arm_summary_rolls_up_seed_and_recall: OK")


def case_paired_recall_delta():
    a = [_ep(0, 1, 0.0, False), _ep(1, 0, 0.2, False), _ep(2, 0, 0.1, False)]
    b = [_ep(0, 1, 0.0, False), _ep(1, 0, 0.5, True), _ep(2, 0, 0.1, False)]
    dsoft, dsucc = ll.paired_recall_delta(a, b)
    assert dsoft == [0.3, 0.0], dsoft           # 0.5-0.2, 0.1-0.1
    assert dsucc == [1.0, 0.0], dsucc           # True-False, False-False
    print("  case paired_recall_delta: OK")


def case_verdict_no_write():
    assert ll.redundancy_verdict(0, 0, []).startswith("NO-WRITE")
    print("  case verdict_no_write: OK")


def case_verdict_not_recalled():
    assert ll.redundancy_verdict(1, 0, [0.1]).startswith("WRITE-NOT-RECALLED")
    print("  case verdict_not_recalled: OK")


def case_verdict_redundant_helps_hurts():
    assert ll.redundancy_verdict(1, 2, [0.0, 0.01]).startswith("REDUNDANT")   # within tie band
    assert ll.redundancy_verdict(1, 2, [0.2, 0.3]).startswith("HELPS")
    assert ll.redundancy_verdict(1, 2, [-0.2, -0.3]).startswith("HURTS")
    print("  case verdict_redundant_helps_hurts: OK")


def case_bootstrap_ci_brackets_mean():
    st = ll.bootstrap_stats([0.1, 0.2, 0.3, 0.2, 0.15], iters=500, seed=1)
    assert st["n"] == 5
    assert st["lo"] <= st["mean"] <= st["hi"], st
    print("  case bootstrap_ci_brackets_mean: OK")


def case_bootstrap_deterministic():
    a = ll.bootstrap_stats([0.1, -0.2, 0.3, 0.0], iters=500, seed=7)
    b = ll.bootstrap_stats([0.1, -0.2, 0.3, 0.0], iters=500, seed=7)
    assert a == b, (a, b)
    print("  case bootstrap_deterministic: OK")


def case_bootstrap_all_positive_excludes_zero():
    st = ll.bootstrap_stats([0.2, 0.25, 0.3, 0.22, 0.28, 0.26], iters=1000, seed=3)
    assert st["lo"] > 0.0, st          # CI excludes 0
    assert st["p"] < 0.10, st          # significant two-sided
    print("  case bootstrap_all_positive_excludes_zero: OK")


def case_bootstrap_symmetric_straddles_zero():
    st = ll.bootstrap_stats([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3], iters=1000, seed=3)
    assert st["lo"] < 0.0 < st["hi"], st
    print("  case bootstrap_symmetric_straddles_zero: OK")


def case_bootstrap_empty_and_singleton():
    e = ll.bootstrap_stats([], iters=10)
    assert e["n"] == 0 and math.isnan(e["mean"])
    s = ll.bootstrap_stats([0.4], iters=10)
    assert s["n"] == 1 and s["lo"] == s["hi"] == 0.4
    print("  case bootstrap_empty_and_singleton: OK")


def case_leave_one_cell_out_range():
    # two flat cells + one big cell: dropping the big cell pulls the pooled mean to 0
    band = ll.leave_one_cell_out([[0.0, 0.0], [0.0, 0.0], [0.6, 0.6]])
    lo, hi = band
    assert abs(lo - 0.0) < 1e-9, band     # drop the big cell -> 0
    assert abs(hi - 0.3) < 1e-9, band     # drop a flat cell -> (0,0,0.6,0.6)/4
    # <2 cells with pairs -> nan band
    one = ll.leave_one_cell_out([[0.1, 0.2]])
    assert math.isnan(one[0]) and math.isnan(one[1])
    print("  case leave_one_cell_out_range: OK")


def main() -> int:
    cases = [
        case_split_seed_recall,
        case_arm_summary_rolls_up_seed_and_recall,
        case_paired_recall_delta,
        case_verdict_no_write,
        case_verdict_not_recalled,
        case_verdict_redundant_helps_hurts,
        case_bootstrap_ci_brackets_mean,
        case_bootstrap_deterministic,
        case_bootstrap_all_positive_excludes_zero,
        case_bootstrap_symmetric_straddles_zero,
        case_bootstrap_empty_and_singleton,
        case_leave_one_cell_out_range,
    ]
    print(f"running {len(cases)} analyze_lifelong_ab cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
