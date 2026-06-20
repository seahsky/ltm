"""
TDD for analyze_lifelong_ab pure helpers (no filesystem). Loaded standalone.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_analyze_lifelong_ab.py
"""
from __future__ import annotations

import importlib.util
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


def main() -> int:
    cases = [
        case_split_seed_recall,
        case_arm_summary_rolls_up_seed_and_recall,
        case_paired_recall_delta,
        case_verdict_no_write,
        case_verdict_not_recalled,
        case_verdict_redundant_helps_hurts,
    ]
    print(f"running {len(cases)} analyze_lifelong_ab cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
