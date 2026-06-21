"""
TDD for check_seed_not_los pure helpers (no captioner / sim / SBERT download).
Loaded standalone.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_check_seed_not_los.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cs = _load("_check_seed_not_los_under_test", _THIS / "check_seed_not_los.py")


# ---- verdict --------------------------------------------------------------

def case_verdict_green_when_absent_and_low_cos():
    ok, reason = cs.seed_not_los_verdict("a hallway with a closed door", "bed", 0.10)
    assert ok and reason.startswith("GREEN"), reason
    print("  case verdict_green_when_absent_and_low_cos: OK")


def case_verdict_red_token_present():
    ok, reason = cs.seed_not_los_verdict("a tidy bed by the window", "bed", 0.10)
    assert not ok and "names goal token" in reason, reason
    print("  case verdict_red_token_present: OK")


def case_verdict_red_high_cos():
    # token absent (synonym) but cosine over the bar → still RED
    ok, reason = cs.seed_not_los_verdict("a place to sleep with pillows", "bed", 0.40)
    assert not ok and "goal cos" in reason, reason
    print("  case verdict_red_high_cos: OK")


def case_verdict_red_both():
    ok, reason = cs.seed_not_los_verdict("a bed", "bed", 0.50)
    assert not ok and "AND cos" in reason, reason
    print("  case verdict_red_both: OK")


def case_verdict_none_cos_is_below_bar():
    # a missing cosine must not block on the cos check (token check still applies)
    ok, _ = cs.seed_not_los_verdict("an empty corridor", "bed", None)
    assert ok
    ok2, r2 = cs.seed_not_los_verdict("a bed frame", "bed", None)
    assert not ok2 and "names goal token" in r2, r2
    print("  case verdict_none_cos_is_below_bar: OK")


def case_verdict_bar_is_inclusive():
    # cos exactly at the bar counts as high (>=)
    ok, _ = cs.seed_not_los_verdict("a sofa", "bed", 0.23, cos_bar=0.23)
    assert not ok
    ok2, _ = cs.seed_not_los_verdict("a sofa", "bed", 0.2299, cos_bar=0.23)
    assert ok2
    print("  case verdict_bar_is_inclusive: OK")


# ---- goal_query_cosine (stub encoder) -------------------------------------

def _stub_encode(vocab):
    """Map known strings to fixed vectors; everything else → zero vector."""
    def enc(texts):
        return np.asarray([vocab.get(t, [0.0, 0.0]) for t in texts], dtype=np.float64)
    return enc


def case_goal_query_cosine_aligned():
    enc = _stub_encode({"a bed": [1.0, 0.0], "there is a bed": [1.0, 0.0]})
    c = cs.goal_query_cosine("a bed", "bed", enc)
    assert abs(c - 1.0) < 1e-9, c
    print("  case goal_query_cosine_aligned: OK")


def case_goal_query_cosine_orthogonal():
    enc = _stub_encode({"a wall": [0.0, 1.0], "there is a bed": [1.0, 0.0]})
    c = cs.goal_query_cosine("a wall", "bed", enc)
    assert abs(c - 0.0) < 1e-9, c
    print("  case goal_query_cosine_orthogonal: OK")


def case_goal_query_cosine_normalizes_unnormalized():
    # unnormalized parallel vectors → cosine still 1.0
    enc = _stub_encode({"a bed": [3.0, 0.0], "there is a bed": [10.0, 0.0]})
    c = cs.goal_query_cosine("a bed", "bed", enc)
    assert abs(c - 1.0) < 1e-9, c
    print("  case goal_query_cosine_normalizes_unnormalized: OK")


def case_goal_query_cosine_zero_norm_is_zero():
    enc = _stub_encode({"there is a bed": [1.0, 0.0]})   # caption → zero vec
    c = cs.goal_query_cosine("unknown", "bed", enc)
    assert c == 0.0, c
    print("  case goal_query_cosine_zero_norm_is_zero: OK")


def main() -> int:
    cases = [
        case_verdict_green_when_absent_and_low_cos,
        case_verdict_red_token_present,
        case_verdict_red_high_cos,
        case_verdict_red_both,
        case_verdict_none_cos_is_below_bar,
        case_verdict_bar_is_inclusive,
        case_goal_query_cosine_aligned,
        case_goal_query_cosine_orthogonal,
        case_goal_query_cosine_normalizes_unnormalized,
        case_goal_query_cosine_zero_norm_is_zero,
    ]
    print(f"running {len(cases)} check_seed_not_los cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
