"""
TDD for the pure helpers of diagnose_normal_anomaly_calib (best_threshold /
verdict). Loaded standalone (no scipy/CLAP needed for these) — same pattern as
test_audio.py.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_diagnose_normal_anomaly_calib.py
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


cal = _load("_calib_under_test", _THIS / "diagnose_normal_anomaly_calib.py")


def case_perfect_separation():
    s = cal.best_threshold(anom_margins=[0.30, 0.25, 0.40], benign_margins=[-0.10, 0.00, 0.05])
    assert s["ok"] and s["perfect"] is True, s
    assert s["youden"]["tpr"] == 1.0 and s["youden"]["fpr"] == 0.0
    # recommended delta sits in the gap (max benign 0.05 .. min anom 0.25)
    assert 0.05 < s["recommend_delta"] < 0.25, s["recommend_delta"]
    assert s["eer"] == 0.0
    v, _ = cal.verdict(s)
    assert v == "GO", v
    print("  case perfect_separation: OK")


def case_overlap_borderline():
    # one benign clip leaks above one anomaly clip -> not perfect, small EER
    s = cal.best_threshold(anom_margins=[0.10, 0.20, 0.30, 0.40],
                           benign_margins=[-0.10, 0.00, 0.05, 0.15])
    assert s["ok"] and s["perfect"] is False, s
    assert 0.0 < s["eer"] <= 0.30, s["eer"]
    v, _ = cal.verdict(s)
    assert v in ("GO", "BORDERLINE"), v
    print("  case overlap_borderline: OK")


def case_inseparable_stop():
    # fully interleaved -> high EER -> STOP
    s = cal.best_threshold(anom_margins=[0.0, 0.1, 0.2], benign_margins=[0.05, 0.15, 0.25])
    assert s["ok"] and not s["perfect"]
    v, _ = cal.verdict(s)
    assert v in ("BORDERLINE", "STOP"), (v, s["eer"])
    # the worst case (anom <= benign everywhere) must be STOP
    s2 = cal.best_threshold(anom_margins=[0.0, 0.05], benign_margins=[0.30, 0.40])
    v2, _ = cal.verdict(s2)
    assert v2 == "STOP", (v2, s2)
    print("  case inseparable_stop: OK")


def case_empty_group_not_ok():
    s = cal.best_threshold(anom_margins=[], benign_margins=[0.1])
    assert s["ok"] is False
    v, _ = cal.verdict(s)
    assert v == "STOP"
    print("  case empty_group_not_ok: OK")


def case_youden_breaks_ties_low():
    # two thresholds give the same J; the lower delta is chosen
    s = cal.best_threshold(anom_margins=[0.5, 0.6], benign_margins=[0.0, 0.1])
    assert s["youden"]["delta"] <= 0.5
    print("  case youden_breaks_ties_low: OK")


def main() -> int:
    cases = [
        case_perfect_separation,
        case_overlap_borderline,
        case_inseparable_stop,
        case_empty_group_not_ok,
        case_youden_breaks_ties_low,
    ]
    print(f"running {len(cases)} calib cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
