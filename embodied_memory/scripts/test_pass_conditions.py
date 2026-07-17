"""
TDD for the setting-aware run-verdict gate (pass_gate).

A --setting 1 run is memory-OFF (S1 / S1+), so the memory-liveness pass-
conditions can only ever FAIL. Gating the process exit code on them turns every
healthy baseline into ❌ exit 1 — the false alarm that surfaced on the r1nav-s1
smoke (30/30 completed, no crash, still exit 1). The fix: a memory-off run is
healthy iff it did not crash; every memory-ON setting keeps the strict full set.

Run: PYTHONPATH=. python embodied_memory/scripts/test_pass_conditions.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embodied_memory.pass_gate import required_pass_conditions, run_passed  # noqa: E402


def _memory_off_dict(no_crash: bool = True):
    # The exact shape r1nav-s1 produced: every memory gate False, no_crash True.
    return {
        "fine_layer_nonempty": False,
        "rerank_retrieves_always": False,
        "memory_influences_at_least_once": False,
        "all_four_modules_invoked": False,
        "no_crash": no_crash,
    }


def _memory_on_dict(all_true: bool = True):
    v = all_true
    return {
        "fine_layer_nonempty": v,
        "rerank_retrieves_always": v,
        "memory_influences_at_least_once": v,
        "all_four_modules_invoked": v,
        "no_crash": True,
    }


def case_setting1_requires_only_no_crash():
    assert required_pass_conditions(1) == ("no_crash",)
    print("  case_setting1_requires_only_no_crash: OK")


def case_setting1_healthy_baseline_passes():
    # The r1nav-s1 regression: this MUST pass now.
    assert run_passed(_memory_off_dict(no_crash=True), setting=1) is True
    print("  case_setting1_healthy_baseline_passes: OK")


def case_setting1_crash_still_fails():
    assert run_passed(_memory_off_dict(no_crash=False), setting=1) is False
    print("  case_setting1_crash_still_fails: OK")


def case_setting3_keeps_full_strict_set():
    assert set(required_pass_conditions(3)) == {
        "fine_layer_nonempty", "rerank_retrieves_always",
        "memory_influences_at_least_once", "all_four_modules_invoked", "no_crash",
    }
    assert run_passed(_memory_on_dict(all_true=True), setting=3) is True
    print("  case_setting3_keeps_full_strict_set: OK")


def case_setting3_inert_memory_fails():
    # A setting-3 run where memory never fired must still FAIL — the gate's
    # original purpose is preserved for memory-ON runs.
    assert run_passed(_memory_off_dict(no_crash=True), setting=3) is False
    print("  case_setting3_inert_memory_fails: OK")


def case_setting2_stays_strict():
    # STM-only is NOT loosened: it has no fine LTM layer, so it keeps failing
    # the memory gates exactly as before (no silent regression).
    assert "fine_layer_nonempty" in required_pass_conditions(2)
    assert run_passed(_memory_off_dict(no_crash=True), setting=2) is False
    print("  case_setting2_stays_strict: OK")


def case_unknown_setting_stays_strict():
    # None (no --setting) must not accidentally relax the gate.
    assert set(required_pass_conditions(None)) == set(required_pass_conditions(3))
    assert run_passed(_memory_off_dict(no_crash=True), setting=None) is False
    print("  case_unknown_setting_stays_strict: OK")


def case_no_strict_pass_overrides_any_setting():
    assert run_passed(_memory_off_dict(no_crash=False), setting=3, no_strict_pass=True) is True
    print("  case_no_strict_pass_overrides_any_setting: OK")


def main() -> int:
    print("running pass_gate verdict tests…")
    case_setting1_requires_only_no_crash()
    case_setting1_healthy_baseline_passes()
    case_setting1_crash_still_fails()
    case_setting3_keeps_full_strict_set()
    case_setting3_inert_memory_fails()
    case_setting2_stays_strict()
    case_unknown_setting_stays_strict()
    case_no_strict_pass_overrides_any_setting()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
