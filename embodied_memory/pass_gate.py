"""Pure run-verdict helpers — no torch / faiss / habitat imports.

Which per-run pass-conditions gate the process exit code, per ablation setting.
Dependency-free so the exit-code rule is unit-testable without a GPU host
(scripts/test_pass_conditions.py) and importable directly, bypassing the package
__init__.
"""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

# Memory-liveness conditions: they can only PASS when the LTM stack is exercised.
# On a memory-OFF run they can only FAIL, so requiring them turns every healthy
# baseline into exit 1 — a false ❌ that trains the operator to ignore ❌ on
# baselines and, with it, real crashes (alarm fatigue; grilling 2026-07-17).
_MEMORY_CONDITIONS: Tuple[str, ...] = (
    "fine_layer_nonempty",
    "rerank_retrieves_always",
    "memory_influences_at_least_once",
    "all_four_modules_invoked",
)
_ALWAYS: Tuple[str, ...] = ("no_crash",)

MEMORY_OFF_SETTING = 1


def required_pass_conditions(setting: Optional[int]) -> Tuple[str, ...]:
    """The pass-conditions that gate the exit code for this ablation setting.

    Setting 1 is memory-OFF by construction (S1 / S1+), so a run is healthy iff
    it did not crash. Every other setting — including unknown/None and the
    STM-only setting 2 — keeps the strict full set, so this can never silently
    loosen a run that is meant to exercise memory.
    """
    if setting == MEMORY_OFF_SETTING:
        return _ALWAYS
    return _MEMORY_CONDITIONS + _ALWAYS


def run_passed(
    pass_conditions: Mapping[str, bool],
    setting: Optional[int],
    no_strict_pass: bool = False,
) -> bool:
    """Whether the process should exit 0, given the honest per-condition dict.

    The per-condition booleans stay honest (a memory-off run still reports its
    empty fine layer as FAIL); only the SET that gates the exit code is
    setting-aware.
    """
    if no_strict_pass:
        return True
    return all(
        bool(pass_conditions.get(k, False))
        for k in required_pass_conditions(setting)
    )
