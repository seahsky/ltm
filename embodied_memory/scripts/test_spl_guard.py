"""
Sanity test for ``spl_guard.guard_zero_start_distance`` — the runtime shim that
keeps Habitat's SPL / SoftSPL measures from crashing on a zero start-to-goal
geodesic.

The revisit eval's COLD seeding episodes start the agent ON the goal viewpoint
(so the step-0 caption provably captures the goal). Habitat's SoftSPL then
evaluates ``1 - distance_to_goal / start_end_distance`` with
``start_end_distance == 0`` and raises ``ZeroDivisionError`` mid-step, aborting
the seeding run. The cold episode's SPL is never read by Gate A (warm visits
only), so the guard turns that 0/0 into ``metric = 0.0`` and the episode keeps
running and seeds memory.

Stdlib-only — exercises the guard against fake measure classes, so it runs
without habitat/faiss installed (locally AND in the RACE pre-verify).

Invoke with::

    python embodied_memory/scripts/test_spl_guard.py
"""

from __future__ import annotations

import os
import sys

# Import the module directly (top-level, not via the embodied_memory package)
# so we don't trigger the package __init__'s faiss import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import spl_guard  # noqa: E402


class _RaisingMeasure:
    """Fake SoftSPL: update_metric does the 0/0 that crashes on a cold start."""
    _metric = None

    def update_metric(self, *args, **kwargs):
        start_end_distance = 0.0
        # mirrors Habitat's `1 - d2g / start_end_distance`
        self._metric = 1.0 - 0.0 / start_end_distance
        return self._metric


class _NormalMeasure:
    """Fake SPL: a well-defined episode, no division by zero."""
    _metric = None

    def update_metric(self, *args, **kwargs):
        self._metric = 0.5
        return self._metric


# ----------------------------------------------------------------------


def case_guards_zero_division():
    cls = type("M", (_RaisingMeasure,), {})
    n = spl_guard.guard_zero_start_distance([cls])
    assert n == 1, n
    m = cls()
    m.update_metric()          # must NOT raise
    assert m._metric == 0.0, m._metric
    print("  case guards_zero_division: OK")


def case_passthrough_for_normal_episode():
    cls = type("M", (_NormalMeasure,), {})
    spl_guard.guard_zero_start_distance([cls])
    m = cls()
    assert m.update_metric() == 0.5
    assert m._metric == 0.5, m._metric
    print("  case passthrough_for_normal_episode: OK")


def case_idempotent():
    cls = type("M", (_RaisingMeasure,), {})
    assert spl_guard.guard_zero_start_distance([cls]) == 1
    # second call patches nothing (already guarded)
    assert spl_guard.guard_zero_start_distance([cls]) == 0
    m = cls()
    m.update_metric()
    assert m._metric == 0.0, m._metric
    print("  case idempotent: OK")


def case_subclass_patched_independently():
    # SoftSPL subclasses SPL and OVERRIDES update_metric. The guard flag must be
    # checked on each class's OWN __dict__, else the subclass inherits the
    # parent's flag and never gets its (distinct) update_metric wrapped.
    parent = type("Parent", (_RaisingMeasure,), {})
    child = type("Child", (parent,), {
        "update_metric": _RaisingMeasure.update_metric,  # own override
    })
    n = spl_guard.guard_zero_start_distance([parent, child])
    assert n == 2, n
    c = child()
    c.update_metric()          # subclass must be guarded too
    assert c._metric == 0.0, c._metric
    print("  case subclass_patched_independently: OK")


def main() -> int:
    print("SPL/SoftSPL zero-start-distance guard sanity tests")
    case_guards_zero_division()
    case_passthrough_for_normal_episode()
    case_idempotent()
    case_subclass_patched_independently()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
