"""
TDD for the ObjectNav failure-mode buckets (diagnose_spin.classify).

r1smoke returned native SPL ~0.03 with a spin signature in the tail (250-step cap,
forward ~2, never STOP). This buckets each episode so the absolute-SPL bottleneck
is attributable: controller spin vs STOP-localization vs genuine miss.

Run: PYTHONPATH=. python embodied_memory/scripts/test_diagnose_spin.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embodied_memory.scripts.diagnose_spin import classify, summarize  # noqa: E402


def case_spin_timeout():
    # cap hit, barely moved, never stopped -> controller spin (real ep29).
    assert classify({"n_steps": 249, "action_forward": 2, "action_stop": 0,
                     "success": False}, 250) == "spin_timeout"
    print("  case_spin_timeout: OK")


def case_explore_timeout():
    # cap hit but kept translating -> not a spin, just ran out of budget (real ep26).
    assert classify({"n_steps": 249, "action_forward": 28, "action_stop": 0,
                     "success": False}, 250) == "explore_timeout"
    print("  case_explore_timeout: OK")


def case_stop_miss_is_localization():
    # called STOP but outside 0.1 m -> localization-bound, NOT spin (real ep27).
    assert classify({"n_steps": 87, "action_forward": 41, "action_stop": 1,
                     "success": False}, 250) == "stop_miss"
    print("  case_stop_miss_is_localization: OK")


def case_success_wins_over_stop():
    assert classify({"n_steps": 38, "action_forward": 22, "action_stop": 1,
                     "success": True}, 250) == "success"
    print("  case_success_wins_over_stop: OK")


def case_short_no_stop_is_other():
    # ended early without STOP and not at cap -> not classifiable as spin.
    assert classify({"n_steps": 40, "action_forward": 20, "action_stop": 0,
                     "success": False}, 250) == "other"
    print("  case_short_no_stop_is_other: OK")


def case_summarize_counts_and_reach():
    eps = [
        {"n_steps": 249, "action_forward": 2, "action_stop": 0, "success": False,
         "soft_spl": 0.05, "min_distance_to_goal": 5.0},                       # spin, no reach
        {"n_steps": 38, "action_forward": 22, "action_stop": 1, "success": True,
         "soft_spl": 0.5, "min_distance_to_goal": 0.05},                       # success, reach
    ]
    counts, mean_soft, reach = summarize(eps, 250)
    assert counts == {"spin_timeout": 1, "success": 1}, counts
    assert reach["success"] == 1.0 and reach["spin_timeout"] == 0.0
    print("  case_summarize_counts_and_reach: OK")


def main() -> int:
    print("running diagnose_spin tests…")
    case_spin_timeout()
    case_explore_timeout()
    case_stop_miss_is_localization()
    case_success_wins_over_stop()
    case_short_no_stop_is_other()
    case_summarize_counts_and_reach()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
