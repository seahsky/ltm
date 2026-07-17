"""
TDD for the benchmark-standard ObjectNav SPL helper (compute_benchmark_spl).

ADR-0005: R1 (Table 1) must report a number cross-quotable to VLFM's SPL 0.304
and VLingNav's 0.429. The harness's native `spl` is scored at the 0.1 m ring
(localization-bound) and `success_1m` is a STOP-INDEPENDENT reach diagnostic —
neither is the benchmark. Benchmark SPL is:

    success = the agent CALLED STOP within `success_radius` geodesic of a goal
              viewpoint (STOP-gated, unlike success_1m).
    spl     = success * L_opt / max(L_taken, L_opt).

This is the pure math; the runner wiring (geodesic L_opt for single-goal, the
terminal-STOP flag, dist-at-stop) is staged for the VM where habitat can verify
it end-to-end. Testing the math here means the headline number can't be silently
wrong once wired.

Run: PYTHONPATH=. python embodied_memory/scripts/test_metrics.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embodied_memory.metrics import compute_benchmark_spl  # noqa: E402


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def case_perfect_path_is_spl_one():
    ok, spl = compute_benchmark_spl(
        stopped=True, dist_at_stop=0.4, geodesic_optimal=10.0, path_len_taken=10.0)
    assert ok is True and _close(spl, 1.0), (ok, spl)
    print("  case_perfect_path_is_spl_one: OK")


def case_double_path_halves_spl():
    ok, spl = compute_benchmark_spl(
        stopped=True, dist_at_stop=0.9, geodesic_optimal=10.0, path_len_taken=20.0)
    assert ok is True and _close(spl, 0.5), (ok, spl)
    print("  case_double_path_halves_spl: OK")


def case_stop_beyond_radius_fails():
    # Within 1.0 m is the ring; a STOP at 1.5 m is not a benchmark success.
    ok, spl = compute_benchmark_spl(
        stopped=True, dist_at_stop=1.5, geodesic_optimal=10.0, path_len_taken=10.0)
    assert ok is False and _close(spl, 0.0), (ok, spl)
    print("  case_stop_beyond_radius_fails: OK")


def case_timeout_without_stop_fails_even_if_close():
    # The whole point of STOP-gating: near the goal but never called STOP.
    ok, spl = compute_benchmark_spl(
        stopped=False, dist_at_stop=0.2, geodesic_optimal=10.0, path_len_taken=12.0)
    assert ok is False and _close(spl, 0.0), (ok, spl)
    print("  case_timeout_without_stop_fails_even_if_close: OK")


def case_none_dist_fails():
    ok, spl = compute_benchmark_spl(
        stopped=True, dist_at_stop=None, geodesic_optimal=10.0, path_len_taken=10.0)
    assert ok is False and _close(spl, 0.0), (ok, spl)
    print("  case_none_dist_fails: OK")


def case_start_on_goal_is_spl_one():
    # L_opt == 0 (start already at the goal viewpoint), stopped there -> perfect.
    ok, spl = compute_benchmark_spl(
        stopped=True, dist_at_stop=0.05, geodesic_optimal=0.0, path_len_taken=0.0)
    assert ok is True and _close(spl, 1.0), (ok, spl)
    print("  case_start_on_goal_is_spl_one: OK")


def case_ratio_capped_at_one():
    # Defensive: a path shorter than the precomputed optimal cannot exceed 1.0.
    ok, spl = compute_benchmark_spl(
        stopped=True, dist_at_stop=0.3, geodesic_optimal=10.0, path_len_taken=8.0)
    assert ok is True and _close(spl, 1.0), (ok, spl)
    print("  case_ratio_capped_at_one: OK")


def case_custom_radius_tightens_ring():
    # Same STOP distance, a tighter ring (0.1 m) now fails.
    ok, spl = compute_benchmark_spl(
        stopped=True, dist_at_stop=0.5, geodesic_optimal=10.0, path_len_taken=10.0,
        success_radius=0.1)
    assert ok is False and _close(spl, 0.0), (ok, spl)
    print("  case_custom_radius_tightens_ring: OK")


def main() -> int:
    print("running compute_benchmark_spl tests…")
    case_perfect_path_is_spl_one()
    case_double_path_halves_spl()
    case_stop_beyond_radius_fails()
    case_timeout_without_stop_fails_even_if_close()
    case_none_dist_fails()
    case_start_on_goal_is_spl_one()
    case_ratio_capped_at_one()
    case_custom_radius_tightens_ring()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
