"""Pure eval-scoring helpers — no torch / faiss / habitat imports.

Dependency-free on purpose: the correctness-critical scoring math must be
unit-testable without a GPU host (see scripts/test_metrics.py) and safe to
import from a file directly, bypassing the package __init__.
"""
from __future__ import annotations

from typing import Optional, Tuple


def compute_benchmark_spl(
    *,
    stopped: bool,
    dist_at_stop: Optional[float],
    geodesic_optimal: float,
    path_len_taken: float,
    success_radius: float = 1.0,
) -> Tuple[bool, float]:
    """Benchmark-standard ObjectNav SPL at ``success_radius`` (default 1.0 m).

    The cross-quotable number R1 (Table 1) reports against VLFM's SPL 0.304 and
    VLingNav's 0.429 (ADR-0005). Distinct from the harness's native ``spl``
    (scored at the 0.1 m ring, localization-bound) and from ``success_1m`` (a
    STOP-INDEPENDENT reach diagnostic).

        success = the agent CALLED STOP within ``success_radius`` geodesic of a
                  goal viewpoint.  (STOP-gating is what separates this from
                  success_1m, which only asks whether the path ever came close.)
        spl     = success * L_opt / max(L_taken, L_opt).

    Args:
        stopped: the episode ended because the agent emitted ACTION_STOP, not a
            step-budget timeout.
        dist_at_stop: geodesic distance (m) to the nearest goal viewpoint at the
            STOP pose, or None when unavailable.
        geodesic_optimal: L_opt — the start->goal shortest-path length (m).
        path_len_taken: L_taken — the agent's realized path length (m).
        success_radius: benchmark success ring (m); match VLFM's ring (ADR-0005).

    Returns:
        (benchmark_success, benchmark_spl).
    """
    success = bool(
        stopped
        and dist_at_stop is not None
        and float(dist_at_stop) <= float(success_radius)
    )
    if not success:
        return False, 0.0
    denom = max(float(path_len_taken), float(geodesic_optimal))
    if denom <= 0.0:
        # Started on the goal viewpoint and stopped there: L_opt == L_taken == 0.
        return True, 1.0
    return True, float(geodesic_optimal) / denom
