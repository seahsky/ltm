"""Pure eval-scoring helpers — no torch / faiss / habitat imports.

Dependency-free on purpose: the correctness-critical scoring math must be
unit-testable without a GPU host (see earshot/tests/mac/test_metrics.py) and
safe to import from a file directly, bypassing the package __init__.

Ported near-verbatim from ``embodied_memory/metrics.py``; only this pointer
moved, because the path it named is deleted by ticket 10 phase 3.

``compute_soft_spl`` is the one addition, and it is a re-derivation rather than a
port: the old tree read ``soft_spl`` off habitat-lab's own ``SoftSPL`` measure
(``episode_runner.py:2426`` reads ``step.info["softspl"]``), and habitat-lab is
deliberately not a dependency of the clean room. Task spec §6 requires soft-SPL be
computed — not headlined, but computed, because the follow-on memory effort inherits
it already wired — so the arithmetic is written out here from habitat-lab's source
with the citation, exactly as ``task/episodes.py`` did for the dataset loader.
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


def compute_soft_spl(
    *,
    dist_to_goal_final: Optional[float],
    start_end_distance: float,
    path_len_taken: float,
) -> float:
    """habitat-lab's ``SoftSPL``, re-derived from source rather than imported.

    Citation: ``habitat_lab-0.3.320250127``, ``habitat/tasks/nav/nav.py``,
    ``SoftSPL.update_metric``::

        ep_soft_success = max(0, (1 - distance_to_target / start_end_episode_distance))
        metric = ep_soft_success * (
            start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance)
        )

    Two properties worth naming, because both are easy to get wrong from the formula
    alone. ``start_end_episode_distance`` is the episode's *initial* geodesic distance
    to the nearest goal view point, and it plays two roles: the normaliser for progress
    and ``L_opt`` in the efficiency ratio. And ``agent_episode_distance`` is the
    realized **path length** — the sum of per-step Euclidean displacements — not the
    displacement from start to finish.

    Unlike ``compute_benchmark_spl`` this does **not** depend on the agent calling STOP.
    It is a progress measure, which is exactly why §6 computes it and does not headline
    it: with memory out of this build nothing consumes the delta it was the primary
    metric for.

    ``dist_to_goal_final`` is ``None`` when the goal is unreachable from the final pose,
    and scores 0.0 rather than raising — an unreachable goal is a legitimate episode
    outcome, and ``None`` reaching an arithmetic expression is the failure this signature
    exists to prevent.

    **One divergence, at a case habitat-lab divides by zero on.** With
    ``start_end_distance == 0`` the agent began on a goal view point, which this project
    has produced before (the cold-start-on-goal that ``spl_guard`` existed for). habitat-
    lab's expression is ``1 - d/0``; here it is 1.0 if the agent is still there and 0.0
    if it left, which is what the measure means at that boundary.
    """
    if dist_to_goal_final is None:
        return 0.0
    final = float(dist_to_goal_final)
    start = float(start_end_distance)
    if start <= 0.0:
        return 1.0 if final <= 0.0 else 0.0
    progress = max(0.0, 1.0 - final / start)
    return float(progress * (start / max(start, float(path_len_taken))))
