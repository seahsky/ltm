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

``compute_source_spl`` and ``compute_dtg_source_final`` are ADR-0018's addition: the
memory matrix needs an SPL and a final-pose distance scored against the ANOMALY SOURCE,
not the primary ObjectNav goal, and neither of the two functions above can be
reparameterised into that shape (see each docstring for why). Both follow the house
pattern set by ``compute_sws`` below: an explicit ``Optional`` return, and ``None`` —
never ``0.0`` — for an input that was not measured rather than a real answer of zero.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple


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


def sws_episode(
    *,
    offset_step: Optional[int],
    n_loop_steps: int,
    source_reached_step: Optional[int],
) -> Tuple[bool, bool]:
    """``(eligible, reached_after_offset)`` for one episode. ADR-0017's SWS, per episode.

    SWS -- "success when silent" -- is the fraction of episodes the agent completes by
    reaching the goal AFTER the sounding window closed. Chen et al., CVPR 2021, §5,
    adopted **verbatim** rather than adapted, so the number stays cross-quotable against
    SAVi and SAVN-CE. ``CONTEXT.md``'s note rides with it: *avoid reporting it without SR
    beside it*.

    **Which goal.** SWS counts reaching the SOUND SOURCE, not the primary ObjectNav
    goal. ADR-0017 makes the source the find-task, and the primary find has nothing to do
    with the sounding window -- an SWS computed over ``find_sr_1m`` would be a number
    about a different task that happened to be running at the same time. There are two
    successes in this record and collapsing them silently is exactly the ambiguity
    ``CONTEXT.md`` warns about under Mission.

    **Who is in the denominator.** An episode is eligible when it RAN PAST its own offset
    step. An episode that ended first never had a silent phase and cannot answer the
    question SWS asks. An episode that ran past the offset step but reached the source
    BEFORE it is eligible and is **not** in the numerator, because it did not complete
    while the source was silent -- that is the definition's sharp edge and it is the
    whole content of the metric. A continuous-arm episode has no offset step and is never
    eligible.

    Zero eligible episodes is NOT_RUN and never 0.0 (ADR-0014), which is why
    ``compute_sws`` below returns ``Optional``.
    """
    if offset_step is None:
        return False, False
    eligible = int(offset_step) < int(n_loop_steps)
    if not eligible:
        return False, False
    reached_after = (
        source_reached_step is not None
        and int(source_reached_step) >= int(offset_step)
    )
    return True, bool(reached_after)


def post_offset_audible_steps(
    *,
    readings: Sequence[Tuple[int, float]],
    offset_step: Optional[int],
    bed_rms: Optional[float],
    tolerance: float,
) -> Optional[int]:
    """How many silent-phase steps the agent could still tell from the bed. ADR-0017.

    **The measurement ``tail_is_active`` cannot make.** That predicate reads the record
    and the record carries configuration -- a hop, a read-window width, an IR width, a
    tail length -- so it separates "a tail was configured" from "a tail ran" and stops
    there. It cannot separate either from *a tail that carried no energy*, and at the
    shipped defaults that is not hypothetical: ``audio/tail.py`` measures a 0.6 s
    transient looped on a 5 s period falling to 0.002 of its settled level on the FIRST
    silent step, which is a hard cut to the bed and is exactly the artefact ADR-0017
    commissioned the accumulator to prevent.

    The bar is ``AudioConfig.pre_onset_rms_tol``, and it is borrowed rather than invented:
    that is the tolerance §3.1 already uses to decide a reading IS the bed. A silent-phase
    step inside it is a step the agent could not distinguish from
    ``bed.heard_signal(playing=False)``.

    ``None`` is NOT_RUN and never 0 (ADR-0014). No offset step means the continuous arm
    and there is no silent phase to measure; no bed level means the record predates the
    calibration block and the comparison has no reference. Zero is a real answer and a
    loud one: **the silent phase was a hard cut**, and an SWS over such episodes is a
    number about the mechanism ADR-0017 replaced.

    **Since ADR-0019 the readings are the CUE trace, so this counts steps at which the
    ROOM was still audible** rather than steps at which the analysis window still held
    source. Its values FALL -- measured, the cue decays 0.4480 0.0358 0.0000 at the box's
    numbers where the clip readout decayed over seven steps -- and that is the correction,
    not a regression: the 5 s window was smearing a hard cut into a plausible decay. A
    zero now carries a sharper meaning than it did. Either the IR is narrower than one
    step, or the loop's last ring landed more than one fold before the offset, and
    ``SoundingWindowRecord.cue_tail_steps`` separates those two.

    A run mixing pre- and post-split episodes must never be pooled on this number, and
    ``cue_tail_steps`` being present on a record is the reliable marker of which domain
    its trace is in. It stays reported and never gated, for the reason
    ``runner.tail_is_active``'s docstring already gives: a hard cut on a transient
    recording is a fact about the clip and the loop rather than a broken accumulator.
    """
    if offset_step is None or bed_rms is None:
        return None
    bed = float(bed_rms)
    if bed <= 0.0:
        return None
    margin = bed * abs(float(tolerance))
    offset = int(offset_step)
    return sum(
        1
        for step, measured in readings
        if int(step) >= offset and abs(float(measured) - bed) > margin
    )


def compute_source_spl(
    *,
    source_reached: bool,
    dist_at_reach: Optional[float],
    geodesic_optimal_to_source: Optional[float],
    path_len_at_reach: Optional[float],
    success_radius: float = 1.0,
) -> Optional[Tuple[bool, float]]:
    """SPL against the ANOMALY SOURCE, scored at the controller's CHECK transition.

    ``compute_benchmark_spl`` cannot serve this: its ``success`` requires the primary
    ``stopped`` flag, and that is still False when the source is reached mid-episode, so
    parameterising the old function by radius would score every source find as 0.0. This
    is a new function, not a reparameterisation.

        success = source_reached and dist_at_reach is not None
                  and dist_at_reach <= success_radius
        spl     = success * L_opt / max(L_taken, L_opt)

    ``L_opt`` (``geodesic_optimal_to_source``) is the start-pose-to-source navmesh route;
    ``L_taken`` (``path_len_at_reach``) is the agent's realized path length ACCUMULATED UP
    TO THE STEP THE SOURCE WAS REACHED, not the whole episode's path length -- charging
    the source find for metres walked afterwards on the primary task would make the
    number a function of how long the primary search ran after the source was already
    found. ``dist_at_reach`` is a horizontal distance (the shape the runner already
    records at the investigation stop pose) while ``geodesic_optimal_to_source`` is a
    route length -- two axes, inherited from what the runner already computes rather than
    invented here.

    ``None`` -- NOT_RUN, never ``(False, 0.0)`` -- in exactly two cases, both real
    absences and not measured zeros (ADR-0014):

    * ``geodesic_optimal_to_source is None``: no navmesh route from the start pose to the
      source at all (23 of ``yield-2``'s 365 episodes). An unwinnable episode scored
      ``(False, 0.0)`` would sit in the same bucket as a episode that had a route and
      failed to take it, which is the exact confusion ``min_route_to_source_in_window_m``
      was added to prevent.
    * ``source_reached`` is True and either ``dist_at_reach`` or ``path_len_at_reach`` is
      ``None``: the source was reached but the measurement that would score the reach was
      not captured, which is an accounting gap and not a fact about the episode.

    ``(False, 0.0)`` is the real, reportable zero: the episode ran, had a route to the
    source, and did not reach it (or reached it outside ``success_radius``).

    ``denom <= 0.0`` returns ``(True, 1.0)`` -- started on the source -- matching
    ``compute_benchmark_spl``'s boundary case verbatim.
    """
    if geodesic_optimal_to_source is None:
        return None
    if source_reached and (dist_at_reach is None or path_len_at_reach is None):
        return None
    success = bool(
        source_reached
        and dist_at_reach is not None
        and float(dist_at_reach) <= float(success_radius)
    )
    if not success:
        return False, 0.0
    # ``path_len_at_reach`` is not None here: `success` required it above.
    denom = max(float(path_len_at_reach), float(geodesic_optimal_to_source))
    if denom <= 0.0:
        # Started on the source and reached it with zero displacement: L_opt == 0.
        return True, 1.0
    return True, float(geodesic_optimal_to_source) / denom


def compute_dtg_source_final(*, geodesic_to_source: Optional[float]) -> Optional[float]:
    """Final-pose distance-to-goal, taken against the SOURCE rather than the primary goal.

    The source-side twin of the ``dist_to_goal_final`` argument computed inline (and
    discarded) at the ``compute_soft_spl`` call site in ``runner.py`` -- a second,
    separate pathfinder query, not a re-derivation of the first.

    Pure and single-axis: it never falls back to a horizontal distance when the route is
    missing. Substituting one axis for the other is how an un-measured detour comes to
    look like a converging one (``report/audit.py``'s own warning against exactly this).

    ``None`` in (unreachable from the final pose, or the query was never made) -> ``None``
    out. Non-finite in (``inf``, ``nan``) -> ``None`` also: ``World.geodesic_distance``
    already returns ``None`` for unreachable, but a caller that seeds an accumulator with
    ``inf`` and forgets to convert it back would otherwise hand this function a value
    whose ``json.dump`` writes a bare ``Infinity`` -- not valid JSON, and the failure
    ``runner.py``'s own comment on ``min_d2source`` names.

    Raises ``ValueError`` on a negative distance -- an arithmetic bug in the caller, and
    clamping it (``max(0.0, value)``) would publish it as a real number instead.
    """
    if geodesic_to_source is None:
        return None
    value = float(geodesic_to_source)
    if not math.isfinite(value):
        return None
    if value < 0.0:
        raise ValueError(
            "geodesic_to_source cannot be negative: {}".format(value)
        )
    return value


def compute_sws(
    *,
    n_eligible: int,
    n_reached_after_offset: int,
    n_tail_active: Optional[int] = None,
) -> Optional[float]:
    """SWS over a run -- ``None`` when nothing was eligible, never 0.0.

    ``None`` is NOT_RUN and is the whole reason the return is ``Optional``. Two incidents
    are behind that rule: a probe that skipped and reported success, and a canary that
    was never armed reading as a pass. An SWS of 0.0 says *the agent never succeeded in
    silence*; an SWS over zero eligible episodes says *nobody asked*, and a reader cannot
    tell those apart from a float.

    Raises rather than clamping on impossible counts. A numerator larger than its
    denominator is an accounting bug in the caller, and the quiet fix -- ``min(a, b)`` --
    would publish a rate of 1.0 for it.

    ``n_tail_active`` is ADR-0017's bar carried INTO the primitive: how many of the
    eligible episodes had a record showing the accumulation buffer folded a real render.
    Short of the denominator and this raises, because an SWS counting an episode whose
    silent phase was a hard cut to the bed is a number about the mechanism ADR-0017
    replaced. It is ``Optional`` and not required for one reason only -- this function
    takes two ints and cannot fetch the records itself -- so a caller that HAS the
    evidence must pass it and a caller that does not is publishing an unverified rate.
    ``runner.SilentPhaseTally`` is the value that always has it and always passes it;
    reach for that rather than for this.
    """
    eligible = int(n_eligible)
    numerator = int(n_reached_after_offset)
    if eligible < 0 or numerator < 0:
        raise ValueError(
            "SWS counts cannot be negative: {} of {} eligible".format(
                numerator, eligible
            )
        )
    if numerator > eligible:
        raise ValueError(
            "SWS numerator {} exceeds its denominator {} -- more episodes reached the "
            "source after their offset step than ran past one".format(
                numerator, eligible
            )
        )
    if n_tail_active is not None and int(n_tail_active) < eligible:
        raise ValueError(
            "{} of the {} eligible episodes carry no active reverb tail, so an SWS over "
            "them would be measured on a hard cut to the bed rather than on a decaying "
            "source. ADR-0017 bars reporting an SWS before the accumulation buffer is "
            "in.".format(eligible - int(n_tail_active), eligible)
        )
    if eligible == 0:
        return None
    return float(numerator) / float(eligible)
