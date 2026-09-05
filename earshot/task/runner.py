"""The episode loop: where the four layers stop being modules and start being a run.

``task/`` is the only layer ADR-0013's graph lets import everything, so this is the one
place the simulator, the audio sensor, the agent and the report meet.

**Three shapes worth naming before the code.**

*The loop takes its world and its sensor as objects, not as a bundle of callables.*
``audio/`` and ``agent/`` inject callables because the layer graph forbids them from
importing ``sim`` at all; ``task/`` has no such rule, so the injection here would buy
nothing but a dozen lambdas at every call site. What it does buy — a Mac-testable loop —
is bought instead by duck typing: ``run_episode`` names only the handful of methods
``World`` and ``AudioSensorHandle`` publish, and ``tests/mac/test_task_runner.py`` drives
the whole thing with two fakes. ``run()`` is the box-only half, and it is deliberately
thin: it constructs, it loops, it writes.

*Every model is constructed eagerly at startup* (map requirement 9). Ticket 15 measured
the full stack at 5.547 GiB against 31.73 GiB usable, so there is no lazy-loading seam to
build and the layout must not grow one — a model that appears at step 200 is a stall
inside the per-step audio budget criterion 7 audits.

*Nothing may write to fd 1 or 2 concurrently.* ``guarded_observe`` captures both
descriptors around every render. ADR-0013 narrowed this from ticket 18's original
requirement: ``capture_habitat_logs`` flushes Python's buffers on entry and exit, so an
interleaved in-thread ``print()`` between steps is safe. What is forbidden is a
*concurrent* writer — a background thread, a timer-driven progress bar, a subprocess that
inherited the descriptor, a logging handler flushed off-thread. This module therefore
prints progress and starts nothing.

**The calibration is per episode, and that is a reading of §2.3 rather than a departure
from it.** The spec says ``onset_rms`` is derived "at run start", from the anomaly's RMS
distribution across the audible band. That distribution is a property of *this episode's*
source placement and *this scene's* geometry, so with one positioned source per episode
(ADR-0009) the faithful form is one sweep per source. It is measurement either way — the
gate still fails on overlap and the correction is still ``globalVolume`` — but a run of
several episodes carries several thresholds, and each lands on its own audit record with
its own separation margin.

**The sweep's renders are outside the loop, and the record says so.** Smoke criterion 1
is "render count equals step count exactly". Arming the guard costs one render and the
calibration costs ``sweep_poses`` more, all before the first step, so the criterion is
checkable on ``n_renders_in_loop`` versus ``n_loop_steps`` — both recorded — rather than
on the simulator's lifetime counter, which would fail for a reason that is not a defect.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from earshot.agent.config import PlannerConfig
from earshot.agent.controller import (
    ACT_STOP,
    CAST_STEPS,
    RISING_WINDOW,
    SCAN_STEPS,
    ControllerState,
    NavMode,
    climb_eps,
    is_diverting,
    step_controller,
)
from earshot.agent.detector import GoalDetector, OracleDetector
from earshot.agent.proposers import SOURCE_INVESTIGATE, Candidate, FrontierProposer
from earshot.agent.reachability import assert_pool, reachable_pool
from earshot.agent.scorer import pick_waypoint
from earshot.audio.bed import bed_signal
from earshot.audio.calibration import (
    CalibrationError,
    CalibrationResult,
    band_poses,
    calibrate_onset,
    sweep_cue_rms,
    sweep_loop_scatter,
    sweep_render_scatter,
)
from earshot.audio.clap import (
    audio_embedding,
    classify_anomaly,
    heard_clip_for_clap,
    is_anomaly,
    testimony_bank,
)
from earshot.audio.clips import load_anomaly_clip, render_through_ir, resolve_anomaly_clip, rms
from earshot.audio.ir import anechoic_like
from earshot.audio.lateral import lateral_sign
from earshot.audio.normality import NullRoomLabeler, RoomLabeler, is_anomalous_here
from earshot.audio.onset import OnsetState, assert_provenance, observe_step
from earshot.audio.tail import heard_clip_window, heard_step, hop_samples, open_tail
from earshot.audio.window import plan_window
from earshot.config import (
    CastPolicy,
    ClimbRule,
    Detector,
    IrPolicy,
    LateralCue,
    Localization,
    RunConfig,
)
from earshot.env_check import assert_env
from earshot.memory.store import EpisodicStore, MemoryCondition, SemanticStore
from earshot.metrics import (
    compute_benchmark_spl,
    compute_dtg_source_final,
    compute_soft_spl,
    compute_source_spl,
    compute_sws,
    post_offset_audible_steps,
    sws_episode,
)
from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_env_report, write_episode, write_run_summary
from earshot.report.audit import (
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    SoundingWindowRecord,
    StepRecord,
)
from earshot.task.dataset import AnomalyEpisode, EmptyDatasetError, build_anomaly_episodes
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene
from earshot.task.memory_build import stores_for_cell
from earshot.task.memory_prior import (
    RUN_DISCLOSURE,
    MemoryContext,
    PriorMiss,
    points_by_category_for_cell,
    resolve_prior,
)
from earshot.task.prior_build import anchor_of_run_class
from earshot.types import NoRouteError, Pose, Xyz

__all__ = [
    "EpisodeResult",
    "RunSummary",
    "SilentPhaseTally",
    "TailNotActiveError",
    "silent_phase_tally",
    "tail_is_active",
    "calibration_poses",
    "calibrate_episode",
    "ir_under_policy",
    "make_detector",
    "run_episode",
    "run",
]

# How many navmesh draws the calibration sweep samples before matching them to the
# band's target distances. Generous against `AudioConfig.sweep_poses` (16): the draws are
# free — `random_navigable_point` touches no renderer — and the cost of too few is a
# sweep whose poses cluster at whatever distances happened to come up, which is the
# distribution the threshold is then placed inside.
CALIBRATION_DRAWS = 512

# How many recent loudness readings the greedy climb keeps. DERIVED from the rule rather
# than chosen: `is_rising` compares two adjacent windows and therefore reads exactly
# `2 * RISING_WINDOW` entries, so a hand-picked constant here is a silent way to feed the
# agent a shorter baseline than its own predicate asks for — and than a replay of the
# record would reconstruct. It was 8 against a rule that needed 6; the two-sided rule
# needs 10, and nothing about the old number said so.
ENERGY_HISTORY = 2 * RISING_WINDOW

# The candidate id the oracle arm's investigate divert is injected with.
# `FrontierProposer._emit` issues ids from 1, so 0 is never a proposed candidate and the
# divert is identifiable in the audit by its id as well as by its source.
DIVERT_CANDIDATE_ID = 0


def ir_under_policy(impulse: Any, policy: IrPolicy) -> Any:
    """The impulse response the run actually convolves, under ADR-0018's ``IrPolicy``.

    A module-level function rather than a closure per call site, because there are THREE
    places an IR leaves the sensor — the calibration sweep, §2.5's start-pose audibility
    probe, and the step loop — and ADR-0017's rule is that the sweep and the loop must
    take the same path. Two closures in two functions is two places for that rule to be
    edited apart; ``calibrate_episode``'s comment already names the failure it causes,
    which is a threshold calibrated in a domain the loop does not run in.

    ``FULL`` hands the IR back untouched, so the default configuration convolves exactly
    the array the sensor produced and no arm-shaped code runs at all.
    """
    if policy is IrPolicy.FULL:
        return impulse
    return anechoic_like(impulse)


@dataclass(frozen=True)
class EpisodeResult:
    """One episode's two artefacts, before they are written."""

    report: AgentReport
    audit: EpisodeAudit


class TailNotActiveError(RuntimeError):
    """An SWS number was about to be produced for a run whose reverb tail did not run.

    ADR-0017 line 49 bars it outright: *no sounding-window run may report an SWS before
    the accumulation buffer is in*. The reason is not bookkeeping. Without the tail the
    silent phase arrives as a hard step to the bed, so the level the agent reads after
    the offset step is a different signal from the one a real room would deliver, and an
    SWS measured on it is a number about an artefact.

    This raises rather than skipping, and that is the point. The failure mode of a soft
    answer here is an SWS quietly computed over a subset -- which is the shape ADR-0014
    exists to close, and the shape ``NOT_RUN`` reads as red for.
    """


def tail_is_active(window: Optional[SoundingWindowRecord]) -> bool:
    """Did the accumulation buffer actually carry this episode's source?

    Answered from the RECORD rather than from the runner's live state. The input is a
    ``SoundingWindowRecord``, which ``EpisodeAudit.from_dict`` rebuilds out of
    ``audit.json``, so the question is *askable* of a run directory a year from now --
    but **no reader in ``tools/`` asks it today**. ``silent_phase_tally`` is the only
    caller and it runs inside the run, over audits still in memory. Saying it answers
    what a later reader asks would be the log canary's own claim: the shape is there and
    the caller is not, and the gap is a fact about this build rather than a promise.
    Every clause is evidence the buffer folded a real render:

    - ``hop_samples`` is ``round(step_seconds * sample_rate)`` and exists only because
      ``open_tail`` was called;
    - ``max_ir_samples`` is the widest IR the accumulator was HANDED, and it stays 0
      until a sounding step convolved one into the buffer -- so it is the clause that
      separates "a tail was configured" from "a tail ran";
    - ``tail_steps`` is derived from that width and is how long the source COULD have
      outlived its own offset step.

    ``analysis_window_samples`` is the CLIP readout's width and the buffer's geometry.
    The CUE readout's width is ``hop_samples``, which is already a clause above, so
    ADR-0019 added no fifth clause and no new field for it -- and ``cue_tail_steps``,
    which the split did add to the record, is deliberately NOT a clause: it is evidence
    about the ROOM, not evidence that the buffer folded a render.

    An old record (``sounding_window is None``) answers False, which is correct: absent
    is unknown, and unknown is not evidence the tail was there.

    **What it cannot answer, said here so nobody reads it as more than it is.** Every
    clause above is CONFIGURATION. A buffer can be correctly built, correctly handed a
    real IR, and still carry no energy past the offset step -- ``audio/tail.py`` measures
    a transient clip falling to 0.002 of its settled level on the first silent step, and
    this predicate returns True for it. That is what
    ``SoundingWindowRecord.post_offset_audible_steps`` measures and what
    ``SilentPhaseTally`` prints beside every SWS. This one raises; that one is reported,
    because a hard cut on a transient recording is a fact about the clip and the loop
    rather than a broken accumulator, and refusing to run would be answering ADR-0017's
    open question by making four of ESC-50's five classes unusable.

    All four clauses are exercised in ``test_task_runner.py`` -- individually, with a
    record built to fail exactly one. Three of them asserted nothing when this shipped.
    """
    if window is None:
        return False
    return bool(
        window.hop_samples is not None
        and int(window.hop_samples) > 0
        and window.analysis_window_samples is not None
        and int(window.analysis_window_samples) > 0
        and window.max_ir_samples is not None
        and int(window.max_ir_samples) > 0
        and window.tail_steps is not None
        and int(window.tail_steps) > 0
    )


@dataclass(frozen=True)
class SilentPhaseTally:
    """ADR-0017's run-level counts: SWS, and SR structurally beside it.

    ``n_source_reached`` is a FIELD of this type rather than something a caller is
    trusted to fetch, because ``CONTEXT.md``'s note on SWS is *"avoid reporting it
    without SR beside it"* and a convention held by good intentions is a convention that
    is broken by the next reader in a hurry. A type that cannot express one without the
    other is the version of that rule the code can hold.

    ``n_window_closed`` is SWS's denominator and it is not ``n_episodes``: an episode
    that ended before its own offset step never had a silent phase and cannot answer the
    question SWS asks. An SWS over zero eligible episodes is **NOT_RUN, never 0.0**
    (ADR-0014) -- ``sws`` returns ``None`` there and ``as_dict`` emits an explicit
    ``sws_status`` beside the null, so a JSON reader cannot read the absence as a zero.

    ``n_tail_audible`` rides along for the same structural reason and is the second thing
    an SWS must never be read without. ``tail_is_active`` proves the accumulator was
    BUILT; only this counts the eligible episodes whose silent phase the agent could
    still tell from the bed. ``audio/tail.py`` measures a transient clip cutting to the
    bed in one step at the shipped defaults, and an SWS over such episodes is a number
    about the mechanism ADR-0017 replaced. A denominator with a zero here is a red flag
    in the artefact rather than an exception in the log.

    **``n_tail_active`` is ADR-0017's bar carried as a FIELD, and that is the difference
    between a rule and a habit.** The bar used to live at the two call sites that happen
    to exist today -- this module's tally and ``run_episode`` -- while this type's
    constructor took bare counts and asked nothing. A later cross-run aggregator walking
    ``audit.json`` files and building one of these would have published an SWS having
    never asked whether a single accumulator folded a render, and nothing would have
    stopped it. Now the type refuses: a tally that STATES fewer active tails than its own
    denominator cannot be constructed at all.

    What it deliberately does not do is refuse ``None``. This value is also built by
    hand -- by a summary printer, by a test, by a reader who has counts and no records --
    and forcing a fabricated number there would buy a false claim rather than a check. So
    ``None`` is *unverified* and says so in the artefact: ``sws_status`` reads
    ``measured_tail_unverified`` and the printed line carries the same words. An
    unverified SWS is legible as unverified; an unverifiABLE one raises.
    """

    n_episodes: int
    # SWS's denominator: episodes that ran past their own offset step.
    n_window_closed: int
    # SWS's numerator: of those, the ones that reached the source at or after it.
    n_reached_after_offset: int
    # Anomaly-response SR's numerator, over every episode. Never optional.
    n_source_reached: int
    # Of the eligible episodes, how many had a MEASURED audible tail. Defaulted so a
    # tally built by an older caller still constructs; `silent_phase_tally` always fills
    # it. See `SoundingWindowRecord.post_offset_audible_steps`.
    n_tail_audible: int = 0
    # Of the eligible episodes, how many carried an ACTIVE accumulator (`tail_is_active`).
    # `None` is "nobody asked", which the artefact prints rather than hides.
    n_tail_active: Optional[int] = None

    def __post_init__(self) -> None:
        """The bar, at construction, so no path reaches ``sws`` with the counts wrong.

        Checked here rather than in ``sws`` because a value that cannot be published is a
        value that should not exist: ``as_dict``, ``summary`` and every future reader go
        through this constructor, and a property-side check would have to be repeated in
        each of them -- which is the two-call-site arrangement this field replaces.
        """
        if self.n_tail_active is None:
            return
        if int(self.n_tail_active) < int(self.n_window_closed):
            raise TailNotActiveError(
                "{} of the {} eligible episodes carry no active reverb tail, so this "
                "tally cannot express an SWS over them: it would be measured on a hard "
                "cut to the bed rather than on a decaying source. ADR-0017 bars "
                "reporting an SWS before the accumulation buffer is in.".format(
                    int(self.n_window_closed) - int(self.n_tail_active),
                    int(self.n_window_closed),
                )
            )

    @property
    def tail_verified(self) -> bool:
        """Was the accumulator's presence CHECKED for every episode in the denominator?

        False means unverified, never "the tail was missing" -- a missing tail is
        unconstructible above. It is what separates ``measured`` from
        ``measured_tail_unverified`` in the artefact.
        """
        return self.n_tail_active is not None

    @property
    def sws(self) -> Optional[float]:
        """Success when silent, or ``None`` when no episode was eligible."""
        return compute_sws(
            n_eligible=int(self.n_window_closed),
            n_reached_after_offset=int(self.n_reached_after_offset),
            n_tail_active=self.n_tail_active,
        )

    @property
    def anomaly_response_sr(self) -> Optional[float]:
        """Source reaches over episodes -- ``None`` on a run with no episodes."""
        if int(self.n_episodes) <= 0:
            return None
        return float(self.n_source_reached) / float(self.n_episodes)

    def as_dict(self) -> Dict[str, Any]:
        value = self.sws
        return {
            "n_episodes": int(self.n_episodes),
            "n_window_closed": int(self.n_window_closed),
            "n_reached_after_offset": int(self.n_reached_after_offset),
            "n_source_reached": int(self.n_source_reached),
            "n_tail_audible": int(self.n_tail_audible),
            "n_tail_active": (
                None if self.n_tail_active is None else int(self.n_tail_active)
            ),
            "sws": None if value is None else float(value),
            # The null above and this string say the same thing twice on purpose. A
            # reader that treats a missing number as zero produces a 0.0 SWS for a run
            # nobody measured, and that is the exact confusion `CriterionStatus.NOT_RUN`
            # was introduced for. The third value is the same idea one step weaker: the
            # rate is real, and nothing checked that the accumulator was behind it.
            "sws_status": (
                "not_run"
                if value is None
                else ("measured" if self.tail_verified else "measured_tail_unverified")
            ),
            "anomaly_response_sr": self.anomaly_response_sr,
        }


def silent_phase_tally(audits: Sequence[EpisodeAudit]) -> SilentPhaseTally:
    """Count ADR-0017's four numbers over a run's episode records. Pure.

    **Raises ``TailNotActiveError`` on an eligible episode whose record shows no tail.**
    That is ADR-0017's bar held in code rather than in a comment: the only path from an
    episode to an SWS numerator runs through here, and it refuses to count an episode
    whose accumulation buffer never folded a render. A run of such episodes therefore
    produces no SWS at all -- not a 0.0, not a partial rate over the episodes that did.

    ``n_tail_audible`` is counted here rather than re-derived by a reader, off the
    per-episode ``post_offset_audible_steps`` the runner measured. An episode whose record
    predates that field contributes nothing to it, which is why it is reported as a count
    beside the denominator and not as a rate: ``0 of 20`` and ``0 of 0`` are different
    facts and a rate would collapse them.

    ``n_tail_active`` goes onto the returned value because this function is the thing
    that asked. The raise above already refuses the bad case, so the count it hands over
    always equals the denominator -- what it buys is that the VALUE says the check
    happened, and a tally assembled anywhere else says the opposite.
    """
    n_episodes = 0
    n_window_closed = 0
    n_reached_after_offset = 0
    n_source_reached = 0
    n_tail_audible = 0
    n_tail_active = 0
    for audit in audits:
        n_episodes += 1
        if audit.funnel_stage >= FunnelStage.SOURCE_REACHED:
            n_source_reached += 1
        window = audit.sounding_window
        eligible, reached_after = sws_episode(
            offset_step=None if window is None else window.offset_step,
            n_loop_steps=len(audit.steps),
            source_reached_step=audit.source_reached_step,
        )
        if not eligible:
            continue
        if not tail_is_active(window):
            raise TailNotActiveError(
                "episode {} ran past its offset step but its record carries no active "
                "reverb tail ({!r}), so an SWS counting it would be measured on a hard "
                "cut to the bed rather than on a decaying source. ADR-0017 bars "
                "reporting an SWS before the accumulation buffer is in.".format(
                    audit.episode_index, window
                )
            )
        n_window_closed += 1
        n_tail_active += 1
        if reached_after:
            n_reached_after_offset += 1
        audible = window.post_offset_audible_steps
        if audible is not None and int(audible) > 0:
            n_tail_audible += 1
    return SilentPhaseTally(
        n_episodes=n_episodes,
        n_window_closed=n_window_closed,
        n_reached_after_offset=n_reached_after_offset,
        n_source_reached=n_source_reached,
        n_tail_audible=n_tail_audible,
        n_tail_active=n_tail_active,
    )


@dataclass(frozen=True)
class RunSummary:
    """What a whole run reached, for the operator and for ticket 26's smoke."""

    run_dir: str
    scene_label: str
    n_episodes: int
    funnel: Dict[str, int]
    skipped: Tuple[Tuple[str, str], ...] = ()
    # Positioned last and defaulted: `tests/mac/test_notify.py` and
    # `tests/mac/test_yield_report.py` both construct this by keyword against the older
    # shape, and a run written before ADR-0017 legitimately has none.
    silent_phase: Optional[SilentPhaseTally] = None

    def as_dict(self) -> Dict[str, Any]:
        """The whole-run record, for `summary.json`.

        ``skipped`` is the reason this is written rather than printed. It is the run's
        **attrition** — how many of a scene's ObjectNav episodes could not express a
        decoupled anomaly response, and why — and until now it existed only in a console
        line. A number that bounds every ``n`` a paper can quote does not belong in a log
        the next reader will not have.

        The reasons are kept whole, not first-line'd like ``summary()`` does for the
        terminal: the builder's skip text carries the per-rule counts (*"11 too near, 4 on
        another floor, 0 with no view point"*), which is the difference between knowing the
        yield and knowing which rule is costing it.
        """
        return {
            "run_dir": self.run_dir,
            "scene": self.scene_label,
            "n_episodes": self.n_episodes,
            "n_skipped": len(self.skipped),
            "funnel": dict(self.funnel),
            "silent_phase": (
                None if self.silent_phase is None else self.silent_phase.as_dict()
            ),
            "skipped": [
                {"episode_id": episode_id, "reason": reason}
                for episode_id, reason in self.skipped
            ],
        }

    def summary(self) -> str:
        lines = [
            "run: {} episode(s) in {} -> {}".format(
                self.n_episodes, self.scene_label, self.run_dir
            ),
            "funnel (§6, denominator is stage 2):",
        ]
        for stage in FunnelStage:
            lines.append("  {}. {:<20} {}".format(
                int(stage), stage.name, self.funnel.get(stage.name, 0)
            ))
        if self.silent_phase is not None:
            tally = self.silent_phase
            # SWS and SR on one line, always. `CONTEXT.md`: never report SWS without SR.
            if tally.sws is None:
                lines.append(
                    "SWS: NOT_RUN (no episode ran past its offset step)   "
                    "SR: {}/{}".format(tally.n_source_reached, tally.n_episodes)
                )
            else:
                lines.append(
                    "SWS: {:.3f} ({}/{} ran past the offset step)   SR: {}/{}{}".format(
                        tally.sws,
                        tally.n_reached_after_offset,
                        tally.n_window_closed,
                        tally.n_source_reached,
                        tally.n_episodes,
                        # A rate whose denominator nobody checked for an accumulator. It
                        # prints rather than raises because this branch is also reached by
                        # a hand-built tally, and an unverified number that says so is
                        # more use to an operator than one that refuses to print.
                        "" if tally.tail_verified else "   (tail unverified)",
                    )
                )
            # The second number the SWS must not be read without, and it is not the same
            # question as `tail_is_active`: that says the buffer was built, this says the
            # agent could still hear something after the offset step. Zero here means
            # every silent phase was a hard cut to the bed and the SWS above describes
            # the mechanism ADR-0017 replaced.
            lines.append(
                "  audible tail after the offset step: {}/{} eligible episodes{}".format(
                    tally.n_tail_audible,
                    tally.n_window_closed,
                    "   <- the silent phase was a hard cut to the bed"
                    if tally.n_window_closed > 0 and tally.n_tail_audible == 0
                    else "",
                )
            )
        for episode_id, reason in self.skipped:
            lines.append("  skipped {:<12} {}".format(episode_id, reason.splitlines()[0]))
        return "\n".join(lines)


# ----------------------------------------------------------------------
# §2.3's calibration, and the half ticket 25 owns
# ----------------------------------------------------------------------


def calibration_poses(
    world: Any,
    source: Xyz,
    band: Tuple[float, float],
    n_poses: int,
    *,
    n_draws: int = CALIBRATION_DRAWS,
) -> List[Xyz]:
    """Navigable positions spread across the band, by geodesic distance to the source.

    ``audio/calibration.band_poses`` returns **distances** and says why: turning one into
    a pose needs the navmesh, which lives behind ``sim/`` and reaches this layer. This is
    that other half.

    Draw a pile of navigable points, measure each one's geodesic distance to the source,
    and assign the closest match to each target distance without reusing a point. Greedy
    and deliberately not exact: the targets are where the sweep would *like* to sample,
    and a scene simply may not have navmesh at 8 m from the source. What matters is that
    the samples span the band rather than clustering, which is what stops the threshold
    being placed inside a distribution the episode never visits.

    Euclidean nearness is not the question — a point 2 m away through a wall is 12 m of
    walking and hears the source like a point 12 m away, so the geodesic is the axis the
    band is defined on.
    """
    targets = band_poses(band, n_poses)
    draws: List[Tuple[float, Xyz]] = []
    for _ in range(int(n_draws)):
        point = world.random_navigable_point()
        distance = world.geodesic_distance(point, [source])
        if distance is None or not math.isfinite(float(distance)):
            continue
        draws.append((float(distance), point))
    if not draws:
        raise CalibrationError(
            "no navigable point in this scene has a geodesic route to the source at {} "
            "over {} draws — the source is on a disconnected navmesh island, so no pose "
            "can hear it and no threshold can be placed".format(source, n_draws)
        )

    chosen: List[Xyz] = []
    used = set()
    for target in targets:
        best_index = None
        best_error = None
        for index, (distance, _point) in enumerate(draws):
            if index in used:
                continue
            error = abs(distance - target)
            if best_error is None or error < best_error:
                best_index, best_error = index, error
        if best_index is None:
            break
        used.add(best_index)
        chosen.append(draws[best_index][1])
    return chosen


def calibrate_episode(
    world: Any,
    handle: Any,
    source: Xyz,
    clip: Any,
    cfg: RunConfig,
) -> Tuple[CalibrationResult, List[Xyz]]:
    """Run §2.3's sweep against this episode's source and derive ``onset_rms``.

    Seats the agent at each swept pose and renders once, so the distribution is the
    anomaly's own received level — measured through ``clips.render_through_ir``, not off
    the IR's energy, because the threshold is applied to what the agent hears and an IR is
    not in that domain.

    **Leaves the agent wherever the last pose was.** The caller re-seats it at the episode
    start, which it has to do anyway; restoring it here would hide that the sweep moves
    the agent, and a caller that forgot would start its episode at a calibration pose with
    nothing saying so.
    """
    poses = calibration_poses(
        world, source, cfg.audio.audible_band_m, cfg.audio.sweep_poses
    )

    def render_at(position: Xyz) -> Any:
        world.set_pose(position)
        observation, _guard = handle.observe()
        # The FIRST of `ir_under_policy`'s three sites. If the sweep took the real IR
        # while the loop took the anechoic stand-in, `onset_rms` and the scatter
        # `climb_eps` reads would both be calibrated in a domain the loop never runs in —
        # the exact failure the comment below this call names.
        return ir_under_policy(handle.audio_of(observation), cfg.ir_policy)

    # THE SWEEP AND THE LOOP MUST TAKE THE SAME PATH (ADR-0017), AND SINCE ADR-0019 THAT
    # PATH IS THE CUE READOUT. `onset_rms` and the scatter `climb_eps` reads are both
    # derived here and both applied to what `tail.heard_step` produces, and the
    # accumulator's settled level is a measured 1-13% above bare `render_through_ir`
    # depending on how reverberant the IR is. `onset.py:81-84` already names this exact
    # failure -- it "would silently move the domain the threshold was calibrated in" --
    # and nothing raises on it, so a shared code path is the only real fix.
    #
    # `sweep_cue_rms` renders each pose once and folds the accumulator through a whole
    # loop period, and its `level` is the QUADRATIC MEAN of the cue readout's per-fold
    # RMSs. That aggregation is what keeps `onset_rms` where it was: the `phase_folds`
    # cue windows tile the settled period exactly, so their quadratic mean EQUALS the
    # clip readout's RMS -- ratio 1.000000000000 in all four configurations this tree
    # ships. The sweep changed domain and the threshold's LEVEL did not move.
    hop = hop_samples(
        step_seconds=cfg.audio.step_seconds, sample_rate=cfg.audio.sample_rate
    )
    samples = sweep_cue_rms(poses, render_at, clip, hop=hop)
    # The threshold is placed against the LEVELS. The phases ride along to the record so
    # a bursty clip -- audible on one fold in five -- is identifiable after the fact
    # rather than merely suffered; nothing here is gated on them.
    levels = [sample.level for sample in samples]

    # The renderer's disagreement with itself, measured at ONE pose so distance is held
    # constant — the sweep above cannot supply this, because its 16 poses sit at different
    # distances and its spread is the gradient rather than the noise. The MIDDLE pose by
    # target distance, because `poses` comes back ordered by `band_poses` and the middle of
    # the audible band is where the climb actually stalls: `detour-2` put the abandoned
    # arm's plateau windows at a median 5.34 m and the reached arm's at 2.74 m.
    #
    # THREE ARMS AT THAT POSE, and only one of them costs extra renders.
    #
    # `climb_eps` reads the CUE arm, because the cue readout is what `is_rising`
    # compares: since ADR-0019 the agent's reading is `hop` samples wide, so the spread
    # of the reading the climb clears has to be measured on `hop` samples too.
    #
    # The CLIP arm is the ADR-0017 estimator and it is FREE -- `sweep_loop_scatter`
    # takes both readouts off the same folds, so it costs the folds it already pays for.
    # It is kept because every `eps` written between ADR-0017 and ADR-0019 is that
    # number.
    #
    # The SINGLE arm is the pre-ADR-0017 estimator and the only one that costs extra
    # renders -- 12 more at one pose, ~0.33 s at ticket 06's 27 ms, once per episode,
    # beside a sweep that already pays for 16. It is kept because every `eps` on disk
    # before ADR-0017 is that number, and a change nobody can price against its own
    # history is a change nobody can undo.
    #
    # THE HISTORY, labelled with the comparison it is: SINGLE against CLIP, measured at a
    # held pose over 400 repeats before the split -- 3.490e-04 for independent draws
    # against 1.830e-04 for the loop's own readings, a ratio of 1.91 (3.55 under a second
    # noise model), lag-1 autocorrelation 0.804 against 0.022. That is what put a loop arm
    # in this file at all. SINGLE against CUE is a DIFFERENT comparison and it is
    # unmeasured on hardware: `calibration.py` pre-registers `single > cue > clip` and
    # records a Mac counter-measurement that doubts it, and the box's first run after
    # this change is what settles it.
    single_render_scatter: List[float] = []
    loop_cue: Sequence[float] = ()
    loop_clip: Sequence[float] = ()
    if poses:
        held_pose = poses[len(poses) // 2]
        single_render_scatter = sweep_render_scatter(held_pose, render_at, clip)
        loop = sweep_loop_scatter(held_pose, render_at, clip, hop=hop)
        loop_cue, loop_clip = loop.cue, loop.clip

    # The distance each swept sample was taken at. `calibration_poses` chose every pose BY
    # geodesic distance and then discarded the number; re-measuring it costs one pathfinder
    # query per pose and no render at all. It is what turns the sweep into a field profile
    # — received level against distance — and that curve is the premise the whole climb
    # rests on: past the critical distance the reverberant field dominates, level stops
    # falling with distance, and no threshold or ray count recovers a gradient that is not
    # there. Pairs whose route cannot be measured are dropped rather than defaulted, so an
    # unroutable pose cannot enter the profile as a distance of zero.
    profile: List[Tuple[float, float]] = []
    for pose, level in zip(poses, levels):
        distance = world.geodesic_distance(pose, [source])
        if distance is None or not math.isfinite(float(distance)):
            continue
        profile.append((float(distance), float(level)))

    result = calibrate_onset(
        cfg.audio.bed_rms,
        levels,
        global_volume=cfg.audio.global_volume,
        cue_scatter_samples=loop_cue,
        clip_scatter_samples=loop_clip,
        single_render_samples=single_render_scatter,
        profile=profile,
        cue_phases=[sample.phases for sample in samples],
    )
    return result, poses


# ----------------------------------------------------------------------
# the detector arm
# ----------------------------------------------------------------------


def make_detector(cfg: RunConfig, world: Any, anomaly_episode: AnomalyEpisode) -> GoalDetector:
    """The configured arm of ADR-0008's one seam, wired to this episode's two objects.

    The oracle's table is the primary goal's view points plus the anomaly source, keyed by
    object name — the two things ``detects()`` is ever asked about, since the controller's
    active goal is the primary category in SEARCH and the anomaly object in INVESTIGATE.

    **When the source is a same-category instance the two entries merge, and that is
    correct rather than convenient.** ``detects("chair")`` means "a chair is here", and a
    chair at the source is a chair. What it costs is instance discrimination during the
    detour: the visual confirm can fire at a different chair. The builder prefers a
    different category precisely to keep that rare, records ``same_category`` when it
    cannot, and this is where the consequence lands.
    """
    view_points: Dict[str, List[Xyz]] = {
        anomaly_episode.primary_category: [
            view_point.position for view_point in anomaly_episode.episode.view_points()
        ]
    }
    view_points.setdefault(anomaly_episode.source.anomaly_object, []).append(
        anomaly_episode.source.position
    )

    if cfg.detector is Detector.ORACLE:

        def distance_to(obj: str) -> Optional[float]:
            points = view_points.get(obj)
            if not points:
                return None
            return world.geodesic_distance(world.pose().position, points)

        return OracleDetector(distance_to, cfg.detector_config)

    raise RuntimeError(
        "Detector.CAPTION needs a grounding VLM, and `earshot/vlm.py` — the Qwen2-VL-2B "
        "connector ADR-0013's tree names — has not been built. `agent/detector."
        "CaptionDetector` ships live and takes an injected `Grounder`, so the missing "
        "piece is the connector rather than the arm; it belongs to R2, which is out of "
        "scope for this map (the smoke runs Detector.ORACLE, §8). Raising rather than "
        "substituting the oracle: an arm that silently ran the other arm would produce "
        "numbers labelled `caption` in every audit record."
    )


# ----------------------------------------------------------------------
# steering
# ----------------------------------------------------------------------


def _divert_candidate(target: Xyz, pose: Pose) -> Candidate:
    """The oracle arm's investigate waypoint, as a candidate for the pool.

    Injected rather than steered to directly, which is the seam ``agent/scorer.py``
    describes: the divert sorts ahead of everything by rank, so the interrupt is an
    override rather than a high score that a maximal frontier candidate could tie. Going
    around the pool would also skip ``reachability``'s navmesh filter, and a source the
    agent cannot route to is exactly the case that must not become a silent straight-line
    walk into a wall.
    """
    dx, dz = target.x - pose.position.x, target.z - pose.position.z
    return Candidate(
        candidate_id=DIVERT_CANDIDATE_ID,
        position=target,
        source=SOURCE_INVESTIGATE,
        distance_m=math.hypot(dx, dz),
        # The bearing is recomputed at the snapped position by `reachable_pool`, and the
        # scorer ignores it for a divert anyway (the rank puts it first). 0.0 is the
        # honest placeholder rather than a second derivation of the frame convention.
        bearing_rad=0.0,
        raw_score=1.0,
    )


def _choose_waypoint(
    proposer: FrontierProposer,
    pose: Pose,
    *,
    snap_point: Callable[[Xyz], Optional[Xyz]],
    geodesic: Callable[[Xyz, Xyz], Optional[float]],
    planner: PlannerConfig,
    divert: Optional[Candidate] = None,
) -> Tuple[Xyz, str, Dict[str, int]]:
    """Propose, filter on the navmesh, and pick one. Returns ``(waypoint, source, counters)``.

    Two stages because ADR-0008's invariant has two ways to be met. The frontier pool can
    be empty of *cells* — the proposer answers that itself with the compass fan — or full
    of cells the navmesh rejects wholesale, which is the occupancy-versus-navmesh
    disagreement the invariant exists for. Only the caller can answer the second, which is
    why ``compass_fan`` is public.
    """
    proposed = list(proposer.propose(pose))
    if divert is not None:
        proposed.insert(0, divert)
    report = reachable_pool(
        proposed, pose, snap_point=snap_point, geodesic=geodesic, cfg=planner
    )
    kept, counters = report.candidates, report.counters()
    if not kept:
        fan = list(proposer.compass_fan(pose))
        if divert is not None:
            fan.insert(0, divert)
        report = reachable_pool(
            fan, pose, snap_point=snap_point, geodesic=geodesic, cfg=planner
        )
        kept = assert_pool(report, stage="compass fan after the frontier pool emptied")
        counters = report.counters()
    scored = pick_waypoint(kept)
    return scored.candidate.position, scored.candidate.source, counters


# ----------------------------------------------------------------------
# the episode
# ----------------------------------------------------------------------


def run_episode(
    world: Any,
    handle: Any,
    anomaly_episode: AnomalyEpisode,
    cfg: RunConfig,
    *,
    clip: Any,
    detector: GoalDetector,
    index: int = 0,
    room_labeler: Optional[RoomLabeler] = None,
    clap_encoder: Optional[Any] = None,
    calibration: Optional[CalibrationResult] = None,
    memory: Optional[MemoryContext] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> EpisodeResult:
    """One mission: the primary find-task, the interrupt, the detour, the resume.

    ``world`` and ``handle`` are duck-typed (see the module docstring). ``calibration``
    is passed in when the caller has already run the sweep — ``run()`` does, because it
    also wants the sweep's poses for the record — and derived here otherwise.

    The order inside a step is fixed and each part of it is load-bearing: render once
    (the shared observation, so RGB, depth and the IR are the same instant), mix the bed
    and measure, fold the onset, feed the map and the detector, ask the controller, apply
    what it decided, record the row. A detector observing a frame from a different render
    than the onset measured would make "peak-or-plateau plus visual confirm" a statement
    about two moments.
    """
    episode = anomaly_episode.episode
    source = anomaly_episode.source.position
    t_anom = int(anomaly_episode.t_anom)
    # ADR-0017's window, planned here rather than in `task/dataset.py`: this function
    # already holds `cfg`, `index` and `t_anom`, the planning is pure, and
    # `build_anomaly_episodes` stays deterministic and seedless. The draw (DRAWN only) is
    # a pure function of (`seed`, `index`) so `tools/episode_diff.py` can still pair the
    # same episode index across two sweeps and be comparing the same task.
    window = plan_window(
        t_anom=t_anom,
        max_steps=int(cfg.max_steps),
        policy=cfg.sounding_policy,
        sounding_steps=int(cfg.sounding_steps),
        budget_fraction=float(cfg.sounding_budget_fraction),
        draw_steps_range=(
            int(cfg.sounding_draw_steps[0]),
            int(cfg.sounding_draw_steps[1]),
        ),
        seed=int(cfg.seed),
        episode_index=int(index),
    )
    realizable = cfg.localization is Localization.REALIZABLE
    # ADR-0018's three controller arms, mapped to plain primitives exactly as
    # `realizable` above is. `agent/` may not import `earshot.config` (ADR-0013's layer
    # graph), so the enum stops here and the controller stays a decision function over
    # bools and ints that `tests/mac/test_agent_controller.py` can drive with no config
    # at all.
    climb_enabled = cfg.climb_rule is ClimbRule.LIVE
    lateral_cue_enabled = cfg.lateral_cue is LateralCue.LIVE
    cast_steps = CAST_STEPS if cfg.cast_policy is CastPolicy.CAST else 0
    labeler = room_labeler if room_labeler is not None else NullRoomLabeler()
    say = progress if progress is not None else (lambda _message: None)

    if memory is not None and clap_encoder is None:
        # The store's rows are CLAP embeddings, so the query has to be one too. Without an
        # encoder the prior could never fire and every episode of the cell would record a
        # miss it did not actually make -- four cells of identical numbers, arrived at
        # silently. This is the one wiring mistake that would look exactly like a result.
        raise ValueError(
            "a memory arm was passed ({}) but no CLAP encoder; the semantic store is "
            "queried with an audio embedding and there is nothing here to make one, so "
            "the prior would silently never fire".format(memory.condition)
        )

    goal_positions = [view_point.position for view_point in episode.view_points()]
    world.set_pose(episode.start_position, episode.start_rotation)
    start_pose = world.pose()
    start_end_distance = world.geodesic_distance(start_pose.position, goal_positions)

    if calibration is None:
        calibration, _poses = calibrate_episode(world, handle, source, clip, cfg)
        world.set_pose(episode.start_position, episode.start_rotation)
        start_pose = world.pose()

    hop = hop_samples(
        step_seconds=cfg.audio.step_seconds, sample_rate=cfg.audio.sample_rate
    )
    # TWO BEDS, EACH NORMALISED AT ITS OWN LENGTH, AND NEVER ONE SLICED. The split
    # readout gave the runner two signal lengths, and the cheap-looking move -- slice the
    # last `hop` samples off the clip-length bed -- spends exactly the tolerance
    # `bed.py`'s own docstring promises it protects. A slice of `n` Gaussian samples has a
    # relative RMS error of about `1/sqrt(2n)`, and `n` here is `hop`, which is a free
    # parameter (`step_seconds` x `sample_rate`). Measured against the fixed `BED_SEED`:
    # the worst disjoint hop-slice is 0.31% at the shipped hop of 44100, 6.79% at this
    # file's own fixture hop of 441, and 17.73% at the tail fixture's hop of 100, against
    # an `AudioConfig.pre_onset_rms_tol` of 5%. So a slice raises `ProvenanceError` at two
    # configurations this tree ships tests at, and the cost scales the WRONG WAY -- the
    # smaller the step, the worse it gets. Two beds are exact to ~1e-6% at every length.
    #
    # The two are NOT sample-aligned (same `BED_SEED`, each normalised by its own RMS),
    # so nothing may compare their samples. Nothing does: `bed_cue` feeds the onset and
    # `bed_clip` feeds CLAP, and neither is diffed against the other.
    bed_cue = bed_signal(hop, cfg.audio.bed_rms)
    # Built once per episode and used at most ONCE, at the classification step. Building
    # it lazily inside the loop would put a 1.76 MB allocation inside the bracket
    # criterion 7 audits, for a value the loop reads on one step out of hundreds.
    bed_clip = bed_signal(len(clip), cfg.audio.bed_rms)
    # The guard's arming render already measured THIS scene's IR width, so handing it
    # over means the accumulation buffer is preallocated in the common case and
    # `open_tail`'s growth path is the safety net for a wider pose rather than the norm.
    # `AudioContextReport()`'s default `ir_shape` is `None` (the Mac fake), which has to
    # reach 0 rather than crash -- there is no `maxIRLength` anywhere in this tree to
    # fall back on, which is exactly why the buffer grows instead of truncating.
    ir_shape = getattr(getattr(handle, "report", None), "ir_shape", None)
    tail = open_tail(
        window=len(clip),
        hop=hop,
        headroom=0 if ir_shape is None else max(0, int(ir_shape[1]) - 1),
    )
    proposer = FrontierProposer(cfg=cfg.planner)
    proposer.reset(start_pose)
    follow = world.follower()
    state = ControllerState.for_episode(anomaly_episode.primary_category)
    onset = OnsetState()

    def geodesic(a: Xyz, b: Xyz) -> Optional[float]:
        return world.geodesic_distance(a, [b])

    def route_to_source(position: Xyz) -> Optional[float]:
        """How far the agent is from the source ALONG THE NAVMESH, or None if unrouted.

        Recorded per step because the audit's derived series is horizontal `xz` distance,
        and the two separate exactly where the climb dies: at 5-12 m the agent is usually
        in another room, where `xz` shrinks while the walk (and the sound's path) does not.
        Non-finite is None — an unrouted pose has no distance, and a zero there would put
        a phantom sample at the source.
        """
        distance = geodesic(position, source)
        if distance is None or not math.isfinite(float(distance)):
            return None
        return float(distance)

    # `L_opt` AGAINST THE SOURCE: the navmesh route from this episode's own start pose to
    # the source, which is `compute_source_spl`'s numerator and the floor of its
    # denominator. Nothing on disk carried it — `start_end_distance` above is the route to
    # the PRIMARY goal — so no SPL against the source was computable from any artefact
    # this tree wrote.
    #
    # `None` is the load-bearing case and not an edge one: 23 of `yield-2`'s 365 episodes
    # have no navmesh route to their source at all, and an SPL of 0.0 there would put an
    # unwinnable episode in the same bucket as one that had a route and failed to walk it.
    # `compute_source_spl` returns None on it and the metric key is then simply absent.
    #
    # Taken from `start_pose` and not from `world.pose()`: the calibration sweep leaves
    # the agent at its last swept pose, and the block above re-seats it and re-reads
    # `start_pose`. Reading the position rather than the agent puts this out of that
    # ordering's reach entirely.
    source_start_route = route_to_source(start_pose.position)

    # §2.5's one smoke exception: verify audibility at the episode's own start pose, once,
    # with a calibration render, so the smoke is deterministic. It is a measurement rather
    # than a screen — nothing is rejected on it, and it is recorded next to the threshold
    # it is compared against.
    observation, _guard = handle.observe()
    # The SECOND site. The probe is compared against `calibration.onset_rms`, which came
    # off the sweep above, so it has to be rendered through the same IR the sweep used or
    # `start_pose_audible` is a comparison between two domains.
    start_anomaly_rms = rms(
        render_through_ir(
            ir_under_policy(handle.audio_of(observation), cfg.ir_policy), clip
        )
    )
    n_renders_before = int(getattr(world, "n_renders", 0))

    steps: List[StepRecord] = []
    energy: List[float] = []
    counters: Dict[str, int] = {}
    waypoint: Optional[Xyz] = None
    saved_waypoint: Optional[Xyz] = None
    room: Optional[str] = None
    anomaly_class: Optional[str] = None
    verdict: Optional[bool] = None
    classified = False
    entered_investigate = False
    path_len = 0.0
    n_actions = 0
    n_no_action = 0
    min_d2g: Optional[float] = None
    min_d2source = float("inf")
    # The same, restricted to the steps the source was still sounding on. It is the
    # RIGHT-CENSORING evidence ADR-0017's open duration question needs: `onset_delay_steps`
    # exists only for episodes that fired, so a window too short to hear leaves the
    # distribution truncated with no marker. How close the agent got before the source
    # stopped is what says whether "a bit longer" would have been enough.
    min_d2source_in_window = float("inf")
    # The same in ROUTE metres. `None` rather than `inf`: an episode whose source is on a
    # disconnected navmesh island never has one, and that is a different fact from a
    # window that closed too early.
    min_route_in_window: Optional[float] = None
    # The step the SOURCE was reached, which SWS needs and which nothing on disk carried.
    source_reached_step: Optional[int] = None
    # The path length AT that step, which is `compute_source_spl`'s `L_taken`. The
    # whole-episode `path_len` cannot serve: the detour ends mid-episode and the primary
    # search keeps walking afterwards, so an SPL against the source computed on it would
    # be a function of how long the primary task ran after the source was already found.
    path_len_at_reach: Optional[float] = None
    # What the CLIP read window held on the step CLAP was handed it -- see
    # `TailState.clip_source_fill`, whose prefix is load-bearing since the split readout
    # -- how it was rotated, which step that was, and how long the classification waited
    # for the buffer to fill. All four are set together, at the classification step.
    clap_window_fill: Optional[float] = None
    clap_step: Optional[int] = None
    clap_rotation_phase: Optional[int] = None
    clap_deferred_steps: Optional[int] = None
    # The vector the memory prior queries its store with, captured at the classification
    # step from the SAME `audio_embedding` path that produced the store's rows. Two
    # processes have to agree on it -- the prior pass wrote the store, this run reads it --
    # and a second encode here would be a second place for them to drift apart.
    # `Any` rather than `np.ndarray`: this module imports no numpy today and adding the
    # import for one local annotation would be the only reason it did.
    heard_embedding: Optional[Any] = None
    # The prior is resolved ONCE, at the first investigate step after the source has gone
    # silent, and then reused. Re-resolving every step would let the recalled category
    # change under the agent mid-detour, which is not memory, and would pay the k-NN and
    # the navmesh queries per step for an answer that cannot change.
    memory_prior = None
    memory_miss: Optional[PriorMiss] = None
    memory_consulted = False
    # The step the onset FIRED, which is the classification step only when the buffer was
    # already full. The two apart is the deferral; the first set with the second still
    # None at the end of the loop is an episode that ended mid-ramp.
    clap_onset_step: Optional[int] = None
    stopped = False
    collided = False  # no action has been taken yet, so nothing has been hit
    wall_clock_0 = time.perf_counter()

    for step in range(int(cfg.max_steps)):
        audio_t0 = time.perf_counter()
        observation, _guard = handle.observe()
        # The THIRD site, and the one the agent actually hears. Applied here rather than
        # inside `heard_step` so the whole cost of the substitution sits inside the
        # bracket criterion 7 audits, like every other part of the per-step audio bill.
        impulse = ir_under_policy(handle.audio_of(observation), cfg.ir_policy)
        sounding = window.is_sounding(step)
        tail, cue = heard_step(
            tail, ir=impulse, clip=clip, bed_cue=bed_cue, sounding=sounding
        )
        measured = rms(cue)
        # The whole per-step audio bill, not the render alone: criterion 7 audits what
        # live-every-step costs, and the convolution, the bed mix and the guard's two
        # tempfiles are all part of it now (ticket 06's 27.2 ms was the render only).
        # The accumulator is inside this bracket, which is correct AND cheaper than what
        # it replaced: its FFT is over `hop + L` rather than `N + L`, measured at 7.4 ms
        # against 22.9 ms for the whole-clip render on a Mac.
        audio_s = time.perf_counter() - audio_t0
        lateral = lateral_sign(cue)

        onset = observe_step(
            onset,
            step=step,
            measured_rms=measured,
            t_anom=t_anom,
            onset_rms=calibration.onset_rms,
            bed_rms=cfg.audio.bed_rms,
            tolerance=cfg.audio.pre_onset_rms_tol,
        )
        if onset.fired and not classified:
            if clap_onset_step is None:
                clap_onset_step = int(step)
                say("  step {}: onset at RMS {:.6g} (threshold {:.6g})".format(
                    step, measured, calibration.onset_rms
                ))
            # WHAT CLAP IS ABOUT TO BE GIVEN, AND WHEN. Since ADR-0019 the deferral and
            # the fill below read the CLIP readout, which is the domain ADR-0018's bank of
            # record was measured on and which this change deliberately does not touch;
            # the controller reads the CUE and gets there sooner. Everything in this block
            # is about the clip readout, `tail.heard_clip_window(tail, bed_clip=bed_clip)`,
            # and it is a different domain from the full-length
            # `render_through_ir` clips ADR-0018's bank of record (anchor recall 0.911 /
            # 0.895) and the CLAP separation gate (`task/clap_gate.py`) were measured on.
            # It differs in two ways, both measured, neither of which moves the RMS:
            #
            # ROTATION. The settled readout is the circular convolution of the LOOPED
            # clip -- verified as `np.roll(circular_conv, -132300)` to 3e-8..1.1e-7 over
            # nine IR/clip pairs. Its RMS matches `render_through_ir` to 1.0000-1.0134
            # while the largest sample difference is 1.00 to 1.92 times the peak, because
            # the waveform is a rotation of it. With a transient clip the event's peak
            # lands at sample 177988 / 133888 / 89788 / 45688 / 1588 as the step index
            # cycles mod `N/hop`, where `render_through_ir` always put it at 1588.
            #
            # FILL. The read window is only full of source after `ceil(N/hop)` folds, and
            # `onset_rms` comes off a 16-pose sweep spanning the audible band, so an agent
            # standing nearer than the sweep's median clears the threshold MID-RAMP: CLAP
            # would receive a 5 s waveform whose first 3 s is bed.
            #
            # So the classification WAITS while it can. Deferring is not a re-render --
            # §4.3 wants the signal the onset fired on, and this is still exactly that
            # signal, read a few steps later off the same accumulator -- and the delay is
            # bounded by the ramp, `ceil(N/hop) - 1` steps at most. Once the window has
            # closed, waiting makes it worse (the buffer empties), so what there is gets
            # classified and the record says it was a tail. The alternative is the one
            # thing that must not happen: a half-empty buffer classified in silence.
            #
            # The cost, named: on a run WITH an encoder the interrupt waits with the
            # classification (`onset_for_controller` below), so a mid-ramp onset delays
            # the detour by up to `ceil(N/hop) - 1` steps. That is the price of the gate
            # meaning anything at all -- the alternative is diverting on an undecided
            # verdict, which cannot be taken back. On a run without one there is nothing
            # to wait for and the timing is the pre-ADR-0017 timing to the step.
            fill = float(tail.clip_source_fill)
            if fill >= 1.0 or not sounding:
                classified = True
                room = labeler.label(observation.get("rgb"))
                clap_window_fill = fill
                clap_step = int(step)
                # The rotation, exactly: the clip index the NEXT hop starts at, which is
                # what the readout is rotated by. It cycles with period `N/hop`, so two
                # episodes at the same pose whose onset step differs by one carry
                # different values here -- which is the confound, written down.
                clap_rotation_phase = int(tail.phase)
                clap_deferred_steps = int(step) - int(clap_onset_step)
                if clap_encoder is not None:
                    waveform, sample_rate = heard_clip_for_clap(
                        heard_clip_window(tail, bed_clip=bed_clip), cfg.audio.sample_rate
                    )
                    # TWO BANKS, TWO QUESTIONS, AND THEY ARE NOT THE SAME BANK.
                    # `is_anomaly` answers "was that anything at all" and keeps
                    # `ANOMALY_CLASSES`, because `ANOMALY_GATE_DELTA` / `_TAU` were
                    # calibrated against exactly those prompts and HAZARD 2 forbids
                    # quoting them for a wider one. Its `best_class` is an argmax over
                    # three emergency names and is NOT what was heard on a run whose
                    # source is a flush -- copying it into the report is how the agent's
                    # testimony came to say "alarm" on a `toilet_flush` episode.
                    # ONE forward pass of a 153.5 M-param audio encoder, shared by every
                    # question asked about this clip. `test_task_runner` pins it: the
                    # gate, the testimony and the memory write must be about the same
                    # render, not three renders that agree by luck.
                    heard_embedding = audio_embedding(
                        waveform, sample_rate, clap_encoder
                    )
                    fired, gate_class, _scores = is_anomaly(
                        waveform, sample_rate, clap_encoder, embedding=heard_embedding
                    )
                    # `classify_anomaly` answers "what was it, given that it was one of
                    # these", over the bank this run's own source was drawn from. Forced
                    # argmax, so it cannot say "normal" -- which is right here, because
                    # the gate above already decided that.
                    anomaly_class, _testimony = classify_anomaly(
                        waveform,
                        sample_rate,
                        clap_encoder,
                        classes=testimony_bank(cfg.anomaly_class),
                        embedding=heard_embedding,
                    )
                    # The room arm reads what was HEARD, so it takes the testimony class.
                    # A no-op today and asserted to be one: `ROOM_PRIOR`'s normal sets
                    # name `running_water` and `appliance_hum`, which are in neither bank,
                    # so `room_conditioned_anomaly` can only return True or abstain.
                    # `test_audio_normality.py` fails if that stops being true, and then
                    # this line is a decision someone has to make rather than a surprise.
                    verdict = is_anomalous_here(fired, anomaly_class, room)
                    # `heard_embedding` above is the memory's query too. Captured from
                    # the clip the AGENT heard, not from the source file: the store was
                    # learned the same way, through a real IR at a real stop, and querying
                    # it with clean audio would be asking a different question from the
                    # one it was taught.
                # With no CLAP the verdict stays None, which `step_controller` reads as
                # "nothing conditioned this, so any onset interrupts" — the smoke's case
                # (§4.3: one sound, the anomaly by construction), and honest about it: the
                # report's `anomaly_class` stays None rather than being copied off the
                # dataset, which would be the task telling the agent what it heard.
                if clap_deferred_steps or fill < 1.0:
                    say("  step {}: classified after {} deferred step(s), read window "
                        "{:.0%} source{}".format(
                            step, clap_deferred_steps, fill,
                            "" if sounding else " (the window had already closed)",
                        ))

        pose = world.pose()
        depth = observation.get("depth")
        proposer.observe(depth, pose)
        detector.observe(rgb=observation.get("rgb"), depth=depth, pose=pose)
        energy.append(measured)
        del energy[:-ENERGY_HISTORY]

        diverting = is_diverting(state.mode)
        visual_confirm = bool(detector.detects(state.active_goal)) if diverting else False
        primary_reached = (
            False if diverting else bool(detector.detects(anomaly_episode.primary_category))
        )

        distance_to_goal = world.geodesic_distance(pose.position, goal_positions)
        if distance_to_goal is not None:
            min_d2g = distance_to_goal if min_d2g is None else min(min_d2g, distance_to_goal)
        min_d2source = min(min_d2source, pose.position.horizontal_distance_to(source))
        # One pathfinder query, read twice: the per-step record wants it and so does the
        # censoring evidence below. Asking twice would double the loop's route queries for
        # a number that cannot have changed between the two calls.
        route = route_to_source(pose.position)
        if sounding:
            min_d2source_in_window = min(
                min_d2source_in_window, pose.position.horizontal_distance_to(source)
            )
            # THE ROUTE, not the straight line, because "would a longer window have been
            # enough" is a question about walking. The two separate exactly where it
            # matters: at 5-12 m the agent is usually in another room, where `xz` shrinks
            # while the walk does not. `None` is an unrouted pose and stays out of the
            # minimum -- 23 of `yield-2`'s 365 episodes have no navmesh route to their
            # source at all, and a zero there would put a phantom sample at the source.
            if route is not None and (
                min_route_in_window is None or route < min_route_in_window
            ):
                min_route_in_window = route

        # THE INTERRUPT WAITS FOR THE VERDICT, AND ONLY WHERE THERE IS ONE TO WAIT FOR.
        # `step_controller`'s SEARCH branch reads `is_anomaly is None` as "nothing
        # conditioned this, so any onset interrupts" -- correct for a run with no encoder,
        # and a licence to divert on an undecided classification for a run with one. That
        # would spend §4.3's gate on every mid-ramp onset: the detour latches
        # (`investigated` and `investigate_aborted` are both terminal in SEARCH, and
        # `is_anomaly` is not read again once INVESTIGATE is entered), so a benign verdict
        # arriving four steps later could not pull the agent back and `n_benign_ignored`
        # would never count it. Measured on the fake with a benign verdict: the funnel
        # caps at ONSET_FIRED and 34 benign steps are counted, where the undecided reading
        # diverted at the onset step and reached INVESTIGATE_ENTERED regardless.
        #
        # A run WITHOUT an encoder has nothing pending and keeps its pre-ADR-0017 timing to
        # the step, which is what keeps this an audio change rather than a policy one --
        # the whole Mac suite and the no-CLAP smoke run that arm.
        onset_for_controller = bool(onset.fired) and not (
            clap_encoder is not None and not classified
        )
        state, decision = step_controller(
            state,
            cfg.controller,
            onset_fired=onset_for_controller,
            is_anomaly=verdict,
            primary_goal_reached=primary_reached,
            realizable=realizable,
            # The realizable arm is never handed the coordinate at all. The controller
            # would ignore it, but "the arm does not have it" is a stronger property than
            # "the arm does not read it", and it costs one conditional.
            source_xyz=None if realizable else source,
            arrived_at_source=(
                False
                if realizable
                else pose.position.horizontal_distance_to(source)
                <= float(cfg.controller.investigate_arrive_radius_m)
            ),
            anomaly_class=anomaly_class,
            anomaly_object=anomaly_episode.source.anomaly_object,
            energy_history=energy,
            lateral_sign=lateral,
            visual_confirm=visual_confirm,
            # Last step's collision, which is the only tense available: the flag comes
            # back from the action, so the climb reacts to the wall it just hit rather
            # than to one it has not touched yet.
            pose=pose,
            # The rise has to clear the renderer's own scatter, measured on THIS episode's
            # geometry. `climb_eps` owns the `None` case — the fallback is the old `1e-6`,
            # which is not a safe default so much as the previous behaviour, kept so a
            # caller that skips the scatter render still runs. It lives beside the rule
            # rather than here so `tools/detour_report` replays the same threshold this
            # episode ran at; the audit records which of the two was in force.
            #
            # THE CUE ARM, because the cue readout is what `is_rising` compares. Since
            # ADR-0019 the agent's reading is `hop` samples wide, so an epsilon measured
            # on the 5 s clip readout would be the spread of a quantity nothing reads.
            # The other two arms are on the record beside it -- `clip_render_scatter` for
            # the ADR-0017 era and `single_render_scatter` for the one before it -- so
            # this episode's `eps` can be priced against both.
            rising_eps=climb_eps(calibration.cue_render_scatter),
            # ADR-0018's three controller arms. `scan_steps` is passed EXPLICITLY at
            # `SCAN_STEPS` rather than left to the default two functions away: the scan
            # length is not an arm, and a value the call site does not name is a value a
            # reader of this call has to go and find.
            cast_steps=cast_steps,
            scan_steps=SCAN_STEPS,
            climb_enabled=climb_enabled,
            lateral_cue_enabled=lateral_cue_enabled,
        )
        # SWS's numerator, captured here and NOT in the controller. `state.investigated`
        # flips exactly once, in both localization arms, and the runner already holds the
        # state every step -- so the ABSOLUTE step is recoverable with no surgery on
        # `agent/controller.py`, which is what keeps this diff a renderer-and-record
        # change rather than a policy change. An audio change and a policy change in one
        # commit is a confound.
        if state.investigated and source_reached_step is None:
            source_reached_step = step
            # THE TENSE IS DELIBERATE. `path_len` here excludes this step's own
            # displacement, which is added further down after the action is applied. The
            # reach is decided BEFORE the action is taken, so the distance walked to earn
            # it is the distance walked before this step — not after. Do not "fix" this
            # by moving the capture past the `path_len +=` below.
            path_len_at_reach = path_len
        if state.mode is NavMode.INVESTIGATE or decision.mode is NavMode.INVESTIGATE:
            entered_investigate = True
        if decision.save_primary_state:
            saved_waypoint = waypoint
            say("  step {}: INVESTIGATE — primary state saved".format(step))
        if decision.restore_primary_state:
            waypoint = saved_waypoint
            say("  step {}: {} — primary state restored".format(step, decision.mode.name))
        if decision.force_requery:
            proposer.request_replan()

        # **Both localization arms steer through the pool** (ticket 26). The realizable
        # arm used to hand back a low-level action applied straight to the simulator,
        # which left the detour with no planner and no map: `move_forward` was its only
        # translation and the energy gradient chose where forward pointed, so a blocked
        # line to the source was a measured livelock. It now names a probe point the same
        # way the oracle arm names the source, and the follower routes to it.
        # `realizable_action` survives on the decision as the diagnostic of what the cue
        # said, and is no longer the thing that moves the agent.
        action: Optional[str] = None
        if decision.mode is NavMode.COMPLETE:
            # The primary STOP, which is a task decision and never reaches the simulator
            # (`sim.world.step` refuses it). Recorded as the action taken, because "the
            # episode ended here and why" is exactly what the per-step record is for.
            action = ACT_STOP
        elif action is None:
            target = decision.investigate_waypoint or decision.investigate_probe
            # THE MEMORY SPEAKS WHEN THE ROOM HAS GONE QUIET, and only then. While the
            # source is sounding the live cue is real evidence about where it is and a
            # recalled category must not override it. After ADR-0017's offset step the cue
            # has nothing left to say: `_probe_for` still names a place every step, but it
            # is a 2 m hop in whatever direction the scan/cast cycle last chose, and
            # `abl-1` priced that at SWS 27 of 272. That silence is the headroom, and it is
            # the regime the four cells are meant to differ in.
            if memory is not None and is_diverting(decision.mode) and not sounding:
                if not memory_consulted:
                    memory_consulted = True
                    if memory.is_live and heard_embedding is not None:
                        memory_prior, memory_miss = resolve_prior(
                            memory.semantic,
                            heard_embedding,
                            k=int(memory.k),
                            points_by_category=memory.points_by_category,
                            # The NAVMESH, not a straight line: an instance behind a wall
                            # is not nearer than one down the hall, and a point with no
                            # route is excluded rather than ranked at some large number.
                            distance_to=lambda point: geodesic(pose.position, point),
                        )
                    else:
                        # An empty store or a scene with no annotated object cannot name a
                        # place. That is `NO_PREDICTION`, recorded rather than left blank
                        # because a blank would mean "never consulted". It is NOT the
                        # `not_heard` cells' usual path: their store keeps the bank's other
                        # classes and votes a confident wrong category (`PriorMiss`'s own
                        # docstring has the full account).
                        memory_prior, memory_miss = (None, PriorMiss.NO_PREDICTION)
                    say("  step {}: the window has closed — memory says {}".format(
                        step,
                        "{} at {:.1f} m".format(
                            memory_prior.category, memory_prior.distance_m
                        ) if memory_prior is not None else
                        "nothing ({})".format(memory_miss.value),
                    ))
                if memory_prior is not None:
                    target = memory_prior.target
            divert = _divert_candidate(target, pose) if target is not None else None
            waypoint, action, step_counters = _steer(
                proposer,
                pose,
                waypoint,
                follow=follow,
                snap_point=world.snap_point,
                geodesic=geodesic,
                planner=cfg.planner,
                divert=divert,
            )
            for key, value in step_counters.items():
                counters[key] = counters.get(key, 0) + int(value)

        displacement: Optional[float] = None
        if action is not None and action != ACT_STOP:
            before = pose.position
            # `World.step` returns habitat's collision flag and the first version of this
            # loop discarded it, which left nothing in §3.2's record separating a forward
            # that moved from one that hit a wall — the one number that decides whether a
            # stalled climb is obstacle-blind or merely short of budget.
            collided = bool(world.step(action))
            after = world.pose().position
            displacement = math.hypot(after.x - before.x, after.z - before.z)
            path_len += displacement
            n_actions += 1
        else:
            # A STOP or a step not taken collides with nothing, and carrying the previous
            # step's flag forward would have the climb turning away from a wall it is no
            # longer facing.
            collided = False
            if action is None:
                n_no_action += 1

        steps.append(
            StepRecord(
                step=step,
                measured_rms=float(measured),
                lateral_sign=int(lateral),
                # The field keeps EXACTLY its meaning -- "the source emitted on this
                # step" -- so no record written before ADR-0017 is reinterpreted. What
                # changed is that it can now go back to False.
                source_playing=bool(sounding),
                source_is_visible=handle.source_is_visible(),
                action=action,
                audio_render_s=float(audio_s),
                collided=collided,
                displacement_m=displacement,
                # The pose the reading was taken AT, not the one the action left behind:
                # `measured_rms` on this row was rendered here, and pairing the energy
                # with the place it was measured is the whole point of recording it.
                position=pose.position,
                # The route, beside the position the route is FROM. One pathfinder query
                # per step, no render: the field profile's axis has to be the distance the
                # sound travels rather than the one a straight line measures, and only
                # this layer can ask.
                geodesic_to_source=route,
                # What the CUE said, beside what the agent DID. Since ticket 26 the two
                # differ — the rule names a probe and the follower routes to it — and
                # until now only the follower's half was written down, which left an
                # analyst able to recompute the rule but with nothing to check the
                # recomputation against.
                realizable_action=decision.realizable_action,
            )
        )

        if state.mode is NavMode.COMPLETE:
            stopped = True
            say("  step {}: primary goal reached — STOP".format(step))
            break

    wall_clock = time.perf_counter() - wall_clock_0
    n_renders_in_loop = int(getattr(world, "n_renders", 0)) - n_renders_before

    # §3.1, on the recorded state, before anything is written. It RAISES: an episode that
    # reached here with an impossible onset step or a drifted pre-onset reading is one an
    # analyst would otherwise quote, and both causes are silent-fabrication bugs.
    assert_provenance(
        onset,
        t_anom=t_anom,
        bed_rms=cfg.audio.bed_rms,
        tolerance=cfg.audio.pre_onset_rms_tol,
    )

    final_pose = world.pose()
    primary_dist_at_stop = (
        world.geodesic_distance(final_pose.position, goal_positions) if stopped else None
    )
    event = state.investigation_event
    stop_pose = event.stopped_at_pose if event is not None else None
    source_dist_at_stop = (
        stop_pose.position.horizontal_distance_to(source) if stop_pose is not None else None
    )
    # THE FINAL-POSE ROUTE TO THE SOURCE, and it is a DIFFERENT quantity from
    # `min_d2source_m` two screens down. That one is the CLOSEST APPROACH over the whole
    # episode, horizontal, and it answers "did the agent ever get near". This answers
    # "where did the agent end up" — the source-side twin of the `dist_to_goal_final`
    # expression that is computed inline for `compute_soft_spl` below and then thrown
    # away. An episode that walked to the source and then walked back to the primary goal
    # has a small closest approach and a large final distance, and reporting either as the
    # other is the confusion this comment exists to prevent.
    #
    # A second pathfinder query rather than a reuse: the last per-step `route` was taken
    # at the last step's PRE-action pose, and the final pose is after that action.
    dtg_source_final = compute_dtg_source_final(
        geodesic_to_source=route_to_source(final_pose.position)
    )
    # SPL against the SOURCE. `source_dist_at_stop` is reused rather than recomputed: it
    # already IS the distance at the reach, measured at the investigation stop pose.
    source_spl = compute_source_spl(
        source_reached=source_reached_step is not None,
        dist_at_reach=source_dist_at_stop,
        geodesic_optimal_to_source=source_start_route,
        path_len_at_reach=path_len_at_reach,
        success_radius=1.0,
    )

    report = AgentReport(
        primary_completed=state.mode in (NavMode.COMPLETE, NavMode.REPORTED),
        heard_at_step=onset.onset_step,
        room=room,
        anomaly_class=anomaly_class,
        stopped_at_pose=stop_pose,
        visual_confirm_object=event.visual_confirm_object if event is not None else None,
        investigate_aborted=state.investigate_aborted,
        resumed=state.resumed,
        n_benign_ignored=state.n_benign_ignored,
    )

    success_1m, spl_1m = compute_benchmark_spl(
        stopped=stopped,
        dist_at_stop=primary_dist_at_stop,
        geodesic_optimal=float(start_end_distance or 0.0),
        path_len_taken=path_len,
        success_radius=1.0,
    )
    _success_01, spl_01 = compute_benchmark_spl(
        stopped=stopped,
        dist_at_stop=primary_dist_at_stop,
        geodesic_optimal=float(start_end_distance or 0.0),
        path_len_taken=path_len,
        success_radius=0.1,
    )
    metrics: Dict[str, float] = {
        # Find-SR, both rings (§6): 1.0 m primary, 0.1 m the localization-bound
        # diagnostic. Both are STOP-gated.
        "find_sr_1m": float(bool(success_1m)),
        "find_sr_0_1m": float(bool(_success_01)),
        "benchmark_spl_1m": float(spl_1m),
        "benchmark_spl_0_1m": float(spl_01),
        # STOP-independent reach diagnostic, and never a success number (§6).
        "reach_1m": float(min_d2g is not None and min_d2g <= 1.0),
        "soft_spl": compute_soft_spl(
            dist_to_goal_final=world.geodesic_distance(final_pose.position, goal_positions),
            start_end_distance=float(start_end_distance or 0.0),
            path_len_taken=path_len,
        ),
        "start_end_distance_m": float(start_end_distance or 0.0),
        "path_len_m": float(path_len),
        "stopped": float(stopped),
        "n_loop_steps": float(len(steps)),
        "n_renders_in_loop": float(n_renders_in_loop),
        "n_sim_actions": float(n_actions),
        "n_no_action_steps": float(n_no_action),
        "wall_clock_s": float(wall_clock),
        # §2.5's smoke exception, as two numbers rather than a verdict.
        "start_pose_anomaly_rms": float(start_anomaly_rms),
        "start_pose_audible": float(start_anomaly_rms >= calibration.onset_rms),
        "source_separation_m": float(anomaly_episode.source.separation_m),
        "source_dy_m": float(anomaly_episode.source.height_difference_m),
        # The start-to-source drop, which the builder recorded and no run ever showed.
        # `source_dy_m` is measured against the primary ANCHOR, so it read 0.000 on an
        # episode whose source sat 2.6 m below the agent's start — and nothing in the run
        # said so. The rule now bounds this, and the number is here so a future violation
        # is visible in the record rather than inferred from raw coordinates.
        "source_dy_start_m": float(anomaly_episode.source.height_difference_to_start_m),
        "source_same_category": float(anomaly_episode.source.same_category),
        # Whether the source landed on the class's OWN anchor category, or on the geometric
        # fallback because no instance of that anchor qualified. The memory prior recalls a
        # category, so an episode with this at 0.0 is one the prior could not have got
        # right -- and a readout that pooled the two would charge the memory for episodes
        # that did not follow the rule it learned.
        "source_at_class_anchor": float(anomaly_episode.source.at_class_anchor),
        # Did the source stop at all? 0.0 is the CONTINUOUS control arm, and it is a
        # fact about the run rather than a missing measurement, so it is always present.
        "sounding_window_closed": float(window.offset_step is not None),
    }
    # Three measurements are recorded only when they exist, rather than as NaN or inf.
    # `json.dump` writes those as bare `NaN` / `Infinity`, which is not valid JSON and
    # which a strict reader rejects — and "unreachable" and "not measured" are different
    # facts from "the number happens to be missing", which is what an absent key says.
    if min_d2g is not None:
        metrics["min_d2g_m"] = float(min_d2g)
    if primary_dist_at_stop is not None:
        metrics["primary_dist_at_stop_m"] = float(primary_dist_at_stop)
    if math.isfinite(min_d2source):
        # §4.2's replacement for ADR-0001's asserted ~1 m ceiling: measured, per episode,
        # and pooled across a run into the distribution §6 asks for.
        metrics["min_d2source_m"] = float(min_d2source)
    if source_reached_step is not None:
        metrics["source_reached_step"] = float(source_reached_step)
    if dtg_source_final is not None:
        # ABSENT means "the final pose has no navmesh route to the source", which is a
        # real fact about a disconnected island and is NOT a distance of zero. The key
        # is simply not written, the same way `min_d2g_m` is not.
        metrics["dtg_source_final_m"] = float(dtg_source_final)
    if source_spl is not None:
        # WRITTEN TOGETHER OR NOT AT ALL. An SPL with no success flag beside it is
        # unreadable — 0.0 could be "did not reach" or "reached outside the ring" — and
        # `compute_source_spl` returns both from one decision, so splitting them across
        # two conditionals is how they would come to disagree.
        source_success, source_spl_value = source_spl
        metrics["source_find_sr_1m"] = float(source_success)
        metrics["source_spl"] = float(source_spl_value)
    # ALWAYS present, and it is what makes the delay distribution below readable. A
    # `onset_delay_steps` recorded only when the onset fired is RIGHT-CENSORED by
    # construction -- every value in it is smaller than the window that produced it -- so
    # a run whose window was too short reports a comfortable-looking median and hides the
    # episodes it truncated. The censored set is not recoverable from the funnel: a
    # T_ANOM_REACHED without an onset reads identically to ordinary §2.5 attrition.
    metrics["onset_fired"] = float(bool(onset.fired))
    # ONE definition of the ramp, now owned by `TailState.clip_ramp_steps` and written to
    # both the metrics bag and the record: `ceil(N/hop)`, the folds the CLIP read window
    # takes to fill. This tree has already paid for the same number meaning two things in
    # two files (`bed.py` said "once per run" while `audio/config.py` said "per step"),
    # and a ramp of 5 beside a ramp of 4 is that bug with a shorter fuse -- so the
    # expression moved onto the state rather than being written out a second time here.
    #
    # ITS VALUE IS UNCHANGED AND ITS CONSUMER HAS MOVED. It no longer corrects
    # `onset_delay_steps`: the cue readout is written whole by one sounding fold
    # (`tail.CUE_RAMP_STEPS` is 1), so there is no fill bias left to correct. What it
    # bounds now is the CLAP deferral below -- `clip_ramp_steps - 1` steps at most.
    ramp_steps = int(tail.clip_ramp_steps)
    metrics["sounding_ramp_steps"] = float(ramp_steps)
    # THE TWO BOUNDS THE CUE READOUT'S DELAY IS MADE OF, both recorded because neither is
    # re-derivable from the metrics bag. `sounding_cue_tail_steps` is the room's own
    # build-up and decay, `ceil((hop + L - 1)/hop)`; `sounding_phase_folds` is the loop's
    # period, `N // gcd(N, hop)`, which bounds how long a clip whose energy sits inside
    # one hop can keep the cue near silence.
    metrics["sounding_cue_tail_steps"] = float(tail.cue_tail_steps)
    metrics["sounding_phase_folds"] = float(tail.phase_folds)
    if onset.fired and onset.onset_step is not None:
        # THE NUMBER ADR-0017'S OPEN DURATION QUESTION IS DECIDED ON. A window has to
        # outlast this delay or the episode can never hear the source at all, so one
        # sweep at the provisional default produces the distribution a fixed count, a
        # budget fraction and a draw range all have to be chosen against.
        #
        # **Measured from the window OPENING, and since ADR-0019 the FILL BIAS IS GONE.**
        # The reading is the cue readout, which one sounding fold writes whole, so the
        # crossing carries no ramp bias at all. The curve that used to be the correction
        # -- 0.441 0.629 0.772 0.891 0.997 of settled over the first five sounding steps
        # at a fixed pose, and 0.222 0.485 0.695 0.859 0.995 for the Mac fake's 0.05 s
        # burst at hop 0.01 s -- is retired as a correction and kept here labelled as the
        # CLIP readout's, so a reader of a run written before the split can still read
        # that run.
        #
        # What is left is TWO BOUNDED TERMS, both recorded above, neither a constant an
        # analyst may subtract:
        #
        # - the ROOM's own build-up. The cue's level approaches steady state over
        #   `sounding_cue_tail_steps` folds, because the reverberation builds up over `L`
        #   samples, so a crossing can be up to `sounding_cue_tail_steps - 1` steps late.
        # - the LOOP's intermittency. The clip loops with period `sounding_phase_folds`,
        #   so a clip whose energy sits inside one hop is loud on one fold and quiet on
        #   the others and the crossing can be delayed by up to
        #   `sounding_phase_folds - 1` steps. It cannot be PREVENTED -- `observe_step` is
        #   one-shot and monotone-latching -- and 4 steps at the shipped defaults is the
        #   same magnitude as the fill ramp it replaces. What changed is that it now has a
        #   physical cause instead of being an artefact of the analysis window.
        metrics["onset_delay_steps"] = float(int(onset.onset_step) - t_anom)
    # THE CENSORED HALF OF THAT DISTRIBUTION, which is the half that decides the duration
    # question and the half nothing recorded. `onset_delay_steps` exists only where the
    # onset fired, so the sample is right-censored BY THE WINDOW BEING LONG ENOUGH: every
    # value in it is smaller than the window that produced it, and the episodes the
    # default was too short for are invisible except as a funnel shortfall that reads
    # exactly like ordinary §2.5 attrition.
    #
    # A censored episode is not empty. It carries a censoring TIME -- the sounding steps
    # it actually got, so the delay is known to exceed it -- and a distance, which says
    # whether "a bit longer" would have been enough or whether the agent was in another
    # room the whole time. Together they are a survival sample rather than a gap.
    #
    # `heard_within_window` stays conditioned on the onset, deliberately: it answers
    # *source or tail*, and a censored episode heard NEITHER. Its rate is still
    # computable over the whole arm -- sum of `heard_within_window` over the count of
    # `sounding_window_closed` -- because this key names the episodes that are missing
    # from the numerator.
    metrics["onset_delay_censored"] = float(not onset.fired)
    if not onset.fired:
        metrics["onset_delay_censored_at_steps"] = float(
            sum(1 for row in steps if window.is_sounding(row.step))
        )
        if math.isfinite(min_d2source_in_window):
            metrics["min_d2source_in_window_m"] = float(min_d2source_in_window)
        if min_route_in_window is not None:
            metrics["min_route_to_source_in_window_m"] = float(min_route_in_window)
    if clap_step is not None:
        # WHAT CLAP WAS ACTUALLY HANDED, so the confound is in `audit.json` rather than in
        # a reviewer's head. ADR-0018's bank of record and `task/clap_gate.py`'s
        # separation were both derived on full-length `render_through_ir` clips, and this
        # buffer is neither full-length nor identically aligned: `clap_window_fill` is how
        # much of the read window a sounding fold wrote, `clap_rotation_phase_samples` is
        # the rotation the loop left it at (period `N/hop`), `clap_deferred_steps` is how
        # long the classification waited for the fill, and `clap_after_offset` says the
        # buffer was a decaying tail rather than a source.
        metrics["clap_step"] = float(clap_step)
        metrics["clap_window_fill"] = float(clap_window_fill or 0.0)
        metrics["clap_rotation_phase_samples"] = float(clap_rotation_phase or 0)
        metrics["clap_deferred_steps"] = float(clap_deferred_steps or 0)
        if window.offset_step is not None:
            metrics["clap_after_offset"] = float(
                int(clap_step) >= int(window.offset_step)
            )
    elif clap_onset_step is not None:
        # The onset fired and the episode ended while the classification was still
        # waiting for the buffer to fill, so nothing was classified. Without this key the
        # episode is indistinguishable from a no-CLAP run: `anomaly_class` is None on
        # both, and one of them means "no encoder" while the other means "the agent heard
        # it and stopped before we could say what it was".
        metrics["clap_deferred_unresolved"] = 1.0
    window_record = SoundingWindowRecord(
        opens_at=int(window.opens_at),
        offset_step=window.offset_step,
        policy=cfg.sounding_policy.value,
        step_seconds=float(cfg.audio.step_seconds),
        hop_samples=int(tail.hop),
        analysis_window_samples=int(tail.window),
        max_ir_samples=int(tail.max_ir_samples),
        n_buffer_grows=int(tail.n_grows),
        # The field name predates the split and is KEPT: every audit.json on disk uses
        # it, and renaming a serialised field reinterprets every record ever written. It
        # is the CLIP tail -- see `SoundingWindowRecord`'s docstring for the role change.
        tail_steps=int(tail.clip_tail_steps),
        # The CUE tail beside it, `ceil((hop + L - 1)/hop)`: what the agent's own reading
        # takes to empty, what smoke criterion 4's fence post is measured from, and the
        # first number on this record that is evidence the geometric acoustics did work.
        cue_tail_steps=int(tail.cue_tail_steps),
        # The ramp beside the two tails, from the one definition above. `tail_steps` says
        # how long the CLIP window takes to EMPTY and this says how long it took to FILL,
        # which is what bounds the CLAP deferral.
        ramp_steps=ramp_steps,
        # THE MEASURED HALF, and since ADR-0019 it is measured on the CUE trace, so it
        # counts steps at which the ROOM was still audible rather than steps at which the
        # analysis window still held source. Its values FALL, and that is the correction.
        # `cue_tail_steps` is arithmetic off the IR's width and says how long the room
        # COULD outlive the offset step; this counts the steps the agent's own reading
        # actually stayed outside `pre_onset_rms_tol` of the bed. They come apart badly on
        # a transient clip -- `audio/tail.py` measures 0 audible steps against a
        # `cue_tail_steps` of 3 -- and `tail_is_active` cannot see the difference.
        post_offset_audible_steps=post_offset_audible_steps(
            readings=[(row.step, row.measured_rms) for row in steps],
            offset_step=window.offset_step,
            bed_rms=cfg.audio.bed_rms,
            tolerance=cfg.audio.pre_onset_rms_tol,
        ),
    )
    if window.offset_step is not None:
        offset_step = int(window.offset_step)
        metrics["offset_step"] = float(offset_step)
        # `window.duration_steps` rather than the subtraction again: one definition of
        # how long the source sounded, in the type that owns the window's boundaries.
        metrics["sounding_duration_steps"] = float(window.duration_steps or 0)
        # Counted through `SoundingWindow.is_silent` rather than subtracted here, because
        # the offset step being the FIRST silent step is that type's decision and it is
        # one CONTEXT.md's two sentences can be read either way. A subtraction at this
        # call site is a second copy of the boundary, and the copy is the one that drifts.
        metrics["silent_phase_steps"] = float(
            sum(1 for row in steps if window.is_silent(row.step))
        )
        metrics["post_offset_audible_steps"] = float(
            window_record.post_offset_audible_steps or 0
        )
        if onset.fired and onset.onset_step is not None:
            # Hearing the SOURCE and hearing its REVERB are different events. The tail
            # runs `tail_steps` past the offset step and only reaches the bed at its
            # end, so `onset_step > offset_step` is REACHABLE and is not a bug -- this
            # key is what separates the two in the record.
            metrics["heard_within_window"] = float(
                int(onset.onset_step) < offset_step
            )
        eligible, reached_after = sws_episode(
            offset_step=offset_step,
            n_loop_steps=len(steps),
            source_reached_step=source_reached_step,
        )
        metrics["sws_eligible"] = float(eligible)
        if eligible:
            # ADR-0017 line 49, held in code rather than in a comment: an eligible
            # episode is one an SWS would count, and counting it without the
            # accumulation buffer would measure the metric on a hard cut to the bed
            # rather than on a decaying source. This raises BEFORE the key is written,
            # so no artefact carries a per-episode SWS the tail did not earn.
            if not tail_is_active(window_record):
                raise TailNotActiveError(
                    "episode {} ran past its offset step {} but the accumulator "
                    "recorded no folded render ({!r}) -- ADR-0017 bars an SWS measured "
                    "without the reverb tail".format(
                        int(index), window.offset_step, window_record
                    )
                )
            metrics["sws"] = float(reached_after)
    metrics.update({key: float(value) for key, value in proposer.stats().items()})
    metrics.update({key: float(value) for key, value in counters.items()})
    if memory_prior is not None:
        # The numeric half only. The recalled category is a string and lives in the typed
        # field above -- `metrics` is `Mapping[str, float]` and every reader does
        # `float(value)` on it.
        metrics.update(memory_prior.as_metrics())

    audit = EpisodeAudit(
        episode_index=int(index),
        scene_id=episode.scene_id,
        localization_arm=cfg.localization.value,
        detector_arm=cfg.detector.value,
        # The four ADR-0018 arms, per episode. `summary.json` carries the run config once
        # per RUN, and every comparison this tree makes is per EPISODE
        # (`tools/episode_diff.py` pairs by index), so an arm recorded only at run level
        # is an arm a paired diff cannot check it was comparing like with like on.
        climb_rule=cfg.climb_rule.value,
        lateral_cue=cfg.lateral_cue.value,
        cast_policy=cfg.cast_policy.value,
        ir_policy=cfg.ir_policy.value,
        # The matrix cell, on the same terms and for the same reason as the four arms
        # above. `None` here means no memory arm ran, which `MemoryCondition.NONE` also
        # means -- so a run WITH a context writes the string even for `NONE`, and a run
        # without one writes nothing. The two are told apart by the presence of a value.
        memory_condition=(None if memory is None else str(memory.condition.value)),
        # Exactly one of these, or neither. Neither means the prior was never consulted:
        # the source never went silent while the agent was investigating.
        memory_prior_category=(None if memory_prior is None else memory_prior.category),
        memory_prior_miss=(None if memory_miss is None else memory_miss.value),
        source_xyz=source,
        t_anom=t_anom,
        sounding_window=window_record,
        source_reached_step=source_reached_step,
        dist_at_stop=source_dist_at_stop,
        funnel_stage=_funnel_stage(
            n_steps=len(steps),
            t_anom=t_anom,
            onset_fired=onset.fired,
            entered_investigate=entered_investigate,
            investigated=state.investigated,
            resumed=state.resumed,
        ),
        onset=OnsetRecord(
            onset_step=onset.onset_step,
            pre_onset_rms=onset.pre_onset_rms,
            n_pre_onset_readings=onset.n_pre_onset_readings,
            provenance_asserted=True,
        ),
        calibration=CalibrationRecord(
            onset_rms=calibration.onset_rms,
            bed_rms=calibration.bed_rms,
            separation_db=calibration.separation_db,
            n_poses=calibration.n_poses,
            global_volume=calibration.global_volume,
            passed=calibration.passed,
            cue_render_scatter=calibration.cue_render_scatter,
            cue_scatter_repeats=calibration.cue_scatter_repeats,
            clip_render_scatter=calibration.clip_render_scatter,
            clip_scatter_repeats=calibration.clip_scatter_repeats,
            single_render_scatter=calibration.single_render_scatter,
            single_render_repeats=calibration.single_render_repeats,
            cue_phase_folds=calibration.cue_phase_folds,
            cue_phase_crest=calibration.cue_phase_crest,
            cue_phase_min_ratio=calibration.cue_phase_min_ratio,
            cue_phase_aggregation=calibration.cue_phase_aggregation,
            profile=calibration.profile,
        ),
        audio_context=getattr(handle, "report", None),
        steps=tuple(steps),
        metrics=metrics,
    )
    return EpisodeResult(report=report, audit=audit)


def _steer(
    proposer: FrontierProposer,
    pose: Pose,
    waypoint: Optional[Xyz],
    *,
    follow: Callable[[Xyz], Optional[str]],
    snap_point: Callable[[Xyz], Optional[Xyz]],
    geodesic: Callable[[Xyz, Xyz], Optional[float]],
    planner: PlannerConfig,
    divert: Optional[Candidate],
) -> Tuple[Optional[Xyz], Optional[str], Dict[str, int]]:
    """The next action toward a waypoint, re-proposing once if the current one is spent.

    Two attempts, because the two ways a waypoint stops being answerable both resolve by
    picking another one: the follower reports arrival (``None``), or it reports no route
    (``NoRouteError``). A single attempt would spend a whole step standing still after
    every arrival.

    **After two attempts it returns ``None`` and the runner takes no simulator step.**
    That is deliberate: the old tree's answer here was a straight-line fallback, which is
    how "a waypoint was chosen" and "the agent got there" came apart with nothing in the
    code marking where. A recorded ``action: null`` is a visible symptom in the per-step
    record; an invented action is a trajectory that lies.
    """
    counters: Dict[str, int] = {}
    for _attempt in range(2):
        if waypoint is None or proposer.is_decision_step():
            waypoint, _source, counters = _choose_waypoint(
                proposer,
                pose,
                snap_point=snap_point,
                geodesic=geodesic,
                planner=planner,
                divert=divert,
            )
        try:
            action = follow(waypoint)
        except NoRouteError:
            proposer.request_replan()
            waypoint = None
            continue
        if action is not None:
            return waypoint, action, counters
        proposer.request_replan()
        waypoint = None
    return waypoint, None, counters


def _funnel_stage(
    *,
    n_steps: int,
    t_anom: int,
    onset_fired: bool,
    entered_investigate: bool,
    investigated: bool,
    resumed: bool,
) -> FunnelStage:
    """§6's staged funnel, as the highest stage this episode reached.

    The stages nest, so this is a ladder rather than a classification. ``T_ANOM_REACHED``
    is ``n_steps > t_anom`` because the step indices are zero-based — an episode of
    exactly ``t_anom`` steps ended on the step *before* the source started playing.

    **The nesting is enforced rather than assumed** (ticket 26). The first version read
    each flag independently on the premise that *an episode that resumed necessarily
    investigated*, and the abort path falsifies it: the step-budget abort transitions
    straight to RESUME with ``investigated`` False, so a stage-4 episode was promoted to
    6 and the first box run printed a 6/6 funnel while its own trace showed six
    INVESTIGATE entries and five aborts. Smoke criterion 5 is read off this number, so an
    over-credited stage is the gate asserting a loop that did not run.
    """
    stage = FunnelStage.RUN
    if n_steps > int(t_anom):
        stage = FunnelStage.T_ANOM_REACHED
    if stage < FunnelStage.T_ANOM_REACHED:
        return stage
    if not onset_fired:
        return stage
    stage = FunnelStage.ONSET_FIRED
    if not entered_investigate:
        return stage
    stage = FunnelStage.INVESTIGATE_ENTERED
    if not investigated:
        return stage
    stage = FunnelStage.SOURCE_REACHED
    if resumed:
        stage = FunnelStage.PRIMARY_RESUMED
    return stage


# ----------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------


def _pick_scene(split_dir: str, scenes_dir: str, wanted: str) -> Any:
    """The named scene, or the first one in the split whose mesh is on this machine.

    The search exists because the box carries a partial HM3D download and a run that
    fails on the fourth scene, forty minutes in, is a worse failure than one that starts
    somewhere it can.
    """
    labels = [wanted] if wanted else list(available_scenes(split_dir))
    tried = []
    for label in labels:
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            return dataset
        tried.append(dataset.scene_path)
    raise FileNotFoundError(
        "no scene mesh found under {} (tried {} scene(s); first few: {})".format(
            scenes_dir, len(tried), ", ".join(tried[:3])
        )
    )


def run(
    cfg: RunConfig,
    *,
    progress: Optional[Callable[[str], None]] = None,
    memory_condition: Optional[MemoryCondition] = None,
    memory_prior_stores: Optional[Tuple[SemanticStore, EpisodicStore]] = None,
    memory_k: int = 5,
) -> RunSummary:
    """Assert the environment, build the dataset, run the episodes, write the artefacts.

    The box-only half. Everything expensive is constructed once, before the first step:
    the clip, the models, the simulator, the audio sensor and its armed guard (map
    requirement 9).

    ``env_report.json`` carries the resolved **configuration** beside the resolved
    environment. One artefact, two kinds of fact, and both answer the same question about
    a run directory a year from now — which is the argument ``agent/config.py`` makes for
    ``DetectorConfig`` existing at all.

    **The matrix cell, not on ``RunConfig``.** ``memory.store.MemoryCondition``'s own
    docstring gives the reason: the config layer has no edge to ``memory/`` (ADR-0013),
    and a cell selected by a branch inside the config rather than by which stores the
    caller built is exactly what that docstring forbids. ``memory_prior_stores`` is the
    UNFILTERED pair a prior pass built (``memory_build.dump_stores`` / ``load_stores``);
    ``stores_for_cell`` filters it to the one condition, once, before the episode loop --
    the filter depends only on the scene under test and ``cfg.anomaly_class``, both fixed
    for the whole call, so building it once and reusing it every episode is exact, not an
    approximation. Passing a condition with no stores raises rather than silently running
    every episode under ``MemoryCondition.NONE``, which would be a matrix cell that looks
    populated and measures nothing.
    """
    say = progress if progress is not None else print

    say("env_check: probing (clap={})".format(cfg.clap))
    env = assert_env(clap=cfg.clap)
    say(env.summary())

    split_dir = find_split_dir(cfg.split, root=cfg.data_root)
    scenes_dir = find_scenes_dir(root=cfg.data_root)
    dataset = _pick_scene(split_dir, scenes_dir, cfg.scene)

    memory: Optional[MemoryContext] = None
    if memory_condition is not None:
        if memory_prior_stores is None:
            raise ValueError(
                "memory_condition={!r} was given with no memory_prior_stores; the "
                "matrix cell cannot be realised without the stores a prior pass "
                "built (see memory_build.dump_stores / load_stores)".format(
                    memory_condition
                )
            )
        full_semantic, full_episodic = memory_prior_stores
        cell_semantic, cell_episodic = stores_for_cell(
            full_semantic,
            full_episodic,
            memory_condition,
            sound_class=cfg.anomaly_class,
            scene=dataset.scene_label,
        )
        memory = MemoryContext(
            condition=memory_condition,
            semantic=cell_semantic,
            points_by_category=points_by_category_for_cell(
                dataset, cell_episodic, dataset.scene_label
            ),
            k=memory_k,
        )
        say("memory: {} ({} semantic row(s)) -- {}".format(
            memory_condition.value, len(memory.semantic), RUN_DISCLOSURE
        ))

    try:
        build = build_anomaly_episodes(
            dataset,
            anomaly_class=cfg.anomaly_class,
            # The source now goes at the object its CLASS belongs at, when the scene has a
            # qualifying one. Before this the placement was geometric and a semantic memory
            # had nothing to predict -- see `place_anomaly_source` rule 4. The lookup is
            # here rather than inside the builder because the builder imports no `audio/`.
            anchor_category=anchor_of_run_class(cfg.anomaly_class),
            t_anom=cfg.t_anom,
            category=cfg.category,
            n_episodes=cfg.n_episodes,
            min_sep_m=cfg.min_source_sep_m,
            max_dy_m=cfg.max_source_dy_m,
            min_start_sep_m=cfg.min_source_start_sep_m,
        )
    except EmptyDatasetError as exc:
        # A 0% yield is the most informative point a denominator has, and it used to reach
        # nobody: the raise happened here, before `write_run_summary`, so a scene that
        # could place nothing left no record and `yield_report` aggregated the scenes that
        # yielded *something* while calling the result the yield of all of them. yield-1
        # lost `mL8ThkuaVTM` (99 candidates, 0 placed) that way, in both invocations.
        #
        # Written, then re-raised. The record is the measurement; the raise is still true
        # — a run asked for episodes and produced none.
        say(exc.build.summary())
        write_run_summary(
            cfg.run_dir,
            RunSummary(
                run_dir=str(cfg.run_dir),
                scene_label=exc.scene_label,
                n_episodes=0,
                # Every stage zero, spelled out rather than left empty: a funnel with no
                # keys and a funnel of zeros read the same to `dict.get(..., 0)` and mean
                # different things to a person.
                funnel={stage.name: 0 for stage in FunnelStage},
                # No episode ran, so there is nothing to tally and NOT_RUN is the honest
                # reading. `None` rather than a tally of zeros, which would publish an
                # SWS denominator of 0 as if it had been measured.
                silent_phase=None,
                skipped=exc.build.skipped,
            ).as_dict(),
            overwrite=cfg.overwrite,
        )
        raise
    say(build.summary())

    clip_path = resolve_anomaly_clip(cfg.anomaly_class, cfg.anomaly_clip, cfg.audio.clip_dir)
    clip = load_anomaly_clip(
        clip_path, cfg.audio.sample_rate, cfg.audio.target_norm_rms_db
    )
    say("anomaly clip: {} ({} samples at {} Hz)".format(
        clip_path, len(clip), cfg.audio.sample_rate
    ))

    # THE ACCUMULATOR'S ONE CONFIGURATION REFUSAL, PAID FOR HERE AND IN SECONDS. Both
    # numbers are known now: the hop is `round(step_seconds * sample_rate)` and the read
    # window is the clip that was just loaded. Left to `open_tail` inside `run_episode`
    # this raised PER EPISODE, after habitat, CLAP, the scene and 16 calibration renders
    # had all been paid for, and it raised in the accumulator's own words -- samples, a
    # read window, a sensor -- about a mistake that was made in `AudioConfig`. A config
    # typo should cost seconds and name the field it is in.
    preflight_hop = hop_samples(
        step_seconds=cfg.audio.step_seconds, sample_rate=cfg.audio.sample_rate
    )
    clip_seconds = len(clip) / float(cfg.audio.sample_rate)
    if preflight_hop >= len(clip):
        raise ValueError(
            "AudioConfig.step_seconds is {:.6g} s, so one simulator step emits {:.6g} s "
            "({} samples) of source -- at or past the whole {:.6g} s ({} sample) anomaly "
            "clip {}. The accumulator's read window IS the clip, so consecutive steps "
            "would share no samples and everything between two steps would be dropped: a "
            "different sensor, which has to be asked for rather than fallen into. Set "
            "step_seconds below {:.6g} s or stage a longer clip.".format(
                float(cfg.audio.step_seconds),
                preflight_hop / float(cfg.audio.sample_rate),
                preflight_hop,
                clip_seconds,
                len(clip),
                clip_path,
                clip_seconds,
            )
        )
    # Constructed anyway rather than trusted: ADR-0014's rule is that a capability is
    # exercised, and the check above states the bound while this proves the buffer the
    # episodes will build can actually be built.
    preflight = open_tail(window=len(clip), hop=preflight_hop)
    # `clip_ramp_steps` off the state rather than the expression written out again. This
    # line and the record's `ramp_steps` were two copies of `ceil(N/hop)`, which is the
    # shape the record's own comment names as a bug with a shorter fuse.
    say("reverb tail: hop {} samples, read window {}, ramp {} steps".format(
        preflight.hop,
        preflight.window,
        preflight.clip_ramp_steps,
    ))

    clap_encoder = None
    if cfg.clap:
        from earshot.task.models import load_clap_encoder

        clap_encoder = load_clap_encoder()
        say("CLAP: loaded")

    write_env_report(
        cfg.run_dir,
        dict(env.as_dict(), run_config=cfg.as_dict(), scene=dataset.scene_label),
        overwrite=cfg.overwrite,
    )

    # Imported here rather than at module scope: `sim/world.py` imports habitat_sim, so
    # a top-level import would make every Mac test in this file's suite uncollectable —
    # the same reason `audio/clips.py` imports scipy inside its two callers.
    from earshot.audio.sensor import AudioSensorHandle
    from earshot.audio.spec import audio_sensor_spec
    from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs

    spec, binaural = audio_spec_parts()
    audio_sensor_spec(spec, cfg.audio, binaural)
    audio_uuid = str(spec.uuid)

    funnel: Dict[str, int] = {stage.name: 0 for stage in FunnelStage}
    audits: List[EpisodeAudit] = []
    world = World(dataset.scene_path, list(camera_sensor_specs()) + [spec])
    try:
        world.seed_navmesh(cfg.seed)
        handle = None
        for index, anomaly_episode in enumerate(build.episodes):
            episode = anomaly_episode.episode
            say("episode {}: {} in {}, source at {} ({})".format(
                index,
                anomaly_episode.primary_category,
                episode.scene_label,
                anomaly_episode.source.anomaly_object,
                anomaly_episode.source.position,
            ))
            world.set_pose(episode.start_position, episode.start_rotation)
            if handle is None:
                handle = AudioSensorHandle(
                    world.sensor_handle(audio_uuid),
                    world.observe,
                    anomaly_episode.source.position,
                    uuid=audio_uuid,
                )
                say("  audio context armed: {} vertices, IR {}".format(
                    handle.report.n_vertices, handle.report.ir_shape
                ))
            else:
                handle.set_source(anomaly_episode.source.position)

            calibration, poses = calibrate_episode(
                world, handle, anomaly_episode.source.position, clip, cfg
            )
            say("  calibration: onset_rms {:.6g}, separation {:.2f} dB over {} pose(s)".format(
                calibration.onset_rms, calibration.separation_db, len(poses)
            ))

            result = run_episode(
                world,
                handle,
                anomaly_episode,
                cfg,
                clip=clip,
                detector=make_detector(cfg, world, anomaly_episode),
                index=index,
                clap_encoder=clap_encoder,
                calibration=calibration,
                memory=memory,
                progress=say,
            )
            write_episode(
                cfg.run_dir,
                index,
                result.report,
                result.audit,
                overwrite=cfg.overwrite,
            )
            audits.append(result.audit)
            for stage in FunnelStage:
                if result.audit.funnel_stage >= stage:
                    funnel[stage.name] += 1
            say("  funnel: {}   audio {}".format(
                result.audit.funnel_stage.name, result.audit.audio_render_summary()
            ))
    finally:
        world.close()

    summary = RunSummary(
        run_dir=cfg.run_dir,
        scene_label=dataset.scene_label,
        n_episodes=len(build.episodes),
        funnel=funnel,
        skipped=build.skipped,
        silent_phase=silent_phase_tally(audits),
    )
    say(summary.summary())
    write_run_summary(cfg.run_dir, summary.as_dict(), overwrite=cfg.overwrite)
    return summary
