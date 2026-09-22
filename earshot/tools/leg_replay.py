"""Does a finished cast leg know which way the source is? ADR-0029's gate, run off-box.

    python -m earshot.tools.leg_replay runs/abl-2/full runs/oracle-1/full runs/oracle-2/full

**ADR-0029 proposes that the cast read its own legs**, and this measures whether a leg
has anything to read before any controller code ships or any night is booked. Today a
leg is one turn and ``CAST_STEPS`` forwards, and the next leg turns by a rule that looks
only at the leg's ordinal. ``READ_LEGS`` would fit the cue against the agent's own
travel at each leg's end: LOUDER keeps the heading, QUIETER reverses it, INCONCLUSIVE
casts as today.

Every argument is one realizable arm's directory, ``<tag>/<arm>/<scene>/``, and the
gate names three: ``full`` in ``abl-2``, ``oracle-1`` and ``oracle-2``. That is the same
arm at identical behaviour, rendered three times.

What a leg is, and which legs get a verdict
-------------------------------------------

**Legs come from the controller's own cycle, replayed and checked step by step.** The
replay rebuilds ``rising`` and the plateau count through ``detour_report``'s
``rising_flags`` and ``plateau_index``, which call the controller's own ``is_rising``.
It then checks every detour step's reconstructed action against the recorded
``realizable_action``. A leg with one step that disagrees is UNVERIFIED and is never
graded.

**Only a COMPLETED leg gets a verdict**: one whose ``CAST_STEPS`` forwards all ran and
whose next step is the next leg's turn. That is the one place ``READ_LEGS`` acts. A
surge, a STOP or the end of the detour ends the leg first, and the reader would never
have been asked. The replay counts those legs and does not grade them.

The verdict is ``controller.leg_verdict`` over ``controller.leg_t``, imported and never
re-spelled, so the replay prices the function the arm would run. Its readings are the
ones the controller would hold at the next leg's turn: the pose after the leg's turn,
through the pose after its last forward. The audit records position but not yaw, so the
leg's axis is its net displacement, and each reading's displacement is its projection
onto that axis.

How a verdict is graded
-----------------------

Against ``Δroute``: ``geodesic_to_source`` at the leg's last reading minus the value at
its first. That is analyst-only, and it never reaches the controller. A leg is
**informative** when ``|Δroute| >= 0.5 m``. A leg that ran across the source's bearing
has no right answer, so it is not graded; it gives the false-decisive rate instead.
LOUDER is right when the route fell by at least that much, and QUIETER when it rose by at
least that much. Chance is 50%.

The gate, pre-registered in ADR-0029 before this ran
-----------------------------------------------------

- **BUILD**: at one ``T_LEG`` from ``T_LEG_GRID``, BOTH branches are right at least 75% of
  the time pooled and at least 70% in every run, and together they are decisive on at
  least 25% of informative legs.
- **ONE BRANCH**: one branch passes those accuracy tests and is decisive on at least 25%
  of informative legs by itself, and the other does not pass.
- **STOP**: anything else.

Where several ``T_LEG`` values pass, the one decisive on the most legs is chosen, because
that is the one that changes the most of the cast at the accuracy the gate demands.

What it cannot say
------------------

**It prices the verdict, not the sweep.** These legs were walked under blind alternation.
A reader that acts on them changes which legs get walked, so the replay says how often a
verdict would be right, and not how many episodes would reach the source.

**``T_LEG`` is a critical value, not a p-value.** Readings along a leg share the cue tail
and the loop phase, so they are autocorrelated. The false-decisive column is the
measured false-positive rate, and it is the one to read.

**On a completed leg, ``is_rising`` said "not rising" at every step, by construction**,
because a surge would have ended the leg. ADR-0029's fifth item, "how often ``is_rising``
fired on those same legs", is therefore zero on every leg graded here. The replay reports
the useful form of that number: how many completed legs walked 0.5 m or more toward the
source with no surge. The current reader missed each of them, and a LOUDER verdict is what
``READ_LEGS`` would add there.

By sounding state, added after the gate read STOP
-------------------------------------------------

The first readout found every leg pulled toward QUIETER, whichever way it walked.
``full`` runs ADR-0017's windowed task, so the source stops at ``offset_step`` and a
detour can outlast it. ``sounding_state`` puts each completed leg in one state off the
record's own window: read before the offset, spanning it, or read on the bed alone.
``by_sounding`` then prints the grid and the median ``t`` per state. **It is not a
gate**, because the split was chosen after the result.

The loop, removed, added after the sounding split
-------------------------------------------------

The sounding split left legs read while the source sounded pulled quieter whichever way
they walked. The candidate it named is the clip's loop. The source loops every
``sounding_phase_folds`` steps, and a nine-reading fit spans no whole number of loops,
so the loop can put a slope into a leg. ``same_phase_t`` asks ``leg_t``'s question of
readings one whole loop apart, which cancels any level that repeats with the loop,
wherever the clip's energy sits in it. ``loop_removed`` prints both readers over the
same sounding legs. If the loop is the pull, it goes from the same-phase reader and the
direction stays. **It is not a gate** either.

By straight line, added after the loop check
--------------------------------------------

The pull stayed with the loop cancelled. The next suspect is the grader's axis: a leg
is graded on the route, and the level may follow the straight line through the walls
instead. ``by_line`` grades the same sounding legs on the change in horizontal
distance to ``source_xyz``, with both readers, and counts how often the route and the
line agree on a leg's direction. If the pull goes here, it was the axis. If it stays,
the level falls along a leg whatever the geometry. **Not a gate.**

Read-only, no GPU, seconds to minutes. A run that lacks ``realizable_action``,
``geodesic_to_source`` or ``CalibrationRecord.cue_render_scatter``, or that is not
``full``'s rule, is refused by name.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.agent.controller import (
    ACT_STOP,
    CAST_STEPS,
    LEG_INCONCLUSIVE,
    LEG_LOUDER,
    LEG_QUIETER,
    SCAN_STEPS,
    climb_eps,
    leg_t,
    leg_verdict,
)
from earshot.config import CastPolicy, ClimbRule, LateralCue, Localization
from earshot.report.artifacts import episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, SoundingWindowRecord
from earshot.tools.detour_report import (
    BAND_EDGES_M,
    load_traces,
    plateau_index,
    rising_flags,
    rule_action,
)
from earshot.types import Xyz

__all__ = [
    "COMPLETED",
    "CUT_BY_SURGE",
    "CUT_BY_STOP",
    "CUT_BY_END",
    "UNVERIFIED",
    "BUILD",
    "ONE_BRANCH",
    "STOP",
    "INFORMATIVE_ROUTE_M",
    "T_LEG_GRID",
    "GATE_ACCURACY",
    "GATE_RUN_ACCURACY",
    "GATE_DECISIVE_RATE",
    "PREREGISTERED_RUNS",
    "Leg",
    "EpisodeReplay",
    "RunReplay",
    "episode_legs",
    "load_run",
    "score",
    "evaluate_gate",
    "field_by_scene",
    "SOUNDING",
    "SPANS_OFFSET",
    "SILENT",
    "SOUNDING_STATES",
    "sounding_state",
    "by_sounding",
    "same_phase_t",
    "loop_removed",
    "by_line",
    "format_report",
    "main",
]

# How a leg that started ended. Only COMPLETED legs reach the point where READ_LEGS acts.
COMPLETED = "completed"
CUT_BY_SURGE = "cut_by_surge"  # `is_rising` fired mid-leg, and the surge acted on the cue
CUT_BY_STOP = "cut_by_stop"    # the confirm fired mid-leg, and the detour ended there
CUT_BY_END = "cut_by_end"      # the budget or the episode ran out mid-leg
UNVERIFIED = "unverified"      # a step's reconstruction disagreed with the record

# The three branches ADR-0029 pre-registered.
BUILD = "BUILD"
ONE_BRANCH = "ONE BRANCH"
STOP = "STOP"

# Where the source was while a completed leg was read (see `sounding_state`). A leg
# whose record cannot say carries None and is reported as unknown.
SOUNDING = "sounding"          # every reading before the offset step
SPANS_OFFSET = "spans_offset"  # the source, or its cue tail, stopped during the leg
SILENT = "silent"              # every reading is the bed alone
SOUNDING_STATES = (SOUNDING, SPANS_OFFSET, SILENT)

# provenance: ADR-0029 — the route change a leg needs before its verdict has a right
# answer. Two forwards' worth; a nominal leg is eight.
INFORMATIVE_ROUTE_M = 0.5

# provenance: ADR-0029 — the critical values the gate may choose among. Fixed here, in
# the commit that builds the replay, before it has read a single run.
T_LEG_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0)

# provenance: ADR-0029 — the gate's three numbers. Judgment, not derivation, and set
# before the replay ran; see the ADR's branches.
GATE_ACCURACY = 0.75
GATE_RUN_ACCURACY = 0.70
GATE_DECISIVE_RATE = 0.25

# How many independent renders ADR-0029's gate names. Fewer can be read, and the report
# says it is not the pre-registered gate.
PREREGISTERED_RUNS = 3

# provenance: fake — where the breakdowns are printed when the gate chose no T_LEG and
# none was asked for. The grid value nearest 2.36, the textbook two-sided 5% critical
# value for a nine-reading fit. It never enters the gate.
DISPLAY_T_LEG = 2.5

# One turn, then the leg's forwards: the step count from one leg's turn to the next's.
# Read from the controller, never re-spelled.
LEG_PERIOD = 1 + CAST_STEPS

# Net-displacement buckets for "is length what limits a verdict". A clear leg is
# CAST_STEPS x 0.25 m = 2.0 m.
LENGTH_EDGES_M = (0.0, 0.5, 1.0, 1.5)

# `full`'s rule on the four fields that change what the cast cycle does. Values are
# read off the enums, so a renamed value fails here rather than refusing every run.
FULL_ARM = (
    ("localization_arm", Localization.REALIZABLE.value),
    ("climb_rule", ClimbRule.LIVE.value),
    ("lateral_cue", LateralCue.LIVE.value),
    ("cast_policy", CastPolicy.CAST.value),
)

_BRANCHES = (LEG_LOUDER, LEG_QUIETER)


@dataclass(frozen=True)
class Leg:
    """One cast leg that started. The verdict inputs are set on COMPLETED legs only."""

    run: str
    scene: str
    episode: int
    start_step: int
    outcome: str
    t: Optional[float] = None
    length_m: Optional[float] = None
    route_start_m: Optional[float] = None
    delta_route_m: Optional[float] = None
    sounding: Optional[str] = None
    # The loop's period in steps, off the record, and the same-phase reader's t. That t
    # is set on SOUNDING legs only: where the source stopped, the loop stopped with it.
    phase_folds: Optional[int] = None
    t_same_phase: Optional[float] = None
    # Horizontal straight-line distance to the source at the leg's first reading, and its
    # change to the last: the axis the route was chosen over (see `by_line`).
    line_start_m: Optional[float] = None
    delta_line_m: Optional[float] = None


@dataclass(frozen=True)
class EpisodeReplay:
    """One episode's legs, and how many of its detour steps the reconstruction matched."""

    legs: Tuple[Leg, ...]
    n_steps_checked: int
    n_steps_agree: int
    has_detour: bool
    eps_measured: bool


@dataclass(frozen=True)
class RunReplay:
    """One run directory's legs, or the named reasons it cannot be replayed."""

    label: str
    path: str
    n_episodes: int
    n_detours: int
    legs: Tuple[Leg, ...]
    n_steps_checked: int
    n_steps_agree: int
    n_eps_unmeasured: int
    refusals: Tuple[str, ...]


def _share(numerator: int, denominator: int) -> Optional[float]:
    """A ratio, or None for an empty denominator. 0 of 0 is not 0%."""
    return (float(numerator) / float(denominator)) if denominator else None


def _detour(audit: EpisodeAudit) -> List[int]:
    """Indices into ``audit.steps`` of the detour, where the realizable rule ran.

    The first step that recorded ``realizable_action`` is the tick that opened the
    detour, where ``step_controller`` starts a fresh plateau count. The detour runs until
    the first step with none: the budget's abort tick records none, and there is one
    detour per episode.
    """
    indices: List[int] = []
    for index, row in enumerate(audit.steps):
        if row.realizable_action is not None:
            indices.append(index)
        elif indices:
            break
    return indices


def _along_leg(positions: Sequence[Xyz]) -> Tuple[List[float], float]:
    """Each position's displacement along the leg's net heading, and the leg's length.

    Horizontal only. Zero length gives zero displacements, which ``leg_t`` reads as no
    spread and so as no verdict.
    """
    first, last = positions[0], positions[-1]
    dx, dz = last.x - first.x, last.z - first.z
    length = math.hypot(dx, dz)
    if length <= 0.0:
        return [0.0 for _ in positions], 0.0
    return (
        [((p.x - first.x) * dx + (p.z - first.z) * dz) / length for p in positions],
        length,
    )


def episode_legs(audit: EpisodeAudit, *, run: str, scene: str) -> EpisodeReplay:
    """Every cast leg the episode's detour started, classified and measured. Pure."""
    detour = _detour(audit)
    scatter = None if audit.calibration is None else audit.calibration.cue_render_scatter
    if not detour:
        return EpisodeReplay((), 0, 0, False, scatter is not None)

    # `rising` over the WHOLE episode, then sliced to the detour: the controller's history
    # has been filling since step 0, so a detour-local recomputation would invent a surge
    # on the first detour step (see `detour_report.rising_flags`).
    all_flags = rising_flags([r.measured_rms for r in audit.steps], eps=climb_eps(scatter))
    rows = [audit.steps[i] for i in detour]
    flags = [all_flags[i] for i in detour]
    counts = plateau_index(flags)
    recorded = [r.realizable_action for r in rows]
    # STOP needs `visual_confirm`, which no record carries, so a STOP step is excluded
    # from the check by name rather than counted as agreement.
    agree = [
        rec == ACT_STOP
        or rule_action(flag, r.lateral_sign, plateau_steps=count) == rec
        for r, flag, count, rec in zip(rows, flags, counts, recorded)
    ]
    checked = sum(1 for rec in recorded if rec != ACT_STOP)
    agreed = sum(1 for rec, ok in zip(recorded, agree) if rec != ACT_STOP and ok)
    period = _loop_period(audit)

    legs: List[Leg] = []
    for start, (flag, count) in enumerate(zip(flags, counts)):
        if flag or count < SCAN_STEPS or (count - SCAN_STEPS) % LEG_PERIOD:
            continue
        if recorded[start] == ACT_STOP:
            continue  # the confirm fired where the leg would have opened: no leg
        outcome = _leg_outcome(start, flags, recorded, agree)
        leg = Leg(
            run=run, scene=scene, episode=int(audit.episode_index),
            start_step=int(rows[start].step), outcome=outcome)
        if outcome == COMPLETED:
            leg = _measured(
                leg, rows[start + 1 : start + LEG_PERIOD + 1], audit.sounding_window,
                period, audit.source_xyz)
        legs.append(leg)
    return EpisodeReplay(tuple(legs), checked, agreed, True, scatter is not None)


def _leg_outcome(
    start: int, flags: Sequence[bool], recorded: Sequence[Optional[str]],
    agree: Sequence[bool],
) -> str:
    """How the leg opened at ``start`` ended, walking to the next leg's turn."""
    for offset in range(LEG_PERIOD + 1):
        index = start + offset
        if index >= len(recorded):
            return CUT_BY_END
        if recorded[index] == ACT_STOP:
            return CUT_BY_STOP
        if not agree[index]:
            return UNVERIFIED
        if offset and flags[index]:
            return CUT_BY_SURGE
    return COMPLETED


def sounding_state(
    first_step: int, last_step: int, window: Optional[SoundingWindowRecord]
) -> Optional[str]:
    """Where the source was while a leg's readings were taken, from its window. Pure.

    The fence posts are the smoke gate's (criterion 4). ``offset_step`` is the first
    silent step, and the cue carries the room's tail until
    ``offset_step + cue_tail_steps - 1``, the first step whose reading is exactly the bed.
    So a leg is SOUNDING if its last reading comes before the offset step, SILENT if its
    first reading is at or after the bed step, and SPANS_OFFSET otherwise: somewhere in
    it the level fell because the source stopped, whichever way the agent walked.

    A window with no offset step is ``WindowPolicy.CONTINUOUS``, so the source never
    stopped and every leg is SOUNDING. ``None`` where the record cannot say: no window
    at all, or no ``cue_tail_steps`` (a record from before ADR-0019's split).
    """
    if window is None:
        return None
    if window.offset_step is None:
        return SOUNDING
    if window.cue_tail_steps is None:
        return None
    offset = int(window.offset_step)
    bed_from = offset + int(window.cue_tail_steps) - 1
    if int(last_step) < offset:
        return SOUNDING
    if int(first_step) >= bed_from:
        return SILENT
    return SPANS_OFFSET


def _loop_period(audit: EpisodeAudit) -> Optional[int]:
    """The clip loop's period in steps, ``metrics["sounding_phase_folds"]``, or None.

    None where the record does not carry it. A value that is not a whole number of steps
    is a writer fault, because ``tail.phase_folds`` is an integer by construction.
    """
    value = audit.metrics.get("sounding_phase_folds")
    if value is None:
        return None
    period = float(value)
    if period < 1.0 or period != math.floor(period):
        raise ValueError(
            "episode {} records sounding_phase_folds {}. The loop's period is a whole "
            "number of steps, at least 1, so this is a writer fault".format(
                audit.episode_index, value))
    return int(period)


def same_phase_t(
    displacements: Sequence[float], levels: Sequence[float], *, lag: int
) -> Optional[float]:
    """``leg_t``'s question asked of readings one loop apart, as a t-statistic. Pure.

    Each pair is a reading and the one ``lag`` readings after it. A sounding source
    emits the same fold of its loop at both, so any level that repeats every ``lag``
    readings cancels in the pair's difference, whatever the clip's envelope. The slope is
    the least-squares fit of the level differences on the displacement differences,
    through the origin, over ``m`` pairs with ``m - 1`` degrees of freedom. Nine readings
    at a period of 5 give four pairs, and no reading is in two of them.

    ``None`` where ``leg_t`` would give none: fewer than two pairs, no travel within any
    pair, or a residual below what the levels can resolve.
    """
    xs = [float(x) for x in displacements]
    ys = [float(y) for y in levels]
    if len(xs) != len(ys):
        raise ValueError(
            "a leg needs one level per displacement: got {} displacement(s) and {} "
            "level(s)".format(len(xs), len(ys)))
    if int(lag) < 1:
        raise ValueError("a loop's period is at least one reading, got {}".format(lag))
    pairs = [(xs[j + lag] - xs[j], ys[j + lag] - ys[j]) for j in range(len(xs) - lag)]
    m = len(pairs)
    if m < 2:
        return None
    suu = sum(u * u for u, _ in pairs)
    if suu <= 0.0:
        return None
    slope = sum(u * d for u, d in pairs) / suu
    rss = sum((d - slope * u) ** 2 for u, d in pairs)
    # The floor `leg_t` uses, on the levels themselves: a difference of two readings
    # carries their rounding, not its own.
    resolution = len(ys) * sys.float_info.epsilon * max(abs(y) for y in ys)
    sd = math.sqrt(rss / (m - 1))
    if sd <= resolution:
        return None
    return slope / (sd / math.sqrt(suu))


def _line_m(position: Xyz, source: Xyz) -> float:
    """Horizontal straight-line distance, the ``xz`` axis the audit's docstring names."""
    return math.hypot(position.x - source.x, position.z - source.z)


def _measured(
    leg: Leg, readings: Sequence[Any], window: Optional[SoundingWindowRecord],
    period: Optional[int], source: Optional[Xyz],
) -> Leg:
    """A completed leg's verdict inputs, off the readings the controller would hold."""
    positions = [r.position for r in readings]
    if any(p is None for p in positions):
        raise ValueError(
            "episode {} of {}/{} has a detour step with no position. Every record since "
            "yield-1 carries one, so this is a writer fault, and a leg without positions "
            "has no axis to fit along".format(leg.episode, leg.run, leg.scene))
    displacements, length = _along_leg(positions)
    levels = [r.measured_rms for r in readings]
    first = readings[0].geodesic_to_source
    last = readings[-1].geodesic_to_source
    sounding = sounding_state(readings[0].step, readings[-1].step, window)
    line_start = None if source is None else _line_m(positions[0], source)
    return Leg(
        run=leg.run, scene=leg.scene, episode=leg.episode, start_step=leg.start_step,
        outcome=leg.outcome,
        t=leg_t(displacements, levels),
        length_m=length,
        route_start_m=None if first is None else float(first),
        delta_route_m=None if first is None or last is None else float(last) - float(first),
        sounding=sounding,
        phase_folds=period,
        t_same_phase=(
            same_phase_t(displacements, levels, lag=period)
            if sounding == SOUNDING and period is not None else None),
        line_start_m=line_start,
        delta_line_m=(
            None if source is None or line_start is None
            else _line_m(positions[-1], source) - line_start),
    )


def _label(path: pathlib.Path) -> str:
    """``abl-2/full`` for ``runs/abl-2/full``: the tag names the render, the arm the rule."""
    parts = path.resolve().parts
    return "/".join(parts[-2:])


def load_run(arm_dir: str) -> RunReplay:
    """Every leg under one realizable arm's directory, or the reasons it is refused."""
    from earshot.task.smoke import episode_indices

    root = pathlib.Path(arm_dir)
    label = _label(root)
    _, episodes_dir = run_paths(root)
    if episodes_dir.is_dir():
        return RunReplay(label, str(root), 0, 0, (), 0, 0, 0, (
            "{} is a scene directory: it holds episodes itself. Pass the arm directory "
            "above it, <tag>/<arm>.".format(root),))

    legs: List[Leg] = []
    n_episodes = n_detours = checked = agreed = unmeasured = 0
    routed = scattered = False
    mismatched: Dict[Tuple[str, Optional[str]], int] = {}
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for index in episode_indices(str(scene_dir)):
            _, audit_path = episode_paths(scene_dir, index)
            audit = read_audit(audit_path)
            n_episodes += 1
            for name, wanted in FULL_ARM:
                got = getattr(audit, name)
                if got != wanted:
                    mismatched[(name, got)] = mismatched.get((name, got), 0) + 1
            routed = routed or any(r.geodesic_to_source is not None for r in audit.steps)
            replay = episode_legs(audit, run=label, scene=scene_dir.name)
            scattered = scattered or replay.eps_measured
            if replay.has_detour:
                n_detours += 1
                unmeasured += 0 if replay.eps_measured else 1
            legs.extend(replay.legs)
            checked += replay.n_steps_checked
            agreed += replay.n_steps_agree

    refusals: List[str] = []
    if not n_episodes:
        refusals.append("no episode records under {}".format(root))
    else:
        for (name, got), count in sorted(mismatched.items(), key=lambda kv: str(kv[0])):
            refusals.append(
                "not `full`'s rule: {} is {} on {} of {} episode(s). The replay rebuilds "
                "`full`'s cast cycle and no other".format(
                    name, "unrecorded" if got is None else repr(got), count, n_episodes))
        if not n_detours:
            refusals.append(
                "no episode recorded `realizable_action`. The run predates the field or "
                "is not a realizable arm, and legs cannot be checked against the rule "
                "that ran")
        if not routed:
            refusals.append(
                "no step recorded `geodesic_to_source`. The run predates the field, and "
                "a verdict with no route change has nothing to be graded against")
        if not scattered:
            refusals.append(
                "no episode recorded `cue_render_scatter`. The run predates ADR-0019's "
                "split, so its climb ran at a threshold this replay cannot rebuild")
    return RunReplay(
        label, str(root), n_episodes, n_detours, tuple(legs), checked, agreed,
        unmeasured, tuple(refusals))


def score(legs: Sequence[Leg], t_leg: float) -> Dict[str, Any]:
    """Verdict counts at one ``t_leg``, graded on the route. Pure."""
    graded = [l for l in legs if l.outcome == COMPLETED and l.delta_route_m is not None]
    informative = [l for l in graded if abs(l.delta_route_m) >= INFORMATIVE_ROUTE_M]
    uninformative = [l for l in graded if abs(l.delta_route_m) < INFORMATIVE_ROUTE_M]
    verdicts = [leg_verdict(l.t, t_leg=t_leg) for l in informative]
    louder = [l for l, v in zip(informative, verdicts) if v == LEG_LOUDER]
    quieter = [l for l, v in zip(informative, verdicts) if v == LEG_QUIETER]
    return {
        "t_leg": float(t_leg),
        "n_informative": len(informative),
        "n_approached": sum(1 for l in informative if l.delta_route_m < 0),
        "n_receded": sum(1 for l in informative if l.delta_route_m > 0),
        "n_uninformative": len(uninformative),
        "louder_fired": len(louder),
        "louder_right": sum(1 for l in louder if l.delta_route_m < 0),
        "quieter_fired": len(quieter),
        "quieter_right": sum(1 for l in quieter if l.delta_route_m > 0),
        "uninformative_decisive": sum(
            1 for l in uninformative
            if leg_verdict(l.t, t_leg=t_leg) != LEG_INCONCLUSIVE),
    }


def _branch(pooled: Mapping[str, Any], runs: Mapping[str, Mapping[str, Any]], name: str
            ) -> Dict[str, Any]:
    """One branch's accuracy tests at one ``t_leg``. An undefined accuracy fails."""
    accuracy = _share(pooled[name + "_right"], pooled[name + "_fired"])
    per_run = {
        label: _share(row[name + "_right"], row[name + "_fired"])
        for label, row in runs.items()
    }
    defined = [a for a in per_run.values() if a is not None]
    worst = min(defined) if len(defined) == len(per_run) and defined else None
    return {
        "accuracy": accuracy,
        "per_run_accuracy": per_run,
        "worst_run_accuracy": worst,
        "rate": _share(pooled[name + "_fired"], pooled["n_informative"]),
        "passes": (
            accuracy is not None and accuracy >= GATE_ACCURACY
            and worst is not None and worst >= GATE_RUN_ACCURACY),
    }


def evaluate_gate(
    legs_by_run: Mapping[str, Sequence[Leg]],
    *,
    grid: Sequence[float] = T_LEG_GRID,
) -> Dict[str, Any]:
    """ADR-0029's pre-registered branch, over every ``T_LEG`` in ``grid``. Pure."""
    pooled_legs = [leg for legs in legs_by_run.values() for leg in legs]
    rows: List[Dict[str, Any]] = []
    for t_leg in grid:
        pooled = score(pooled_legs, t_leg)
        runs = {label: score(legs, t_leg) for label, legs in legs_by_run.items()}
        branches = {name: _branch(pooled, runs, name) for name in _BRANCHES}
        decisive = _share(
            pooled["louder_fired"] + pooled["quieter_fired"], pooled["n_informative"])
        rows.append({
            "t_leg": float(t_leg),
            "pooled": pooled,
            "runs": runs,
            "branches": branches,
            "decisive_rate": decisive,
            "false_decisive_rate": _share(
                pooled["uninformative_decisive"], pooled["n_uninformative"]),
            "build": (
                all(b["passes"] for b in branches.values())
                and decisive is not None and decisive >= GATE_DECISIVE_RATE),
        })

    verdict, chosen_t, chosen_branch = STOP, None, None
    builds = [row for row in rows if row["build"]]
    if builds:
        best = max(builds, key=lambda row: (row["decisive_rate"], -row["t_leg"]))
        verdict, chosen_t = BUILD, best["t_leg"]
    else:
        singles = [
            (row, name) for row in rows for name in _BRANCHES
            if row["branches"][name]["passes"]
            and (row["branches"][name]["rate"] or 0.0) >= GATE_DECISIVE_RATE
        ]
        if singles:
            row, name = max(singles, key=lambda pair: (
                pair[0]["branches"][pair[1]]["rate"],
                pair[0]["branches"][pair[1]]["accuracy"],
                -pair[0]["t_leg"]))
            verdict, chosen_t, chosen_branch = ONE_BRANCH, row["t_leg"], name
    return {
        "verdict": verdict,
        "t_leg": chosen_t,
        "branch": chosen_branch,
        "n_runs": len(legs_by_run),
        "preregistered": len(legs_by_run) >= PREREGISTERED_RUNS,
        "rows": rows,
    }


def field_by_scene(run_dirs: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """``detour_report``'s plateau ``sig/sc`` per scene, pooled over the runs given.

    The at-scale re-measurement of ADR-0029's 1.94, which came from one scene of oracle
    trajectories. Pooling a scene across runs is sound: it is the same room and the same
    episodes rendered again. Pooling across scenes is not, so the all-scene figure is a
    median of window values and nothing more.

    ``louder_nearer`` is the share of fitted windows whose slope on the route is
    negative, louder as the route falls. ``sig/sc`` is unsigned, so it is printed beside
    it, because a window that got quieter on the approach scores the same.
    """
    scenes: Dict[str, Dict[str, List[float]]] = {}
    for run_dir in run_dirs:
        for scene_dir in sorted(p for p in pathlib.Path(run_dir).iterdir() if p.is_dir()):
            entry = scenes.setdefault(scene_dir.name, {"sig": [], "slope": []})
            for trace in load_traces(str(scene_dir)):
                for window in trace.get("plateaus") or []:
                    if window.get("signal_to_scatter") is not None:
                        entry["sig"].append(float(window["signal_to_scatter"]))
                        entry["slope"].append(float(window["slope_per_m"]))
    return {
        name: {
            "n_windows": len(entry["sig"]),
            "median_sig_sc": statistics.median(entry["sig"]) if entry["sig"] else None,
            "louder_nearer": _share(
                sum(1 for s in entry["slope"] if s < 0), len(entry["slope"])),
            "sig": entry["sig"],
            "slope": entry["slope"],
        }
        for name, entry in sorted(scenes.items())
    }


def _pct(value: Optional[float]) -> str:
    return "  n/a" if value is None else "{:5.1f}%".format(100.0 * value)


def _bucket(edges: Sequence[float], value: Optional[float]) -> Optional[str]:
    """``"1-2"`` for a bounded bucket, ``"8+"`` for the open one, None for no value."""
    if value is None:
        return None
    for index, low in enumerate(edges):
        high = edges[index + 1] if index + 1 < len(edges) else None
        if value >= low and (high is None or value < high):
            return "{:g}+".format(low) if high is None else "{:g}-{:g}".format(low, high)
    return None


def _right_fired(row: Mapping[str, Any], name: str) -> Tuple[str, str]:
    """``("9/12", " 75.0%")``: right over fired, and the accuracy that makes."""
    right, fired = row[name + "_right"], row[name + "_fired"]
    return "{}/{}".format(right, fired), _pct(_share(right, fired))


def _accounting_lines(runs: Sequence[RunReplay]) -> List[str]:
    lines = [
        "THE LEGS. A leg is one turn and {} forwards after a {}-turn scan. Only a leg "
        "that".format(CAST_STEPS, SCAN_STEPS),
        "reaches its next turn gets a verdict: a surge, a STOP or the detour's end acts "
        "first.",
        "  {:<22} {:>8} {:>7} {:>8} {:>9} {:>6} {:>5} {:>5} {:>10}".format(
            "run", "episodes", "detours", "started", "completed", "surge", "stop", "end",
            "unverified"),
    ]
    for run in runs:
        outcomes = [leg.outcome for leg in run.legs]
        lines.append("  {:<22} {:>8} {:>7} {:>8} {:>9} {:>6} {:>5} {:>5} {:>10}".format(
            run.label, run.n_episodes, run.n_detours, len(outcomes),
            outcomes.count(COMPLETED), outcomes.count(CUT_BY_SURGE),
            outcomes.count(CUT_BY_STOP), outcomes.count(CUT_BY_END),
            outcomes.count(UNVERIFIED)))
    checked = sum(run.n_steps_checked for run in runs)
    agreed = sum(run.n_steps_agree for run in runs)
    lines.append(
        "  rule reconstruction: {} of {} detour step(s) agree with the recorded "
        "realizable_action".format(agreed, checked))
    if agreed != checked:
        lines.append(
            "  RECONSTRUCTION DISAGREES on {} step(s). Legs that contain one are "
            "UNVERIFIED and not graded.".format(checked - agreed))
    unmeasured = sum(run.n_eps_unmeasured for run in runs)
    if unmeasured:
        lines.append(
            "  {} detour(s) carried no cue_render_scatter and are replayed at the "
            "controller's fallback, the value they ran at.".format(unmeasured))
    return lines


def _population_lines(legs: Sequence[Leg]) -> List[str]:
    completed = [l for l in legs if l.outcome == COMPLETED]
    unrouted = sum(1 for l in completed if l.delta_route_m is None)
    base = score(legs, T_LEG_GRID[0])
    return [
        "",
        "  completed: {}; graded on the route: {} ({} without a route at one end)".format(
            len(completed), len(completed) - unrouted, unrouted),
        "  informative (|route change| >= {:.2f} m): {}, of which {} approached and {} "
        "receded".format(
            INFORMATIVE_ROUTE_M, base["n_informative"], base["n_approached"],
            base["n_receded"]),
        "  uninformative: {}".format(base["n_uninformative"]),
        "  WHAT THE CURRENT READER MISSED: {} completed leg(s) walked {:.2f} m or more "
        "toward the".format(base["n_approached"], INFORMATIVE_ROUTE_M),
        "  source with no surge. is_rising said \"not rising\" at every step of each, or "
        "the leg",
        "  would have been cut. A right LOUDER verdict is what READ_LEGS adds there.",
    ]


def _grid_lines(gate: Mapping[str, Any]) -> List[str]:
    lines = [
        "",
        "THE GRID. LOUDER is right when the route fell by >= {0:.2f} m, QUIETER when it "
        "rose by".format(INFORMATIVE_ROUTE_M),
        "as much. Graded on informative legs; 50% is chance. false-decisive is the share "
        "of",
        "uninformative legs that got a verdict anyway, which is the measured false-positive "
        "rate.",
        "LOUDER and QUIETER are right/fired, pooled; worst run is the lowest accuracy in "
        "any one run.",
    ]
    return lines + _grid_table(gate["rows"], indent="")


def _grid_table(rows: Sequence[Mapping[str, Any]], *, indent: str) -> List[str]:
    lines = [indent + _GRID_ROW.format(
        "T_LEG", "decisive", "LOUDER", "right", "worst run", "QUIETER", "right",
        "worst run", "false-decisive")]
    for row in rows:
        pooled, branches = row["pooled"], row["branches"]
        lines.append(indent + _GRID_ROW.format(
            "{:.1f}".format(row["t_leg"]), _pct(row["decisive_rate"]),
            *_right_fired(pooled, LEG_LOUDER),
            _pct(branches[LEG_LOUDER]["worst_run_accuracy"]),
            *_right_fired(pooled, LEG_QUIETER),
            _pct(branches[LEG_QUIETER]["worst_run_accuracy"]),
            _pct(row["false_decisive_rate"])))
    return lines


_GRID_ROW = "  {:>5}  {:>8}  {:>9} {:>6} {:>9}  {:>9} {:>6} {:>9}  {:>14}"
_BUCKET_ROW = "    {:<7} {:>11} {:>8}  {:>9} {:>6}  {:>9} {:>6}"


def _gate_lines(gate: Mapping[str, Any]) -> List[str]:
    lines = [""]
    head = "THE GATE (ADR-0029, pre-registered): "
    if gate["verdict"] == BUILD:
        row = next(r for r in gate["rows"] if r["t_leg"] == gate["t_leg"])
        lines.append(head + "BUILD at T_LEG {:.1f}.".format(gate["t_leg"]))
        for name in _BRANCHES:
            branch = row["branches"][name]
            lines.append("  {} right {} pooled, {} in the worst run, decisive on {} of "
                         "informative legs".format(
                             name.upper(), _pct(branch["accuracy"]).strip(),
                             _pct(branch["worst_run_accuracy"]).strip(),
                             _pct(branch["rate"]).strip()))
        lines.append("  Together decisive on {} of informative legs. Implement READ_LEGS "
                     "at this value.".format(_pct(row["decisive_rate"]).strip()))
    elif gate["verdict"] == ONE_BRANCH:
        row = next(r for r in gate["rows"] if r["t_leg"] == gate["t_leg"])
        branch = row["branches"][gate["branch"]]
        other = LEG_QUIETER if gate["branch"] == LEG_LOUDER else LEG_LOUDER
        lines.append(head + "ONE BRANCH: {} at T_LEG {:.1f}.".format(
            gate["branch"].upper(), gate["t_leg"]))
        lines.append("  {} right {} pooled, {} in the worst run, decisive on {} of "
                     "informative legs.".format(
                         gate["branch"].upper(), _pct(branch["accuracy"]).strip(),
                         _pct(branch["worst_run_accuracy"]).strip(),
                         _pct(branch["rate"]).strip()))
        lines.append("  Ship that branch and leave {} INCONCLUSIVE.".format(other.upper()))
    else:
        lines.append(head + "STOP.")
        lines.append(
            "  No T_LEG in the grid gives a branch >= {} right pooled and >= {} in every "
            "run at".format(_pct(GATE_ACCURACY).strip(), _pct(GATE_RUN_ACCURACY).strip()))
        lines.append(
            "  >= {} of informative legs. The lever closes with no box time spent; the "
            "grid says how far.".format(_pct(GATE_DECISIVE_RATE).strip()))
    if not gate["preregistered"]:
        lines.append(
            "  NOT THE PRE-REGISTERED GATE: ADR-0029 names {} independent renders and "
            "this read has {}.".format(PREREGISTERED_RUNS, gate["n_runs"]))
    return lines


def _breakdown_lines(legs: Sequence[Leg], t_leg: float, why: str) -> List[str]:
    lines = [
        "",
        "AT T_LEG {:.1f} ({}).".format(t_leg, why),
    ]
    for title, edges, key in (
        ("by route distance at the leg's start, m", BAND_EDGES_M, "route_start_m"),
        ("by leg length (net displacement), m", LENGTH_EDGES_M, "length_m"),
    ):
        lines.append("  " + title)
        lines.append(_BUCKET_ROW.format(
            "bucket", "informative", "decisive", "LOUDER", "right", "QUIETER", "right"))
        for index in range(len(edges)):
            label = _bucket(edges, edges[index])
            picked = [l for l in legs if _bucket(edges, getattr(l, key)) == label]
            row = score(picked, t_leg)
            lines.append(_BUCKET_ROW.format(
                label, row["n_informative"],
                _pct(_share(row["louder_fired"] + row["quieter_fired"],
                            row["n_informative"])),
                *_right_fired(row, LEG_LOUDER), *_right_fired(row, LEG_QUIETER)))
    return lines


def _field_lines(field: Mapping[str, Mapping[str, Any]]) -> List[str]:
    lines = [
        "",
        "THE FIELD, per scene: detour_report's plateau windows, slope of the cue on the "
        "route,",
        "|slope| x span / residual SD. ADR-0029's 1.94 was one scene of oracle "
        "trajectories.",
        "  {:<14} {:>8} {:>14} {:>14}".format(
            "scene", "windows", "median sig/sc", "louder nearer"),
    ]
    sig: List[float] = []
    slope: List[float] = []
    for name, entry in field.items():
        sig.extend(entry["sig"])
        slope.extend(entry["slope"])
        median = entry["median_sig_sc"]
        lines.append("  {:<14} {:>8} {:>14} {:>14}".format(
            name, entry["n_windows"], "n/a" if median is None else "{:.2f}".format(median),
            _pct(entry["louder_nearer"])))
    lines.append("  {:<14} {:>8} {:>14} {:>14}".format(
        "all windows", len(sig),
        "n/a" if not sig else "{:.2f}".format(statistics.median(sig)),
        _pct(_share(sum(1 for s in slope if s < 0), len(slope)))))
    return lines


def _median_t(legs: Sequence[Leg]) -> Optional[float]:
    values = [float(leg.t) for leg in legs if leg.t is not None]
    return statistics.median(values) if values else None


def by_sounding(runs: Sequence[RunReplay]) -> Dict[str, Dict[str, Any]]:
    """The grid again, once per sounding state. Pure. **Not a gate.**

    ``SOUNDING``, ``SPANS_OFFSET``, ``SILENT`` and ``"unknown"`` (a record that cannot
    say), in that order. Each state carries its completed and informative counts, the
    median ``t`` on legs that approached and on legs that receded, and the rows
    ``evaluate_gate`` computes over that state's legs alone.

    **The median ``t`` is the number that answers the question the split was built for**,
    and it needs no ``T_LEG``. A leg that reads direction has a positive median on
    approaching legs and a negative one on receding legs. A trend that pulls every leg
    quieter pushes both negative, and it shows in the state where it lives.

    The split was chosen after the gate read STOP, so a state that passes here has not
    passed the gate. It would need its own pre-registration before it could.
    """
    labels = [run.label for run in runs]
    states: Dict[str, Dict[str, Any]] = {}
    for state in SOUNDING_STATES + ("unknown",):
        wanted = None if state == "unknown" else state
        picked = {
            run.label: [
                leg for leg in run.legs
                if leg.outcome == COMPLETED and leg.sounding == wanted]
            for run in runs}
        pooled = [leg for label in labels for leg in picked[label]]
        graded = [leg for leg in pooled if leg.delta_route_m is not None]
        approached = [l for l in graded if l.delta_route_m <= -INFORMATIVE_ROUTE_M]
        receded = [l for l in graded if l.delta_route_m >= INFORMATIVE_ROUTE_M]
        states[state] = {
            "n_completed": len(pooled),
            "n_informative": len(approached) + len(receded),
            "n_approached": len(approached),
            "n_receded": len(receded),
            "median_t_approached": _median_t(approached),
            "median_t_receded": _median_t(receded),
            "rows": evaluate_gate(picked)["rows"],
        }
    return states


_STATE_NAMES = {SOUNDING: "sounding", SPANS_OFFSET: "spans offset", SILENT: "silent",
                "unknown": "unknown"}
_STATE_ROW = "  {:<13} {:>9} {:>11} {:>10} {:>8}  {:>20} {:>8}"


def _t_cell(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:+.2f}".format(value)


def _sounding_lines(states: Mapping[str, Mapping[str, Any]]) -> List[str]:
    lines = [
        "",
        "BY SOUNDING STATE. NOT A GATE: this split was chosen after the gate read STOP.",
        "sounding: every reading before the offset step. spans offset: the source or its "
        "cue tail",
        "stopped during the leg. silent: every reading at or after offset_step + "
        "cue_tail_steps - 1,",
        "where the cue is exactly the bed. Median t needs no T_LEG. If the leg reads "
        "direction it is",
        "positive on approaching legs and negative on receding ones. If something pulls "
        "every leg",
        "quieter, both are negative.",
        _STATE_ROW.format("state", "completed", "informative", "approached", "receded",
                          "median t: approached", "receded"),
    ]
    shown = [s for s in SOUNDING_STATES + ("unknown",)
             if s != "unknown" or states[s]["n_completed"]]
    for state in shown:
        entry = states[state]
        lines.append(_STATE_ROW.format(
            _STATE_NAMES[state], entry["n_completed"], entry["n_informative"],
            entry["n_approached"], entry["n_receded"],
            _t_cell(entry["median_t_approached"]), _t_cell(entry["median_t_receded"])))
    for state in shown:
        if not states[state]["n_informative"]:
            continue
        lines.append("")
        lines.append("  the grid, {} legs only".format(_STATE_NAMES[state]))
        lines += _grid_table(states[state]["rows"], indent="  ")
    return lines


# The two readers `loop_removed` compares, as (key, the Leg field holding its t).
_LOOP_READERS = (("fit", "t"), ("same_phase", "t_same_phase"))


def _reader(legs: Sequence[Leg], attr: str, delta: str = "delta_route_m") -> Dict[str, Any]:
    """One reader's medians and sign shares over legs informative on ``delta``. Pure.

    "Approached" means the grading axis fell over the leg, and "receded" that it rose.
    """
    approached = [float(getattr(l, attr)) for l in legs if getattr(l, delta) < 0]
    receded = [float(getattr(l, attr)) for l in legs if getattr(l, delta) > 0]
    return {
        "median_t_approached": statistics.median(approached) if approached else None,
        "median_t_receded": statistics.median(receded) if receded else None,
        "approached_read_up": _share(sum(1 for t in approached if t > 0), len(approached)),
        "receded_read_down": _share(sum(1 for t in receded if t < 0), len(receded)),
    }


def loop_removed(runs: Sequence[RunReplay]) -> Dict[str, Any]:
    """The 9-reading fit and the same-phase reader, over the same legs. Pure. **Not a gate.**

    Informative SOUNDING legs only: where the source stopped, the loop stopped with it,
    and a silent leg has no cue. ``periods`` counts those legs by the loop period their
    record carries, ``"unrecorded"`` included, so a run that never wrote the period
    shows as such and never as a loop of some length. Both readers are then read over
    the legs where both are defined, so a difference between them is the reader's and
    not the population's.

    **The question is whether the pull goes and the direction stays.** A reader with no
    pull reads up on about as many approaching legs as it reads down on receding ones.
    A pull quieter shows as more receding legs read down than approaching legs read up,
    and as both medians negative.
    """
    informative = [
        leg for run in runs for leg in run.legs
        if leg.outcome == COMPLETED and leg.sounding == SOUNDING
        and leg.delta_route_m is not None
        and abs(leg.delta_route_m) >= INFORMATIVE_ROUTE_M]
    periods: Dict[str, int] = {}
    for leg in informative:
        key = "unrecorded" if leg.phase_folds is None else str(leg.phase_folds)
        periods[key] = periods.get(key, 0) + 1
    paired = [l for l in informative if l.t is not None and l.t_same_phase is not None]
    return {
        "n_informative": len(informative),
        "periods": dict(sorted(periods.items())),
        "n_paired": len(paired),
        "n_approached": sum(1 for l in paired if l.delta_route_m < 0),
        "n_receded": sum(1 for l in paired if l.delta_route_m > 0),
        "readers": {key: _reader(paired, attr) for key, attr in _LOOP_READERS},
    }


_READER_NAMES = {"fit": "9-reading fit", "same_phase": "same phase"}
_READER_ROW = "  {:<14} {:>20} {:>8}  {:>18}  {:>17}"


def _loop_lines(entry: Mapping[str, Any]) -> List[str]:
    periods = ", ".join(
        "{} on {}".format(
            "unrecorded" if key == "unrecorded" else "{} steps".format(key), count)
        for key, count in entry["periods"].items()) or "none"
    lines = [
        "",
        "THE LOOP, REMOVED. NOT A GATE: added after the sounding split, sounding legs "
        "only.",
        "The source loops every sounding_phase_folds steps, and 9 readings span no whole "
        "number",
        "of loops. The same-phase reader fits each reading against the one a whole loop "
        "later,",
        "which cancels any level that repeats with the loop. If the loop is the pull, it "
        "goes from",
        "that reader and the direction stays.",
        "  informative sounding legs: {}; loop period: {}".format(
            entry["n_informative"], periods),
        "  both readers defined on {}: {} approached, {} receded".format(
            entry["n_paired"], entry["n_approached"], entry["n_receded"]),
        _READER_ROW.format("reader", "median t: approached", "receded",
                           "approached read up", "receded read down"),
    ]
    return lines + _reader_rows(entry["readers"])


def _reader_rows(readers: Mapping[str, Mapping[str, Any]]) -> List[str]:
    lines = []
    for key, _ in _LOOP_READERS:
        reader = readers[key]
        lines.append(_READER_ROW.format(
            _READER_NAMES[key], _t_cell(reader["median_t_approached"]),
            _t_cell(reader["median_t_receded"]), _pct(reader["approached_read_up"]),
            _pct(reader["receded_read_down"])))
    return lines


def by_line(runs: Sequence[RunReplay]) -> Dict[str, Any]:
    """Both readers over the sounding legs again, graded on the straight line. Pure.

    **Not a gate.** The loop check left about three quarters of informative sounding
    legs quieter along the leg, whichever way they walked by route. The route was chosen
    over the straight line because past a few metres in a house the two come apart and
    the sound was taken to follow the walk (``StepRecord.geodesic_to_source``). Nothing
    measured that choice. If the level follows the straight line instead, legs that
    shortened the route while lengthening the line read quieter, and the pull is the
    grader's axis rather than the field.

    A leg is informative on the line when the horizontal distance to ``source_xyz``
    changed by at least ``INFORMATIVE_ROUTE_M``, the route's own bar. The route is not
    needed, so legs with no route at one end are read here too. ``n_both`` and
    ``n_agree`` count the legs informative on both axes and those where the two agree
    on the direction, which says how far the two gradings can differ at all.
    """
    informative = [
        leg for run in runs for leg in run.legs
        if leg.outcome == COMPLETED and leg.sounding == SOUNDING
        and leg.delta_line_m is not None
        and abs(leg.delta_line_m) >= INFORMATIVE_ROUTE_M]
    paired = [l for l in informative if l.t is not None and l.t_same_phase is not None]
    both = [
        l for l in informative
        if l.delta_route_m is not None and abs(l.delta_route_m) >= INFORMATIVE_ROUTE_M]
    return {
        "n_informative": len(informative),
        "n_paired": len(paired),
        "n_approached": sum(1 for l in paired if l.delta_line_m < 0),
        "n_receded": sum(1 for l in paired if l.delta_line_m > 0),
        "n_both": len(both),
        "n_agree": sum(1 for l in both if (l.delta_line_m < 0) == (l.delta_route_m < 0)),
        "readers": {
            key: _reader(paired, attr, "delta_line_m") for key, attr in _LOOP_READERS},
    }


def _line_lines(entry: Mapping[str, Any]) -> List[str]:
    return [
        "",
        "BY STRAIGHT LINE. NOT A GATE: added after the loop check, sounding legs only.",
        "The same two readers, graded on the horizontal straight-line distance to the "
        "source in",
        "place of the route. If the level follows the line and not the route, the pull "
        "goes here:",
        "legs that closed the line read up about as often as legs that opened it read down.",
        "  informative on the line (|change| >= {:.2f} m): {}".format(
            INFORMATIVE_ROUTE_M, entry["n_informative"]),
        "  both readers defined on {}: {} closed, {} opened".format(
            entry["n_paired"], entry["n_approached"], entry["n_receded"]),
        "  route and line agree on {} of the {} legs informative on both ({})".format(
            entry["n_agree"], entry["n_both"],
            _pct(_share(entry["n_agree"], entry["n_both"])).strip()),
        _READER_ROW.format("reader", "median t: closed", "opened", "closed read up",
                           "opened read down"),
    ] + _reader_rows(entry["readers"])


def format_report(
    runs: Sequence[RunReplay],
    gate: Mapping[str, Any],
    *,
    display_t_leg: float,
    display_why: str,
    field: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> str:
    """The replay as text. Pure, so the gate's wording is Mac-testable."""
    legs = [leg for run in runs for leg in run.legs]
    lines = ["leg replay: ADR-0029's gate over {} run(s): {}".format(
        len(runs), ", ".join(run.label for run in runs)), ""]
    lines += _accounting_lines(runs)
    lines += _population_lines(legs)
    lines += _grid_lines(gate)
    lines += _gate_lines(gate)
    lines += _breakdown_lines(legs, display_t_leg, display_why)
    lines += _sounding_lines(by_sounding(runs))
    lines += _loop_lines(loop_removed(runs))
    lines += _line_lines(by_line(runs))
    if field is not None:
        lines += _field_lines(field)
    lines += [
        "",
        "WHAT THIS CANNOT SAY. These legs were walked under blind alternation, and a "
        "reader",
        "that acts on them changes which legs get walked. This prices the verdict, not the "
        "sweep.",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dirs", nargs="+",
        help="one realizable arm's directory per run, <tag>/<arm>. ADR-0029's gate names "
             "abl-2/full, oracle-1/full and oracle-2/full")
    parser.add_argument(
        "--t-leg", type=float, default=None,
        help="print the breakdowns at this T_LEG. Never changes the gate (default: the "
             "gate's choice, else {:.1f})".format(DISPLAY_T_LEG))
    parser.add_argument(
        "--no-field", action="store_true",
        help="skip the per-scene sig/sc section, which reads every audit a second time")
    parser.add_argument("--json", action="store_true", help="emit the replay as JSON")
    args = parser.parse_args(argv)

    missing = [d for d in args.run_dirs if not pathlib.Path(d).is_dir()]
    if missing:
        print("no such run directory: {}".format(", ".join(missing)))
        return 2
    resolved = [str(pathlib.Path(d).resolve()) for d in args.run_dirs]
    if len(set(resolved)) != len(resolved):
        print("the same run directory was given twice. Its legs would count twice and "
              "its render would pass the per-run test twice")
        return 2

    runs = [load_run(d) for d in args.run_dirs]
    refused = [run for run in runs if run.refusals]
    if refused:
        for run in refused:
            print("REFUSED {}:".format(run.path))
            for reason in run.refusals:
                print("  " + reason)
        return 2
    if not any(leg.outcome == COMPLETED for run in runs for leg in run.legs):
        print("no completed cast leg in {} run(s): nothing to grade".format(len(runs)))
        return 2

    gate = evaluate_gate({run.label: run.legs for run in runs})
    if args.t_leg is not None:
        display, why = float(args.t_leg), "asked for with --t-leg"
    elif gate["t_leg"] is not None:
        display, why = float(gate["t_leg"]), "the gate's choice"
    else:
        display, why = DISPLAY_T_LEG, "the display default; the gate chose none"
    field = None if args.no_field else field_by_scene(args.run_dirs)

    if args.json:
        print(json.dumps({
            "runs": [
                {key: value for key, value in asdict(run).items() if key != "legs"}
                for run in runs],
            "gate": gate,
            "by_sounding": by_sounding(runs),
            "loop_removed": loop_removed(runs),
            "by_line": by_line(runs),
            "field": field,
            "legs": [asdict(leg) for run in runs for leg in run.legs],
        }, indent=2))
    else:
        print(format_report(runs, gate, display_t_leg=display, display_why=why,
                            field=field))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
