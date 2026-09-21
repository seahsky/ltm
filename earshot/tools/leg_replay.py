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
from earshot.report.audit import EpisodeAudit
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
            leg = _measured(leg, rows[start + 1 : start + LEG_PERIOD + 1])
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


def _measured(leg: Leg, readings: Sequence[Any]) -> Leg:
    """A completed leg's verdict inputs, off the readings the controller would hold."""
    positions = [r.position for r in readings]
    if any(p is None for p in positions):
        raise ValueError(
            "episode {} of {}/{} has a detour step with no position. Every record since "
            "yield-1 carries one, so this is a writer fault, and a leg without positions "
            "has no axis to fit along".format(leg.episode, leg.run, leg.scene))
    displacements, length = _along_leg(positions)
    first = readings[0].geodesic_to_source
    last = readings[-1].geodesic_to_source
    return Leg(
        run=leg.run, scene=leg.scene, episode=leg.episode, start_step=leg.start_step,
        outcome=leg.outcome,
        t=leg_t(displacements, [r.measured_rms for r in readings]),
        length_m=length,
        route_start_m=None if first is None else float(first),
        delta_route_m=None if first is None or last is None else float(last) - float(first),
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
        _GRID_ROW.format(
            "T_LEG", "decisive", "LOUDER", "right", "worst run", "QUIETER", "right",
            "worst run", "false-decisive"),
    ]
    for row in gate["rows"]:
        pooled, branches = row["pooled"], row["branches"]
        lines.append(_GRID_ROW.format(
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
            "field": field,
            "legs": [asdict(leg) for run in runs for leg in run.legs],
        }, indent=2))
    else:
        print(format_report(runs, gate, display_t_leg=display, display_why=why,
                            field=field))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
