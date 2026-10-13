"""Is the detour's loudest place nearer the source than where silence finds it? ADR-0031's gate.

    python -m earshot.tools.silent_replay runs/no-tc/no-tc runs/regate-a/no-tc \\
        runs/regate-b/no-tc runs/legs-1/full runs/legs-1/full-b

**ADR-0031 proposes that, once the source stops, the detour go back to where it was
loudest** and run the unchanged cast from there. This prices that change on runs already
on disk, before any arm is built or any night is booked. Read-only, no GPU, minutes.

Every argument is one realizable arm's directory, ``<tag>/<arm>/<scene>/``, and the gate
names five: ``full``'s rule at TC 0, rendered five times over the same 282 episodes.

What it rebuilds
----------------

The detour is the steps that recorded ``realizable_action``, as ``leg_replay`` reads it,
and ``leg_replay.episode_legs`` checks every one of them against the rule that ran. An
episode with one step that disagrees is never read.

Over the detour's own readings it runs the two functions the arm would run, imported from
``agent/controller.py`` and never re-spelled: ``back_at_bed`` decides the source has
stopped, and ``loudest_place`` keeps the loudest whole loop before that step. The bed is
the agent's own ``pre_onset_rms`` and the tolerance is the run's recorded
``pre_onset_rms_tol``, so nothing the agent could not know enters either function.

What it measures
----------------

All distances are ``geodesic_to_source``, which is analyst-only. ``d_off`` is the route
where the detector fired, ``d_loudest`` the route at the loudest place, ``d_min`` the
nearest the detour came up to and including the step the detector fired, and ``d_end``
the route at the detour's last step.

**The price.** One logistic curve of "reached the source after the detector fired" on
``d_off``, fitted on every run's eligible episodes that are not false fires. Each
episode that would return prices ``p(d_loudest) - p(d_off)``, and one that would not
prices zero. A run's price is the sum in points of the run's episodes, and the pooled
price is the mean of the run prices. ADR-0031 lists why the price is optimistic.

The branches are ADR-0031's, checked in order: NOT_RUN, STOP, BUILD.
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
    ACT_FORWARD,
    SILENT_RETURN_M,
    LoudestPlace,
    back_at_bed,
    loudest_place,
)
from earshot.report.artifacts import ENV_REPORT_NAME, episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.tools.leg_replay import FULL_ARM, detour_steps, episode_legs
from earshot.types import Xyz

__all__ = [
    "NOT_RUN",
    "STOP",
    "BUILD",
    "ELIGIBLE",
    "NO_DETOUR",
    "UNROUTED_START",
    "UNVERIFIED",
    "NO_BED_LEVEL",
    "NO_OFFSET",
    "NEVER_FIRED",
    "NO_LOUDEST",
    "REACHED_FIRST",
    "UNROUTED_POSE",
    "STATUSES",
    "PREREGISTERED_RUNS",
    "INFORMATIVE_M",
    "BAND_EDGES_M",
    "GATE_PRICE_POINTS",
    "GATE_ACCURACY",
    "GATE_RUN_ACCURACY",
    "MIN_ELIGIBLE",
    "MIN_INFORMATIVE",
    "MIN_SILENT_REACHES",
    "MAX_FALSE_FIRE",
    "Silence",
    "SilentRun",
    "episode_silence",
    "scene_audio",
    "load_run",
    "fit_reach_curve",
    "reach_probability",
    "evaluate",
    "format_report",
    "main",
]

NOT_RUN = "NOT_RUN"
STOP = "STOP"
BUILD = "BUILD"

# Why an episode is or is not read, in the order the conditions are checked. Every
# episode gets exactly one, and the report counts each, so nothing is dropped unseen.
ELIGIBLE = "eligible"
NO_DETOUR = "no_detour"            # never entered INVESTIGATE
UNROUTED_START = "unrouted_start"  # the detour's first step has no route to the source
UNVERIFIED = "unverified"          # a detour step's rebuilt action disagrees with the record
NO_BED_LEVEL = "no_bed_level"      # no `pre_onset_rms`: the agent's bed level is unknown
NO_OFFSET = "no_offset"            # no window or no offset step on the record
NEVER_FIRED = "never_fired"        # the detector did not fire before the detour ended
NO_LOUDEST = "no_loudest"          # it fired before one whole loop was kept
REACHED_FIRST = "reached_first"    # the source was reached at or before that step
UNROUTED_POSE = "unrouted_pose"    # no route at the firing step or at the loudest place
STATUSES = (ELIGIBLE, NO_DETOUR, UNROUTED_START, UNVERIFIED, NO_BED_LEVEL, NO_OFFSET,
            NEVER_FIRED, NO_LOUDEST, REACHED_FIRST, UNROUTED_POSE)

# provenance: ADR-0031 -- the gate's numbers, judgment and not derivation, set before the
# replay read a run (PR #173, accepted as written 2026-10-12).
PREREGISTERED_RUNS = 5
INFORMATIVE_M = 1.0              # |d_loudest - d_off| for the accuracy test
BAND_EDGES_M = (2.0, 3.0, 5.0, 8.0)
GATE_PRICE_POINTS = 7.30         # one 282-episode night's MDE at the 19.1% TC-0 flip
GATE_ACCURACY = 0.75
GATE_RUN_ACCURACY = 0.70
MIN_ELIGIBLE = 100               # per run
MIN_INFORMATIVE = 30             # per run, or its accuracy is not measured
MIN_SILENT_REACHES = 30          # pooled, or the curve is not fitted
MAX_FALSE_FIRE = 0.10            # of the detector's fires, per run

# provenance: fake -- the fit's own stopping rule, fixed with the tool. A logistic fit
# that has not settled in 100 Newton steps has separated data, and its curve is not one.
FIT_MAX_ITERATIONS = 100
FIT_TOLERANCE = 1e-9

# Where the curve is printed, one distance inside each band.
_PRINT_AT_M = (1.5, 2.5, 4.0, 6.5, 10.0)


@dataclass(frozen=True)
class Silence:
    """One episode's silent half, or the named reason it is not read.

    The detector's fields are set on every episode it fired on, eligible or not, because
    the false-fire share is over all of those. The distances are set on eligible ones.
    """

    run: str
    scene: str
    episode: int
    status: str
    fire_step: Optional[int] = None
    false_fire: Optional[bool] = None
    delay_steps: Optional[int] = None    # firing step minus the record's offset step
    loudest_step: Optional[int] = None
    d_off: Optional[float] = None
    d_loudest: Optional[float] = None
    d_min: Optional[float] = None
    d_end: Optional[float] = None
    reached_after: Optional[bool] = None
    return_m: Optional[float] = None     # agent's own horizontal distance to the place
    would_return: Optional[bool] = None
    forwards_back: Optional[int] = None  # forwards taken from the place to the firing step


@dataclass(frozen=True)
class SilentRun:
    """One run directory's episodes, or the named reasons it cannot be read."""

    label: str
    path: str
    n_episodes: int
    episodes: Tuple[Silence, ...]
    refusals: Tuple[str, ...]


def _share(numerator: int, denominator: int) -> Optional[float]:
    """A ratio, or None for an empty denominator. 0 of 0 is not 0%."""
    return (float(numerator) / float(denominator)) if denominator else None


def _horizontal(a: Xyz, b: Xyz) -> float:
    return math.hypot(a.x - b.x, a.z - b.z)


def episode_silence(
    audit: EpisodeAudit, *, run: str, scene: str, tolerance: float
) -> Silence:
    """The silent half of one episode's detour, as ADR-0031's arm would have read it. Pure.

    ``tolerance`` is the run's recorded ``pre_onset_rms_tol``. The loop length is the
    calibration's ``cue_phase_folds``, which the caller has checked is at least 1.
    """
    base = dict(run=run, scene=scene, episode=int(audit.episode_index))
    detour = detour_steps(audit)
    if audit.funnel_stage < FunnelStage.INVESTIGATE_ENTERED or not detour:
        return Silence(status=NO_DETOUR, **base)
    rows = [audit.steps[i] for i in detour]
    if rows[0].geodesic_to_source is None:
        return Silence(status=UNROUTED_START, **base)
    replay = episode_legs(audit, run=run, scene=scene)
    if replay.n_steps_agree != replay.n_steps_checked:
        return Silence(status=UNVERIFIED, **base)
    bed = None if audit.onset is None else audit.onset.pre_onset_rms
    if bed is None:
        return Silence(status=NO_BED_LEVEL, **base)
    window = audit.sounding_window
    if window is None or window.offset_step is None:
        return Silence(status=NO_OFFSET, **base)
    if any(r.position is None for r in rows):
        raise ValueError(
            "episode {} of {}/{} has a detour step with no position. Every record since "
            "yield-1 carries one, so this is a writer fault".format(
                audit.episode_index, run, scene))
    if audit.calibration is None or int(audit.calibration.cue_phase_folds) < 1:
        raise ValueError(
            "episode {} of {}/{} records no loop length; `load_run` refuses such a run "
            "before reading an episode".format(audit.episode_index, run, scene))
    folds = int(audit.calibration.cue_phase_folds)

    levels = [float(r.measured_rms) for r in rows]
    positions = [r.position for r in rows]
    kept: Optional[LoudestPlace] = None
    fire: Optional[int] = None
    for k in range(len(rows)):
        if back_at_bed(levels[: k + 1], bed_level=float(bed), tolerance=tolerance,
                       folds=folds):
            fire = k
            break
        kept = loudest_place(kept, levels[: k + 1], positions[: k + 1],
                             last_step=int(rows[k].step), folds=folds)
    if fire is None:
        return Silence(status=NEVER_FIRED, **base)

    fired = rows[fire]
    detector = dict(
        fire_step=int(fired.step),
        false_fire=bool(fired.source_playing),
        delay_steps=int(fired.step) - int(window.offset_step),
    )
    if kept is None:
        return Silence(status=NO_LOUDEST, **base, **detector)
    reached = audit.source_reached_step
    if reached is not None and int(reached) <= int(fired.step):
        return Silence(status=REACHED_FIRST, **base, **detector)
    loud = next(j for j, r in enumerate(rows) if int(r.step) == kept.step)
    d_off, d_loudest = fired.geodesic_to_source, rows[loud].geodesic_to_source
    if d_off is None or d_loudest is None:
        return Silence(status=UNROUTED_POSE, loudest_step=kept.step, **base, **detector)

    routes = [float(r.geodesic_to_source) for r in rows[: fire + 1]
              if r.geodesic_to_source is not None]
    end = rows[-1].geodesic_to_source
    return_m = _horizontal(fired.position, kept.position)
    return Silence(
        status=ELIGIBLE,
        loudest_step=kept.step,
        d_off=float(d_off),
        d_loudest=float(d_loudest),
        d_min=min(routes),
        d_end=None if end is None else float(end),
        reached_after=reached is not None and int(reached) > int(fired.step),
        return_m=return_m,
        would_return=return_m > SILENT_RETURN_M,
        forwards_back=sum(1 for r in rows[loud:fire] if r.action == ACT_FORWARD),
        **base, **detector)


def _label(path: pathlib.Path) -> str:
    """``legs-1/full`` for ``runs/legs-1/full``: the tag names the render, the arm the rule."""
    return "/".join(path.resolve().parts[-2:])


def scene_audio(scene_dir: pathlib.Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """The scene's recorded ``AudioConfig``, or why it cannot be read as a TC-0 render.

    ``env_report.json``'s ``run_config.audio``, which the runner writes per scene
    directory before any episode. A render whose ``temporal_coherence`` is not exactly
    ``False`` is refused, ``None`` included: that is a record from before ADR-0030 made
    the value explicit, and it rendered with the key on.
    """
    report = scene_dir / ENV_REPORT_NAME
    if not report.is_file():
        return None, ("{} is missing, so the scene's render cannot be confirmed as TC 0"
                      .format(report))
    audio = json.loads(report.read_text(encoding="utf-8"))["run_config"]["audio"]
    if audio.get("temporal_coherence") is not False:
        return None, ("{} rendered with temporal_coherence {!r}. The reader takes `full`'s "
                      "rule at TC 0 alone".format(scene_dir.name,
                                                  audio.get("temporal_coherence")))
    return dict(audio), ""


def load_run(arm_dir: str) -> SilentRun:
    """Every episode under one arm's directory, or the reasons the run is refused."""
    from earshot.task.smoke import episode_indices

    root = pathlib.Path(arm_dir)
    label = _label(root)
    _, episodes_dir = run_paths(root)
    if episodes_dir.is_dir():
        return SilentRun(label, str(root), 0, (), (
            "{} is a scene directory: it holds episodes itself. Pass the arm directory "
            "above it, <tag>/<arm>.".format(root),))

    episodes: List[Silence] = []
    refusals: List[str] = []
    mismatched: Dict[Tuple[str, Optional[str]], int] = {}
    n_episodes = unfolded = 0
    routed = positioned = realized = False
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        indices = episode_indices(str(scene_dir))
        if not indices:
            continue  # a zero-yield scene: nothing to read and nothing to refuse
        audio, why = scene_audio(scene_dir)
        if audio is None:
            refusals.append(why)
            continue
        tolerance = audio.get("pre_onset_rms_tol")
        if tolerance is None:
            refusals.append(
                "{} records no pre_onset_rms_tol, the bound the detector reads the bed "
                "with".format(scene_dir.name))
            continue
        for index in indices:
            _, audit_path = episode_paths(scene_dir, index)
            audit = read_audit(audit_path)
            n_episodes += 1
            for name, wanted in FULL_ARM:
                got = getattr(audit, name)
                if got != wanted:
                    mismatched[(name, got)] = mismatched.get((name, got), 0) + 1
            routed = routed or any(r.geodesic_to_source is not None for r in audit.steps)
            positioned = positioned or any(r.position is not None for r in audit.steps)
            realized = realized or any(r.realizable_action is not None for r in audit.steps)
            if audit.calibration is None or int(audit.calibration.cue_phase_folds) < 1:
                unfolded += 1
                continue
            episodes.append(episode_silence(
                audit, run=label, scene=scene_dir.name, tolerance=float(tolerance)))

    if not n_episodes and not refusals:
        refusals.append("no episode records under {}".format(root))
    elif n_episodes:
        for (name, got), count in sorted(mismatched.items(), key=lambda kv: str(kv[0])):
            refusals.append(
                "not `full`'s rule: {} is {} on {} of {} episode(s)".format(
                    name, "unrecorded" if got is None else repr(got), count, n_episodes))
        for flag, field in ((routed, "geodesic_to_source"), (positioned, "position"),
                            (realized, "realizable_action")):
            if not flag:
                refusals.append("no step recorded `{}`".format(field))
        if unfolded:
            refusals.append(
                "{} of {} episode(s) record no loop length (`cue_phase_folds` under 1). "
                "A window of zero readings would fire the detector on the detour's first "
                "step".format(unfolded, n_episodes))
    return SilentRun(label, str(root), n_episodes, tuple(episodes), tuple(refusals))


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_reach_curve(
    points: Sequence[Tuple[float, bool]]
) -> Optional[Tuple[float, float]]:
    """Maximum-likelihood ``(a, b)`` of ``p(d) = 1 / (1 + exp(-(a + b d)))``. Pure.

    Newton's method from the base rate. ``None`` where no curve exists: every outcome the
    same, or a fit that has not settled in ``FIT_MAX_ITERATIONS`` steps, which is what
    separated data does.
    """
    n = len(points)
    k = sum(1 for _, y in points if y)
    if k == 0 or k == n:
        return None
    a, b = math.log(k / (n - k)), 0.0
    for _ in range(FIT_MAX_ITERATIONS):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in points:
            p = _sigmoid(a + b * float(x))
            w = p * (1.0 - p)
            r = (1.0 if y else 0.0) - p
            g0 += r
            g1 += r * float(x)
            h00 += w
            h01 += w * float(x)
            h11 += w * float(x) * float(x)
        det = h00 * h11 - h01 * h01
        if det <= 0.0:
            return None
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a, b = a + da, b + db
        if max(abs(da), abs(db)) < FIT_TOLERANCE:
            return a, b
    return None


def reach_probability(curve: Tuple[float, float], distance_m: float) -> float:
    """The fitted curve at one route distance. Pure."""
    a, b = curve
    return _sigmoid(a + b * float(distance_m))


def _informative(s: Silence) -> bool:
    return abs(float(s.d_loudest) - float(s.d_off)) >= INFORMATIVE_M


def _run_price(run: SilentRun, curve: Tuple[float, float]) -> Optional[float]:
    if not run.n_episodes:
        return None
    total = sum(
        reach_probability(curve, s.d_loudest) - reach_probability(curve, s.d_off)
        for s in run.episodes if s.status == ELIGIBLE and s.would_return)
    return 100.0 * total / run.n_episodes


def evaluate(runs: Sequence[SilentRun]) -> Dict[str, Any]:
    """Every number the branches read, and the branch. Pure."""
    eligible = [s for run in runs for s in run.episodes if s.status == ELIGIBLE]
    fit_points = [(float(s.d_off), bool(s.reached_after))
                  for s in eligible if not s.false_fire]
    silent_reaches = sum(1 for _, y in fit_points if y)
    curve = fit_reach_curve(fit_points) if silent_reaches >= MIN_SILENT_REACHES else None

    per_run: Dict[str, Dict[str, Any]] = {}
    for run in runs:
        mine = [s for s in run.episodes if s.status == ELIGIBLE]
        fired = [s for s in run.episodes if s.fire_step is not None]
        informative = [s for s in mine if _informative(s)]
        nearer = sum(1 for s in informative if s.d_loudest < s.d_off)
        per_run[run.label] = {
            "n_episodes": run.n_episodes,
            "status": {name: sum(1 for s in run.episodes if s.status == name)
                       for name in STATUSES},
            "n_eligible": len(mine),
            "n_fired": len(fired),
            "n_false_fire": sum(1 for s in fired if s.false_fire),
            "false_fire_share": _share(sum(1 for s in fired if s.false_fire), len(fired)),
            "n_informative": len(informative),
            "n_nearer": nearer,
            "accuracy": _share(nearer, len(informative)),
            "price_points": None if curve is None else _run_price(run, curve),
        }

    n_informative = sum(r["n_informative"] for r in per_run.values())
    n_nearer = sum(r["n_nearer"] for r in per_run.values())
    prices = [r["price_points"] for r in per_run.values()]
    pooled_price = (sum(prices) / len(prices)
                    if prices and all(p is not None for p in prices) else None)
    pooled_accuracy = _share(n_nearer, n_informative)

    not_run: List[str] = []
    if len(runs) != PREREGISTERED_RUNS:
        not_run.append("ADR-0031 names {} renders and {} were read".format(
            PREREGISTERED_RUNS, len(runs)))
    for run in runs:
        if run.refusals:
            not_run.append("{} is refused".format(run.label))
    for label, r in per_run.items():
        if r["n_eligible"] < MIN_ELIGIBLE:
            not_run.append("{} has {} eligible episode(s), under {}".format(
                label, r["n_eligible"], MIN_ELIGIBLE))
        share = r["false_fire_share"]
        if share is not None and share > MAX_FALSE_FIRE:
            not_run.append(
                "{}: the detector fired while the source sounded on {} of {} fire(s), "
                "over {:.0%}".format(label, r["n_false_fire"], r["n_fired"], MAX_FALSE_FIRE))
    if silent_reaches < MIN_SILENT_REACHES:
        not_run.append("{} reach(es) after the detector fired, pooled, under {}: the curve "
                       "is not fitted".format(silent_reaches, MIN_SILENT_REACHES))
    elif curve is None:
        not_run.append("the reach curve did not converge in {} steps".format(
            FIT_MAX_ITERATIONS))

    stop: List[str] = []
    if not not_run:
        if curve is not None and curve[1] >= 0.0:
            stop.append("the reach curve does not fall with distance (slope {:+.3f}), so no "
                        "return to a nearer place can pay".format(curve[1]))
        if pooled_price is None or pooled_price < GATE_PRICE_POINTS:
            stop.append("the pooled price is {} points, under {:.2f}".format(
                "unmeasured" if pooled_price is None else "{:.2f}".format(pooled_price),
                GATE_PRICE_POINTS))
        for label, r in per_run.items():
            if r["price_points"] is None or r["price_points"] <= 0.0:
                stop.append("{} prices at {}".format(label, r["price_points"]))
        if pooled_accuracy is None or pooled_accuracy < GATE_ACCURACY:
            stop.append("the loudest place is nearer on {} of {} informative episodes "
                        "pooled, under {:.0%}".format(n_nearer, n_informative,
                                                      GATE_ACCURACY))
        for label, r in per_run.items():
            if r["n_informative"] < MIN_INFORMATIVE:
                stop.append("{} has {} informative episode(s), under {}: its accuracy is "
                            "not measured".format(label, r["n_informative"],
                                                  MIN_INFORMATIVE))
            elif r["accuracy"] < GATE_RUN_ACCURACY:
                stop.append("{} is nearer on {} of {}, under {:.0%}".format(
                    label, r["n_nearer"], r["n_informative"], GATE_RUN_ACCURACY))

    verdict = NOT_RUN if not_run else (STOP if stop else BUILD)
    return {
        "verdict": verdict,
        "reasons": not_run or stop,
        "runs": per_run,
        "n_eligible": len(eligible),
        "n_fit": len(fit_points),
        "silent_reaches": silent_reaches,
        "curve": None if curve is None else {"a": curve[0], "b": curve[1]},
        "pooled_price_points": pooled_price,
        "n_informative": n_informative,
        "n_nearer": n_nearer,
        "pooled_accuracy": pooled_accuracy,
    }


def _band(distance_m: float) -> int:
    return sum(1 for edge in BAND_EDGES_M if float(distance_m) >= edge)


_BAND_NAMES = tuple(
    ["under {:g} m".format(BAND_EDGES_M[0])]
    + ["{:g} to {:g} m".format(lo, hi) for lo, hi in zip(BAND_EDGES_M, BAND_EDGES_M[1:])]
    + ["{:g} m and over".format(BAND_EDGES_M[-1])])


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.1%}".format(value)


def _points(value: Optional[float]) -> str:
    return "unmeasured" if value is None else "{:+.2f} points".format(value)


def _spread(values: Sequence[float]) -> str:
    if not values:
        return "none"
    ordered = sorted(values)
    if len(ordered) < 2:
        return "n={} value {:+.2f}".format(len(ordered), ordered[0])
    q1, q2, q3 = statistics.quantiles(ordered, n=4)
    return "n={} median {:+.2f}  quartiles {:+.2f} / {:+.2f}  min {:+.2f}  max {:+.2f}".format(
        len(ordered), q2, q1, q3, ordered[0], ordered[-1])


def format_report(runs: Sequence[SilentRun], result: Mapping[str, Any]) -> str:
    eligible = [s for run in runs for s in run.episodes if s.status == ELIGIBLE]
    lines = ["ADR-0031's GATE: in silence, does the detour's loudest place know more than",
             "where silence found the agent? Pre-registered in ADR-0031.", ""]

    lines.append("WHAT WAS READ, per run (every episode has exactly one status)")
    for run in runs:
        status = result["runs"][run.label]["status"]
        lines.append("  {}  {} episode(s): {}".format(run.label, run.n_episodes, "  ".join(
            "{} {}".format(name, status[name]) for name in STATUSES if status[name])))
    lines.append("")

    lines.append("1-2. WHERE SILENCE FINDS THE AGENT, and what the silent cast reaches from "
                 "there")
    lines.append("     each cell: eligible episodes with the route at firing in that band, "
                 "and the share reached after the detector fired")
    lines.append("  {:<28}".format("") + "".join("{:>16}".format(n) for n in _BAND_NAMES))
    groups = [(run.label, [s for s in run.episodes if s.status == ELIGIBLE]) for run in runs]
    for label, mine in groups + [("pooled", eligible)]:
        cells = []
        for b in range(len(_BAND_NAMES)):
            band = [s for s in mine if _band(s.d_off) == b]
            hits = sum(1 for s in band if s.reached_after)
            share = _share(hits, len(band))
            cells.append("{:>16}".format("{} {}".format(
                len(band), "" if share is None else "({:.0%})".format(share))))
        lines.append("  {:<28}".format(label) + "".join(cells))
    lines.append("  d_end - d_off (m), pooled, by the same bands:")
    for b, name in enumerate(_BAND_NAMES):
        lines.append("    {:<14} {}".format(name, _spread(
            [s.d_end - s.d_off for s in eligible
             if _band(s.d_off) == b and s.d_end is not None])))
    lines.append("")

    lines.append("THE SPLIT: nearest the detour came (rows) against where silence found it "
                 "(columns)")
    lines.append("  {:<16}".format("") + "".join("{:>14}".format(n) for n in _BAND_NAMES))
    for row, name in enumerate(_BAND_NAMES):
        counts = [sum(1 for s in eligible if _band(s.d_min) == row and _band(s.d_off) == col)
                  for col in range(len(_BAND_NAMES))]
        lines.append("  {:<16}".format(name) + "".join("{:>14}".format(c) for c in counts))
    lines.append("  A detour that never came near has nothing near to return to. One that "
                 "came near and left is the case ADR-0031 is for.")
    lines.append("")

    lines.append("3. WHAT THE LOUDEST PLACE KNOWS: nearer than the firing pose, on "
                 "informative episodes (|d_loudest - d_off| >= {:g} m)".format(INFORMATIVE_M))
    for label, r in result["runs"].items():
        lines.append("  {:<28} {:>4} of {:>4} = {}".format(
            label, r["n_nearer"], r["n_informative"], _pct(r["accuracy"])))
    lines.append("  {:<28} {:>4} of {:>4} = {}".format(
        "pooled", result["n_nearer"], result["n_informative"],
        _pct(result["pooled_accuracy"])))
    lines.append("  d_loudest - d_off (m): {}".format(
        _spread([s.d_loudest - s.d_off for s in eligible])))
    lines.append("")

    lines.append("4. LOUDEST AGAINST NEAREST: d_loudest - d_min (m), zero or more by "
                 "construction")
    for label, mine in groups + [("pooled", eligible)]:
        lines.append("  {:<28} {}".format(
            label, _spread([s.d_loudest - s.d_min for s in mine])))
    lines.append("")

    lines.append("5. THE OFFSET DETECTOR")
    for run in runs:
        r = result["runs"][run.label]
        delays = [float(s.delay_steps) for s in run.episodes if s.delay_steps is not None]
        lines.append("  {:<28} fired on {}, while the source sounded on {} ({})".format(
            run.label, r["n_fired"], r["n_false_fire"], _pct(r["false_fire_share"])))
        lines.append("  {:<28} delay after the offset step, steps: {}".format(
            "", _spread(delays)))
    lines.append("")

    returning = [s for s in eligible if s.would_return]
    lines.append("6. THE RETURN: {} of {} eligible would return (over {:g} m from the "
                 "loudest place)".format(len(returning), len(eligible), SILENT_RETURN_M))
    for label, mine in groups + [("pooled", eligible)]:
        back = [s for s in mine if s.would_return]
        lines.append("  {:<28} {} of {} return".format(label, len(back), len(mine)))
        lines.append("  {:<28} straight-line distance back (m): {}".format(
            "", _spread([s.return_m for s in back])))
        lines.append("  {:<28} forwards taken since the loudest place: {}".format(
            "", _spread([float(s.forwards_back) for s in back])))
    lines.append("  That bounds the return's distance, not its steps: the follower turns "
                 "30 degrees a tick.")
    lines.append("")

    curve = result["curve"]
    lines.append("7. THE PRICE (optimistic; ADR-0031 lists why)")
    if curve is None:
        lines.append("  no curve: {} reach(es) after firing over {} fitted episode(s)".format(
            result["silent_reaches"], result["n_fit"]))
    else:
        fitted = (curve["a"], curve["b"])
        lines.append("  reach curve: p(d) = 1/(1+exp(-({:+.3f} {:+.3f} d))), {} reach(es) over "
                     "{} episode(s)".format(curve["a"], curve["b"], result["silent_reaches"],
                                            result["n_fit"]))
        lines.append("  " + "  ".join("p({:g} m) {:.1%}".format(d, reach_probability(fitted, d))
                                      for d in _PRINT_AT_M))
        for label, r in result["runs"].items():
            lines.append("  {:<28} {}".format(label, _points(r["price_points"])))
        lines.append("  {:<28} {} (the mean of the runs), against {:.2f}".format(
            "pooled", _points(result["pooled_price_points"]), GATE_PRICE_POINTS))
    lines.append("")

    lines.append("BRANCH: {}".format(result["verdict"]))
    for reason in result["reasons"]:
        lines.append("  - " + reason)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "run_dirs", nargs="+",
        help="one arm's directory per render, <tag>/<arm>. ADR-0031 names no-tc/no-tc, "
             "regate-a/no-tc, regate-b/no-tc, legs-1/full and legs-1/full-b")
    parser.add_argument("--json", action="store_true", help="emit the replay as JSON")
    args = parser.parse_args(argv)

    missing = [d for d in args.run_dirs if not pathlib.Path(d).is_dir()]
    if missing:
        print("no such run directory: {}".format(", ".join(missing)))
        return 2
    resolved = [str(pathlib.Path(d).resolve()) for d in args.run_dirs]
    if len(set(resolved)) != len(resolved):
        print("the same run directory was given twice. Its episodes would count twice and "
              "its render would pass the per-run tests twice")
        return 2

    runs = [load_run(d) for d in args.run_dirs]
    refused = [run for run in runs if run.refusals]
    if refused:
        for run in refused:
            print("REFUSED {}:".format(run.path))
            for reason in run.refusals:
                print("  " + reason)
        print("BRANCH: {} (a refused run is not read)".format(NOT_RUN))
        return 2

    result = evaluate(runs)
    if args.json:
        print(json.dumps({
            "gate": result,
            "episodes": [asdict(s) for run in runs for s in run.episodes],
        }, indent=2))
    else:
        print(format_report(runs, result))
    # NOT_RUN is red: a gate that could not be evaluated never reads as a result.
    return 2 if result["verdict"] == NOT_RUN else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
