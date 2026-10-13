"""While the source sounds, could the detour reach it, and does the ear cue point at it?

    python -m earshot.tools.sounding_replay runs/no-tc/no-tc runs/regate-a/no-tc \\
        runs/regate-b/no-tc runs/legs-1/full runs/legs-1/full-b

ADR-0031's replay split the loss by half of the detour: of the detours that had not
arrived when the source stopped, 44.8% never came within 5 m of it. This asks two
questions of the same records before anything is built for the sounding half. Read-only,
no GPU, minutes. NOT A GATE: no branch is pre-registered, and a change it points at
needs its own ADR and gate.

THE CLOCK. The source sounds for a fixed number of steps. A detour whose route to the
source, less the arrival ring, is longer than those steps can walk could not arrive while
it sounds, however well it steered. That bound is optimistic for the agent: it charges no
turn and no wall. The share of detours past it is what no change to steering in the
sounding half can reach.

THE EAR. ``lateral_sign`` is the interaural level sign the agent heard at every step:
+1 the source to its right, -1 left, 0 nothing. The controller reads it in its first scan
and at its first leg's turn, and the cast ignores it after that. ``abl-2`` priced that one
use at 14.9 points (the ``no-cue`` arm). This grades the sign at every sounding detour
step against the side the source was on, by straight line, from the agent's position and
heading. If the sign is right well above chance where detours stall, a controller that
steers by it for the whole sounding window is the candidate. If it is near chance there,
the cue does not carry the side at those distances, and no steering rule can use it.

The audit records no heading, so the replay rebuilds it. A forward that moved the agent
gives its heading off the displacement. Turns between two such forwards are 30 degrees
each (``sim/world.OBJECTNAV_HM3D``). A step is graded only between two forwards whose
headings agree with the turns recorded between them, so a heading the record cannot
confirm is never used. The report counts the pairs that disagree.

Straight line is not route. Through a wall the sound arrives by the door, so the report
splits the grade by how much longer the route is than the line.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from earshot.agent.config import DetectorConfig
from earshot.agent.controller import ACT_FORWARD, ACT_TURN_LEFT, ACT_TURN_RIGHT
from earshot.audio.lateral import LATERAL_AMBIGUOUS, bearing_lateral_sign
from earshot.report.artifacts import episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.tools.leg_replay import FULL_ARM, detour_steps
from earshot.tools.silent_replay import scene_audio
from earshot.types import Pose, Xyz

__all__ = [
    "READ",
    "NO_DETOUR",
    "UNROUTED_START",
    "NO_OFFSET",
    "TURN_RAD",
    "STEP_M",
    "ARRIVAL_RING_M",
    "HEADING_TOLERANCE_RAD",
    "EarStep",
    "Sounding",
    "SoundingRun",
    "headings",
    "episode_sounding",
    "load_run",
    "summarize",
    "format_report",
    "main",
]

READ = "read"
NO_DETOUR = "no_detour"            # never entered INVESTIGATE
UNROUTED_START = "unrouted_start"  # no route to the source at the detour's first step
NO_OFFSET = "no_offset"            # no window or no offset step on the record
STATUSES = (READ, NO_DETOUR, UNROUTED_START, NO_OFFSET)

# provenance: source -- the embodiment, `sim/world.OBJECTNAV_HM3D` (30 degree turns, 0.25 m
# forwards). Not imported: `sim/` imports habitat_sim, which a Mac cannot load. The
# heading check below would fail on every pair if these were wrong.
TURN_RAD = math.radians(30.0)
STEP_M = 0.25
# A forward that moved at least this far is an anchor. Less is a slide along a wall,
# whose direction is the wall's and not the heading's.
ANCHOR_MIN_M = 0.2
# provenance: source -- the arrival ring the oracle detector confirms at, Find-SR's 1.0 m.
ARRIVAL_RING_M = DetectorConfig().oracle_radius_m
# provenance: fake -- how far two anchors' headings may disagree after the recorded turns
# between them are applied. A turn is 30 degrees, so 10 degrees separates "the same
# heading, rounded" from "a turn the record missed".
HEADING_TOLERANCE_RAD = math.radians(10.0)
# The red line for the heading rebuild: more disagreeing anchor pairs than this, and the
# ear's grade rests on headings the record does not support.
MAX_HEADING_DISAGREE = 0.05

LINE_EDGES_M = (2.0, 4.0, 8.0)
START_EDGES_M = (3.0, 5.0, 8.0, 12.0)
OPEN_RATIO = 1.2  # route no more than this many times the line: the path is open


@dataclass(frozen=True)
class EarStep:
    """One sounding detour step whose heading the record confirms, graded."""

    line_m: float
    route_m: Optional[float]
    heard: int  # the recorded `lateral_sign`
    truth: int  # the side of the source by straight line, from the rebuilt pose


@dataclass(frozen=True)
class Sounding:
    """One episode's sounding half: the clock, and the ear's graded steps."""

    run: str
    scene: str
    episode: int
    status: str
    d_start: Optional[float] = None
    sounding_steps: Optional[int] = None   # detour steps taken while the source sounded
    in_time: Optional[bool] = None         # the route, less the ring, fits in those steps
    d_min_sounding: Optional[float] = None
    reached_sounding: Optional[bool] = None
    anchor_pairs: int = 0
    anchor_disagree: int = 0
    ear: Tuple[EarStep, ...] = ()


@dataclass(frozen=True)
class SoundingRun:
    label: str
    path: str
    n_episodes: int
    episodes: Tuple[Sounding, ...]
    refusals: Tuple[str, ...]


def _turn(action: Optional[str]) -> float:
    if action == ACT_TURN_LEFT:
        return TURN_RAD
    if action == ACT_TURN_RIGHT:
        return -TURN_RAD
    return 0.0


def _wrap(angle: float) -> float:
    """An angle in (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def headings(
    positions: Sequence[Xyz], actions: Sequence[Optional[str]]
) -> Tuple[List[Optional[float]], int, int]:
    """Each step's heading where the record confirms it, and the anchor pairs checked. Pure.

    ``positions[i]`` is the pose a step's reading was taken at and ``actions[i]`` what the
    agent did next, so ``positions[i + 1]`` is where that action left it. A forward that
    moved the agent at least ``ANCHOR_MIN_M`` is an anchor: its heading is the
    displacement's, ``atan2(-dx, -dz)`` in habitat's frame (forward is ``-z`` at yaw 0).
    Between two anchors the recorded turns are applied to the first, and if the result
    lands within ``HEADING_TOLERANCE_RAD`` of the second the steps between them get
    headings. Steps before the first anchor, after the last, or inside a pair that
    disagrees get ``None``. Returns the headings, the pairs checked, and the pairs that
    disagreed.
    """
    n = len(positions)
    if len(actions) != n:
        raise ValueError("one action per position: got {} and {}".format(len(actions), n))
    anchors: List[Tuple[int, float]] = []
    for i in range(n - 1):
        if actions[i] != ACT_FORWARD:
            continue
        dx = positions[i + 1].x - positions[i].x
        dz = positions[i + 1].z - positions[i].z
        if math.hypot(dx, dz) >= ANCHOR_MIN_M:
            anchors.append((i, math.atan2(-dx, -dz)))
    out: List[Optional[float]] = [None] * n
    disagree = 0
    for (p, yaw_p), (a, yaw_a) in zip(anchors, anchors[1:]):
        yaw = yaw_p
        rebuilt = [yaw]
        for j in range(p, a):
            yaw += _turn(actions[j])
            rebuilt.append(yaw)
        if abs(_wrap(rebuilt[-1] - yaw_a)) > HEADING_TOLERANCE_RAD:
            disagree += 1
            continue
        for offset, value in enumerate(rebuilt):
            out[p + offset] = _wrap(value)
    return out, max(0, len(anchors) - 1), disagree


def _line_m(position: Xyz, source: Xyz) -> float:
    return math.hypot(position.x - source.x, position.z - source.z)


def episode_sounding(audit: EpisodeAudit, *, run: str, scene: str) -> Sounding:
    """The sounding half of one episode's detour. Pure."""
    base = dict(run=run, scene=scene, episode=int(audit.episode_index))
    detour = detour_steps(audit)
    if audit.funnel_stage < FunnelStage.INVESTIGATE_ENTERED or not detour:
        return Sounding(status=NO_DETOUR, **base)
    rows = [audit.steps[i] for i in detour]
    if rows[0].geodesic_to_source is None:
        return Sounding(status=UNROUTED_START, **base)
    window = audit.sounding_window
    if window is None or window.offset_step is None:
        return Sounding(status=NO_OFFSET, **base)
    if any(r.position is None for r in rows):
        raise ValueError(
            "episode {} of {}/{} has a detour step with no position, which is a writer "
            "fault".format(audit.episode_index, run, scene))

    offset = int(window.offset_step)
    sounding = [r for r in rows if r.source_playing and int(r.step) < offset]
    d_start = float(rows[0].geodesic_to_source)
    routes = [float(r.geodesic_to_source) for r in sounding
              if r.geodesic_to_source is not None]
    reached = audit.source_reached_step
    need = max(0.0, d_start - ARRIVAL_RING_M) / STEP_M

    yaws, pairs, disagree = headings([r.position for r in rows], [r.action for r in rows])
    source = audit.source_xyz
    ear: List[EarStep] = []
    if source is not None:
        for r, yaw in zip(rows, yaws):
            if yaw is None or not r.source_playing or int(r.step) >= offset:
                continue
            if r.lateral_sign is None:
                continue
            truth = bearing_lateral_sign(Pose(position=r.position, yaw_rad=yaw), source)
            if truth == LATERAL_AMBIGUOUS:
                continue
            ear.append(EarStep(
                line_m=_line_m(r.position, source),
                route_m=None if r.geodesic_to_source is None else float(r.geodesic_to_source),
                heard=int(r.lateral_sign), truth=int(truth)))
    return Sounding(
        status=READ,
        d_start=d_start,
        sounding_steps=len(sounding),
        in_time=need <= len(sounding),
        d_min_sounding=min(routes) if routes else None,
        reached_sounding=reached is not None and int(reached) < offset,
        anchor_pairs=pairs,
        anchor_disagree=disagree,
        ear=tuple(ear),
        **base)


def _label(path: pathlib.Path) -> str:
    return "/".join(path.resolve().parts[-2:])


def load_run(arm_dir: str) -> SoundingRun:
    """Every episode under one arm's directory, or the reasons the run is refused."""
    from earshot.task.smoke import episode_indices

    root = pathlib.Path(arm_dir)
    label = _label(root)
    _, episodes_dir = run_paths(root)
    if episodes_dir.is_dir():
        return SoundingRun(label, str(root), 0, (), (
            "{} is a scene directory: it holds episodes itself. Pass the arm directory "
            "above it, <tag>/<arm>.".format(root),))
    episodes: List[Sounding] = []
    refusals: List[str] = []
    mismatched: Dict[Tuple[str, Optional[str]], int] = {}
    n_episodes = 0
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        indices = episode_indices(str(scene_dir))
        if not indices:
            continue
        audio, why = scene_audio(scene_dir)
        if audio is None:
            refusals.append(why)
            continue
        for index in indices:
            _, audit_path = episode_paths(scene_dir, index)
            audit = read_audit(audit_path)
            n_episodes += 1
            for name, wanted in FULL_ARM:
                got = getattr(audit, name)
                if got != wanted:
                    mismatched[(name, got)] = mismatched.get((name, got), 0) + 1
            episodes.append(episode_sounding(audit, run=label, scene=scene_dir.name))
    if not n_episodes and not refusals:
        refusals.append("no episode records under {}".format(root))
    for (name, got), count in sorted(mismatched.items(), key=lambda kv: str(kv[0])):
        refusals.append("not `full`'s rule: {} is {} on {} of {} episode(s)".format(
            name, "unrecorded" if got is None else repr(got), count, n_episodes))
    return SoundingRun(label, str(root), n_episodes, tuple(episodes), tuple(refusals))


def _share(numerator: int, denominator: int) -> Optional[float]:
    return (float(numerator) / float(denominator)) if denominator else None


def _band(value: float, edges: Sequence[float]) -> int:
    return sum(1 for edge in edges if float(value) >= edge)


def _band_names(edges: Sequence[float]) -> Tuple[str, ...]:
    return tuple(["under {:g} m".format(edges[0])]
                 + ["{:g} to {:g} m".format(lo, hi) for lo, hi in zip(edges, edges[1:])]
                 + ["{:g} m and over".format(edges[-1])])


def _grade(steps: Sequence[EarStep]) -> Dict[str, Any]:
    right = sum(1 for s in steps if s.heard == s.truth)
    silent = sum(1 for s in steps if s.heard == LATERAL_AMBIGUOUS)
    heard = len(steps) - silent
    return {"n": len(steps), "right": right, "heard_nothing": silent,
            "right_of_heard": _share(right, heard), "right_of_all": _share(right, len(steps))}


def summarize(runs: Sequence[SoundingRun]) -> Dict[str, Any]:
    """Every number the report prints. Pure."""
    def clock(episodes: Sequence[Sounding]) -> Dict[str, Any]:
        read = [s for s in episodes if s.status == READ]
        never5 = [s for s in read if not s.reached_sounding
                  and (s.d_min_sounding is None or s.d_min_sounding >= 5.0)]
        return {
            "n_read": len(read),
            "status": {name: sum(1 for s in episodes if s.status == name)
                       for name in STATUSES},
            "out_of_time": sum(1 for s in read if not s.in_time),
            "reached_sounding": sum(1 for s in read if s.reached_sounding),
            "never_within_5m": len(never5),
            "never_within_5m_out_of_time": sum(1 for s in never5 if not s.in_time),
            "median_d_start": statistics.median([s.d_start for s in read]) if read else None,
            "median_sounding_steps": (statistics.median([s.sounding_steps for s in read])
                                      if read else None),
        }

    pooled_eps = [s for run in runs for s in run.episodes]
    read = [s for s in pooled_eps if s.status == READ]
    steps = [e for s in read for e in s.ear]
    by_start = []
    for b, name in enumerate(_band_names(START_EDGES_M)):
        mine = [s for s in read if _band(s.d_start, START_EDGES_M) == b]
        by_start.append({
            "band": name, "n": len(mine),
            "reached_sounding": sum(1 for s in mine if s.reached_sounding),
            "out_of_time": sum(1 for s in mine if not s.in_time),
            "median_d_min_sounding": (
                statistics.median([s.d_min_sounding for s in mine
                                   if s.d_min_sounding is not None])
                if any(s.d_min_sounding is not None for s in mine) else None),
        })
    by_line = [dict(_grade([e for e in steps if _band(e.line_m, LINE_EDGES_M) == b]),
                    band=name)
               for b, name in enumerate(_band_names(LINE_EDGES_M))]
    open_steps = [e for e in steps if e.route_m is not None and e.line_m > 0.0
                  and e.route_m <= OPEN_RATIO * e.line_m]
    walled = [e for e in steps if e.route_m is not None and e.line_m > 0.0
              and e.route_m > OPEN_RATIO * e.line_m]
    pairs = sum(s.anchor_pairs for s in read)
    disagree = sum(s.anchor_disagree for s in read)
    return {
        "runs": {run.label: dict(clock(run.episodes),
                                 ear=_grade([e for s in run.episodes for e in s.ear]))
                 for run in runs},
        "pooled": dict(clock(pooled_eps), ear=_grade(steps)),
        "by_start": by_start,
        "ear_by_line": by_line,
        "ear_open": _grade(open_steps),
        "ear_walled": _grade(walled),
        "anchor_pairs": pairs,
        "anchor_disagree": disagree,
        "heading_share_disagree": _share(disagree, pairs),
        "red": (not steps) or (pairs > 0 and disagree / pairs > MAX_HEADING_DISAGREE),
    }


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.1%}".format(value)


def _ear_cell(grade: Dict[str, Any]) -> str:
    return "{:>7} steps  right {:>6} of those heard ({} of all; heard nothing on {})".format(
        grade["n"], _pct(grade["right_of_heard"]), _pct(grade["right_of_all"]),
        grade["heard_nothing"])


def format_report(runs: Sequence[SoundingRun], summary: Dict[str, Any]) -> str:
    lines = ["THE SOUNDING HALF: could the detour reach the source while it sounded, and",
             "does the ear cue point at it? NOT A GATE.", ""]
    lines.append("WHAT WAS READ, per run")
    for label, r in summary["runs"].items():
        lines.append("  {:<16} {}".format(label, "  ".join(
            "{} {}".format(k, v) for k, v in r["status"].items() if v)))
    lines.append("")

    lines.append("1. THE CLOCK: could any steering arrive before the source stopped?")
    lines.append("   out of time = (route at the detour's start - {:g} m) / {:g} m is more "
                 "steps than it sounded,".format(ARRIVAL_RING_M, STEP_M))
    lines.append("   which charges no turn and no wall, so it is optimistic for the agent")
    for label, r in list(summary["runs"].items()) + [("pooled", summary["pooled"])]:
        lines.append(
            "  {:<16} read {:>4}  reached while sounding {:>4}  out of time {:>4} ({})  "
            "never within 5 m {:>4}, of them out of time {:>4}".format(
                label, r["n_read"], r["reached_sounding"], r["out_of_time"],
                _pct(_share(r["out_of_time"], r["n_read"])), r["never_within_5m"],
                r["never_within_5m_out_of_time"]))
    pooled = summary["pooled"]
    lines.append("  median route at the detour's start {}; median steps sounded {}".format(
        "n/a" if pooled["median_d_start"] is None
        else "{:.2f} m".format(pooled["median_d_start"]),
        pooled["median_sounding_steps"]))
    lines.append("")
    lines.append("  by route at the detour's start, pooled:")
    lines.append("  {:<16} {:>8} {:>22} {:>14} {:>26}".format(
        "", "detours", "reached while sounding", "out of time",
        "nearest while sounding (median)"))
    for row in summary["by_start"]:
        lines.append("  {:<16} {:>8} {:>22} {:>14} {:>26}".format(
            row["band"], row["n"],
            "{} ({})".format(row["reached_sounding"],
                             _pct(_share(row["reached_sounding"], row["n"]))),
            "{} ({})".format(row["out_of_time"], _pct(_share(row["out_of_time"], row["n"]))),
            "n/a" if row["median_d_min_sounding"] is None
            else "{:.2f} m".format(row["median_d_min_sounding"])))
    lines.append("")

    lines.append("2. THE EAR: does the heard left/right sign match the side the source is on?")
    lines.append("   sounding detour steps with a confirmed heading; the side is by straight "
                 "line; 50% is chance")
    for label, r in summary["runs"].items():
        lines.append("  {:<16} {}".format(label, _ear_cell(r["ear"])))
    lines.append("  {:<16} {}".format("pooled", _ear_cell(summary["pooled"]["ear"])))
    lines.append("  by straight-line distance, pooled:")
    for row in summary["ear_by_line"]:
        lines.append("  {:<16} {}".format(row["band"], _ear_cell(row)))
    lines.append("  by how open the path is, pooled:")
    lines.append("  {:<16} {}".format("route <= {:g}x line".format(OPEN_RATIO),
                                      _ear_cell(summary["ear_open"])))
    lines.append("  {:<16} {}".format("around a wall", _ear_cell(summary["ear_walled"])))
    lines.append("")

    lines.append("3. THE HEADING REBUILD: {} anchor pair(s), {} disagree ({})".format(
        summary["anchor_pairs"], summary["anchor_disagree"],
        _pct(summary["heading_share_disagree"])))
    lines.append("")
    lines.append("WHAT EACH ANSWER MEANS (written before the read, not a branch)")
    lines.append("  - Most 'never within 5 m' detours out of time: the sounding window, not "
                 "the steering, is the limit there.")
    lines.append("  - The ear right well above 50% where detours stall: steering by it for "
                 "the whole window is the candidate, under its own ADR and gate.")
    lines.append("  - The ear near 50% there: the cue does not carry the side at those "
                 "distances, and no steering rule can use it.")
    if summary["red"]:
        lines.append("")
        lines.append("RED: no step was graded, or more than {:.0%} of heading pairs "
                     "disagree. The ear table is not a measurement.".format(
                         MAX_HEADING_DISAGREE))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+",
                        help="one arm's directory per render, <tag>/<arm>, `full`'s rule "
                             "at TC 0")
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args(argv)

    missing = [d for d in args.run_dirs if not pathlib.Path(d).is_dir()]
    if missing:
        print("no such run directory: {}".format(", ".join(missing)))
        return 2
    resolved = [str(pathlib.Path(d).resolve()) for d in args.run_dirs]
    if len(set(resolved)) != len(resolved):
        print("the same run directory was given twice, so its episodes would count twice")
        return 2
    runs = [load_run(d) for d in args.run_dirs]
    refused = [run for run in runs if run.refusals]
    if refused:
        for run in refused:
            print("REFUSED {}:".format(run.path))
            for reason in run.refusals:
                print("  " + reason)
        return 2
    summary = summarize(runs)
    if args.json:
        print(json.dumps({
            "summary": summary,
            "episodes": [dict((k, v) for k, v in asdict(s).items() if k != "ear")
                         for run in runs for s in run.episodes],
        }, indent=2))
    else:
        print(format_report(runs, summary))
    return 2 if summary["red"] else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
