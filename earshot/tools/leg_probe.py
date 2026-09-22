"""Does the level fall along a walking leg, and does ``temporalCoherence`` make it?

    nrun bash earshot/tools/leg_probe.sh --tag walk-1
    python -m earshot.tools.leg_probe read runs/walk-1

**Why it exists.** The leg replay (PRs #149 to #156) found the cue falling along about
three quarters of sounding cast legs whichever way they walked, by a median 9.0% per
loop, and that is the number the controller reads: ``is_rising`` compares five readings
against five along a leg, and a level that falls whatever the agent does leans every one
of those comparisons toward "not rising".

The hold probe then rendered the standing first scans again on the box (``hold-2``, PR
#159) and read them FLAT in both arms against a recorded -11.0% per loop. That ruled out
the preset, the renderer at a fixed pose, the pipeline after the renderer and the scan's
heading -- for a standing agent. It could not rule them out for a walking one. Its only
sequence that moves, the rescan, is also the only row that leaned: -2.6% per loop with
the preset on against -0.9% with it off, at p = 0.05 over 76 poses. That is a hint, on
ten steps of walking, in the wrong sequence for the question.

This probe walks. It renders the recorded legs again, one per episode, with
``temporalCoherence`` on and with it off, and reads the level along the leg the way
``leg_replay`` reads the recorded one.

What each pose renders
----------------------

A pose is one COMPLETED leg that was read while the source sounded: the population
``by_sounding`` and ``loop_removed`` grade. One leg per episode, the first that
qualifies, because the sign test's unit is the episode and two legs of one episode share
a room, a source and a walk. The same episode in two runs is one sample, so a later
run's leg of an episode already posed is counted as a duplicate and left out.

Seated ``walk_in`` steps before the leg's first reading at the heading rebuilt from the
record, the agent takes the recorded actions through the leg. Every position is checked
against the record at ``POSITION_TOLERANCE_M``, so a heading rebuilt wrong or a motion
model that disagrees with habitat's shows up as a divergence and that pose is not read.
The leg's ``LEG_PERIOD`` readings are then graded by ``same_phase_change`` at the run's
own loop period, which is what the recorded number is.

Every render goes through the run's own ``heard_step``, with the run's clip and bed at
the audio configuration the runs recorded in ``env_report.json``, and the source
sounding throughout.

**One World per scene**, with ``place_source`` between poses, which is the structure the
run itself has. A pose's own walk-in stands between it and the previous pose's renders.
The hold probe needed one World per sequence KIND because it rendered three sequences at
one pose; this probe renders one.

**The instrument's own arm** (ADR-0014). FORCED folds the same renders with the source's
IR scaled by ``FORCED_DECAY`` per leg reading: a fall of known size that the readout must
see, in the same poses, the same walk and the same noise. Where the source is weak the
bed dilutes it, so the check is stricter than the number it is checking. The hold probe's
FROZEN arm is not repeated here: it measured the pipeline after the renderer flat on real
IRs in both arms (+0.0% and -0.0% over 60 readings), and folding one IR again along a
walk asks that same question a second time.

The RECORDED row is the second arm and the thing being explained: the same legs' own
``change_per_loop`` off the record. If it does not fall, these are not the legs the chain
is about and the branch is NOT_RUN.

The branch, pre-registered
--------------------------

FALLS needs an exact sign-test p at or under ``SIGN_ALPHA`` and a median change at or
under ``-MIN_CHANGE`` per loop, over at least ``MIN_POSES`` poses. Read in this order:

1. **NOT_RUN** -- a population is UNREAD, the recorded legs do not fall, or FORCED does
   not fall in an arm whose walk does not rise. The instrument did not answer.
2. **RENDERER** -- the walk falls with the preset on and with it off. The renderer's
   behaviour under motion is the cause, and it is not a preset knob: the next question is
   the renderer's own configuration (ray count, IR length), measured the same way.
3. **PRESET** -- the walk falls with ``temporalCoherence`` on only. The A/B ticket 01
   asked for, answered on a leg. The next run is ``full`` with the preset off beside
   ``full``, which is a sweep and not a probe.
4. **MIXED** -- the walk falls with the preset off only, or rises in either arm. Read the
   table before anything else; one of the two arms is not doing what its name says.
5. **SELECTION** -- flat in both arms. The legs' fall does not reproduce either, so
   neither the scans' fall nor the legs' is in the render. What both populations share is
   that ``is_rising`` decides which of them is read: a leg the level rose through is cut
   by a surge and never graded, exactly as a scan is. The next check is read-only and
   already built for the scans (``leg_replay.by_cut_scan``): grade the legs the surge
   cut, off ``_leg_outcome``'s own CUT_BY_SURGE.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Dict, Iterator, List, Mapping, Optional
from typing import Sequence, Tuple

import numpy as np

from earshot.report.audit import EpisodeAudit
from earshot.tools.hold_probe import (
    ARM_ORDER,
    ARMS,
    DEFAULT_WALK_IN,
    FALLS,
    MIN_WALK_IN,
    MIXED,
    NOT_RUN,
    PRESET,
    RENDERER,
    RISES,
    SELECTION,
    UNREAD,
    ProbeIO,
    ProbePose,
    Trace,
    plan_window,
    read_tag,
    render_arm,
    select_poses,
    trace_rescan,
    trend,
)
from earshot.tools.leg_replay import (
    COMPLETED,
    LEG_PERIOD,
    SOUNDING,
    EpisodeReplay,
    Leg,
    same_phase_change,
)

__all__ = [
    "LEG_ROWS",
    "LegResult",
    "plan_leg",
    "sounding_legs",
    "select_legs",
    "probe_legs",
    "render_legs",
    "decide",
    "readout",
    "read_walk",
    "format_readout",
    "main",
]

# The rows the readout prints. The first three decide the branch; the level of the
# room's own impulse response decides nothing and is printed because a walk that reads
# flat at the ears while the room's level moves is a different finding from both arms
# being flat.
LEG_ROWS: Tuple[str, ...] = ("walk", "walk_forced", "ir_walk")
_DECIDING = ("walk", "walk_forced")

# `hold` is the hold probe's header field and a walk has none: a leg's length is the
# record's, `LEG_PERIOD` readings. It is written as zero so one reader serves both.
NO_HOLD = 0


@dataclass(frozen=True)
class LegResult:
    """One leg's walk in one arm."""

    pose: ProbePose
    walk: Trace

    @property
    def diverged(self) -> bool:
        return self.walk.diverged_at is not None

    def as_dict(self) -> Dict[str, Any]:
        return {"pose": self.pose.as_dict(), "walk": self.walk.as_dict()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LegResult":
        return cls(pose=ProbePose.from_dict(data["pose"]),
                   walk=Trace.from_dict(data["walk"]))


def plan_leg(
    audit: EpisodeAudit, leg: Leg, *, walk_in: int
) -> Tuple[Optional[ProbePose], Optional[str]]:
    """The pose for one recorded leg, or why it cannot be one. Pure.

    A leg's readings start one step AFTER the turn that opened it, which is the slice
    ``leg_replay._measured`` grades, so the pose reads from there and keeps the leg's own
    ``start_step`` as its name.
    """
    if leg.phase_folds is None or leg.change_per_loop is None:
        raise ValueError(
            "leg {} of episode {} of {}/{} was not read: plan only the legs graded while "
            "the source sounded".format(leg.start_step, leg.episode, leg.run, leg.scene))
    return plan_window(
        audit, run=leg.run, scene=leg.scene, episode=int(leg.episode),
        start_step=int(leg.start_step), first_step=int(leg.start_step) + 1,
        n_readings=LEG_PERIOD, walk_in=walk_in, period=int(leg.phase_folds),
        recorded_change=float(leg.change_per_loop),
        what="leg {}".format(leg.start_step))


def sounding_legs(
    audit: EpisodeAudit, replay: EpisodeReplay, *, walk_in: int
) -> Iterator[Tuple[Optional[ProbePose], Optional[str]]]:
    """The episode's FIRST completed sounding leg, planned. Pure.

    One leg per episode and not every leg: the sign test counts poses as independent,
    and two legs of one episode share a room, a source and most of a walk. The first is
    taken rather than the longest or the loudest, because any rule that reads the levels
    would choose the poses by the quantity under test.
    """
    for leg in replay.legs:
        if (leg.outcome == COMPLETED and leg.sounding == SOUNDING
                and leg.change_per_loop is not None and leg.phase_folds is not None):
            yield plan_leg(audit, leg, walk_in=walk_in)
            return


def select_legs(
    arm_dirs: Sequence[str], *, walk_in: int, scenes: Optional[Sequence[str]] = None
):
    """``select_poses`` over legs: the same runs, refusals and de-duplication."""
    return select_poses(arm_dirs, walk_in=walk_in, scenes=scenes,
                        candidates=sounding_legs)


def probe_legs(
    poses: Sequence[ProbePose],
    fresh: Callable[[], ContextManager[ProbeIO]],
    *,
    clip: np.ndarray,
    bed_cue: np.ndarray,
    hop: int,
    hold: int = NO_HOLD,
    progress: Callable[[str], None] = print,
) -> List[LegResult]:
    """Every pose in one scene, walked in one World. ``hold`` is the header's and unused."""
    results: List[LegResult] = []
    with fresh() as io:
        for pose in poses:
            io.place_source(pose.source)
            trace = trace_rescan(pose, io, clip=clip, bed_cue=bed_cue, hop=hop)
            results.append(LegResult(pose=pose, walk=trace))
            progress("  walk episode {} leg {}: {}".format(
                pose.episode, pose.start_step,
                "rendered" if trace.diverged_at is None
                else "DIVERGED at render {}".format(trace.diverged_at)))
    return results


def render_legs(
    *, out_dir: str, arm: str, selection: Any, data_root: str,
    progress: Callable[[str], None] = print,
) -> int:
    """Render every leg in one arm. The box half, through the hold probe's own writer."""
    return render_arm(out_dir=out_dir, arm=arm, selection=selection, hold=NO_HOLD,
                      data_root=data_root, probe=probe_legs, progress=progress)


def _per_pose(result: LegResult) -> Dict[str, Optional[float]]:
    """Every row's change at one pose in one arm."""
    trace, lag = result.walk, int(result.pose.period)
    read = slice(int(trace.read_from), None)
    return {
        "walk": same_phase_change(list(trace.cue)[read], lag=lag),
        "walk_forced": same_phase_change(list(trace.forced)[read], lag=lag),
        "ir_walk": same_phase_change(list(trace.ir_level)[read], lag=lag),
    }


def decide(verdicts: Mapping[str, Mapping[str, str]]) -> Tuple[str, str]:
    """The pre-registered branch, and why. Pure.

    ``verdicts`` is ``{row: {arm: verdict}}`` plus ``{"recorded": {"run": verdict}}``.
    The order is the module docstring's and was fixed before the first run.
    """
    unread = [
        "{} {}".format(arm, row) for row in ("recorded",) + _DECIDING
        for arm, verdict in verdicts[row].items() if verdict == UNREAD]
    if unread:
        return NOT_RUN, (
            "no verdict on {}: too few poses read, or none. A population that could not "
            "be read is red".format(", ".join(sorted(unread))))
    if verdicts["recorded"]["run"] != FALLS:
        return NOT_RUN, (
            "the recorded legs read {} and not {}. These are not the legs the chain is "
            "about, and nothing rendered here explains a fall they do not have".format(
                verdicts["recorded"]["run"], FALLS))
    blind = [arm for arm in ARM_ORDER
             if verdicts["walk_forced"][arm] != FALLS and verdicts["walk"][arm] != RISES]
    if blind:
        return NOT_RUN, (
            "the forced fall was not detected in arm(s) {}: the readout cannot see a fall "
            "of the size it exists to explain, so its flat rows mean nothing".format(
                ", ".join(blind)))
    falls = [arm for arm in ARM_ORDER if verdicts["walk"][arm] == FALLS]
    rises = [arm for arm in ARM_ORDER if verdicts["walk"][arm] == RISES]
    if len(falls) == len(ARM_ORDER):
        return RENDERER, (
            "the level falls along the leg with temporalCoherence on AND off, at "
            "positions checked against the record. The preset is not the cause and the "
            "renderer under motion is. Next is the renderer's own configuration -- ray "
            "count and IR length -- measured this way, not a preset knob")
    if falls == ["tc1"]:
        return PRESET, (
            "the level falls along the leg with temporalCoherence ON and not with it "
            "off, at the same poses and the same walk. The preset is the cause. The next "
            "run is a sweep, `full` with the preset off beside `full`, and it prices what "
            "the fall costs the controller")
    if falls or rises:
        return MIXED, (
            "the arms disagree in a way the branch did not anticipate: falls {}, rises "
            "{}. Read the table before anything else -- one arm is not rendering what its "
            "name says".format(falls or "none", rises or "none"))
    return SELECTION, (
        "the recorded legs fall and the same legs walked again are FLAT with "
        "temporalCoherence on and FLAT with it off. Neither the scans' fall (hold-2) nor "
        "the legs' is in the render. What both share is that `is_rising` chooses which "
        "of them is read: a leg the level rose through is cut by a surge and never "
        "graded, as a scan is. The next check is read-only -- grade the legs the surge "
        "cut, the way `leg_replay.by_cut_scan` grades the scans")


def readout(
    arms: Mapping[str, Mapping[Tuple[str, str, int, int], LegResult]]
) -> Dict[str, Any]:
    """Pair the arms by pose, read every row, and decide. Pure.

    The same shape ``hold_probe.readout`` returns, so the two readouts are read the same
    way: a pose is read only if both arms rendered it and neither diverged from the
    record.
    """
    keys = sorted(set.intersection(*(set(arms[a]) for a in ARM_ORDER)))
    diverged = {a: sum(1 for k in keys if arms[a][k].diverged) for a in ARM_ORDER}
    read = [k for k in keys if not any(arms[a][k].diverged for a in ARM_ORDER)]
    per_pose = {a: [_per_pose(arms[a][k]) for k in read] for a in ARM_ORDER}
    rows: Dict[str, Dict[str, Dict[str, Any]]] = {
        "recorded": {"run": trend([arms[ARM_ORDER[0]][k].pose.recorded_change
                                   for k in read])}}
    for row in LEG_ROWS:
        rows[row] = {a: trend([p[row] for p in per_pose[a] if p[row] is not None])
                     for a in ARM_ORDER}
    verdicts = {row: {col: cell["verdict"] for col, cell in cols.items()}
                for row, cols in rows.items()}
    branch, why = decide(verdicts)
    return {
        "n_paired": len(keys),
        "only_in": {a: len(set(arms[a]) - set(keys)) for a in ARM_ORDER},
        "diverged": diverged,
        "n_read": len(read),
        "rows": rows,
        "branch": branch,
        "why": why,
    }


def read_walk(tag_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """The tag's two arms, read back off disk as legs."""
    arms, headers, errors = read_tag(tag_dir, result_type=LegResult)
    return arms, headers, errors


_ROWS = (("recorded", "recorded leg, the run's own renders"),
         ("walk", "walk: the same walk-in and leg"),
         ("walk_forced", "CHECK forced fall, the walk"),
         ("ir_walk", "IR level: the walk"))
_ROW = "  {:<38} {:>5} {:>5} {:>7} {:>7} {:>9} {:>8}  {}"


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else "{:.1f}%".format(100.0 * value)


def format_readout(
    result: Mapping[str, Any], *, tag: str, errors: Sequence[str]
) -> str:
    """The readout as text. Pure, so the branch's wording is Mac-testable."""
    lines = [
        "leg probe  {}".format(tag),
        "  arms: tc1 = temporalCoherence on (the shipped preset), tc0 = off",
        "  legs in both arms: {}   only in tc1: {}   only in tc0: {}".format(
            result["n_paired"], result["only_in"]["tc1"], result["only_in"]["tc0"]),
        "  diverged from the record: tc1 {}, tc0 {}   READ: {}".format(
            result["diverged"]["tc1"], result["diverged"]["tc0"], result["n_read"]),
        "",
        "  change per loop (pairs one loop apart, over the leg's readings)",
        _ROW.format("", "arm", "read", "rose", "fell", "median", "sign p", "reads"),
    ]
    for key, name in _ROWS:
        for arm, entry in result["rows"][key].items():
            median = entry["median"]
            lines.append(_ROW.format(
                name if arm in ("run", "tc1") else "", arm, entry["n"],
                _pct(None if entry["n"] == 0 else entry["rose"] / entry["n"]),
                _pct(None if entry["n"] == 0 else entry["fell"] / entry["n"]),
                "n/a" if median is None else "{:+.1f}%".format(
                    round(100.0 * median, 1) or 0.0),
                "n/a" if entry["p"] is None else "{:.4f}".format(entry["p"]),
                entry["verdict"]))
            name = ""
    for error in errors:
        lines.append("  SCENE ERROR {}".format(error))
    lines += [
        "",
        "  BRANCH: {}".format(result["branch"]),
        "  {}".format(result["why"]),
        "",
        "  Pre-registered in leg_probe.py before the first box run. FALLS needs a "
        "sign-test",
        "  p <= 0.01 and a median change <= -3% per loop over >= 20 poses. The IR level "
        "row",
        "  decides nothing.",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m earshot.tools.leg_probe",
        description="Does the level fall along a walking leg? TC on against off.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("select", "render"):
        cmd = sub.add_parser(name)
        cmd.add_argument("runs", nargs="+",
                         help="`full` arm directories, <tag>/full; the first wins an episode")
        cmd.add_argument("--walk-in", type=int, default=DEFAULT_WALK_IN)
        cmd.add_argument("--scenes", default="", help="comma-separated scene labels")
    render = sub.choices["render"]
    render.add_argument("--arm", required=True, choices=list(ARMS))
    render.add_argument("--out", required=True, help="the tag directory, runs/<tag>")
    render.add_argument("--data-root", default=".")
    read = sub.add_parser("read")
    read.add_argument("tag_dir")
    args = parser.parse_args(argv)

    if args.command == "read":
        arms, _headers, errors = read_walk(args.tag_dir)
        result = readout(arms)
        print(format_readout(result, tag=args.tag_dir, errors=errors))
        return 2 if result["branch"] == NOT_RUN else 0

    if int(args.walk_in) < MIN_WALK_IN:
        print("FATAL: --walk-in {} is under the {} folds the cue takes to settle".format(
            args.walk_in, MIN_WALK_IN))
        return 2
    selection = select_legs(
        args.runs, walk_in=int(args.walk_in),
        scenes=None if not args.scenes else args.scenes.split(","))
    for refusal in selection.refusals:
        print("REFUSED: {}".format(refusal))
    if selection.refusals:
        return 2
    print("legs offered by the runs: {}   poses: {}".format(
        selection.n_first_scans, len(selection.poses)))
    for why, count in selection.excluded.items():
        print("  excluded {}: {}".format(count, why))
    if selection.zero_yield:
        print("  zero-yield, no episode to probe: {}".format(
            " ".join(selection.zero_yield)))
    if not selection.poses:
        print("FATAL: no pose to render")
        return 2
    if args.command == "select":
        return 0
    rendered = render_legs(out_dir=args.out, arm=args.arm, selection=selection,
                           data_root=args.data_root)
    print("arm {}: {} of {} scene(s) rendered".format(
        args.arm, rendered, len({p.scene for p in selection.poses})))
    return 0 if rendered else 1


if __name__ == "__main__":
    sys.exit(main())
