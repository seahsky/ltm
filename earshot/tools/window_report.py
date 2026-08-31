"""What one windowed sweep measured, per arm. The reader ``window_pilot.sh`` should
have had.

**This module exists because the pilot's reader was a shell heredoc and nothing could
test it.** ``pilot-1`` ran 120 episodes across three arms in 42 minutes, wrote every one
of them to disk, and printed ``NO EPISODES ON DISK -- this arm did not run`` three times:
the walk matched a file named ``audit.json`` and this tree writes ``ep0000.audit.json``
(``report/artifacts.py``). Nothing was wrong with the run. The three numbers the pilot was
built to produce were simply never read, and no test in the tree could have said so,
because the reader was forty lines of Python inside a bash string.

So the enumeration is not re-derived here. ``smoke.episode_indices`` is the one function
in this repo that knows what an episode file is called, and it is called rather than
copied — a second implementation of that glob is precisely what cost the 42 minutes.

**SWS is likewise taken and not re-derived.** ``runner.silent_phase_tally`` carries
ADR-0017's bar (an eligible episode whose record shows no active reverb tail *raises*),
and a reader that recomputed ``reached >= offset`` in a loop would publish an SWS the
tally refuses to publish. The refusal is caught and REPORTED here rather than allowed to
kill the readout: an arm whose SWS is barred is a finding about that arm, and the other
arms' numbers are still what the pilot ran for.

Every number below is written by the runner into the audit record. A disagreement between
this module and an audit is a bug in this module and never a second opinion about a run.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from earshot.report.artifacts import episode_paths, read_audit, run_paths
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.task.runner import SilentPhaseTally, TailNotActiveError, silent_phase_tally
from earshot.task.smoke import episode_indices

__all__ = [
    "PILOT_ARMS",
    "ArmReading",
    "scene_dirs",
    "load_arm_audits",
    "read_arm",
    "read_sweep",
    "format_arm",
    "format_report",
    "main",
]

# The arms `window_pilot.sh` runs, in the order it runs them: the control first, so a
# crash in the windowed arms still leaves the arm every comparison needs on disk. Held
# here so the reader prints them in a fixed order rather than in `os.listdir` order, and
# so an arm directory missing entirely is visible as missing rather than as absent.
PILOT_ARMS = ("cont-alarm", "win-alarm", "win-burst")


@dataclass(frozen=True)
class ArmReading:
    """One arm of a sweep, read off the audit records. Pure data.

    ``sws`` and ``sws_refused`` are both ``Optional`` and never both set. ``None`` on
    each means nothing was eligible, which is NOT_RUN and never 0.0. A populated
    ``sws_refused`` is ADR-0017's bar firing: the tally would not publish a rate over
    episodes whose silent phase carried no tail.
    """

    arm: str
    scenes: Tuple[str, ...]
    n_episodes: int
    n_source_reached: int
    mean_steps: Optional[float]
    mean_audio_s: Optional[float]
    max_step_audio_s: Optional[float]
    tally: Optional[SilentPhaseTally]
    sws: Optional[float]
    sws_refused: Optional[str]
    onset_delays: Tuple[float, ...]
    n_onset_censored: int
    n_windowed: int
    n_heard_within_window: int
    cue_tail_steps: Tuple[int, ...]
    phase_folds: Tuple[int, ...]

    @property
    def reached_rate(self) -> Optional[float]:
        """``None`` on an empty arm, because 0 of 0 is not 0.0."""
        if self.n_episodes == 0:
            return None
        return self.n_source_reached / float(self.n_episodes)


def scene_dirs(arm_dir: str) -> Tuple[pathlib.Path, ...]:
    """Every scene directory under one arm, in name order.

    A scene directory is one that holds an ``episodes/`` directory — the shape
    ``report/artifacts.py`` writes. A directory without one is skipped rather than
    counted as a scene that produced nothing: the pilot puts its per-run logs beside the
    scene directories, and a log is not a barren scene.
    """
    root = pathlib.Path(arm_dir)
    if not root.is_dir():
        return ()
    return tuple(
        child
        for child in sorted(root.iterdir())
        if child.is_dir() and run_paths(child)[1].is_dir()
    )


def load_arm_audits(arm_dir: str) -> Tuple[Tuple[str, ...], Tuple[EpisodeAudit, ...]]:
    """``(scene_ids, audits)`` for one arm. The only function here that touches the disk.

    The scene ids come from the directory names rather than from ``audit.scene_id``: a
    scene that built zero episodes has a directory and no audit to read an id off, and
    the sweep's shape has to survive into the reading.
    """
    scenes: List[str] = []
    audits: List[EpisodeAudit] = []
    for scene in scene_dirs(arm_dir):
        scenes.append(scene.name)
        for index in episode_indices(str(scene)):
            _agent_path, audit_path = episode_paths(str(scene), index)
            audits.append(read_audit(audit_path))
    return tuple(scenes), tuple(audits)


def _metric(audit: EpisodeAudit, key: str) -> Optional[float]:
    value = audit.metrics.get(key)
    return None if value is None else float(value)


def read_arm(arm_dir: str, *, arm: str) -> ArmReading:
    """Read one arm directory into an ``ArmReading``.

    Never raises on a barred SWS. ``silent_phase_tally``'s ``TailNotActiveError`` is
    caught and carried as ``sws_refused``, because the refusal is a finding and the arm's
    other numbers — the episode's cost above all — are still worth printing.
    """
    scenes, audits = load_arm_audits(arm_dir)
    if not audits:
        return ArmReading(
            arm=arm,
            scenes=scenes,
            n_episodes=0,
            n_source_reached=0,
            mean_steps=None,
            mean_audio_s=None,
            max_step_audio_s=None,
            tally=None,
            sws=None,
            sws_refused=None,
            onset_delays=(),
            n_onset_censored=0,
            n_windowed=0,
            n_heard_within_window=0,
            cue_tail_steps=(),
            phase_folds=(),
        )

    steps = [float(len(audit.steps)) for audit in audits]
    renders = [audit.audio_render_summary() for audit in audits]
    totals = [row["total_s"] for row in renders if row]
    maxima = [row["max_s"] for row in renders if row]

    tally: Optional[SilentPhaseTally] = None
    sws: Optional[float] = None
    refused: Optional[str] = None
    try:
        tally = silent_phase_tally(audits)
        sws = tally.sws
    except TailNotActiveError as error:
        refused = str(error)

    delays = tuple(
        value
        for value in (_metric(audit, "onset_delay_steps") for audit in audits)
        if value is not None
    )
    cue_tail = tuple(
        int(value)
        for value in (_metric(audit, "sounding_cue_tail_steps") for audit in audits)
        if value is not None
    )
    folds = tuple(
        int(value)
        for value in (_metric(audit, "sounding_phase_folds") for audit in audits)
        if value is not None
    )

    return ArmReading(
        arm=arm,
        scenes=scenes,
        n_episodes=len(audits),
        # `funnel_stage >= SOURCE_REACHED` and NOT `source_reached_step is not None`,
        # because that is how `SilentPhaseTally` counts the SR it prints beside SWS. Two
        # definitions of "reached" in one report is how a reader comes to quote the
        # wrong one; `test_window_report.py` asserts the two agree on a real record.
        n_source_reached=sum(
            1 for audit in audits if audit.funnel_stage >= FunnelStage.SOURCE_REACHED
        ),
        mean_steps=statistics.mean(steps),
        mean_audio_s=statistics.mean(totals) if totals else None,
        max_step_audio_s=max(maxima) if maxima else None,
        tally=tally,
        sws=sws,
        sws_refused=refused,
        onset_delays=delays,
        n_onset_censored=sum(
            1 for audit in audits if _metric(audit, "onset_delay_censored")
        ),
        # `heard_within_window` is written by the runner ONLY under `offset_step is not
        # None`, so on the continuous arm it is absent on every episode. Its denominator
        # is therefore the windowed episodes and not all of them: `pilot-1`'s first
        # readable output said "heard while still sounding: 0 of 40" for `cont-alarm`
        # beside "40 of 40" for the windowed arms, which reads as a difference between
        # the arms and is a metric that does not exist on one of them.
        n_windowed=sum(
            1
            for audit in audits
            if audit.sounding_window is not None
            and audit.sounding_window.offset_step is not None
        ),
        n_heard_within_window=sum(
            1 for audit in audits if _metric(audit, "heard_within_window")
        ),
        cue_tail_steps=cue_tail,
        phase_folds=folds,
    )


def read_sweep(root: str, *, arms: Sequence[str] = PILOT_ARMS) -> Tuple[ArmReading, ...]:
    """Every named arm under one sweep directory, in the order given.

    An arm with no directory at all comes back as an empty reading rather than dropped.
    A missing arm is NOT_RUN, and NOT_RUN is red — a report that quietly lists two arms
    when three were asked for is the shape of failure this repo keeps paying for.
    """
    base = pathlib.Path(root)
    return tuple(read_arm(str(base / arm), arm=arm) for arm in arms)


def _histogram(values: Sequence[int]) -> str:
    counts: Dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return " ".join(
        "{}x{}".format(count, value) for value, count in sorted(counts.items())
    )


def format_arm(reading: ArmReading) -> List[str]:
    """One arm as printable lines. Numbers rather than a verdict (ADR-0014)."""
    if reading.n_episodes == 0:
        return [
            "  {:11s}  NOT_RUN: no episode on disk under this arm ({} scene dir(s))"
            .format(reading.arm, len(reading.scenes))
        ]

    lines = [
        "  {:11s} n={:3d} over {} scene(s)   source reached {:3d} ({:5.1%})".format(
            reading.arm,
            reading.n_episodes,
            len(reading.scenes),
            reading.n_source_reached,
            reading.reached_rate or 0.0,
        ),
        "  {:11s}   THE COST: {:5.1f} steps/ep, {:6.2f} audio s/ep, worst single step "
        "{:.4f} s".format(
            "",
            reading.mean_steps or 0.0,
            reading.mean_audio_s or 0.0,
            reading.max_step_audio_s or 0.0,
        ),
    ]

    tally = reading.tally
    if reading.sws_refused is not None:
        lines.append(
            "  {:11s}   SWS REFUSED — ADR-0017's bar fired, and that is the finding:"
            .format("")
        )
        lines.append("  {:11s}     {}".format("", reading.sws_refused))
    elif tally is None or tally.n_window_closed == 0:
        lines.append(
            "  {:11s}   SWS NOT_RUN: no episode ran past its own offset step. On the "
            "continuous arm that is by design; anywhere else the budget ends before "
            "the window does.".format("")
        )
    else:
        lines.append(
            "  {:11s}   SWS {}/{} = {:.3f}   silent-phase tail audible in {} of them"
            .format(
                "",
                tally.n_reached_after_offset,
                tally.n_window_closed,
                reading.sws if reading.sws is not None else 0.0,
                tally.n_tail_audible,
            )
        )

    if reading.onset_delays:
        lines.append(
            "  {:11s}   onset delay steps: n={} median {:.1f} max {:.1f}   CENSORED "
            "(never heard it): {}".format(
                "",
                len(reading.onset_delays),
                statistics.median(reading.onset_delays),
                max(reading.onset_delays),
                reading.n_onset_censored,
            )
        )
    else:
        lines.append(
            "  {:11s}   ONSET NEVER FIRED in any episode of this arm — the climb had "
            "nothing to climb ({} censored)".format("", reading.n_onset_censored)
        )
    if reading.n_windowed == 0:
        lines.append(
            "  {:11s}   heard while still sounding: n/a — no offset step on this arm, "
            "so the runner writes no such metric".format("")
        )
    else:
        lines.append(
            "  {:11s}   heard while still sounding: {} of {} windowed".format(
                "", reading.n_heard_within_window, reading.n_windowed
            )
        )
    if reading.cue_tail_steps:
        lines.append(
            "  {:11s}   cue_tail_steps {}   phase folds {}".format(
                "", _histogram(reading.cue_tail_steps), _histogram(reading.phase_folds) or "-"
            )
        )
    return lines


def format_report(readings: Sequence[ArmReading]) -> str:
    lines: List[str] = []
    for reading in readings:
        lines.extend(format_arm(reading))
        lines.append("")
    lines.append(
        "  A pilot times the episode and exposes mechanism failures. It does not resolve"
    )
    lines.append(
        "  a difference between arms: repeat-1 measured a 16.2% flip rate on "
        "BYTE-IDENTICAL reruns."
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", help="a tag directory holding one directory per arm")
    parser.add_argument(
        "--arms",
        default=" ".join(PILOT_ARMS),
        help="space-separated arm directory names, in report order",
    )
    args = parser.parse_args(argv)

    if not pathlib.Path(args.sweep_dir).is_dir():
        print("no such sweep directory: {}".format(args.sweep_dir))
        return 2
    readings = read_sweep(args.sweep_dir, arms=tuple(args.arms.split()))
    print(format_report(readings))
    # Nonzero when NOTHING was read. A reader that exits 0 over an empty sweep is how
    # `pilot-1` announced three dead arms and let the driver print its summary anyway.
    return 0 if any(reading.n_episodes for reading in readings) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
