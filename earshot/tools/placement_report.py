"""Did the anomaly sound land at the object its class is anchored at? One sweep, by arm.

    python -m earshot.tools.placement_report runs/<tag>
    python -m earshot.tools.placement_report runs/<tag> --arms "full scan-only"
    python -m earshot.tools.placement_report runs/<tag>/full        # one arm directory

Read-only, no GPU, seconds. It reads the audit records a finished run already wrote, so a
sweep can be re-read without re-running it.

**Why it exists.** ADR-0022 made `place_anomaly_source` prefer the object category the
anomaly class is anchored at, and every number measured before that date was produced under
geometric placement. `abl-2` re-baselined the ablation table against the new placement and
its SRs moved. THAT MOVEMENT IS NOT EVIDENCE THE CHANGE TOOK: a re-run of the same task moves
too, by 3.0 points on byte-identical bytes (`repeat-1`). The only record that says which
branch an episode took is `source_at_class_anchor`, and nothing in `tools/` read it, so a
sweep could re-measure the old task with new dice and print a clean table. Asking the
question with a shell heredoc is how `pilot-1` lost 42 minutes of V100 time to a filename, so
the episode enumeration here is `smoke.episode_indices` and `artifacts.episode_paths` called,
never a second glob written.

**MISSING IS NOT FALSE.** The field arrived with ADR-0022. An episode whose audit does not
carry it was written by a runner that could not have anchored anything, so it gets its own
column and is never folded into the geometric count. An arm that is entirely MISSING exits
nonzero, on the standing rule that a criterion which could not be evaluated is never green.

**The two branches are reported separately and are never pooled.** The preference is a
preference: where no instance of the anchor qualifies under ADR-0010's separation and floor
rules, the ranking falls through to the pre-2026-09-02 ordering. The memory prior recalls a
CATEGORY, so an episode placed geometrically is one the prior COULD NOT have got right, and a
reached-rate over both branches charges the memory for episodes that never followed the rule
it learned. That is the same shape of error as counting an unroutable source as a miss.

**Zero anchored is not always a bug.** A class with no `vocabulary.anchor_object` row places
geometrically in every scene by construction — `glass_break` is one — so the verdict names the
run's anomaly class as the thing to read out of `provenance.txt` before calling it a defect.

"Reached" is `funnel_stage >= SOURCE_REACHED`, which is `window_report`'s definition and not a
second one. Scenes are keyed by DIRECTORY NAME, because the runner records `scene_id` as a full
`.glb` path while the sweep names the directory after the bare id: grouping on the recorded
value matches no directory at all, and the first draft of this file reported all nineteen live
scenes as barren while missing the one that truly was.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from earshot.report.artifacts import episode_paths, read_audit
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.task.smoke import episode_indices
from earshot.tools.window_report import scene_dirs

__all__ = [
    "ABLATION_ARMS",
    "ANCHOR_METRIC",
    "ScenePlacement",
    "ArmPlacement",
    "branch_of",
    "read_arm",
    "read_sweep",
    "format_arm",
    "format_report",
    "main",
]

# The metric `runner.run_episode` publishes from `SourcePlacement.at_class_anchor`.
ANCHOR_METRIC = "source_at_class_anchor"

# `ablation_sweep.sh`'s arms, in the order it runs them. Held here so the reader defaults to
# the sweep it was written for; `test_placement_report.py` asserts this list still equals the
# driver's own `ARM_NAMES`, because two lists of arm names that drift apart is a reader that
# silently skips an arm and reports a complete sweep.
ABLATION_ARMS = ("full", "no-climb", "no-cue", "scan-only", "anechoic")


@dataclass(frozen=True)
class ScenePlacement:
    """One scene of one arm, split by which branch of ADR-0022's ranking it took.

    The three counts are disjoint and sum to `n`. `n_missing` is the episodes whose audit
    does not carry `source_at_class_anchor` at all, which is a different fact from an episode
    that carried it as false, and the two must not be added together.
    """

    scene: str
    n_anchored: int
    n_geometric: int
    n_missing: int
    n_reached_anchored: int
    n_reached_geometric: int
    n_reached_missing: int

    @property
    def n(self) -> int:
        return self.n_anchored + self.n_geometric + self.n_missing


@dataclass(frozen=True)
class ArmPlacement:
    """One arm directory. `barren` is the scene directories that built no episode at all.

    A barren scene is a measurement (`mL8ThkuaVTM` places no episode in any sweep this repo
    has run) and never a failure, so it is carried and printed rather than dropped.
    """

    arm: str
    scenes: Tuple[ScenePlacement, ...]
    barren: Tuple[str, ...]

    def _total(self, field: str) -> int:
        return sum(getattr(scene, field) for scene in self.scenes)

    @property
    def n(self) -> int:
        return self._total("n")

    @property
    def n_anchored(self) -> int:
        return self._total("n_anchored")

    @property
    def n_geometric(self) -> int:
        return self._total("n_geometric")

    @property
    def n_missing(self) -> int:
        return self._total("n_missing")

    @property
    def n_reached_anchored(self) -> int:
        return self._total("n_reached_anchored")

    @property
    def n_reached_geometric(self) -> int:
        return self._total("n_reached_geometric")

    @property
    def evaluable(self) -> bool:
        """Did ANY episode in this arm record the field? An arm that recorded it nowhere
        cannot answer the question it was read to answer, and NOT_RUN is red."""
        return self.n_anchored + self.n_geometric > 0


def branch_of(audit: EpisodeAudit) -> Optional[bool]:
    """`True` anchored, `False` geometric, `None` the field is absent.

    The tri-state is the whole point. A reader that returned a bool would turn every record
    written before ADR-0022 into evidence that the placement change did not take.
    """
    value = audit.metrics.get(ANCHOR_METRIC)
    return None if value is None else bool(value)


def _reached(audit: EpisodeAudit) -> bool:
    return audit.funnel_stage >= FunnelStage.SOURCE_REACHED


def read_arm(arm_dir: str, *, arm: str) -> ArmPlacement:
    """Read one arm directory. The only function here that touches the disk.

    Scenes are keyed by DIRECTORY NAME and never by the audit's ``scene_id``. The runner
    records the scene as its full ``.glb`` path and the sweep names the directory after the
    bare id, so a reader that grouped on the recorded value cannot match a directory to the
    episodes inside it: the first draft did exactly that and reported all nineteen live
    scenes as barren while missing the one that truly was.

    ``episode_indices`` and ``episode_paths`` are called rather than copied. What
    ``window_report``'s header forbids is a second implementation of the FILENAME, and that
    knowledge stays in ``report/artifacts.py`` where it belongs.

    Every child directory is walked, not only the ones holding an ``episodes/`` directory.
    ``window_report.scene_dirs`` requires that directory, which is right for a reader of
    episodes and wrong here: a scene that built nothing may never have had one made, and
    that scene is the measurement this column exists to carry.
    """
    scenes: List[ScenePlacement] = []
    barren: List[str] = []
    root = pathlib.Path(arm_dir)
    children = sorted(root.iterdir()) if root.is_dir() else []
    for scene_dir in (child for child in children if child.is_dir()):
        row = [0, 0, 0, 0, 0, 0]
        indices = episode_indices(str(scene_dir))
        if not indices:
            barren.append(scene_dir.name)
            continue
        for index in indices:
            _agent_path, audit_path = episode_paths(str(scene_dir), index)
            audit = read_audit(audit_path)
            branch = branch_of(audit)
            column = 0 if branch is True else (1 if branch is False else 2)
            row[column] += 1
            if _reached(audit):
                row[column + 3] += 1
        scenes.append(
            ScenePlacement(
                scene=scene_dir.name,
                n_anchored=row[0],
                n_geometric=row[1],
                n_missing=row[2],
                n_reached_anchored=row[3],
                n_reached_geometric=row[4],
                n_reached_missing=row[5],
            )
        )
    return ArmPlacement(arm=arm, scenes=tuple(scenes), barren=tuple(barren))


def read_sweep(
    sweep_dir: str, *, arms: Sequence[str] = ABLATION_ARMS
) -> Tuple[ArmPlacement, ...]:
    """Every named arm under `sweep_dir`, in the given order.

    When none of the named arms exists as a subdirectory and `sweep_dir` itself holds scene
    directories, `sweep_dir` IS the arm and is read under its own basename. That is the
    single-arm invocation, and it is a rule rather than a guess: a directory either has arm
    subdirectories or it has scene subdirectories.
    """
    root = pathlib.Path(sweep_dir)
    present = [arm for arm in arms if (root / arm).is_dir()]
    if not present and scene_dirs(sweep_dir):
        return (read_arm(sweep_dir, arm=root.name),)
    return tuple(read_arm(str(root / arm), arm=arm) for arm in present)


def _rate(n_reached: int, n: int) -> str:
    """`n/a` on an empty branch, because 0 of 0 is not 0.0%."""
    if n == 0:
        return "        n/a"
    return "{:>3}/{:<3} ({:5.1f}%)".format(n_reached, n, 100.0 * n_reached / n)


_ROW = "  {:<16s} {:>5s} {:>9s} {:>10s} {:>9s}   {:>18s} {:>18s}"


def _counts_row(label, n, n_anchored, n_geometric, n_missing, anchored_rate, geom_rate):
    return _ROW.format(
        label, str(n), str(n_anchored), str(n_geometric), str(n_missing),
        anchored_rate, geom_rate,
    )


def format_arm(placement: ArmPlacement) -> List[str]:
    lines = ["=== {} ===".format(placement.arm)]
    if not placement.scenes and not placement.barren:
        lines.append("  NO EPISODES AND NO SCENE DIRECTORIES. This arm did not run.")
        return lines

    lines.append(
        _ROW.format(
            "scene", "eps", "anchored", "geometric", "MISSING",
            "reached@anchor", "reached@geometric",
        )
    )
    for scene in placement.scenes:
        lines.append(
            _counts_row(
                scene.scene, scene.n, scene.n_anchored, scene.n_geometric,
                scene.n_missing,
                _rate(scene.n_reached_anchored, scene.n_anchored),
                _rate(scene.n_reached_geometric, scene.n_geometric),
            )
        )
    for name in placement.barren:
        lines.append(
            "  {:<16s} {:>5d}   no episodes built — zero yield, measured, not a "
            "failure".format(name, 0)
        )
    lines.append(
        _counts_row(
            "TOTAL", placement.n, placement.n_anchored, placement.n_geometric,
            placement.n_missing,
            _rate(placement.n_reached_anchored, placement.n_anchored),
            _rate(placement.n_reached_geometric, placement.n_geometric),
        )
    )
    lines.extend("  " + line for line in _verdict(placement))
    return lines


def _verdict(placement: ArmPlacement) -> List[str]:
    if placement.n == 0:
        return ["NOT_RUN: every scene directory is barren, so nothing was placed at all."]
    if not placement.evaluable:
        return [
            "NOT RECORDED on all {} episode(s). ADR-0022's `{}` is absent from every "
            "audit, so this arm was rendered by a runner from BEFORE the placement "
            "change. Whatever its SR moved by, it did not move because of the "
            "anchor.".format(placement.n, ANCHOR_METRIC),
        ]
    if placement.n_missing:
        return [
            "MIXED: {} of {} episode(s) carry no `{}`. This arm directory holds records "
            "from two different runners, so the split below is over a subset and the "
            "arm is not one task.".format(placement.n_missing, placement.n, ANCHOR_METRIC),
        ]
    if placement.n_anchored == 0:
        return [
            "RECORDED AND NEVER TRUE on all {} episode(s): no source in this arm sat at "
            "its class's anchor. Read the run's anomaly class out of `provenance.txt` "
            "first — a class with no `vocabulary.anchor_object` row places geometrically "
            "in every scene by construction, and that is the design and not a "
            "defect.".format(placement.n),
        ]
    if placement.n_geometric == 0:
        return [
            "ANCHORED ON EVERY EPISODE. The fallback branch was never taken, so this arm "
            "carries no within-run control for the placement.",
        ]
    return [
        "BOTH BRANCHES PRESENT: {} anchored, {} geometric. The two rates above are the "
        "only ones a memory prior may be scored against; pooling them charges the memory "
        "for episodes that did not follow the rule it learned.".format(
            placement.n_anchored, placement.n_geometric
        ),
    ]


def format_report(placements: Sequence[ArmPlacement]) -> str:
    if not placements:
        return "no arm directories found."
    lines: List[str] = []
    for placement in placements:
        lines.extend(format_arm(placement))
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep_dir", help="a tag directory holding one directory per arm")
    parser.add_argument(
        "--arms",
        default=" ".join(ABLATION_ARMS),
        help="space-separated arm directory names, in report order",
    )
    args = parser.parse_args(argv)

    if not pathlib.Path(args.sweep_dir).is_dir():
        print("no such sweep directory: {}".format(args.sweep_dir))
        return 2
    placements = read_sweep(args.sweep_dir, arms=tuple(args.arms.split()))
    print(format_report(placements))
    # Red unless at least one arm could actually answer the question. Nothing read, and a
    # sweep whose every episode predates the field, are both NOT_RUN.
    return 0 if any(placement.evaluable for placement in placements) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
