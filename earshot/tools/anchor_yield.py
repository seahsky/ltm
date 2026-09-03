"""If the sound chose the scene, how many episodes would land at its class's anchor?

    python -m earshot.tools.anchor_yield                       # every sounding class, val
    python -m earshot.tools.anchor_yield --classes "alarm toilet_flush"
    python -m earshot.tools.anchor_yield --split val --n-episodes 15 --limit 0
    python -m earshot.tools.anchor_yield --out runs/anchor_yield.json

Read-only, **no GPU and no simulator**, seconds. `task/episodes.py` is stdlib gzip and json
and `task/dataset.py` places sources with pure arithmetic (ADR-0013 keeps habitat-sim in one
file), so this runs the REAL builder over the real published goals.

**Why it exists.** `abl-2` measured what `placement_report` then read back: with
`--anomaly-class alarm` fixed for every episode, only 134 of 282 sources sat at the class's
anchor. `alarm` anchors at `bed`, and three of nineteen scenes never placed one there. The
other 148 episodes are ones a memory prior that recalls a CATEGORY could not have got right
whatever it remembered, so ADR-0018's heard axis arrives pre-diluted by half before a single
run. Widening the class per scene is the obvious answer and it is worth a night only if the
anchored fraction actually rises. This counts it first.

**It runs the builder rather than modelling it.** `build_anomaly_episodes` takes only the
dataset and its flags — no seed, no simulator — so a build here is byte-for-byte the build a
sweep would do, and the answer is a prediction only in the sense that the run has not happened
yet. The check that this is true is printed: `alarm` over the val split at 15 episodes must
reproduce `abl-2`'s measured 134 of 282, and a disagreement is a defect here and never a
second opinion about that run.

**A scene that hosts no anchor is a fact about HM3D, not a failure.** `mL8ThkuaVTM` places no
episode at all, in any sweep this repo has run, and `EmptyDatasetError` carries its whole build
for exactly this reason. Nothing here raises on one. The counts are the finding.

**A class with no `vocabulary.anchor_object` row is reported, not hidden.** `glass_break` is
one, and it anchors nowhere by construction: its 0% is the design and reads differently from a
class whose anchor exists and never qualified.

**This bounds the CEILING and not the cell size.** It says how many episodes COULD carry a
learnable association. Whether the agent then reaches the source is `ablation_sweep.sh`'s
question and needs a GPU and a night.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from earshot.audio.clips import ANOMALY_CLASSES, SOUNDING_CLASSES
from earshot.task.dataset import (
    DatasetBuild,
    EmptyDatasetError,
    PlacementError,
    build_anomaly_episodes,
)
from earshot.task.episodes import (
    EpisodeDataError,
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    load_scene,
)
from earshot.task.prior_build import anchor_of_run_class

__all__ = [
    "CellYield",
    "ClassYield",
    "anchored_in_build",
    "cell_yield",
    "fold_by_class",
    "best_class_per_scene",
    "format_report",
    "main",
]

# What `ablation_sweep.sh` runs at, so the reproduction check compares like with like.
SWEEP_N_EPISODES = 15

# `abl-2`, measured: `--anomaly-class alarm`, val split, 15 episodes per scene. Held here so
# the reproduction is asserted rather than eyeballed. A build that disagrees with a finished
# run is this module being wrong.
ABL2_ALARM_ANCHORED = 134
ABL2_ALARM_BUILT = 282


@dataclass(frozen=True)
class CellYield:
    """One (scene, class) cell: what the real builder produced for it.

    `n_anchored` counts placements whose `at_class_anchor` is true. `n_built` is every
    episode the scene could express, anchored or not, because ADR-0022's preference falls
    through to geometric placement rather than refusing — so the two are a fraction and never
    a pass/fail.
    """

    scene: str
    anomaly_class: str
    anchor_category: Optional[str]
    n_built: int
    n_anchored: int
    n_skipped: int
    error: Optional[str] = None

    @property
    def rate(self) -> Optional[float]:
        """`None` on a scene that built nothing. 0 of 0 is not 0.0."""
        if self.n_built == 0:
            return None
        return self.n_anchored / float(self.n_built)


@dataclass(frozen=True)
class ClassYield:
    """One class folded over every scene."""

    anomaly_class: str
    anchor_category: Optional[str]
    n_built: int
    n_anchored: int
    n_scenes_with_any: int
    n_scenes_built: int

    @property
    def rate(self) -> Optional[float]:
        if self.n_built == 0:
            return None
        return self.n_anchored / float(self.n_built)


def anchored_in_build(build: DatasetBuild) -> int:
    """How many of a build's episodes sat at the class's anchor."""
    return sum(1 for episode in build.episodes if episode.source.at_class_anchor)


def cell_yield(
    dataset: object,
    *,
    scene: str,
    anomaly_class: str,
    n_episodes: int,
) -> CellYield:
    """Build one (scene, class) cell with the REAL builder and count the anchored ones.

    `EmptyDatasetError` is caught and folded into a zero-yield cell rather than propagated:
    a scene that can express nothing is the single most informative point a denominator has,
    and the error already carries the whole build so nothing is lost by not raising.
    """
    anchor = anchor_of_run_class(anomaly_class)
    try:
        build = build_anomaly_episodes(
            dataset,
            anomaly_class=anomaly_class,
            anchor_category=anchor,
            n_episodes=n_episodes,
        )
    except EmptyDatasetError as empty:
        return CellYield(
            scene=scene,
            anomaly_class=anomaly_class,
            anchor_category=anchor,
            n_built=0,
            n_anchored=0,
            n_skipped=len(empty.build.skipped),
        )
    except (PlacementError, ValueError) as exc:
        # Recorded, never dropped. A cell that would not build is not a cell with no anchor.
        return CellYield(
            scene=scene,
            anomaly_class=anomaly_class,
            anchor_category=anchor,
            n_built=0,
            n_anchored=0,
            n_skipped=0,
            error=str(exc),
        )
    return CellYield(
        scene=scene,
        anomaly_class=anomaly_class,
        anchor_category=anchor,
        n_built=len(build.episodes),
        n_anchored=anchored_in_build(build),
        n_skipped=len(build.skipped),
    )


def fold_by_class(cells: Sequence[CellYield]) -> Tuple[ClassYield, ...]:
    """Every cell folded onto its class, in the order the classes first appear."""
    order: List[str] = []
    grouped: Dict[str, List[CellYield]] = {}
    for cell in cells:
        if cell.anomaly_class not in grouped:
            order.append(cell.anomaly_class)
            grouped[cell.anomaly_class] = []
        grouped[cell.anomaly_class].append(cell)
    return tuple(
        ClassYield(
            anomaly_class=name,
            anchor_category=grouped[name][0].anchor_category,
            n_built=sum(cell.n_built for cell in grouped[name]),
            n_anchored=sum(cell.n_anchored for cell in grouped[name]),
            n_scenes_with_any=sum(1 for cell in grouped[name] if cell.n_anchored),
            n_scenes_built=sum(1 for cell in grouped[name] if cell.n_built),
        )
        for name in order
    )


def best_class_per_scene(cells: Sequence[CellYield]) -> Dict[str, CellYield]:
    """The class each scene anchors most episodes of.

    This is the matrix's real design question: a sweep is free to pick the class per SCENE, so
    the ceiling on a learnable association is the sum of these and not any single class's
    total. Ties break on class name so the answer is stable between runs.
    """
    best: Dict[str, CellYield] = {}
    for cell in cells:
        current = best.get(cell.scene)
        if current is None or (cell.n_anchored, cell.anomaly_class) > (
            current.n_anchored, current.anomaly_class
        ):
            best[cell.scene] = cell
    return best


def _pct(n: int, total: int) -> str:
    return "   n/a" if total == 0 else "{:5.1f}%".format(100.0 * n / total)


def format_report(
    cells: Sequence[CellYield],
    *,
    scenes: Sequence[str],
    n_episodes: int,
    split: str,
) -> str:
    lines: List[str] = []
    by_class = fold_by_class(cells)
    best = best_class_per_scene(cells)

    lines.append("")
    lines.append("=== anchored episodes per class ({}, {} scene(s), {} ep/scene) ===".format(
        split, len(scenes), n_episodes
    ))
    lines.append("  {:<18s} {:<12s} {:>7s} {:>9s} {:>8s}   {}".format(
        "class", "anchor", "built", "anchored", "rate", "scenes anchoring any"
    ))
    for entry in sorted(by_class, key=lambda e: (-(e.rate or 0.0), e.anomaly_class)):
        lines.append("  {:<18s} {:<12s} {:>7d} {:>9d} {:>8s}   {} of {}".format(
            entry.anomaly_class,
            entry.anchor_category or "NONE",
            entry.n_built,
            entry.n_anchored,
            _pct(entry.n_anchored, entry.n_built),
            entry.n_scenes_with_any,
            entry.n_scenes_built,
        ))
    lines.append("")
    lines.append("  A class with anchor NONE has no `vocabulary.anchor_object` row and cannot")
    lines.append("  anchor anywhere. Its 0% is the design, not a scene that failed to host it.")

    lines.append("")
    lines.append("=== if the sweep picked the best class PER SCENE ===")
    lines.append("  {:<18s} {:<18s} {:<12s} {:>7s} {:>9s} {:>8s}".format(
        "scene", "class", "anchor", "built", "anchored", "rate"
    ))
    total_built = 0
    total_anchored = 0
    for scene in scenes:
        cell = best.get(scene)
        if cell is None:
            continue
        total_built += cell.n_built
        total_anchored += cell.n_anchored
        if cell.n_built == 0:
            lines.append("  {:<18s} {}".format(
                scene, "no episode built by ANY class — zero yield, measured"
            ))
            continue
        lines.append("  {:<18s} {:<18s} {:<12s} {:>7d} {:>9d} {:>8s}".format(
            scene,
            cell.anomaly_class,
            cell.anchor_category or "NONE",
            cell.n_built,
            cell.n_anchored,
            _pct(cell.n_anchored, cell.n_built),
        ))
    lines.append("  {:<18s} {:<18s} {:<12s} {:>7d} {:>9d} {:>8s}".format(
        "TOTAL", "", "", total_built, total_anchored, _pct(total_anchored, total_built)
    ))

    lines.extend(_reproduction_lines(by_class, n_episodes, split))

    lines.append("")
    lines.append("  THIS IS A CEILING, NOT A RESULT. It bounds how many episodes COULD carry a")
    lines.append("  learnable class-to-category association. Whether the agent then reaches the")
    lines.append("  source is `ablation_sweep.sh`'s question and costs a GPU and a night.")
    lines.append("")
    return "\n".join(lines)


def _reproduction_lines(
    by_class: Sequence[ClassYield], n_episodes: int, split: str
) -> List[str]:
    """The check that this module is building what the sweep built.

    Only meaningful at the sweep's own settings; on anything else it says so rather than
    comparing two different questions and calling the difference a defect.
    """
    alarm = next((entry for entry in by_class if entry.anomaly_class == "alarm"), None)
    if alarm is None:
        return []
    lines = ["", "=== reproduction check against abl-2 (measured) ==="]
    if split != "val" or n_episodes != SWEEP_N_EPISODES:
        lines.append(
            "  SKIPPED: abl-2 ran split=val at {} ep/scene; this ran split={} at {}. "
            "Not the same question.".format(SWEEP_N_EPISODES, split, n_episodes)
        )
        return lines
    lines.append("  abl-2 measured   alarm anchored {} of {} episodes".format(
        ABL2_ALARM_ANCHORED, ABL2_ALARM_BUILT
    ))
    lines.append("  this build gives alarm anchored {} of {} episodes".format(
        alarm.n_anchored, alarm.n_built
    ))
    if (alarm.n_anchored, alarm.n_built) == (ABL2_ALARM_ANCHORED, ABL2_ALARM_BUILT):
        lines.append("  AGREES. The builder here is the builder that ran.")
    else:
        lines.append(
            "  DISAGREES. A build with no seed and no simulator should reproduce a finished "
            "run exactly, so this is a defect in `anchor_yield` or a scene set that differs "
            "from abl-2's — it is NOT a second opinion about abl-2."
        )
    return lines


def _resolve_classes(raw: Optional[str]) -> Tuple[str, ...]:
    """The classes to build. Default is every sounding class plus the carried three.

    `ANOMALY_CLASSES` is included even though `SOUNDING_CLASSES` is the matrix's set, because
    `alarm` is what every run on disk used and dropping it would drop the reproduction check.
    """
    if raw:
        return tuple(raw.split())
    seen: List[str] = []
    for name in tuple(SOUNDING_CLASSES) + tuple(ANOMALY_CLASSES):
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count how many episodes would land at each sound class's anchor. No GPU."
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--root", default=".")
    parser.add_argument("--classes", default=None, help="space-separated; default is all")
    parser.add_argument("--n-episodes", type=int, default=SWEEP_N_EPISODES)
    parser.add_argument("--limit", type=int, default=0, help="cap the scene count; 0 is all")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(None if argv is None else list(argv))

    if args.n_episodes < 1:
        print("FATAL: --n-episodes must be >= 1", file=sys.stderr)
        return 2

    split_dir = find_split_dir(args.split, root=args.root)
    scenes_dir = find_scenes_dir(root=args.root)

    # The same discovery `ablation_sweep.sh` makes, and for the same reason: a content file
    # whose mesh is not on this box fails at load, which is a different fact from a scene
    # that cannot pose the task.
    scenes: List[str] = []
    datasets: Dict[str, object] = {}
    skipped: List[Tuple[str, str]] = []
    for label in available_scenes(split_dir):
        try:
            dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        except (EpisodeDataError, OSError, ValueError) as exc:
            skipped.append((label, str(exc)))
            continue
        if not os.path.exists(dataset.scene_path):
            skipped.append((label, "no mesh on this box"))
            continue
        scenes.append(label)
        datasets[label] = dataset
    if args.limit > 0:
        scenes = scenes[: args.limit]

    if not scenes:
        print("FATAL: no scene with a mesh on this box under {}".format(split_dir),
              file=sys.stderr)
        return 2

    classes = _resolve_classes(args.classes)
    cells: List[CellYield] = []
    for name in classes:
        for scene in scenes:
            cells.append(
                cell_yield(
                    datasets[scene],
                    scene=scene,
                    anomaly_class=name,
                    n_episodes=args.n_episodes,
                )
            )

    print(format_report(cells, scenes=scenes, n_episodes=args.n_episodes, split=args.split))
    if skipped:
        print("  SCENES NOT COUNTED ({}) — not scenes with zero anchors:".format(len(skipped)))
        for label, reason in skipped:
            print("    {:24s} {}".format(label, reason))
        print("")

    errors = [cell for cell in cells if cell.error]
    if errors:
        print("  CELLS THAT RAISED ({}):".format(len(errors)))
        for cell in errors[:10]:
            print("    {:18s} {:18s} {}".format(cell.scene, cell.anomaly_class, cell.error))
        print("")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as sink:
            json.dump(
                {
                    "split": args.split,
                    "n_episodes": args.n_episodes,
                    "scenes": scenes,
                    "cells": [
                        {
                            "scene": cell.scene,
                            "anomaly_class": cell.anomaly_class,
                            "anchor_category": cell.anchor_category,
                            "n_built": cell.n_built,
                            "n_anchored": cell.n_anchored,
                            "n_skipped": cell.n_skipped,
                            "error": cell.error,
                        }
                        for cell in cells
                    ],
                    "skipped_scenes": [{"scene": s, "reason": r} for s, r in skipped],
                },
                sink,
                indent=2,
                sort_keys=True,
            )
        print("  written: {}".format(args.out))

    # Red when NOTHING anchored anywhere. That is either a broken lookup or a scene set with
    # no anchor instances at all, and both are findings that must not exit 0.
    return 0 if any(cell.n_anchored for cell in cells) else 1


if __name__ == "__main__":
    raise SystemExit(main())
