"""Build the episode plan: cross the scenes with the bank, balance it, price it.

    python -m earshot.tools.build_plan --bank runs/bank_of_record.json
    python -m earshot.tools.build_plan --n-episodes 2000        # 8000 runs, 60 hours
    python -m earshot.tools.build_plan --role verification      # a different clip block

Read-only apart from `--out`, no GPU, seconds. It reads the published ObjectNav goals through
`room_yield.scene_rooms`, so it needs the dataset on disk but never a simulator.

**"Dataset size" is ambiguous and the tool refuses to pick for you.** An episode is
`(scene, class, instance, recording)`; the four cells are conditions on the PRIOR PHASE, so
every episode is run four times. So a plan of `n` episodes is `4n` episode-runs, and "2000"
could mean either:

| distinct episodes | runs | wall clock | paired MDE |
|---|---|---|---|
| 500 | 2000 | 15 h | 5.0 pt |
| 2000 | 8000 | 60 h | 2.5 pt |

`--n-episodes` is the distinct count and both numbers are printed every time, because quoting
one while budgeting for the other is a two-and-a-half-day mistake.

**Wall clock is extrapolated from the OLD task**, 27 s an episode measured over the
anomaly-response sweeps. The sounding-window episode has never been timed. Treat the hours as
an order of magnitude, not a booking.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

from earshot.task.episode_plan import (
    MEMORY_CONDITIONS,
    balance_report,
    plan_episodes,
    supply,
)
from earshot.tools.dataset_split import ROLES, split_recordings
from earshot.tools.power import mde_between_cells, mde_paired
from earshot.tools.room_yield import scene_rooms

__all__ = ["load_bank", "main"]

# provenance: measured on the anomaly-response sweeps (365 episodes in 2h45m at 500 rays).
# The sounding-window episode has never been timed.
SECONDS_PER_EPISODE = 27.0


def load_bank(path: str) -> Dict[str, List[str]]:
    """`{room: [class]}` from a `bank_of_record.json`.

    Raises rather than falling back to the candidate vocabulary. The candidate set carries the
    weak-affinity and unresolvable classes on purpose; building a plan from it would put
    `mouse_click` (0.308 anchor recall) and `coughing` (a sound made in every room) into the
    dataset, and neither belongs in a measurement of room memory.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    by_anchor = payload.get("by_anchor")
    if not by_anchor:
        raise ValueError(
            "{} carries no 'by_anchor' map; it is not a bank of record. Generate one with "
            "`python -m earshot.tools.bank_intersect <run-a> <run-b>`.".format(path)
        )
    bank = {room: sorted(names) for room, names in by_anchor.items() if names}
    if not bank:
        raise ValueError("{} lists no class under any room".format(path))
    return bank


def _rooms_by_scene(split: str, root: str) -> Dict[str, Dict[str, int]]:
    from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene

    split_dir = find_split_dir(split, root=root)
    scenes_dir = find_scenes_dir(root=root)
    out: Dict[str, Dict[str, int]] = {}
    for label in available_scenes(split_dir):
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        rooms = scene_rooms(dataset)
        if rooms:
            out[label] = rooms
    return out


def _print_sizes(n_episodes: int) -> None:
    print("")
    print("=== what 'dataset size' means here ===")
    print("  An episode is run under all {} memory conditions, so distinct != runs.".format(
        len(MEMORY_CONDITIONS)))
    print("")
    print("  {:>10s} {:>8s} {:>10s} {:>12s} {:>12s}".format(
        "distinct", "runs", "wall", "paired MDE", "unpaired"))
    for n in sorted({200, 500, 2000, n_episodes}):
        runs = n * len(MEMORY_CONDITIONS)
        print("  {:>10d} {:>8d} {:>9.1f}h {:>11.1f}pt {:>11.1f}pt{}".format(
            n, runs, runs * SECONDS_PER_EPISODE / 3600.0,
            100 * mde_paired(n), 100 * mde_between_cells(n),
            "   <- this plan" if n == n_episodes else ""))
    print("")
    print("  Paired is available because the four cells share the SAME episodes. Quoting the")
    print("  unpaired column would understate the design by roughly 1.8x.")
    print("  Hours extrapolate 27 s/episode from the OLD task. The new episode is untimed.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build a balanced episode plan. No GPU.")
    parser.add_argument("--bank", default="runs/bank_of_record.json")
    parser.add_argument("--split", default="val")
    parser.add_argument("--root", default=".")
    parser.add_argument("--n-episodes", type=int, default=500,
                        help="DISTINCT episodes; runs are 4x this")
    parser.add_argument("--role", default=ROLES[0], choices=list(ROLES))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(None if argv is None else list(argv))

    try:
        bank = load_bank(args.bank)
    except (OSError, ValueError) as exc:
        print("FATAL: {}".format(exc), file=sys.stderr)
        return 2

    try:
        rooms_by_scene = _rooms_by_scene(args.split, args.root)
    except Exception as exc:
        print("FATAL: could not read scenes for split={!r}: {}".format(args.split, exc),
              file=sys.stderr)
        return 2
    if not rooms_by_scene:
        print("FATAL: no scene in split={!r} publishes an anchor room".format(args.split),
              file=sys.stderr)
        return 2

    block = {b.role: b for b in split_recordings()}[args.role]
    clips = [int(index) for index in block.members]

    print("")
    print("=== inputs ===")
    print("  bank      {} classes over {} rooms  ({})".format(
        sum(len(v) for v in bank.values()), len(bank), args.bank))
    for room in sorted(bank):
        print("    {:12s} {}".format(room, ", ".join(bank[room])))
    print("  scenes    {} in split={}".format(len(rooms_by_scene), args.split))
    print("  clips     {} from the {} block, indices {}..{}".format(
        len(clips), args.role, clips[0], clips[-1]))

    counts = supply(rooms_by_scene, bank, len(clips))
    print("")
    print("=== supply, before balancing ===")
    print("  {:12s} {:>8s} {:>8s} {:>9s} {:>14s}".format(
        "room", "scenes", "anchors", "classes", "combinations"))
    for room in sorted(counts["per_room"]):
        item = counts["per_room"][room]
        print("  {:12s} {:>8d} {:>8d} {:>9d} {:>14,d}".format(
            room, item["scenes"], item["anchors"], item["classes"], item["combinations"]))
    print("  {:12s} {:>8s} {:>8d} {:>9s} {:>14,d}".format(
        "TOTAL", "", counts["anchors"], "", counts["combinations"]))
    print("")
    print("  The largest room holds {:.0%} of the raw anchors. Uniform sampling would hand".format(
        counts["largest_room_share"]))
    print("  it exactly that share of the dataset, which is why the plan does not sample.")

    _print_sizes(args.n_episodes)

    try:
        specs = plan_episodes(rooms_by_scene, bank, clips, args.n_episodes)
    except ValueError as exc:
        print("")
        print("FATAL: {}".format(exc), file=sys.stderr)
        return 2
    report = balance_report(specs)

    print("")
    print("=== the plan: {} distinct episodes, {} runs ===".format(
        report["n_episodes"], report["n_episodes"] * len(MEMORY_CONDITIONS)))
    print("  by room   {}".format(dict(report["by_room"])))
    print("  by class  {}".format(dict(report["by_class"])))
    hostable = [
        count for scene, count in report["by_scene"].items()
        if scene not in report["scenes_incomplete"]
    ]
    print("  by scene  {}..{} across {} scenes that host every room".format(
        min(hostable), max(hostable), len(hostable)))
    if report["scenes_incomplete"]:
        print("  short     {} -- cannot host every room, so structurally below the rest".format(
            ", ".join(
                "{} ({})".format(scene, report["by_scene"][scene])
                for scene in report["scenes_incomplete"]
            )))
    print("  ratio     {:.3f} over complete scenes, {:.3f} over all".format(
        report["scene_ratio_complete"], report["scene_ratio"]))
    print("  clips     {} distinct".format(report["n_distinct_recordings"]))

    if args.out:
        directory = os.path.dirname(args.out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as sink:
            json.dump(
                {
                    "n_episodes": len(specs),
                    "n_runs": len(specs) * len(MEMORY_CONDITIONS),
                    "conditions": list(MEMORY_CONDITIONS),
                    "split": args.split,
                    "clip_role": args.role,
                    "clips": clips,
                    "bank": {room: list(names) for room, names in sorted(bank.items())},
                    "balance": {
                        key: value for key, value in report.items() if key != "by_scene"
                    },
                    "by_scene": dict(report["by_scene"]),
                    "episodes": [spec.as_dict() for spec in specs],
                },
                sink,
                indent=2,
                sort_keys=True,
            )
        print("")
        print("  written: {}".format(args.out))
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
