"""How many HM3D scenes can host each anchor room, and therefore how big a cell can be.

    python -m earshot.tools.room_yield              # the val split
    python -m earshot.tools.room_yield --split train

Read-only, **no GPU and no simulator**. `task/episodes.py` is stdlib gzip and json (ADR-0013
keeps habitat-sim in one file), so this reads the published ObjectNav goals straight off disk in
seconds.

**Why it exists.** ADR-0018's generalization matrix needs the seen/unseen axis over SCENES and
the heard/not-heard axis over CLASSES, and a class lives in a room. A scene that publishes no
`toilet` goal cannot host a bathroom episode at all. So the real cell sizes are not "365 divided
by four": they are bounded by how many scenes can host each room, and that has never been
counted.

The power question ADR-0018 leaves open is being argued at roughly 90 episodes a cell against a
measured 16.2% flip rate. Arguing it without this number is arguing about a denominator nobody
has looked up. `yield_sweep.sh` asks the same shape of question about sources and needs a GPU
and hours; this needs neither, so there is no reason not to have run it first.

**A scene missing a room is a fact about HM3D, not a defect.** Nothing here fails on one. The
output is counts, and the counts are the finding.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from earshot.audio.vocabulary import CANDIDATE_VOCABULARY, ROOMS, ROOM_OF_ANCHOR
from earshot.task.dataset import goal_table
from earshot.task.episodes import (
    EpisodeDataError,
    available_scenes,
    find_scenes_dir,
    find_split_dir,
    load_scene,
)

__all__ = ["scene_rooms", "summarise_yield", "main"]


def scene_rooms(dataset: object) -> Dict[str, int]:
    """`{room: instance count}` for one scene, from its own published goals.

    Counts INSTANCES rather than categories, because the living room is reachable through
    `sofa`, `tv_monitor` or `chair` and a scene with all three is not three living rooms. The
    room key collapses them; the count says how much furniture backs it.
    """
    rooms: Dict[str, int] = {}
    for category, goals in goal_table(dataset).items():  # type: ignore[arg-type]
        room = ROOM_OF_ANCHOR.get(category)
        if room is None:
            continue
        usable = [goal for goal in goals if goal.view_points]
        if usable:
            rooms[room] = rooms.get(room, 0) + len(usable)
    return rooms


def summarise_yield(
    per_scene: Mapping[str, Mapping[str, int]],
) -> Dict[str, object]:
    """Fold the per-scene table into the numbers that size a cell.

    `scenes_per_room` is the ceiling on the seen row for that room: a scene absent here can
    never appear in it. `scenes_with_all` is the ceiling on a design that wants every scene to
    offer every room, which is the cleanest split but also the most expensive.
    """
    scenes_per_room = {room: 0 for room in ROOMS}
    for rooms in per_scene.values():
        for room in rooms:
            scenes_per_room[room] += 1
    complete = sorted(
        label for label, rooms in per_scene.items() if set(rooms) == set(ROOMS)
    )
    histogram = collections.Counter(len(rooms) for rooms in per_scene.values())
    return {
        "n_scenes": len(per_scene),
        "scenes_per_room": scenes_per_room,
        "scenes_with_all_rooms": complete,
        "rooms_per_scene_histogram": {str(k): v for k, v in sorted(histogram.items())},
    }


def _classes_per_room() -> Dict[str, int]:
    counts: Dict[str, int] = {room: 0 for room in ROOMS}
    for entry in CANDIDATE_VOCABULARY:
        room = ROOM_OF_ANCHOR.get(entry.anchor_object)
        if room is not None and entry.room_affinity in ("strong", "moderate"):
            counts[room] += 1
    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count which HM3D scenes can host which anchor rooms. No GPU."
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(None if argv is None else list(argv))

    split_dir = find_split_dir(args.split, root=args.root)
    scenes_dir = find_scenes_dir(root=args.root)
    labels = list(available_scenes(split_dir))
    if not labels:
        print("FATAL: no scenes in {}".format(split_dir), file=sys.stderr)
        return 2

    per_scene: Dict[str, Dict[str, int]] = {}
    skipped: List[Tuple[str, str]] = []
    for label in labels:
        try:
            dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        except (EpisodeDataError, OSError, ValueError) as exc:
            # Recorded, never dropped. A scene that would not load is not a scene with no
            # rooms, and folding the two together is how a denominator quietly shrinks.
            skipped.append((label, str(exc)))
            continue
        if not os.path.exists(dataset.scene_path):
            skipped.append((label, "no mesh on this box"))
            continue
        per_scene[label] = scene_rooms(dataset)

    print("")
    print("=== anchor rooms per scene ({}) ===".format(args.split))
    print("  {:24s} {:10s} {:10s} {:12s}  rooms".format("scene", "bathroom", "bedroom", "living"))
    for label in sorted(per_scene):
        rooms = per_scene[label]
        print(
            "  {:24s} {:<10d} {:<10d} {:<12d}  {}/{}".format(
                label,
                rooms.get("bathroom", 0),
                rooms.get("bedroom", 0),
                rooms.get("living_room", 0),
                len(rooms),
                len(ROOMS),
            )
        )
    if skipped:
        print("")
        print("  SCENES NOT COUNTED ({}) -- these are not scenes with zero rooms:".format(len(skipped)))
        for label, reason in skipped:
            print("    {:24s} {}".format(label, reason))

    summary = summarise_yield(per_scene)
    scenes_per_room = summary["scenes_per_room"]
    classes = _classes_per_room()

    print("")
    print("=== the ceiling on each cell ===")
    print("  {:12s} {:>8s} {:>10s}  {}".format("room", "scenes", "classes", "note"))
    for room in ROOMS:
        n_scenes = scenes_per_room[room]
        note = ""
        if n_scenes < len(per_scene):
            note = "{} scene(s) cannot host it".format(len(per_scene) - n_scenes)
        print("  {:12s} {:>8d} {:>10d}  {}".format(room, n_scenes, classes[room], note))
    print("")
    print(
        "  scenes offering all {} rooms: {} of {}".format(
            len(ROOMS), len(summary["scenes_with_all_rooms"]), summary["n_scenes"]
        )
    )
    print("  rooms per scene: {}".format(summary["rooms_per_scene_histogram"]))

    print("")
    print("  READ THIS AS A CEILING, NOT A CELL SIZE. It bounds how many scenes a room's")
    print("  episodes can be drawn from; how many EPISODES each scene yields is a separate")
    print("  question that needs the source placement rules and a GPU (yield_sweep.sh).")
    print("  The seen/unseen split halves the scene count again, and a room hosted by few")
    print("  scenes is the one that will run out first.")
    print("")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as sink:
            json.dump(
                {
                    "split": args.split,
                    "per_scene": per_scene,
                    "skipped": [{"scene": s, "reason": r} for s, r in skipped],
                    "classes_per_room": classes,
                    **summary,
                },
                sink,
                indent=2,
                sort_keys=True,
            )
        print("  written: {}".format(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
