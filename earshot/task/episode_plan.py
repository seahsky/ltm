"""Crossing scenes with sounds into an episode plan: count the supply, then balance it.

The combination space is (scene, room, anchor instance, sound class, recording, start pose).
It is enormous, and that is the trap: **the supply is wildly unbalanced, so expanding it
naively builds a dataset that looks large and carries very little.**

HM3D val publishes 40 bathroom anchors, 68 bedroom and **296 living-room**. Sample instances
uniformly and 73% of the dataset is living room. One scene (`cvZr5TUy5C5`) holds 37 anchors
against `mL8ThkuaVTM`'s 8, so uniform sampling also hands one house four times the weight of
another. Neither imbalance is visible in a success rate; both would move it.

**So the plan is built by balance, not by enumeration.** Rooms get equal shares, classes get
equal shares inside their room, and scenes are filled instance-major: every scene contributes
its first anchor before any scene contributes a second. That is the fix for 296 against 40.

**The structural finding, which is why this module exists at all.** The four cells of the
matrix are four MEMORY conditions over the same task. An episode is defined by (scene, class,
instance, recording); whether it lands in seen-heard or unseen-not-heard is a property of the
PRIOR PHASE, not of the episode. So one episode set can be run under all four conditions, and
then the cells are PAIRED on identical episodes rather than being four independent samples.

At the pre-registered 800 episode-runs that is 200 distinct episodes crossed with 4 conditions,
and the comparison becomes paired: MDE **8.0 points** against the unpaired **14.0**, for exactly
the same GPU time. `tools/power.py` computes both; this module only arranges for the pairing to
be possible.

Pure: no simulator, no numpy, no randomness. The plan is a function of its inputs, so two
callers a month apart get the same episodes.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "MEMORY_CONDITIONS",
    "EpisodeSpec",
    "anchor_slots",
    "balanced_slots",
    "supply",
    "plan_episodes",
    "balance_report",
]

# The four cells, named as what they are: conditions on the prior phase, not on the episode.
# `seen` means the prior pass toured this scene; `heard` means it sounded this CLASS somewhere,
# which the scene-agnostic semantic store makes possible in a scene the agent never entered.
MEMORY_CONDITIONS: Tuple[str, str, str, str] = (
    "seen_heard",
    "seen_unheard",
    "unseen_heard",
    "unseen_unheard",
)


@dataclass(frozen=True)
class EpisodeSpec:
    """One episode, independent of which memory condition it is run under.

    `instance` indexes the anchor objects of `room` within `scene`, so two episodes at the same
    scene and class but different instances are different rooms of the same house.
    """

    index: int
    scene: str
    room: str
    sound_class: str
    instance: int
    recording: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "index": self.index,
            "scene": self.scene,
            "room": self.room,
            "sound_class": self.sound_class,
            "instance": self.instance,
            "recording": self.recording,
        }


def anchor_slots(rooms_by_scene: Mapping[str, Mapping[str, int]], room: str) -> List[Tuple[str, int]]:
    """`(scene, instance)` for one room, ordered INSTANCE-MAJOR.

    Every scene offers its first anchor before any scene offers its second, so a house with 28
    living-room anchors does not drown one with 3. Scene-major order would do exactly that,
    and the resulting dataset would be a study of `cvZr5TUy5C5`.
    """
    counts = {
        scene: int(rooms.get(room, 0))
        for scene, rooms in rooms_by_scene.items()
        if int(rooms.get(room, 0)) > 0
    }
    if not counts:
        return []
    slots: List[Tuple[str, int]] = []
    for instance in range(max(counts.values())):
        for scene in sorted(counts):
            if instance < counts[scene]:
                slots.append((scene, instance))
    return slots


def supply(
    rooms_by_scene: Mapping[str, Mapping[str, int]],
    classes_by_room: Mapping[str, Sequence[str]],
    n_recordings: int,
) -> Dict[str, object]:
    """How many distinct episodes the crossing admits, and how lopsided it is.

    Reported per room as well as in total, because the total is the number that flatters and
    the per-room split is the number that decides whether the plan can be balanced.
    """
    if n_recordings < 1:
        raise ValueError("n_recordings must be at least 1, got {}".format(n_recordings))
    per_room: Dict[str, Dict[str, int]] = {}
    for room in sorted(classes_by_room):
        slots = anchor_slots(rooms_by_scene, room)
        n_classes = len(classes_by_room[room])
        per_room[room] = {
            "scenes": len({scene for scene, _i in slots}),
            "anchors": len(slots),
            "classes": n_classes,
            "combinations": len(slots) * n_classes * n_recordings,
        }
    total = sum(item["combinations"] for item in per_room.values())
    anchors = sum(item["anchors"] for item in per_room.values())
    biggest = max(per_room.values(), key=lambda item: item["anchors"])["anchors"] if per_room else 0
    return {
        "per_room": per_room,
        "combinations": total,
        "anchors": anchors,
        # The headline imbalance: what fraction of raw anchors the largest room holds. Uniform
        # sampling over anchors would hand it exactly this share of the dataset.
        "largest_room_share": (biggest / anchors) if anchors else 0.0,
    }


def balanced_slots(
    rooms_by_scene: Mapping[str, Mapping[str, int]], room: str, n: int, offset: int = 0
) -> List[Tuple[str, int]]:
    """`n` `(scene, instance)` draws with EQUAL weight per scene, rotating instances.

    `anchor_slots` enumerates the supply and is proportional to it: a house with five
    bathrooms contributes five slots against another's one, so cycling it hands the big house
    five times the episodes. Instance-major ordering fixes only the partial final lap.

    This draws scene-major instead: episode k goes to `scenes[(k + offset) % n_scenes]`, and its
    instance is the lap number modulo that scene's own anchor count. Every scene gets the same
    number of episodes to within one, and a scene with several anchors rotates through them
    rather than being sampled more often.

    `offset` moves where the remainder lands. A room's share rarely divides its scene count, so
    a few scenes get one extra episode; without an offset those are the same alphabetically-first
    scenes in every room, and three rooms stack their extras onto the same houses. Measured on
    HM3D val that took the busiest-to-quietest ratio to 1.71; offsetting per room brings it down.
    """
    if n < 0:
        raise ValueError("n must not be negative, got {}".format(n))
    counts = {
        scene: int(rooms.get(room, 0))
        for scene, rooms in rooms_by_scene.items()
        if int(rooms.get(room, 0)) > 0
    }
    if not counts:
        raise ValueError(
            "room {!r} has no anchor instance in any scene, so its share of the plan "
            "cannot be filled".format(room)
        )
    scenes = sorted(counts)
    draws: List[Tuple[str, int]] = []
    for k in range(n):
        scene = scenes[(k + int(offset)) % len(scenes)]
        draws.append((scene, (k // len(scenes)) % counts[scene]))
    return draws


def _shares(total: int, parts: int) -> List[int]:
    """`total` split into `parts` near-equal integers, largest first, summing exactly."""
    if parts < 1:
        raise ValueError("parts must be at least 1, got {}".format(parts))
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def plan_episodes(
    rooms_by_scene: Mapping[str, Mapping[str, int]],
    classes_by_room: Mapping[str, Sequence[str]],
    recordings: Sequence[int],
    n_episodes: int,
) -> Tuple[EpisodeSpec, ...]:
    """A balanced, deterministic episode set. Run it under all four memory conditions.

    Balance is imposed top-down: rooms take equal shares of `n_episodes`, classes take equal
    shares of their room's, and each class draws anchors instance-major and recordings in a
    cycle. Nothing is sampled, so the plan is a pure function of its inputs.

    Raises when a room has no anchors or a class list is empty. A cell quietly missing a room
    is the failure this whole balancing exercise exists to prevent, so it is not a warning.
    """
    if n_episodes < 1:
        raise ValueError("n_episodes must be at least 1, got {}".format(n_episodes))
    if not recordings:
        raise ValueError("no recordings given; an episode needs audio to render")
    rooms = sorted(classes_by_room)
    if not rooms:
        raise ValueError("no rooms; nothing to plan")

    specs: List[EpisodeSpec] = []
    for room_index, (room, room_share) in enumerate(zip(rooms, _shares(n_episodes, len(rooms)))):
        names = sorted(classes_by_room[room])
        if not names:
            raise ValueError("room {!r} carries no sound class".format(room))
        # Drawn for the ROOM's whole share, then sliced across its classes, so scene balance
        # holds over the room rather than being re-started (and re-biased) per class.
        # A stride of 7 rather than 1: with three rooms and ~20 scenes, consecutive offsets
        # would still overlap the remainder windows. 7 is coprime to 19 and to 20.
        slot_cycle = iter(
            balanced_slots(rooms_by_scene, room, room_share, offset=7 * room_index)
        )
        clip_cycle = itertools.cycle(sorted(recordings))
        for name, class_share in zip(names, _shares(room_share, len(names))):
            for _ in range(class_share):
                scene, instance = next(slot_cycle)
                specs.append(
                    EpisodeSpec(
                        index=len(specs),
                        scene=scene,
                        room=room,
                        sound_class=name,
                        instance=instance,
                        recording=next(clip_cycle),
                    )
                )
    return tuple(specs)


def balance_report(specs: Sequence[EpisodeSpec]) -> Dict[str, object]:
    """Counts by room, class and scene, and TWO scene ratios, because one of them lies.

    `scene_ratio` over every scene is dominated by scenes that structurally cannot host every
    room: `QaLdnwvtxbs` publishes no toilet, so it can never take a bathroom episode and sits
    a third below the rest whatever the planner does. Reporting only that number makes a
    well-balanced plan look worse than a badly-balanced one, which is how it was nearly used
    to reject the fix that improved the spread from 9-12 to 10-11.

    `scene_ratio_complete` is the one that measures the PLANNER: it covers only the scenes that
    appear with every room, so it moves when balancing changes and stays put when the dataset
    is merely lopsided. Both are reported, and `scenes_incomplete` names the difference.
    """
    if not specs:
        raise ValueError("no episodes to report on; an empty plan is NOT_RUN")
    by_room: Dict[str, int] = {}
    by_class: Dict[str, int] = {}
    by_scene: Dict[str, int] = {}
    rooms_seen: Dict[str, set] = {}
    for spec in specs:
        by_room[spec.room] = by_room.get(spec.room, 0) + 1
        by_class[spec.sound_class] = by_class.get(spec.sound_class, 0) + 1
        by_scene[spec.scene] = by_scene.get(spec.scene, 0) + 1
        rooms_seen.setdefault(spec.scene, set()).add(spec.room)

    n_rooms = len(by_room)
    complete = sorted(scene for scene, rooms in rooms_seen.items() if len(rooms) == n_rooms)
    incomplete = sorted(set(by_scene) - set(complete))

    def ratio(scenes):
        counts = sorted(by_scene[scene] for scene in scenes)
        if not counts or not counts[0]:
            return float("inf")
        return counts[-1] / counts[0]

    return {
        "n_episodes": len(specs),
        "by_room": dict(sorted(by_room.items())),
        "by_class": dict(sorted(by_class.items())),
        "by_scene": dict(sorted(by_scene.items())),
        "scene_ratio": ratio(by_scene),
        "scene_ratio_complete": ratio(complete) if complete else float("inf"),
        "scenes_incomplete": incomplete,
        "n_distinct_recordings": len({spec.recording for spec in specs}),
    }
