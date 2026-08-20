"""The candidate sounding class vocabulary, and the placement mapping that is not the agent's.

`CONTEXT.md` defines the terms this module holds: the **sounding class vocabulary**, the
**sound-room mapping**, **sound-room affinity**, and the **pruned vocabulary**. ADR-0017
records the pivot; ADR-0018 records the matrix and carries the amendment that made the
anchor a ROOM.

**The mapping is placement ground truth and the agent must never read it.** An episode puts
its source at the room's object -- a flush at the toilet -- and the agent is supposed to LEARN
that association by hearing flushes in bathrooms on prior visits. Hand it `anchor_object` or
`room_of` and the unseen-and-heard cell measures this table rather than the agent's semantic
store. Same fence `sim/world.py` puts around `sourceIsVisible()`, held by
`tests/mac/test_audio_vocabulary.py` rather than by good intentions.

**The anchor is a room, and the object is only where the source sits.** The first taxonomy
graded against OBJECTS and the `clapsmoke-3` gate refuted it: `plant` scored 0.383 with 187 of
480 rows landing on `toilet`, because water sounds mean bathroom and a houseplant is not what
pouring water predicts. Rooms are the level the sounds encode. This costs no new capability --
the object stays the navigable target, and nothing here needs a room labeller, which this tree
does not have.

**The grades were NOT re-derived from the gate's recall.** Fitting the ground truth to the
measurement would make the matrix circular: the semantic store would then be scored against a
table built from the very classifier it depends on. Affinity is a judgement about what a sound
PREDICTS, made at the room level, and it is independent of what CLAP can hear. `mouse_click`
keeps a moderate grade despite scoring 0.017, and the separation gate is what cuts it.

**The candidate set is deliberately generous.** It carries weak grades on purpose so the gate
prunes rather than the author. Do not prune this tuple by hand.

numpy is not imported here and nothing renders: a table plus predicates, importable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "HM3D_GOAL_CATEGORIES",
    "ROOMS",
    "ROOM_OF_ANCHOR",
    "AFFINITY_GRADES",
    "SoundClass",
    "CANDIDATE_VOCABULARY",
    "ABSENT_CLASSES",
    "class_names",
    "prompts",
    "prompt_of",
    "by_affinity",
    "anchor_object",
    "room_of",
]

# provenance: source -- HM3D ObjectNav's goal set, the six the episode JSON carries and the
# only objects `task/dataset.place_anomaly_source` can anchor a source at. Measured across the
# val scenes at PHASE2_ABLATION_REPORT.md:2157.
HM3D_GOAL_CATEGORIES: Tuple[str, ...] = (
    "chair",
    "bed",
    "plant",
    "toilet",
    "sofa",
    "tv_monitor",
)

# The room each anchor object sits in. THE ANCHOR TAXONOMY, as of the 2026-08-20 amendment.
#
# `chair`, `sofa` and `tv_monitor` all resolve to the living room: they are the same acoustic
# scene -- a person in a shared room -- and holding them apart is what put `sofa` at 0.792 and
# `chair` at 0.683 with their mass going to each other's neighbours. HM3D homes have no
# reliable office, so a study would be a room with one class and no way to split it.
#
# `plant` maps to nothing. A houseplant has no characteristic sound, and the classes assigned
# to it were the author reasoning from what could happen near an object. Its former classes are
# now ABSENT_CLASSES: an outdoor sound heard indoors is a sound with no room, which is exactly
# what the forced-failure arm should be testing.
ROOM_OF_ANCHOR: Dict[str, str] = {
    "toilet": "bathroom",
    "bed": "bedroom",
    "sofa": "living_room",
    "tv_monitor": "living_room",
    "chair": "living_room",
}

ROOMS: Tuple[str, ...] = ("bathroom", "bedroom", "living_room")

# How strongly the sound implies ONE ROOM rather than any other. Not a confidence in the
# recording, not a CLAP score, and not derived from one.
AFFINITY_GRADES: Tuple[str, ...] = ("strong", "moderate", "weak")


@dataclass(frozen=True)
class SoundClass:
    """One row of the candidate vocabulary.

    `anchor_object` is where the source is PLACED and must be one of HM3D's six. `room_affinity`
    grades against the ROOM that object sits in, which is the level the semantic store learns at.
    The two are deliberately separate: placement is a navmesh fact, affinity is a semantic claim.
    """

    name: str
    esc50_category: str
    prompt: str
    anchor_object: str
    room_affinity: str

    def __post_init__(self) -> None:
        if self.anchor_object not in HM3D_GOAL_CATEGORIES:
            raise ValueError(
                "{!r} anchors at {!r}, which is not one of HM3D ObjectNav's six goal "
                "categories {}; a source cannot be placed at an object the episode builder "
                "cannot find".format(self.name, self.anchor_object, HM3D_GOAL_CATEGORIES)
            )
        if self.anchor_object not in ROOM_OF_ANCHOR:
            raise ValueError(
                "{!r} anchors at {!r}, which resolves to no room. An object with no room "
                "cannot carry a sound class: the semantic store learns at the room level, so "
                "there would be nothing for it to learn.".format(self.name, self.anchor_object)
            )
        if self.room_affinity not in AFFINITY_GRADES:
            raise ValueError(
                "{!r} has room_affinity {!r}, expected one of {}".format(
                    self.name, self.room_affinity, AFFINITY_GRADES
                )
            )


# Grades, at the ROOM level:
#   strong   -- the sound is made by that room's fixtures, or by the one activity it hosts.
#   moderate -- the sound belongs to that room and is uncommon elsewhere in a home.
#   weak     -- the sound happens in every room; the semantic store has nothing to learn.
CANDIDATE_VOCABULARY: Tuple[SoundClass, ...] = (
    # -- bathroom (source placed at the toilet) -----------------------------
    SoundClass("toilet_flush", "toilet_flush", "a toilet flushing", "toilet", "strong"),
    SoundClass("brushing_teeth", "brushing_teeth", "someone brushing their teeth", "toilet", "strong"),
    # A dripping tap is a bathroom or kitchen sound, and kitchen is not an HM3D category.
    SoundClass("water_drops", "water_drops", "water dripping from a tap", "toilet", "strong"),
    # Moved here from `plant`. The room taxonomy is what makes the old assignment untenable:
    # pouring water predicts a room with plumbing, and a houseplant is not one.
    SoundClass("pouring_water", "pouring_water", "water being poured", "toilet", "moderate"),
    # -- bedroom (source placed at the bed) ---------------------------------
    SoundClass("clock_alarm", "clock_alarm", "an alarm clock beeping", "bed", "strong"),
    SoundClass("snoring", "snoring", "a person snoring", "bed", "strong"),
    SoundClass("breathing", "breathing", "a person breathing slowly", "bed", "moderate"),
    SoundClass("clock_tick", "clock_tick", "a clock ticking", "bed", "moderate"),
    SoundClass("crying_baby", "crying_baby", "a baby crying", "bed", "moderate"),
    # -- living room (source placed at a sofa, a tv or a chair) -------------
    SoundClass("clapping", "clapping", "an audience clapping on television", "tv_monitor", "moderate"),
    SoundClass("laughing", "laughing", "people laughing on television", "tv_monitor", "moderate"),
    SoundClass("keyboard_typing", "keyboard_typing", "typing on a computer keyboard", "chair", "moderate"),
    SoundClass("mouse_click", "mouse_click", "clicking a computer mouse", "chair", "moderate"),
    SoundClass("vacuum_cleaner", "vacuum_cleaner", "a vacuum cleaner running", "sofa", "weak"),
    SoundClass("coughing", "coughing", "a person coughing", "sofa", "weak"),
    SoundClass("drinking_sipping", "drinking_sipping", "someone sipping a drink", "sofa", "weak"),
    SoundClass("door_wood_creaks", "door_wood_creaks", "wood creaking", "chair", "weak"),
)

# The forced-failure arm (ADR-0014: a detector ships both arms). Sounds with NO room in a
# home, supplied as audio and never placed in the prompt bank, so an open-set rule that
# accepts them is accepting anything.
#
# The last three arrived here from the candidate set when `plant` lost its room. That is a
# promotion rather than a deletion: birdsong heard indoors is precisely a sound whose source
# is not in the house, which is the hardest and most realistic negative this arm can have.
# It also strengthens the arm from five classes to eight.
ABSENT_CLASSES: Tuple[str, ...] = (
    "chainsaw",
    "helicopter",
    "airplane",
    "church_bells",
    "sea_waves",
    "chirping_birds",
    "crickets",
    "rain",
)


def class_names() -> Tuple[str, ...]:
    """Every candidate class name, in table order."""
    return tuple(entry.name for entry in CANDIDATE_VOCABULARY)


def prompts() -> Dict[str, str]:
    """The prompt bank `audio/clap.py` scores against: class name to CLAP text prompt.

    The whole of what the agent may know about the vocabulary. It carries no object and no
    room, so a controller holding this dict cannot recover the placement mapping.
    """
    return {entry.name: entry.prompt for entry in CANDIDATE_VOCABULARY}


def prompt_of(name: str) -> str:
    """The prompt for one class, raising rather than returning a plausible default."""
    bank = prompts()
    if name not in bank:
        raise KeyError(
            "{!r} is not in the candidate vocabulary; present: {}".format(
                name, ", ".join(sorted(bank))
            )
        )
    return bank[name]


def by_affinity(grade: str) -> Tuple[SoundClass, ...]:
    """Every candidate at one room-affinity grade, for the strong-versus-weak breakdown."""
    if grade not in AFFINITY_GRADES:
        raise ValueError(
            "unknown affinity {!r}, expected one of {}".format(grade, AFFINITY_GRADES)
        )
    return tuple(entry for entry in CANDIDATE_VOCABULARY if entry.room_affinity == grade)


def anchor_object(name: str) -> str:
    """ANALYST AND DATASET BUILDER ONLY -- the object a source of this class is placed at.

    **The controller must never call this.** It is placement ground truth, and the agent's
    semantic store exists to learn the association from prior visits. The fence is
    `tests/mac/test_audio_vocabulary.py`, which fails if anything under `earshot/agent/`
    reaches this module at all and holds an allowlist of call sites besides.
    """
    for entry in CANDIDATE_VOCABULARY:
        if entry.name == name:
            return entry.anchor_object
    raise KeyError(
        "{!r} is not in the candidate vocabulary; present: {}".format(
            name, ", ".join(sorted(class_names()))
        )
    )


def room_of(name: str) -> str:
    """ANALYST AND DATASET BUILDER ONLY -- the room a source of this class is placed in.

    `anchor_object` composed with `ROOM_OF_ANCHOR`, and fenced for the same reason. This is
    the level the semantic store learns at, so it is the one that must never leak.
    """
    return ROOM_OF_ANCHOR[anchor_object(name)]
