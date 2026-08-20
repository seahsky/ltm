"""The candidate sounding class vocabulary, and the placement mapping that is not the agent's.

`CONTEXT.md` defines four terms this module holds: the **sounding class vocabulary**, the
**sound-object mapping**, **sound-object affinity**, and the **pruned vocabulary**. The
grilling session of 2026-08-20 settled all four, and ADR-0017 records why.

**The mapping is placement ground truth and the agent must never read it.** An episode
puts its source at `anchor_object`'s view point because a flush comes from a toilet; the
agent is supposed to LEARN that association by hearing flushes near toilets on prior
visits. Hand it `anchor_object` and the unseen-and-heard cell of the matrix measures this
table rather than the agent's semantic store, and the number is worthless. This is the
same fence `sim/world.py` puts around `sourceIsVisible()`, and it is held by
`tests/mac/test_vocabulary_fence.py` rather than by good intentions.

**Affinity binds harder than CLAP accuracy.** A class CLAP separates perfectly is still
disqualified when its affinity is flat: coughing and laughing happen on every one of
HM3D's six goal categories, so a vocabulary carrying them at face value asks the semantic
store to learn noise, and it will correctly fail to. The grades here are the author's
judgement, declared rather than hidden, so the analysis can ask the obvious question --
did the strong pairs transfer better than the weak ones -- instead of assuming they did.

**The candidate set is deliberately generous.** It carries the weak anchors on purpose:
the separation gate cuts what CLAP cannot tell apart through reverb, so the surviving
vocabulary is a measured artefact rather than a pre-registered guess. Do not prune this
tuple by hand -- prune it with `audio/separation.py` against a gate run.

numpy is not imported here and nothing in this module renders: it is a table plus the
predicates over it, so it stays importable anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

__all__ = [
    "HM3D_GOAL_CATEGORIES",
    "AFFINITY_GRADES",
    "SoundClass",
    "CANDIDATE_VOCABULARY",
    "ABSENT_CLASSES",
    "class_names",
    "prompts",
    "prompt_of",
    "by_affinity",
    "anchor_object",
    "ROOM_OF_ANCHOR",
    "room_of",
]

# provenance: source -- HM3D ObjectNav's goal set, the six the ObjectNav episode JSON
# actually carries and the only objects `task/dataset.place_anomaly_source` can anchor a
# source at. Measured across the val scenes at PHASE2_ABLATION_REPORT.md:2157 (97
# achievable scene-by-category cells). Six is thin, and ADR-0018 owns that it is: it caps
# the semantic transfer at a 6-way mapping, which is the price of ADR-0007 keeping MP3D
# and its 21 categories out of scope.
HM3D_GOAL_CATEGORIES: Tuple[str, ...] = (
    "chair",
    "bed",
    "plant",
    "toilet",
    "sofa",
    "tv_monitor",
)

# How strongly the sound implies ONE of the six rather than any other. Not a confidence in
# the recording and not a CLAP score: it is about what the signal predicts.
AFFINITY_GRADES: Tuple[str, ...] = ("strong", "moderate", "weak")

# The room each anchor object sits in. ANALYST-ONLY, same fence as `anchor_object`.
#
# It exists because the clapsmoke-3 gate refuted the object-level taxonomy this table was
# built on. `plant` scored 0.383 with 187 of 480 rows landing on `toilet`: water sounds mean
# BATHROOM, and "a houseplant is a thing you pour water near" was the author reasoning from
# what could plausibly happen at an object rather than from what the sound predicts. Rooms
# are the level the sounds actually encode, and an object is still the navigable target, so
# grouping costs no new capability -- notably NOT a room labeller, which this tree does not
# have (`NullRoomLabeler` is what runs).
#
# `chair` joins the living room rather than getting a study of its own: HM3D homes do not
# reliably have offices, and a lone-class room cannot be split heard-from-not-heard.
# `plant` keeps its own room deliberately, so the re-score can say whether grouping rescues
# it or whether the class assignments under it were simply wrong.
ROOM_OF_ANCHOR: Dict[str, str] = {
    "toilet": "bathroom",
    "bed": "bedroom",
    "sofa": "living_room",
    "tv_monitor": "living_room",
    "chair": "living_room",
    "plant": "greenery",
}


def room_of(name: str) -> str:
    """ANALYST AND DATASET BUILDER ONLY -- the room a source of this class is placed in.

    The composition of `anchor_object` with `ROOM_OF_ANCHOR`, and fenced for the same reason:
    it is placement ground truth, and an agent that reads it measures this table instead of
    its own semantic store.
    """
    return ROOM_OF_ANCHOR[anchor_object(name)]



@dataclass(frozen=True)
class SoundClass:
    """One row of the candidate vocabulary.

    `name` is ours and `esc50_category` is the ESC-50 `category` column; they coincide
    today and are kept separate because a future clip source will not be ESC-50 and the
    prompt bank should not have to change when the corpus does.
    """

    name: str
    esc50_category: str
    prompt: str
    anchor_object: str
    affinity: str

    def __post_init__(self) -> None:
        if self.anchor_object not in HM3D_GOAL_CATEGORIES:
            raise ValueError(
                "{!r} anchors at {!r}, which is not one of HM3D ObjectNav's six goal "
                "categories {}; a source cannot be placed at an object the episode "
                "builder cannot find".format(
                    self.name, self.anchor_object, HM3D_GOAL_CATEGORIES
                )
            )
        if self.affinity not in AFFINITY_GRADES:
            raise ValueError(
                "{!r} has affinity {!r}, expected one of {}".format(
                    self.name, self.affinity, AFFINITY_GRADES
                )
            )


# The generous candidate set. Grades are declared judgements, argued in ADR-0018:
#   strong   -- the sound is made BY the object, or by the one activity that object hosts.
#   moderate -- the sound belongs to that object's room and rarely to another.
#   weak     -- the sound is plausible near the object and also plausible near others.
# Every weak row is here to be CUT by the gate if CLAP cannot carry it, not to be trusted.
CANDIDATE_VOCABULARY: Tuple[SoundClass, ...] = (
    # -- toilet / bathroom --------------------------------------------------
    SoundClass("toilet_flush", "toilet_flush", "a toilet flushing", "toilet", "strong"),
    SoundClass("brushing_teeth", "brushing_teeth", "someone brushing their teeth", "toilet", "strong"),
    SoundClass("water_drops", "water_drops", "water dripping from a tap", "toilet", "moderate"),
    # -- bed / bedroom ------------------------------------------------------
    SoundClass("clock_alarm", "clock_alarm", "an alarm clock beeping", "bed", "strong"),
    SoundClass("snoring", "snoring", "a person snoring", "bed", "strong"),
    SoundClass("breathing", "breathing", "a person breathing slowly", "bed", "moderate"),
    SoundClass("clock_tick", "clock_tick", "a clock ticking", "bed", "moderate"),
    SoundClass("crying_baby", "crying_baby", "a baby crying", "bed", "moderate"),
    # -- chair / desk -------------------------------------------------------
    SoundClass("keyboard_typing", "keyboard_typing", "typing on a computer keyboard", "chair", "strong"),
    SoundClass("mouse_click", "mouse_click", "clicking a computer mouse", "chair", "strong"),
    SoundClass("door_wood_creaks", "door_wood_creaks", "a wooden chair or door creaking", "chair", "weak"),
    # -- plant / window -----------------------------------------------------
    SoundClass("pouring_water", "pouring_water", "water being poured", "plant", "moderate"),
    SoundClass("chirping_birds", "chirping_birds", "birds chirping outside a window", "plant", "weak"),
    SoundClass("crickets", "crickets", "crickets chirping", "plant", "weak"),
    SoundClass("rain", "rain", "rain falling outside a window", "plant", "weak"),
    # -- sofa / living room -------------------------------------------------
    SoundClass("vacuum_cleaner", "vacuum_cleaner", "a vacuum cleaner running", "sofa", "weak"),
    SoundClass("coughing", "coughing", "a person coughing", "sofa", "weak"),
    SoundClass("drinking_sipping", "drinking_sipping", "someone sipping a drink", "sofa", "weak"),
    # -- tv_monitor ---------------------------------------------------------
    SoundClass("clapping", "clapping", "an audience clapping on television", "tv_monitor", "weak"),
    SoundClass("laughing", "laughing", "people laughing on television", "tv_monitor", "weak"),
)

# The forced-failure arm (ADR-0014: a detector ships both arms, the healthy path passing
# AND the induced failure firing). These are ESC-50 categories with NO plausible anchor in
# a home interior, so a gate that accepts them is accepting anything. They are supplied as
# AUDIO ONLY and never enter the prompt bank -- that is the point: the open-set rejection
# has to fire on a class it was never told about, which is the arm the `anommxv` gate that
# rejected 0 of 8 never had.
ABSENT_CLASSES: Tuple[str, ...] = (
    "chainsaw",
    "helicopter",
    "airplane",
    "church_bells",
    "sea_waves",
)


def class_names() -> Tuple[str, ...]:
    """Every candidate class name, in table order."""
    return tuple(entry.name for entry in CANDIDATE_VOCABULARY)


def prompts() -> Dict[str, str]:
    """The prompt bank `audio/clap.py` scores against: class name to CLAP text prompt.

    This is the whole of what the agent may know about the vocabulary. It carries no
    object, so a controller holding this dict cannot recover the placement mapping.
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
    """Every candidate at one affinity grade, for the strong-versus-weak breakdown."""
    if grade not in AFFINITY_GRADES:
        raise ValueError(
            "unknown affinity {!r}, expected one of {}".format(grade, AFFINITY_GRADES)
        )
    return tuple(entry for entry in CANDIDATE_VOCABULARY if entry.affinity == grade)


def anchor_object(name: str) -> str:
    """ANALYST AND DATASET BUILDER ONLY -- the object a source of this class is placed at.

    **The controller must never call this.** It is the sound-object mapping, and the
    agent's semantic store exists to learn it from prior visits. A controller that reads
    it turns the unseen-and-heard cell into a measurement of the table above.

    The fence is `tests/mac/test_vocabulary_fence.py`, which fails if anything under
    `earshot/agent/` reaches this module at all. That test is the enforcement; this
    docstring is only the reason.
    """
    for entry in CANDIDATE_VOCABULARY:
        if entry.name == name:
            return entry.anchor_object
    raise KeyError(
        "{!r} is not in the candidate vocabulary; present: {}".format(
            name, ", ".join(sorted(class_names()))
        )
    )
