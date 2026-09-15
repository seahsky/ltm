"""DREAM §III.E: the three levels of long-term memory, and the abstraction between them.

    M^L   = {M^E, M^P, M^K}                          (eq. 14)
    m^E_i = (h^av_i, h^traj_i, c^obj_i, y_i)         (eq. 15)
    M^P   = G(M^E)                                   (eq. 16)
    m^P_k = (c^audio_k, c^obj_k, c^sce_k, rho_k)     (eq. 17)
    m^K_l = (c^sound_l, c^obj_l, c^sce_l, c^spat_l)  (eq. 18)

`D*` from `consolidate.py` (eq. 13) is what fills `M^E`. `G` turns `M^E` into `M^P`. This
module is the destination PR 3 had none of.

**M^K IS THE STORE THAT ALREADY EXISTS, AND IS NOT REBUILT HERE.** Eq. 18's four fields
land on `store.SemanticEntry` almost exactly — `c^sound` is `sound_class`, `c^object` is
`category`, `c^scene` is `room` — and that store is already written by
`task.prior_pass` and queried by `task.runner` through ONE embedding path. Forking it into
a second sound-to-object table would be two paths answering the same question with no
symptom, which is the hazard `audio.clap.audio_embedding`'s docstring exists to name. So
`LongTermMemory` HOLDS a `SemanticStore` as its K level rather than defining a new row.
Eq. 18's `c^spatial` has no field on `SemanticEntry`; the tree realises the spatial prior
through `store.EpisodicStore` instead, which is ADR-0018's "seen" axis.

**THREE THINGS EQ. 15 AND 17 DO NOT SAY, AND ONE THEY SAY THAT CANNOT BE TAKEN AT FACE
VALUE.** Each is argued where it is made:

  1. `h^traj_i` "encodes the navigation trajectory" and is not defined. It is a
     SCENE-INDEPENDENT SHAPE DESCRIPTOR here, never coordinates — see
     `trajectory_descriptor`, where the paper's own §III.A forbids the alternative.
  2. `G` is "the experience abstraction operation" and is not defined. It groups
     SUCCESSFUL experiences by their concept triple and summarises each group — see
     `abstract`.
  3. `rho_k` "summarizes a reusable navigation strategy" and is not defined. See
     `Strategy`.
  4. Eq. 17 needs `c^audio_k` and `c^sce_k`, and eq. 15 gives `m^E_i` NEITHER. `G` cannot
     produce an acoustic concept out of a fused vector and an object name, so the
     experience row carries both — marked in `ExperienceEntry` as the fields eq. 15 does
     not list.

**"EPISODIC" NOW MEANS TWO DIFFERENT THINGS IN THIS PACKAGE, SO NEITHER IS CALLED THAT
HERE.** `store.EpisodicStore` is ADR-0018's matrix axis: where a category was literally
seen on a prior tour, as coordinates, used to realise the seen/unseen split. DREAM's
"episodic experience memory" (eq. 15) is a different object with a different purpose, and
it is `ExperienceStore` in this module for that reason. Nothing converts between them.

**WHAT NOVELTY CAN RANGE OVER (eq. 12) IS NARROWER THAN `M^L`.** `N_j = 1 - max_{m in
M^L} sim(h_j, m)` reads as a single max over all three levels, and it cannot be: `M^E` and
`M^P` are keyed by the fused `z^av` and `M^K` by a CLAP vector of half that width, so
`sim` is not defined between `h_j` and an `m^K`. The paper's own eq. 20-22 retrieve from
the three levels SEPARATELY rather than from their union, which is the same fact. So
`novelty_vectors` returns the two levels that share `h_j`'s space and says why.

**THE LAYER TABLE MOVES HERE.** `LAYER_IMPORTS["memory"]` grows from `("types",)` to
`("memory", "types")`, because `LongTermMemory` holds a `SemanticStore` and so this module
imports `store.py`. That is the widening PR #110's body said would be needed for
consolidation and was not; it is needed now, and it is a SELF-edge only — `memory` still
reaches neither `audio` nor `agent`, which are the two absences that do the work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from earshot.memory.store import SemanticStore
from earshot.types import Xyz

__all__ = [
    "Outcome",
    "TRAJECTORY_COMPONENTS",
    "trajectory_descriptor",
    "ExperienceEntry",
    "ExperienceStore",
    "Strategy",
    "Pattern",
    "PatternStore",
    "abstract",
    "LongTermMemory",
]

# The names of `h^traj`'s components, in order. Named rather than left as anonymous axes
# because a descriptor whose fields nobody can recite is a vector that will be compared
# against the wrong thing eventually.
TRAJECTORY_COMPONENTS: Tuple[str, ...] = (
    "path_length_m",
    "net_displacement_m",
    "straightness",
    "mean_abs_turn_rad",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError(
            "a zero-norm vector reached a long-term memory key. It cannot be normalised, "
            "and storing it unchanged would make it score 0.0 against every query -- a "
            "row that can never be retrieved and never says so"
        )
    return values / norm


@dataclass(frozen=True)
class Outcome:
    """`y_i` (eq. 15): how the episode that produced this experience ended.

    Two fields rather than a bool, because the tree measures two things and both decide
    whether an experience is worth abstracting: `reached` is Find-SR@1m's own predicate,
    and `final_gap_m` is the DTG the runner already writes as `dtg_source_final_m`. A
    bool alone would make a 1.1 m miss and a 12 m miss the same experience.
    """

    reached: bool
    final_gap_m: float

    def __post_init__(self) -> None:
        gap = float(self.final_gap_m)
        if gap < 0.0:
            raise ValueError(
                "Outcome.final_gap_m must be >= 0, got {}; a negative distance to the "
                "source is a sign error upstream, not an outcome".format(gap)
            )
        object.__setattr__(self, "final_gap_m", gap)
        object.__setattr__(self, "reached", bool(self.reached))


def trajectory_descriptor(positions: Sequence[Xyz]) -> np.ndarray:
    """`h^traj_i` (eq. 15): the SHAPE of a walk, in units that mean the same in any home.

    **THE PAPER FORBIDS THE OBVIOUS IMPLEMENTATION, IN ITS OWN WORDS.** §III.A: "`M^L` is
    designed to capture transferable audio-visual-semantic and spatial knowledge rather
    than memorizing environment-specific trajectories or target locations." A `h^traj`
    built from coordinates is exactly the memorised trajectory that sentence rules out,
    and it would make every `M^E` row dead the moment the scene changed -- which is three
    of the four cells of ADR-0018's matrix.

    So the four components of `TRAJECTORY_COMPONENTS`, all of them properties of the walk
    and none of them properties of the house:

      * `path_length_m` -- how far the agent actually walked;
      * `net_displacement_m` -- how far it ended from where it began;
      * `straightness` -- their ratio, in [0, 1]. 1.0 is a straight run at the target;
        near 0 is a search that came back to itself. This is the component that separates
        "walked to it" from "wandered into it", which no single distance can;
      * `mean_abs_turn_rad` -- the mean unsigned heading change per step.

    **UNSIGNED turns, and that is not laziness.** `types.py` holds no bearing helper
    because ticket 09 found the lateral sign silently inverting between world and agent
    frame with no code change. A signed turn here would assert the same convention this
    tree refuses to assert anywhere but `audio/lateral.py`, measured by a box test. The
    magnitude of a turn is the same number in either frame, so taking `abs` makes the
    descriptor frame-free rather than frame-wrong.

    Horizontal distance throughout (y up, ignored), the tree's convention: a descriptor
    that charged a walk for a staircase would make the same route two different
    experiences on two floors.

    NOT normalised. These are interpretable physical quantities, they never enter a
    cosine -- retrieval keys on `h^av`, not on this -- and unit-normalising them would
    trade the one property that makes them readable in an artefact for nothing.
    """
    points = list(positions)
    if not points:
        raise ValueError(
            "trajectory_descriptor got no positions; an experience with no walk behind "
            "it is not an experience, and a zero descriptor would read as a perfectly "
            "straight walk of zero length"
        )
    path = sum(
        points[index].horizontal_distance_to(points[index + 1])
        for index in range(len(points) - 1)
    )
    net = points[0].horizontal_distance_to(points[-1])
    straightness = (net / path) if path > 0.0 else 0.0
    turns: List[float] = []
    for index in range(len(points) - 2):
        first, second, third = points[index], points[index + 1], points[index + 2]
        before = np.arctan2(second.z - first.z, second.x - first.x)
        after = np.arctan2(third.z - second.z, third.x - second.x)
        turns.append(abs(float(np.arctan2(np.sin(after - before), np.cos(after - before)))))
    mean_turn = (sum(turns) / len(turns)) if turns else 0.0
    return np.array(
        [path, net, straightness, mean_turn], dtype=np.float32
    )


@dataclass(frozen=True, eq=False)
class ExperienceEntry:
    """`m^E_i` (eq. 15), plus the two concepts eq. 17 needs and eq. 15 does not carry.

    `eq=False` for the arrays, the reason `store.SemanticEntry` sets it: a generated
    `__eq__` would compare a tuple holding a numpy array, whose `bool()` raises.

    **`sound_concept` AND `room_concept` ARE NOT IN EQ. 15.** They are here because eq. 17
    requires `c^audio_k` and `c^sce_k` on every pattern, and `G` has no way to derive an
    acoustic concept from a fused vector and an object name. Either the experience row
    carries them or `G` invents them; it carries them, and this paragraph is the record
    that the paper's two equations do not line up.

    **`room_concept` IS `Optional` BECAUSE THE LABELLER ABSTAINS.**
    `audio.normality.RoomLabeler`'s contract is "a room type or `None`", and the labeller
    the runs actually use is `NullRoomLabeler`, which always abstains. So `None` is the
    ordinary value today, and it makes the scene axis of `G` visibly flat in the artefact
    instead of quietly flat. A GT room label would make the whole level privileged.

    **`room_concept` IS A ROOM, NEVER A SCENE ID.** Eq. 17 says "the associated scene" and
    the M^K paragraph says "scene/region prior", so the transferable reading is the
    region. Keying a pattern on the mesh would build a pattern that can never fire in a
    new home, which is the one thing §III.A rules out.
    """

    context: np.ndarray
    trajectory: np.ndarray
    target_concept: str
    outcome: Outcome
    sound_concept: str
    room_concept: Optional[str]

    def __post_init__(self) -> None:
        context = np.array(self.context, dtype=np.float32, copy=True).reshape(-1)
        if context.size == 0:
            raise ValueError(
                "ExperienceEntry was given an empty h^av; a 0-length vector cannot be a "
                "retrieval key and must not reach M^E"
            )
        if not bool(np.isfinite(context).all()):
            raise ValueError(
                "ExperienceEntry was given a non-finite h^av. A NaN key silently loses "
                "every comparison in eq. 12 and eq. 20, so the row is stored and can "
                "never be retrieved"
            )
        context.flags.writeable = False
        object.__setattr__(self, "context", context)

        trajectory = np.array(self.trajectory, dtype=np.float32, copy=True).reshape(-1)
        if trajectory.size != len(TRAJECTORY_COMPONENTS):
            raise ValueError(
                "h^traj must have the {} components of TRAJECTORY_COMPONENTS {}, got {}; "
                "a descriptor of another width was built by something other than "
                "trajectory_descriptor".format(
                    len(TRAJECTORY_COMPONENTS), TRAJECTORY_COMPONENTS, trajectory.size
                )
            )
        trajectory.flags.writeable = False
        object.__setattr__(self, "trajectory", trajectory)

        for name in ("target_concept", "sound_concept"):
            if not str(getattr(self, name)).strip():
                raise ValueError(
                    "ExperienceEntry.{} is empty. An unnamed concept groups with every "
                    "other unnamed concept in G and produces a pattern about "
                    "nothing".format(name)
                )


@dataclass(frozen=True)
class ExperienceStore:
    """`M^E` (eq. 14/15): consolidated experiences, newest last. Immutable.

    Width-guarded the way `SemanticStore` is: a store mixing two encoders' keys cannot be
    retrieved from coherently, and finding out at the query is finding out too late.
    """

    entries: Tuple[ExperienceEntry, ...] = ()

    def __post_init__(self) -> None:
        widths = sorted({int(entry.context.size) for entry in self.entries})
        if len(widths) > 1:
            raise ValueError(
                "ExperienceStore holds h^av of mismatched width {}; a store mixing "
                "encoders is a wiring bug, not a store to be retrieved from".format(widths)
            )

    @property
    def dim(self) -> Optional[int]:
        """The key width every row shares, or `None` on an empty store — NEVER 0."""
        if not self.entries:
            return None
        return int(self.entries[0].context.size)

    def __len__(self) -> int:
        return len(self.entries)

    def extend(self, entries: Sequence[ExperienceEntry]) -> "ExperienceStore":
        """A NEW store with `entries` appended. Never mutates the receiver.

        What a finished episode calls with its `D*`. Returning a new store is what lets
        the audit hold the memory as it stood at the start of each episode, which is the
        only state a retrieval can be reproduced against afterwards.
        """
        return ExperienceStore(entries=tuple(self.entries) + tuple(entries))

    @property
    def keys(self) -> Tuple[np.ndarray, ...]:
        """Every `h^av_i`, in order. What eq. 20 retrieves over."""
        return tuple(entry.context for entry in self.entries)


@dataclass(frozen=True)
class Strategy:
    """`rho_k` (eq. 17): "a reusable navigation strategy", which the paper leaves open.

    **THE CHOICE.** What is reusable about a group of successful episodes that heard the
    same sound and reached the same kind of object is: what the group SOUNDED AND LOOKED
    like when it worked (`signature`), what shape of walk got there (`trajectory`), how
    close it got (`mean_final_gap_m`), and how many times (`support`). Those four are
    computable with no training and they are exactly what eq. 26's `S_mem` will need to
    score a candidate plan against a pattern.

    `support` is not in eq. 17. It is here because a pattern abstracted from two episodes
    and a pattern abstracted from forty must not weigh the same in a retrieval, and
    nothing else on the row carries that.

    `signature` is unit-normalised (it is a retrieval key and is compared by cosine);
    `trajectory` is not (it is a physical descriptor and is read, not compared).
    """

    support: int
    signature: np.ndarray
    trajectory: np.ndarray
    mean_final_gap_m: float

    def __post_init__(self) -> None:
        if int(self.support) < 1:
            raise ValueError(
                "Strategy.support must be >= 1, got {}; a pattern abstracted from no "
                "experience is not an abstraction".format(self.support)
            )
        object.__setattr__(self, "support", int(self.support))
        for name in ("signature", "trajectory"):
            vector = np.array(getattr(self, name), dtype=np.float32, copy=True).reshape(-1)
            vector.flags.writeable = False
            object.__setattr__(self, name, vector)
        object.__setattr__(self, "mean_final_gap_m", float(self.mean_final_gap_m))


@dataclass(frozen=True, eq=False)
class Pattern:
    """`m^P_k` (eq. 17): one recurring regularity, keyed by its concept triple."""

    sound_concept: str
    object_concept: str
    room_concept: Optional[str]
    strategy: Strategy

    @property
    def concepts(self) -> Tuple[str, str, Optional[str]]:
        """The triple `G` grouped on. What makes two patterns the same pattern."""
        return (self.sound_concept, self.object_concept, self.room_concept)


@dataclass(frozen=True)
class PatternStore:
    """`M^P` (eq. 14/16): the abstraction of `M^E`. Built by `abstract`, never by hand."""

    patterns: Tuple[Pattern, ...] = ()

    def __len__(self) -> int:
        return len(self.patterns)

    @property
    def keys(self) -> Tuple[np.ndarray, ...]:
        """Every `rho_k.signature`, in order. What eq. 21 retrieves over."""
        return tuple(pattern.strategy.signature for pattern in self.patterns)


def abstract(experiences: ExperienceStore, *, min_support: int) -> PatternStore:
    """`M^P = G(M^E)` (eq. 16). The paper names `G` and does not define it.

    **THE CHOICE.** Group the experiences by their concept triple
    `(c^audio, c^obj, c^sce)` — which is what eq. 17 says a pattern is keyed by — and
    summarise each group into one `Strategy`. Training-free, deterministic, and its output
    is readable in an artefact, which a learned abstraction's would not be.

    **SUCCESSFUL EXPERIENCES ONLY, AND HERE THE PAPER SAYS SO.** §III.E.2: "DREAM
    therefore abstracts recurring regularities from related SUCCESSFUL experiences." That
    is a different situation from `C_j` in `consolidate.py`, where the same word appeared
    in a clause about a segment's contribution and the success gate was REFUSED because
    gating there would have discarded roughly two thirds of all experience before novelty
    ever saw it. Here the gate is at abstraction rather than at retention: `M^E` still
    keeps every failure, and only the pattern level is success-only. The two readings are
    consistent, and the difference is which stage the filter sits at.

    `min_support` has no default and is required. A "recurring regularity" that occurred
    once is not one, and the number of times something has to recur before it counts is
    the whole content of the word — so the caller states it and the audit records it.

    Deterministic order: patterns come out sorted by their concept triple, with an
    absent room last. Not insertion order, which would make two runs over the same
    experiences produce two different `M^P` and make a retrieval irreproducible.
    """
    if int(min_support) < 1:
        raise ValueError(
            "min_support must be >= 1, got {}; a pattern needs at least one experience "
            "behind it, and 0 would emit a pattern for a group that does not exist".format(
                min_support
            )
        )
    groups: Dict[Tuple[str, str, Optional[str]], List[ExperienceEntry]] = {}
    for entry in experiences.entries:
        if not entry.outcome.reached:
            continue
        key = (entry.sound_concept, entry.target_concept, entry.room_concept)
        groups.setdefault(key, []).append(entry)

    patterns: List[Pattern] = []
    # `None` cannot be compared with `str` under Python 3, so the room sorts on a
    # (present?, value) pair and an absent room lands last rather than raising.
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2] is None, k[2] or "")):
        members = groups[key]
        if len(members) < int(min_support):
            continue
        sound, obj, room = key
        patterns.append(Pattern(
            sound_concept=sound,
            object_concept=obj,
            room_concept=room,
            strategy=Strategy(
                support=len(members),
                signature=_unit(np.stack([m.context for m in members]).mean(axis=0)),
                trajectory=np.stack([m.trajectory for m in members]).mean(axis=0),
                mean_final_gap_m=sum(m.outcome.final_gap_m for m in members) / len(members),
            ),
        ))
    return PatternStore(patterns=tuple(patterns))


@dataclass(frozen=True)
class LongTermMemory:
    """`M^L = {M^E, M^P, M^K}` (eq. 14). The three levels, held together and nothing more.

    `knowledge` is `store.SemanticStore` — the table `task.prior_pass` already writes and
    `task.runner` already queries. See the module docstring on why eq. 18 does not get a
    new row type.
    """

    experience: ExperienceStore
    pattern: PatternStore
    knowledge: SemanticStore

    def novelty_vectors(self) -> Tuple[np.ndarray, ...]:
        """The rows eq. 12's `max_{m in M^L}` can actually range over: `M^E` and `M^P`.

        **`M^K` IS EXCLUDED, AND THE PAPER FORCES IT.** `h_j` and the `M^E`/`M^P` keys are
        fused `z^av` vectors; an `m^K` is keyed by a CLAP embedding of half that width, so
        `sim(h_j, m^K)` is not a defined quantity and `consolidate.novelty` raises on it
        rather than comparing across two spaces. Eq. 12 reads as one max over the union of
        all three levels and cannot be, which the paper's own eq. 20-22 already concede by
        retrieving from the three levels SEPARATELY rather than from their union.

        This is a statement about the equation, not a shortcut: including `M^K` here would
        mean either silently padding one space into the other or dropping the width check
        that catches a wrong encoder.
        """
        return self.experience.keys + self.pattern.keys

    def __repr__(self) -> str:
        return "LongTermMemory(M^E={} rows, M^P={} patterns, M^K={} rows)".format(
            len(self.experience), len(self.pattern), len(self.knowledge)
        )
