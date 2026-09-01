"""The two learned tables ADR-0018's matrix ablates: what was heard, and what was seen.

`SemanticEntry` / `SemanticStore` hold a sound-class embedding and the room it was heard
in — a prior visit's "I heard a flush; this is the bathroom" — and `predict_room`
answers a query embedding with a k-NN vote over cosine similarity. `EpisodicEntry` /
`EpisodicStore` hold where a category was actually seen on a prior tour — the geometric
half, `TourRecord.legs` filtered to `leg.reached` — and `points_for_room` answers where
to look. `without_class` and `without_scene` are the pure filters that carve the matrix's
four cells (heard/not-heard crossed with seen/unseen) out of one full store each; a task-
layer caller composes them, this module does not know the cells exist.

**This module must never be able to answer from the placement table.** `room_of` /
`anchor_object` / `ROOM_OF_ANCHOR` in `earshot.audio.vocabulary` are the ground truth an
episode places its source with, and the unseen-and-heard cell's whole claim is that the
semantic store LEARNED its room association by hearing the class on prior visits — a
store that could read the table instead would measure the author's map, not the agent's
memory. `tests/mac/test_audio_vocabulary.py`'s `TestAnchorFence` widens to scan this
package, and `LAYER_IMPORTS["memory"]` (`tests/mac/_tree.py`) does not include `audio`,
so the naive leak does not even compile. Belt and braces, as that file's own docstring
puts it.

**Embedding dtype and shape: `np.float32`, 1-D, shape `(D,)`, `D` fixed per store.**
`np.float32` because every producer already is — `audio.clap._unit` does
`np.asarray(vector, dtype=np.float32).reshape(-1)`, and this module cannot import
`audio.clap` to share that constant, so the dtype is restated here rather than imported.
Storing float64 would make this the one float64 surface in the tree and silently double
a table with one row per prior-tour observation. 1-D and never `(1, D)` or `(D, 1)`:
`np.dot` of two 1-D arrays is a Python-castable scalar, while a 2-D shape's `float()` is
a deprecation warning under the pinned numpy (`<1.24`) and an outright error after it.
`D` is read from the store's own first entry rather than hard-coded to CLAP's 512, so a
differently-sized encoder is a loud `ValueError` at the query that used it, not a value
silently broadcast against the wrong axis.

**A `SemanticEntry` stores the raw encoder vector, not a pre-normalised one.**
Normalising at construction would hide a zero-norm input behind a NaN or a zero unit
vector forever, and would fix the store to whatever metric happened to normalise it. A
raw vector can be re-scored under a different metric later, and a zero-norm query or
entry is caught explicitly, at the moment it would otherwise silently score everything
0.0 -- see `predict_room` below.

**`SemanticEntry` re-copies its embedding in `__post_init__`** —
`np.array(embedding, dtype=np.float32, copy=True).reshape(-1)`, `writeable` cleared on
*that copy* — so the constructor is pure with respect to its caller's array: nothing the
caller does to the array it passed in afterwards can reach back into a stored entry, and
nothing this store does can reach forward into the caller's.

**`SemanticEntry` is `eq=False`.** A generated `__eq__` would compare the field tuple
with `==`, and a numpy array in that tuple makes `==` return an array whose `bool()`
raises. `eq=False` keeps identity equality and a working `__hash__` — `agent/occupancy.py`
`OccupancyGrid` sets the same flag for the same reason. `EpisodicEntry` carries no array
and stays a plain frozen dataclass.

Nothing in this module imports `agent/`, and nothing in `agent/` may import this module
(`LAYER_IMPORTS["agent"] = ("agent", "vlm", "types")`, unwidened) — the controller reaches
a store by injection, the same rule `task.prior_pass.walk_tour`'s `observe` callback and
`agent.detector.OracleDetector`'s `distance_to` already follow. The candidate this store
feeds into the scorer is built in `task/`, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from earshot.types import Xyz

__all__ = [
    "SemanticEntry",
    "EpisodicEntry",
    "SemanticStore",
    "EpisodicStore",
    "MemoryCondition",
    "without_class",
    "without_scene",
]


@dataclass(frozen=True, eq=False)
class SemanticEntry:
    """One prior "I heard `sound_class` here" observation, in `room` of `donor_scene`.

    `donor_scene` is provenance, not a filter axis — see `SemanticStore.donor_scenes`
    and the module docstring on why the unseen column is realised on the episodic store
    alone.
    """

    sound_class: str
    room: str
    embedding: np.ndarray
    donor_scene: str

    def __post_init__(self) -> None:
        vector = np.array(self.embedding, dtype=np.float32, copy=True).reshape(-1)
        if vector.size == 0:
            raise ValueError(
                "SemanticEntry({!r}, {!r}) was given an empty embedding; a 0-d or "
                "zero-length vector cannot be a store row".format(
                    self.sound_class, self.room
                )
            )
        vector.flags.writeable = False
        object.__setattr__(self, "embedding", vector)


@dataclass(frozen=True)
class EpisodicEntry:
    """One prior tour leg that was actually REACHED: where `category` sits in `room`.

    Exactly `prior_pass.TourStop(room, category, point)` plus the scene the leg walked
    in — the task-layer adapter is `EpisodicEntry(scene=record.scene, room=leg.stop.room,
    category=leg.stop.category, point=leg.stop.point)` over `TourRecord.legs` where
    `leg.reached`, and these field names must not drift from that source.
    """

    scene: str
    room: str
    category: str
    point: Xyz


@dataclass(frozen=True)
class SemanticStore:
    """A k-NN table over sound-class embeddings, each tagged with the room it was heard in.

    Immutable: both mutating operations the matrix needs — dropping a class, dropping a
    scene's provenance — are the free functions `without_class` / `without_scene` below,
    which return a NEW store and never touch this one's `entries`.
    """

    entries: Tuple[SemanticEntry, ...] = ()

    def __post_init__(self) -> None:
        dims = sorted({int(entry.embedding.size) for entry in self.entries})
        if len(dims) > 1:
            raise ValueError(
                "SemanticStore holds entries of mismatched embedding size {}; a store "
                "mixing encoders is a wiring bug, not a store to be queried".format(dims)
            )

    @property
    def dim(self) -> Optional[int]:
        """The embedding width every entry shares, or `None` on an empty store — NEVER 0."""
        if not self.entries:
            return None
        return int(self.entries[0].embedding.size)

    @property
    def sound_classes(self) -> Tuple[str, ...]:
        """Every class held, sorted and de-duplicated."""
        return tuple(sorted({entry.sound_class for entry in self.entries}))

    @property
    def donor_scenes(self) -> Tuple[str, ...]:
        """Every scene an entry was donated from, sorted and de-duplicated. Provenance
        only — see the module docstring on why this is not a third filter axis."""
        return tuple(sorted({entry.donor_scene for entry in self.entries}))

    def __len__(self) -> int:
        return len(self.entries)

    def predict_room(self, embedding: np.ndarray, k: int) -> Optional[Tuple[str, float]]:
        """The k nearest entries by cosine, grouped by room, the highest mean wins.

        `k` has no default: a knob with a default is a knob that reaches no artefact, so
        the caller passes it and the runner records it beside the prediction.

        Raises `ValueError` if `k < 1`, or if the query's flattened size does not match
        `self.dim` — a dimension mismatch is a wiring bug (a different encoder reached
        this store) and must never read as a low score.

        Returns `None` — never `(room, 0.0)` — when the store is empty, when the query
        vector's norm is 0, or when every entry that could be scored also has a zero
        norm: a degenerate vector is not evidence of anything and must not silently
        outvote a real one.

        Pure: sorts a local list, never mutates `self.entries` or the caller's `embedding`.
        """
        if k < 1:
            raise ValueError("predict_room needs k >= 1, got {}".format(k))
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not self.entries:
            return None
        if query.size != self.dim:
            raise ValueError(
                "query embedding has size {} but this store's entries are size {}".format(
                    query.size, self.dim
                )
            )
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return None
        unit_query = query / query_norm

        scored: List[Tuple[float, str]] = []
        for entry in self.entries:
            entry_norm = float(np.linalg.norm(entry.embedding))
            if entry_norm == 0.0:
                # A zero-norm entry carries no direction to compare against; excluding
                # it is not the same as it losing every comparison, which is why this
                # is a skip and not a cosine of 0.0.
                continue
            cosine = float(np.dot(unit_query, entry.embedding / entry_norm))
            scored.append((cosine, entry.room))
        if not scored:
            return None

        scored.sort(key=lambda item: -item[0])
        nearest = scored[: min(k, len(scored))]

        by_room: Dict[str, List[float]] = {}
        for cosine, room in nearest:
            by_room.setdefault(room, []).append(cosine)
        means = [(room, sum(cosines) / len(cosines)) for room, cosines in by_room.items()]
        # Ties broken by room name ascending -- data-independent and reproducible from a
        # log, unlike insertion order or ROOMS' declaration order (unreachable here by
        # design: that table lives in the fenced `audio.vocabulary`).
        means.sort(key=lambda item: (-item[1], item[0]))
        winner_room, winner_score = means[0]
        return (winner_room, winner_score)


@dataclass(frozen=True)
class EpisodicStore:
    """Where categories were actually reached on prior tours, keyed by scene and room."""

    entries: Tuple[EpisodicEntry, ...] = ()

    @property
    def scenes(self) -> Tuple[str, ...]:
        """Every scene held, sorted and de-duplicated."""
        return tuple(sorted({entry.scene for entry in self.entries}))

    def __len__(self) -> int:
        return len(self.entries)

    def points_for_room(self, scene: str, room: str) -> Tuple[Xyz, ...]:
        """Every reached point in `room` of `scene`, in stored order.

        `()` — never `None` — when nothing matches: an empty tuple here is a complete
        and real answer ("this store holds no such point"), not an unanswered query.
        """
        return tuple(
            entry.point
            for entry in self.entries
            if entry.scene == scene and entry.room == room
        )


class MemoryCondition(Enum):
    """Which cell of ADR-0018's matrix an episode ran in.

    Not a `RunConfig` field: the cell is realised by WHICH stores the caller built and
    handed to the episode (`without_class` / `without_scene` applied or not), not by a
    branch inside the run, and putting it on `RunConfig` would force
    `LAYER_IMPORTS["config"]` to grow a `memory` edge for a value the config cannot act
    on. The cell is therefore selected by the sweep driver rather than typed on
    `RunConfig`, so the audit record is the only place it can be witnessed.

    **`EpisodeAudit.memory_condition` DOES NOT EXIST YET, deliberately.** Nothing in
    `task/` builds a store, so a typed field here would be `None` on every record ever
    written while its `None` documented two different facts ("predates the field" and
    "ran outside the matrix") — the shape of this repo's two absence incidents. It lands
    with its writer, as a typed field and not a metric, because a string cannot live in
    a `Mapping[str, float]`.
    """

    NONE = "none"
    HEARD_SEEN = "heard_seen"
    HEARD_UNSEEN = "heard_unseen"
    NOT_HEARD_SEEN = "not_heard_seen"
    NOT_HEARD_UNSEEN = "not_heard_unseen"


def without_class(store: SemanticStore, name: str) -> SemanticStore:
    """A new `SemanticStore` holding every entry except those of sound class `name`.

    Total and pure: a `name` absent from the store returns a store equal in content to
    the input (no raise, no warning), and `store.entries` is never mutated -- the input
    `store` is unchanged after this call, which is the property the matrix's four cells
    depend on (the same full store is filtered four different ways).
    """
    return SemanticStore(
        entries=tuple(entry for entry in store.entries if entry.sound_class != name)
    )


def without_scene(store: EpisodicStore, scene: str) -> EpisodicStore:
    """The episodic twin of `without_class`: drop every entry walked in `scene`.

    Total and pure, on the same terms: an absent `scene` is a no-op copy, and the input
    `store` is unchanged after this call.
    """
    return EpisodicStore(
        entries=tuple(entry for entry in store.entries if entry.scene != scene)
    )
