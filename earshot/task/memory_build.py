"""From a walked tour to the two stores, and from the two stores to ADR-0018's four cells.

`task/prior_pass.py` walks a scripted route and returns a `TourRecord`. `memory/store.py`
holds the two tables the matrix ablates. Nothing joined them, and this module is that
join: **pure, Mac-testable, and the only place that knows both shapes.**

Layering (ADR-0013). `memory/` sits at `("memory", "types")` and cannot import `audio`,
which is what stops a store reading `ROOM_OF_ANCHOR` — the answer key the unseen-and-heard
cell's whole claim rests on not being read. `task/` is the wiring layer and may import
both, so the adapter lives here rather than on either side. A `from_tour` method on
`SemanticStore` would have dragged `prior_pass` into `memory/`'s import surface for no
gain; a `to_store` method on `TourRecord` would have done the same in the other direction.

**The four cells are carved from ONE built store each, never from four tours.** That is
ADR-0018's amendment (a) and it is what makes the matrix paired: the same episode runs in
all four cells, so `episode_diff` compares an episode against itself. Building four stores
from four differently-toured scene sets would make every cell a different experiment and
drop the design from a 5.04-point paired MDE to 14.0 unpaired at the same rendering cost.

**A malformed observation RAISES; it is never skipped.** `walk_tour`'s `observe` callback
returns `Mapping[str, object]` because that module refuses to know what a store is, so the
payload arrives untyped and this is where it is validated. An observation missing its
embedding means the render or the encoder failed at that stop, and a store quietly one row
short is a store whose k-NN vote is different for a reason nothing on disk records. The
tour's own abandoned legs are already recorded as abandoned; a silent drop here would be
the one loss with no record.

**Round-trip through JSON is lossless for everything a query reads.** The prior pass runs
on the box and the store is consumed by a later run in the same sweep, so the stores cross
a process boundary and the format is part of the contract. `float32` in, `float32` out,
at full precision (`repr`-grade floats, not a fixed number of decimals): a store whose
cosines shift in the eighth digit between the tour and the run is a store that answers a
different question from the one it learned.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from earshot.memory.store import (
    EpisodicEntry,
    EpisodicStore,
    MemoryCondition,
    SemanticEntry,
    SemanticStore,
    without_class,
    without_scene,
)
from earshot.task.prior_pass import TourRecord, TourStop
from earshot.types import Xyz

__all__ = [
    "OBSERVATION_KEYS",
    "MemoryBuildError",
    "tour_observation",
    "episodic_from_tour",
    "semantic_from_tour",
    "stores_for_cell",
    "dump_stores",
    "load_stores",
    "STORE_FORMAT_VERSION",
]

# What `walk_tour`'s `observe` callback must put in every payload. Named here rather than
# in `prior_pass` because `prior_pass` is deliberately ignorant of stores, and named at
# all because the callback's return type is `Mapping[str, object]` and an untyped
# contract that lives only in a docstring is one this repo has already watched drift.
OBSERVATION_KEYS: Tuple[str, ...] = ("sound_class", "room", "category", "embedding")

# Bumped when the on-disk shape changes in a way an old reader would misread. `load_stores`
# refuses an unknown version rather than guessing at it: a store read under the wrong
# schema answers queries confidently and wrongly, which is the failure mode with no symptom.
STORE_FORMAT_VERSION = 1


class MemoryBuildError(ValueError):
    """A tour record that cannot become a store, with the stop that broke it named."""


def tour_observation(
    stop: TourStop, *, sound_class: str, embedding: Sequence[float]
) -> Dict[str, Any]:
    """The payload `walk_tour`'s `observe` callback returns at one reached stop.

    A constructor rather than a dict literal at the call site, so the box-side callback and
    the Mac-side tests build the same shape from the same code. `embedding` is stored as a
    plain list of Python floats because this crosses a JSON boundary; the `float32` cast
    happens once, in `SemanticEntry.__post_init__`, and is not done twice in two places.
    """
    values = [float(value) for value in embedding]
    if not values:
        raise MemoryBuildError(
            "the encoder returned an empty embedding at {} in {} — a store row with no "
            "direction scores every query 0.0 and must not be written".format(
                stop.category, stop.room
            )
        )
    return {
        "sound_class": str(sound_class),
        "room": str(stop.room),
        "category": str(stop.category),
        "embedding": values,
    }


def episodic_from_tour(record: TourRecord) -> EpisodicStore:
    """Every REACHED leg of one tour as episodic rows. Abandoned legs are not in the store.

    `leg.reached` is the filter and it is not negotiable: a leg that ran out of budget left
    the agent somewhere short of the stop, so the stop's point was never confirmed
    navigable-to from that route. `TourRecord.complete` is what a caller checks to decide
    whether the SCENE counts as seen; this function's job is only to carry across what the
    tour actually reached.
    """
    return EpisodicStore(
        entries=tuple(
            EpisodicEntry(
                scene=record.scene,
                room=leg.stop.room,
                category=leg.stop.category,
                point=leg.stop.point,
            )
            for leg in record.legs
            if leg.reached
        )
    )


def semantic_from_tour(record: TourRecord) -> SemanticStore:
    """Every observation of one tour as semantic rows, tagged with the tour's scene.

    `donor_scene` is `record.scene` — provenance, and never a filter axis (`store.py`).
    The unseen column is realised on the EPISODIC store alone, so a semantic row donated
    by the scene under test is not a leak: what the semantic store knows is "this class is
    heard in this kind of room", which is exactly the transferable fact the heard axis is
    about.

    Raises on any observation missing a key or carrying an unusable embedding. See the
    module docstring on why this is not a skip.
    """
    entries: List[SemanticEntry] = []
    for index, payload in enumerate(record.observations):
        missing = [key for key in OBSERVATION_KEYS if key not in payload]
        if missing:
            raise MemoryBuildError(
                "observation {} of the tour of {} is missing {} — the observe callback "
                "and memory_build.tour_observation have drifted apart".format(
                    index, record.scene, ", ".join(missing)
                )
            )
        raw = payload["embedding"]
        if not isinstance(raw, (list, tuple, np.ndarray)):
            raise MemoryBuildError(
                "observation {} of the tour of {} carries an embedding of type {}; a "
                "sequence of floats was expected".format(
                    index, record.scene, type(raw).__name__
                )
            )
        vector = np.asarray(raw, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise MemoryBuildError(
                "observation {} of the tour of {} carries an empty embedding".format(
                    index, record.scene
                )
            )
        entries.append(
            SemanticEntry(
                sound_class=str(payload["sound_class"]),
                room=str(payload["room"]),
                embedding=vector,
                donor_scene=record.scene,
            )
        )
    # `SemanticStore.__post_init__` raises on mixed embedding widths, which is the check
    # for "two encoders reached one tour". Constructed rather than validated by hand.
    return SemanticStore(entries=tuple(entries))


def stores_for_cell(
    semantic: SemanticStore,
    episodic: EpisodicStore,
    condition: MemoryCondition,
    *,
    sound_class: str,
    scene: str,
) -> Tuple[SemanticStore, EpisodicStore]:
    """The two stores ONE episode of ONE cell is allowed to consult.

    The whole matrix, in one total function over `MemoryCondition`:

    | condition          | semantic                  | episodic                |
    |--------------------|---------------------------|-------------------------|
    | `HEARD_SEEN`       | full                      | full                    |
    | `HEARD_UNSEEN`     | full                      | `without_scene(scene)`  |
    | `NOT_HEARD_SEEN`   | `without_class(class)`    | full                    |
    | `NOT_HEARD_UNSEEN` | `without_class(class)`    | `without_scene(scene)`  |
    | `NONE`             | empty                     | empty                   |

    `NONE` returns EMPTY STORES rather than the full ones. It is the condition an episode
    outside the matrix runs under, and handing it a populated store would make "no memory
    condition recorded" and "both memories" the same run — the exact conflation
    `MemoryCondition`'s own docstring refuses to allow into `EpisodeAudit`.

    Pure on both inputs: `without_class` and `without_scene` return new stores, and the
    full-store branches return the arguments unchanged (both are frozen, so sharing is
    safe and copying would only hide an aliasing bug rather than prevent one).
    """
    if condition is MemoryCondition.NONE:
        return SemanticStore(), EpisodicStore()
    heard = condition in (MemoryCondition.HEARD_SEEN, MemoryCondition.HEARD_UNSEEN)
    seen = condition in (MemoryCondition.HEARD_SEEN, MemoryCondition.NOT_HEARD_SEEN)
    return (
        semantic if heard else without_class(semantic, sound_class),
        episodic if seen else without_scene(episodic, scene),
    )


def _semantic_as_dict(store: SemanticStore) -> List[Dict[str, Any]]:
    return [
        {
            "sound_class": entry.sound_class,
            "room": entry.room,
            "donor_scene": entry.donor_scene,
            # `tolist()` on a float32 array yields Python floats that `repr` round-trips
            # exactly, so the reader's `float32` cast lands on the same bits. Rounding
            # here to shorten the file would move cosines in the digits a k-NN vote
            # actually turns on.
            "embedding": entry.embedding.tolist(),
        }
        for entry in store.entries
    ]


def _episodic_as_dict(store: EpisodicStore) -> List[Dict[str, Any]]:
    return [
        {
            "scene": entry.scene,
            "room": entry.room,
            "category": entry.category,
            "point": list(entry.point.as_tuple()),
        }
        for entry in store.entries
    ]


def dump_stores(
    path: str,
    semantic: SemanticStore,
    episodic: EpisodicStore,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
) -> pathlib.Path:
    """Write both stores to one JSON file, with whatever provenance the caller carries.

    ONE file rather than two, because the two are only ever meaningful together: a semantic
    store paired with the wrong tour's episodic store is a silently different experiment,
    and two paths are two chances to pair them wrongly.

    `provenance` is written verbatim under its own key and is never read back into a store.
    It exists so a store on disk can answer "which commit, which scenes, which classes,
    which encoder" a month later, which is the question `runs/` directories keep failing
    to answer.
    """
    payload = {
        "format_version": STORE_FORMAT_VERSION,
        "provenance": dict(provenance or {}),
        "semantic": _semantic_as_dict(semantic),
        "episodic": _episodic_as_dict(episodic),
    }
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename, in the target's own directory so the rename is atomic: a sweep
    # killed mid-write must leave either the old store or the new one, never half of one
    # that loads without complaint and votes with half its rows.
    scratch = target.with_name(target.name + ".partial")
    scratch.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    scratch.replace(target)
    return target


def load_stores(path: str) -> Tuple[SemanticStore, EpisodicStore, Dict[str, Any]]:
    """`(semantic, episodic, provenance)` from a file `dump_stores` wrote.

    Refuses an unknown `format_version` rather than reading what it can. A store read under
    the wrong schema answers every query confidently, and there is no symptom to notice.
    """
    payload = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    version = payload.get("format_version")
    if version != STORE_FORMAT_VERSION:
        raise MemoryBuildError(
            "{} was written at store format version {!r}; this reader is version {}. "
            "Rebuild the store rather than reading it under the wrong schema.".format(
                path, version, STORE_FORMAT_VERSION
            )
        )
    semantic = SemanticStore(
        entries=tuple(
            SemanticEntry(
                sound_class=str(row["sound_class"]),
                room=str(row["room"]),
                embedding=np.asarray(row["embedding"], dtype=np.float32),
                donor_scene=str(row["donor_scene"]),
            )
            for row in payload["semantic"]
        )
    )
    episodic = EpisodicStore(
        entries=tuple(
            EpisodicEntry(
                scene=str(row["scene"]),
                room=str(row["room"]),
                category=str(row["category"]),
                point=Xyz.from_sequence(row["point"]),
            )
            for row in payload["episodic"]
        )
    )
    return semantic, episodic, dict(payload.get("provenance", {}))
