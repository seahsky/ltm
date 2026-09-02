"""The prior pass, minus the simulator: what to hear where, and how tours become stores.

`prior_pass.walk_tour` drives the robot and calls an `observe` callback at each REACHED
stop. This module is everything around that callback that does not need habitat-sim: which
sound class belongs at a stop, how one tour's observations become a `SemanticStore`, and how
many scenes' tours merge into the one store the matrix carves its four cells from. The box
driver supplies the two things that do need a simulator -- a world to walk and an encoder to
hear with -- and nothing else.

**THIS MODULE IS BUILT AND THE TASK CANNOT CURRENTLY USE IT. Read this before wiring a
sweep to it.** The mechanism is "the class was heard at this object category on prior
tours, so look for it at this scene's instance of that category". Two facts about the tree
as it stands make that unlearnable, and both are about the TASK rather than about this code:

1. **The test episode does not place the source at the class's anchor.**
   `task.dataset.place_anomaly_source` ranks candidates by
   `(same_category, separation, category)` -- geometry only. Nothing in `task/` calls
   `vocabulary.anchor_object`; only `tools/` analysis scripts do. So in every episode this
   repo has ever run, `abl-1` included, the alarm sits at whatever object the separation
   rules picked. A store that learned "alarm at bed" has nothing to predict.
2. **At run-class granularity the association is nearly constant.** `clips.ANOMALY_CLASSES`
   is three names; `alarm` and `baby_cry` both anchor at `bed` and `glass_break` has no
   vocabulary row at all. `categories_with_a_sound` over HM3D's six goal categories returns
   exactly `{"bed": "alarm"}`. A predictor with one answer is not a predictor.

Making it work needs a decision that changes the task and re-baselines `abl-1`: anchor the
placement to the class, and widen the sounding class set from the three emergency names to
the vocabulary's seventeen. Neither is taken here. What IS here is correct and tested, and
it is the half that has no opinion about that decision.

**What the store would learn, once the decision is made.** `class_at_category` reads
`audio.vocabulary`'s `anchor_object`, which is placement ground truth and is fenced from
`agent/` and `memory/` for exactly that reason. Calling it here is the dataset builder's
licence: it decides which sound is PLAYED at which stop, the same decision the episode
builder would then make for the test. What the store holds is a CLAP embedding of that
sound as the agent actually heard it, through a real IR at a real pose, tagged with the
category it was heard at. The agent never sees the table; it sees vectors it recorded
itself, and the `not_heard` cells run the same episodes with that class filtered out, so
the delta is what the association was worth.

Pure throughout. `observation_for` takes the embedding rather than computing it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from earshot.audio.clips import ANOMALY_CLASSES, CLASS_TO_ESC50
from earshot.audio.vocabulary import CANDIDATE_VOCABULARY
from earshot.memory.store import EpisodicEntry, EpisodicStore, SemanticEntry, SemanticStore
from earshot.task.memory_build import (
    MemoryBuildError,
    episodic_from_tour,
    semantic_from_tour,
    tour_observation,
)
from earshot.task.prior_pass import TourRecord, TourStop

__all__ = [
    "anchor_of_run_class",
    "class_at_category",
    "categories_with_a_sound",
    "observation_for",
    "merge_stores",
    "stores_from_records",
]


# THE TREE HAS TWO NAMES FOR ONE SOUND AND THEY MUST NOT MEET IN A STORE.
# A run says `--anomaly-class alarm` (`clips.ANOMALY_CLASSES`); the vocabulary that knows
# where an alarm belongs calls it `clock_alarm` (an ESC-50 name). `clips.CLASS_TO_ESC50` is
# the bridge, and everything this module returns is in the RUN's namespace -- because
# `memory.without_class` carves the matrix's not-heard column by class NAME, and a store
# holding `clock_alarm` while the sweep filters `alarm` would filter nothing at all and
# hand the not-heard cells a full store. That is a four-identical-cells bug that no test
# downstream of it could see.
_ANCHOR_OF_ESC50 = {entry.name: entry.anchor_object for entry in CANDIDATE_VOCABULARY}


def anchor_of_run_class(name: str) -> Optional[str]:
    """The object category a run-level anomaly class is anchored at, or `None`.

    `None` for a class the vocabulary has no row for, which is a real state: `clips` locks
    three emergency classes and `glass_break` has no ESC-50 vocabulary entry, so it has no
    anchor and can teach no association.
    """
    return _ANCHOR_OF_ESC50.get(CLASS_TO_ESC50.get(name, name))


def class_at_category(category: str, *, classes: Optional[Sequence[str]] = None) -> Optional[str]:
    """The RUN-level sound class whose source is anchored at `category`, or `None`.

    Deterministic when several classes share an anchor: the name that sorts first wins, so
    two prior passes over the same scene teach the same association and a store is
    reproducible from its provenance rather than from vocabulary declaration order.

    `classes` restricts the answer to a run's own class set -- the sweep runs one anomaly
    class at a time, and a tour that taught the store about classes the test never plays
    would inflate `n` in the store with rows no query can reach.
    """
    allowed = set(ANOMALY_CLASSES if classes is None else classes)
    names = sorted(
        name for name in allowed if anchor_of_run_class(name) == category
    )
    return names[0] if names else None


def categories_with_a_sound(
    categories: Iterable[str], *, classes: Optional[Sequence[str]] = None
) -> Dict[str, str]:
    """`{category: class}` for every category some class is anchored at. Total, never raises.

    A category with no sound is simply absent, which is what lets the caller plan a tour over
    a scene's whole goal table and then visit only the stops that can teach anything.
    """
    found = {}
    for category in categories:
        name = class_at_category(category, classes=classes)
        if name is not None:
            found[category] = name
    return found


def observation_for(
    stop: TourStop, embedding: Sequence[float], *, classes: Optional[Sequence[str]] = None
) -> Optional[Mapping[str, Any]]:
    """One reached stop's store row, or `None` when nothing is heard at that category.

    `None` rather than a raise: a tour legitimately walks past a `bed` when the run's class
    is `alarm`, and `walk_tour` treats a `None` return as "nothing observed here". A
    malformed EMBEDDING still raises, through `tour_observation` -- the two are different
    failures and only the second is a bug.
    """
    name = class_at_category(stop.category, classes=classes)
    if name is None:
        return None
    return tour_observation(stop, sound_class=name, embedding=embedding)


def merge_stores(
    pairs: Sequence[Tuple[SemanticStore, EpisodicStore]]
) -> Tuple[SemanticStore, EpisodicStore]:
    """Concatenate several scenes' stores into the one the matrix carves from.

    Order is the caller's, preserved, so the file a sweep reads is a function of the scene
    order its provenance records. `SemanticStore.__post_init__` raises here if two scenes
    were toured under different encoders, which is the check that a merged store is worth
    querying at all -- and it fires at the merge rather than at some later cosine.
    """
    semantic: List[SemanticEntry] = []
    episodic: List[EpisodicEntry] = []
    for semantic_store, episodic_store in pairs:
        semantic.extend(semantic_store.entries)
        episodic.extend(episodic_store.entries)
    return (SemanticStore(entries=tuple(semantic)), EpisodicStore(entries=tuple(episodic)))


def stores_from_records(
    records: Sequence[TourRecord],
) -> Tuple[SemanticStore, EpisodicStore]:
    """Every tour, as one pair of stores. Raises with the scene named on a malformed record.

    The scene is added to the message because a merged store is built from many tours and
    `semantic_from_tour`'s own error names the observation index within ONE of them, which
    on its own is not enough to go and look.
    """
    pairs = []
    for record in records:
        try:
            pairs.append((semantic_from_tour(record), episodic_from_tour(record)))
        except MemoryBuildError as error:
            raise MemoryBuildError(
                "the tour of {} cannot become a store: {}".format(record.scene, error)
            ) from error
    return merge_stores(pairs)
