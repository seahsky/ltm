"""What the memory says to do when the room has gone quiet.

ADR-0017's source stops sounding at the offset step. After that the live cue has nothing
left to say: `_probe_for` still names a place every step, but it is a 2 m hop in whatever
direction the scan/cast cycle last chose, and `abl-1` measured what that is worth --
SWS 27 of 272, so the baseline recovers the source after the silence in one episode in ten.
That is the headroom this module exists to spend.

**The mechanism, and why it is this one.** ADR-0018's `unseen_heard` cell is the one whose
mechanism has to be argued for rather than assumed: in an unseen scene `without_scene` has
emptied the episodic store, so `points_for_room` returns `()` and a recalled ROOM points at
no coordinate the agent can walk to. What transfers across scenes is the object CATEGORY
the class was heard at -- an alarm heard at a stove on prior tours is an alarm to look for
at this scene's stove -- because the scene under test has its own instances of that
category. `SemanticEntry.category` carries it and survives `without_scene`; this module
turns it into a place.

**The privilege this takes, stated rather than discovered later.** The instances come from
the scene's own ObjectNav annotations, which are ground truth. That is the same privilege
`agent.detector.OracleDetector` already takes for the primary stop, and it is disclosed the
same way: the prior answers "where are this scene's stoves", never "where is the source".
The association from class to category is LEARNED, from real audio at real stops on prior
tours, and the fenced `audio.vocabulary` placement table is unreachable from `memory/` by
construction. A run that uses this must say so, and `RUN_DISCLOSURE` is what it says.

**Every failure is named, and none of them is a zero.** A prior can fail three different
ways and they are three different facts about the method: the store answered nothing
(`NO_PREDICTION`), it named a category this scene has no instance of (`CATEGORY_ABSENT`),
or the instances exist and none has a navmesh route (`UNREACHABLE`). Collapsing them into
`None` would make "the memory is empty" and "the memory was right but the house has no
stove" the same measurement, and this repo has paid for that shape twice.

Pure throughout. Nothing here renders, steps, or reads the simulator: the caller supplies
the scene's category table and a distance function, so every path below is decidable on a
machine with no habitat-sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from earshot.memory.store import SemanticStore
from earshot.types import Xyz

__all__ = [
    "PriorMiss",
    "MemoryPrior",
    "MemoryContext",
    "RUN_DISCLOSURE",
    "category_points",
    "resolve_prior",
]

# §8's required disclosure for any run whose memory arm is live, printed beside the oracle
# STOP's. A reader must not have to infer from a flag that the target instances were
# annotated rather than perceived.
RUN_DISCLOSURE = (
    "the memory prior located its recalled category from the scene's ObjectNav "
    "annotations, so instance detection is not exercised; the class-to-category "
    "association is learned from the prior pass and is not annotated"
)


class PriorMiss(Enum):
    """Why a prior named no place. Each value is a different fact about the method."""

    NO_PREDICTION = "no_prediction"
    """The store answered nothing: it is empty, or the query vector is degenerate. This is
    the `not_heard` cells' expected value and must never read as a wrong prediction."""

    CATEGORY_ABSENT = "category_absent"
    """A category was predicted and this scene has no instance of it. The memory answered;
    the house did not have the thing. A generalization failure of a different kind from a
    wrong recall, and the two must be counted apart."""

    UNREACHABLE = "unreachable"
    """Instances exist and the navmesh routes to none of them. The same disconnected-island
    fact `window_report` already reports as ABSENT rather than as a distance of 0."""


@dataclass(frozen=True)
class MemoryPrior:
    """A place the memory named, with everything needed to audit the naming.

    `confidence` is the k-NN mean cosine the vote won by, carried so the audit can separate
    a confident recall that was wrong from a marginal one that happened to be right.
    `n_instances` is how many of the category the scene held, because a prior that picked
    the nearest of one is a different claim from the nearest of nine.
    """

    category: str
    confidence: float
    target: Xyz
    distance_m: float
    n_instances: int

    def as_metrics(self) -> Dict[str, float]:
        """The numeric half, for the audit's `Mapping[str, float]`. The category is a
        string and goes in the typed field, never here."""
        return {
            "memory_prior_confidence": float(self.confidence),
            "memory_prior_distance_m": float(self.distance_m),
            "memory_prior_instances": float(self.n_instances),
        }


@dataclass(frozen=True)
class MemoryContext:
    """One matrix cell, resolved: the stores it was carved to, and the scene it runs in.

    Built by the sweep driver, handed to `run_episode`, and never assembled inside the run
    -- which is the same rule `MemoryCondition`'s docstring already states. The cell is
    realised by WHICH stores are in here (`without_class` / `without_scene` applied or not),
    so a run cannot select its own cell and the audit's `memory_condition` is a record of
    what the caller did rather than of a branch the runner took.

    `condition is MemoryCondition.NONE` with empty stores is the historic behaviour and is
    what every non-matrix run passes by passing nothing at all: `run_episode`'s parameter
    defaults to `None` and the steering below is unreachable without one.

    `k` lives here rather than on `RunConfig` for the reason `MemoryCondition` is not on it
    either: the config cannot act on a store it has no edge to, and ADR-0013 does not widen
    for a value only the caller uses.
    """

    condition: object  # memory.store.MemoryCondition; typed loosely to avoid a cycle
    semantic: SemanticStore
    points_by_category: Mapping[str, Sequence[Xyz]]
    k: int = 5

    def __post_init__(self) -> None:
        if int(self.k) < 1:
            raise ValueError(
                "MemoryContext needs k >= 1, got {} -- a vote over no neighbours is not "
                "a weaker prediction, it is no prediction".format(self.k)
            )

    @property
    def is_live(self) -> bool:
        """True when this context can name a place at all: a store with rows AND a scene
        with objects. Either half empty is a cell that will only ever return a miss, and
        saying so here keeps the runner from encoding that rule twice."""
        return bool(len(self.semantic)) and bool(self.points_by_category)


def category_points(dataset: object) -> Dict[str, Tuple[Xyz, ...]]:
    """Every goal position in an `episodes.EpisodeDataset`, grouped by object category.

    Duck-typed on purpose: this takes anything with `.episodes`, each with
    `.object_category` and `.goals`, each goal with `.position`. That is the shape
    `task.episodes` publishes, and typing it structurally keeps this module testable
    without building a real dataset.

    De-duplicated by rounded coordinate, because `goals_by_category` hoists one copy per
    category across every episode of the scene (`object_nav_dataset.py:38-58`) and the same
    stove would otherwise be counted nine times and skew `n_instances`.
    """
    grouped: Dict[str, Dict[Tuple[float, float, float], Xyz]] = {}
    for episode in getattr(dataset, "episodes", ()):
        category = str(episode.object_category)
        bucket = grouped.setdefault(category, {})
        for goal in episode.goals:
            point = goal.position
            key = (round(point.x, 4), round(point.y, 4), round(point.z, 4))
            bucket.setdefault(key, point)
    return {
        category: tuple(bucket[key] for key in sorted(bucket))
        for category, bucket in grouped.items()
    }


def resolve_prior(
    store: SemanticStore,
    embedding: np.ndarray,
    *,
    k: int,
    points_by_category: Mapping[str, Sequence[Xyz]],
    distance_to: Callable[[Xyz], Optional[float]],
) -> Tuple[Optional[MemoryPrior], Optional[PriorMiss]]:
    """The recalled category's nearest reachable instance, or the named reason there is none.

    Exactly one of the two returned values is ever set. `k` has no default for the same
    reason `predict_category`'s has none: a knob with a default is a knob that reaches no
    artefact, so the caller passes it and the audit records it.

    `distance_to` returns `None` for a point with no route, which is how an instance on a
    disconnected island is excluded rather than being ranked at some large number. When it
    excludes every instance the answer is `UNREACHABLE`, which is not the same fact as the
    scene having no such object.

    Pure: reads `store` and `points_by_category`, mutates neither, and calls `distance_to`
    once per candidate instance.
    """
    predicted = store.predict_category(embedding, k=k)
    if predicted is None:
        return (None, PriorMiss.NO_PREDICTION)
    category, confidence = predicted

    instances = tuple(points_by_category.get(category, ()))
    if not instances:
        return (None, PriorMiss.CATEGORY_ABSENT)

    routed = [
        (distance, point)
        for distance, point in ((distance_to(point), point) for point in instances)
        if distance is not None
    ]
    if not routed:
        return (None, PriorMiss.UNREACHABLE)

    # Ties broken by coordinate so the choice is reproducible from the audit rather than
    # from whichever order the annotations happened to be written in.
    routed.sort(key=lambda item: (item[0], item[1].x, item[1].y, item[1].z))
    distance, target = routed[0]
    return (
        MemoryPrior(
            category=category,
            confidence=float(confidence),
            target=target,
            distance_m=float(distance),
            n_instances=len(instances),
        ),
        None,
    )
