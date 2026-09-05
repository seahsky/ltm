"""What the memory says to do when the room has gone quiet.

ADR-0017's source stops sounding at the offset step. After that the live cue has nothing
left to say: `_probe_for` still names a place every step, but it is a 2 m hop in whatever
direction the scan/cast cycle last chose, and `abl-1` measured what that is worth --
SWS 27 of 272, so the baseline recovers the source after the silence in one episode in ten.
That is the headroom this module exists to spend.

**The mechanism, and why it is this one.** ADR-0018's `unseen_heard` cell is the one whose
mechanism has to be argued for rather than assumed: in an unseen scene `without_scene` has
emptied the episodic store, so a recalled CATEGORY has no prior-tour instance of its own to
point at. What transfers across scenes is the object CATEGORY the class was heard at -- an
alarm heard at a stove on prior tours is an alarm to look for at this scene's stove --
because the scene under test has its own instances of that category. `SemanticEntry.category`
carries it and survives `without_scene`; this module turns it into a place.

**The privilege this takes, and where it stops.** `points_by_category_for_cell` prefers the
agent's OWN recalled instance from a prior tour of the scene under test -- real navigation
history, not a privilege -- and falls back to the scene's ObjectNav annotations only for a
category the tour never reached, or in a scene it never toured at all. That fallback IS
ground truth, taken the same way `agent.detector.OracleDetector` already takes it for the
primary stop, and disclosed the same way: where recall has nothing to say, the prior answers
"where are this scene's stoves", never "where is the source". The association from class to
category is LEARNED, from real audio at real stops on prior tours, and the fenced
`audio.vocabulary` placement table is unreachable from `memory/` by construction. A run that
uses this must say so, and `RUN_DISCLOSURE` is what it says.

**This is the whole mechanism the seen axis has.** The semantic store is scene-agnostic by
design (ADR-0018: "the only store that can return anything useful in a scene the agent has
never entered"), so `stores_for_cell` never filters it by scene -- `HEARD_SEEN` and
`HEARD_UNSEEN` vote on the identical store and recall the identical category. Everything
that can make the two cells diverge has to happen AFTER the vote, in how the category
becomes a point, which is exactly what `points_by_category_for_cell` does and
`category_points` alone does not: it is scene ground truth, constant across every cell,
and a `MemoryContext` built from it directly cannot tell a seen scene from an unseen one.

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

from earshot.memory.store import EpisodicStore, SemanticStore
from earshot.types import Xyz

__all__ = [
    "PriorMiss",
    "MemoryPrior",
    "MemoryContext",
    "RUN_DISCLOSURE",
    "category_points",
    "points_by_category_for_cell",
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
    """The store answered nothing: it is empty, or the query vector is degenerate.

    NOT the `not_heard` cells' expected value, although an earlier revision said so and
    matrix-1's review caught it: `without_class` strips only the run's own class, so a
    multi-class store still votes and a `not_heard` episode receives a confident
    WRONG-class prediction — `predict_category` has no abstain. Under the matrix bank
    this value fires only for an empty store (a `NONE` cell, or a cell whose whole bank
    is one class) or a degenerate query. The heard-vs-not-heard contrast is therefore
    right-prior vs wrong-prior until a `NONE` arm runs beside them."""

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


def points_by_category_for_cell(
    dataset: object, episodic: EpisodicStore, scene: str
) -> Dict[str, Tuple[Xyz, ...]]:
    """`category_points(dataset)`, with a scene's own prior-tour recall standing in front
    of it wherever the tour has one.

    **This is what makes the seen axis a seen axis.** `stores_for_cell` filters the
    episodic store per `MemoryCondition` (`without_scene` empties it for the unseen
    cells), but `resolve_prior` votes on the SEMANTIC store alone -- and the semantic
    store is never filtered by scene, because ADR-0018 makes it scene-agnostic on
    purpose ("the only store that can return anything useful in a scene the agent has
    never entered"). So a `HEARD_SEEN` and a `HEARD_UNSEEN` episode recall the identical
    category from the identical store; without this function they would also resolve it
    through the identical `category_points(dataset)` table and be indistinguishable at
    every layer downstream of the vote. Route the filtered episodic store's OWN recalled
    points in here instead, and the two cells diverge exactly where the seen axis says
    they should: at a category the prior tour actually reached.

    For a category the tour visited, the returned points are the ones `EpisodicEntry`
    recorded -- the agent's own past location, not detection. For every other category
    (including every one when `episodic` was emptied by `without_scene`) this falls back
    to `category_points`, which is `RUN_DISCLOSURE`'s privileged stand-in: the scene's
    ObjectNav ground truth, disclosed because no detector produced it.

    Pure: builds a new mapping, mutates neither argument.
    """
    merged: Dict[str, Tuple[Xyz, ...]] = dict(category_points(dataset))
    categories = {entry.category for entry in episodic.entries if entry.scene == scene}
    for category in categories:
        points = episodic.points_for_category(scene, category)
        if points:
            merged[category] = points
    return merged


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
