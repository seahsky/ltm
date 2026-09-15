"""DREAM §III.F: the query, the three retrievals, and the omega the paper turns on.

    q_t   = f_q(z^av_t, M^S_t)                                  (eq. 19)
    K^E_t = Retrieve(q_t, M^E, k_E)                             (eq. 20)
    K^P_t = Retrieve(q_t, M^P, k_P)                             (eq. 21)
    K^K_t = Retrieve(q_t, M^K, k_K)                             (eq. 22)
    omega_t = softmax[f_omega(q_t, K^E_t, K^P_t, K^K_t)]        (eq. 23)
    K_t   = w^E K^E_t + w^P K^P_t + w^K K^K_t                   (eq. 24)

**THIS IS THE MECHANISM THE PAPER'S CENTRAL CLAIM IS ABOUT.** §III.F states it in prose
and it is the one part of DREAM specified precisely enough to test: "Under familiar
navigation conditions, concrete episodic experience can provide direct guidance. As the
current condition becomes less similar to previous experience, higher-level pattern and
semantic-spatial memories provide more transferable priors." `TestTheOmegaShift` in the
mac suite asserts exactly that sentence, because it is the hypothesis and not a comment.

**EQ. 24 DOES NOT CLOSE AS WRITTEN, AND THIS IS THE SECOND PLACE IT DOES NOT.** PR #111
found that eq. 12's single `max` over `M^L` cannot range over all three levels, because
`M^E`/`M^P` are keyed by the fused `z^av` and `M^K` by a CLAP vector of half that width.
Eq. 24 is the same fact in a worse place: it adds `K^E`, `K^P` and `K^K` as if they were
three vectors in one space, and they are not. `K^K` is not even a vector — retrieving from
a sound-to-object table returns an object CONCEPT, which is a different KIND of guidance
from "here is a past experience that resembled this one".

So `K_t` is a weighted BUNDLE here, not a sum:

  * `fused_context()` is eq. 24 exactly, over the two levels that do share a space —
    `w^E K^E + w^P K^P`, renormalised;
  * `knowledge` stays a `(category, score)` beside it, carrying `w^K`.

**THE PAPER'S OWN eq. 26 AGREES.** §III.G defines `S_mem` as measuring "consistency with
retrieved successful experiences and navigation patterns" — `M^E` and `M^P`, and it does
not name `M^K`. The knowledge level's job in this tree is the sound-to-object prior that
says WHAT to look for, which is `store.SemanticStore.predict_category` and is already how
`resolve_prior` works. Two different kinds of output, used at two different points. The
sum in eq. 24 was compression.

**f_q IS A BLEND, AND THE STM ARRIVES AS A VECTOR.** `memory` may not import `agent`, so
`query` takes the context ARRAY that `agent.stm.ShortTermMemory.context(decay=)` returns,
never the memory object — the same shape `consolidate` takes vectors rather than
`StmEntry`s. `None` there means the STM is empty, which is the ordinary state at t=0, and
`q_t` is then `z^av_t` alone rather than a blend with nothing.

**M^K IS QUERIED WITH THE AUDIO HALF OF q_t, AND THE STORE'S OWN `dim` SAYS WHERE IT
STARTS.** This is what PR #109's `f_fuse` kept both halves intact for. The split is the
LAST `knowledge.dim` components, which rests on `fuse` concatenating `[visual, audio]` in
that order — pinned by `tests/mac/test_agent_stm.py::
test_both_halves_survive_intact_and_in_order`, and tied to this module by
`TestTheKnowledgeQueryIsTheAudioHalf`. Reading the width off the store rather than taking
it as an argument means a store built under a different encoder raises inside
`predict_category` instead of being silently sliced to the wrong place.

**NO KNOB HAS A DEFAULT.** `present_weight`, `k_experience`, `k_pattern`, `k_knowledge`
and `temperature`, all keyword-only and required. `temperature` is the ablation of the
whole hypothesis: at a high enough value `omega_t` is uniform, which is DREAM with the
dynamic retrieval switched off and every other part intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from earshot.memory.longterm import ExperienceEntry, LongTermMemory, Pattern

__all__ = [
    "ExperienceHit",
    "PatternHit",
    "KnowledgeHit",
    "Weights",
    "RetrievedContext",
    "query",
    "retrieve",
]


def _unit(vector: np.ndarray, where: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError(
            "{} produced a zero-norm vector. It has no direction, so every cosine "
            "against it is 0.0 and every level of M^L would tie -- which is a retrieval "
            "that looks like it happened".format(where)
        )
    return values / norm


def query(
    observation: np.ndarray,
    context: Optional[np.ndarray],
    *,
    present_weight: float,
) -> np.ndarray:
    """``q_t = f_q(z^av_t, M^S_t)`` (eq. 19): the present blended with the recent past.

    The paper names `f_q` and does not define it. `present_weight` in [0, 1] is the mass
    on `z^av_t`; the rest goes to the STM context, which is itself the recency-weighted
    mean of the window (`agent.stm.ShortTermMemory.context`). At 1.0 the query is the
    current observation alone and `M^S` is ablated out of eq. 19 with nothing else
    changing, which is why the knob is worth having.

    `context` is `None` on an empty STM — the ordinary state at the first step of every
    episode — and `q_t` is then `z^av_t` alone. Blending with a zero vector instead would
    silently halve the query's own contribution for the first `K` steps of every episode.

    Both halves are unit-normalised before the blend, for the reason `f_fuse` normalises
    its halves: otherwise the weight that decides the balance is not the only thing
    deciding it.
    """
    weight = float(present_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError(
            "present_weight must be in [0, 1]; got {}. Outside it the blend is an "
            "extrapolation away from both the observation and the context, which is "
            "neither of the two things eq. 19 names".format(weight)
        )
    present = _unit(observation, "f_q's observation half")
    if context is None:
        return present
    past = _unit(context, "f_q's STM context half")
    blended = weight * present + (1.0 - weight) * past
    return _unit(blended, "f_q").astype(np.float32)


@dataclass(frozen=True, eq=False)
class ExperienceHit:
    """One row of `K^E_t` (eq. 20): a past experience and how much it resembles `q_t`."""

    entry: ExperienceEntry
    score: float


@dataclass(frozen=True, eq=False)
class PatternHit:
    """One row of `K^P_t` (eq. 21): an abstracted pattern and its similarity to `q_t`."""

    pattern: Pattern
    score: float


@dataclass(frozen=True)
class KnowledgeHit:
    """`K^K_t` (eq. 22): the object concept the sound-to-object table votes for.

    A CONCEPT and not a vector, which is why eq. 24's sum cannot include it. `score` is
    the vote's mean cosine, exactly what `SemanticStore.predict_category` returns.
    """

    category: str
    score: float


@dataclass(frozen=True)
class Weights:
    """`omega_t = [w^E, w^P, w^K]` (eq. 23). A softmax, so these sum to 1."""

    experience: float
    pattern: float
    knowledge: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.experience, self.pattern, self.knowledge)


@dataclass(frozen=True)
class RetrievedContext:
    """`K_t` (eq. 24), as a weighted bundle rather than a sum. See the module docstring.

    `weights` is `None` when NO level retrieved anything — an empty `M^L`, or a query
    every level refused. That is the honest state ("memory said nothing") and it is not
    the same as three zero weights or a uniform split over nothing; a caller that blends
    on `None` would be blending with a memory that does not exist.
    """

    experience: Tuple[ExperienceHit, ...]
    pattern: Tuple[PatternHit, ...]
    knowledge: Optional[KnowledgeHit]
    weights: Optional[Weights]

    def __bool__(self) -> bool:
        return self.weights is not None

    def fused_context(self) -> Optional[np.ndarray]:
        """Eq. 24 over the two levels that share `q_t`'s space: `w^E K^E + w^P K^P`.

        Each level is summarised by the mean of its retrieved keys before weighting, so a
        level that returned three rows does not outweigh one that returned one by count
        alone — the weight is `omega_t`'s job and nothing else's.

        `None` when neither of those two levels retrieved anything, even if `M^K` did: the
        knowledge level contributes a concept, and there is no vector to return.
        """
        parts: List[np.ndarray] = []
        if self.weights is None:
            return None
        if self.experience:
            parts.append(
                self.weights.experience
                * np.stack([hit.entry.context for hit in self.experience]).mean(axis=0)
            )
        if self.pattern:
            parts.append(
                self.weights.pattern
                * np.stack(
                    [hit.pattern.strategy.signature for hit in self.pattern]
                ).mean(axis=0)
            )
        if not parts:
            return None
        total = np.sum(np.stack(parts), axis=0)
        if float(np.linalg.norm(total)) == 0.0:
            return None
        return _unit(total, "eq. 24").astype(np.float32)


def _top_k(
    query_vector: np.ndarray, keys: Sequence[np.ndarray], k: int, where: str
) -> Tuple[Tuple[int, float], ...]:
    """The `Retrieve` of eq. 20-22: the `k` nearest keys by cosine, best first.

    The paper names `Retrieve` and does not define it. Cosine top-k, which is the metric
    `store.SemanticStore._vote` already uses and the one `f_fuse` unit-normalises for.

    Zero-norm keys are SKIPPED, not scored 0.0 — `store.py::_vote` and
    `consolidate.novelty` both do this, and for the same reason: a row with no direction
    is not evidence, and scoring it 0.0 lets it outrank a genuinely dissimilar row.

    Ties break on index ascending, which is store order and is reproducible from a log.
    """
    if int(k) < 1:
        raise ValueError(
            "{} needs k >= 1, got {}; a retrieval of nothing is expressed by an empty "
            "store, not by asking for zero rows".format(where, k)
        )
    scored: List[Tuple[int, float]] = []
    for index, key in enumerate(keys):
        values = np.asarray(key, dtype=np.float32).reshape(-1)
        if values.size != query_vector.size:
            raise ValueError(
                "{}: a key of width {} was scored against a q_t of width {}. Retrieval "
                "across two embedding spaces is not defined, and a silent slice here is "
                "how a wrong encoder reaches a memory unnoticed".format(
                    where, values.size, query_vector.size
                )
            )
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            continue
        scored.append((index, float(np.dot(query_vector, values / norm))))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return tuple(scored[: int(k)])


def _softmax(logits: Sequence[Optional[float]], temperature: float) -> Tuple[float, ...]:
    """`omega_t` (eq. 23). `None` for a level that retrieved nothing gets weight 0.

    A level with no rows is EXCLUDED rather than given a logit of 0.0. A logit of 0.0 is a
    real number and takes real softmax mass, so an empty `M^P` would quietly draw weight
    away from the levels that actually answered — and `fused_context` would then be
    scaled by a weight belonging to nothing.
    """
    present = [(index, value) for index, value in enumerate(logits) if value is not None]
    if not present:
        return tuple(0.0 for _ in logits)
    scaled = np.array([value for _index, value in present], dtype=np.float64) / temperature
    shifted = np.exp(scaled - scaled.max())
    normalised = shifted / shifted.sum()
    out = [0.0] * len(logits)
    for (index, _value), weight in zip(present, normalised):
        out[index] = float(weight)
    return tuple(out)


def retrieve(
    query_vector: np.ndarray,
    memory: LongTermMemory,
    *,
    k_experience: int,
    k_pattern: int,
    k_knowledge: int,
    temperature: float,
) -> RetrievedContext:
    """Eq. 20-24 in one pass: three retrievals, `omega_t`, and the bundle.

    **f_omega IS THE PER-LEVEL BEST SIMILARITY.** The paper gives `f_omega` as a learned
    function and then describes exactly what it must do: episodic guidance when the
    condition is familiar, higher levels when it is not. The logit for each level is that
    level's own best retrieval score, so a level that matched well is weighted up and a
    level that did not is weighted down — which is that sentence, with no parameters and
    nothing trained. Eq. 23 passes `q_t` to `f_omega` as well; the scores ARE functions of
    `q_t`, so nothing is lost by not passing it again.

    Whether weight actually climbs the hierarchy as the episodic match degrades is then a
    MEASUREMENT rather than a construction, and it is the paper's central claim. The mac
    suite asserts the mechanism; a sweep decides the magnitude.

    `temperature` scales the logits. Above about 10 the weights are uniform to three
    places, which is the ablation arm — DREAM with dynamic retrieval off and everything
    else intact — and below about 0.05 it is an argmax over the three levels. It has no
    default, like every other knob here.
    """
    if float(temperature) <= 0.0:
        raise ValueError(
            "temperature must be > 0, got {}. At 0 the softmax is a division by zero and "
            "below it the weights INVERT -- the level that matched worst would be "
            "weighted highest, which is eq. 23 run backwards".format(temperature)
        )
    unit_query = _unit(query_vector, "retrieve's q_t")

    experience_hits = tuple(
        ExperienceHit(entry=memory.experience.entries[index], score=score)
        for index, score in _top_k(
            unit_query, memory.experience.keys, k_experience, "eq. 20 (M^E)"
        )
    )
    pattern_hits = tuple(
        PatternHit(pattern=memory.pattern.patterns[index], score=score)
        for index, score in _top_k(
            unit_query, memory.pattern.keys, k_pattern, "eq. 21 (M^P)"
        )
    )

    knowledge_hit: Optional[KnowledgeHit] = None
    width = memory.knowledge.dim
    if int(k_knowledge) < 1:
        raise ValueError(
            "eq. 22 (M^K) needs k >= 1, got {}".format(k_knowledge)
        )
    if width is not None:
        if unit_query.size <= width:
            raise ValueError(
                "q_t is {} wide and M^K's keys are {}; the audio half of a fused query "
                "must be strictly narrower than the whole of it, so these two were not "
                "built by the same pair of encoders".format(unit_query.size, width)
            )
        # The LAST `width` components. `f_fuse` concatenates [visual, audio] in that
        # order and `tests/mac/test_agent_stm.py` pins it; this module's own
        # `TestTheKnowledgeQueryIsTheAudioHalf` ties the two together so the convention
        # cannot drift on one side alone.
        voted = memory.knowledge.predict_category(unit_query[-width:], k=int(k_knowledge))
        if voted is not None:
            knowledge_hit = KnowledgeHit(category=voted[0], score=float(voted[1]))

    logits: Tuple[Optional[float], ...] = (
        experience_hits[0].score if experience_hits else None,
        pattern_hits[0].score if pattern_hits else None,
        knowledge_hit.score if knowledge_hit is not None else None,
    )
    if all(value is None for value in logits):
        return RetrievedContext(
            experience=(), pattern=(), knowledge=None, weights=None
        )
    shares = _softmax(logits, float(temperature))
    return RetrievedContext(
        experience=experience_hits,
        pattern=pattern_hits,
        knowledge=knowledge_hit,
        weights=Weights(
            experience=shares[0], pattern=shares[1], knowledge=shares[2]
        ),
    )
