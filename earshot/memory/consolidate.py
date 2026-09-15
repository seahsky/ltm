"""DREAM §III.D: which of an episode's experience is worth keeping.

    tau = {e_1, ..., e_T}                            (eq. 8)
    tau = {delta_1, ..., delta_J}                    (eq. 9)
    I_j = alpha*C_j + beta*U_j + gamma*N_j           (eq. 10)
    h_j = f_seg(delta_j)                             (eq. 11)
    N_j = 1 - max_{m in M^L} sim(h_j, m)             (eq. 12)
    D*  = {delta_j | I_j > eta}                      (eq. 13)

**THE PAPER GIVES EQ. 10, 12 AND 13 AND LEAVES EVERYTHING ELSE OPEN.** `f_seg` is named
and not defined; the segmentation of eq. 9 is called "temporally coherent" and not
specified; `C_j` and `U_j` are described in one clause each; `alpha`, `beta`, `gamma` and
`eta` have no values anywhere. Four choices had to be made here, and each one is argued in
the function that makes it rather than buried. All four are TRAINING-FREE, because nothing
else in this stack is trained and a learned `f_seg` would be the only component whose
behaviour could not be read off the code.

**NO KNOB HAS A DEFAULT.** Seven of them reach this module (`coherence`, `min_length`,
`max_length`, `alpha`, `beta`, `gamma`, `eta`) and every one is keyword-only and required,
for the reason `resolve_prior`'s `k` and `ShortTermMemory.context`'s `decay` are: a knob
with a default is a knob that reaches no artefact, so the caller passes it and the audit
records it.

**C_j DOES NOT SEE THE SOURCE.** "Contribution to successful navigation" reads most
directly as progress toward the true source, and that reading is refused here. The memory
is written BY the agent, so a consolidation that scores segments against ground truth
makes every downstream retrieval GT-privileged — the exact defect
`audio-localization-stop-closed` recorded against the old energy-STOP. `contribution`
takes an explicit `target` and the wiring passes the episode's own FINAL position, which
equals the source on a success and is honest on a failure. Passing the true source is one
argument away and must never be what an arm claiming to measure memory is wired with.

**A LEAF, STILL (ADR-0013: `"memory": ("types",)`).** I expected this module to force the
layer table open, and it does not: eq. 12 ranges over `m in M^L`, whose three levels have
three different row types, so the only thing they share is a vector in the memory
embedding space — and `novelty` therefore takes vectors and imports nothing. `store.py`
stays the one module in this package and the table is unchanged.

**`TrajectoryStep` IS NOT `StmEntry`, AND THAT IS NOT DUPLICATION.** `memory` may not
import `agent` (the edge is absent on purpose and stays absent), so this module names its
own input row, exactly as `EpisodicEntry` names the shape `task/` adapts `TourRecord.legs`
into. It is also a genuinely smaller and different record: consolidation reads the FUSED
observation rather than the two halves, needs the position rather than the full pose, and
needs the agent's source belief, which `e_t` (eq. 5) does not carry at all. The adapter
lives in `task/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from earshot.types import Xyz

__all__ = [
    "TrajectoryStep",
    "Segment",
    "ImportanceWeights",
    "segment_trajectory",
    "contribution",
    "surprise",
    "novelty",
    "importance",
    "retain",
]


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError(
            "a zero-norm vector reached f_seg. It cannot be normalised, and the "
            "alternative -- returning it unchanged -- would score 0.0 against every "
            "memory row and so read as MAXIMUM novelty (eq. 12), which is how a "
            "degenerate vector outvotes a real one"
        )
    return values / norm


def _shares(amounts: Sequence[float]) -> Tuple[float, ...]:
    """Each non-negative amount as a share of their total; all zeros when the total is 0.

    Both `C_j` and `U_j` are shares of an episode total rather than absolute quantities,
    which is what lets them sit in eq. 10 beside `N_j` with no metre-to-cosine conversion
    constant between them. A scale knob there would be an eighth number the paper does not
    have, and it would set the balance of the whole score.
    """
    positive = [max(0.0, float(amount)) for amount in amounts]
    total = float(sum(positive))
    if total == 0.0:
        return tuple(0.0 for _ in positive)
    return tuple(amount / total for amount in positive)


@dataclass(frozen=True, eq=False)
class TrajectoryStep:
    """One element of `tau` (eq. 8) as CONSOLIDATION sees it. `eq=False` for the array.

    The task-layer adapter is
    `TrajectoryStep(observation=entry.fused, position=entry.pose.position, belief=...)`
    over the episode's `ShortTermMemory` entries, and these field names must not drift
    from that source.

    `belief` is the agent's current estimate of where the source is, or `None` when it has
    none — which is the ordinary state before the anomaly fires, not a fault. It is what
    `U_j` (eq. 10) is measured on: "unexpected changes in the agent's predictions" needs a
    prediction, and the agent's committed target is the one it makes every step.
    """

    observation: np.ndarray
    position: Xyz
    belief: Optional[Xyz]

    def __post_init__(self) -> None:
        vector = np.array(self.observation, dtype=np.float32, copy=True).reshape(-1)
        if vector.size == 0:
            raise ValueError(
                "TrajectoryStep was given an empty observation; a 0-length vector cannot "
                "be a z^av and must not reach f_seg"
            )
        if not bool(np.isfinite(vector).all()):
            raise ValueError(
                "TrajectoryStep was given a non-finite observation. A NaN propagates "
                "through f_seg into every similarity in eq. 12 and turns the whole "
                "novelty comparison into a silent False"
            )
        vector.flags.writeable = False
        object.__setattr__(self, "observation", vector)


@dataclass(frozen=True, eq=False)
class Segment:
    """`delta_j` (eq. 9): a contiguous, temporally coherent run of steps, oldest first."""

    steps: Tuple[TrajectoryStep, ...]

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError(
                "a Segment with no steps is not a member of a partition of tau; "
                "`segment_trajectory` returns () for an empty trajectory rather than a "
                "tuple holding an empty segment"
            )
        widths = sorted({int(step.observation.size) for step in self.steps})
        if len(widths) > 1:
            raise ValueError(
                "this segment mixes observation widths {}; f_seg cannot average them and "
                "a segment spanning two encoders is a wiring bug, not a segment".format(
                    widths
                )
            )

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def representation(self) -> np.ndarray:
        """`h_j = f_seg(delta_j)` (eq. 11): the unit-normalised MEAN of the segment's
        fused observations.

        The paper names `f_seg` and does not define it. This is the choice, and it is the
        one that costs nothing and loses least: it adds no parameters, and it puts `h_j`
        in the same space as the observations themselves, which eq. 12 requires — `sim`
        there is taken against rows of `M^L`, so a representation in some other space
        could not be compared with them at all.

        FLAT, with no recency weighting, which is the one place this deliberately differs
        from `ShortTermMemory.context(decay=)`. A query has a "now" and leans on it; a
        segment being filed does not. Weighting the end of a segment would make `h_j`
        describe the boundary that ended it rather than the run it summarises.
        """
        stacked = np.stack([step.observation for step in self.steps])
        return _unit(stacked.mean(axis=0)).astype(np.float32)


@dataclass(frozen=True)
class ImportanceWeights:
    """`alpha`, `beta`, `gamma` of eq. 10. All three required; the paper gives no values.

    Rejected: any negative weight, and an all-zero triple. A negative weight inverts the
    meaning of its term rather than reducing it — a negative `gamma` retains what memory
    already holds and discards what is new, which is eq. 12 run backwards. An all-zero
    triple makes every `I_j` exactly 0, so eq. 13 retains everything or nothing depending
    on the sign of `eta` alone, and the artefact would record a consolidation that never
    consolidated.
    """

    alpha: float
    beta: float
    gamma: float

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "gamma"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(
                    "ImportanceWeights.{} must be >= 0, got {}. A negative weight runs "
                    "its term backwards rather than down-weighting it".format(name, value)
                )
            object.__setattr__(self, name, value)
        if self.alpha == 0.0 and self.beta == 0.0 and self.gamma == 0.0:
            raise ValueError(
                "ImportanceWeights(0, 0, 0) makes every I_j exactly 0, so eq. 13 keeps "
                "every segment or no segment on the sign of eta alone and nothing on "
                "disk would say consolidation had stopped discriminating"
            )


def segment_trajectory(
    steps: Sequence[TrajectoryStep],
    *,
    coherence: float,
    min_length: int,
    max_length: int,
) -> Tuple[Segment, ...]:
    """`tau = {delta_1, ..., delta_J}` (eq. 9). A partition, in order, covering every step.

    **THE CHOICE.** The paper asks for "temporally coherent segments" and says no more.
    A segment is cut when the next observation falls below `coherence` cosine of the
    segment's own running mean — its `h_j` so far — so within a segment every step
    resembles the thing the segment will be filed as. That is what coherence means here,
    and it is measurable from the trajectory alone.

    Against the RUNNING MEAN rather than the previous step, which is the difference that
    matters: a slow walk down a corridor has every consecutive pair at cosine 0.99 and its
    two ends at 0.5, so a pairwise rule never cuts it and files one segment whose `h_j`
    describes neither end. `tests/mac/test_memory_consolidate.py` holds that arm.

    `min_length` stops one outlier frame from splitting a run into slivers; `max_length`
    bounds a corridor of near-identical frames, which no similarity rule would ever cut.
    Both are required.

    The final segment may be shorter than `min_length`. A partition has to cover `tau`,
    and folding a short tail into the segment before it would silently breach
    `max_length` — the one bound that keeps `h_j` a summary rather than an average of the
    whole episode.
    """
    if not 0.0 < float(coherence) <= 1.0:
        raise ValueError(
            "coherence is a cosine threshold and must be in (0, 1]; got {}. At or below "
            "0 a segment never ends on similarity and only max_length cuts it".format(
                coherence
            )
        )
    if int(min_length) < 1:
        raise ValueError("min_length must be >= 1, got {}".format(min_length))
    if int(max_length) < int(min_length):
        raise ValueError(
            "max_length {} is below min_length {}; no cut rule can satisfy both".format(
                max_length, min_length
            )
        )
    if not steps:
        return ()

    segments: List[List[TrajectoryStep]] = []
    current: List[TrajectoryStep] = [steps[0]]
    running = _unit(steps[0].observation)
    for step in steps[1:]:
        if step.observation.size != running.size:
            raise ValueError(
                "step observations change width mid-trajectory ({} then {}); tau spans "
                "two encoders and cannot be segmented".format(
                    running.size, step.observation.size
                )
            )
        at_cap = len(current) >= int(max_length)
        incoherent = (
            len(current) >= int(min_length)
            and float(np.dot(_unit(step.observation), running)) < float(coherence)
        )
        if at_cap or incoherent:
            segments.append(current)
            current = [step]
            running = _unit(step.observation)
            continue
        current.append(step)
        running = _unit(np.stack([s.observation for s in current]).mean(axis=0))
    segments.append(current)
    return tuple(Segment(steps=tuple(group)) for group in segments)


def _transitions(segments: Sequence[Segment]) -> List[List[Tuple[TrajectoryStep, TrajectoryStep]]]:
    """Every consecutive (earlier, later) pair, bucketed into the segment holding `later`.

    ONE tiling rule, applied to both `C_j` and `U_j`: a transition belongs to the segment
    containing its later step. Every transition of the episode lands in exactly one
    bucket, so both quantities sum over segments to the episode's own total and each share
    is a share of something real. The first step of the episode has no predecessor and so
    contributes to neither.
    """
    buckets: List[List[Tuple[TrajectoryStep, TrajectoryStep]]] = []
    previous: Optional[TrajectoryStep] = None
    for segment in segments:
        pairs: List[Tuple[TrajectoryStep, TrajectoryStep]] = []
        for step in segment.steps:
            if previous is not None:
                pairs.append((previous, step))
            previous = step
        buckets.append(pairs)
    return buckets


def contribution(segments: Sequence[Segment], *, target: Xyz) -> Tuple[float, ...]:
    """`C_j` (eq. 10): each segment's share of the distance the episode actually closed.

    **`target` IS NOT THE SOURCE.** See the module docstring: the wiring passes the
    episode's own final position, so nothing ground-truth reaches a memory the agent
    wrote. This function does not know which it was given, which is why the caller has to
    say and the audit records it.

    Horizontal distance, y ignored (`types.Xyz.horizontal_distance_to`) — the tree's
    convention everywhere, and a 3D distance here would charge a segment for a staircase.

    **Backward motion scores 0, not a negative.** A segment that walked away contributed
    nothing to arriving, and scoring it negative would make `alpha` a penalty scale as
    well as a weight, which eq. 10 gives no room for. A detour that was worth keeping is
    exactly what `U_j` and `N_j` are in the sum to catch.

    **No success gate.** "Contribution to SUCCESSFUL navigation" could be read as zero for
    every segment of a failed episode; that reading is refused. `abl-2` reached the source
    in 35.8% of episodes, so gating would discard roughly two thirds of all experience
    before `N_j` ever saw it, and a failed episode still contains the part of the route
    that worked. The outcome is recorded where the paper puts it — `y_i` on the `M^E` row
    (eq. 15) — not folded into `C_j`.
    """
    gaps = [
        sum(
            earlier.position.horizontal_distance_to(target)
            - later.position.horizontal_distance_to(target)
            for earlier, later in pairs
        )
        for pairs in _transitions(segments)
    ]
    return _shares(gaps)


def surprise(segments: Sequence[Segment]) -> Tuple[float, ...]:
    """`U_j` (eq. 10): each segment's share of how far the agent's belief moved.

    The paper's clause is "unexpected changes in the agent's predictions". The agent's
    prediction is where it currently believes the source to be, so the change is how far
    that estimate travelled between steps, and the share is over the episode's total.

    **A belief appearing is not a belief changing.** `None -> Xyz` is the anomaly firing
    and the agent forming its first estimate; scoring that as a large revision would put
    the maximum surprise of every episode on whichever segment happened to contain the
    onset, in every episode, which measures the task's structure and not the agent's. The
    same holds for `Xyz -> None`. Only two present beliefs make a transition.
    """
    revisions = [
        sum(
            earlier.belief.horizontal_distance_to(later.belief)
            for earlier, later in pairs
            if earlier.belief is not None and later.belief is not None
        )
        for pairs in _transitions(segments)
    ]
    return _shares(revisions)


def novelty(representation: np.ndarray, memory: Sequence[np.ndarray]) -> float:
    """`N_j = 1 - max_{m in M^L} sim(h_j, m)` (eq. 12), `sim` being cosine.

    `memory` is a sequence of vectors rather than of rows because eq. 12 ranges over all
    of `M^L` (eq. 14) — three levels with three different row types whose only common
    surface is a vector in the memory embedding space. That is also what keeps this module
    a leaf.

    **An EMPTY memory scores 1.0, not 2.0.** `max` over nothing is undefined and the paper
    does not say. 1.0 is the value an orthogonal memory would give, and it is the one that
    does not break the mechanism: with cosine in [-1, 1] the formula's range is [0, 2], so
    answering 2.0 would let the very first episode's segments outscore every later
    segment on novelty purely because the store was empty, fill `M^E` with one episode,
    and then threshold out everything that followed.

    Returned unclamped otherwise, exactly as written — a negative similarity really is
    more novel than an orthogonal one, and clamping to [0, 1] would be a fifth undocumented
    choice on top of the four this module already makes.
    """
    query = _unit(representation)
    best: Optional[float] = None
    for row in memory:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size != query.size:
            raise ValueError(
                "a memory row of width {} was compared against an h_j of width {}; "
                "eq. 12's sim is not defined across two embedding spaces".format(
                    values.size, query.size
                )
            )
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            # A zero-norm row has no direction to be similar to. Skipping it is not the
            # same as it scoring 0.0, which would cap N_j at 1.0 for every query the
            # moment one degenerate row entered the store. `store.py::_vote` skips for
            # the same reason.
            continue
        similarity = float(np.dot(query, values / norm))
        best = similarity if best is None else max(best, similarity)
    if best is None:
        return 1.0
    return 1.0 - best


def importance(
    segments: Sequence[Segment],
    *,
    target: Xyz,
    memory: Sequence[np.ndarray],
    weights: ImportanceWeights,
) -> Tuple[float, ...]:
    """`I_j = alpha*C_j + beta*U_j + gamma*N_j` (eq. 10), one score per segment, in order.

    Separate from `retain` on purpose: the scores are what the audit records and what a
    later sweep needs in order to say whether `eta` was set anywhere sensible. A `retain`
    that scored internally would discard them at the moment they became evidence.
    """
    contributions = contribution(segments, target=target)
    surprises = surprise(segments)
    return tuple(
        weights.alpha * c
        + weights.beta * u
        + weights.gamma * novelty(segment.representation, memory)
        for segment, c, u in zip(segments, contributions, surprises)
    )


def retain(
    segments: Sequence[Segment], scores: Sequence[float], *, eta: float
) -> Tuple[Segment, ...]:
    """`D* = {delta_j | I_j > eta}` (eq. 13). Strictly greater, as written.

    Pure and total: it filters, it does not re-score, and a `segments`/`scores` length
    mismatch raises rather than zipping to the shorter of the two — a silently truncated
    `D*` is a consolidation that dropped its tail with nothing saying so.
    """
    if len(segments) != len(scores):
        raise ValueError(
            "retain got {} segments and {} scores; a mismatch means the scores are not "
            "these segments' scores".format(len(segments), len(scores))
        )
    return tuple(
        segment for segment, score in zip(segments, scores) if float(score) > float(eta)
    )
