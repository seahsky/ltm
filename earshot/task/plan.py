"""DREAM §III.G: scoring a candidate waypoint against memory, and the override that holds.

    P_t = {P^1, ..., P^L} = f_plan(o_t, M^S_t, K_t)                   (eq. 25)
    Score(P^i) = l1 S_plan(P^i) + l2 S_mem(P^i) + l3 S_feas(P^i)      (eq. 26)
    P*_t = argmax_i Score(P^i_t)                                      (eq. 27)

**THIS LIVES IN `task/` BECAUSE IT IS A JOIN, AND `store.py` ALREADY SAID SO.** A score
here needs a `Candidate` (from `agent/`) and a `RetrievedContext` (from `memory/`), and
neither package may import the other -- `LAYER_IMPORTS["agent"]` has no `memory` and
`LAYER_IMPORTS["memory"]` has no `agent`, deliberately. `store.py`'s docstring states the
resolution in advance: "The candidate this store feeds into the scorer is built in
`task/`, not here." So does this.

**TWO OF EQ. 26's THREE TERMS ALREADY EXIST AND ARE NOT REBUILT.**

`S_plan` is "the planner confidence", and `agent.scorer.score_candidate` is exactly that:
the geometric blend `0.5 raw + 0.3 bearing + 0.2 distance`, clipped to [0, 1], frozen by
ADR-0008. `S_feas` "evaluates the feasibility of the plan under the current environmental
state", and `reachability.reachable_pool` has already asked the navmesh and written
`Candidate.geodesic_m`; the feasibility of a candidate is how directly the agent can
actually reach it. Only `S_mem` is new.

`f_plan` (eq. 25) is not rebuilt either: the proposers ARE the plan generator, and
`agent.scorer`'s own docstring names the seam a memory-derived proposer would use
(`Candidate.source`, "a memory proposer would add one branch here"). Generating candidates
FROM memory is a separate change and is not this one -- eq. 25 passes `K_t` to `f_plan`,
and here `K_t` reaches the scoring and not the generation. That is a narrowing of the
paper and it is stated rather than hidden.

**THE INVESTIGATE DIVERT STAYS A STRUCTURAL OVERRIDE, AND MEMORY IS WHY IT HAS TO.**
`agent/scorer.py` documents a regression at length: the old tree gave the divert
`score = 1.0` and called it maximal, which held only while a rerank blended a memory term
on top. With memory dropped the physics score WAS the final score, a maximal frontier tied
the divert exactly, and the emission-order tie-break handed the pick away -- making the
anomaly interrupt advisory. `tests/mac/test_agent_scorer.py` caught it and the fix was to
sort the divert first by RANK rather than by arithmetic.

Eq. 26 puts a memory term back into that blend. A divert whose `S_mem` is low would lose
to a frontier the memory likes, which is the same defect wearing DREAM's notation. So
`pick_plan` carries `scorer._rank`'s shape: **the divert sorts ahead of everything
regardless of `Score`**, and `test_the_divert_survives_a_memory_that_hates_it` is the arm.

**`S_mem` IS GEOMETRIC, AND IT IS WEAK. HERE IS EXACTLY WHY IT CANNOT BE STRONGER.**
Eq. 26 says `S_mem` measures "consistency with retrieved successful experiences and
navigation patterns". A retrieved experience carries `h^av` (a fused audio-visual
embedding of a past segment) and `h^traj` (that segment's walk SHAPE). A candidate is a
point on the navmesh. There is no shared space between a fused embedding and a point, so
the only honest comparison is between `h^traj` and the walk the candidate implies: its
geodesic length, and its straightness (straight-line over geodesic).

The strong version would score each candidate by what the agent would SEE there against
the retrieved `c^obj` -- and it is not merely unbuilt, it is unavailable: rendering at a
candidate pose requires moving the agent there, which costs a simulator step and corrupts
the episode. There is no way to observe a place you have not gone. `M^K`'s category prior
is how this tree gets semantic guidance instead, and it arrives through
`resolve_prior`/`predict_category` rather than through this term.

**`M^K` CONTRIBUTES NO `S_mem`, WHICH IS THE PAPER'S OWN READING.** §III.G names
"experiences and patterns" and not knowledge, and PR #112 found the matching fact in eq.
24: `K^K` is a CONCEPT, not a vector, and carries no `h^traj` to compare against. So the
two weights that reach `S_mem` are `omega^E` and `omega^P`, renormalised between
themselves.

**WHEN MEMORY SAYS NOTHING, `S_mem` IS 0.0 AND THAT IS SAFE, NOT ARBITRARY.** An empty
`M^L` gives every candidate the same `S_mem`, and adding a constant to every candidate
cannot change an argmax -- so eq. 27 is identical to the pre-memory pick, which is the
property that makes a no-memory arm a real control. `PlanScore.memory_informed` records
which case it was, so the audit can tell "memory said nothing" from "memory said this is
bad".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from earshot.agent.proposers import SOURCE_INVESTIGATE, Candidate
from earshot.agent.scorer import score_candidate
from earshot.memory.retrieve import RetrievedContext

__all__ = [
    "PlanWeights",
    "PlanScore",
    "feasibility",
    "memory_consistency",
    "score_plan",
    "score_plans",
    "pick_plan",
    "as_measurements",
]

# `h^traj`'s component order, restated as the two indices this module reads. Named rather
# than inlined as `[0]` and `[2]` because `TRAJECTORY_COMPONENTS` is the source of truth
# and a silent reindex here would compare a path length against a turn angle.
_TRAJ_PATH_LENGTH = 0
_TRAJ_STRAIGHTNESS = 2


@dataclass(frozen=True)
class PlanWeights:
    """`l1`, `l2`, `l3` of eq. 26. All three required; the paper gives no values.

    The same two refusals `memory.consolidate.ImportanceWeights` makes, for the same
    reasons: a negative weight runs its term backwards rather than down-weighting it, and
    an all-zero triple makes every `Score` exactly 0 so eq. 27 becomes the tie-break
    alone with nothing on disk saying the planner had stopped discriminating.

    `l2 = 0` IS allowed and is the arm that matters: it is DREAM's planner with memory
    removed from eq. 26 and every other component intact.
    """

    plan: float
    memory: float
    feasibility: float

    def __post_init__(self) -> None:
        for name in ("plan", "memory", "feasibility"):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(
                    "PlanWeights.{} must be >= 0, got {}. A negative weight runs its "
                    "term backwards rather than down-weighting it".format(name, value)
                )
            object.__setattr__(self, name, value)
        if self.plan == 0.0 and self.memory == 0.0 and self.feasibility == 0.0:
            raise ValueError(
                "PlanWeights(0, 0, 0) makes every Score exactly 0, so eq. 27 is decided "
                "by the tie-break alone and nothing on disk would say the planner had "
                "stopped discriminating"
            )


@dataclass(frozen=True)
class PlanScore:
    """One candidate's eq. 26 breakdown, kept whole so the audit carries every term.

    `total` is not recomputed by any consumer: a reader that re-derived it from the three
    parts and the weights could disagree with what the pick actually used, which is the
    class of divergence this tree removes rather than documents.
    """

    candidate: Candidate
    plan: float
    memory: float
    feasibility: float
    total: float
    memory_informed: bool


def feasibility(candidate: Candidate) -> float:
    """`S_feas` (eq. 26): how directly the agent can actually reach this candidate.

    Straight-line over geodesic, clipped to [0, 1]. 1.0 is a clear shot; a candidate
    reachable only around a long detour scores low, which is exactly "feasibility under
    the current environmental state" for an agent that must walk there.

    **`geodesic_m is None` RAISES.** `Candidate.geodesic_m`'s own contract is that `None`
    means "not yet asked" and that an UNREACHABLE candidate is not in the pool at all --
    so a `None` here means the caller skipped `reachability.reachable_pool`. Scoring it as
    feasible would silently promote a candidate nothing has routed to, and scoring it 0.0
    would silently demote one that may be fine. `agent.scorer.pick_waypoint` raises on an
    empty pool for the same reason and says the same thing.
    """
    if candidate.geodesic_m is None:
        raise ValueError(
            "candidate {} has no geodesic_m, which means the navmesh was never asked "
            "about it -- reachability.reachable_pool has been skipped. S_feas cannot be "
            "guessed from a straight line".format(candidate.candidate_id)
        )
    geodesic = float(candidate.geodesic_m)
    straight = float(candidate.distance_m)
    if geodesic <= 0.0:
        # Standing on it. Perfectly feasible, and the ratio is 0/0.
        return 1.0
    return float(min(1.0, max(0.0, straight / geodesic)))


def _agreement(mine: float, theirs: float) -> float:
    """Symmetric ratio agreement in [0, 1]: 1.0 when equal, falling as they diverge.

    `1 - |a - b| / (a + b)`, which is scale-free and therefore needs no metre-to-score
    conversion constant -- the same reason `consolidate`'s `C_j` and `U_j` are shares
    rather than absolute quantities. Both zero is perfect agreement, not a division.
    """
    total = abs(float(mine)) + abs(float(theirs))
    if total == 0.0:
        return 1.0
    return float(max(0.0, 1.0 - abs(float(mine) - float(theirs)) / total))


def _leg_agreement(candidate: Candidate, trajectory: Sequence[float]) -> float:
    """How much the walk this candidate implies resembles one remembered `h^traj`.

    Two comparable quantities and no more: the length of the leg, and its straightness.
    The remembered vector's other two components (`net_displacement_m`,
    `mean_abs_turn_rad`) have no counterpart on a candidate the agent has not walked yet,
    and inventing one would be scoring against a number this side cannot produce.
    """
    geodesic = float(candidate.geodesic_m or 0.0)
    straight = float(candidate.distance_m)
    implied_straightness = (straight / geodesic) if geodesic > 0.0 else 1.0
    implied_straightness = min(1.0, max(0.0, implied_straightness))
    length = _agreement(geodesic, float(trajectory[_TRAJ_PATH_LENGTH]))
    shape = 1.0 - abs(implied_straightness - float(trajectory[_TRAJ_STRAIGHTNESS]))
    return float((length + max(0.0, shape)) / 2.0)


def memory_consistency(
    candidate: Candidate, context: RetrievedContext
) -> Tuple[float, bool]:
    """`S_mem` (eq. 26), and whether memory had anything to say.

    Returns `(score, informed)`. `informed` is `False` when neither `M^E` nor `M^P`
    retrieved anything -- see the module docstring on why the score is then 0.0 and why
    that is safe rather than arbitrary.

    The two levels are combined by `omega^E` and `omega^P` RENORMALISED BETWEEN
    THEMSELVES. `omega^K` is dropped because `M^K` returns a concept, not a walk, and
    §III.G names only experiences and patterns. Renormalising rather than leaving the
    weights short means `S_mem` stays in [0, 1] whatever share the knowledge level took,
    so `l2` weighs the same amount of evidence from step to step.
    """
    if context.weights is None:
        return (0.0, False)
    parts: List[Tuple[float, float]] = []
    if context.experience:
        parts.append((
            context.weights.experience,
            sum(_leg_agreement(candidate, hit.entry.trajectory)
                for hit in context.experience) / len(context.experience),
        ))
    if context.pattern:
        parts.append((
            context.weights.pattern,
            sum(_leg_agreement(candidate, hit.pattern.strategy.trajectory)
                for hit in context.pattern) / len(context.pattern),
        ))
    if not parts:
        return (0.0, False)
    total_weight = sum(weight for weight, _score in parts)
    if total_weight <= 0.0:
        # Both levels answered but omega gave them nothing, which can only happen if the
        # knowledge level took all the mass. Fall back to an unweighted mean rather than
        # dividing by zero or silently reporting 0.0 -- the levels DID answer.
        return (float(sum(score for _weight, score in parts) / len(parts)), True)
    return (
        float(sum(weight * score for weight, score in parts) / total_weight),
        True,
    )


def score_plan(
    candidate: Candidate,
    context: RetrievedContext,
    *,
    weights: PlanWeights,
) -> PlanScore:
    """`Score(P^i) = l1 S_plan + l2 S_mem + l3 S_feas` (eq. 26), kept whole.

    `S_plan` is `agent.scorer.score_candidate` unchanged, so an arm with `l2 = l3 = 0` and
    `l1 = 1` reproduces the pre-DREAM ranking exactly -- which is what makes the
    pre-DREAM behaviour a control rather than a memory.
    """
    plan = float(score_candidate(candidate))
    memory, informed = memory_consistency(candidate, context)
    feasible = feasibility(candidate)
    return PlanScore(
        candidate=candidate,
        plan=plan,
        memory=memory,
        feasibility=feasible,
        total=(
            weights.plan * plan
            + weights.memory * memory
            + weights.feasibility * feasible
        ),
        memory_informed=informed,
    )


def _rank(scored: PlanScore) -> Tuple[int, float, int]:
    """Sort key, carried from `agent.scorer._rank` and load-bearing for the same reason.

    The divert first BY RANK, then `Score` descending, then emission order. See the module
    docstring: eq. 26 puts a memory term back into the blend that made the divert's 1.0
    tie-able, so without this a frontier the memory likes would outrank the anomaly the
    controller has already decided to investigate.
    """
    is_divert = 0 if scored.candidate.source == SOURCE_INVESTIGATE else 1
    return (is_divert, -scored.total, scored.candidate.candidate_id)


def score_plans(
    candidates: Sequence[Candidate],
    context: RetrievedContext,
    *,
    weights: PlanWeights,
) -> List[PlanScore]:
    """Every candidate scored under eq. 26, best first, divert ahead of all."""
    scored = [score_plan(c, context, weights=weights) for c in candidates]
    scored.sort(key=_rank)
    return scored


def pick_plan(
    candidates: Sequence[Candidate],
    context: RetrievedContext,
    *,
    weights: PlanWeights,
) -> PlanScore:
    """`P*_t = argmax_i Score(P^i_t)` (eq. 27), with the divert override intact.

    Raises on an empty pool rather than returning `None`, exactly as
    `agent.scorer.pick_waypoint` does and for the identical reason: ADR-0008's invariant
    is that the pool is never empty, `reachability.assert_pool` enforces it, and a `None`
    here would read as "no action this step" -- the failure that looks like standing
    still.
    """
    if not candidates:
        raise ValueError(
            "pick_plan got an empty pool; ADR-0008's invariant is asserted in "
            "reachability.assert_pool, which the caller has skipped"
        )
    return score_plans(candidates, context, weights=weights)[0]


def as_measurements(scored: PlanScore) -> Tuple[Tuple[str, float], ...]:
    """The eq. 26 breakdown as named numbers for the audit record.

    A tuple of pairs rather than a dict so the ORDER is fixed and a diff of two audits
    lines up. `memory_informed` rides as 1.0/0.0 because the audit's metric mapping holds
    floats -- the same reason `MemoryCondition` is a typed field rather than a metric.
    """
    return (
        ("plan_s_plan", float(scored.plan)),
        ("plan_s_mem", float(scored.memory)),
        ("plan_s_feas", float(scored.feasibility)),
        ("plan_score_total", float(scored.total)),
        ("plan_memory_informed", 1.0 if scored.memory_informed else 0.0),
    )
