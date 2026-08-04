"""One waypoint out of the pool. ``FrontierPhysicsScorer``, minus what memory owned.

The old scorer sat inside a reranker as the coherence slot, blended
``0.30 * S_sim + 0.70 * S_phys`` where ``S_sim`` was the LTM's goal-versus-caption
cosine, and dispatched on candidate source across four branches. Memory is out of this
build by the map's chart-time decision, so ``S_sim`` has no supplier and the rerank
collapses to the physics term. Two branches survive:

- **geometric**, for frontier and compass candidates:
  ``0.5 * raw_score + 0.3 * bearing_alignment + 0.2 * distance_band``, carried verbatim.
- **the investigate divert**, which is an **override rather than a high score**. The
  controller has decided the agent must go to the anomaly source; a waypoint that loses
  the pick would make the interrupt advisory.

**That override is structural, and it had to be.** The old scorer gave the divert
``score = 1.0`` and called that "max physics score", which was true only because the
final rerank blended ``0.30 * S_sim + 0.70 * S_phys`` and the memory term broke ties.
With memory dropped the physics score *is* the final score, and the geometric branch is
clipped to 1.0 — so a maximal frontier candidate ties the divert exactly, and the
emission-order tie-break hands the pick to the frontier. Found by
``tests/mac/test_agent_scorer.py``, which asserted the divert beats the best possible
frontier and went red. The divert therefore sorts ahead of everything by rank, and its
1.0 survives only as the number the audit record carries.

Dropped with their suppliers: the memory-cosine branch (``_MEM_COS_NULL`` / ``_MEM_COS_FULL``
were calibrated against SBERT caption cosines nothing here produces), the coarse-affordance
branch (built, measured never-chosen, arc closed), and the semantic-frontier ceiling
(guarding a crowd-out between two absent components). The seam they used is intact — it is
``Candidate.source``, and a memory proposer would add one branch here.

**The pick is deterministic.** Highest score, and on a tie the lowest ``candidate_id``,
which is emission order. The old sort was stable over an unstable input order, so a tie
resolved on whichever cluster the grid scan happened to reach first — and the compass
fan's pre-occupancy version tied on *every* direction, so the tie-break was the whole
decision (Run-5 smoke 6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from earshot.agent.proposers import SOURCE_INVESTIGATE, Candidate

__all__ = ["ScoredCandidate", "score_candidate", "score_pool", "pick_waypoint"]

# The geometric blend's weights, carried verbatim from `FrontierPhysicsScorer.score`.
_W_RAW = 0.5
_W_BEARING = 0.3
_W_DISTANCE = 0.2

# The distance band: full credit at 2 m, falling linearly to nothing at 6 m or on top of
# the agent. Not the same kernel as `proposers.frontier_score`'s 2.5 m Gaussian, and that
# is carried rather than reconciled — one is the proposer's intrinsic preference and the
# other the scorer's, and collapsing them would change the ranking ADR-0008 froze.
_DIST_PEAK_M = 2.0
_DIST_SPAN_M = 4.0


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate and its physics score, so the audit record can carry both."""

    candidate: Candidate
    score: float


def score_candidate(candidate: Candidate) -> float:
    """The candidate's physics score in ``[0, 1]``."""
    if candidate.source == SOURCE_INVESTIGATE:
        return 1.0
    distance = float(candidate.distance_m)
    if distance <= 0.0:
        distance_term = 0.0
    else:
        distance_term = max(0.0, 1.0 - abs(distance - _DIST_PEAK_M) / _DIST_SPAN_M)
    # Bearing alignment on the MAGNITUDE of the relative bearing: heading somewhere
    # straight ahead is cheap, and turning 180 degrees costs six turn actions. The sign
    # is deliberately not read — which is exactly why the old tree's 180-degree frame
    # error was invisible here (see `occupancy.py`): `|bearing|` stayed a plausible
    # number and simply ranked the pool backwards.
    bearing_term = max(0.0, 1.0 - abs(float(candidate.bearing_rad)) / math.pi)
    score = _W_RAW * float(candidate.raw_score) + _W_BEARING * bearing_term + _W_DISTANCE * distance_term
    return float(min(1.0, max(0.0, score)))


def _rank(scored: ScoredCandidate) -> Tuple[int, float, int]:
    """Sort key: the divert first, then score descending, then emission order.

    The first element is what makes the override structural rather than arithmetic — see
    the module docstring. Emission order is the last tie-break because it is the one
    ordering that is reproducible from a log; the old sort was stable over an input order
    that depended on which cluster the grid scan reached first.
    """
    is_divert = 0 if scored.candidate.source == SOURCE_INVESTIGATE else 1
    return (is_divert, -scored.score, scored.candidate.candidate_id)


def score_pool(candidates: Sequence[Candidate]) -> List[ScoredCandidate]:
    """Every candidate scored, best first, with the investigate divert ahead of all."""
    scored = [ScoredCandidate(candidate=c, score=score_candidate(c)) for c in candidates]
    scored.sort(key=_rank)
    return scored


def pick_waypoint(candidates: Sequence[Candidate]) -> ScoredCandidate:
    """The one waypoint the follower steers to.

    Raises on an empty pool rather than returning ``None``. ADR-0008's invariant is that
    the pool is never empty and ``reachability.assert_pool`` is where that is enforced, so
    reaching here with nothing means the runner skipped the filter — and a ``None`` here
    would be read as "no action this step", which is the failure that looks like standing
    still.
    """
    if not candidates:
        raise ValueError(
            "pick_waypoint got an empty pool; ADR-0008's invariant is asserted in "
            "reachability.assert_pool, which the caller has skipped"
        )
    return score_pool(candidates)[0]
