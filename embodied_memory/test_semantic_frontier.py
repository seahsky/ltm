#!/usr/bin/env python
"""Unit tests for the LTM_SEMANTIC_FRONTIER lever (VLFM-style goal-semantic
frontier value map) — the pure logic, no habitat/CLIP.

Covers:
  * FrontierPlanner.observe_value  — FOV-cone accumulation, confidence-weighted
    running mean, no-op when the weight is off (byte-identical default path).
  * FrontierPlanner._semantic_value_at — windowed max over observed cells.
  * FrontierPlanner.propose — geometric path byte-identical when off; blend +
    semantic flag when on; a frontier toward a high-value region outranks an
    otherwise-equal one toward a low-value region.
  * FrontierPhysicsScorer — unflagged candidate byte-identical; a flagged
    semantic frontier is scaled to the ceiling, which sits in the gap between a
    memory NON-match and a memory MATCH (so it never crowds out a true recall).

Run: /opt/anaconda3/envs/ltm-embodied/bin/python embodied_memory/test_semantic_frontier.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embodied_memory.frontier_planner import (  # noqa: E402
    CELL_FREE,
    FrontierCandidate,
    FrontierPlanner,
)


def _geom(cluster_size, dist):
    return 0.6 * math.tanh(cluster_size / 10.0) + 0.4 * math.exp(-((dist - 2.5) ** 2) / 4.0)


# ----------------------------------------------------------------------
# observe_value / _semantic_value_at
# ----------------------------------------------------------------------

def case_observe_value_noop_when_off():
    p = FrontierPlanner(semantic_frontier_weight=0.0)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    p.observe_value(np.array([0.0, 0.0, 0.0]), 0.0, 0.9)
    assert float(p.value_map.max()) == 0.0
    assert float(p.value_conf.max()) == 0.0


def case_observe_value_writes_in_front_only():
    p = FrontierPlanner(semantic_frontier_weight=0.5)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    # yaw=0 -> heading is +z (bearing = atan2(dx, dz) = 0 when dz>0). The FOV
    # cone is in front; a cell behind (-z) must stay unobserved.
    p.observe_value(np.array([0.0, 0.0, 0.0]), 0.0, 0.7)
    front = p.grid.world_to_grid(0.0, 2.0)   # +z, in front
    back = p.grid.world_to_grid(0.0, -2.0)   # -z, behind
    assert p.value_conf[front] > 0.0, "front cell should be observed"
    assert abs(float(p.value_map[front]) - 0.7) < 1e-5, p.value_map[front]
    assert p.value_conf[back] == 0.0, "behind cell must not be observed"
    assert float(p.value_map[back]) == 0.0


def case_observe_value_confidence_weighted_mean():
    p = FrontierPlanner(semantic_frontier_weight=0.5)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    # Two head-on observations at the same pose average (conf = cos(0) = 1 each).
    p.observe_value(np.array([0.0, 0.0, 0.0]), 0.0, 0.7)
    p.observe_value(np.array([0.0, 0.0, 0.0]), 0.0, 0.3)
    front = p.grid.world_to_grid(0.0, 2.0)
    assert abs(float(p.value_map[front]) - 0.5) < 1e-5, p.value_map[front]


def case_semantic_value_at_windowed_max_else_zero():
    p = FrontierPlanner(semantic_frontier_weight=0.5)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    p.value_map[50, 50] = 0.6
    p.value_conf[50, 50] = 1.0
    assert abs(p._semantic_value_at((52, 52), radius=10) - 0.6) < 1e-6
    # nothing observed near (150,150) -> 0.0
    assert p._semantic_value_at((150, 150), radius=10) == 0.0


def case_semantic_value_at_ignores_zero_confidence_cells():
    p = FrontierPlanner(semantic_frontier_weight=0.5)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    # A value with no confidence (never observed) must not count.
    p.value_map[50, 50] = 0.9
    p.value_conf[50, 50] = 0.0
    assert p._semantic_value_at((50, 50), radius=5) == 0.0


# ----------------------------------------------------------------------
# propose() blend
# ----------------------------------------------------------------------

def _carve_two_frontiers(p):
    """Two FREE blocks ~2 m fore/aft of the agent (each auto-frontier since the
    surrounding cells are UNKNOWN). Returns nothing; mutates the grid."""
    p.grid.grid[78:83, 98:103] = CELL_FREE    # ~ -z (front-ish), centroid ~(80,100)
    p.grid.grid[118:123, 98:103] = CELL_FREE  # ~ +z, centroid ~(120,100)


def case_propose_byte_identical_when_off():
    p = FrontierPlanner(decision_period=1, n_candidates=4, semantic_frontier_weight=0.0)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    _carve_two_frontiers(p)
    cands = p.propose(np.array([0.0, 0.0, 0.0]), 0.0)
    assert len(cands) >= 2, len(cands)
    for c in cands:
        assert "semantic_frontier" not in c.metadata, c.metadata
        exp = _geom(c.cluster_size, c.distance_m)
        assert abs(c.raw_score - exp) < 1e-6, (c.raw_score, exp)


def case_propose_blends_and_flags_when_on():
    p = FrontierPlanner(decision_period=1, n_candidates=4, semantic_frontier_weight=0.5)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    _carve_two_frontiers(p)
    # High value at the -z cluster (rows ~78-82), zero at the +z cluster.
    p.value_map[74:86, 94:106] = 0.9
    p.value_conf[74:86, 94:106] = 1.0
    cands = p.propose(np.array([0.0, 0.0, 0.0]), 0.0)
    assert len(cands) >= 2
    for c in cands:
        assert c.metadata.get("semantic_frontier") is True
        exp = 0.5 * c.metadata["geom_score"] + 0.5 * c.metadata["semantic_value"]
        assert abs(c.raw_score - exp) < 1e-6, (c.raw_score, exp)
    # The high-value (-z) cluster must outrank the zero-value (+z) one and be first.
    hi = [c for c in cands if c.world_xy[1] < 0.0]
    lo = [c for c in cands if c.world_xy[1] > 0.0]
    assert hi and lo, [c.world_xy.tolist() for c in cands]
    assert hi[0].raw_score > lo[0].raw_score, (hi[0].raw_score, lo[0].raw_score)
    assert cands[0] is hi[0], "high-value frontier should sort first"
    assert hi[0].metadata["semantic_value"] > 0.5
    assert lo[0].metadata["semantic_value"] == 0.0


def case_propose_reports_semantic_spread():
    # The vacuous-arm guard for R1 (S1 geometric vs S1+ BLIP-2 frontier).
    #
    # With the weight on but nothing ever observed into the value map, every
    # frontier reads semantic_value=0.0, so raw_score = (1-w)*geom_score — a
    # uniform rescale, which preserves the geometric ranking EXACTLY. S1+ then
    # picks the same frontiers in the same order as S1 while every candidate
    # still carries semantic_frontier=True, so a "did the branch fire" check
    # reports green on an arm that contributed nothing. Same family as the S2
    # audio-DOA head (zero-sum bonus, inert by construction) and M4: fires,
    # changes nothing.
    #
    # Absence is not the only way to be inert: a CONSTANT nonzero value is too,
    # since only SPREAD can reorder. That is not hypothetical — it is the CLIP
    # flatness measured at 0.020 separation three times, and whether BLIP-2 does
    # better is the question R1 exists to answer. So propose() must report the
    # spread it actually saw, or Table 1 cannot tell "BLIP-2 is flat" from
    # "BLIP-2 never loaded" and would publish the first as the second.
    #
    # Both halves are one spec on purpose: the flat half alone would pass a
    # hardcoded spread of 0.0.
    p = FrontierPlanner(decision_period=1, n_candidates=4, semantic_frontier_weight=0.5)
    p.reset(agent_pos=np.array([0.0, 0.0, 0.0]))
    _carve_two_frontiers(p)

    # FLAT: value map never observed => every frontier reads 0.0 => inert.
    flat = p.propose(np.array([0.0, 0.0, 0.0]), 0.0)
    assert len(flat) >= 2, len(flat)
    d = p._last_semantic_diag
    assert d.get("n_scored") == len(flat), d
    assert d.get("spread") == 0.0, f"an unobserved value map must read as zero spread: {d}"

    # POPULATED: one cluster valued => spread > 0 => the blend can reorder.
    p.value_map[74:86, 94:106] = 0.9
    p.value_conf[74:86, 94:106] = 1.0
    live = p.propose(np.array([0.0, 0.0, 0.0]), 0.0)
    assert len(live) >= 2, len(live)
    d2 = p._last_semantic_diag
    assert d2.get("n_scored") == len(live), d2
    assert d2.get("spread") > 0.5, f"a valued cluster must show spread: {d2}"


# ----------------------------------------------------------------------
# FrontierPhysicsScorer ceiling renorm (the crowd-out guardrail)
# ----------------------------------------------------------------------

def _scorer():
    from embodied_memory.memory_bridge import FrontierPhysicsScorer
    return FrontierPhysicsScorer()


def _planner_cand(raw, bearing=0.0, dist=2.0, semantic=False):
    meta = {"semantic_frontier": True} if semantic else {}
    return FrontierCandidate(
        candidate_id=1, world_xy=np.array([0.0, dist], dtype=np.float32),
        grid_rc=(0, 0), distance_m=dist, bearing_rad=bearing,
        cluster_size=10, raw_score=raw, source="planner", metadata=meta,
    )


def _mem_cand(cos, dist=2.0):
    return FrontierCandidate(
        candidate_id=2, world_xy=np.array([0.0, dist], dtype=np.float32),
        grid_rc=(0, 0), distance_m=dist, bearing_rad=0.0,
        cluster_size=0, raw_score=cos, source="memory", metadata={},
    )


def _score(scorer, cand):
    return scorer.score("", None, {"frontier_candidate": cand})


def case_scorer_unflagged_planner_byte_identical():
    sc = _scorer()
    cand = _planner_cand(raw=1.0, bearing=0.0, dist=2.0, semantic=False)
    # 0.5*1 + 0.3*1 + 0.2*dist_score(2.0); dist_score = 1 - |2-2|/4 = 1.0 -> 1.0
    assert abs(_score(sc, cand) - 1.0) < 1e-6, _score(sc, cand)


def case_scorer_flagged_semantic_clamped_to_ceiling():
    sc = _scorer()
    ceil = sc._SEMANTIC_FRONTIER_CEILING
    # raw=1.0 -> unflagged score 1.0 (> ceiling) -> flagged CLAMPS to the ceiling.
    cand = _planner_cand(raw=1.0, bearing=0.0, dist=2.0, semantic=True)
    assert abs(_score(sc, cand) - ceil) < 1e-6, (_score(sc, cand), ceil)


def case_scorer_flagged_below_ceiling_unchanged():
    # The clamp (not a scale) must LEAVE a frontier whose score is already below the
    # ceiling UNCHANGED — this is what preserves the explore fallback (a scale would
    # shrink it below the geometric baseline, which made the agent spin).
    sc = _scorer()
    weak = _planner_cand(raw=0.2, bearing=1.2, dist=2.0, semantic=False)
    weak_flagged = _planner_cand(raw=0.2, bearing=1.2, dist=2.0, semantic=True)
    base = _score(sc, weak)            # ~0.485, below the 0.70 ceiling
    assert base < sc._SEMANTIC_FRONTIER_CEILING, base
    assert abs(_score(sc, weak_flagged) - base) < 1e-6, (_score(sc, weak_flagged), base)


def case_semantic_frontier_loses_to_memory_match_beats_nonmatch():
    sc = _scorer()
    # strongest possible semantic frontier
    sem = _score(sc, _planner_cand(raw=1.0, bearing=0.0, dist=2.0, semantic=True))
    match = _score(sc, _mem_cand(cos=0.44))      # >= _MEM_COS_FULL -> saturates
    nonmatch = _score(sc, _mem_cand(cos=0.25))   # <= _MEM_COS_NULL -> ~0
    assert match > sem, (match, sem)             # a true recall always wins
    assert sem > nonmatch, (sem, nonmatch)       # but a semantic frontier beats junk


def case_scorer_ceiling_is_order_preserving():
    sc = _scorer()
    # two flagged frontiers with different geometric strength keep their order
    strong = _score(sc, _planner_cand(raw=0.9, bearing=0.0, dist=2.0, semantic=True))
    weak = _score(sc, _planner_cand(raw=0.2, bearing=1.2, dist=2.0, semantic=True))
    assert strong > weak, (strong, weak)


def main():
    case_observe_value_noop_when_off()
    case_observe_value_writes_in_front_only()
    case_observe_value_confidence_weighted_mean()
    case_semantic_value_at_windowed_max_else_zero()
    case_semantic_value_at_ignores_zero_confidence_cells()
    case_propose_byte_identical_when_off()
    case_propose_blends_and_flags_when_on()
    case_propose_reports_semantic_spread()
    case_scorer_unflagged_planner_byte_identical()
    case_scorer_flagged_semantic_clamped_to_ceiling()
    case_scorer_flagged_below_ceiling_unchanged()
    case_semantic_frontier_loses_to_memory_match_beats_nonmatch()
    case_scorer_ceiling_is_order_preserving()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
