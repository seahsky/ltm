"""The scorer: the geometric blend, the investigate override, and a deterministic pick.

Two properties are worth more than the arithmetic. The investigate divert must **win**,
or the anomaly interrupt is advisory rather than an interrupt. And the pick must be
deterministic, because the old compass fan tied on every direction and the tie-break was
therefore the whole decision (Run-5 smoke 6) — a pick that depends on dict or scan order
is a pick nobody can reproduce from a log.
"""

import math
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.proposers import (
    SOURCE_COMPASS,
    SOURCE_FRONTIER,
    SOURCE_INVESTIGATE,
    Candidate,
)
from earshot.agent.scorer import pick_waypoint, score_candidate, score_pool
from earshot.types import Xyz


def candidate(
    cid=1, source=SOURCE_FRONTIER, distance_m=2.0, bearing_rad=0.0, raw_score=0.5
):
    return Candidate(
        candidate_id=cid,
        position=Xyz(0.0, 0.0, -distance_m),
        source=source,
        distance_m=distance_m,
        bearing_rad=bearing_rad,
        raw_score=raw_score,
    )


class TestTheGeometricBlend(unittest.TestCase):
    def test_the_blend_is_the_carried_formula(self):
        """0.5 * raw + 0.3 * bearing_alignment + 0.2 * distance_band, at the band's peak."""
        self.assertAlmostEqual(
            score_candidate(candidate(raw_score=0.8, distance_m=2.0, bearing_rad=0.0)),
            0.5 * 0.8 + 0.3 * 1.0 + 0.2 * 1.0,
            places=9,
        )

    def test_a_nearer_bearing_beats_a_turned_one(self):
        ahead = score_candidate(candidate(bearing_rad=0.0))
        behind = score_candidate(candidate(bearing_rad=math.pi))
        self.assertGreater(ahead, behind)

    def test_the_bearing_term_ignores_the_sign(self):
        """Which is exactly why the old tree's 180-degree frame error was invisible here."""
        self.assertEqual(
            score_candidate(candidate(bearing_rad=1.2)),
            score_candidate(candidate(bearing_rad=-1.2)),
        )

    def test_the_distance_band_peaks_at_two_metres(self):
        peak = score_candidate(candidate(distance_m=2.0))
        self.assertGreater(peak, score_candidate(candidate(distance_m=0.2)))
        self.assertGreater(peak, score_candidate(candidate(distance_m=9.0)))

    def test_a_zero_distance_candidate_gets_no_distance_credit(self):
        """A waypoint on top of the agent is not somewhere to go."""
        self.assertAlmostEqual(
            score_candidate(candidate(raw_score=0.0, distance_m=0.0, bearing_rad=0.0)),
            0.3,
            places=9,
        )

    def test_scores_stay_inside_the_unit_interval(self):
        for raw in (-3.0, 0.0, 0.5, 4.0):
            for distance in (0.0, 2.0, 100.0):
                score = score_candidate(candidate(raw_score=raw, distance_m=distance))
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_a_compass_candidate_scores_on_the_same_branch(self):
        self.assertEqual(
            score_candidate(candidate(source=SOURCE_COMPASS, raw_score=0.7)),
            score_candidate(candidate(source=SOURCE_FRONTIER, raw_score=0.7)),
        )


class TestTheInvestigateOverride(unittest.TestCase):
    def test_the_divert_scores_the_maximum(self):
        self.assertEqual(score_candidate(candidate(source=SOURCE_INVESTIGATE)), 1.0)

    def test_the_divert_beats_the_best_possible_frontier(self):
        """Otherwise the interrupt is a suggestion the planner is free to ignore.

        This went red on the first run and found a real defect: a maximal frontier scores
        exactly 1.0, ties the divert, and the emission-order tie-break then hands it the
        pick. The old tree's memory term hid that. The override is now a sort rank.
        """
        best_frontier = candidate(cid=1, raw_score=1.0, distance_m=2.0, bearing_rad=0.0)
        self.assertEqual(score_candidate(best_frontier), 1.0)
        divert = candidate(cid=2, source=SOURCE_INVESTIGATE, distance_m=12.0, bearing_rad=math.pi)
        self.assertEqual(pick_waypoint([best_frontier, divert]).candidate.candidate_id, 2)

    def test_the_divert_wins_from_either_input_position(self):
        divert = candidate(cid=9, source=SOURCE_INVESTIGATE)
        frontier = candidate(cid=1, raw_score=1.0)
        self.assertEqual(pick_waypoint([divert, frontier]).candidate.candidate_id, 9)
        self.assertEqual(pick_waypoint([frontier, divert]).candidate.candidate_id, 9)

    def test_the_divert_ignores_its_own_geometry(self):
        near = score_candidate(candidate(source=SOURCE_INVESTIGATE, distance_m=0.5))
        far = score_candidate(candidate(source=SOURCE_INVESTIGATE, distance_m=30.0))
        self.assertEqual(near, far)


class TestThePick(unittest.TestCase):
    def test_the_pool_comes_back_best_first(self):
        pool = [
            candidate(1, raw_score=0.1),
            candidate(2, raw_score=0.9),
            candidate(3, raw_score=0.5),
        ]
        self.assertEqual([s.candidate.candidate_id for s in score_pool(pool)], [2, 3, 1])

    def test_a_tie_is_broken_by_emission_order(self):
        pool = [candidate(7, raw_score=0.5), candidate(3, raw_score=0.5)]
        self.assertEqual(pick_waypoint(pool).candidate.candidate_id, 3)

    def test_the_pick_does_not_depend_on_input_order(self):
        pool = [candidate(1, raw_score=0.5), candidate(2, raw_score=0.5), candidate(3, raw_score=0.9)]
        self.assertEqual(
            pick_waypoint(pool).candidate.candidate_id,
            pick_waypoint(list(reversed(pool))).candidate.candidate_id,
        )

    def test_the_score_travels_with_the_pick(self):
        picked = pick_waypoint([candidate(raw_score=0.8, distance_m=2.0)])
        self.assertAlmostEqual(picked.score, score_candidate(picked.candidate), places=12)

    def test_an_empty_pool_raises_rather_than_returning_nothing(self):
        """A ``None`` here reads as "no action", which is the failure that looks like
        standing still. ADR-0008's invariant is enforced in ``reachability.assert_pool``."""
        with self.assertRaises(ValueError):
            pick_waypoint([])


if __name__ == "__main__":
    unittest.main(verbosity=2)
