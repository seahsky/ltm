"""DREAM §III.G (eq. 25-27), and the regression a memory term reopens.

PR 6a of 6 building DREAM as `ICRA2027_Memory` specifies it.

**THE TEST THAT MATTERS MOST HERE IS NOT ABOUT DREAM.** `agent/scorer.py` records a
regression at length: the investigate divert's `score = 1.0` was only maximal while a
rerank blended a memory term on top of it, and once memory was dropped a maximal frontier
TIED the divert and won on emission order -- making the anomaly interrupt advisory.
Eq. 26 puts a memory term back into that blend, so
`test_the_divert_survives_a_memory_that_hates_it` is the arm that says the structural
override still holds under the new arithmetic.

The rest:

  * `S_feas` raising on an unrouted candidate rather than guessing from a straight line;
  * `S_mem` reading only the two `h^traj` components a candidate has a counterpart for;
  * an empty memory scoring every candidate the same, which is what makes a no-memory arm
    a control rather than a different planner;
  * `l1 = 1, l2 = l3 = 0` reproducing the pre-DREAM ranking EXACTLY, which is the same
    claim from the other side.

No torch, no simulator.
"""

import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.proposers import SOURCE_COMPASS, SOURCE_FRONTIER, SOURCE_INVESTIGATE, Candidate
from earshot.agent.scorer import pick_waypoint, score_candidate
from earshot.memory.longterm import (
    ExperienceEntry,
    ExperienceStore,
    LongTermMemory,
    Outcome,
    PatternStore,
    abstract,
)
from earshot.memory.retrieve import (
    ExperienceHit,
    KnowledgeHit,
    PatternHit,
    RetrievedContext,
    Weights,
)
from earshot.memory.store import SemanticStore
from earshot.task.plan import (
    PlanWeights,
    as_measurements,
    feasibility,
    memory_consistency,
    pick_plan,
    score_plan,
    score_plans,
)
from earshot.types import Xyz

EVEN = PlanWeights(plan=1.0, memory=1.0, feasibility=1.0)
PLAN_ONLY = PlanWeights(plan=1.0, memory=0.0, feasibility=0.0)


def _candidate(
    candidate_id=0,
    source=SOURCE_FRONTIER,
    distance_m=2.0,
    geodesic_m=2.0,
    bearing_rad=0.0,
    raw_score=0.5,
):
    return Candidate(
        candidate_id=candidate_id,
        position=Xyz(float(distance_m), 0.0, 0.0),
        source=source,
        distance_m=float(distance_m),
        bearing_rad=float(bearing_rad),
        raw_score=float(raw_score),
        geodesic_m=None if geodesic_m is None else float(geodesic_m),
    )


def _traj(path_length, straightness):
    """An `h^traj` with only the two components `S_mem` reads set to anything."""
    return np.asarray([path_length, 0.0, straightness, 0.0], dtype=np.float32)


def _entry(path_length=2.0, straightness=1.0, sound="alarm", obj="fireplace"):
    return ExperienceEntry(
        context=np.asarray([1.0, 0.0], dtype=np.float32),
        trajectory=_traj(path_length, straightness),
        target_concept=obj,
        outcome=Outcome(reached=True, final_gap_m=0.2),
        sound_concept=sound,
        room_concept=None,
    )


def _context(experiences=(), patterns=(), weights=None, knowledge=None):
    return RetrievedContext(
        experience=tuple(ExperienceHit(entry=e, score=1.0) for e in experiences),
        pattern=tuple(PatternHit(pattern=p, score=1.0) for p in patterns),
        knowledge=knowledge,
        weights=weights,
    )


def _empty_context():
    return RetrievedContext(experience=(), pattern=(), knowledge=None, weights=None)


def _pattern(path_length=2.0, straightness=1.0):
    store = ExperienceStore().extend([
        _entry(path_length, straightness), _entry(path_length, straightness)
    ])
    return abstract(store, min_support=2).patterns[0]


class TestFeasibility(unittest.TestCase):
    """`S_feas` (eq. 26)."""

    def test_a_straight_shot_is_fully_feasible(self):
        self.assertAlmostEqual(feasibility(_candidate(distance_m=3.0, geodesic_m=3.0)), 1.0)

    def test_a_long_detour_is_less_feasible(self):
        self.assertAlmostEqual(
            feasibility(_candidate(distance_m=3.0, geodesic_m=12.0)), 0.25, places=6
        )

    def test_it_is_bounded_to_zero_and_one(self):
        """A geodesic shorter than the straight line is a navmesh artefact, not a
        candidate that is better than feasible."""
        self.assertAlmostEqual(
            feasibility(_candidate(distance_m=5.0, geodesic_m=1.0)), 1.0
        )

    def test_standing_on_it_is_feasible_and_not_a_division(self):
        self.assertAlmostEqual(feasibility(_candidate(distance_m=0.0, geodesic_m=0.0)), 1.0)

    def test_an_unrouted_candidate_raises(self):
        """`geodesic_m is None` means the navmesh was never asked. Scoring it as feasible
        would promote a candidate nothing has routed to; scoring it 0.0 would demote one
        that may be fine. `pick_waypoint` raises on an empty pool for the same reason."""
        with self.assertRaises(ValueError) as caught:
            feasibility(_candidate(geodesic_m=None))
        self.assertIn("reachable_pool has been skipped", str(caught.exception))


class TestMemoryConsistency(unittest.TestCase):
    """`S_mem` (eq. 26). Geometric, and weak for a reason the module states."""

    def test_an_empty_memory_is_uninformed_and_scores_zero(self):
        score, informed = memory_consistency(_candidate(), _empty_context())
        self.assertEqual(score, 0.0)
        self.assertFalse(informed)

    def test_a_candidate_matching_the_remembered_leg_scores_high(self):
        context = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        score, informed = memory_consistency(
            _candidate(distance_m=2.0, geodesic_m=2.0), context
        )
        self.assertTrue(informed)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_a_candidate_at_the_wrong_length_scores_lower(self):
        context = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        near, _ = memory_consistency(_candidate(distance_m=2.0, geodesic_m=2.0), context)
        far, _ = memory_consistency(_candidate(distance_m=20.0, geodesic_m=20.0), context)
        self.assertGreater(near, far)

    def test_a_candidate_at_the_wrong_shape_scores_lower(self):
        """The component a length cannot supply: two candidates the same distance away,
        one a straight shot and one around a wall."""
        context = _context(
            experiences=[_entry(path_length=4.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        direct, _ = memory_consistency(_candidate(distance_m=4.0, geodesic_m=4.0), context)
        around, _ = memory_consistency(_candidate(distance_m=1.0, geodesic_m=4.0), context)
        self.assertGreater(direct, around)

    def test_the_two_levels_are_weighted_by_omega(self):
        """`omega^E` and `omega^P`, and moving the weight moves the score toward the
        level that agrees."""
        context_e = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            patterns=[_pattern(path_length=30.0, straightness=0.1)],
            weights=Weights(experience=0.99, pattern=0.01, knowledge=0.0),
        )
        context_p = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            patterns=[_pattern(path_length=30.0, straightness=0.1)],
            weights=Weights(experience=0.01, pattern=0.99, knowledge=0.0),
        )
        candidate = _candidate(distance_m=2.0, geodesic_m=2.0)
        leaning_e, _ = memory_consistency(candidate, context_e)
        leaning_p, _ = memory_consistency(candidate, context_p)
        self.assertGreater(leaning_e, leaning_p)

    def test_the_knowledge_weight_is_renormalised_away_not_left_short(self):
        """**THE ARM THAT KEEPS `l2` WEIGHING A CONSTANT AMOUNT OF EVIDENCE.** `M^K`
        returns a concept and carries no `h^traj`, so its weight cannot reach `S_mem`.
        Leaving it in the denominator would shrink `S_mem` in proportion to how much mass
        the knowledge level happened to take, which is a scale that changes step to step
        for a reason unrelated to the candidate."""
        without = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        with_k = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=0.2, pattern=0.0, knowledge=0.8),
            knowledge=KnowledgeHit(category="toilet", score=0.9),
        )
        candidate = _candidate(distance_m=2.0, geodesic_m=2.0)
        self.assertAlmostEqual(
            memory_consistency(candidate, without)[0],
            memory_consistency(candidate, with_k)[0],
            places=6,
        )

    def test_a_knowledge_only_retrieval_is_uninformed(self):
        """The retrieval happened and `omega_t` is defined, but no level carries a walk
        to be consistent with."""
        context = _context(
            weights=Weights(experience=0.0, pattern=0.0, knowledge=1.0),
            knowledge=KnowledgeHit(category="toilet", score=0.9),
        )
        score, informed = memory_consistency(_candidate(), context)
        self.assertEqual(score, 0.0)
        self.assertFalse(informed)

    def test_the_score_stays_in_zero_to_one(self):
        for path_length, straightness, distance, geodesic in (
            (0.0, 0.0, 0.0, 0.0), (100.0, 1.0, 0.1, 50.0), (1.0, 0.0, 30.0, 30.0)
        ):
            context = _context(
                experiences=[_entry(path_length, straightness)],
                weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
            )
            score, _ = memory_consistency(
                _candidate(distance_m=distance, geodesic_m=geodesic), context
            )
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_several_hits_from_one_level_are_averaged(self):
        one = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        two = _context(
            experiences=[
                _entry(path_length=2.0, straightness=1.0),
                _entry(path_length=2.0, straightness=1.0),
            ],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        candidate = _candidate(distance_m=2.0, geodesic_m=2.0)
        self.assertAlmostEqual(
            memory_consistency(candidate, one)[0],
            memory_consistency(candidate, two)[0],
            places=6,
        )


class TestPlanWeights(unittest.TestCase):
    def test_a_negative_weight_raises(self):
        for kwargs in ({"plan": -1.0}, {"memory": -1.0}, {"feasibility": -1.0}):
            fields = {"plan": 1.0, "memory": 1.0, "feasibility": 1.0}
            fields.update(kwargs)
            with self.assertRaises(ValueError) as caught:
                PlanWeights(**fields)
            self.assertIn("must be >= 0", str(caught.exception))

    def test_an_all_zero_triple_raises(self):
        with self.assertRaises(ValueError) as caught:
            PlanWeights(plan=0.0, memory=0.0, feasibility=0.0)
        self.assertIn("stopped discriminating", str(caught.exception))

    def test_zeroing_the_memory_term_is_allowed(self):
        """The arm that matters: DREAM's planner with memory removed from eq. 26 and
        every other component intact."""
        self.assertEqual(PlanWeights(plan=1.0, memory=0.0, feasibility=1.0).memory, 0.0)

    def test_the_weights_have_no_defaults(self):
        with self.assertRaises(TypeError):
            PlanWeights()


class TestScorePlan(unittest.TestCase):
    """`Score(P^i)` (eq. 26), kept whole."""

    def test_s_plan_is_the_existing_scorer_unchanged(self):
        """An arm with `l1 = 1, l2 = l3 = 0` must reproduce the pre-DREAM number
        exactly, or the pre-DREAM behaviour is not a control."""
        candidate = _candidate(raw_score=0.7, distance_m=2.0, geodesic_m=2.0)
        scored = score_plan(candidate, _empty_context(), weights=PLAN_ONLY)
        self.assertAlmostEqual(scored.plan, score_candidate(candidate), places=7)
        self.assertAlmostEqual(scored.total, score_candidate(candidate), places=7)

    def test_the_total_is_the_weighted_sum_of_the_three(self):
        candidate = _candidate(distance_m=2.0, geodesic_m=4.0, raw_score=0.6)
        context = _context(
            experiences=[_entry(path_length=4.0, straightness=0.5)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        weights = PlanWeights(plan=2.0, memory=3.0, feasibility=5.0)
        scored = score_plan(candidate, context, weights=weights)
        self.assertAlmostEqual(
            scored.total,
            2.0 * scored.plan + 3.0 * scored.memory + 5.0 * scored.feasibility,
            places=6,
        )

    def test_memory_informed_records_which_case_it_was(self):
        """So the audit can tell "memory said nothing" from "memory said this is bad" --
        two states that both produce `S_mem = 0.0`."""
        self.assertFalse(
            score_plan(_candidate(), _empty_context(), weights=EVEN).memory_informed
        )
        informed = score_plan(
            _candidate(),
            _context(
                experiences=[_entry()],
                weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
            ),
            weights=EVEN,
        )
        self.assertTrue(informed.memory_informed)

    def test_the_breakdown_reaches_the_audit_in_a_fixed_order(self):
        scored = score_plan(_candidate(), _empty_context(), weights=EVEN)
        names = [name for name, _value in as_measurements(scored)]
        self.assertEqual(names, [
            "plan_s_plan", "plan_s_mem", "plan_s_feas", "plan_score_total",
            "plan_memory_informed",
        ])
        self.assertEqual(dict(as_measurements(scored))["plan_memory_informed"], 0.0)


class TestPickPlan(unittest.TestCase):
    """`P*_t = argmax Score` (eq. 27), and the override that outranks it."""

    def test_an_empty_pool_raises(self):
        """`None` here would read as "no action this step", which is the failure that
        looks like standing still."""
        with self.assertRaises(ValueError) as caught:
            pick_plan([], _empty_context(), weights=EVEN)
        self.assertIn("reachability.assert_pool", str(caught.exception))

    def test_the_best_total_wins_among_ordinary_candidates(self):
        poor = _candidate(candidate_id=0, raw_score=0.0, distance_m=1.0, geodesic_m=9.0)
        good = _candidate(candidate_id=1, raw_score=1.0, distance_m=2.0, geodesic_m=2.0)
        picked = pick_plan([poor, good], _empty_context(), weights=EVEN)
        self.assertEqual(picked.candidate.candidate_id, 1)

    def test_ties_break_on_emission_order(self):
        """The one ordering reproducible from a log, carried from `agent.scorer._rank`."""
        first = _candidate(candidate_id=3, raw_score=0.5)
        second = _candidate(candidate_id=7, raw_score=0.5)
        picked = pick_plan([second, first], _empty_context(), weights=EVEN)
        self.assertEqual(picked.candidate.candidate_id, 3)

    def test_the_divert_survives_a_memory_that_hates_it(self):
        """**THE ARM THIS FILE EXISTS FOR.** `agent/scorer.py` records the regression:
        the divert's `score = 1.0` was maximal only while a rerank blended a memory term,
        and once memory was dropped a maximal frontier tied it and won on emission order,
        making the anomaly interrupt ADVISORY. Eq. 26 puts a memory term back, so the
        structural override has to still hold under the new arithmetic.

        The divert below is deliberately the worst candidate on every arithmetic count: a
        long detour, and a leg nothing in memory resembles. The frontier is the best on
        every count and is emitted FIRST, so emission order favours it too."""
        divert = _candidate(
            candidate_id=1, source=SOURCE_INVESTIGATE,
            distance_m=1.0, geodesic_m=40.0, raw_score=0.0, bearing_rad=3.1,
        )
        frontier = _candidate(
            candidate_id=0, source=SOURCE_FRONTIER,
            distance_m=2.0, geodesic_m=2.0, raw_score=1.0, bearing_rad=0.0,
        )
        context = _context(
            experiences=[_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        scored = score_plans([frontier, divert], context, weights=EVEN)
        self.assertEqual(
            scored[0].candidate.source, SOURCE_INVESTIGATE,
            "a frontier the memory liked outranked the anomaly the controller had "
            "already decided to investigate, so the interrupt is advisory again",
        )
        self.assertGreater(
            scored[1].total, scored[0].total,
            "this fixture was meant to make the divert lose on arithmetic and it does "
            "not, so the override is not being tested",
        )

    def test_the_divert_wins_under_every_weighting(self):
        """Including one that zeroes the term the override was originally about."""
        divert = _candidate(
            candidate_id=9, source=SOURCE_INVESTIGATE,
            distance_m=1.0, geodesic_m=40.0, raw_score=0.0, bearing_rad=3.1,
        )
        frontier = _candidate(candidate_id=0, raw_score=1.0, distance_m=2.0, geodesic_m=2.0)
        for weights in (
            EVEN, PLAN_ONLY,
            PlanWeights(plan=0.0, memory=1.0, feasibility=0.0),
            PlanWeights(plan=0.0, memory=0.0, feasibility=1.0),
        ):
            picked = pick_plan([frontier, divert], _empty_context(), weights=weights)
            self.assertEqual(picked.candidate.source, SOURCE_INVESTIGATE)

    def test_an_empty_memory_leaves_the_pre_dream_ranking_untouched(self):
        """**WHY `S_mem = 0.0` IS SAFE.** Adding the same constant to every candidate
        cannot change an argmax, so an empty `M^L` picks exactly what
        `agent.scorer.pick_waypoint` picks -- which is what makes a no-memory arm a
        control rather than a different planner."""
        pool = [
            _candidate(candidate_id=0, raw_score=0.2, distance_m=1.0, geodesic_m=1.0),
            _candidate(candidate_id=1, raw_score=0.9, distance_m=2.0, geodesic_m=2.0),
            _candidate(candidate_id=2, source=SOURCE_COMPASS, raw_score=0.5,
                       distance_m=3.0, geodesic_m=3.0),
        ]
        dream = pick_plan(pool, _empty_context(), weights=PLAN_ONLY)
        legacy = pick_waypoint(pool)
        self.assertEqual(dream.candidate.candidate_id, legacy.candidate.candidate_id)
        self.assertAlmostEqual(dream.total, legacy.score, places=7)

    def test_memory_can_change_the_pick_among_ordinary_candidates(self):
        """The control for the test above. If memory never moved the pick, the whole term
        would be inert and every green here would mean nothing."""
        near = _candidate(candidate_id=0, raw_score=0.5, distance_m=2.0, geodesic_m=2.0)
        far = _candidate(candidate_id=1, raw_score=0.5, distance_m=20.0, geodesic_m=20.0)
        remembers_far = _context(
            experiences=[_entry(path_length=20.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        without = pick_plan([near, far], _empty_context(),
                            weights=PlanWeights(plan=0.0, memory=1.0, feasibility=1.0))
        with_memory = pick_plan([near, far], remembers_far,
                                weights=PlanWeights(plan=0.0, memory=1.0, feasibility=1.0))
        self.assertEqual(without.candidate.candidate_id, 0)
        self.assertEqual(
            with_memory.candidate.candidate_id, 1,
            "a memory of long successful legs did not move the pick toward the long "
            "candidate, so S_mem is inert",
        )


class TestTheWholePath(unittest.TestCase):
    """Eq. 25-27 over memory the other modules built, rather than hand-made hits."""

    def test_a_real_retrieved_context_scores_a_pool(self):
        from earshot.memory.retrieve import retrieve

        store = ExperienceStore().extend([
            _entry(path_length=3.0, straightness=0.9),
            _entry(path_length=3.2, straightness=0.9),
        ])
        memory = LongTermMemory(
            experience=store,
            pattern=abstract(store, min_support=2),
            knowledge=SemanticStore(),
        )
        context = retrieve(
            np.asarray([1.0, 0.0], dtype=np.float32), memory,
            k_experience=2, k_pattern=1, k_knowledge=1, temperature=0.5,
        )
        pool = [
            _candidate(candidate_id=0, distance_m=3.0, geodesic_m=3.3, raw_score=0.5),
            _candidate(candidate_id=1, distance_m=12.0, geodesic_m=14.0, raw_score=0.5),
        ]
        scored = score_plans(pool, context, weights=EVEN)
        self.assertEqual(len(scored), 2)
        self.assertTrue(all(s.memory_informed for s in scored))
        self.assertEqual(scored[0].candidate.candidate_id, 0)

    def test_an_empty_long_term_memory_is_uninformed_end_to_end(self):
        from earshot.memory.retrieve import retrieve

        memory = LongTermMemory(
            experience=ExperienceStore(), pattern=PatternStore(), knowledge=SemanticStore()
        )
        context = retrieve(
            np.asarray([1.0, 0.0], dtype=np.float32), memory,
            k_experience=1, k_pattern=1, k_knowledge=1, temperature=0.5,
        )
        scored = score_plan(
            _candidate(distance_m=2.0, geodesic_m=2.0), context, weights=EVEN
        )
        self.assertFalse(scored.memory_informed)
        self.assertEqual(scored.memory, 0.0)


if __name__ == "__main__":
    unittest.main()
