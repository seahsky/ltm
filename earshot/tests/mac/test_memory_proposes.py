"""ADR-0026: the semantic prior PROPOSES a waypoint and is ranked, instead of replacing one.

`matrix-2` measured the episodic store costing 10.3 points against a RIGHT prior, and the
mechanism was one line: `runner.py` did `target = memory_prior.target` while diverting, so
the prior overwrote the acoustic estimate outright. The divert class then held exactly one
candidate, and the structural override in `_rank` made it the pick with nothing ranking
it. A mechanism that is redundant when it agrees with the cue and unchecked when it does
not can only add variance, which is the shape that was measured.

What is tested here is the fix and, at least as importantly, the two things the fix must
not break: the interrupt must stay an override, and `proposes=False` must still be the
behaviour every earlier result was measured under.

The rail gets its own class because an unbounded proposal is `matrix-2`'s failure mode one
rank lower, and a rail that suppressed everything would read as a live arm that measured
nothing rather than as a bug.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.proposers import SOURCE_FRONTIER, SOURCE_INVESTIGATE
from earshot.memory.retrieve import Weights
from earshot.task.plan import score_plans
from earshot.memory.store import MemoryCondition, SemanticStore
from earshot.task.memory_prior import MemoryContext
from earshot.task.runner import (
    DIVERT_CANDIDATE_ID,
    MEMORY_CANDIDATE_ID,
    _divert_candidate,
    _memory_candidate,
)
from earshot.types import Pose, Xyz

# The eq. 26 fixtures, from the file that owns them. Rebuilding them here would let
# the override arm and its own regression test drift apart silently.
from test_task_memory_arm import MemoryArmCase
from test_task_plan import EVEN as PLAN_EVEN
from test_task_plan import _candidate as _plan_candidate
from test_task_plan import _context as _plan_context
from test_task_plan import _entry as _plan_entry

POSE = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)
ACOUSTIC = Xyz(3.0, 0.0, 0.0)


def euclid(a, b):
    return ((a.x - b.x) ** 2 + (a.z - b.z) ** 2) ** 0.5


def propose(target, *, acoustic=ACOUSTIC, geodesic=euclid, max_offset_m=6.0):
    return _memory_candidate(
        target, POSE, acoustic=acoustic, geodesic=geodesic, max_offset_m=max_offset_m
    )


class TestTheProposalIsACandidateAndNotAReplacement(unittest.TestCase):
    def test_it_is_emitted_into_the_divert_class(self):
        """SOURCE_INVESTIGATE is what keeps the interrupt an override. A proposal emitted
        as a frontier would lose to the acoustic divert by rank and never be read."""
        candidate, why = propose(Xyz(4.0, 0.0, 1.0))
        self.assertEqual(why, "emitted")
        self.assertEqual(candidate.source, SOURCE_INVESTIGATE)

    def test_its_id_is_reserved_and_distinct_from_the_acoustic_divert(self):
        """`_emit` issues from 1 and the acoustic divert holds 0, so all three are
        distinguishable in an audit by id alone."""
        candidate, _ = propose(Xyz(4.0, 0.0, 1.0))
        self.assertEqual(candidate.candidate_id, MEMORY_CANDIDATE_ID)
        self.assertNotEqual(MEMORY_CANDIDATE_ID, DIVERT_CANDIDATE_ID)

    def test_the_acoustic_estimate_wins_an_exact_tie(self):
        """`_rank`'s last key is the id, and `score_candidate` gives EVERY divert a hard
        1.0, so on a run with no memory term the two tie exactly. The prior must lose
        that tie, or `proposes=True` is replacement one rank lower. A draft with a
        negative id did exactly that and this is the arm that caught it."""
        self.assertGreater(MEMORY_CANDIDATE_ID, DIVERT_CANDIDATE_ID)

    def test_it_arrives_indistinguishable_from_the_divert_except_in_position(self):
        """THE CONTRAST MUST BE DECIDED BY THE RANKING, NOT BY THE CONSTRUCTOR. A
        different `raw_score` here would settle `memory-propose` against `memory-replace`
        before eq. 26 ever saw the pool."""
        target = Xyz(4.0, 0.0, 1.0)
        proposal, _ = propose(target)
        divert = _divert_candidate(ACOUSTIC, POSE)
        self.assertEqual(proposal.raw_score, divert.raw_score)
        self.assertEqual(proposal.source, divert.source)
        self.assertEqual(proposal.bearing_rad, divert.bearing_rad)
        self.assertEqual(proposal.position, target)

    def test_the_acoustic_estimate_is_not_consumed(self):
        """The whole decision: after proposing, the cue's own target is still available to
        be the other member of the class."""
        proposal, _ = propose(Xyz(4.0, 0.0, 1.0))
        divert = _divert_candidate(ACOUSTIC, POSE)
        self.assertEqual(divert.position, ACOUSTIC)
        self.assertNotEqual(proposal.position, divert.position)


class TestTheRail(unittest.TestCase):
    """`matrix-2`'s failure mode is a wrong prior with nothing above it. Ranking fixes the
    ordering half and NOT the magnitude half: eq. 26 could still prefer a proposal on the
    far side of the house. The rail is the magnitude half."""

    def test_a_proposal_inside_the_bound_is_emitted(self):
        candidate, why = propose(Xyz(3.0, 0.0, 5.0), max_offset_m=6.0)
        self.assertEqual(why, "emitted")
        self.assertIsNotNone(candidate)

    def test_a_proposal_beyond_the_bound_never_enters_the_pool(self):
        candidate, why = propose(Xyz(3.0, 0.0, 40.0), max_offset_m=6.0)
        self.assertEqual(why, "railed")
        self.assertIsNone(candidate)

    def test_the_bound_is_measured_from_the_acoustic_estimate_not_the_agent(self):
        """A prior 1 m from the cue's target is a small correction however far the agent
        is standing from either. Measuring from the pose would rail exactly the useful
        case: a correction offered early in a long detour."""
        near_cue_far_from_agent = Xyz(3.0, 0.0, 1.0)
        self.assertGreater(euclid(POSE.position, near_cue_far_from_agent), 3.0)
        _, why = propose(near_cue_far_from_agent, max_offset_m=2.0)
        self.assertEqual(why, "emitted")

    def test_an_unrouted_proposal_is_suppressed(self):
        """`None` from the navmesh is a disconnected island, not a distance of 0. The
        conservative direction is to keep the estimate the cue gave."""
        _, why = propose(Xyz(4.0, 0.0, 1.0), geodesic=lambda a, b: None)
        self.assertEqual(why, "unrouted")

    def test_a_removed_rail_emits_what_a_rail_would_have_stopped(self):
        """`None` is an arm to measure against, and it has to actually differ."""
        _, railed = propose(Xyz(3.0, 0.0, 40.0), max_offset_m=6.0)
        candidate, why = propose(Xyz(3.0, 0.0, 40.0), max_offset_m=None)
        self.assertEqual(railed, "railed")
        self.assertEqual(why, "emitted")
        self.assertIsNotNone(candidate)

    def test_no_acoustic_estimate_means_no_proposal(self):
        """There is nothing to compete with and nothing to measure the rail against, so
        the prior does NOT get the pick by default. That would be replacement again."""
        candidate, why = propose(Xyz(4.0, 0.0, 1.0), acoustic=None)
        self.assertEqual(why, "no_acoustic")
        self.assertIsNone(candidate)

    def test_every_suppression_names_itself(self):
        """A reader must be able to tell a rail from a routing failure from an arm that
        never ran. `dream-1` is why."""
        reasons = {
            propose(Xyz(3.0, 0.0, 40.0))[1],
            propose(Xyz(4.0, 0.0, 1.0), geodesic=lambda a, b: None)[1],
            propose(Xyz(4.0, 0.0, 1.0), acoustic=None)[1],
            propose(Xyz(4.0, 0.0, 1.0))[1],
        }
        self.assertEqual(
            reasons, {"railed", "unrouted", "no_acoustic", "emitted"}
        )


class TestTheInterruptStaysAnOverride(unittest.TestCase):
    """`plan.py:30-42` records the regression: a maximal frontier tied the divert and won
    on emission order, making the anomaly interrupt ADVISORY. A second member of the
    divert class must not reopen it.

    Run through `score_plans`, the real ranking path, and not through `_rank` on a raw
    candidate: the override this protects lives in eq. 26's ordering and a unit test on
    the key function would not see a regression in how the pool reaches it.
    """

    def _frontier(self, candidate_id=7, raw_score=1.0):
        return _plan_candidate(
            candidate_id=candidate_id, source=SOURCE_FRONTIER,
            distance_m=1.0, geodesic_m=1.0, raw_score=raw_score, bearing_rad=0.0,
        )

    def test_a_memory_that_hates_both_diverts_still_cannot_pick_a_frontier(self):
        """`test_the_divert_survives_a_memory_that_hates_it` with the class widened. The
        frontier is best on every arithmetic count and is emitted FIRST."""
        proposal = _plan_candidate(
            candidate_id=MEMORY_CANDIDATE_ID, source=SOURCE_INVESTIGATE,
            distance_m=1.0, geodesic_m=41.0, raw_score=0.0, bearing_rad=3.1,
        )
        divert = _plan_candidate(
            candidate_id=DIVERT_CANDIDATE_ID, source=SOURCE_INVESTIGATE,
            distance_m=1.0, geodesic_m=40.0, raw_score=0.0, bearing_rad=3.1,
        )
        context = _plan_context(
            experiences=[_plan_entry(path_length=1.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        ranked = score_plans(
            [self._frontier(), divert, proposal], context, weights=PLAN_EVEN
        )
        self.assertEqual(
            ranked[0].candidate.source, SOURCE_INVESTIGATE,
            "a frontier the memory liked outranked the anomaly the controller had "
            "already decided to investigate, so the interrupt is advisory again",
        )
        self.assertEqual(
            ranked[1].candidate.source, SOURCE_INVESTIGATE,
            "only one member of the divert class sorted ahead of the frontier, so the "
            "class is not a class",
        )

    def test_the_proposal_can_win_the_pick_from_the_acoustic_divert(self):
        """THE ARM THE ADR EXISTS FOR. Both are diverts, so the override cannot decide
        between them and eq. 26 must. The proposal's leg is the one memory resembles."""
        proposal = _plan_candidate(
            candidate_id=MEMORY_CANDIDATE_ID, source=SOURCE_INVESTIGATE,
            distance_m=2.0, geodesic_m=2.0, raw_score=1.0, bearing_rad=0.0,
        )
        divert = _plan_candidate(
            candidate_id=DIVERT_CANDIDATE_ID, source=SOURCE_INVESTIGATE,
            distance_m=1.0, geodesic_m=30.0, raw_score=1.0, bearing_rad=0.0,
        )
        context = _plan_context(
            experiences=[_plan_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        ranked = score_plans([divert, proposal], context, weights=PLAN_EVEN)
        self.assertEqual(
            ranked[0].candidate.candidate_id, MEMORY_CANDIDATE_ID,
            "eq. 26 could not prefer the proposal even when it is the better leg on "
            "every count, so the class has two members and still only one outcome",
        )

    def test_the_acoustic_divert_wins_when_the_memory_does_not_prefer_the_proposal(self):
        """The other arm. The prior has to EARN the pick; losing must be reachable, or
        the mechanism is replacement with extra steps."""
        proposal = _plan_candidate(
            candidate_id=MEMORY_CANDIDATE_ID, source=SOURCE_INVESTIGATE,
            distance_m=1.0, geodesic_m=30.0, raw_score=1.0, bearing_rad=0.0,
        )
        divert = _plan_candidate(
            candidate_id=DIVERT_CANDIDATE_ID, source=SOURCE_INVESTIGATE,
            distance_m=2.0, geodesic_m=2.0, raw_score=1.0, bearing_rad=0.0,
        )
        context = _plan_context(
            experiences=[_plan_entry(path_length=2.0, straightness=1.0)],
            weights=Weights(experience=1.0, pattern=0.0, knowledge=0.0),
        )
        ranked = score_plans([proposal, divert], context, weights=PLAN_EVEN)
        self.assertEqual(ranked[0].candidate.candidate_id, DIVERT_CANDIDATE_ID)


class TestTheContextKnob(unittest.TestCase):
    def _context(self, **kwargs):
        return MemoryContext(
            condition=MemoryCondition.NONE,
            semantic=SemanticStore(),
            points_by_category={},
            **kwargs,
        )

    def test_replacement_is_the_default(self):
        """Every result measured before ADR-0026 ran under replacement, so a context that
        does not ask for the new behaviour must not get it."""
        self.assertFalse(self._context().proposes)

    def test_a_rail_of_zero_is_refused(self):
        """It suppresses every proposal, which is `proposes=False` wearing a knob's name
        and would read as a live arm that measured nothing."""
        with self.assertRaises(ValueError):
            self._context(propose_max_offset_m=0.0)
        with self.assertRaises(ValueError):
            self._context(propose_max_offset_m=-1.0)

    def test_a_removed_rail_is_allowed_because_it_is_an_arm(self):
        self.assertIsNone(self._context(propose_max_offset_m=None).propose_max_offset_m)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTheWiringInARealEpisode(MemoryArmCase):
    """ADR-0014's two arms for this mechanism, end to end through `run_episode`.

    `MemoryArmCase`'s geometry is what makes it worth running here: the source is 5 m
    BEHIND the agent and the recalled stove is 6 m AHEAD, 11 m apart, so the prior and the
    cue disagree about as hard as this fixture can make them. That is the case where
    replacement is most dangerous and where the rail has the most to do.
    """

    @classmethod
    def setUpClass(cls):
        case = cls()
        cls.replace = case.run_arm(case.context())
        cls.propose_railed = case.run_arm(case.context(proposes=True))
        cls.propose_open = case.run_arm(
            case.context(proposes=True, propose_max_offset_m=50.0)
        )

    def test_the_replace_arm_records_no_proposal_counters_at_all(self):
        """Absent, not zero. Every result before ADR-0026 ran this way, and an arm that
        never proposed must not be readable as one that proposed and never won."""
        keys = [k for k in self.replace.audit.metrics if k.startswith("memory_propose_")]
        self.assertEqual(keys, [])

    def test_the_proposing_arm_records_them_even_when_it_never_emitted(self):
        """The other half of the same rule: a live arm that was suppressed every step is
        a MEASUREMENT, and it has to be distinguishable from an arm that did not run."""
        metrics = self.propose_railed.audit.metrics
        for key in (
            "memory_propose_eligible",
            "memory_propose_ranked_first",
            "memory_propose_emitted",
            "memory_propose_railed",
        ):
            self.assertIn(key, metrics, "{} is missing, so a reader cannot tell a "
                                        "suppressed arm from an absent one".format(key))
        self.assertEqual(metrics["memory_propose_rail_m"], 6.0)

    def test_the_rail_suppresses_proposals_a_wider_one_admits(self):
        """The acoustic estimate MOVES -- it is the cue's current probe target, not a
        fixed point -- so the offset the rail measures changes step to step and railing is
        intermittent by construction. What has to hold is that the narrow rail stops
        something the wide one admits, which is the rail doing its job."""
        narrow = self.propose_railed.audit.metrics
        wide = self.propose_open.audit.metrics
        print("\n  [rail 6m ] railed={} emitted={}\n  [rail 50m] railed={} emitted={}"
              .format(narrow.get("memory_propose_railed"),
                      narrow.get("memory_propose_emitted"),
                      wide.get("memory_propose_railed"),
                      wide.get("memory_propose_emitted")), flush=True)
        self.assertGreater(narrow["memory_propose_railed"], 0.0)
        self.assertEqual(wide["memory_propose_railed"], 0.0)
        self.assertGreater(
            wide["memory_propose_emitted"], narrow["memory_propose_emitted"],
            "the wider rail admitted no more proposals than the narrow one, so the "
            "bound is not what is deciding",
        )

    def test_an_open_rail_puts_the_proposal_in_the_pool(self):
        """THE ARM THAT PROVES THE MECHANISM RUNS. Without this the file tests a rail and
        a constructor and never shows eq. 26 being given a second divert."""
        metrics = self.propose_open.audit.metrics
        print("\n  [open] emitted={} eligible={} ranked_first={} rail={}".format(
            metrics.get("memory_propose_emitted"),
            metrics.get("memory_propose_eligible"),
            metrics.get("memory_propose_ranked_first"),
            metrics.get("memory_propose_rail_m"),
        ), flush=True)
        self.assertGreater(metrics["memory_propose_emitted"], 0.0)
        self.assertGreater(metrics["memory_propose_eligible"], 0.0)
        self.assertEqual(metrics["memory_propose_rail_m"], 50.0)

    def test_the_prior_still_resolves_in_every_arm(self):
        """If the recall broke, the three arms above would agree for a reason that has
        nothing to do with what they are testing."""
        for name, result in (
            ("replace", self.replace),
            ("railed", self.propose_railed),
            ("open", self.propose_open),
        ):
            self.assertEqual(result.audit.memory_prior_category, "stove", name)

    def test_without_a_memory_term_proposing_reduces_to_the_acoustic_behaviour(self):
        """**THE ARM THAT MATTERS MOST, and it is a reduction rather than an effect.**

        This fixture runs no DREAM context, so `_choose_waypoint` uses `pick_waypoint` and
        eq. 26 never runs. `score_candidate` hands every divert a hard 1.0, so the two
        members of the class tie and the id breaks it. The acoustic estimate must win,
        which means the proposing arm walks the REPLACEMENT arm's path only if the
        replacement arm was going to the same place anyway.

        A draft with a negative id inverted this: the proposal took every tie, the agent
        walked to the stove in both arms, and the two paths matched step for step. That
        read as "the mechanism is inert" when what it actually was is replacement wearing
        the new name. The assertion is therefore on the ACTIONS, and the two arms must
        differ, because replacement sends the agent 6 m forward to the stove and the cue
        sends it 5 m back to the source.
        """
        def actions(result):
            return tuple(step.action for step in result.audit.steps)

        replace_path = actions(self.replace)
        propose_path = actions(self.propose_open)
        differ_at = next(
            (i for i, (a, b) in enumerate(zip(replace_path, propose_path)) if a != b),
            None,
        )
        print("\n  [paths] replace={} propose={} first_difference={}".format(
            len(replace_path), len(propose_path), differ_at), flush=True)
        self.assertNotEqual(
            replace_path, propose_path,
            "the proposing arm reproduced replacement step for step. Either the prior is "
            "taking the pick without earning it, or the acoustic divert never entered "
            "the pool",
        )

    def test_the_proposal_never_won_a_pick_without_eq_26(self):
        """The counter half of the reduction above, stated as a number rather than
        inferred from a trajectory."""
        metrics = self.propose_open.audit.metrics
        self.assertGreater(metrics["memory_propose_eligible"], 0.0)
        self.assertEqual(
            metrics["memory_propose_ranked_first"], 0.0,
            "a proposal won a pick on a run with no memory term, so something other "
            "than eq. 26 is choosing between the two diverts",
        )
