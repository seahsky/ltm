"""The detour arithmetic, against injected records — the runs need a GPU, the maths does not.

`trace_one` and `aggregate` turn per-step audits into the one measurement that separates
yield-1's two candidate diagnoses: *were the twelve abandoned detours walking at the
source when the budget cut them off, or were they wandering?* Ticket 19's third row,
applied to a diagnosis rather than to a gate — given records that say the agent walked
twenty metres and closed one, does the report say so, or does the number come out looking
like a converging climb that ran out of steps.

The two arms are the point. A synthetic "abandoned" trace proves nothing on its own; the
reached ones are the control, and this asserts they are reported side by side (CLAUDE.md:
a claim that X failed because of Y needs the arm where Y is absent).
"""

import inspect
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_STOP,
    ACT_TURN_LEFT,
    ACT_TURN_RIGHT,
    realizable_investigate_step,
)
from earshot.report.audit import EpisodeAudit, FunnelStage, OnsetRecord, StepRecord
from earshot.tools.detour_report import (
    ABANDONED,
    NO_DETOUR,
    REACHED,
    RISING_EPS,
    aggregate,
    fit_slope,
    format_report,
    plateau_windows,
    rising_flags,
    rule_action,
    trace_one,
)
from earshot.types import Xyz

SOURCE = Xyz(0.0, 0.0, 0.0)
ONSET_STEP = 5


def steps(distances, *, onset=ONSET_STEP, displacement=0.25, collided=False):
    """One step per distance, laid out along +x so `horizontal_distance_to` is the number.

    `displacement` is what the agent actually moved that step, which is deliberately NOT
    derived from the distance series: an agent can walk a metre and end up no closer, and
    that gap between path length and progress is exactly what the report measures.
    """
    return tuple(
        StepRecord(step=onset + i, measured_rms=0.05, position=Xyz(float(d), 0.0, 0.0),
                   displacement_m=displacement, collided=collided)
        for i, d in enumerate(distances)
    )


def audit(distances, *, stage, index=0, **kwargs):
    return EpisodeAudit(
        episode_index=index,
        source_xyz=SOURCE,
        funnel_stage=stage,
        onset=OnsetRecord(onset_step=ONSET_STEP),
        steps=steps(distances, **kwargs),
    )


class TestOneDetour(unittest.TestCase):
    def test_a_climb_that_walked_straight_at_the_source_reads_near_one(self):
        """8 m closed over 8 m walked: short of steps, not lost."""
        row = trace_one(audit([10.0, 8.0, 6.0, 4.0, 2.0], stage=FunnelStage.INVESTIGATE_ENTERED,
                              displacement=2.0))
        self.assertEqual(row["outcome"], ABANDONED)
        self.assertAlmostEqual(row["gap_closed_m"], 8.0)
        self.assertAlmostEqual(row["walked_m"], 10.0)
        self.assertAlmostEqual(row["walked_per_metre_closed"], 1.25)

    def test_a_wander_reads_an_order_of_magnitude_higher(self):
        """Twenty steps of real movement, half a metre closed. No threshold needed to see it."""
        row = trace_one(audit([6.0] * 10 + [5.5] * 10, stage=FunnelStage.INVESTIGATE_ENTERED,
                              displacement=1.0))
        self.assertAlmostEqual(row["gap_closed_m"], 0.5)
        self.assertAlmostEqual(row["walked_m"], 20.0)
        self.assertAlmostEqual(row["walked_per_metre_closed"], 40.0)

    def test_a_stuck_agent_shows_it_in_walked_and_in_collisions(self):
        row = trace_one(audit([6.0] * 20, stage=FunnelStage.INVESTIGATE_ENTERED,
                              displacement=0.0, collided=True))
        self.assertEqual(row["walked_m"], 0.0)
        self.assertEqual(row["gap_closed_m"], 0.0)
        self.assertEqual((row["n_collided"], row["n_moves"]), (20, 20))
        self.assertIsNone(row["walked_per_metre_closed"],
                          "a gap of zero has no metres-per-metre; a number here would be "
                          "a division that invented progress")

    def test_reaching_the_source_is_the_other_arm(self):
        row = trace_one(audit([6.0, 3.0, 0.4], stage=FunnelStage.PRIMARY_RESUMED))
        self.assertEqual(row["outcome"], REACHED)
        self.assertTrue(row["walked_is_upper_bound"],
                        "the window overshoots into the resumed primary search, and the "
                        "report has to say so rather than quietly over-count the detour")

    def test_an_episode_that_never_diverted_has_no_detour(self):
        row = trace_one(audit([6.0], stage=FunnelStage.ONSET_FIRED))
        self.assertEqual(row["outcome"], NO_DETOUR)
        self.assertNotIn("walked_m", row)

    def test_the_budget_clips_the_window(self):
        """An abandoned detour ends at the budget by definition; steps after it are the
        resumed primary search and are not the detour's cost."""
        row = trace_one(audit([9.0, 8.0, 7.0, 6.0, 5.0], stage=FunnelStage.INVESTIGATE_ENTERED),
                        budget=2)
        self.assertEqual(row["detour_steps"], 3)  # onset, +1, +2
        self.assertAlmostEqual(row["d_end_m"], 7.0)

    def test_records_without_a_position_read_absent_rather_than_zero(self):
        """Every audit written before StepRecord.position. A distance of n/a and a
        distance of zero are different claims and only one is a measurement."""
        bare = EpisodeAudit(
            source_xyz=SOURCE,
            funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
            onset=OnsetRecord(onset_step=ONSET_STEP),
            steps=tuple(StepRecord(step=ONSET_STEP + i, measured_rms=0.05,
                                   displacement_m=0.25) for i in range(5)),
        )
        row = trace_one(bare)
        self.assertIsNone(row["d_onset_m"])
        self.assertIsNone(row["gap_closed_m"])
        self.assertAlmostEqual(row["walked_m"], 1.25)  # still knows it moved


class TestTheTwoArms(unittest.TestCase):
    def test_the_arms_are_counted_and_reported_separately(self):
        traces = [
            trace_one(audit([10.0, 9.5], stage=FunnelStage.INVESTIGATE_ENTERED, index=i,
                            displacement=3.0))
            for i in range(12)
        ] + [
            trace_one(audit([6.0, 3.0, 0.4], stage=FunnelStage.PRIMARY_RESUMED, index=12 + i))
            for i in range(8)
        ]
        agg = aggregate(traces)
        self.assertEqual(agg["n_episodes"], 20)
        self.assertEqual(agg["arms"][ABANDONED]["n"], 12)
        self.assertEqual(agg["arms"][REACHED]["n"], 8)
        self.assertGreater(agg["arms"][ABANDONED]["walked_per_metre_closed"],
                           agg["arms"][REACHED]["walked_per_metre_closed"])

    def test_an_empty_run_aggregates_to_nothing_rather_than_crashing(self):
        agg = aggregate([])
        self.assertEqual(agg["n_episodes"], 0)
        self.assertIsNone(agg["arms"][ABANDONED]["walked_m"])

    def test_the_report_names_both_arms_and_flags_missing_positions(self):
        bare = EpisodeAudit(source_xyz=SOURCE, funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
                            onset=OnsetRecord(onset_step=ONSET_STEP),
                            steps=(StepRecord(step=ONSET_STEP, measured_rms=0.05),))
        text = format_report(aggregate([
            trace_one(audit([10.0, 5.0], stage=FunnelStage.INVESTIGATE_ENTERED)),
            trace_one(audit([6.0, 0.4], stage=FunnelStage.PRIMARY_RESUMED, index=1)),
            trace_one(bare),
        ]))
        self.assertIn(ABANDONED, text)
        self.assertIn(REACHED, text)
        self.assertIn("carry no per-step position", text)


def rms_steps(pairs, *, onset=ONSET_STEP, action=None, realizable=None, lateral=0):
    """Steps from explicit ``(distance, measured_rms)`` pairs.

    The helper above pins ``measured_rms`` at a constant because the questions it serves
    are about distance. The plateau questions are about the *energy*, so this one lets a
    test say what the render did at each pose — which is the whole input to `rising`.

    ``realizable`` takes one action for every step, or a sequence for one per step. The
    per-step form is what the reconstruction check needs: the FIRST detour step has no
    predecessor in the window and so is always rising, and a test that hands it a turn is
    asserting against the rule rather than against the tool.
    """
    per_step = realizable if isinstance(realizable, (list, tuple)) else [realizable] * len(pairs)
    return tuple(
        StepRecord(step=onset + i, measured_rms=float(rms), position=Xyz(float(d), 0.0, 0.0),
                   displacement_m=0.25, lateral_sign=lateral, action=action,
                   realizable_action=per_step[i])
        for i, (d, rms) in enumerate(pairs)
    )


def rms_audit(pairs, *, stage=FunnelStage.INVESTIGATE_ENTERED, index=0, **kwargs):
    return EpisodeAudit(
        episode_index=index,
        source_xyz=SOURCE,
        funnel_stage=stage,
        onset=OnsetRecord(onset_step=ONSET_STEP),
        steps=rms_steps(pairs, **kwargs),
    )


class TestTheRulesOwnPredicate(unittest.TestCase):
    """`rising` and the turn it implies, carried rather than re-invented."""

    def test_the_epsilon_is_the_rules_own_default(self):
        """The drift guard. Two copies of this constant is two controllers."""
        signature = inspect.signature(realizable_investigate_step)
        self.assertEqual(RISING_EPS, signature.parameters["eps"].default)

    def test_the_first_reading_is_rising_and_a_repeat_is_not(self):
        self.assertEqual(rising_flags([0.05, 0.05]), [True, False])

    def test_a_rise_inside_epsilon_does_not_count(self):
        """The rule needs `current > previous + eps`, not merely `>`."""
        self.assertEqual(rising_flags([0.05, 0.05 + RISING_EPS / 2]), [True, False])
        self.assertEqual(rising_flags([0.05, 0.05 + RISING_EPS * 10]), [True, True])

    def test_the_rule_forwards_while_rising_and_turns_by_the_lateral_sign(self):
        self.assertEqual(rule_action(True, -1), ACT_FORWARD)
        self.assertEqual(rule_action(False, 1), ACT_TURN_RIGHT)
        self.assertEqual(rule_action(False, -1), ACT_TURN_LEFT)

    def test_an_ambiguous_or_absent_sign_scans_left_like_the_rule_does(self):
        self.assertEqual(rule_action(False, 0), ACT_TURN_LEFT)
        self.assertEqual(rule_action(False, None), ACT_TURN_LEFT)

    def test_the_reconstruction_matches_the_rule_it_reconstructs(self):
        """Both halves against the real function, not against a copy of its body."""
        history = [0.05, 0.05, 0.06]
        for i in range(1, len(history) + 1):
            window = history[:i]
            flag = rising_flags(window)[-1]
            self.assertEqual(
                rule_action(flag, -1),
                realizable_investigate_step(window, -1, False),
            )


class TestPlateauWindows(unittest.TestCase):
    def test_maximal_runs_of_not_rising_are_the_windows(self):
        self.assertEqual(plateau_windows([True, False, False, True, False]),
                         [(1, 3), (4, 5)])

    def test_a_run_that_reaches_the_end_is_closed(self):
        self.assertEqual(plateau_windows([True, False, False]), [(1, 3)])

    def test_an_always_rising_climb_has_no_plateau(self):
        self.assertEqual(plateau_windows([True, True, True]), [])

    def test_a_flat_trace_is_one_window_not_many(self):
        self.assertEqual(plateau_windows([False, False, False]), [(0, 3)])

    def test_rising_is_computed_over_the_episode_not_the_detour(self):
        """The specific bug: a window-local recompute calls the onset step rising.

        The detour starts at ONSET_STEP + 1 here and the reading has not changed since the
        step before it, so the agent was already plateaued when it diverted. Slicing first
        and recomputing second would invent a FORWARD the agent never took.
        """
        row = trace_one(rms_audit([(3.0, 0.05), (3.0, 0.05), (3.0, 0.05)]))
        self.assertEqual(row["n_plateaus"], 1)
        self.assertEqual(row["plateau_steps"], 2)
        self.assertEqual(row["plateaus"][0]["start_step"], ONSET_STEP + 1)


class TestFitSlope(unittest.TestCase):
    def test_a_clean_line_recovers_its_slope_and_scatters_nothing(self):
        slope, resid = fit_slope([1.0, 2.0, 3.0, 4.0], [0.10, 0.08, 0.06, 0.04])
        self.assertAlmostEqual(slope, -0.02)
        self.assertAlmostEqual(resid, 0.0)

    def test_two_points_fit_a_line_but_report_no_scatter(self):
        """A line through two points leaves nothing to scatter; None, never 0.0, which
        would read as a noiseless measurement rather than an unmeasurable one."""
        slope, resid = fit_slope([1.0, 2.0], [0.10, 0.08])
        self.assertAlmostEqual(slope, -0.02)
        self.assertIsNone(resid)

    def test_an_agent_that_never_moved_has_no_slope(self):
        self.assertEqual(fit_slope([2.0, 2.0, 2.0], [0.05, 0.09, 0.04]), (None, None))

    def test_one_point_is_not_a_fit(self):
        self.assertEqual(fit_slope([2.0], [0.05]), (None, None))


class TestTheTwoDiagnoses(unittest.TestCase):
    """The measurement this half exists for: is the cue exhausted, or missed?"""

    def test_a_real_plateau_reads_flat_while_the_agent_keeps_moving(self):
        """The agent closes 1.25 m of gap and the render does not budge. Cue exhausted.

        The window is the six steps AFTER the onset: the first detour step has nothing
        before it inside the episode and so is rising by the rule's own definition.
        """
        row = trace_one(rms_audit([(3.0, 0.05)] + [(3.0 - 0.25 * i, 0.05) for i in range(1, 7)]))
        window = row["plateaus"][0]
        self.assertEqual(window["n_steps"], 6)
        self.assertFalse(window["static"])
        self.assertAlmostEqual(window["slope_per_m"], 0.0)
        self.assertAlmostEqual(window["d_span_m"], 1.25)

    def test_a_spurious_plateau_keeps_a_negative_slope_under_the_jitter(self):
        """A live gradient the single-step test cannot see.

        The agent is being carried AWAY from the source and the render tracks it: quieter
        at every step, so `rising` is false at every step, so the rule answers a turn at
        every step and the whole stretch is one window. But regressed against distance the
        cue is unmistakable — louder near, quieter far, a clean negative slope well clear
        of its own scatter. This is the (ii) signature: the field was informative and the
        one-step comparison could not use it.
        """
        pairs = [(1.9, 0.0720),
                 (2.0, 0.0700), (2.2, 0.0679), (2.4, 0.0662),
                 (2.6, 0.0639), (2.8, 0.0621), (3.0, 0.0600)]
        row = trace_one(rms_audit(pairs))
        window = row["plateaus"][0]
        self.assertEqual(window["n_steps"], 6)
        self.assertLess(window["slope_per_m"], 0.0)
        self.assertAlmostEqual(window["d_span_m"], 1.0)
        # The rays-1 gate: the cue clears its own noise by a wide margin here, so a
        # windowed estimator would have found it and the ray count is not the lever.
        self.assertGreater(window["signal_to_scatter"], 1.0)

    def test_an_agent_turning_in_place_is_static_and_its_slope_is_withheld(self):
        """The ill-conditioned case, reported rather than regressed: no translation means
        no test of the field, and a slope fitted here would be noise wearing a number."""
        row = trace_one(rms_audit([(2.3, 0.05), (2.3, 0.06), (2.3, 0.04), (2.3, 0.07)]))
        window = row["plateaus"][0]
        self.assertTrue(window["static"])
        self.assertIsNone(window["slope_per_m"])
        self.assertIsNone(window["signal_to_scatter"])
        self.assertEqual(row["n_static_plateaus"], 1)

    def test_static_windows_are_counted_in_the_aggregate_not_dropped(self):
        agg = aggregate([trace_one(rms_audit([(2.3, 0.05)] * 4, index=0))])
        arm = agg["plateaus"][ABANDONED]
        self.assertEqual(arm["n_windows"], 1)
        self.assertEqual(arm["n_static"], 1)
        self.assertEqual(arm["n_fitted"], 0)


class TestTheReconstructionCheck(unittest.TestCase):
    """ADR-0014's both arms: the check passing, and the check firing."""

    def test_a_record_that_agrees_reads_as_fully_checked(self):
        """Rising on the first detour step, plateaued after: FORWARD then two turns."""
        row = trace_one(rms_audit(
            [(3.0, 0.05), (2.8, 0.05), (2.6, 0.05)],
            realizable=[ACT_FORWARD, ACT_TURN_LEFT, ACT_TURN_LEFT], lateral=-1))
        self.assertEqual(row["n_rule_checked"], 3)
        self.assertEqual(row["n_rule_agree"], 3)
        self.assertEqual(aggregate([row])["rule_check"]["agreement"], 1.0)

    def test_a_record_that_disagrees_is_caught_and_named(self):
        """Forced failure: the trace is plateaued so the rule must answer a turn, and the
        record says the cue answered FORWARD. Silence here would let every plateau above
        rest on a model of the controller no one had tested."""
        row = trace_one(rms_audit([(3.0, 0.05), (2.8, 0.05), (2.6, 0.05)],
                                  realizable=ACT_FORWARD, lateral=-1))
        self.assertEqual(row["n_rule_checked"], 3)
        self.assertEqual(row["n_rule_agree"], 1)  # the rising first step, and nothing else
        text = format_report(aggregate([row]))
        self.assertIn("DISAGREEMENT IS THE FINDING", text)

    def test_a_stop_is_excluded_by_name_rather_than_counted_as_a_disagreement(self):
        """The rule's STOP needs `visual_confirm`, which no record carries."""
        row = trace_one(rms_audit([(3.0, 0.05), (0.4, 0.05)],
                                  realizable=ACT_STOP, lateral=-1))
        self.assertEqual(row["n_rule_checked"], 0)
        self.assertEqual(row["n_rule_stop"], 2)

    def test_a_pre_field_record_reads_unvalidated_and_says_so(self):
        """yield-2's shape. 0/0 must never render as agreement."""
        agg = aggregate([trace_one(rms_audit([(3.0, 0.05), (2.8, 0.05)]))])
        self.assertEqual(agg["rule_check"]["n_checked"], 0)
        self.assertIsNone(agg["rule_check"]["agreement"])
        self.assertIn("RECONSTRUCTION UNVALIDATED", format_report(agg))

    def test_the_plateau_table_names_both_arms(self):
        text = format_report(aggregate([
            trace_one(rms_audit([(3.0, 0.05)] * 4, index=0)),
            trace_one(rms_audit([(3.0, 0.05), (0.4, 0.05)], index=1,
                                stage=FunnelStage.PRIMARY_RESUMED)),
        ]))
        self.assertIn("plateau windows", text)
        self.assertIn("sig/sc", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
