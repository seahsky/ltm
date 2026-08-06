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

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.audit import EpisodeAudit, FunnelStage, OnsetRecord, StepRecord
from earshot.tools.detour_report import (
    ABANDONED,
    NO_DETOUR,
    REACHED,
    aggregate,
    format_report,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
