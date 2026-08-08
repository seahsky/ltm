"""The subtraction that scores a change, against injected records.

`eps-1` came back GREEN at 33% source-reached with no arm beside it, and the one thing
that would have scored it — `yield-2`'s funnel, built by the same builder — was a
subtraction nobody had done. This is that arithmetic, and the property that matters most
is not the subtraction: it is the REFUSAL to subtract two things that are not an arm-pair.
`detour-1`'s 8/20 was compared against a later run across an intervening builder change
once already, and the scene labels matched while the episode sets did not.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.audit import FunnelStage
from earshot.tools.funnel_diff import FLIP_RATE, diff, format_report


def summary(scene, built, *, reached, resumed=None, onset=None):
    """One scene's `summary.json`, with the funnel filled down the ladder.

    The stages NEST (`FunnelStage` is an IntEnum for that reason), so a fixture that set
    SOURCE_REACHED above INVESTIGATE_ENTERED would be describing a run that cannot happen
    and would let a bug in the pooling pass.
    """
    entered = onset if onset is not None else built
    return {
        "scene": scene,
        "n_episodes": built,
        "funnel": {
            FunnelStage.RUN.name: built,
            FunnelStage.T_ANOM_REACHED.name: built,
            FunnelStage.ONSET_FIRED.name: entered,
            FunnelStage.INVESTIGATE_ENTERED.name: entered,
            FunnelStage.SOURCE_REACHED.name: reached,
            FunnelStage.PRIMARY_RESUMED.name: reached if resumed is None else resumed,
        },
    }


class TestTheSubtraction(unittest.TestCase):
    def test_the_headline_is_source_reached_and_the_sign_is_the_change_helping(self):
        agg = diff([summary("A", 20, reached=5)], [summary("A", 20, reached=9)])
        self.assertEqual(agg["before"], 5)
        self.assertEqual(agg["after"], 9)
        self.assertEqual(agg["delta"], 4)
        self.assertAlmostEqual(agg["delta_rate"], 0.2)

    def test_a_change_that_hurt_reads_negative(self):
        """The arm this tool exists for is as likely to be the losing one."""
        agg = diff([summary("A", 20, reached=8)], [summary("A", 20, reached=3)])
        self.assertEqual(agg["delta"], -5)

    def test_the_whole_ladder_is_reported_not_just_the_headline(self):
        """A funnel that moved at stage 3 and not at stage 5 is a different finding from
        one that moved at 5, and a single number cannot tell them apart."""
        agg = diff(
            [summary("A", 20, reached=5, onset=20)],
            [summary("A", 20, reached=5, onset=12)],
        )
        by_stage = {row["stage"]: row for row in agg["stages"]}
        self.assertEqual(by_stage[FunnelStage.ONSET_FIRED.name]["delta"], -8)
        self.assertEqual(by_stage[FunnelStage.SOURCE_REACHED.name]["delta"], 0)

    def test_another_stage_can_be_the_headline(self):
        agg = diff(
            [summary("A", 20, reached=5, resumed=5)],
            [summary("A", 20, reached=5, resumed=2)],
            stage=FunnelStage.PRIMARY_RESUMED,
        )
        self.assertEqual(agg["stage"], FunnelStage.PRIMARY_RESUMED.name)
        self.assertEqual(agg["delta"], -3)


class TestWhatItRefusesToSubtract(unittest.TestCase):
    """THE FORCED-FAILURE ARM. A tool that happily subtracts mismatched sets is worse than
    no tool: it produces a number, and a number gets quoted."""

    def test_a_scene_built_to_a_different_count_is_unpaired_and_excluded(self):
        agg = diff(
            [summary("A", 20, reached=8), summary("B", 20, reached=4)],
            [summary("A", 17, reached=3), summary("B", 20, reached=6)],
        )
        self.assertEqual(agg["n_paired_scenes"], 1)
        self.assertEqual(agg["built"], 20)
        self.assertEqual(agg["delta"], 2, "only B is an arm-pair")
        self.assertEqual(len(agg["unpaired"]), 1)
        self.assertIn("different episode sets", agg["unpaired"][0]["reason"])

    def test_a_scene_missing_from_one_side_is_unpaired(self):
        agg = diff([summary("A", 20, reached=8)],
                   [summary("A", 20, reached=8), summary("B", 20, reached=6)])
        self.assertEqual(agg["n_paired_scenes"], 1)
        self.assertEqual(agg["unpaired"][0]["scene"], "B")
        self.assertIn("only in after", agg["unpaired"][0]["reason"])

    def test_a_zero_yield_scene_pairs_with_its_own_zero(self):
        """A scene that could not pose the task in either arm is 0 against 0, which is a
        legitimate pairing and belongs in the denominator's story rather than dropped."""
        agg = diff([summary("Z", 0, reached=0)], [summary("Z", 0, reached=0)])
        self.assertEqual(agg["n_paired_scenes"], 1)
        self.assertEqual(agg["built"], 0)
        self.assertIsNone(agg["delta_rate"], "no episodes is not a rate of zero")

    def test_two_runs_pooled_under_one_tag_raise(self):
        """The failure `yield-1` shipped: two invocations in one directory, whose funnel
        is neither run's."""
        with self.assertRaises(ValueError) as caught:
            diff([summary("A", 20, reached=8), summary("A", 20, reached=5)],
                 [summary("A", 20, reached=8)])
        self.assertIn("twice", str(caught.exception))


class TestTheReport(unittest.TestCase):
    def test_the_unpaired_scenes_are_named_in_the_output(self):
        text = format_report(diff(
            [summary("A", 20, reached=8), summary("B", 20, reached=4)],
            [summary("A", 17, reached=3), summary("B", 20, reached=6)],
        ))
        self.assertIn("NOT SUBTRACTED", text)
        self.assertIn("A", text)

    def test_the_flip_noise_is_printed_beside_the_total(self):
        """A delta inside the renderer's own run-to-run variation is not a result, and the
        reader is told the size of that variation rather than expected to recall it."""
        text = format_report(diff([summary("A", 20, reached=8)],
                                  [summary("A", 20, reached=9)]))
        self.assertIn("{:.0%}".format(FLIP_RATE), text)
        self.assertIn("not paired episodes", text)

    def test_the_labels_name_the_two_runs(self):
        text = format_report(
            diff([summary("A", 20, reached=8)], [summary("A", 20, reached=9)]),
            labels=("yield-2", "eps-1"))
        self.assertIn("yield-2", text)
        self.assertIn("eps-1", text)
