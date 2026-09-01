"""The power arithmetic, and the two mistakes it exists to stop.

**Mistake one: 2 sigma called an MDE.** The first table put in front of the n-per-cell decision
used a 2.0 multiplier. Two sigma is the significance threshold; an effect exactly there is
detected half the time. At 80% power the multiplier is 2.80, so that table understated every
requirement by about 40%.

**Mistake two: the PAIRED formula used on an UNPAIRED design.** "MDE = 15 episodes = 4.1 points
at n=365" comes from `SD = sqrt(flip_rate * n)`, which is valid because `episode_diff` compares
the same episode in two arms. Cells of the 2x2 share no episodes. Carrying that number across
would claim about 1.8x the sensitivity the design has -- 5.9 points paired against 10.4
unpaired at the same total rendering cost. That ratio was asserted as "three times" before it
was computed, so it is pinned below rather than described.
"""

import contextlib
import io
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.power import (
    HISTORICAL_EFFECT,
    MEASURED_FLIP_RATE,
    episodes_for_mde,
    main,
    mde_between_cells,
    mde_paired,
    sd_between_cells,
    sd_paired,
    sign_test_threshold,
)


class TestTheMultiplierIsNotTwo(unittest.TestCase):
    def test_the_mde_is_two_point_eight_sigma_not_two(self):
        sd = sd_between_cells(200)
        mde = mde_between_cells(200)
        self.assertAlmostEqual(mde / sd, 2.802, places=2)
        self.assertGreater(mde, 2.0 * sd)

    def test_fifty_percent_power_recovers_the_significance_threshold(self):
        """At power 0.5 the z_beta term vanishes, leaving 1.96 sigma. The sanity anchor."""
        sd = sd_between_cells(200)
        self.assertAlmostEqual(mde_between_cells(200, power=0.5) / sd, 1.96, places=2)

    def test_the_headline_choice_detects_the_historical_effect(self):
        """200 per cell was chosen on 2026-08-21. This is the check that it is enough.

        The M3 revisit headline was +0.171. The MDE at 200/cell is 0.140, so the effect this
        project has actually measured sits above the bar with margin.
        """
        self.assertLess(mde_between_cells(200), HISTORICAL_EFFECT)
        self.assertGreater(mde_between_cells(90), HISTORICAL_EFFECT)


class TestBetweenCells(unittest.TestCase):
    def test_the_sd_halves_when_n_quadruples(self):
        self.assertAlmostEqual(
            sd_between_cells(400), sd_between_cells(100) / 2.0, places=9
        )

    def test_p_of_one_half_is_the_worst_case(self):
        """The default must be the pessimistic one or the MDE flatters itself."""
        worst = sd_between_cells(200, p=0.5)
        for p in (0.1, 0.3, 0.7, 0.9):
            self.assertLess(sd_between_cells(200, p=p), worst)

    def test_episodes_for_mde_inverts_mde_between_cells(self):
        for target in (0.05, 0.10, 0.14, 0.20):
            n = episodes_for_mde(target)
            self.assertLessEqual(mde_between_cells(n), target)
            # And it is not wasteful: one episode fewer would miss.
            self.assertGreater(mde_between_cells(n - 1), target)

    def test_a_smaller_target_costs_more_episodes(self):
        self.assertGreater(episodes_for_mde(0.05), episodes_for_mde(0.10))

    def test_impossible_inputs_raise(self):
        with self.assertRaises(ValueError):
            sd_between_cells(0)
        with self.assertRaises(ValueError):
            mde_between_cells(200, alpha=0.0)
        with self.assertRaises(ValueError):
            mde_between_cells(200, power=1.0)
        with self.assertRaises(ValueError):
            episodes_for_mde(0.0)


class TestThePairedFormulaIsDifferent(unittest.TestCase):
    def test_it_reproduces_the_recorded_sd_at_n_365(self):
        """`SD(difference) 7.7` is on record from the byte-identical re-run. Anchor it."""
        self.assertAlmostEqual(sd_paired(365, MEASURED_FLIP_RATE), 7.69, places=2)

    def test_the_paired_design_is_more_sensitive_and_by_how_much(self):
        """Pinned, not described. The first estimate of this ratio was "3x" and was wrong."""
        paired = mde_paired(365)
        unpaired = mde_between_cells(365)
        self.assertAlmostEqual(100 * paired, 5.90, places=1)
        self.assertAlmostEqual(100 * unpaired, 10.37, places=1)
        self.assertAlmostEqual(unpaired / paired, 1.76, places=2)

    def test_a_flip_rate_of_zero_makes_any_difference_detectable(self):
        self.assertEqual(sd_paired(365, 0.0), 0.0)


class TestTheSceneLevelTest(unittest.TestCase):
    def test_ten_scenes_need_nine_to_agree(self):
        """The matrix's actual position, and the reason more episodes do not help."""
        self.assertEqual(sign_test_threshold(10), 9)

    def test_more_scenes_lower_the_required_fraction(self):
        ten = sign_test_threshold(10) / 10
        forty = sign_test_threshold(40) / 40
        self.assertLess(forty, ten)

    def test_a_tiny_scene_count_cannot_reach_significance_at_all(self):
        """NOT_RUN is red, and so is a test no outcome can pass. It must say so."""
        self.assertIsNone(sign_test_threshold(5))

    def test_the_threshold_is_always_a_majority(self):
        for n_scenes in (10, 15, 20, 30, 40):
            threshold = sign_test_threshold(n_scenes)
            self.assertIsNotNone(threshold)
            self.assertGreater(threshold, n_scenes / 2)
            self.assertLessEqual(threshold, n_scenes)

    def test_a_zero_scene_count_raises(self):
        with self.assertRaises(ValueError):
            sign_test_threshold(0)


class TestTheCliPricesTheSweepInFrontOfIt(unittest.TestCase):
    """`ablation_sweep.sh` prints its own MDE before it spends ten hours earning it.

    The paired block was fixed at the n=365 of the run that measured the flip rate, so a
    driver could only quote its own number in a comment -- which is the shape of every
    figure ADR-0018 spent three amendments arguing about. `--paired-n` and `--n-scenes`
    exist so the driver prints what it is buying, and these tests hold the two numbers
    the ablation sweep's header states.
    """

    def _run(self, *argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = main(list(argv))
        self.assertEqual(code, 0)
        return buffer.getvalue()

    def test_paired_n_prints_the_sweeps_own_mde_and_not_only_the_historic_one(self):
        out = self._run("--n-per-cell", "285", "--paired-n", "285")
        self.assertIn("THIS SWEEP, paired at n=285", out)
        # 6.68 points is what `ablation_sweep.sh`'s header claims at 15 episodes over 19
        # scenes. If this moves, that header is wrong and this test is how it is found.
        self.assertIn("MDE {:.2f} points".format(100 * mde_paired(285)), out)
        self.assertIn("6.68", out)
        # The historic reference stays, and stays labelled as the other thing.
        self.assertIn("n=365", out)

    def test_the_paired_block_is_silent_about_this_sweep_when_not_asked(self):
        out = self._run("--n-per-cell", "285")
        self.assertNotIn("THIS SWEEP", out)
        self.assertIn("n=365", out)

    def test_n_scenes_adds_its_own_row_without_dropping_the_reference_rows(self):
        out = self._run("--n-scenes", "19")
        self.assertIn("15 of 19", out)
        for reference in ("9 of 10", "12 of 15", "15 of 20"):
            self.assertIn(reference, out)

    def test_a_scene_count_already_in_the_table_is_not_printed_twice(self):
        out = self._run("--n-scenes", "20")
        self.assertEqual(out.count("15 of 20"), 1)


if __name__ == "__main__":
    unittest.main()
