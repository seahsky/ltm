"""The windowed rise test, and the noise floor it is judged against.

`detour-2` is the run behind this file. It measured 325 of 336 abandoned-arm plateau
windows as a SINGLE step with zero travel, while the windows that could be fitted put the
cue well above its own scatter (sig/sc 6.12 and 7.41). Those two facts together say the
cue was there and the test missed it — so what is asserted here is the *forced failure*
as much as the healthy path (ADR-0014): a flat field with render noise on it, where the
old single-step rule turns and the windowed rule holds forward.

The numbers are not invented. `eps` in the fixtures is 2.8e-3, the residual SD
`detour-2` printed, and the old `1e-6` appears verbatim as the control arm.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_TURN_LEFT,
    RISING_WINDOW,
    is_rising,
    realizable_investigate_step,
)
from earshot.audio.calibration import (
    CalibrationError,
    calibrate_onset,
    render_scatter_of,
    sweep_render_scatter,
)
from earshot.tools.detour_report import _length_histogram

# provenance: measured — the residual SD `detour-2` reported for both arms (2.8e-3
# abandoned, 3.0e-3 reached). The old threshold, three thousand times smaller.
MEASURED_SCATTER = 2.8e-3
OLD_EPS = 1e-6


class RisingWindowTest(unittest.TestCase):
    """The predicate itself, on series built to isolate one behaviour each."""

    def test_short_history_degrades_to_the_old_comparison(self):
        """With one prior reading the median IS that reading — early steps unchanged."""
        self.assertTrue(is_rising([0.010, 0.020], eps=MEASURED_SCATTER))
        self.assertFalse(is_rising([0.020, 0.010], eps=MEASURED_SCATTER))

    def test_no_history_probes_forward(self):
        self.assertTrue(is_rising([], eps=MEASURED_SCATTER))
        self.assertTrue(is_rising([0.01], eps=MEASURED_SCATTER))

    def test_one_unlucky_reading_no_longer_stops_a_live_climb(self):
        """THE FORCED FAILURE ARM. A rising series with a single low sample in it.

        The old rule compares against that one sample's successor and answers "not
        rising" on the tick after the dip. The windowed rule sees a median of the last
        five and holds. This is the 1-step, zero-travel window, reproduced.
        """
        climbing = [0.010, 0.012, 0.014, 0.016, 0.018]
        dipped = climbing + [0.0125]        # one bad render
        recovered = dipped + [0.020]        # the climb, resumed

        # the control: the single-step rule turns on the tick after the dip
        self.assertFalse(is_rising(dipped[-2:], eps=OLD_EPS, window=1))
        # the fix: judged against the window, the recovery clears the median
        self.assertTrue(is_rising(recovered, eps=MEASURED_SCATTER))

    def test_a_flat_field_with_noise_does_not_read_as_rising(self):
        """The other forced failure: noise on a DEAD gradient must not answer FORWARD.

        A window that smoothed everything into "keep going" would trade a false turn for
        a false forward, which is worse — the agent would never turn toward the source.
        """
        flat = [0.0100, 0.0102, 0.0098, 0.0101, 0.0099, 0.0103]
        self.assertFalse(is_rising(flat, eps=MEASURED_SCATTER))
        # and the old threshold is exactly the coin flip this replaces: the same series,
        # judged at 1e-6, calls a 2e-4 wobble a rise.
        self.assertTrue(is_rising(flat, eps=OLD_EPS, window=1))

    def test_a_real_rise_still_clears_the_noise_floor(self):
        """The healthy path. A climb steeper than the scatter is still a climb."""
        climbing = [0.010, 0.014, 0.018, 0.022, 0.026, 0.030]
        self.assertTrue(is_rising(climbing, eps=MEASURED_SCATTER))

    def test_a_rise_smaller_than_the_scatter_is_refused(self):
        """A gradient under the noise floor is not evidence, and is not treated as any.

        This is the property that makes `eps` worth deriving: the same series is a rise
        at the old threshold and is not one at the measured floor.
        """
        creeping = [0.01000, 0.01005, 0.01010, 0.01015, 0.01020, 0.01025]
        self.assertFalse(is_rising(creeping, eps=MEASURED_SCATTER))
        self.assertTrue(is_rising(creeping, eps=OLD_EPS))

    def test_the_window_bounds_how_far_back_the_baseline_reaches(self):
        """A long-dead history must not keep a stalled agent walking forward."""
        series = [0.001] * 20 + [0.010, 0.010, 0.010, 0.010, 0.010, 0.010]
        # the ancient quiet readings are outside the window, so the recent flat run wins
        self.assertFalse(is_rising(series, eps=MEASURED_SCATTER, window=RISING_WINDOW))
        # reaching back far enough to include them would call this a rise
        self.assertTrue(is_rising(series, eps=MEASURED_SCATTER, window=20))


class RisingDrivesTheRuleTest(unittest.TestCase):
    """The predicate reaches the action, and `eps` reaches the predicate."""

    def test_a_noisy_flat_field_turns_instead_of_walking(self):
        flat = [0.0100, 0.0102, 0.0098, 0.0101, 0.0099, 0.0103]
        self.assertEqual(
            realizable_investigate_step(flat, -1, False, eps=MEASURED_SCATTER),
            ACT_TURN_LEFT)

    def test_the_same_field_walked_forward_at_the_old_threshold(self):
        """The control run. Without this arm the fix is a claim rather than a measurement."""
        flat = [0.0100, 0.0102, 0.0098, 0.0101, 0.0099, 0.0103]
        self.assertEqual(
            realizable_investigate_step(flat, -1, False, eps=OLD_EPS, window=1),
            ACT_FORWARD)


class RenderScatterTest(unittest.TestCase):
    """Where `eps` comes from: repeats at ONE pose, never the 16-pose sweep."""

    def test_scatter_is_the_sample_sd_of_the_repeats(self):
        # sample (n-1) SD of these three is exactly 1e-3
        self.assertAlmostEqual(
            render_scatter_of([0.009, 0.010, 0.011]), 1e-3, places=12)

    def test_a_renderer_that_agrees_with_itself_reports_zero_not_a_floor(self):
        """Zero is a finding, so it is returned rather than clamped away."""
        self.assertEqual(render_scatter_of([0.01, 0.01, 0.01]), 0.0)

    def test_one_sample_has_no_spread_and_raises(self):
        with self.assertRaises(CalibrationError):
            render_scatter_of([0.01])

    def test_the_sweep_holds_the_pose_fixed(self):
        """The whole point: distance is constant, so what varies is the renderer."""
        seen = []
        impulse = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        def render_at(pose):
            seen.append(pose)
            return impulse

        sweep_render_scatter("one-pose", render_at, [0.5, 0.5, 0.5], repeats=3)
        self.assertEqual(seen, ["one-pose"] * 3)

    def test_fewer_than_two_repeats_is_refused(self):
        with self.assertRaises(CalibrationError):
            sweep_render_scatter("p", lambda p: p, [0.5], repeats=1)

    def test_calibration_carries_the_scatter_without_moving_the_threshold(self):
        without = calibrate_onset(0.001, [0.01, 0.02, 0.03])
        with_scatter = calibrate_onset(
            0.001, [0.01, 0.02, 0.03], scatter_samples=[0.009, 0.010, 0.011])
        self.assertEqual(without.onset_rms, with_scatter.onset_rms)
        self.assertIsNone(without.render_scatter)
        self.assertEqual(without.scatter_repeats, 0)
        self.assertAlmostEqual(with_scatter.render_scatter, 1e-3, places=12)
        self.assertEqual(with_scatter.scatter_repeats, 3)

    def test_unmeasured_scatter_is_none_and_never_zero(self):
        """`None` and `0.0` mean opposite things and the record must not conflate them."""
        result = calibrate_onset(0.001, [0.01, 0.02, 0.03], scatter_samples=[])
        self.assertIsNone(result.as_dict()["render_scatter"])


class LengthHistogramTest(unittest.TestCase):
    """What the median was hiding: `detour-2` read 1 step while 60% of steps plateaued."""

    def test_every_window_lands_in_exactly_one_bucket(self):
        windows = [{"n_steps": n} for n in [1] * 325 + [2, 3, 4, 7, 8, 12, 15, 40, 60]]
        histogram = _length_histogram(windows)
        self.assertEqual(sum(count for _label, count in histogram), len(windows))
        self.assertEqual(
            histogram,
            [("1", 325), ("2-4", 3), ("5-9", 2), ("10-19", 2), ("20+", 2)])

    def test_the_tail_the_median_could_not_show_is_separable(self):
        """A hail of 1s and a handful of long stalls must not read as one number."""
        flickering = _length_histogram([{"n_steps": 1}] * 100)
        stalled = _length_histogram([{"n_steps": 60}] * 100)
        self.assertEqual(dict(flickering)["1"], 100)
        self.assertEqual(dict(flickering)["20+"], 0)
        self.assertEqual(dict(stalled)["1"], 0)
        self.assertEqual(dict(stalled)["20+"], 100)

    def test_no_windows_is_zeros_rather_than_an_absent_row(self):
        self.assertEqual(
            [count for _label, count in _length_histogram([])], [0, 0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
