"""``audio/calibration.py`` — where ``onset_rms`` comes from, and when it refuses.

The arithmetic is Mac-testable because the sweep injects its renderer. What is *not*
settled here is whether a real scene separates at all: that is the gate's job on the
box, and the number it prints is the deliverable.
"""

import math
import unittest

import numpy as np

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.calibration import (
    CalibrationError,
    band_poses,
    calibrate_onset,
    sweep_anomaly_rms,
)
from earshot.audio.clips import rms


class TestCalibrate(unittest.TestCase):
    def test_the_threshold_sits_strictly_between_the_two_distributions(self):
        result = calibrate_onset(0.001, [0.05, 0.08, 0.2, 0.11])
        self.assertGreater(result.onset_rms, result.bed_rms)
        self.assertLess(result.onset_rms, result.anomaly_low)

    def test_the_placement_is_geometric_so_the_margin_is_symmetric_in_db(self):
        """The arithmetic midpoint of 0.001 and 0.1 sits 34 dB above the bed and 6 dB
        below the anomaly, which is not "between" in the sense that matters."""
        result = calibrate_onset(0.001, [0.1] * 8)
        self.assertAlmostEqual(result.onset_rms, math.sqrt(0.001 * 0.1), places=9)
        up = 20.0 * math.log10(result.onset_rms / result.bed_rms)
        down = 20.0 * math.log10(result.anomaly_low / result.onset_rms)
        self.assertAlmostEqual(up, down, places=6)

    def test_the_gate_number_is_the_separation_in_db(self):
        result = calibrate_onset(0.001, [0.01] * 5)
        self.assertAlmostEqual(result.separation_db, 20.0, places=6)
        self.assertTrue(result.passed)
        self.assertEqual(result.n_poses, 5)

    def test_the_low_percentile_is_used_not_the_minimum(self):
        """One pose behind a closed door is §2.5's attrition, not the threshold's job."""
        samples = [0.0011] + [0.2] * 19
        result = calibrate_onset(0.001, samples)
        self.assertGreater(result.anomaly_low, 0.0011)
        self.assertEqual(result.anomaly_min, 0.0011)

    def test_overlap_fails_the_gate_and_names_the_correction(self):
        with self.assertRaises(CalibrationError) as caught:
            calibrate_onset(0.05, [0.01, 0.02, 0.04])
        message = str(caught.exception)
        self.assertIn("OVERLAP", message)
        self.assertIn("globalVolume", message)
        self.assertNotIn("lower the threshold", message)

    def test_too_little_daylight_fails_even_though_a_threshold_would_fit(self):
        """There is room for a threshold but not for a margin, so a small drift
        re-crosses it."""
        with self.assertRaises(CalibrationError) as caught:
            calibrate_onset(0.01, [0.015] * 6)
        self.assertIn("dB", str(caught.exception))
        self.assertIn("globalVolume", str(caught.exception))

    def test_a_failed_gate_raises_rather_than_returning_a_result(self):
        """A caller who can carry on past a failed gate is a caller who will."""
        with self.assertRaises(CalibrationError):
            calibrate_onset(0.05, [0.05])

    def test_an_empty_sweep_is_an_error_not_a_default(self):
        with self.assertRaises(CalibrationError):
            calibrate_onset(0.001, [])

    def test_a_zero_bed_is_rejected(self):
        with self.assertRaises(CalibrationError) as caught:
            calibrate_onset(0.0, [0.1] * 4)
        self.assertIn("bed_rms", str(caught.exception))

    def test_the_result_serialises_for_the_audit_record(self):
        record = calibrate_onset(0.001, [0.1] * 4).as_dict()
        self.assertEqual(record["n_poses"], 4)
        self.assertIn("separation_db", record)
        self.assertIn("global_volume", record)


class TestSweep(unittest.TestCase):
    def test_one_render_per_pose_measured_on_the_received_signal(self):
        """Not on the IR's own energy: the threshold is applied to what the agent hears,
        and the two differ by the clip's level."""
        rendered = []

        def render_at(pose):
            rendered.append(pose)
            return fakes.synthetic_ir(left=pose, right=pose)

        clip = np.ones(128, dtype=np.float32) * 0.1
        levels = sweep_anomaly_rms([1.0, 0.5, 0.25], render_at, clip)
        self.assertEqual(rendered, [1.0, 0.5, 0.25])
        self.assertEqual(len(levels), 3)
        self.assertGreater(levels[0], levels[1])
        self.assertGreater(levels[1], levels[2])

    def test_the_sweep_measures_the_anomaly_alone(self):
        """The bed is the other side of the comparison, so mixing it in here would
        compare the bed against itself."""
        clip = np.ones(64, dtype=np.float32)
        level = sweep_anomaly_rms([1.0], lambda _: fakes.synthetic_ir(), clip)[0]
        self.assertGreater(level, 0.0)
        self.assertNotAlmostEqual(level, rms(np.zeros(64)))


class TestBandPoses(unittest.TestCase):
    def test_the_band_is_log_spaced_between_the_endpoints(self):
        distances = band_poses((1.0, 8.0), 4)
        self.assertAlmostEqual(distances[0], 1.0)
        self.assertAlmostEqual(distances[-1], 8.0)
        ratios = [b / a for a, b in zip(distances, distances[1:])]
        for ratio in ratios[1:]:
            self.assertAlmostEqual(ratio, ratios[0], places=9)

    def test_linear_spacing_would_crowd_the_quiet_end(self):
        """The reason log spacing is not a stylistic choice."""
        distances = band_poses((1.0, 8.0), 4)
        self.assertLess(distances[1], (1.0 + 8.0) / 2.0)

    def test_a_bad_band_is_rejected(self):
        with self.assertRaises(ValueError):
            band_poses((8.0, 1.0), 4)
        with self.assertRaises(ValueError):
            band_poses((0.0, 8.0), 4)

    def test_a_distribution_needs_more_than_one_pose(self):
        with self.assertRaises(ValueError):
            band_poses((1.0, 8.0), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
