"""``audio/bed.py`` — the properties §2.4 and §3.1 are built on top of.

Three of these are not ordinary unit tests. They are the *premises* of decisions made
elsewhere in the map, checked here so that a later edit cannot quietly remove one:

- the bed is diotic, so it contributes nothing to the lateral cue;
- the bed is position-invariant, so §2.4's absolute threshold is sound;
- before ``t_anom`` the heard signal is the bed **exactly**, so §3.1's provenance
  assertion has content rather than slack.
"""

import unittest

import numpy as np

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.bed import BED_SEED, bed_signal, heard_signal, mix_bed
from earshot.audio.clips import rms
from earshot.audio.lateral import lateral_sign


class TestBedSignal(unittest.TestCase):
    def test_the_level_is_exact_not_in_expectation(self):
        """What lets `pre_onset_rms_tol` be a drift tolerance rather than a slack budget."""
        bed = bed_signal(4096, 1e-3)
        self.assertAlmostEqual(rms(bed), 1e-3, places=9)

    def test_it_is_diotic(self):
        bed = bed_signal(1024, 0.01)
        np.testing.assert_array_equal(bed[0], bed[1])
        self.assertEqual(lateral_sign(bed), 0)

    def test_it_is_reproducible(self):
        np.testing.assert_array_equal(bed_signal(256, 0.1), bed_signal(256, 0.1, BED_SEED))

    def test_a_zero_level_is_silence_rather_than_a_divide(self):
        self.assertEqual(rms(bed_signal(64, 0.0)), 0.0)

    def test_bad_arguments_are_rejected(self):
        with self.assertRaises(ValueError):
            bed_signal(0, 0.1)
        with self.assertRaises(ValueError):
            bed_signal(64, -1.0)


class TestMixBed(unittest.TestCase):
    def test_mixing_adds(self):
        rendered = np.ones((2, 8), dtype=np.float32)
        mixed = mix_bed(rendered, np.full((2, 8), 0.5, dtype=np.float32))
        np.testing.assert_allclose(mixed, 1.5)

    def test_a_length_mismatch_is_a_loud_failure(self):
        """Tiling or cropping would silently move the RMS the threshold was set against."""
        with self.assertRaises(ValueError) as caught:
            mix_bed(np.ones((2, 8)), np.ones((2, 9)))
        self.assertIn("clip's length", str(caught.exception))


class TestHeardSignal(unittest.TestCase):
    def setUp(self):
        self.clip = np.ones(256, dtype=np.float32) * 0.1
        self.bed = bed_signal(256, 1e-3)

    def test_before_t_anom_the_heard_signal_is_the_bed_exactly(self):
        """§3.1's invariant is only assertable because this is equality, not similarity."""
        heard = heard_signal(fakes.synthetic_ir(), self.clip, self.bed, playing=False)
        np.testing.assert_array_equal(heard, self.bed)
        self.assertAlmostEqual(rms(heard), 1e-3, places=9)

    def test_the_anomaly_contributes_exactly_zero_before_it_plays(self):
        """Not a scaled-down render, not a fade — the clip is simply not mixed in."""
        quiet = heard_signal(fakes.synthetic_ir(left=1.0), self.clip, self.bed, playing=False)
        loud = heard_signal(fakes.synthetic_ir(left=1000.0), self.clip, self.bed, playing=False)
        np.testing.assert_array_equal(quiet, loud)

    def test_playing_adds_the_rendered_anomaly(self):
        heard = heard_signal(fakes.synthetic_ir(), self.clip, self.bed, playing=True)
        self.assertEqual(heard.shape, (2, 256))
        self.assertGreater(rms(heard), rms(self.bed))

    def test_the_bed_is_position_invariant_where_the_anomaly_is_not(self):
        """The whole of §2.4: the spatial swing lives entirely in the post-onset term."""
        near = heard_signal(fakes.synthetic_ir(left=1.0, right=1.0), self.clip, self.bed, playing=True)
        far = heard_signal(
            fakes.synthetic_ir(left=0.01, right=0.01), self.clip, self.bed, playing=True
        )
        self.assertGreater(rms(near), 10.0 * rms(far))
        silent_near = heard_signal(fakes.synthetic_ir(left=1.0), self.clip, self.bed, playing=False)
        silent_far = heard_signal(fakes.synthetic_ir(left=0.01), self.clip, self.bed, playing=False)
        self.assertEqual(rms(silent_near), rms(silent_far))

    def test_the_bed_does_not_move_the_lateral_cue(self):
        """A source on the right stays on the right once the bed is mixed in."""
        heard = heard_signal(
            fakes.synthetic_ir(left=1.0, right=6.0), self.clip, self.bed, playing=True
        )
        self.assertEqual(lateral_sign(heard), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
