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


class TestTwoBedsAtTwoLengths(unittest.TestCase):
    """ADR-0019 decision 1: the runner builds TWO beds, and never slices one from the
    other. This is the arm that makes that decision evidence rather than a preference.

    ``tail.heard_step`` composes at ``hop`` and ``tail.heard_clip_window`` at
    ``len(clip)``. The smaller diff was to keep one clip-length bed and slice its last
    ``hop`` samples for the cue; the slice is out of ``AudioConfig.pre_onset_rms_tol`` at
    two of the three configurations this tree ships tests at.
    """

    TOLERANCE = 0.05  # AudioConfig.pre_onset_rms_tol

    def test_a_bed_built_at_its_own_length_is_exact_at_every_length(self):
        """THE HEALTHY ARM. Normalising after generation is what makes the tolerance a
        bound on DRIFT rather than a slack budget for sampling noise."""
        for n_samples in (100, 441, 800, 2205, 44100, 220500):
            self.assertAlmostEqual(rms(bed_signal(n_samples, 1e-3)), 1e-3, places=9)

    def test_a_slice_of_the_clip_bed_is_out_of_tolerance_where_its_own_bed_is_not(self):
        """THE FORCED-FAILURE ARM -- the rejected alternative, measured.

        A slice of ``n`` Gaussian samples carries a relative RMS error of about
        ``1/sqrt(2n)``, and ``n`` is ``hop``, a free parameter. Measured against the fixed
        ``BED_SEED``: at the shipped 220500/44100 the worst disjoint hop-slice deviates
        0.3107% (harmless); at the runner fixture's 2205/441 it deviates 6.7906%; at the
        tail fixture's 800/100 it deviates 17.7320%. Against a 5% tolerance the last two
        would raise ``ProvenanceError`` on the pre-onset step -- which is §3.1's first
        invariant, so a bed built by slicing would take out the assertion that exists to
        catch a fabricated signal.

        The cost scales the wrong way, which is the reason this is a decision rather than
        a tuning: the smaller the step, the worse the slice gets.
        """
        expected = {(220500, 44100): 0.0031, (2205, 441): 0.0679, (800, 100): 0.1773}
        for (window, hop), worst_expected in sorted(expected.items()):
            long_bed = bed_signal(window, 1e-3)
            worst = max(
                abs(rms(long_bed[:, start : start + hop]) - 1e-3) / 1e-3
                for start in range(0, window - hop + 1, hop)
            )
            own = abs(rms(bed_signal(hop, 1e-3)) - 1e-3) / 1e-3
            print(
                "\n  [bed] {}/{}: worst disjoint slice {:.4%}  own bed {:.3e}".format(
                    window, hop, worst, own),
                flush=True,
            )
            self.assertAlmostEqual(worst, worst_expected, places=3, msg=str(hop))
            self.assertLess(own, 1e-6, str(hop))
            if hop <= 441:
                self.assertGreater(worst, self.TOLERANCE, str(hop))
            else:
                # the shipped hop is the one configuration a slice would survive, which
                # is exactly why the tests below it are the ones that matter
                self.assertLess(worst, self.TOLERANCE, str(hop))

    def test_the_two_beds_are_not_sample_aligned_and_nothing_may_compare_them(self):
        """The stated NON-property. Same seed, so the same draws; different scaling, so
        different samples. The cue bed feeds the onset and the clip bed feeds CLAP, and
        neither is diffed against the other -- but a future edit that assumed alignment
        would find the first few samples agreeing and the rest not."""
        cue_bed = bed_signal(441, 1e-3)
        clip_bed = bed_signal(2205, 1e-3)
        self.assertFalse(np.array_equal(cue_bed, clip_bed[:, :441]))
        # ...and they are not merely rescaled copies of one another either
        ratio = clip_bed[0, :441] / cue_bed[0, :441]
        self.assertLess(float(np.std(ratio)), 1e-5)  # same draws
        self.assertNotAlmostEqual(float(np.mean(ratio)), 1.0, places=3)  # own scaling


class TestMixBed(unittest.TestCase):
    def test_mixing_adds(self):
        rendered = np.ones((2, 8), dtype=np.float32)
        mixed = mix_bed(rendered, np.full((2, 8), 0.5, dtype=np.float32))
        np.testing.assert_allclose(mixed, 1.5)

    def test_a_length_mismatch_is_a_loud_failure(self):
        """Tiling or cropping would silently move the RMS the threshold was set against."""
        with self.assertRaises(ValueError) as caught:
            mix_bed(np.ones((2, 8)), np.ones((2, 9)))
        self.assertIn("(2, 9)", str(caught.exception))
        self.assertIn("(2, 8)", str(caught.exception))

    def test_a_cue_length_signal_against_a_clip_length_bed_names_both_lengths(self):
        """The guard ADR-0019 leans on: it is what catches ``heard_step(bed_cue=bed_clip)``.

        The message has to name BOTH lengths, because "the bed is the wrong length" is not
        actionable while "the bed is 2205 and the signal is 441" says immediately which of
        the two readouts the caller meant.
        """
        with self.assertRaises(ValueError) as caught:
            mix_bed(np.ones((2, 441), dtype=np.float32), bed_signal(2205, 1e-3))
        message = str(caught.exception)
        self.assertIn("441", message)
        self.assertIn("2205", message)
        self.assertIn("bed_cue", message)
        self.assertIn("bed_clip", message)


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
