"""The 44.1 kHz to 48 kHz conversion CLAP requires, and the claim that it was not needed.

`ClapFeatureExtractor` raises on any rate but its own 48 kHz. It does NOT resample, though
`ClapEncoder.encode_audio`'s docstring asserted for months that it did. The assertion was
never false-alarmed because the weights had never been loaded on the box: `bootstrap_ss2.sh`
reports `clap_weights_loaded False` and `--clap` defaults off, so the separation gate was the
first caller to reach the line.

scipy is in the box's `ss2` env and nowhere near `mac-requirements.txt`, so the resample
itself cannot be tested here. The RATIO can, and it is the part worth testing: 44100 to 48000
is 160/147, and getting it wrong would silently change every duration CLAP sees while raising
nothing at all.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.config import AudioConfig
from earshot.task.models import CLAP_SAMPLE_RATE, resample_ratio


class TestResampleRatio(unittest.TestCase):
    def test_the_branch_rate_to_clap_is_160_over_147(self):
        self.assertEqual(resample_ratio(44100, 48000), (160, 147))

    def test_the_default_target_is_clap_s_own_rate(self):
        self.assertEqual(resample_ratio(44100), (160, 147))
        self.assertEqual(CLAP_SAMPLE_RATE, 48000)

    def test_matching_rates_are_a_no_op_pair(self):
        """`(1, 1)` so a caller can skip the resample without a second comparison."""
        self.assertEqual(resample_ratio(48000, 48000), (1, 1))

    def test_the_ratio_is_always_in_lowest_terms(self):
        import math

        for source in (8000, 16000, 22050, 32000, 44100, 44800, 96000):
            up, down = resample_ratio(source, CLAP_SAMPLE_RATE)
            self.assertEqual(math.gcd(up, down), 1, "{} -> {}/{}".format(source, up, down))
            # The pair must actually carry out the conversion it claims.
            self.assertAlmostEqual(source * up / down, CLAP_SAMPLE_RATE, places=6)

    def test_a_non_positive_rate_raises(self):
        """Raising, not defaulting: a rate of 0 would make the ratio meaningless."""
        with self.assertRaises(ValueError):
            resample_ratio(0, CLAP_SAMPLE_RATE)
        with self.assertRaises(ValueError):
            resample_ratio(44100, 0)
        with self.assertRaises(ValueError):
            resample_ratio(-44100, CLAP_SAMPLE_RATE)


class TestTheRendererKeepsItsOwnRate(unittest.TestCase):
    def test_audio_config_still_renders_at_the_branch_rate(self):
        """The renderer must NOT be moved to 48 kHz to suit a text-audio encoder.

        `AudioConfig.sample_rate` is the branch's own `sampleRate` and ESC-50 is 44.1 kHz.
        CLAP converts at its own boundary; changing this would change every IR in the tree
        and silently invalidate every calibration on disk.
        """
        self.assertEqual(AudioConfig().sample_rate, 44100)
        self.assertNotEqual(AudioConfig().sample_rate, CLAP_SAMPLE_RATE)


if __name__ == "__main__":
    unittest.main()
