"""``audio/clap.py`` — the gate's logic against a stub encoder.

The **thresholds** are not this file's subject and cannot be: they were measured on the
box at EER 0.00 against RIR-convolved audio. What is checked here is that the gate uses
them, that it can say "normal" at all — the thing ``classify_anomaly`` structurally
cannot — and that the calibrated pair is what a caller gets without asking.
"""

import unittest

import numpy as np

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.clap import (
    AMBIGUOUS_CLASSES,
    ANOMALY_CLASSES,
    ANOMALY_GATE_DELTA,
    ANOMALY_GATE_TAU,
    CLASS_TO_CLAP_PROMPT,
    NORMAL_PROMPTS,
    classify_anomaly,
    heard_clip_for_clap,
    is_anomaly,
)


def encoder_favouring(prompt, strength=1.0, others=0.0):
    """A stub whose audio vector points at exactly one prompt."""
    vectors = {text: fakes.one_hot(1, scale=others) for text in CLASS_TO_CLAP_PROMPT.values()}
    vectors.update({text: fakes.one_hot(1, scale=others) for text in NORMAL_PROMPTS})
    vectors[prompt] = fakes.one_hot(0, scale=1.0)
    return fakes.FakeClapEncoder(fakes.one_hot(0, scale=strength), vectors)


class TestPromptBanks(unittest.TestCase):
    def test_every_class_has_a_prompt(self):
        for name in tuple(ANOMALY_CLASSES) + tuple(AMBIGUOUS_CLASSES):
            self.assertIn(name, CLASS_TO_CLAP_PROMPT)

    def test_the_calibrated_thresholds_are_the_measured_pair(self):
        """Negative delta is not a typo: on convolved audio CLAP's anomaly text margin
        IS negative, and the separation is carried by the small absolute floor."""
        self.assertAlmostEqual(ANOMALY_GATE_DELTA, -0.2557)
        self.assertAlmostEqual(ANOMALY_GATE_TAU, 0.0341)


class TestClassify(unittest.TestCase):
    def test_it_picks_the_closest_class_and_reports_every_cosine(self):
        encoder = encoder_favouring(CLASS_TO_CLAP_PROMPT["glass_break"])
        best, scores = classify_anomaly(np.zeros(16), 44100, encoder)
        self.assertEqual(best, "glass_break")
        self.assertEqual(sorted(scores), sorted(ANOMALY_CLASSES))
        self.assertEqual(encoder.seen_rates, [44100])

    def test_it_can_never_say_normal(self):
        """The reason `is_anomaly` exists: an argmax over anomaly prompts always
        returns one."""
        encoder = encoder_favouring("people talking")
        best, _ = classify_anomaly(np.zeros(16), 44100, encoder)
        self.assertIn(best, ANOMALY_CLASSES)


class TestOpenSetGate(unittest.TestCase):
    def test_it_fires_when_the_anomaly_side_wins(self):
        encoder = encoder_favouring(CLASS_TO_CLAP_PROMPT["alarm"])
        fired, best, scores = is_anomaly(
            np.zeros(16), 44100, encoder, delta=0.0, tau_abs=0.0
        )
        self.assertTrue(fired)
        self.assertEqual(best, "alarm")
        self.assertGreater(scores["margin"], 0.0)

    def test_a_benign_sound_is_heard_and_does_not_fire(self):
        """A merely loud sound must not consume the once-per-episode onset."""
        encoder = encoder_favouring("footsteps")
        fired, best, scores = is_anomaly(
            np.zeros(16), 44100, encoder, delta=0.0, tau_abs=0.0
        )
        self.assertFalse(fired)
        self.assertLess(scores["margin"], 0.0)
        self.assertIn(best, ANOMALY_CLASSES)  # still says what it would have called it

    def test_the_absolute_floor_can_veto_a_won_margin(self):
        floor = 2.0  # unreachable: cosines of unit vectors are at most 1.0
        encoder = encoder_favouring(CLASS_TO_CLAP_PROMPT["alarm"])
        fired, _, scores = is_anomaly(
            np.zeros(16), 44100, encoder, delta=0.0, tau_abs=floor
        )
        self.assertFalse(fired)
        self.assertGreater(scores["margin"], 0.0)
        self.assertLess(scores["s_anom"], floor)

    def test_the_defaults_are_the_calibrated_pair(self):
        """The old signature defaulted to (0.0, 0.0), which is why the plain path needed
        two extra flags to get a working gate."""
        encoder = encoder_favouring(CLASS_TO_CLAP_PROMPT["alarm"])
        self.assertTrue(is_anomaly(np.zeros(16), 44100, encoder)[0])

        # Cosines are scale-free, so "quiet" here means near-orthogonal rather than
        # small: every anomaly prompt sits at cosine ~0.03, just under the measured
        # tau of 0.0341, while the normal bank sits at ~0.001. The margin is won and
        # the absolute floor still vetoes it — which is the whole point of having two.
        weak = fakes.one_hot(1) + fakes.one_hot(0, scale=0.03)
        weaker = fakes.one_hot(2) + fakes.one_hot(0, scale=0.001)
        vectors = {text: weak for text in CLASS_TO_CLAP_PROMPT.values()}
        vectors.update({text: weaker for text in NORMAL_PROMPTS})
        quiet = fakes.FakeClapEncoder(fakes.one_hot(0), vectors)
        fired, _, scores = is_anomaly(np.zeros(16), 44100, quiet)
        self.assertGreater(scores["margin"], ANOMALY_GATE_DELTA)
        self.assertLess(scores["s_anom"], ANOMALY_GATE_TAU)
        self.assertFalse(fired)

    def test_the_scores_carry_the_summary_keys_the_audit_record_wants(self):
        encoder = encoder_favouring(CLASS_TO_CLAP_PROMPT["baby_cry"])
        _, _, scores = is_anomaly(np.zeros(16), 44100, encoder)
        for key in ("s_anom", "s_norm", "margin"):
            self.assertIn(key, scores)


class TestHeardClip(unittest.TestCase):
    def test_the_binaural_signal_becomes_mono_and_keeps_its_rate(self):
        heard = np.stack([np.ones(32), np.full(32, 3.0)]).astype(np.float32)
        waveform, rate = heard_clip_for_clap(heard, 44100)
        self.assertEqual(waveform.shape, (32,))
        np.testing.assert_allclose(waveform, 2.0)
        self.assertEqual(rate, 44100)

    def test_a_mono_signal_passes_through(self):
        waveform, _ = heard_clip_for_clap(np.ones(8, dtype=np.float32), 48000)
        self.assertEqual(waveform.shape, (8,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
