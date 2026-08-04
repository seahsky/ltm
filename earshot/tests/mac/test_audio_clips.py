"""``audio/clips.py`` — the clip domain, the RMS every assertion is written in, and
the convolution that replaced the grid lookup.

``render_through_ir`` is the one worth reading twice: it is where a unit error would
hide. Calibrating on IR energy and thresholding on a received signal produces a
threshold that never fires, and nothing else in the tree would say so.
"""

import os
import tempfile
import unittest

import numpy as np

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.clips import (
    CLASS_TO_ESC50,
    as_binaural,
    load_anomaly_clip,
    normalize_clip,
    render_through_ir,
    resolve_anomaly_clip,
    rms,
    select_esc50_clip,
    synthetic_burst,
)


class TestRms(unittest.TestCase):
    def test_rms_of_a_constant_is_its_magnitude(self):
        self.assertAlmostEqual(rms(np.full(64, 0.25, dtype=np.float32)), 0.25, places=6)

    def test_rms_spans_channels(self):
        signal = np.stack([np.zeros(16), np.full(16, 2.0)])
        self.assertAlmostEqual(rms(signal), np.sqrt(2.0), places=6)

    def test_empty_is_zero_not_nan(self):
        """A NaN here reaches the onset comparison and silently never fires."""
        self.assertEqual(rms(np.zeros(0)), 0.0)


class TestNormalizeClip(unittest.TestCase):
    def test_hits_the_target_level(self):
        clip = normalize_clip(np.random.default_rng(0).standard_normal(4096), -20.0)
        self.assertAlmostEqual(rms(clip), 0.1, places=5)

    def test_silence_stays_silence(self):
        self.assertEqual(rms(normalize_clip(np.zeros(128))), 0.0)

    def test_stereo_is_folded_to_mono(self):
        self.assertEqual(normalize_clip(np.ones((2, 32))).ndim, 1)


class TestAsBinaural(unittest.TestCase):
    def test_a_nested_sequence_converts(self):
        """The observation is NOT a numpy array — ticket 16 measured `.shape` as None."""
        converted = as_binaural([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
        self.assertEqual(converted.shape, (2, 3))
        self.assertEqual(converted.dtype, np.float32)

    def test_mono_is_a_loud_failure_not_a_broadcast(self):
        with self.assertRaises(ValueError) as caught:
            as_binaural(np.zeros(128))
        self.assertIn("Binaural", str(caught.exception))


class TestRenderThroughIr(unittest.TestCase):
    def test_length_is_the_clips_length_not_the_convolutions(self):
        """A fixed output length is what makes per-step RMS comparable between steps."""
        rendered = render_through_ir(fakes.synthetic_ir(n_samples=512), np.ones(300))
        self.assertEqual(rendered.shape, (2, 300))

    def test_a_unit_impulse_reproduces_the_clip(self):
        impulse = np.zeros((2, 8), dtype=np.float32)
        impulse[:, 0] = 1.0
        clip = np.array([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
        rendered = render_through_ir(impulse, clip)
        np.testing.assert_allclose(rendered[0], clip, atol=1e-5)
        np.testing.assert_allclose(rendered[1], clip, atol=1e-5)

    def test_a_delayed_impulse_shifts_the_clip(self):
        """Catches an off-by-one in the transform length, which a spike at 0 would not."""
        impulse = np.zeros((2, 8), dtype=np.float32)
        impulse[:, 2] = 1.0
        clip = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        rendered = render_through_ir(impulse, clip)
        np.testing.assert_allclose(rendered[0], [0.0, 0.0, 1.0, 2.0], atol=1e-5)

    def test_the_ear_imbalance_survives_the_convolution(self):
        """If it did not, the lateral cue would be destroyed by the thing that makes it."""
        rendered = render_through_ir(fakes.synthetic_ir(left=1.0, right=4.0), np.ones(256))
        self.assertGreater(rms(rendered[1]), 3.5 * rms(rendered[0]))

    def test_an_empty_clip_is_rejected(self):
        with self.assertRaises(ValueError):
            render_through_ir(fakes.synthetic_ir(), np.zeros(0))


class TestResolveAndLoad(unittest.TestCase):
    def test_an_explicit_path_wins(self):
        self.assertEqual(resolve_anomaly_clip("alarm", "/tmp/x.wav"), "/tmp/x.wav")

    def test_a_staged_clip_is_found_and_a_missing_one_is_none(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = os.path.join(directory, "alarm.wav")
            with open(staged, "wb") as handle:
                handle.write(b"RIFF")
            self.assertEqual(resolve_anomaly_clip("alarm", None, directory), staged)
            self.assertIsNone(resolve_anomaly_clip("baby_cry", None, directory))
            self.assertIsNone(resolve_anomaly_clip(None, None, directory))

    def test_loading_a_missing_clip_raises_rather_than_synthesising_one(self):
        """The silent fallback is how a run could calibrate CLAP on a real recording
        and then classify a noise burst."""
        with self.assertRaises(FileNotFoundError) as caught:
            load_anomaly_clip("/definitely/not/here.wav", 44100)
        self.assertIn("no synthetic fallback", str(caught.exception))


class TestSyntheticBurst(unittest.TestCase):
    def test_it_is_deterministic_and_normalised(self):
        first = synthetic_burst(1000, 0.5)
        second = synthetic_burst(1000, 0.5)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(rms(first), 0.1, places=5)
        self.assertEqual(first.size, 500)


class TestEsc50Selection(unittest.TestCase):
    ROWS = [
        {"filename": "b.wav", "category": "clock_alarm"},
        {"filename": "a.wav", "category": "clock_alarm"},
        {"filename": "c.wav", "category": "crying_baby"},
    ]

    def test_selection_is_sorted_and_wraps(self):
        self.assertEqual(select_esc50_clip(self.ROWS, "clock_alarm", 0), "a.wav")
        self.assertEqual(select_esc50_clip(self.ROWS, "clock_alarm", 1), "b.wav")
        self.assertEqual(select_esc50_clip(self.ROWS, "clock_alarm", 2), "a.wav")

    def test_an_unknown_category_is_none(self):
        self.assertIsNone(select_esc50_clip(self.ROWS, "vacuum_cleaner", 0))

    def test_the_three_locked_classes_all_have_a_source_category(self):
        self.assertEqual(
            sorted(CLASS_TO_ESC50), ["alarm", "baby_cry", "glass_break"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
