"""``audio/ir.py`` -- the anechoic IR policy, ADR-0018's ``IrPolicy`` arm.

Both arms (ADR-0014): a real IR's energy survives past the first sample (the ON path,
FULL), and the same IR run through ``anechoic_like`` does not (the OFF path, ANECHOIC)
-- the whole point of the function is to make that difference true. Printed, because box
tests in this repo print their measurements and this one is small enough to run here.
"""

import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.ir import anechoic_like


def _tail_energy(ir):
    """Energy in every sample after the first -- what "has a reverb tail" means here."""
    values = np.asarray(ir, dtype=np.float64)
    return float(np.sum(values[:, 1:] ** 2))


class TestBothArms(unittest.TestCase):
    def test_a_real_room_ir_keeps_energy_after_the_first_sample(self):
        """ON: a decaying multi-sample IR is a room with a tail."""
        rng = np.random.default_rng(0)
        n = 64
        decay = 0.9 ** np.arange(n)
        room = (rng.standard_normal((2, n)) * decay).astype(np.float32)
        # Make certain the peak is not on the first sample by construction, so the tail
        # measured is a real decay and not noise past a silent front.
        room[:, 0] = 0.05

        tail = _tail_energy(room)
        print(
            "FULL room IR: tail energy (samples 1..{}) = {:.6f}".format(n - 1, tail)
        )
        self.assertGreater(tail, 0.0, "the fixture room carries no tail to strip")

    def test_the_anechoic_control_has_no_tail(self):
        """OFF: the same room, through anechoic_like, is flat at the peak in sample 0
        and has nothing in any later sample because there IS no later sample."""
        rng = np.random.default_rng(0)
        n = 64
        decay = 0.9 ** np.arange(n)
        room = (rng.standard_normal((2, n)) * decay).astype(np.float32)
        room[:, 0] = 0.05

        anechoic = anechoic_like(room)
        tail = _tail_energy(anechoic)
        print(
            "ANECHOIC control: shape={}, peak={:.6f}, tail energy = {:.6f}".format(
                anechoic.shape, float(np.max(np.abs(anechoic))), tail
            )
        )
        self.assertEqual(tail, 0.0, "a (2, 1) IR has no sample past the first by shape")


class TestShapeAndDtype(unittest.TestCase):
    def test_shape_is_two_by_one(self):
        impulse = np.array([[0.1, 0.2, 0.05], [0.1, 0.2, 0.05]], dtype=np.float32)
        result = anechoic_like(impulse)
        self.assertEqual(result.shape, (2, 1))

    def test_dtype_is_float32(self):
        impulse = np.array([[1.0, 0.5]], dtype=np.float64)
        result = anechoic_like(impulse)
        self.assertEqual(result.dtype, np.float32)

    def test_peak_is_preserved(self):
        impulse = np.array([[0.02, -0.5, 0.1], [0.02, 0.5, -0.1]], dtype=np.float32)
        result = anechoic_like(impulse)
        np.testing.assert_allclose(result, np.full((2, 1), 0.5, dtype=np.float32))

    def test_an_all_zero_ir_floors_rather_than_returning_zero(self):
        impulse = np.zeros((2, 5), dtype=np.float32)
        result = anechoic_like(impulse)
        self.assertTrue(np.all(result > 0.0), "a silent IR must not floor at true zero")
        np.testing.assert_allclose(result, np.full((2, 1), 1e-12, dtype=np.float32))

    def test_the_input_array_is_unchanged_after_the_call(self):
        impulse = np.array([[0.3, 0.1], [0.3, 0.1]], dtype=np.float32)
        before = impulse.copy()
        anechoic_like(impulse)
        np.testing.assert_array_equal(impulse, before)

    def test_an_empty_impulse_raises(self):
        with self.assertRaises(ValueError):
            anechoic_like(np.zeros((2, 0), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
