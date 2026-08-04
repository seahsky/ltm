"""``audio/lateral.py`` — the cue's arithmetic, and the prediction the box checks.

**Read the limit of this file before trusting it.** Everything here is our own
arithmetic on samples we made up. It pins that ``lateral_sign`` reads the ear order it
claims to and that ``bearing_lateral_sign`` computes the geometry it claims to. It says
**nothing** about which ear the real renderer puts first or which frame it renders in,
and those are the two things that actually decide whether the controller turns the right
way. ``tests/box/test_lateral_box.py`` is the assertion that matters; this is the half
of it that can be checked without a V100.

That split is ADR-0014's rule applied literally: the *subject* of "which frame does
RLR render the listener in" is behaviour we did not write.
"""

import math
import unittest

import numpy as np

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.lateral import (
    LATERAL_AMBIGUOUS,
    LATERAL_LEFT,
    LATERAL_RIGHT,
    LEFT_EAR,
    RIGHT_EAR,
    bearing_lateral_sign,
    interaural_level_difference,
    lateral_sign,
)
from earshot.types import Pose, Xyz


def at(x, z, yaw=0.0):
    return Pose(position=Xyz(x, 0.0, z), yaw_rad=yaw)


class TestTheCue(unittest.TestCase):
    def test_the_louder_ear_decides_and_ear_zero_is_left(self):
        self.assertEqual(lateral_sign(fakes.synthetic_ir(left=1.0, right=4.0)), LATERAL_RIGHT)
        self.assertEqual(lateral_sign(fakes.synthetic_ir(left=4.0, right=1.0)), LATERAL_LEFT)
        self.assertEqual(LEFT_EAR, 0)
        self.assertEqual(RIGHT_EAR, 1)

    def test_identical_ears_are_ambiguous_not_a_direction(self):
        self.assertEqual(lateral_sign(fakes.synthetic_ir(left=1.0, right=1.0)), LATERAL_AMBIGUOUS)

    def test_silence_is_ambiguous_rather_than_a_divide(self):
        self.assertEqual(lateral_sign(np.zeros((2, 64))), LATERAL_AMBIGUOUS)
        self.assertEqual(interaural_level_difference(np.zeros((2, 64))), 0.0)

    def test_the_cue_is_fold_invariant(self):
        """The same direction at 1 m and at 8 m: a controller reading it does not have
        to know how loud the source is to know which way to turn."""
        near = interaural_level_difference(fakes.synthetic_ir(left=1.0, right=3.0))
        far = interaural_level_difference(fakes.synthetic_ir(left=0.01, right=0.03))
        self.assertAlmostEqual(near, far, places=6)

    def test_a_non_binaural_signal_is_a_loud_failure(self):
        with self.assertRaises(ValueError):
            lateral_sign(np.zeros(64))


class TestBearingPrediction(unittest.TestCase):
    """``bearing_lateral_sign`` is analyst-only and privileged — see its docstring."""

    def test_facing_forward_a_source_to_positive_x_is_on_the_right(self):
        """Habitat: forward is -z and right is +x at zero yaw."""
        self.assertEqual(bearing_lateral_sign(at(0.0, 0.0), Xyz(1.0, 0.0, -1.0)), LATERAL_RIGHT)
        self.assertEqual(bearing_lateral_sign(at(0.0, 0.0), Xyz(-1.0, 0.0, -1.0)), LATERAL_LEFT)

    def test_turning_around_flips_the_prediction(self):
        """THE decisive pair. The agent frame says this flips; the world frame — what
        the grid rendered, at identity listener yaw — says it does not."""
        source = Xyz(1.0, 0.0, 0.0)
        self.assertEqual(bearing_lateral_sign(at(0.0, 0.0, 0.0), source), LATERAL_RIGHT)
        self.assertEqual(bearing_lateral_sign(at(0.0, 0.0, math.pi), source), LATERAL_LEFT)

    def test_a_source_directly_behind_is_ambiguous(self):
        """At yaw pi/2 the agent faces -x, so a source at +x is dead astern: the lateral
        component is zero and the cue cannot resolve front from back."""
        self.assertEqual(
            bearing_lateral_sign(at(0.0, 0.0, math.pi / 2.0), Xyz(1.0, 0.0, 0.0)),
            LATERAL_AMBIGUOUS,
        )

    def test_height_does_not_enter_the_cue(self):
        """y is up; a source overhead is neither left nor right."""
        low = bearing_lateral_sign(at(0.0, 0.0), Xyz(1.0, 0.0, 0.0))
        high = bearing_lateral_sign(at(0.0, 0.0), Xyz(1.0, 9.0, 0.0))
        self.assertEqual(low, high)

    def test_a_quarter_turn_moves_a_side_source_to_dead_ahead(self):
        source = Xyz(1.0, 0.0, 0.0)
        self.assertEqual(bearing_lateral_sign(at(0.0, 0.0, 0.0), source), LATERAL_RIGHT)
        self.assertEqual(
            bearing_lateral_sign(at(0.0, 0.0, -math.pi / 2.0), source), LATERAL_AMBIGUOUS
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
