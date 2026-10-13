"""ADR-0032's checks before the box: does steering by the ear home, or circle?

A kinematic walk, not the runner: `cast_action` picks each tick's action, a turn moves the
heading 30 degrees and a forward moves 0.25 m along it, which is what the fake world's
follower does with the probe a turn names. No walls, no climb, no surge, the source in
open floor, and sign errors drawn independently each tick. So the numbers price the cast's
geometry and nothing else. They are not a prediction of the night.

ADR-0014's two arms: with a right sign the arm must arrive, and with a sign of 0 it must
be `full` tick for tick. The noisy rows are printed and not asserted, as the ADR says.
"""

import math
import random
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_TURN_LEFT,
    ACT_TURN_RIGHT,
    cast_action,
)
from earshot.agent.occupancy import forward_xz
from earshot.audio.lateral import bearing_lateral_sign
from earshot.types import Pose, Xyz

TURN_RAD = math.radians(30.0)
STEP_M = 0.25
RING_M = 1.0
SOUNDING_TICKS = 60
TRIALS = 240
DISTANCES_M = (2.0, 4.0, 6.0, 8.0, 10.0)


def walk(source, *, ear_legs, rng, right_share=1.0, ticks=SOUNDING_TICKS, silent=False):
    """Ticks to arrive within the ring (or None), the path walked, the actions, and the
    nearest the walk came to the source."""
    pose = Pose(Xyz(0.0, 0.0, 0.0), 0.0)
    path, actions = 0.0, []
    nearest = math.hypot(source.x, source.z)
    for k in range(ticks):
        here = math.hypot(pose.position.x - source.x, pose.position.z - source.z)
        nearest = min(nearest, here)
        if here <= RING_M:
            return k, path, actions, nearest
        sign = 0 if silent else bearing_lateral_sign(pose, source)
        if rng.random() >= right_share:
            sign = -sign
        action = cast_action(k, sign, ear_legs=ear_legs)
        actions.append(action)
        x, z, yaw = pose.position.x, pose.position.z, pose.yaw_rad
        if action == ACT_FORWARD:
            dx, dz = forward_xz(yaw)
            x, z = x + STEP_M * dx, z + STEP_M * dz
            path += STEP_M
        elif action == ACT_TURN_LEFT:
            yaw += TURN_RAD
        elif action == ACT_TURN_RIGHT:
            yaw -= TURN_RAD
        pose = Pose(Xyz(x, 0.0, z), yaw)
    nearest = min(nearest, math.hypot(pose.position.x - source.x, pose.position.z - source.z))
    return None, path, actions, nearest


def sources(distance_m, rng):
    for _ in range(TRIALS):
        bearing = rng.uniform(0.0, 2.0 * math.pi)
        yield Xyz(distance_m * math.sin(bearing), 0.0, -distance_m * math.cos(bearing))


def arrival_rate(distance_m, *, ear_legs, right_share, seed=7):
    """The share of walks that arrive, and the path walked per metre of the gap closed.

    The two arms share a seed, so each walk is paired: the same bearing and the same
    draws. Path per metre closed is summed path over summed (start - nearest): a walk
    that circles closes little for the path it walks. Every walk counts, arrived or not.
    """
    rng = random.Random(seed)
    arrived = 0
    walked = closed = 0.0
    for source in sources(distance_m, rng):
        k, path, _, nearest = walk(source, ear_legs=ear_legs, rng=rng,
                                   right_share=right_share)
        arrived += k is not None
        walked += path
        closed += distance_m - nearest
    return arrived / TRIALS, (walked / closed if closed > 0 else math.inf)


class TestTheEarHomes(unittest.TestCase):
    def test_cast_passes_a_source_abeam_just_outside_the_ring(self):
        """Why `full` never arrives from 4 m out with a right sign: its zigzag's mean
        heading is the bisector of two headings 30 degrees apart, so it runs off the
        scan's aim and passes the source outside the ring. Printed, and checked only
        to be outside the ring."""
        rng = random.Random(0)
        source = Xyz(6.0, 0.0, 0.0)
        _, _, _, cast_nearest = walk(source, ear_legs=False, rng=rng, ticks=120)
        _, _, _, ear_nearest = walk(source, ear_legs=True, rng=rng, ticks=120)
        print("6 m abeam, right sign, 120 ticks: full's nearest pass {:.2f} m, the ear's "
              "{:.2f} m".format(cast_nearest, ear_nearest))
        self.assertGreater(cast_nearest, RING_M)
        self.assertLessEqual(ear_nearest, RING_M)

    def test_with_a_right_sign_the_ear_arrives_from_every_distance(self):
        for distance in DISTANCES_M:
            ear, _ = arrival_rate(distance, ear_legs=True, right_share=1.0)
            full, _ = arrival_rate(distance, ear_legs=False, right_share=1.0)
            print("right sign, {:4.1f} m: ear {:6.1%}  full {:6.1%} within {} ticks".format(
                distance, ear, full, SOUNDING_TICKS))
            self.assertEqual(ear, 1.0, distance)

    def test_a_noisy_ear_is_printed_beside_full(self):
        """The ADR's circling check: arrival within the sounding window, and the path
        walked per metre of the gap closed, both arms, at the ear's measured accuracy
        (71.8% pooled, 66.6% at 8 m and over). Printed, not asserted."""
        for share in (0.718, 0.666):
            for distance in DISTANCES_M:
                ear, ear_ratio = arrival_rate(distance, ear_legs=True, right_share=share)
                full, full_ratio = arrival_rate(distance, ear_legs=False, right_share=share)
                print("sign right {:.1%}, {:4.1f} m: arrive ear {:6.1%} full {:6.1%}; "
                      "path per metre closed ear {:5.2f} full {:5.2f}".format(
                          share, distance, ear, full, ear_ratio, full_ratio))

    def test_in_silence_the_ear_is_full_tick_for_tick(self):
        rng_a, rng_b = random.Random(3), random.Random(3)
        for distance in DISTANCES_M:
            source = Xyz(distance, 0.0, 0.0)
            _, _, ear, _ = walk(source, ear_legs=True, rng=rng_a, silent=True, ticks=120)
            _, _, full, _ = walk(source, ear_legs=False, rng=rng_b, silent=True, ticks=120)
            self.assertEqual(ear, full, distance)


if __name__ == "__main__":
    unittest.main()
