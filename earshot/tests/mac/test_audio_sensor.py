"""``audio/sensor.py`` — arming is construction, and the source precedes the render.

The claims a fake can carry: the ordering, the guard being armed rather than merely
available, and the failure modes of the uuid lookup and the analyst-only LOS probe.

The claim it cannot: whether the real ``setAudioSourceTransform`` and ``sourceIsVisible``
behave as assumed. ``tests/box/`` owns that, and ticket 16 already found one call on
this branch (``RLRA_SetListenerHRTF``) that returns ``Success`` over a failed load.
"""

import unittest

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.guard import AudioContextError, StepGuardReport
from earshot.audio.sensor import AudioSensorHandle
from earshot.types import Xyz

SOURCE = Xyz(1.5, 0.9, -2.25)


class _OrderRecordingWorld(fakes.FakeWorld):
    """Records how many source transforms had been set by the time each render ran."""

    def __init__(self, sensor, **kwargs):
        super().__init__(**kwargs)
        self._sensor = sensor
        self.transforms_at_render = []

    def observe(self):
        self.transforms_at_render.append(len(self._sensor.source_transforms))
        return super().observe()


class TestArming(unittest.TestCase):
    def test_the_source_is_set_before_the_first_render(self):
        """RLR renders an IR *for a source*. Arming with none arms against nothing."""
        sensor = fakes.FakeAudioSensor()
        world = _OrderRecordingWorld(sensor)
        AudioSensorHandle(sensor, world.observe, SOURCE)
        self.assertEqual(world.transforms_at_render, [1])

    def test_constructing_arms_the_guard(self):
        """Requirement 1(b): the mesh upload is lazy, so the guard owns the first render."""
        sensor = fakes.FakeAudioSensor(n_vertices=20_000)
        world = fakes.FakeWorld()
        handle = AudioSensorHandle(sensor, world.observe, SOURCE)
        self.assertEqual(world.n_renders, 1)
        self.assertEqual(handle.report.n_vertices, 20_000)
        self.assertTrue(handle.report.obj_written)
        self.assertTrue(handle.report.log_canary_seen)
        self.assertGreater(handle.report.ir_peak_abs, 0.0)

    def test_a_short_mesh_stops_construction(self):
        """No half-built handle: a zero-geometry context renders plausible audio."""
        sensor = fakes.FakeAudioSensor(n_vertices=42)
        with self.assertRaises(AudioContextError) as caught:
            AudioSensorHandle(sensor, fakes.FakeWorld().observe, SOURCE)
        self.assertIn("42 vertices", str(caught.exception))

    def test_the_source_reaches_the_sensor_as_a_world_coordinate(self):
        """float32, because that is what the binding takes — 0.9 arrives as 0.89999998."""
        sensor = fakes.FakeAudioSensor()
        AudioSensorHandle(sensor, fakes.FakeWorld().observe, SOURCE)
        for actual, expected in zip(sensor.source_transforms[0], [1.5, 0.9, -2.25]):
            self.assertAlmostEqual(actual, expected, places=6)


class TestLiveHandle(unittest.TestCase):
    def setUp(self):
        self.sensor = fakes.FakeAudioSensor()
        self.world = fakes.FakeWorld()
        self.handle = AudioSensorHandle(self.sensor, self.world.observe, SOURCE)

    def test_observe_is_guarded_and_returns_the_whole_observation(self):
        """RGB, depth and the IR come out of one call — there is no separate render."""
        observation, report = self.handle.observe()
        self.assertIsInstance(report, StepGuardReport)
        self.assertTrue(report.log_canary_seen)
        self.assertIn("rgb", observation)
        self.assertIn("audio_sensor", observation)
        self.assertEqual(self.world.n_renders, 2)  # one to arm, one here

    def test_moving_the_source_is_counted(self):
        self.handle.set_source(Xyz(0.0, 0.0, 0.0))
        self.assertEqual(self.handle.n_source_moves, 2)
        self.assertEqual(self.handle.source, Xyz(0.0, 0.0, 0.0))

    def test_a_uuid_mismatch_names_the_keys_that_are_there(self):
        observation, _ = self.handle.observe()
        self.handle.uuid = "audio"
        with self.assertRaises(AudioContextError) as caught:
            self.handle.audio_of(observation)
        self.assertIn("audio_sensor", str(caught.exception))

    def test_source_is_visible_is_a_diagnostic_that_never_ends_an_episode(self):
        self.assertTrue(self.handle.source_is_visible())
        self.sensor.raises_on_visible = True
        self.assertIsNone(self.handle.source_is_visible())

    def test_source_is_visible_is_none_when_the_branch_lacks_it(self):
        """`None` rather than a fabricated `True`: with no geometry nothing occludes."""

        class OlderSensor(fakes.FakeAudioSensor):
            sourceIsVisible = None

        sensor = OlderSensor()
        handle = AudioSensorHandle(sensor, fakes.FakeWorld().observe, SOURCE)
        self.assertIsNone(handle.source_is_visible())


if __name__ == "__main__":
    unittest.main(verbosity=2)
