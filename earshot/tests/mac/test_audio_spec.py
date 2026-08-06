"""``audio/spec.py`` — the only configuration path, and the trap it closes.

What a Mac can settle here is our own logic: that every field goes through the
validator, that the preset is applied rather than inherited, and that a post-condition
fires when something does not take. What it cannot settle is whether the real
``AudioSensorSpec`` swallows what this fake swallows — ticket 04 measured that on the
box, and ``_audio_fakes`` reproduces the measurement structurally.
"""

import unittest

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.config import AudioConfig
from earshot.audio.guard import AudioContextError
from earshot.audio.spec import (
    ACOUSTICS_PRESET,
    AUDIO_SENSOR_UUID,
    audio_config_mapping,
    audio_sensor_spec,
)


class TestConfigMapping(unittest.TestCase):
    def test_the_preset_is_ticket_06s_measured_cheap_preset(self):
        """A "tidy" edit to any of these is a 63x slower render that still looks fine."""
        self.assertEqual(
            ACOUSTICS_PRESET,
            {
                "indirectRayCount": 500,
                "indirectRayDepth": 50,
                "threadCount": 4,
                "temporalCoherence": 1,
            },
        )

    def test_the_physics_knobs_are_not_in_the_preset(self):
        """`transmission` and `diffraction` ARE the non-line-of-sight audio path.

        They are cheaper with the sound effectively switched off, so a preset derived on
        speed alone would have swallowed a task-design decision.
        """
        self.assertNotIn("transmission", ACOUSTICS_PRESET)
        self.assertNotIn("diffraction", ACOUSTICS_PRESET)

    def test_mapping_carries_the_config_and_does_not_mutate_the_preset(self):
        mapping = audio_config_mapping(AudioConfig(), fakes.BINAURAL)
        self.assertEqual(mapping["acousticsConfig"]["sampleRate"], 44100.0)
        self.assertEqual(mapping["acousticsConfig"]["globalVolume"], 1.0)
        self.assertEqual(mapping["acousticsConfig"]["indirectRayCount"], 500)
        self.assertNotIn("sampleRate", ACOUSTICS_PRESET)
        self.assertNotIn("globalVolume", ACOUSTICS_PRESET)

    def test_the_ray_count_is_overridable_from_the_config(self):
        """The knob behind the reproducibility finding: two runs of the same scene at
        500 rays disagreed on 4 of 20 episode outcomes, and `indirectRayCount` is the
        only preset entry that trades render ACCURACY for speed rather than speed alone."""
        mapping = audio_config_mapping(AudioConfig(indirect_ray_count=2500), fakes.BINAURAL)
        self.assertEqual(mapping["acousticsConfig"]["indirectRayCount"], 2500)
        self.assertEqual(ACOUSTICS_PRESET["indirectRayCount"], 500,
                         "the override must not write through to the module constant")

    def test_none_leaves_the_preset_alone(self):
        mapping = audio_config_mapping(AudioConfig(), fakes.BINAURAL)
        self.assertEqual(mapping["acousticsConfig"]["indirectRayCount"], 500)

    def test_it_overrides_one_key_and_leaves_the_rest_of_the_preset(self):
        acoustics = audio_config_mapping(
            AudioConfig(indirect_ray_count=2500), fakes.BINAURAL)["acousticsConfig"]
        for key, value in ACOUSTICS_PRESET.items():
            if key != "indirectRayCount":
                self.assertEqual(acoustics[key], value, key)

    def test_the_config_override_wins_over_an_explicit_preset(self):
        """`acoustics` replaces the base wholesale; the config field then overrides one
        key of whatever base is in force, so a run's ray count is always the number
        `run_config` records."""
        mapping = audio_config_mapping(
            AudioConfig(indirect_ray_count=2500), fakes.BINAURAL,
            acoustics={"indirectRayCount": 17, "indirectRayDepth": 5},
        )
        self.assertEqual(mapping["acousticsConfig"]["indirectRayCount"], 2500)
        self.assertEqual(mapping["acousticsConfig"]["indirectRayDepth"], 5)

    def test_materials_are_off_explicitly_not_inherited(self):
        """ADR-0007 is a decision; this is the line that states it in code."""
        self.assertIs(audio_config_mapping(AudioConfig(), fakes.BINAURAL)["enableMaterials"], False)


class TestAudioSensorSpec(unittest.TestCase):
    def setUp(self):
        self.spec = fakes.FakeAudioSensorSpec()
        self.config = AudioConfig()

    def test_a_fresh_fake_spec_has_only_the_known_dynamic_attribute(self):
        """The fake reproduces the box measurement, or every test below is vacuous."""
        self.assertEqual(list(vars(self.spec)), ["__noise_model_kwargs"])

    def test_configuring_applies_the_preset_and_the_layout(self):
        configured = audio_sensor_spec(self.spec, self.config, fakes.BINAURAL)
        self.assertIs(configured, self.spec)
        self.assertEqual(configured.acousticsConfig.indirectRayCount, 500)
        self.assertEqual(configured.acousticsConfig.indirectRayDepth, 50)
        self.assertEqual(configured.acousticsConfig.threadCount, 4)
        self.assertEqual(configured.acousticsConfig.temporalCoherence, 1)
        self.assertEqual(configured.acousticsConfig.sampleRate, 44100.0)
        self.assertIs(configured.channelLayout.type, fakes.BINAURAL)
        self.assertEqual(configured.channelLayout.channelCount, 2)
        self.assertIs(configured.enableMaterials, False)
        self.assertEqual(configured.uuid, AUDIO_SENSOR_UUID)

    def test_the_defaults_it_replaces_are_the_slow_ones(self):
        """Guards the test above against a fake that already carried the preset."""
        self.assertEqual(fakes.FakeAcousticsConfig().indirectRayCount, 5000)
        self.assertEqual(fakes.FakeAcousticsConfig().indirectRayDepth, 200)

    def test_an_unknown_acoustics_key_is_rejected_before_anything_is_written(self):
        """`irTime` was renamed `maxIRLength`; on the real spec it would be swallowed."""
        with self.assertRaises(AudioContextError) as caught:
            audio_sensor_spec(
                self.spec, self.config, fakes.BINAURAL, acoustics={"irTime": 4.0}
            )
        self.assertIn("irTime", str(caught.exception))
        self.assertIn("maxIRLength", str(caught.exception))  # the valid-field list
        self.assertEqual(self.spec.acousticsConfig.indirectRayCount, 5000)

    def test_a_key_attached_before_configuration_is_caught(self):
        """Belt to the validator's braces: this one never passes through the mapping."""
        setattr(self.spec, "outputDirectory", "/tmp/ir")
        with self.assertRaises(AudioContextError) as caught:
            audio_sensor_spec(self.spec, self.config, fakes.BINAURAL)
        self.assertIn("outputDirectory", str(caught.exception))

    def test_the_layout_check_compares_by_value_not_identity(self):
        """pybind11 re-wraps a C++ enum on read, so `is` would fail on the box against a
        spec that took the value correctly. The fake proves only that `==` is used."""

        class Rewrapping(fakes.FakeChannelLayout):
            """Returns an equal-but-not-identical object, as the binding would."""

            def __init__(self):
                object.__setattr__(self, "type", None)
                object.__setattr__(self, "channelCount", 2)

            def __getattribute__(self, name):
                value = object.__getattribute__(self, name)
                if name == "type" and value is not None:
                    return _EqualTo(value)
                return value

        class _EqualTo:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __eq__(self, other):
                return other is self._wrapped or other == self._wrapped

        self.spec.channelLayout = Rewrapping()
        audio_sensor_spec(self.spec, self.config, fakes.BINAURAL)

    def test_a_layout_that_does_not_take_is_a_loud_failure(self):
        """A silent Mono layout is a lateral sign of zero at every pose."""

        class StubbornLayout(fakes.FakeChannelLayout):
            """The field exists and reads back Mono however often it is assigned.

            That is the shape of the failure worth catching: an assignment that raises
            is already loud, and one that never creates the field is caught by the key
            validator a line earlier.
            """

            def __init__(self):
                object.__setattr__(self, "type", "Mono")
                object.__setattr__(self, "channelCount", 2)

            def __setattr__(self, name, value):
                if name == "type":
                    return  # the assignment "succeeds" and changes nothing
                object.__setattr__(self, name, value)

        self.spec.channelLayout = StubbornLayout()
        with self.assertRaises(ValueError) as caught:
            audio_sensor_spec(self.spec, self.config, fakes.BINAURAL)
        self.assertIn("channelLayout.type did not take", str(caught.exception))

    def test_an_acoustics_knob_that_does_not_take_is_a_loud_failure(self):
        class StubbornAcoustics(fakes.FakeAcousticsConfig):
            def __init__(self):
                super().__init__()
                object.__setattr__(self, "threadCount", 1)

            def __setattr__(self, name, value):
                if name == "threadCount":
                    return
                object.__setattr__(self, name, value)

        self.spec.acousticsConfig = StubbornAcoustics()
        with self.assertRaises(ValueError) as caught:
            audio_sensor_spec(self.spec, self.config, fakes.BINAURAL)
        self.assertIn("threadCount", str(caught.exception))
        self.assertIn("1723 ms/step", str(caught.exception))

    def test_an_explicit_acoustics_override_is_honoured(self):
        audio_sensor_spec(
            self.spec, self.config, fakes.BINAURAL, acoustics={"indirectRayCount": 2000}
        )
        self.assertEqual(self.spec.acousticsConfig.indirectRayCount, 2000)
        # Untouched knobs keep the branch defaults rather than the preset's values.
        self.assertEqual(self.spec.acousticsConfig.threadCount, 1)

    def test_the_config_reaches_the_spec_not_just_the_mapping(self):
        audio_sensor_spec(
            self.spec, AudioConfig(sample_rate=48000, global_volume=2.0), fakes.BINAURAL
        )
        self.assertEqual(self.spec.acousticsConfig.sampleRate, 48000.0)
        self.assertEqual(self.spec.acousticsConfig.globalVolume, 2.0)


class TestTheUuidIsAssignedNotInherited(unittest.TestCase):
    """The defect that survived 560 green tests and died on the box's first render.

    ``AudioSensorSpec`` constructs with ``uuid == 'audio'``;
    ``Simulator._get_audio_observation`` looks up the literal ``"audio_sensor"``. The
    old fake carried ``"audio_sensor"`` as its starting value, so the one assertion that
    would have caught this passed against a spec that already held the answer.
    """

    def test_a_fresh_spec_does_not_already_carry_the_name(self):
        """Guards every test below: a fake born correct cannot fail."""
        self.assertEqual(fakes.FakeAudioSensorSpec().uuid, "audio")
        self.assertNotEqual(fakes.FakeAudioSensorSpec().uuid, AUDIO_SENSOR_UUID)

    def test_configuring_assigns_the_name_habitat_looks_up(self):
        spec = fakes.FakeAudioSensorSpec()
        audio_sensor_spec(spec, AudioConfig(), fakes.BINAURAL)
        self.assertEqual(spec.uuid, AUDIO_SENSOR_UUID)

    def test_a_uuid_that_does_not_take_is_a_loud_failure(self):
        """A build that ignores the write must raise here, not at the first render.

        The failure it stands in for is not hypothetical: it is exactly what the box
        did, one render in, from inside habitat, as a ``KeyError`` naming a string that
        appears nowhere in our code.
        """

        class RefusesTheName(fakes.FakeAudioSensorSpec):
            @property
            def uuid(self):
                return "audio"

            @uuid.setter
            def uuid(self, _value):
                pass

        with self.assertRaises(ValueError) as caught:
            audio_sensor_spec(RefusesTheName(), AudioConfig(), fakes.BINAURAL)
        message = str(caught.exception)
        self.assertIn("audio_sensor", message)
        self.assertIn("_get_audio_observation", message)

    def test_it_is_asserted_before_the_acoustics_are_written(self):
        """Ordering, so the diagnosis names the uuid rather than whichever knob is first.

        A spec that refuses one write is a spec whose other writes are suspect; failing
        on the config would send the reader to the preset, which is not where the
        problem is.
        """

        class RefusesEverything(fakes.FakeAudioSensorSpec):
            @property
            def uuid(self):
                return "audio"

            @uuid.setter
            def uuid(self, _value):
                pass

        spec = RefusesEverything()
        with self.assertRaises(ValueError):
            audio_sensor_spec(spec, AudioConfig(), fakes.BINAURAL)
        self.assertEqual(spec.acousticsConfig.indirectRayCount, 5000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
