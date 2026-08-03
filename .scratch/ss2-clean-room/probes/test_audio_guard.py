#!/usr/bin/env python3
"""Mac-runnable tests for ticket 12's audio context guard.

    python3 .scratch/ss2-clean-room/probes/test_audio_guard.py

No habitat_sim, no simulator, no box. The fakes reproduce the two pybind11 behaviours
the guard depends on, because those behaviours are the whole subject:

- ``FakeSpec`` has ``__dict__`` *and* slot-backed properties, so a real field is a data
  descriptor that never touches the instance dict while an unknown name lands in it —
  the ``py::dynamic_attr()`` semantics on ``AudioSensorSpec``.
- ``FakeAcoustics`` has slots and no ``__dict__``, so an unknown name raises — the
  ``RLRAudioPropagationConfiguration`` semantics.

``make_render`` writes with ``os.write`` rather than to ``sys.stderr``/``sys.stdout``,
because that is how habitat-sim's C++ logger writes and a capture that only sees the
Python objects is the vacuous-pass this ticket exists to prevent.

**Ticket 16 corrected these fakes on two points, both from source, before the probe ran.**
They used to write the whole log to fd 2 and to fabricate a ``[Error]`` severity tag —
and 30 tests passed against both. Neither is true of the binary:

- ``ESP_DEBUG`` is a ``Corrade::Utility::Debug``, whose ``defaultOutput()`` is
  ``&std::cout`` (Corrade ``Debug.cpp:525``), so the ``Vertex count`` canary is on
  **fd 1**. ``ESP_WARNING``/``ESP_ERROR`` go to ``&std::cerr`` (``:526-527``).
- ``buildMessagePrefix`` (``Logging.cpp:149-152``) renders
  ``"[HH:MM:SS:uuuuuu]:[Subsystem] file(line)::func : "`` and Corrade adds no severity
  tag, so **no** ``[Error]`` substring exists to match. Severity is the stream.

The fakes now reproduce both, which is what makes the suite worth anything here.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_guard import (  # noqa: E402
    HABITAT_LOG_PREFIX_RE,
    AudioContextError,
    apply_audio_config,
    arm_audio_context,
    assert_no_swallowed_keys,
    bound_field_names,
    capture_habitat_logs,
    count_obj_vertices,
    pin_habitat_logging,
)


# ----------------------------------------------------------------------
# pybind11-shaped fakes
# ----------------------------------------------------------------------


def _prop(name):
    slot = "_" + name

    def getter(self):
        return getattr(self, slot)

    def setter(self, value):
        setattr(self, slot, value)

    return property(getter, setter)


class FakeAcoustics:
    """RLRAudioPropagationConfiguration: bound fields, NO dynamic_attr."""

    __slots__ = ("_sampleRate", "_maxIRLength", "_transmission", "_threadCount")
    sampleRate = _prop("sampleRate")
    maxIRLength = _prop("maxIRLength")
    transmission = _prop("transmission")
    threadCount = _prop("threadCount")

    def __init__(self):
        self._sampleRate = 44100.0
        self._maxIRLength = 4.0
        self._transmission = 1
        self._threadCount = 1


class FakeChannelLayout:
    __slots__ = ("_type", "_channelCount")
    type = _prop("type")
    channelCount = _prop("channelCount")

    def __init__(self):
        self._type = "Binaural"
        self._channelCount = 2


class FakeSpec:
    """AudioSensorSpec: bound fields in slots, plus a real ``__dict__`` for the trap."""

    __slots__ = ("_uuid", "_enableMaterials", "_acousticsConfig", "_channelLayout", "__dict__")
    uuid = _prop("uuid")
    enableMaterials = _prop("enableMaterials")
    acousticsConfig = _prop("acousticsConfig")
    channelLayout = _prop("channelLayout")

    def __init__(self):
        self._uuid = ""
        self._enableMaterials = False
        self._acousticsConfig = FakeAcoustics()
        self._channelLayout = FakeChannelLayout()


class FakeAudioSensor:
    def __init__(self, n_vertices=392356, write_ok=True, ray_efficiency=0.548, visible=False):
        self.n_vertices = n_vertices
        self.write_ok = write_ok
        self._ray_efficiency = ray_efficiency
        self._visible = visible
        self.obj_paths = []

    def writeSceneMeshOBJ(self, path):
        if not self.write_ok:
            return False
        self.obj_paths.append(path)
        with open(path, "w") as fh:
            fh.write("# fake scene mesh\n")
            for i in range(self.n_vertices):
                fh.write("v {0}.0 {0}.5 {0}.25\n".format(i))
            # Normals and texture coords must not be counted as geometry.
            fh.write("vn 0.0 1.0 0.0\nvt 0.5 0.5\n")
        return True

    def getRayEfficiency(self):
        return self._ray_efficiency

    def sourceIsVisible(self):
        return self._visible


# Every line here is ESP_DEBUG, so all of it lands on fd 1 on the real binary.
HEALTHY_LOG = (
    "[11:02:14:481200]:[Sensor] AudioSensor.cpp(135)::runSimulation : [Audio] Semantic "
    "scene does not exist or materials are disabled, will use default material\n"
    "[11:02:14:481355]:[Sensor] AudioSensor.cpp(495)::loadMesh : [Audio] Loading "
    "non-semantic mesh\n"
    "[11:02:14:481420]:[Sensor] AudioSensor.cpp(499)::loadMesh : Vertex count : 392356 "
    ", Index count : 1185054\n"
)

# A real ESP_ERROR, rendered exactly as buildMessagePrefix formats it — note there is no
# severity marker anywhere in the text. AudioSensor.cpp:181.
ERROR_LOG = (
    "[11:02:14:482010]:[Sensor] AudioSensor.cpp(181)::setListenerHRTF : Couldn't load "
    "custom audio listener HRTF\n"
)

# VERBATIM from the box (ticket 16, writeSceneMeshOBJ against a bad path). The closed RLR
# engine writes this to fd 2 itself, with no habitat prefix, and — for setListenerHRTF —
# returns RLRA_Success anyway, so habitat's own ESP_ERROR never fires. This block is the
# entire evidence that anything went wrong.
RLR_ENGINE_LOG = (
    "File: arvr/libraries/audio/AudioSDK/Research/Source/Wrapper/PropagationWrapper.cpp\n"
    "Function: ovrResult PropagationWrapper::WriteSceneMeshOBJ(const std::string &), Line 1025\n"
    "Error writing scene OBJ mesh at location:\n"
    "/nonexistent-dir/x.obj\n"
)


def make_render(log=HEALTHY_LOG, err="", ir=((0.0, 0.163), (0.02, -0.09))):
    """``log`` goes to fd 1 (ESP_DEBUG), ``err`` to fd 2 (ESP_WARNING/ESP_ERROR)."""

    def render():
        if log:
            os.write(1, log.encode())
        if err:
            os.write(2, err.encode())
        return ir

    return render


# ----------------------------------------------------------------------


class TestFdCapture(unittest.TestCase):
    def test_captures_writes_to_fd_2(self):
        with capture_habitat_logs() as captured:
            os.write(2, b"from the C++ side\n")
        self.assertIn("from the C++ side", captured.stderr)

    def test_captures_writes_to_fd_1(self):
        """ESP_DEBUG goes to std::cout, so fd 1 is where the canary actually lands."""
        with capture_habitat_logs() as captured:
            os.write(1, b"Vertex count : 392356\n")
        self.assertIn("Vertex count", captured.stdout)
        self.assertIn("Vertex count", captured.text)

    def test_keeps_the_two_streams_apart(self):
        """The split IS the severity signal — merging them would discard it."""
        with capture_habitat_logs() as captured:
            os.write(1, b"debug line\n")
            os.write(2, b"error line\n")
        self.assertIn("debug line", captured.stdout)
        self.assertNotIn("error line", captured.stdout)
        self.assertIn("error line", captured.stderr)
        self.assertNotIn("debug line", captured.stderr)

    def test_restores_both_fds_after_an_exception(self):
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            with self.assertRaises(ValueError):
                with capture_habitat_logs():
                    os.write(1, b"noise\n")
                    os.write(2, b"noise\n")
                    raise ValueError("boom")
            os.write(1, b"")  # both are still valid descriptors
            os.write(2, b"")
        finally:
            os.close(saved_out)
            os.close(saved_err)

    def test_does_not_swallow_the_exception(self):
        with self.assertRaises(KeyError):
            with capture_habitat_logs():
                raise KeyError("k")


class TestHabitatLogPrefix(unittest.TestCase):
    def test_matches_a_real_rendered_prefix(self):
        self.assertTrue(HABITAT_LOG_PREFIX_RE.search(ERROR_LOG))

    def test_does_not_match_third_party_stderr(self):
        """Anything else on fd 2 is not habitat-sim and must not count as a severity event."""
        for noise in (
            "UserWarning: torch.cuda is deprecated\n",
            "libGL error: MESA-LOADER: failed to open swrast\n",
            "  File \"/x/y.py\", line 3, in <module>\n",
        ):
            self.assertIsNone(HABITAT_LOG_PREFIX_RE.search(noise), noise)

    def test_no_severity_tag_exists_to_match(self):
        """The regression guard on the old `[Error]` pattern, which matched nothing."""
        self.assertNotIn("[Error]", ERROR_LOG)


class TestObjVertexCount(unittest.TestCase):
    def test_counts_v_lines_only(self):
        path = os.path.join(tempfile.gettempdir(), "audioguard-test-count.obj")
        with open(path, "w") as fh:
            fh.write("# header\nv 0 0 0\nvn 0 1 0\nv 1 1 1\nvt 0.5 0.5\nf 1 2 3\nv 2 2 2\n")
        try:
            self.assertEqual(count_obj_vertices(path), 3)
        finally:
            os.unlink(path)

    def test_empty_mesh_counts_zero(self):
        path = os.path.join(tempfile.gettempdir(), "audioguard-test-empty.obj")
        with open(path, "w") as fh:
            fh.write("# no geometry\n")
        try:
            self.assertEqual(count_obj_vertices(path), 0)
        finally:
            os.unlink(path)


class TestKeyValidation(unittest.TestCase):
    def test_bound_field_names_finds_the_real_fields(self):
        names = bound_field_names(FakeSpec())
        self.assertEqual(names, frozenset({"uuid", "enableMaterials", "acousticsConfig", "channelLayout"}))

    def test_valid_config_applies(self):
        spec = apply_audio_config(
            FakeSpec(),
            {
                "uuid": "audio_sensor",
                "enableMaterials": False,
                "acousticsConfig": {"sampleRate": 22050.0, "threadCount": 4},
                "channelLayout": {"channelCount": 2},
            },
        )
        self.assertEqual(spec.uuid, "audio_sensor")
        self.assertEqual(spec.acousticsConfig.sampleRate, 22050.0)
        self.assertEqual(spec.acousticsConfig.threadCount, 4)
        self.assertEqual(vars(spec), {})

    def test_unknown_top_level_key_is_rejected(self):
        with self.assertRaises(AudioContextError) as ctx:
            apply_audio_config(FakeSpec(), {"irTime": 4.0})
        self.assertIn("irTime", str(ctx.exception))

    def test_unknown_nested_key_is_rejected(self):
        with self.assertRaises(AudioContextError) as ctx:
            apply_audio_config(FakeSpec(), {"acousticsConfig": {"irTime": 4.0}})
        self.assertIn("spec.acousticsConfig", str(ctx.exception))

    def test_rejection_is_atomic(self):
        """A bad key must not leave a half-applied config behind."""
        spec = FakeSpec()
        with self.assertRaises(AudioContextError):
            apply_audio_config(spec, {"uuid": "audio_sensor", "irTime": 4.0})
        self.assertEqual(spec.uuid, "")

    def test_swallowed_key_is_detected_exactly(self):
        spec = FakeSpec()
        spec.irTime = 4.0  # what py::dynamic_attr does: silently attach, never read
        with self.assertRaises(AudioContextError) as ctx:
            assert_no_swallowed_keys(spec)
        self.assertIn("irTime", str(ctx.exception))

    def test_clean_spec_has_no_swallowed_keys(self):
        assert_no_swallowed_keys(FakeSpec())

    def test_allowed_dynamic_attr_is_tolerated(self):
        spec = FakeSpec()
        spec.someLegitimateAttr = 1
        assert_no_swallowed_keys(spec, allowed=("someLegitimateAttr",))

    def test_real_constructor_attribute_is_not_a_swallowed_key(self):
        """The box's stock AudioSensorSpec() leaves __noise_model_kwargs in vars().

        Measured 2026-08-03. The fakes never reproduced it, so the guard raised on
        every healthy spec until KNOWN_DYNAMIC_ATTRS was populated.
        """
        spec = FakeSpec()
        setattr(spec, "__noise_model_kwargs", {})
        assert_no_swallowed_keys(spec)

    def test_known_dynamic_attrs_survive_a_caller_supplied_allowed(self):
        """A caller passing its own `allowed` must not re-open a measured false positive."""
        spec = FakeSpec()
        setattr(spec, "__noise_model_kwargs", {})
        spec.someLegitimateAttr = 1
        assert_no_swallowed_keys(spec, allowed=("someLegitimateAttr",))

    def test_a_real_typo_still_fails_alongside_the_known_attr(self):
        spec = FakeSpec()
        setattr(spec, "__noise_model_kwargs", {})
        spec.irTime = 4.0
        with self.assertRaises(AudioContextError) as ctx:
            assert_no_swallowed_keys(spec)
        self.assertIn("irTime", str(ctx.exception))
        self.assertNotIn("__noise_model_kwargs", str(ctx.exception))


class TestArmAudioContext(unittest.TestCase):
    def test_healthy_context_passes(self):
        report = arm_audio_context(FakeAudioSensor(n_vertices=392356), make_render())
        self.assertEqual(report.n_vertices, 392356)
        self.assertTrue(report.obj_written)
        self.assertTrue(report.log_canary_seen)
        self.assertAlmostEqual(report.ir_peak_abs, 0.163, places=6)
        self.assertAlmostEqual(report.ray_efficiency, 0.548, places=6)
        self.assertEqual(report.fatal_log_lines, [])

    def test_a_healthy_render_logs_only_to_stdout(self):
        """The shape of a good run: all fd 1, empty fd 2. This is the case the old
        fd-2-only capture would have failed."""
        report = arm_audio_context(FakeAudioSensor(), make_render())
        self.assertGreater(report.stdout_chars, 0)
        self.assertEqual(report.stderr_chars, 0)
        self.assertEqual(report.log_chars, report.stdout_chars)

    def test_habitat_prefixed_line_on_stderr_is_fatal(self):
        """No substring in FATAL_LOG_SUBSTRINGS matches this — the stream is what damns it."""
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(), make_render(err=ERROR_LOG))
        self.assertIn("Couldn't load custom audio listener HRTF", str(ctx.exception))

    def test_the_same_line_on_stdout_is_not_fatal(self):
        """ESP_DEBUG echoes plenty of scary-looking text; only fd 2 carries severity."""
        report = arm_audio_context(
            FakeAudioSensor(), make_render(log=HEALTHY_LOG + ERROR_LOG)
        )
        self.assertEqual(report.fatal_log_lines, [])

    def test_rlr_engine_block_on_stderr_is_fatal(self):
        """The case the box found: no habitat prefix, no return code, no ESP_ERROR.

        Nothing in FATAL_LOG_SUBSTRINGS matches it and HABITAT_LOG_PREFIX_RE cannot,
        so without the engine rule a broken context passes silently.
        """
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(), make_render(err=RLR_ENGINE_LOG))
        message = str(ctx.exception)
        self.assertIn("Error writing scene OBJ mesh", message)
        # The whole block, not just the line carrying the marker: the message and the
        # offending path are the informative parts and neither matches on its own.
        self.assertIn("/nonexistent-dir/x.obj", message)

    def test_engine_block_sets_its_own_flag(self):
        report = arm_audio_context(FakeAudioSensor(), make_render())
        self.assertFalse(report.rlr_engine_error)
        with self.assertRaises(AudioContextError):
            arm_audio_context(FakeAudioSensor(), make_render(err=RLR_ENGINE_LOG))

    def test_engine_block_on_stdout_is_not_fatal(self):
        """Same text, benign stream. The engine reports failures on fd 2."""
        report = arm_audio_context(
            FakeAudioSensor(), make_render(log=HEALTHY_LOG + RLR_ENGINE_LOG)
        )
        self.assertEqual(report.fatal_log_lines, [])
        self.assertFalse(report.rlr_engine_error)

    def test_third_party_stderr_noise_is_not_fatal(self):
        """A numpy warning on fd 2 must not fail a healthy audio context."""
        report = arm_audio_context(
            FakeAudioSensor(),
            make_render(err="UserWarning: builtin type SwigPyPacked has no __module__\n"),
        )
        self.assertEqual(report.fatal_log_lines, [])
        self.assertGreater(report.stderr_chars, 0)

    def test_the_obj_is_cleaned_up(self):
        sensor = FakeAudioSensor(n_vertices=20000)
        arm_audio_context(sensor, make_render())
        self.assertFalse(os.path.exists(sensor.obj_paths[0]))

    def test_zero_geometry_context_fails(self):
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(n_vertices=0), make_render())
        self.assertIn("0 vertices", str(ctx.exception))

    def test_degenerate_mesh_fails_the_scene_scale_floor(self):
        """`> 0` would pass this; the floor is why it does not."""
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(n_vertices=3), make_render())
        self.assertIn("below the 10000 floor", str(ctx.exception))

    def test_failed_obj_write_fails(self):
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(write_ok=False), make_render())
        self.assertIn("writeSceneMeshOBJ", str(ctx.exception))

    def test_fatal_log_substring_fails(self):
        log = HEALTHY_LOG + "Could not get the GenericSemanticMeshData\n"
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(), make_render(log=log))
        self.assertIn("GenericSemanticMeshData", str(ctx.exception))

    def test_fatal_substring_is_caught_on_either_stream(self):
        """ResourceManager's cast failure is ESP_ERROR, but the substring rule must not
        depend on which stream a future habitat-sim routes it to."""
        for kwargs in ({"log": HEALTHY_LOG + "Could not get the GenericSemanticMeshData\n"},
                       {"err": "Could not get the GenericSemanticMeshData\n"}):
            with self.assertRaises(AudioContextError) as ctx:
                arm_audio_context(FakeAudioSensor(), make_render(**kwargs))
            self.assertIn("GenericSemanticMeshData", str(ctx.exception))

    def test_missing_canary_fails_rather_than_passing_vacuously(self):
        """A capture that saw nothing has not verified invariant 2."""
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(), make_render(log=""))
        self.assertIn("unverified, not satisfied", str(ctx.exception))

    def test_canary_can_be_waived(self):
        report = arm_audio_context(
            FakeAudioSensor(), make_render(log=""), require_log_canary=False
        )
        self.assertFalse(report.log_canary_seen)

    def test_silent_ir_fails(self):
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(), make_render(ir=((0.0, 0.0), (0.0, 0.0))))
        self.assertIn("silent IR", str(ctx.exception))

    def test_all_failures_are_reported_together(self):
        """The first failure is rarely the diagnosis."""
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(
                FakeAudioSensor(n_vertices=0),
                make_render(log="", ir=((0.0,), (0.0,))),
            )
        message = str(ctx.exception)
        self.assertIn("failed 3 invariant(s)", message)
        self.assertIn("unverified", message)
        self.assertIn("0 vertices", message)
        self.assertIn("silent IR", message)

    def test_sensor_without_the_optional_methods_still_reports(self):
        class Minimal:
            def writeSceneMeshOBJ(self, path):
                with open(path, "w") as fh:
                    fh.write("v 0 0 0\n" * 20000)
                return True

        report = arm_audio_context(Minimal(), make_render())
        self.assertIsNone(report.ray_efficiency)
        self.assertIsNone(report.source_is_visible)


class TestLoggingPin(unittest.TestCase):
    def test_refuses_once_habitat_sim_is_imported(self):
        sys.modules["habitat_sim"] = object()
        try:
            with self.assertRaises(AudioContextError) as ctx:
                pin_habitat_logging()
            self.assertIn("already imported", str(ctx.exception))
        finally:
            del sys.modules["habitat_sim"]

    def test_sets_the_variable(self):
        saved = os.environ.get("HABITAT_SIM_LOG")
        try:
            value = pin_habitat_logging("Sensor=Debug")
            self.assertEqual(os.environ["HABITAT_SIM_LOG"], "Sensor=Debug")
            self.assertEqual(value, "Sensor=Debug")
        finally:
            if saved is None:
                os.environ.pop("HABITAT_SIM_LOG", None)
            else:
                os.environ["HABITAT_SIM_LOG"] = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)
