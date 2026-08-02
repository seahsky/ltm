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

``FakeRender`` writes to file descriptor 2 with ``os.write`` rather than to
``sys.stderr``, because that is how habitat-sim's C++ logger writes and a capture that
only sees ``sys.stderr`` is the vacuous-pass this ticket exists to prevent.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_guard import (  # noqa: E402
    AudioContextError,
    apply_audio_config,
    arm_audio_context,
    assert_no_swallowed_keys,
    bound_field_names,
    capture_fd_stderr,
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


HEALTHY_LOG = (
    "[Audio] Semantic scene does not exist or materials are disabled, "
    "will use default material\n[Audio] Loading non-semantic mesh\n"
    "Vertex count : 392356 , Index count : 1185054\n"
)


def make_render(log=HEALTHY_LOG, ir=((0.0, 0.163), (0.02, -0.09))):
    def render():
        if log:
            os.write(2, log.encode())
        return ir

    return render


# ----------------------------------------------------------------------


class TestFdCapture(unittest.TestCase):
    def test_captures_writes_to_fd_2(self):
        with capture_fd_stderr() as captured:
            os.write(2, b"from the C++ side\n")
        self.assertIn("from the C++ side", captured.text)

    def test_restores_fd_2_after_an_exception(self):
        before = os.dup(2)
        try:
            with self.assertRaises(ValueError):
                with capture_fd_stderr():
                    os.write(2, b"noise\n")
                    raise ValueError("boom")
            os.write(2, b"")  # fd 2 is still a valid descriptor
        finally:
            os.close(before)

    def test_does_not_swallow_the_exception(self):
        with self.assertRaises(KeyError):
            with capture_fd_stderr():
                raise KeyError("k")


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


class TestArmAudioContext(unittest.TestCase):
    def test_healthy_context_passes(self):
        report = arm_audio_context(FakeAudioSensor(n_vertices=392356), make_render())
        self.assertEqual(report.n_vertices, 392356)
        self.assertTrue(report.obj_written)
        self.assertTrue(report.log_canary_seen)
        self.assertAlmostEqual(report.ir_peak_abs, 0.163, places=6)
        self.assertAlmostEqual(report.ray_efficiency, 0.548, places=6)
        self.assertEqual(report.fatal_log_lines, [])

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

    def test_severity_marker_fails(self):
        log = HEALTHY_LOG + "[12:00:00]:[Error]:[Sensor] AudioSensor.cpp(512) : something\n"
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(FakeAudioSensor(), make_render(log=log))
        self.assertIn("AudioSensor.cpp(512)", str(ctx.exception))

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
