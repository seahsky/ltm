#!/usr/bin/env python3
"""The audio guard against the real binary. Needs the V100 and the `ss2` env.

    conda activate ss2
    python -m unittest discover earshot/tests/box

Carried from ``.scratch/ss2-clean-room/probes/audioguard_probe.py`` (tickets 12 and 16),
reshaped from a staged script into a ``unittest`` suite per ADR-0014 so the box side
stops having a different mental model from the Mac side. The four negative controls are
now test methods and the driver concerns moved to ``earshot/tools/box_gate.sh``.

**What this adds over the Mac suite**, which is fully green against fakes: the three
things a fake cannot settle, each of which would otherwise be an inference.

1. **Does the healthy path pass?** Against ticket 04's control — 392,356 verts on
   ``minival/00800-TEEsavR23oF``, non-semantic path — ``arm_audio_context`` must return
   a report rather than raise.
2. **Does the guard actually fire?** A guard that has only ever passed is
   indistinguishable from a guard that cannot fail. ADR-0014 makes both arms mandatory,
   and it is earned by named incidents in both directions: ticket 13's torch layer
   skipped on mere importability (under-firing), and ticket 15 found a false positive in
   ticket 12's guard that would have fired on **every healthy run** (over-firing).
3. **Are the calibrated constants right on the binary?** Ticket 16 settled both from
   source before this ever ran — ``HABITAT_SIM_LOG_PIN`` is correct because
   ``AudioSensor.cpp`` opens ``namespace esp::sensor`` (``:12-13``) so its macros
   resolve to ``Subsystem::sensor``, and the old ``r"\\[Error\\]"`` pattern was not
   miscalibrated but structurally dead, since ``buildMessagePrefix``
   (``Logging.cpp:149-152``) emits no severity tag. Severity is the **stream**.

**These tests print their measurements.** Ticket 16's box trip left numbers, not just
green — 916 chars stdout versus 0 stderr on a healthy render, the +8 vertex gap between
what habitat submits (392,356) and what the engine holds (392,364), 0.814 s and 32.2 MB
for the OBJ write — and those numbers are what made tickets 09, 15 and 17 decidable. A
bare pass/fail run discards exactly the evidence the next decision needs.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import glob
import os
import time
import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time. `pin_habitat_logging` raises rather than no-ops if
# it is already loaded, so this ordering is enforced by Python and not by a comment.
import earshot  # noqa: F401
from earshot.audio.guard import (
    HABITAT_LOG_PREFIX_RE,
    HABITAT_SIM_LOG_PIN,
    MIN_SCENE_VERTICES,
    RLR_ENGINE_RE,
    AudioContextError,
    apply_audio_config,
    arm_audio_context,
    assert_no_swallowed_keys,
    bound_field_names,
    capture_habitat_logs,
    guarded_observe,
)

# Ticket 04's control, on minival/00800-TEEsavR23oF, non-semantic path.
CONTROL_VERTICES = 392356

PLACEMENT_SEED = 20260803
# A source drawn independently of the listener can land across the scene, and the guard
# asserts a non-silent IR — so an unlucky draw would fail the healthy path for a reason
# that has nothing to do with the audio context. Bound the separation instead.
# Euclidean, not geodesic, deliberately: no extra API surface, and the point is only to
# stay in earshot.
SOURCE_MIN_M = 1.0
SOURCE_MAX_M = 8.0
SOURCE_DRAW_TRIES = 64

_SIM = None
_AUDIO_SENSOR = None
_SPEC = None
_SCENE = None
_PLACEMENT = None


def _find_scene():
    explicit = os.environ.get("SS2_SCENE")
    if explicit:
        return explicit
    # Canonical root is data/hm3d/scene_datasets/hm3d/<split> — verified, it is what
    # box_inventory.py counted the 100 val / 10 minival meshes from. The bare
    # data/scene_datasets form is a legacy layout, kept as a fallback.
    patterns = [
        "data/hm3d/scene_datasets/hm3d/minival/**/*.basis.glb",
        "data/hm3d/scene_datasets/hm3d/minival/**/*.glb",
        "data/hm3d/scene_datasets/hm3d/val/**/*.basis.glb",
        "data/scene_datasets/hm3d/minival/**/*.basis.glb",
        "data/scene_datasets/hm3d/minival/**/*.glb",
        "data/scene_datasets/hm3d/val/**/*.basis.glb",
    ]
    for pattern in patterns:
        hits = [
            p
            for p in glob.glob(pattern, recursive=True)
            if "semantic" not in os.path.basename(p)
        ]
        if hits:
            return sorted(hits)[0]
    raise unittest.SkipTest("no HM3D .glb found — set SS2_SCENE")


def _place(sim):
    """Seat the listener and the source, reproducibly.

    Seeded because a RED run that cannot be reproduced is not a negative control.
    """
    import numpy as np

    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    placement = {"seeded": False, "draws": 0, "fallback": False}
    if sim.pathfinder.is_loaded:
        if hasattr(sim.pathfinder, "seed"):
            sim.pathfinder.seed(PLACEMENT_SEED)
            placement["seeded"] = True
        listener = sim.pathfinder.get_random_navigable_point()
        source = None
        candidate = listener
        separation = 0.0
        for attempt in range(SOURCE_DRAW_TRIES):
            candidate = sim.pathfinder.get_random_navigable_point()
            separation = float(
                np.linalg.norm(np.asarray(candidate) - np.asarray(listener))
            )
            placement["draws"] = attempt + 1
            if SOURCE_MIN_M <= separation <= SOURCE_MAX_M:
                source = candidate
                break
        if source is None:
            # Small or oddly-shaped navmesh: take the last draw rather than fail here.
            # The guard's silent-IR check is then the honest report of what happened.
            source = candidate
            placement["fallback"] = True
        placement["separation_m"] = round(separation, 3)
    else:
        listener = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        source = np.array([1.0, 0.0, 1.0], dtype=np.float32)
        placement["navmesh_loaded"] = False
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = listener
    agent.set_state(state)
    audio_sensor.setAudioSourceTransform(np.asarray(source, dtype=np.float32))
    placement["listener"] = [round(float(v), 3) for v in np.asarray(listener).tolist()]
    placement["source"] = [round(float(v), 3) for v in np.asarray(source).tolist()]
    return placement


def setUpModule():
    """Stand up sim + audio sensor once, exactly as the rebuilt runner will.

    Config goes through ``apply_audio_config`` rather than a bare ``setattr``, so this
    exercises the key validator against the real ``AudioSensorSpec`` and not a fake —
    which is the entire reason invariant 3 needs a box test at all.
    """
    global _SIM, _AUDIO_SENSOR, _SPEC, _SCENE, _PLACEMENT
    import quaternion  # noqa: F401  must precede habitat_sim (issue #1813)
    import habitat_sim

    _SCENE = _find_scene()
    print("\n  HABITAT_SIM_LOG pinned to {!r}".format(HABITAT_SIM_LOG_PIN), flush=True)
    print("  MIN_SCENE_VERTICES = {}".format(MIN_SCENE_VERTICES), flush=True)
    print("  scene: {}".format(_SCENE), flush=True)

    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = _SCENE
    for field, value in (("load_semantic_mesh", False), ("enable_physics", False)):
        if hasattr(backend_cfg, field):
            setattr(backend_cfg, field, value)
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    _SIM = habitat_sim.Simulator(habitat_sim.Configuration(backend_cfg, [agent_cfg]))

    _SPEC = habitat_sim.AudioSensorSpec()
    apply_audio_config(
        _SPEC,
        {
            # Permanently off per ADR-0007 — this is the path the clean room runs, and
            # it is SoundSpaces' own HM3D reference configuration.
            "uuid": "audio_sensor",
            "enableMaterials": False,
            "acousticsConfig": {"sampleRate": 44100.0},
        },
    )
    _SPEC.channelLayout.type = (
        habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    )
    _SPEC.channelLayout.channelCount = 2
    _SIM.add_sensor(_SPEC)
    _AUDIO_SENSOR = _SIM.get_agent(0)._sensors["audio_sensor"]
    _PLACEMENT = _place(_SIM)
    print("  placement: {}".format(_PLACEMENT), flush=True)


def tearDownModule():
    if _SIM is not None:
        _SIM.close()


def _render():
    return _SIM.get_sensor_observations()["audio_sensor"]


class TestKeyValidatorOnTheRealSpec(unittest.TestCase):
    """Invariant 3 against ``AudioSensorSpec``, which really does carry dynamic_attr."""

    def test_the_bound_fields_are_introspectable(self):
        import habitat_sim

        spec = habitat_sim.AudioSensorSpec()
        fields = sorted(bound_field_names(spec))
        print("\n  spec fields ({}): {}".format(len(fields), ", ".join(fields)), flush=True)
        print(
            "  acoustics fields: {}".format(len(bound_field_names(spec.acousticsConfig))),
            flush=True,
        )
        self.assertIn("acousticsConfig", fields)
        self.assertIn("channelLayout", fields)

    def test_a_stock_construct_leaves_only_the_known_dynamic_attr(self):
        """Ticket 16's stage-1 question, answered by ticket 15's budget run.

        The real constructor attaches ``__noise_model_kwargs`` as a genuine instance
        attribute, so ``vars(spec)`` is NOT empty on a healthy spec. Without
        ``KNOWN_DYNAMIC_ATTRS`` the guard raised on every good spec — invariant 3 would
        have been a false positive forever, and the Mac fakes never reproduced it.
        """
        import habitat_sim

        spec = habitat_sim.AudioSensorSpec()
        print("  vars(spec) after bare construct: {}".format(sorted(vars(spec)) or "empty"), flush=True)
        assert_no_swallowed_keys(spec)
        apply_audio_config(spec, {"uuid": "audio_sensor", "enableMaterials": False})
        print("  vars(spec) after configure:      {}".format(sorted(vars(spec)) or "empty"), flush=True)
        assert_no_swallowed_keys(spec)

    def test_the_branchs_own_rename_is_rejected_not_swallowed(self):
        """NEGATIVE CONTROL 1. ``irTime`` became ``maxIRLength`` on this branch."""
        import habitat_sim

        with self.assertRaises(AudioContextError) as ctx:
            apply_audio_config(habitat_sim.AudioSensorSpec(), {"irTime": 4.0})
        print("  irTime rejected: {}".format(str(ctx.exception)[:120]), flush=True)

    def test_a_key_attached_behind_the_validators_back_is_still_caught(self):
        """NEGATIVE CONTROL 2. ``vars(spec)`` is an exact probe, not a heuristic.

        ``def_readwrite`` installs a data descriptor on the type, so a real field never
        reaches the instance ``__dict__``; whatever is in there is precisely the set of
        keys ``py::dynamic_attr`` swallowed.
        """
        import habitat_sim

        dirty = habitat_sim.AudioSensorSpec()
        dirty.definitelyNotAField = 1.0
        with self.assertRaises(AudioContextError):
            assert_no_swallowed_keys(dirty)
        print("  swallowed key detected OK", flush=True)


class TestHealthyPath(unittest.TestCase):
    def test_arm_audio_context_passes_on_a_real_scene(self):
        t0 = time.time()
        report = arm_audio_context(_AUDIO_SENSOR, _render)
        elapsed = time.time() - t0
        print("\n  --- healthy path ---", flush=True)
        for key, value in report.as_dict().items():
            if key.endswith("_tail"):
                continue
            print("  {:<24} {}".format(key, value), flush=True)
        print("  {:<24} {:.3f} s".format("guard_total_s", elapsed), flush=True)
        self.assertTrue(report.obj_written)
        self.assertTrue(report.log_canary_seen)
        self.assertGreaterEqual(report.n_vertices, MIN_SCENE_VERTICES)
        self.assertGreater(report.ir_peak_abs, 0.0)
        self.assertEqual(report.fatal_log_lines, [])

        if report.submitted_n_vertices is not None:
            delta = report.n_vertices - report.submitted_n_vertices
            # RECORDED, NEVER ASSERTED. Ticket 16 measured +8: habitat submits 392,356
            # and the engine holds 392,364. That gap is the first direct evidence that
            # invariant 1 reads the ENGINE's geometry rather than habitat's, and the
            # closed .so does not say what it adds.
            print(
                "  submitted {} verts, engine holds {} (delta {}; ticket 16 measured +8)".format(
                    report.submitted_n_vertices, report.n_vertices, delta
                ),
                flush=True,
            )
        print(
            "  matches ticket 04's control ({}): {}".format(
                CONTROL_VERTICES, report.n_vertices == CONTROL_VERTICES
            ),
            flush=True,
        )

    def test_a_healthy_render_logs_to_stdout_and_leaves_stderr_empty(self):
        """The claim ticket 16 pre-flighted from source: ESP_DEBUG is on fd 1.

        Measured 916 chars stdout / 0 stderr. Capturing fd 2 alone — which is what this
        guard did before ticket 16 read Corrade's ``Debug.cpp:525`` — would have raised
        on a perfectly good context, every run.
        """
        with capture_habitat_logs() as captured:
            _render()
        print(
            "\n  render log split: stdout {} chars / stderr {} chars".format(
                len(captured.stdout), len(captured.stderr)
            ),
            flush=True,
        )
        self.assertGreater(len(captured.stdout), 0)
        self.assertEqual(len(captured.stderr), 0)

    def test_the_canary_stays_armed_on_later_renders(self):
        """Ticket 16 measured this True, and its own prediction of False was wrong.

        ``Vertex count`` IS a first-render artefact — the mesh uploads once, via
        ``newInitialization_`` — but the other canary substring is ``logHeader_``, and
        ``runSimulation`` logs ``[Audio] Running the audio simulator``
        (``AudioSensor.cpp:130``) on EVERY render. That is what gives
        ``guarded_observe`` a consumer for the whole episode rather than only at arm
        time.
        """
        with capture_habitat_logs() as captured:
            _render()
        on_stdout = any(m in captured.stdout for m in ("Vertex count", "[Audio]"))
        on_stderr = any(m in captured.stderr for m in ("Vertex count", "[Audio]"))
        print(
            "  canary on later render — stdout {} / stderr {}".format(on_stdout, on_stderr),
            flush=True,
        )
        self.assertTrue(on_stdout)

    def test_guarded_observe_passes_on_every_step_of_a_short_run(self):
        """The per-step half, on the binary. NEW CODE — its fakes never met this.

        Ten steps rather than one, because the failure it exists to catch is a context
        that degrades partway through an episode. Also prices the per-step guard, which
        lands inside the wall-clock the task spec requires reporting every run.
        """
        elapsed = []
        for step in range(10):
            t0 = time.time()
            observation, report = guarded_observe(_render, occasion="step {}".format(step))
            elapsed.append(time.time() - t0)
            self.assertTrue(report.log_canary_seen)
            self.assertEqual(report.fatal_log_lines, [])
            self.assertIsNotNone(observation)
        print(
            "\n  guarded_observe over 10 steps: min {:.1f} ms / mean {:.1f} ms / max {:.1f} ms".format(
                1000 * min(elapsed), 1000 * sum(elapsed) / len(elapsed), 1000 * max(elapsed)
            ),
            flush=True,
        )


class TestNegativeControls(unittest.TestCase):
    """The guard must FIRE. A detector ships with both arms (ADR-0014)."""

    def test_an_impossible_vertex_floor_fires(self):
        """NEGATIVE CONTROL 3. Proves the assertion path is live on the real sensor.

        It does NOT prove a genuinely empty mesh is detectable — that would need a
        scene which actually produces one — and saying so is the difference between a
        control and a claim.
        """
        with self.assertRaises(AudioContextError) as ctx:
            arm_audio_context(_AUDIO_SENSOR, _render, min_vertices=10 ** 9)
        print("\n  impossible floor fires: {}".format(str(ctx.exception)[:160]), flush=True)

    def test_a_provoked_engine_failure_is_visible_on_fd_2(self):
        """NEGATIVE CONTROL 4, and the finding that made invariant 2's engine arm exist.

        Run 1 of ticket 16 found habitat's own prefix on fd 2 for NONE of these three:
        the closed engine writes its own un-prefixed block, and
        ``RLRA_SetListenerHRTF`` returns ``Success`` over a failed load, so
        ``AudioSensor.cpp:181``'s ESP_ERROR never fires. ``RLRA_WriteSceneMeshOBJ``
        correctly returns failure — which is why invariant 1's ``is True`` check is
        load-bearing AND invariant 2's log scan cannot be retired. The return code is
        trustworthy for some calls and not others.
        """
        provocations = [
            ("setListenerHRTF", lambda: _AUDIO_SENSOR.setListenerHRTF("/nonexistent/hrtf.wav")),
            ("setAudioMaterialsJSON", lambda: _AUDIO_SENSOR.setAudioMaterialsJSON("/nonexistent/m.json")),
            ("writeSceneMeshOBJ", lambda: _AUDIO_SENSOR.writeSceneMeshOBJ("/nonexistent-dir/x.obj")),
        ]
        engine_seen = False
        print("\n  --- provocations ---", flush=True)
        for name, call in provocations:
            raised, returned = None, "<not reached>"
            with capture_habitat_logs() as captured:
                try:
                    returned = repr(call())
                except Exception as exc:  # a raising binding is a finding, not a failure
                    raised = repr(exc)
            habitat_prefixed = any(
                HABITAT_LOG_PREFIX_RE.search(ln) for ln in captured.stderr.splitlines()
            )
            engine_block = bool(RLR_ENGINE_RE.search(captured.stderr))
            engine_seen = engine_seen or engine_block
            print(
                "  {:<22} out {:>5}c / err {:>5}c  habitat-prefix {:<5} engine-block {:<5} "
                "returned {} raised {}".format(
                    name,
                    len(captured.stdout),
                    len(captured.stderr),
                    str(habitat_prefixed),
                    str(engine_block),
                    returned,
                    raised,
                ),
                flush=True,
            )
            if captured.stderr.strip():
                print("    stderr tail: {!r}".format(captured.stderr[-400:]), flush=True)
        self.assertTrue(
            engine_seen,
            "no RLR engine block detected on fd 2 across three provoked failures — read "
            "the stderr tails above and fix RLR_ENGINE_RE; invariant 2's generic arm is "
            "blind without it",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
