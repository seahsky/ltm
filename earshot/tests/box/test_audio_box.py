#!/usr/bin/env python3
"""The audio layer against the real renderer. V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

**These tests print their measurements** (ADR-0014). Ticket 16's box trip left numbers,
and those numbers are what made tickets 09, 15 and 17 decidable.

Four things a Mac cannot settle, and one of them is the reason this file exists.

1. **Which frame the lateral cue arrives in.** ``TestLateralFrameConvention`` is ticket
   22's named box test. The grid rendered at identity listener yaw, so the cue was
   **world-frame** and the fusion arc compensated with ``heard == -right(world-bearing)``.
   Live rendering uses the agent's real transform, so the same function should now
   return an **agent-frame** cue with no code change. Ticket 09 found that from source;
   this measures it. The test renders one fixed source from one position twice — facing
   it, then turned 180 degrees away. The agent frame predicts the sign **flips**; the
   world frame predicts it does not. That pair is decisive, and no fake can produce it.
   If it comes back world-frame, ``agent/controller.py`` needs the compensation term
   back and this test says so in its failure message rather than leaving a controller
   that turns the wrong way on every stall looking like a mediocre climb.
2. **Whether the real spec takes the configuration.** ``audio/spec.py`` asserts its own
   post-conditions, but against a fake whose ``py::dynamic_attr`` behaviour is imitated.
   Here the spec is the real one.
3. **What the received signal actually looks like.** The IR is scene- and pose-dependent
   (ticket 06 measured ``[2, 72300]`` against a 4.0 s ``maxIRLength`` cap), and
   ``clips.render_through_ir`` is new code that has never met one.
4. **The per-step cost, in the runner's own units.** Ticket 06 measured 27.2 ms/step for
   the render alone at this preset. Smoke criterion 7 wants it audited on every run
   rather than trusted from one sweep, and the convolution is now part of the per-step
   bill.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import math
import os
import time
import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time.
import earshot  # noqa: F401
import numpy as np

from earshot.audio.bed import bed_signal, heard_signal
from earshot.audio.clips import render_through_ir, rms, synthetic_burst
from earshot.audio.config import AudioConfig
from earshot.audio.lateral import (
    LATERAL_AMBIGUOUS,
    bearing_lateral_sign,
    interaural_level_difference,
    lateral_sign,
)
from earshot.audio.sensor import AudioSensorHandle
from earshot.audio.spec import ACOUSTICS_PRESET, audio_sensor_spec
from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene
from earshot.types import Pose, Xyz

SPLIT = os.environ.get("SS2_SPLIT", "val")
PLACEMENT_SEED = 20260804
# Far enough for the two ears to differ, near enough to stay in one room. The cue is
# fold-invariant so the distance is not critical; what matters is that the source is not
# dead ahead, which is the one bearing at which the sign is legitimately zero.
SOURCE_OFFSET_M = 2.0
LOS_DRAW_TRIES = 32

_DATASET = None
_SCENES_DIR = None


def setUpModule():
    """Find any ObjectNav scene whose mesh is on this box.

    Deliberately simpler than ``test_world_box``'s search: that one prefers an
    *unannotated* scene because ticket 08's claim turns on the absence of a
    ``.semantic.glb``. Nothing here cares — semantics are off either way (ADR-0007) —
    so this takes the first mesh it finds and says which.
    """
    global _DATASET, _SCENES_DIR

    split_dir = find_split_dir(SPLIT)
    _SCENES_DIR = find_scenes_dir()
    scenes = available_scenes(split_dir)
    override = os.environ.get("SS2_SCENE_LABEL")
    for label in [override] if override else list(scenes):
        dataset = load_scene(split_dir, label, scenes_dir=_SCENES_DIR)
        if os.path.exists(dataset.scene_path):
            _DATASET = dataset
            break
    if _DATASET is None:
        raise unittest.SkipTest(
            "no ObjectNav {} scene has its mesh on this box (looked under {})".format(
                SPLIT, _SCENES_DIR
            )
        )
    print("\n  scene: {}".format(_DATASET.scene_label), flush=True)
    print("  mesh:  {}".format(_DATASET.scene_path), flush=True)


def _quaternion_for_yaw(yaw_rad):
    """``[x, y, z, w]`` for a rotation about +y, which is the order ``set_pose`` takes."""
    return [0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0)]


class _AudioWorld:
    """One world with the real audio spec attached, and its armed handle.

    Built through ``sim.world.audio_spec_parts`` and ``audio.spec.audio_sensor_spec`` —
    the tree's one configuration path — rather than by assembling a spec here. A box
    test that built its own spec would measure a configuration nothing runs.
    """

    def __init__(self, config=None):
        from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs

        self.config = config or AudioConfig()
        spec, binaural = audio_spec_parts()
        audio_sensor_spec(spec, self.config, binaural)
        self.audio_uuid = str(spec.uuid)
        self.spec = spec
        self.world = World(_DATASET.scene_path, camera_sensor_specs(width=256, height=256) + [spec])
        self.world.seed_navmesh(PLACEMENT_SEED)
        self.handle = None

    def arm(self, source):
        self.handle = AudioSensorHandle(
            self.world.sensor_handle(self.audio_uuid),
            self.world.observe,
            source,
            uuid=self.audio_uuid,
        )
        return self.handle

    def ir_at(self, position, yaw_rad):
        """Seat the agent, render once, return the raw binaural IR."""
        self.world.set_pose(position, _quaternion_for_yaw(yaw_rad))
        observation, _ = self.handle.observe()
        return self.handle.audio_of(observation)

    def close(self):
        self.world.close()


class TestSpecOnTheRealBinding(unittest.TestCase):
    """The configuration path, against the spec whose ``dynamic_attr`` the fakes imitate."""

    def test_the_preset_and_the_layout_take_on_the_real_spec(self):
        from earshot.sim.world import audio_spec_parts

        spec, binaural = audio_spec_parts()
        before = {key: getattr(spec.acousticsConfig, key) for key in ACOUSTICS_PRESET}
        audio_sensor_spec(spec, AudioConfig(), binaural)
        after = {key: getattr(spec.acousticsConfig, key) for key in ACOUSTICS_PRESET}

        print("\n  --- ticket 22: the spec on the real binding ---", flush=True)
        print("  uuid:            {!r}".format(spec.uuid), flush=True)
        print("  enableMaterials: {}".format(spec.enableMaterials), flush=True)
        print("  channelLayout:   {} / {}".format(
            spec.channelLayout.type, spec.channelLayout.channelCount), flush=True)
        for key in sorted(ACOUSTICS_PRESET):
            print("  {:<20} {} -> {}".format(key, before[key], after[key]), flush=True)
        print("  vars(spec) after configuration: {}".format(sorted(vars(spec))), flush=True)

        self.assertEqual(after, dict(ACOUSTICS_PRESET))
        self.assertEqual(spec.acousticsConfig.sampleRate, 44100.0)
        self.assertIs(spec.enableMaterials, False)
        self.assertEqual(spec.channelLayout.channelCount, 2)


class TestLateralFrameConvention(unittest.TestCase):
    """**Ticket 22's named box test.** Which frame does the live cue arrive in?"""

    @classmethod
    def setUpClass(cls):
        cls.audio = _AudioWorld()
        # A navigable listener position, and a source offset along world +x at ear
        # height. Redrawn until the engine reports line of sight: an occluded pair still
        # has a lateral cue, but a weak one, and this test is about the SIGN rather than
        # about propagation through walls.
        cls.listener = None
        cls.source = None
        for _ in range(LOS_DRAW_TRIES):
            point = cls.audio.world.random_navigable_point()
            source = Xyz(point.x + SOURCE_OFFSET_M, point.y + 0.5, point.z)
            # Seat the agent BEFORE arming. `arm_audio_context` owns the first render
            # and rejects a silent IR, and habitat's default agent pose is wherever the
            # scene puts it — possibly inside geometry, metres from this source.
            cls.audio.world.set_pose(point, _quaternion_for_yaw(0.0))
            if cls.audio.handle is None:
                cls.audio.arm(source)
            else:
                cls.audio.handle.set_source(source)
                cls.audio.handle.observe()
            cls.listener, cls.source = point, source
            if cls.audio.handle.source_is_visible():
                break
        else:  # every draw was occluded: measure it anyway and say the cue may be weak
            print("  WARNING: no line-of-sight draw in {} tries — the cue may be "
                  "weak".format(LOS_DRAW_TRIES), flush=True)

    @classmethod
    def tearDownClass(cls):
        cls.audio.close()

    def test_the_cue_is_agent_frame_and_flips_when_the_agent_turns(self):
        facing_ir = self.audio.ir_at(self.listener, 0.0)
        turned_ir = self.audio.ir_at(self.listener, math.pi)

        facing = np.asarray(facing_ir, dtype=np.float32)
        turned = np.asarray(turned_ir, dtype=np.float32)
        # The simulator's own reading of the pose it was just given — if `set_pose` and
        # `yaw_from_quaternion` disagree, the prediction below is about a different
        # heading than the one that was rendered, and everything after it is noise.
        measured_yaw = self.audio.world.pose().yaw_rad
        self.assertAlmostEqual(abs(measured_yaw), math.pi, places=4)
        predicted_facing = bearing_lateral_sign(Pose(self.listener, 0.0), self.source)
        predicted_turned = bearing_lateral_sign(Pose(self.listener, math.pi), self.source)

        print("\n  --- ticket 22: the lateral frame convention ---", flush=True)
        print("  listener: {}".format(self.listener), flush=True)
        print("  source:   {}  (world +x by {} m)".format(self.source, SOURCE_OFFSET_M), flush=True)
        print("  source_is_visible: {}".format(self.audio.handle.source_is_visible()), flush=True)
        for label, ir in (("facing (yaw 0)", facing), ("turned (yaw pi)", turned)):
            print("  {:<16} shape {}  rms L {:.6g} R {:.6g}  ILD {:+.4f}  sign {:+d}".format(
                label, ir.shape, rms(ir[0]), rms(ir[1]),
                interaural_level_difference(ir), lateral_sign(ir)), flush=True)
        print("  agent-frame prediction: facing {:+d}, turned {:+d}  (measured yaw after "
              "the second pose: {:+.4f} rad)".format(
                  predicted_facing, predicted_turned, measured_yaw), flush=True)
        self.assertEqual(predicted_facing, -predicted_turned)

        heard_facing = lateral_sign(facing)
        heard_turned = lateral_sign(turned)

        self.assertNotEqual(
            heard_facing, LATERAL_AMBIGUOUS,
            "the two ears are numerically identical at a source 2 m off-axis, so the "
            "channel layout is not binaural or the listener transform is not being "
            "applied at all",
        )
        self.assertEqual(
            heard_facing, -heard_turned,
            "THE CUE IS WORLD-FRAME. Turning the agent 180 degrees did not flip the "
            "sign ({:+d} then {:+d}), so live rendering behaves as the grid did and "
            "agent/controller.py needs the `heard == -right(world-bearing)` "
            "compensation ticket 09 said to drop. Do not 'fix' this by inverting "
            "lateral_sign: the frame is the finding.".format(heard_facing, heard_turned),
        )
        self.assertEqual(
            heard_facing, predicted_facing,
            "the sign is agent-frame but INVERTED against the prediction, so ear 0 is "
            "the right channel rather than the left. Flip LEFT_EAR/RIGHT_EAR in "
            "audio/lateral.py and re-run — every other consumer reads the sign, not "
            "the channels.",
        )
        print("  PINNED: agent-frame, ear 0 = left. No compensation term.", flush=True)


class TestTheReceivedSignal(unittest.TestCase):
    """``render_through_ir`` and the bed, on an IR the simulator actually produced."""

    @classmethod
    def setUpClass(cls):
        cls.audio = _AudioWorld()
        cls.point = cls.audio.world.random_navigable_point()
        cls.audio.world.set_pose(cls.point, _quaternion_for_yaw(0.0))
        cls.audio.arm(Xyz(cls.point.x + SOURCE_OFFSET_M, cls.point.y + 0.5, cls.point.z))

    @classmethod
    def tearDownClass(cls):
        cls.audio.close()

    def test_the_ir_is_real_and_the_received_signal_is_in_the_beds_domain(self):
        config = self.audio.config
        clip = synthetic_burst(config.sample_rate, 0.5, config.target_norm_rms_db)
        bed = bed_signal(clip.size, config.bed_rms)

        observation, guard_report = self.audio.handle.observe()
        ir = np.asarray(self.audio.handle.audio_of(observation), dtype=np.float32)
        rendered = render_through_ir(ir, clip)
        silent = heard_signal(ir, clip, bed, playing=False)
        playing = heard_signal(ir, clip, bed, playing=True)

        print("\n  --- ticket 22: the received signal ---", flush=True)
        print("  IR shape {}  ({:.3f} s at {} Hz)  peak {:.6g}".format(
            ir.shape, ir.shape[1] / float(config.sample_rate), config.sample_rate,
            float(np.max(np.abs(ir)))), flush=True)
        print("  clip {} samples  rms {:.6g}".format(clip.size, rms(clip)), flush=True)
        print("  rendered rms {:.6g}   bed rms {:.6g}   ratio {:.1f}x".format(
            rms(rendered), rms(bed), rms(rendered) / max(rms(bed), 1e-12)), flush=True)
        print("  pre-onset heard rms {:.6g} (must equal the bed exactly)".format(
            rms(silent)), flush=True)
        print("  playing  heard rms {:.6g}".format(rms(playing)), flush=True)
        print("  guard: canary {}  engine error {}".format(
            guard_report.log_canary_seen, guard_report.rlr_engine_error), flush=True)

        self.assertEqual(rendered.shape, (2, clip.size))
        self.assertGreater(rms(rendered), 0.0, "the render is silent")
        self.assertEqual(rms(silent), rms(bed))
        self.assertGreater(rms(playing), rms(bed))
        # The IR is trimmed to actual decay, not to maxIRLength — so nothing downstream
        # may assume a fixed width (ticket 06, finding 4).
        self.assertLess(
            ir.shape[1], int(4.0 * config.sample_rate),
            "the IR fills the whole maxIRLength cap, which contradicts ticket 06's "
            "measurement that it is trimmed to decay",
        )

    def test_the_per_step_bill_is_measured_not_assumed(self):
        """Smoke criterion 7 wants this audited every run. Here it is at its source.

        Reported, not gated: ticket 06 measured 2.3x pose variance against ticket 04 on
        the same scene, so a tight bound here would fail for a reason that is not a
        regression. The generous ceiling is the one the runner enforces.
        """
        config = self.audio.config
        clip = synthetic_burst(config.sample_rate, 0.5, config.target_norm_rms_db)
        bed = bed_signal(clip.size, config.bed_rms)

        render_ms, convolve_ms = [], []
        for _ in range(8):
            start = time.time()
            observation, _ = self.audio.handle.observe()
            ir = self.audio.handle.audio_of(observation)
            render_ms.append((time.time() - start) * 1000.0)
            start = time.time()
            heard_signal(ir, clip, bed, playing=True)
            convolve_ms.append((time.time() - start) * 1000.0)

        steady_render = sorted(render_ms[1:])[len(render_ms[1:]) // 2]
        steady_convolve = sorted(convolve_ms[1:])[len(convolve_ms[1:]) // 2]
        print("\n  --- ticket 22: the per-step bill ---", flush=True)
        print("  guarded render (median of 7): {:8.2f} ms   first {:8.2f} ms".format(
            steady_render, render_ms[0]), flush=True)
        print("  heard_signal   (median of 7): {:8.2f} ms".format(steady_convolve), flush=True)
        print("  total per step:               {:8.2f} ms  = {:.1f} s per 500-step "
              "episode".format(steady_render + steady_convolve,
                               (steady_render + steady_convolve) * 500.0 / 1000.0), flush=True)
        print("  ticket 06 measured 27.2 ms for the render alone at this preset; the "
              "guard's two tempfiles and the convolution are the additions.", flush=True)
        self.assertGreater(steady_render, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
