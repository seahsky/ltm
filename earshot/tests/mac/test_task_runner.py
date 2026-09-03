"""The episode loop, driven end to end against fakes.

This is the highest-risk logic in the tree — it is the only place the four layers meet,
and until ticket 26's box trip it is also the only place they have ever met. So the loop
is duck-typed against ``FakeWorld`` and ``FakeAudioSensorHandle`` (see ``_task_fakes``)
and actually run, rather than being checked one function at a time.

**What a green here does and does not license** (ADR-0014). It licenses: the loop's
control flow, the funnel ladder, the artefact split, the criterion-1 accounting, and the
fact that the greedy climb closes on a source whose loudness and lateral cue are real
functions of the pose. It licenses nothing about the follower, the navmesh, the renderer
or the frame the live cue arrives in — ticket 22's box test owns that last one, and this
file's fake deliberately uses ``agent/occupancy``'s frame so that a green here is
consistent with it rather than independent evidence for it.
"""

import dataclasses
import math
import unittest

import numpy as np

import _audio_fakes as audio_fakes
from _interpreter import assert_interpreter  # noqa: F401
from _task_fakes import (
    FakeAudioSensorHandle,
    FakeWorld,
    make_anomaly_episode,
    make_episode,
    make_goal,
)

from earshot.agent.config import ControllerConfig, DetectorConfig
from earshot.agent.proposers import SOURCE_INVESTIGATE
from earshot.agent.reachability import EmptyPoolError
from earshot.audio.bed import bed_signal, heard_signal, mix_bed
from earshot.audio.calibration import CalibrationResult
from earshot.audio.clips import rms, synthetic_burst
from earshot.audio.config import AudioConfig, WindowPolicy
from earshot.audio.guard import AudioContextReport
from earshot.audio.lateral import LATERAL_AMBIGUOUS, bearing_lateral_sign
from earshot.audio.tail import (
    advance_tail,
    clip_readout,
    hop_samples,
    phase_folds,
    steady_state_cue_rms,
)
from earshot.audio.window import plan_window
from earshot.config import (
    CastPolicy,
    ClimbRule,
    Detector,
    IrPolicy,
    LateralCue,
    Localization,
    RunConfig,
)
from earshot.report.agent import SCHEMA_FIELDS
from earshot.report.audit import (
    CalibrationRecord,
    FunnelStage,
    SoundingWindowRecord,
)
from earshot.task.runner import (
    CALIBRATION_DRAWS,
    DIVERT_CANDIDATE_ID,
    SilentPhaseTally,
    TailNotActiveError,
    _divert_candidate,
    _funnel_stage,
    calibrate_episode,
    calibration_poses,
    make_detector,
    run_episode,
    silent_phase_tally,
    tail_is_active,
)
from earshot.memory.store import (
    EpisodicEntry,
    EpisodicStore,
    MemoryCondition,
    SemanticEntry,
    SemanticStore,
)
from earshot.task.memory_prior import RUN_DISCLOSURE
from earshot.types import Pose, Xyz

# A short clip so the convolution is cheap; the level is `AudioConfig`'s own
# `target_norm_rms_db` of -20 dBFS, i.e. RMS 0.1.
CLIP = synthetic_burst(44100, seconds=0.05)

# The threshold a real run derives from the §2.3 sweep. Fixed here so the loop's
# behaviour is the subject rather than the calibration's — `TestCalibrationPoses` owns
# that half. Between the bed (1e-3) and the received level at 5 m (~8e-3), which is
# where `calibrate_onset`'s geometric placement would put it.
CALIBRATION = CalibrationResult(
    onset_rms=0.003,
    bed_rms=1e-3,
    anomaly_low=0.008,
    anomaly_median=0.01,
    anomaly_min=0.005,
    anomaly_max=0.05,
    separation_db=18.0,
    n_poses=16,
    global_volume=1.0,
)


def make_config(**overrides):
    base = dict(
        run_dir="/nonexistent",
        max_steps=80,
        t_anom=2,
        localization=Localization.REALIZABLE,
        detector=Detector.ORACLE,
        # `CLIP` is 0.05 s = 2205 samples, and `open_tail` REFUSES a hop at or past the
        # read window: a step that outruns the window is a different sensor, one that
        # silently drops the audio between two steps. 0.01 s gives a 441-sample hop, a
        # 5-step ramp and a 6-step tail against `FakeAudioSensorHandle.IR_LENGTH = 64` —
        # the same shape the box gets from a 5 s clip at the real 1.0 s step.
        audio=AudioConfig(step_seconds=0.01),
    )
    base.update(overrides)
    return RunConfig(**base)


def run(world, handle, anomaly_episode, cfg, **kwargs):
    kwargs.setdefault("detector", make_detector(cfg, world, anomaly_episode))
    kwargs.setdefault("clip", CLIP)
    return run_episode(world, handle, anomaly_episode, cfg, **kwargs)


class TestTheFullLoop(unittest.TestCase):
    """Criterion 5: SEARCH, onset, INVESTIGATE, CHECK, RESUME, legitimate termination."""

    @classmethod
    def setUpClass(cls):
        # The agent faces -z (habitat's forward at zero yaw), the source is 5 m ahead and
        # the primary goal 9 m ahead, so the detour is on the way rather than behind. The
        # anomaly object differs from the find-target, which is the decoupled regime the
        # builder prefers.
        cls.world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
        cls.source = Xyz(0.0, 0.0, -5.0)
        cls.handle = FakeAudioSensorHandle(cls.world, cls.source)
        cls.episode = make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        cls.anomaly_episode = make_anomaly_episode(
            source=cls.source, episode=cls.episode, t_anom=2
        )
        cls.cfg = make_config()
        cls.result = run(
            cls.world, cls.handle, cls.anomaly_episode, cls.cfg, calibration=CALIBRATION
        )

    def test_the_whole_interrupt_resume_loop_ran(self):
        """CHECK and RESUME must both be reached (§8 criterion 5)."""
        audit = self.result.audit
        self.assertEqual(audit.funnel_stage, FunnelStage.PRIMARY_RESUMED)
        self.assertTrue(self.result.report.resumed)
        self.assertFalse(self.result.report.investigate_aborted)

    def test_the_agent_found_the_source_rather_than_being_told_where_it_was(self):
        """The realizable arm's claim, as an assertion about distance.

        The controller was handed no coordinate (``source_xyz=None`` in this arm), so the
        only thing that could have brought it within a step of the source is the energy
        climb over the fake's distance-dependent IR.

        **The bound is the ring, and it is inclusive now.** With the confirm sufficient
        the agent stops the tick it enters the ring rather than pressing on until the
        climb flattens, so a stop lands AT `oracle_radius_m` instead of inside it. That is
        the trade `cast-1` bought: arrivals counted, final proximity given up. Read from
        the config rather than written as `1.0`, so moving the ring moves this with it.
        """
        ring = DetectorConfig().oracle_radius_m
        self.assertIsNotNone(self.result.audit.dist_at_stop)
        self.assertLessEqual(self.result.audit.dist_at_stop, ring)
        self.assertIsNotNone(self.result.report.stopped_at_pose)

    def test_the_visual_confirm_named_the_object_at_the_source(self):
        self.assertEqual(
            self.result.report.visual_confirm_object,
            self.anomaly_episode.source.anomaly_object,
        )

    def test_render_count_equals_step_count_exactly(self):
        """Smoke criterion 1, measured on the loop rather than on the lifetime counter.

        Arming the guard and the calibration sweep both render before the first step, so
        the simulator's own ``n_renders`` is legitimately larger. The audit records the
        two numbers the criterion is actually about.
        """
        metrics = self.result.audit.metrics
        self.assertEqual(metrics["n_renders_in_loop"], metrics["n_loop_steps"])
        self.assertEqual(self.result.audit.n_render_steps, int(metrics["n_loop_steps"]))

    def test_the_per_step_record_is_every_step_and_carries_the_action(self):
        """§3.2, and the reason ``action`` is on it (ADR-0011).

        A rotation-driven rise in RMS has to be separable from a translation-driven one
        after the fact, and only the action taken separates them.
        """
        steps = self.result.audit.steps
        self.assertEqual([row.step for row in steps], list(range(len(steps))))
        self.assertTrue(all(row.audio_render_s is not None for row in steps))
        self.assertTrue(any(row.action == "move_forward" for row in steps))

    def test_the_pre_onset_signal_is_the_bed_at_every_pose(self):
        """§3.1's first invariant has content because the bed is unrendered (ADR-0009).

        Not "close to" — the bed is generated at the clip's length and RMS-normalised, so
        a healthy pre-onset reading is the bed level to float precision, and the anomaly
        contributes exactly zero rather than a scaled-down render.

        **The predicate is the STEP INDEX now, not ``source_playing``.** Those were the
        same set while the window never closed; with ADR-0017 the silent phase's rows
        also carry ``source_playing=False`` and their level is the decaying tail, so the
        old filter would have quietly changed what this test is about.
        ``test_the_silent_phase_decays_rather_than_cutting_to_the_bed`` is the other half.
        """
        t_anom = self.anomaly_episode.t_anom
        pre_onset = [row for row in self.result.audit.steps if row.step < t_anom]
        self.assertEqual(len(pre_onset), t_anom)
        for row in pre_onset:
            self.assertAlmostEqual(row.measured_rms, self.cfg.audio.bed_rms, places=6)

    def test_the_step_the_source_was_reached_is_on_the_record(self):
        """SWS's numerator, which was recoverable from no artefact this tree wrote.

        The primary STOP is ``len(steps) - 1``; the SOURCE reach was written nowhere —
        ``InvestigationEvent.investigate_steps`` is a relative count of INVESTIGATE ticks
        and the ORACLE localization arm leaves no ``realizable_action`` trail to read it
        off. Captured in the runner off ``ControllerState.investigated``, which flips
        exactly once in both arms, so ``agent/controller.py`` stays out of this diff.
        """
        audit = self.result.audit
        self.assertIsNotNone(audit.source_reached_step)
        self.assertGreaterEqual(audit.source_reached_step, self.anomaly_episode.t_anom)
        self.assertLess(audit.source_reached_step, len(audit.steps))
        # **The metric and the field are written from ONE runner local**, so comparing
        # them holds for any value including a wrong one. The independent anchor is the
        # per-step trace: the oracle detector confirms inside `oracle_radius_m`, so the
        # step the source was reached must be the FIRST step whose recorded pose is in
        # that ring. A reach step copied off the onset, or off by one, disagrees here.
        ring = DetectorConfig().oracle_radius_m
        in_ring = [
            row.step
            for row in audit.steps
            if row.position.horizontal_distance_to(audit.source_xyz) <= ring
        ]
        self.assertEqual(audit.source_reached_step, in_ring[0])
        self.assertEqual(
            audit.metrics["source_reached_step"], float(audit.source_reached_step)
        )
        # The row it indexes is inside the detour: the agent had already diverted, so the
        # step it "reached the source" on is one the controller was investigating at.
        self.assertGreaterEqual(
            audit.source_reached_step, self.result.audit.onset.onset_step
        )

    def test_the_onset_fired_at_or_after_t_anom_and_the_provenance_was_asserted(self):
        onset = self.result.audit.onset
        self.assertIsNotNone(onset.onset_step)
        self.assertGreaterEqual(onset.onset_step, self.anomaly_episode.t_anom)
        self.assertEqual(onset.n_pre_onset_readings, self.anomaly_episode.t_anom)
        # Ticket 16's discipline: the record says the assertion RAN, because an artefact
        # that exists at all looks like proof it passed.
        self.assertTrue(onset.provenance_asserted)
        self.assertEqual(self.result.report.heard_at_step, onset.onset_step)

    def test_the_audit_records_the_t_anom_this_episode_actually_ran(self):
        """It is derived per episode now, so the configuration does not know it.

        ``funnel_stage`` is computed from ``t_anom`` and sits in the same record, so an
        audit that omitted it would carry a stage nobody could re-derive — and smoke
        criterion 4 states §3.1's first invariant against it.
        """
        self.assertEqual(self.result.audit.t_anom, self.anomaly_episode.t_anom)

    def test_the_testimony_and_the_answer_key_are_disjoint_artefacts(self):
        """ADR-0013's boundary, on the objects this loop actually produced.

        ``test_report_boundary.py`` holds it on the types and
        ``test_report_artifacts.py`` on the bytes; this is the third place it can fail —
        a runner that populated the report from privileged state.
        """
        testimony = self.result.report.as_dict()
        self.assertEqual(sorted(testimony), sorted(SCHEMA_FIELDS))
        self.assertNotIn("source_xyz", testimony)
        self.assertIsNotNone(self.result.audit.source_xyz)

    def test_the_arms_are_recorded_because_the_testimony_cannot_show_them(self):
        self.assertEqual(self.result.audit.localization_arm, "realizable")
        self.assertEqual(self.result.audit.detector_arm, "oracle")

    def test_the_audio_bill_is_recorded_for_every_run(self):
        """§6's second new metric. An empty summary would pass a ceiling check vacuously."""
        summary = self.result.audit.audio_render_summary()
        self.assertEqual(summary["n"], self.result.audit.metrics["n_loop_steps"])
        self.assertGreater(summary["max_s"], 0.0)

    def test_clap_is_off_so_the_class_is_null_rather_than_copied_from_the_dataset(self):
        """The honest shape of the smoke's configuration (§4.3).

        The episode knows the anomaly is an alarm. The agent does not, because nothing
        classified it — and a report that named it anyway would be the task telling the
        agent what it heard.
        """
        self.assertIsNone(self.result.report.anomaly_class)
        self.assertEqual(self.anomaly_episode.anomaly_class, "alarm")

    def test_source_visibility_is_recorded_and_never_read_by_the_controller(self):
        """§3.3. The structural half is ``test_analyst_only.py``; this is the record."""
        self.assertTrue(all(row.source_is_visible for row in self.result.audit.steps))
        self.assertEqual(
            len(self.result.audit.source_is_visible_history),
            len(self.result.audit.steps),
        )


def _longest_held_run(rows):
    """``(start index, length)`` of the longest run of consecutive BLOCKED forwards.

    A blocked forward changes neither the pose (``allow_sliding`` is False) nor the
    heading, so a run of them is the only stretch of this fixture where the agent's
    geometry is genuinely held -- which is what a claim about "the reading at a held pose"
    needs. Locating it by search rather than by an offset from the first collision is what
    survives the controller turning at a different step: since ADR-0019 the stall fires
    much earlier and a fixed offset lands mid-cast.
    """
    best_start, best_length = 0, 0
    start, length = 0, 0
    for index, row in enumerate(rows):
        if row.action == "move_forward" and row.collided:
            if length == 0:
                start = index
            length += 1
            if length > best_length:
                best_start, best_length = start, length
        else:
            length = 0
    return best_start, best_length


class TestAWallTheClimbWalksInto(unittest.TestCase):
    """Ticket 26's box finding, reproduced at the seam it broke at.

    The first box episode walked 110 forwards for 6.57 m of path and never reached
    line-of-sight. Rising loudness kept saying forward into a wall the rule could not
    see, and a blocked forward does not move, so the next reading is not *lower* and the
    stall branch never fires either — the climb pushes the same wall until the sub-budget
    aborts.

    **This fixture is what stopped ticket 26 from shipping a collision branch on the
    climb.** With ``allow_sliding`` False a collided forward does not move, the RMS is a
    pure function of pose, so the reading repeats and ADR-0011's stall branch already
    turns — measured over four wall geometries, trajectories byte-identical with the rule
    reading the flag and ignoring it. The flag's job is the record, not the rule.

    It then produced the finding that *did* change the arm: the climb never escaped, in
    any geometry, ending pressed flat against the wall with **zero lateral movement** and
    unchanged by tripling the step budget. ``move_forward`` was its only translation and
    the gradient chose where forward pointed, so no sequence of its actions could go
    around anything. The detour now names a probe point and the follower routes to it.

    **This fixture cannot show that fix working, and does not claim to.** The fake
    follower steers in a straight line (``_task_fakes.FakeWorld.follower``), so it walks
    into the wall on the way to a probe behind it and the livelock persists here for a
    reason that is the fake's, not the system's. Routing is a navmesh capability and
    ``tests/box/test_investigate_route_box.py`` exercises it (ADR-0014). What this holds is
    the record, and the symptom, so a regression to blind stepping is still visible on a
    Mac.
    """

    @classmethod
    def setUpClass(cls):
        # A wall 1 m ahead, across the agent's path to a source directly beyond it. The
        # climb's honest move is forward; the wall is what makes forward useless.
        cls.world = FakeWorld(
            start=Xyz(0.0, 0.0, 0.0), yaw=0.0, wall=lambda p: p.z < -1.0
        )
        cls.source = Xyz(0.0, 0.0, -5.0)
        cls.handle = FakeAudioSensorHandle(cls.world, cls.source)
        cls.anomaly_episode = make_anomaly_episode(
            source=cls.source, t_anom=0, episode=make_episode(start_yaw=0.0)
        )
        cls.cfg = make_config(max_steps=40, t_anom=0)
        cls.result = run(
            cls.world, cls.handle, cls.anomaly_episode, cls.cfg, calibration=CALIBRATION
        )

    def test_the_record_separates_a_wall_from_a_step_that_moved(self):
        summary = self.result.audit.forward_summary()
        self.assertGreater(summary["n_forward"], 0)
        self.assertGreater(
            summary["n_collided"], 0, "the wall was never hit, so this proves nothing"
        )

    def test_a_collided_forward_is_recorded_as_displacing_nothing(self):
        walls = [
            row
            for row in self.result.audit.steps
            if row.action == "move_forward" and row.collided
        ]
        self.assertTrue(walls)
        for row in walls:
            self.assertAlmostEqual(row.displacement_m, 0.0)

    def test_a_held_pose_CYCLES_with_the_clip_loop_rather_than_repeating_the_reading(self):
        """**"A held pose repeats the reading" is FALSE since ADR-0017, and ADR-0019 made
        the cycle FIFTY TIMES BIGGER.**

        ``controller.py`` argues the collision flag is redundant because "the RMS is a
        pure function of pose", which was true while ``heard_signal`` convolved the whole
        clip fresh every step: the pre-ADR-0017 arm reads 0.01503124 nine times
        BIT-FOR-BIT at this wall (``TestTheWallAgainstThePreAdr0017Renderer``). The
        accumulator does not. The clip LOOPS with period ``phase_folds = N // gcd(N, hop)``
        and the cue readout is one of those folds, so the reading is PERIODIC.

        **The reading does not merely rotate, it CYCLES WITH THE CLIP'S OWN ENVELOPE.**
        Measured at this wall over the longest held run, the agent pressed flat against
        it and its heading unchanged:

            0.01577929  0.01628650  0.01598172  0.00736297  0.01458480   then repeat

        Five distinct values spanning 8.92e-03 -- **54.8% of the maximum**, against the
        clip readout's 0.38% on the same fixture. One fold in five carries little of the
        clip's energy and the cue readout says so; the 5 s window was averaging that away.
        That is the intermittency, it is honest, and it is why this test asserts the
        PERIOD and the VALUE SET rather than a tolerance on the spread.

        The CONCLUSION survives: an oscillation is not a climb, ADR-0011's stall branch
        still fires, and the flag stays recorded rather than consumed.
        """
        rows = self.result.audit.steps
        metrics = self.result.audit.metrics
        period = int(metrics["sounding_phase_folds"])
        start, length = _longest_held_run(rows)
        self.assertGreaterEqual(
            length,
            period + 1,
            "no held run long enough to see one loop period, so this proves nothing",
        )
        held = rows[start : start + length]
        for index in range(length - period):
            self.assertEqual(
                held[index].measured_rms,
                held[index + period].measured_rms,
                "the held pose is not periodic at the clip's own loop period, so the "
                "reading is carrying something other than the phase",
            )
        for index in range(min(3, length - 1)):
            self.assertNotEqual(
                held[index].measured_rms,
                held[index + 1].measured_rms,
                "the reading REPEATED at a held pose, which is the pre-ADR-0017 "
                "renderer's behaviour and not this one's",
            )
        levels = [row.measured_rms for row in held[:period]]
        self.assertEqual(len(set(levels)), period)
        spread = max(levels) - min(levels)
        # The clip readout's own spread here was 0.0038 of the maximum. Asserting a FLOOR
        # rather than a ceiling is the point: the intermittency is the measurement.
        self.assertGreater(
            spread / max(levels),
            0.40,
            "the cue readout's spread at a held pose collapsed to the clip readout's, "
            "so the reading is being averaged over the loop again",
        )

    def test_the_accumulators_memory_of_a_closer_pose_is_now_ONE_STEP_DEEP(self):
        """Consequence 2 of ADR-0019, and its number is the deliverable.

        Under the clip readout the reading kept RISING for ``tail_steps`` after a wall
        stopped the agent, because the 5 s window was still filling with the nearer pose
        it had just reached -- a rise ``is_rising`` cannot tell from a climb and the agent
        did not earn. That window is ``cue_tail_steps`` wide now: **2 here against a
        ``tail_steps`` of 6**, and the fake's IR is a single delta, so in this fixture the
        memory is effectively nil.

        What is asserted is the property that number implies: after ONE step at a held
        pose the reading is already purely periodic in the loop phase. The clip readout
        could not be -- it would still be carrying five older poses at that point -- so
        the lead-in being shorter than ``tail_steps`` is half the measurement.
        """
        rows = self.result.audit.steps
        window = self.result.audit.sounding_window
        period = int(self.result.audit.metrics["sounding_phase_folds"])
        self.assertEqual(window.cue_tail_steps, 2)
        self.assertEqual(window.tail_steps, 6)
        start, length = _longest_held_run(rows)
        # How long the pose and heading had been held when the run's FIRST reading was
        # taken. The step before a blocked-forward run is the last thing that could have
        # moved or turned the agent, so the lead-in is exactly one step.
        lead_in = 1
        self.assertLessEqual(
            lead_in,
            window.cue_tail_steps,
            "the run's lead-in is longer than the cue tail, so periodicity from its "
            "first entry says nothing about how deep the memory goes",
        )
        self.assertLess(
            lead_in,
            window.tail_steps,
            "the lead-in is long enough for the CLIP readout to have settled too, so "
            "this fixture cannot separate the two memories",
        )
        held = rows[start : start + length]
        for index in range(length - period):
            self.assertEqual(
                held[index].measured_rms, held[index + period].measured_rms
            )
        self.assertAlmostEqual(held[0].displacement_m, 0.0)

    def test_the_symptom_is_still_visible_here_because_the_fake_cannot_route(self):
        """The guard against a regression to blind stepping, and the fake's stated limit.

        The straight-line follower walks into the wall on the way to a probe behind it, so
        the collision rate stays high here even with the detour routed through the pool.
        That is a property of ``FakeWorld.follower``, not of the system —
        ``tests/box/test_investigate_route_box.py`` is what settles routing. What this
        pins is that the record still shows the wall, so a silent return to stepping
        ``move_forward`` blindly does not pass unnoticed.
        """
        summary = self.result.audit.forward_summary()
        self.assertGreater(summary["n_collided"] / summary["n_forward"], 0.5)


class TestTheStallTurnsTowardTheSource(unittest.TestCase):
    """ADR-0011's third rule, and the frame convention it rests on.

    The agent starts facing **away** from a source that is behind and to its left, so the
    climb's first move takes it further away, the loudness stops rising, and the only
    thing that can recover the episode is the lateral cue. The assertion is not that it
    turned — it is that the sign it read matches the ground-truth bearing and that the
    turn went that way, which is the pair a compensation term would break.

    ``t_anom = 0`` here, so the onset lands on step 0 and the pose at that step is the
    start pose the prediction is computed from. The per-step record deliberately carries
    no pose (§3.2 lists five fields and this is not one), so pinning the cue against
    ground truth needs a step whose pose is known independently. §3.1's pre-onset
    invariant is exercised by ``TestTheFullLoop`` instead, where there are pre-onset
    steps to exercise it on.
    """

    @classmethod
    def setUpClass(cls):
        cls.world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=math.pi)
        cls.source = Xyz(2.0, 0.0, -5.0)
        cls.handle = FakeAudioSensorHandle(cls.world, cls.source)
        # The episode owns the start pose — `run_episode` seats the agent from it — so
        # the yaw has to be here rather than on the world.
        cls.anomaly_episode = make_anomaly_episode(
            source=cls.source, t_anom=0, episode=make_episode(start_yaw=math.pi)
        )
        cls.cfg = make_config(max_steps=120, t_anom=0)
        cls.result = run(
            cls.world, cls.handle, cls.anomaly_episode, cls.cfg, calibration=CALIBRATION
        )

    def test_the_cue_the_agent_read_agrees_with_the_ground_truth_bearing(self):
        """``lateral_sign`` (measured) versus ``bearing_lateral_sign`` (privileged).

        The second is analyst-only for exactly this reason: it states a prediction the
        renderer either matches or refutes. Here the "renderer" is the fake, so what this
        pins is that the loop reads the cue in the frame ``agent/occupancy`` defines —
        the agreement ``test_agent_frame.py`` asserts across the layer boundary.
        """
        first = self.result.audit.steps[self.result.audit.onset.onset_step]
        pose = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=math.pi)
        self.assertEqual(first.lateral_sign, bearing_lateral_sign(pose, self.source))
        self.assertNotEqual(first.lateral_sign, LATERAL_AMBIGUOUS)

    def test_the_first_stall_turns_toward_the_louder_half_plane(self):
        onset_step = self.result.audit.onset.onset_step
        turns = [
            row
            for row in self.result.audit.steps[onset_step:]
            if row.action in ("turn_left", "turn_right")
        ]
        self.assertTrue(turns, "the climb never stalled, so the cue was never used")
        wanted = "turn_right" if turns[0].lateral_sign > 0 else "turn_left"
        self.assertEqual(turns[0].action, wanted)

    def test_it_still_reaches_the_source(self):
        """The whole point of the cue: a climb that starts backwards recovers.

        **This assertion means more than it used to, and it is now §9's budget guard.**
        Ticket 26 made the abort terminal for the interrupt, so an episode that aborted
        can never later reach — which makes ``SOURCE_REACHED`` proof that the whole climb
        fitted inside one ``investigate_max_steps``. It did not before: this climb needs
        **59 steps**, the budget was 40, and it reached the source on a *re-entry* whose
        second attempt started from a pose the first had already improved. The test
        passed on the strength of the bug it was meant to be independent of.

        So lowering the sub-budget below what a real climb costs now fails here, with
        this docstring as the reason.
        """
        self.assertGreaterEqual(
            self.result.audit.funnel_stage, FunnelStage.SOURCE_REACHED
        )


class TestTheOracleArm(unittest.TestCase):
    """The bisection tool (§8): the same loop, steered by a point goal.

    Retained precisely so that if the smoke fails, running this separates audio from
    controller in one step — so it has to work, and the thing that makes it different has
    to be visible in the record.
    """

    @classmethod
    def setUpClass(cls):
        cls.world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
        cls.source = Xyz(1.5, 0.0, -5.0)
        cls.handle = FakeAudioSensorHandle(cls.world, cls.source)
        cls.anomaly_episode = make_anomaly_episode(source=cls.source, t_anom=2)
        cls.cfg = make_config(localization=Localization.ORACLE, max_steps=60)
        cls.result = run(
            cls.world, cls.handle, cls.anomaly_episode, cls.cfg, calibration=CALIBRATION
        )

    def test_it_reaches_the_source_and_says_which_arm_ran(self):
        self.assertGreaterEqual(
            self.result.audit.funnel_stage, FunnelStage.SOURCE_REACHED
        )
        self.assertEqual(self.result.audit.localization_arm, "oracle")

    def test_the_testimony_schema_is_identical_to_the_realizable_arm(self):
        """§5.1's requirement, and the reason the arm is invisible in the testimony.

        If the oracle arm's report carried one extra field, "is this arm realizable"
        would be answerable by reading the schema — which is exactly what the identical
        schema is for on the *other* side.
        """
        self.assertEqual(sorted(self.result.report.as_dict()), sorted(SCHEMA_FIELDS))

    def test_it_stopped_inside_the_oracle_arrival_radius(self):
        """The one arrival criterion that survives in this arm only (§4.2)."""
        self.assertLessEqual(
            self.result.audit.dist_at_stop,
            ControllerConfig().investigate_arrive_radius_m,
        )


class TestTheDivertCandidate(unittest.TestCase):
    def test_it_is_injected_into_the_pool_rather_than_steered_to_directly(self):
        """``agent/scorer.py``'s seam: an override by rank, and navmesh-filtered.

        Going around the pool would skip ``reachability``'s filter, and a source the
        agent cannot route to would become a silent straight-line walk into a wall.
        """
        pose = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)
        candidate = _divert_candidate(Xyz(3.0, 0.0, -4.0), pose)
        self.assertEqual(candidate.source, SOURCE_INVESTIGATE)
        self.assertEqual(candidate.candidate_id, DIVERT_CANDIDATE_ID)
        self.assertAlmostEqual(candidate.distance_m, 5.0)

    def test_id_zero_is_never_issued_by_the_proposer(self):
        """Which is what makes the divert identifiable in the audit by id as well as source."""
        from earshot.agent.proposers import FrontierProposer

        pose = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)
        proposer = FrontierProposer()
        proposer.reset(pose)
        ids = [candidate.candidate_id for candidate in proposer.compass_fan(pose)]
        self.assertNotIn(DIVERT_CANDIDATE_ID, ids)


class TestTheEmptyPoolIsFatal(unittest.TestCase):
    """ADR-0008's invariant, at the seam where the runner could have swallowed it.

    An agent that silently stops choosing looks in a trajectory exactly like an agent
    that chose to stand still, which is the failure the old tree's straight-line fallback
    produced. The runner must let it raise.
    """

    def test_a_navmesh_that_rejects_everything_raises(self):
        world = FakeWorld(blocked=True)
        handle = FakeAudioSensorHandle(world, Xyz(0.0, 0.0, -5.0))
        with self.assertRaises(EmptyPoolError):
            run(
                world,
                handle,
                make_anomaly_episode(t_anom=2),
                make_config(max_steps=5),
                calibration=CALIBRATION,
            )


class TestTheDetectorArm(unittest.TestCase):
    def test_the_oracle_answers_about_the_primary_goal_and_the_source(self):
        world = FakeWorld(start=Xyz(0.0, 0.0, -4.5), yaw=0.0)
        anomaly_episode = make_anomaly_episode(
            source=Xyz(0.0, 0.0, -5.0),
            episode=make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))]),
        )
        detector = make_detector(make_config(), world, anomaly_episode)
        self.assertTrue(detector.detects("sofa"))  # the source, 0.5 m away
        self.assertFalse(detector.detects("chair"))  # the primary goal, 4.5 m away
        self.assertFalse(detector.detects("toilet"))  # not in this episode at all

    def test_the_caption_arm_raises_and_names_what_is_missing(self):
        """An arm that silently ran the other arm would mislabel every audit record."""
        with self.assertRaises(RuntimeError) as caught:
            make_detector(make_config(detector=Detector.CAPTION), FakeWorld(), make_anomaly_episode())
        self.assertIn("vlm.py", str(caught.exception))


class TestCalibrationPoses(unittest.TestCase):
    """§2.3's sweep poses — the half ``audio/calibration.band_poses`` leaves to this layer."""

    def test_the_poses_spread_across_the_band_by_geodesic_distance(self):
        world = FakeWorld()
        source = Xyz(0.0, 0.0, 0.0)
        poses = calibration_poses(world, source, (1.0, 8.0), 6, n_draws=64)
        self.assertEqual(len(poses), 6)
        distances = sorted(world.geodesic_distance(p, [source]) for p in poses)
        self.assertLess(distances[0], 2.0)
        self.assertGreater(distances[-1], 6.0)

    def test_no_pose_is_used_twice(self):
        world = FakeWorld()
        poses = calibration_poses(world, Xyz(0.0, 0.0, 0.0), (1.0, 8.0), 6, n_draws=64)
        self.assertEqual(len({p.as_tuple() for p in poses}), len(poses))

    def test_an_unroutable_source_fails_the_gate_rather_than_returning_nothing(self):
        """A silent empty sweep would reach ``calibrate_onset`` as "no distribution"."""
        from earshot.audio.calibration import CalibrationError

        class Islanded(FakeWorld):
            def geodesic_distance(self, start, ends):
                return None

        with self.assertRaises(CalibrationError):
            calibration_poses(Islanded(), Xyz(0.0, 0.0, 0.0), (1.0, 8.0), 4, n_draws=8)


class TestTheCalibrationProfile(unittest.TestCase):
    """The sweep's distance axis, kept rather than summarised away.

    `calibrate_onset` reduces sixteen rendered poses to four percentiles, which say how
    loud the anomaly is and cannot say whether it gets louder as you approach. That curve
    is the premise of an energy-gradient climb and it was being computed and discarded
    every episode.
    """

    def _calibrate(self, world):
        source = Xyz(3.0, 0.0, 0.0)
        handle = FakeAudioSensorHandle(world, source)
        result, poses = calibrate_episode(
            world, handle, source, CLIP, make_config())
        return result, poses

    def test_the_profile_carries_one_pair_per_pose_and_falls_with_distance(self):
        world = FakeWorld()
        result, poses = self._calibrate(world)
        self.assertEqual(len(result.profile), len(poses))
        by_distance = sorted(result.profile)
        self.assertGreater(
            by_distance[0][1], by_distance[-1][1],
            "the nearest pose must be the loudest, or the axis is not the axis")

    def test_a_pose_with_no_route_is_dropped_rather_than_recorded_at_zero(self):
        """The forced-failure arm (ADR-0014).

        A distance that could not be measured entering the profile as ``0.0`` would put a
        phantom sample at the source, which is where the gradient is steepest — it would
        manufacture the very cue the profile exists to test for.
        """
        class LosesOnePose(FakeWorld):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def geodesic_distance(self, start, ends):
                self.calls += 1
                # The pose draw comes first and takes CALIBRATION_DRAWS calls; the next
                # call is the profile's first pose.
                if self.calls == CALIBRATION_DRAWS + 1:
                    return None
                return super().geodesic_distance(start, ends)

        result, poses = self._calibrate(LosesOnePose())
        self.assertEqual(len(result.profile), len(poses) - 1)
        self.assertTrue(all(distance > 0.0 for distance, _rms in result.profile))

    def test_the_profile_survives_the_audit_round_trip(self):
        record = CalibrationRecord(
            onset_rms=0.01, bed_rms=0.001, separation_db=40.0, n_poses=2,
            global_volume=1.0, profile=((1.0, 0.5), (8.0, 0.1)))
        self.assertEqual(
            CalibrationRecord.from_dict(record.as_dict()).profile,
            ((1.0, 0.5), (8.0, 0.1)))

    def test_a_record_written_before_the_profile_existed_reads_as_absent(self):
        """Empty, not a flat field: `()` means nobody measured, and nothing may infer
        from it that the level did not change with distance."""
        payload = CalibrationRecord(
            onset_rms=0.01, bed_rms=0.001, separation_db=40.0, n_poses=2,
            global_volume=1.0).as_dict()
        del payload["profile"]
        self.assertEqual(CalibrationRecord.from_dict(payload).profile, ())


class TestTheFunnelLadder(unittest.TestCase):
    """§6's stages nest, so the stage is the highest one reached, not a classification."""

    def test_the_ladder_climbs(self):
        base = dict(
            n_steps=100,
            t_anom=30,
            onset_fired=False,
            entered_investigate=False,
            investigated=False,
            resumed=False,
        )
        self.assertEqual(_funnel_stage(**dict(base, n_steps=10)), FunnelStage.RUN)
        self.assertEqual(_funnel_stage(**base), FunnelStage.T_ANOM_REACHED)
        self.assertEqual(
            _funnel_stage(**dict(base, onset_fired=True)), FunnelStage.ONSET_FIRED
        )
        self.assertEqual(
            _funnel_stage(**dict(base, onset_fired=True, entered_investigate=True)),
            FunnelStage.INVESTIGATE_ENTERED,
        )
        self.assertEqual(
            _funnel_stage(
                **dict(base, onset_fired=True, entered_investigate=True, investigated=True)
            ),
            FunnelStage.SOURCE_REACHED,
        )
        self.assertEqual(
            _funnel_stage(
                **dict(
                    base,
                    onset_fired=True,
                    entered_investigate=True,
                    investigated=True,
                    resumed=True,
                )
            ),
            FunnelStage.PRIMARY_RESUMED,
        )

    def test_an_aborted_detour_that_resumed_did_not_reach_the_source(self):
        """The ladder's premise is falsified by the abort path, and the first box run hit it.

        ``an episode that resumed necessarily investigated`` is false: the abort
        transitions straight to RESUME with ``investigated`` False, so a monotone ladder
        promoted a stage-4 episode to 6 and the run printed a 6/6 funnel while its own
        trace showed six INVESTIGATE entries and five aborts. Criterion 5 is read off
        this number, so the over-credit is not cosmetic — it is the smoke asserting the
        loop ran on an episode where CHECK was never reached.
        """
        aborted = dict(
            n_steps=100,
            t_anom=30,
            onset_fired=True,
            entered_investigate=True,
            investigated=False,
            resumed=True,
        )
        self.assertEqual(_funnel_stage(**aborted), FunnelStage.INVESTIGATE_ENTERED)

    def test_resuming_without_ever_entering_cannot_climb_either(self):
        """Same rule, one rung lower: the stages nest or they mean nothing."""
        self.assertEqual(
            _funnel_stage(
                n_steps=100,
                t_anom=30,
                onset_fired=True,
                entered_investigate=False,
                investigated=False,
                resumed=True,
            ),
            FunnelStage.ONSET_FIRED,
        )

    def test_an_episode_that_ended_before_t_anom_did_not_reach_stage_two(self):
        """The denominator for the loop is stage 2, so its boundary is load-bearing.

        Step indices are zero-based, so an episode of exactly ``t_anom`` steps ended on
        the step *before* the source started playing.
        """
        base = dict(
            t_anom=30,
            onset_fired=False,
            entered_investigate=False,
            investigated=False,
            resumed=False,
        )
        self.assertEqual(_funnel_stage(n_steps=30, **base), FunnelStage.RUN)
        self.assertEqual(_funnel_stage(n_steps=31, **base), FunnelStage.T_ANOM_REACHED)


class TestTheRunnerAndTheRealCollaboratorsAgreeOnNames(unittest.TestCase):
    """The one thing a fake can hide, closed the way ticket 23 closed its FOV pin.

    ``run_episode`` is duck-typed, so a rename in ``sim/world.py`` — or a fake that
    drifted from it — would leave this whole file green and fail on the box as an
    ``AttributeError`` forty minutes into a trip. No Mac can import ``sim/world.py`` at
    all (it imports habitat-sim), so the real subject is read out of its ``ast``, which
    is exactly the move ``test_agent_frame.py`` makes for ``PlannerConfig``'s FOV.

    ``run()`` itself is the part of this module that has never executed anywhere. This is
    the cheapest assertion that covers its riskiest failure mode.
    """

    # Every attribute `task/runner.py` reaches for on a `World`. Hand-transcribed on
    # purpose: derived from the runner's own ast it would only assert that the runner
    # equals itself, which is the vacuous shape an adversarial pass on ticket 23 kept
    # finding.
    WORLD_API = (
        "observe",
        "step",
        "pose",
        "set_pose",
        "snap_point",
        "geodesic_distance",
        "random_navigable_point",
        "seed_navmesh",
        "follower",
        "sensor_handle",
        "close",
        "n_renders",
    )

    HANDLE_API = ("observe", "audio_of", "set_source", "source_is_visible", "report")

    @staticmethod
    def _class_members(path, class_name):
        """Method names plus ``self.x = ...`` attributes declared in a class body."""
        import ast

        import _tree

        for node in ast.walk(_tree.parse(path)):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                names = set()
                for child in ast.walk(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        names.add(child.name)
                    elif isinstance(child, ast.Attribute) and isinstance(
                        child.value, ast.Name
                    ):
                        if child.value.id == "self" and isinstance(child.ctx, ast.Store):
                            names.add(child.attr)
                return names
        raise AssertionError("no class {!r} in {}".format(class_name, path))

    def test_the_real_world_publishes_everything_the_runner_calls(self):
        import _tree

        members = self._class_members(_tree.PACKAGE_ROOT / "sim" / "world.py", "World")
        missing = sorted(name for name in self.WORLD_API if name not in members)
        self.assertEqual(
            missing,
            [],
            "task/runner.py calls {} on the world and sim/world.World does not define "
            "them — an AttributeError that only the box would find".format(missing),
        )

    def test_the_fake_world_publishes_the_same_set(self):
        """Otherwise the fake and the real object drift apart in the other direction."""
        # `n_renders` is an instance attribute on both, so ask an instance.
        instance = FakeWorld()
        missing = sorted(name for name in self.WORLD_API if not hasattr(instance, name))
        self.assertEqual(missing, [])

    def test_the_real_handle_publishes_everything_the_runner_calls(self):
        """``audio/sensor.py`` imports no habitat-sim, so this one can be asked directly."""
        from earshot.audio.sensor import AudioSensorHandle

        missing = [
            name
            for name in self.HANDLE_API
            if not hasattr(AudioSensorHandle, name)
            and name not in AudioSensorHandle.__init__.__code__.co_names
        ]
        self.assertEqual(missing, [])

    def test_the_fake_handle_publishes_the_same_set(self):
        handle = FakeAudioSensorHandle(FakeWorld(), Xyz(0.0, 0.0, 0.0))
        missing = sorted(name for name in self.HANDLE_API if not hasattr(handle, name))
        self.assertEqual(missing, [])


class TestBedAndClipAgree(unittest.TestCase):
    """``mix_bed`` refuses a length mismatch, so the runner must build the bed at the clip.

    A tolerant mix would silently change the RMS the onset threshold was calibrated
    against — the two tempting fixes (tile the bed, crop the render) each do exactly
    that.
    """

    def test_the_bed_is_generated_at_the_clips_length(self):
        bed = bed_signal(len(CLIP), 1e-3)
        self.assertEqual(bed.shape, (2, len(CLIP)))


# The speed of sound this file's delayed-IR fake uses, so a propagation delay in SAMPLES
# is derivable from a distance in metres. It is a fixture constant and nothing in
# `earshot/` reads it -- `sim/world.py` gets its own from the renderer's configuration.
SPEED_OF_SOUND_M_S = 343.0


class DelayedIrHandle(FakeAudioSensorHandle):
    """The fake with its direct path where the physics puts it: after the flight time.

    **``_task_fakes``' IR is a delta at index 0 for both ears**, which is a source whose
    sound arrives the instant it is emitted. That was harmless while the readout was the
    5 s clip window, because the whole flight time fits inside one analysis window either
    way. It stopped being harmless at ADR-0019: the cue readout is one step wide, so
    WHICH step a chunk lands in is now the thing being measured, and a delay of zero makes
    every arrival land in the fold that emitted it.

    Here the impulse sits at ``round(distance / 343 * sample_rate)``. At this file's hop
    of 441 samples (0.01 s) a source 5 m away is 643 samples out -- 1.46 hops -- so a
    chunk emitted on the last sounding fold ARRIVES ONE TO TWO STEPS AFTER THE WINDOW HAS
    CLOSED. That is what makes ``onset_step > offset_step`` reachable on a Mac again, and
    it is the same shape the box gets from a real RIR, whose first nonzero sample is the
    direct path's own delay.

    It is deliberately still a single tap: the room's reverberation is a separate
    capability and ``tests/box/test_sounding_window_box.py`` is what exercises it.
    """

    IR_LENGTH = 1024

    def audio_of(self, observation):
        direct = super().audio_of(observation)
        distance = self.world.pose().position.horizontal_distance_to(self.source)
        delay = min(
            int(round(distance / SPEED_OF_SOUND_M_S * 44100)), self.IR_LENGTH - 1
        )
        impulse = np.zeros((2, self.IR_LENGTH), dtype=np.float32)
        impulse[0, delay] = direct[0, 0]
        impulse[1, delay] = direct[1, 0]
        return impulse


class RingingHandle(FakeAudioSensorHandle):
    """The fake with a room behind it: a decaying multi-tap IR instead of one delta.

    ``_task_fakes``' impulse is a single tap, so its ``cue_tail_steps`` is arithmetic off
    a PADDED width and its cue readout is exactly zero on the first silent step -- an
    honest hard cut, and the reason ``post_offset_audible_steps`` is 0 for every fixture
    in this file that uses it. That is the correct reading of an anechoic IR and it is
    also why a second handle has to exist: with only the delta fake, "the cue tail carries
    energy" is a claim no Mac test could make.

    The tail is ``exp(-k/300)`` over 900 samples, sign-randomised off a fixed seed so it
    is a decaying reverberation rather than a comb, at 35% of the direct path's amplitude.
    Two arms, and this is the healthy one; the delta fake is the forced failure.
    """

    IR_LENGTH = 900
    RT_SAMPLES = 300.0
    TAIL_GAIN = 0.35
    # Fixed, because a fixture whose IR changes shape between runs cannot pin a decay
    # curve. It is a seed and not a magic constant: any seed gives the same measurement.
    SEED = 20260831

    def audio_of(self, observation):
        direct = super().audio_of(observation)
        k = np.arange(self.IR_LENGTH, dtype=np.float32)
        envelope = np.exp(-k / self.RT_SAMPLES).astype(np.float32)
        signs = np.random.default_rng(self.SEED).normal(
            0.0, 1.0, size=(2, self.IR_LENGTH)
        ).astype(np.float32)
        impulse = np.zeros((2, self.IR_LENGTH), dtype=np.float32)
        for ear in range(2):
            impulse[ear] = direct[ear, 0] * envelope * signs[ear] * self.TAIL_GAIN
            impulse[ear, 0] = direct[ear, 0]
        return impulse


def _tail_onset_episode(**overrides):
    """The fixture where the source's arrival OUTLIVES its own window.

    A source 5 m away sounding for ONE step from ``t_anom = 2``, through
    ``DelayedIrHandle``. The single sounding fold emits its chunk and reads the bed,
    because the sound is still in flight; the arrival lands on the offset step and the one
    after it. So the loudest steps of the episode are SILENT ones and the first crossing
    is exactly ON the offset step.

    **This fixture replaces one that rested on the fill ramp**, and had to. The old shape
    was "the accumulator's read window is still FILLING when the window closes, so the
    reading keeps climbing for one more step" -- which is the analysis-window artefact
    ADR-0019 removed. The property being pinned is unchanged (``onset_step ==
    offset_step``, so ``heard_within_window`` is 0.0 and ``<`` is separable from ``<=``);
    what changed is that its cause is now the propagation delay, which is physical.
    """
    world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0, wall=lambda p: p.z < -1.0)
    source = Xyz(0.0, 0.0, -5.0)
    handle = DelayedIrHandle(world, source)
    anomaly_episode = make_anomaly_episode(
        source=source,
        t_anom=2,
        episode=make_episode(start_yaw=0.0, goals=[make_goal(Xyz(0.0, 0.0, -40.0))]),
    )
    cfg = make_config(max_steps=20, t_anom=2, sounding_steps=1)
    calibration = dataclasses.replace(
        CALIBRATION, onset_rms=TAIL_ONSET_RMS
    )
    return run(world, handle, anomaly_episode, cfg, calibration=calibration,
               **overrides), cfg


# `onset_rms` for `_tail_onset_episode`, placed on MEASURED levels: the single sounding
# step reads the bed (0.001) because the sound has not arrived, and the offset step reads
# 0.004223. Anywhere between the two puts the first crossing exactly on the offset step,
# which is the only fixture shape that separates `<` from `<=` in `heard_within_window`.
TAIL_ONSET_RMS = 0.003


def _windowed_episode(index=0, **cfg_overrides):
    """One episode that outlives its own window: a wall, a near source, a far goal.

    The wall stops the agent so the episode spends its whole budget rather than ending
    at the primary STOP, which is the only way to get a silent phase long enough to say
    anything about. The goal is 40 m away for the same reason.

    ``index`` is the EPISODE index and not a config field. It reaches ``plan_window``
    and, under ``WindowPolicy.DRAWN``, is half of what the duration is drawn from -- so a
    helper that could not vary it left the whole draw path unreachable from this file.
    """
    world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0, wall=lambda p: p.z < -1.0)
    source = Xyz(0.0, 0.0, -5.0)
    handle = FakeAudioSensorHandle(world, source)
    anomaly_episode = make_anomaly_episode(
        source=source,
        t_anom=2,
        episode=make_episode(start_yaw=0.0, goals=[make_goal(Xyz(0.0, 0.0, -40.0))]),
    )
    calibration = cfg_overrides.pop("calibration_override", CALIBRATION)
    cfg = make_config(max_steps=40, t_anom=2, **cfg_overrides)
    return (
        run(world, handle, anomaly_episode, cfg, index=index, calibration=calibration),
        cfg,
    )


class TestTheSoundingWindow(unittest.TestCase):
    """ADR-0017: the source sounds for a bounded window, goes silent, and is still there.

    Before this, ``playing = step >= t_anom`` was the whole mechanism and nothing closed
    it. Everything below is about the two facts that replaced it — the offset step, and
    the reverb tail that makes the silence physical instead of a hard cut to the bed.

    **Since ADR-0019 this class's own fixture is the ANECHOIC arm and it says so.** Its
    IR is ``_task_fakes``' single delta, so its cue readout cuts to the bed on the offset
    step and its ``post_offset_audible_steps`` is 0. ``RingingHandle`` supplies the other
    arm. Under the old clip readout the two were indistinguishable -- both showed a
    five-step decay -- and that indistinguishability is the defect.
    """

    @classmethod
    def setUpClass(cls):
        cls.result, cls.cfg = _windowed_episode(sounding_steps=6)
        cls.window = cls.result.audit.sounding_window

    def test_the_window_closes_and_the_record_says_when(self):
        """The offset step exists NOWHERE ELSE on disk.

        ``source_playing`` shows when the source stopped; it cannot show what the task
        ASKED for, and a source that failed to stop leaves a trace that agrees with
        itself. That is the argument ``t_anom`` already won on this record.
        """
        self.assertEqual(self.window.opens_at, self.cfg.t_anom)
        self.assertEqual(self.window.offset_step, self.cfg.t_anom + self.cfg.sounding_steps)
        self.assertEqual(self.window.policy, "fixed_steps")
        sounding = [row.step for row in self.result.audit.steps if row.source_playing]
        self.assertEqual(
            sounding, list(range(self.window.opens_at, self.window.offset_step))
        )

    def test_the_silent_phase_decays_when_there_is_a_ROOM_and_cuts_when_there_is_not(self):
        """BOTH ARMS OF THE TAIL, and ADR-0019 is why they are now separable at all.

        Under the clip readout every fixture in this file showed a plausible five-step
        decay after the offset step, INCLUDING this one -- whose IR is a single delta at
        index 0, i.e. a room with no reverberation whatever. That decay was the 5 s
        analysis window emptying, and an anechoic control reproduced it to ~1 point. The
        cue readout is one step wide, so what it shows after the offset step is the room.

        HEALTHY ARM -- ``RingingHandle``, a decaying 900-sample IR. Measured, as a
        fraction of the last sounding step's level:

            0.3455  0.0715  0.0132  (the bed)   against a ``cue_tail_steps`` of 4

        FORCED FAILURE -- the delta fake this class's own fixture uses. Its cue readout is
        EXACTLY the bed on the offset step: an honest hard cut, correctly reported.

        **``onset_step > offset_step`` stays reachable** -- ``TestAnOnsetThatFiredOnThe
        ReverbTail`` produces it off the flight time -- so no post-offset invariant may
        assert the bed at the offset step. What changed is that an episode reading the bed
        there is now saying something true about its room.
        """
        rows = {row.step: row for row in self.result.audit.steps}
        offset = self.window.offset_step
        bed = self.cfg.audio.bed_rms
        self.assertAlmostEqual(
            rows[offset].measured_rms,
            bed,
            places=6,
            msg="the delta fake's cue readout is loud after the offset step, so it is "
                "still carrying the analysis window rather than the room",
        )

        ringing, ring_cfg = _episode_with(RingingHandle)
        window = ringing.audit.sounding_window
        ring_rows = {row.step: row for row in ringing.audit.steps}
        ring_bed = ring_cfg.audio.bed_rms
        ring_offset = window.offset_step
        last_sounding = ring_rows[ring_offset - 1].measured_rms
        curve = [
            ring_rows[ring_offset + i].measured_rms / last_sounding
            for i in range(window.cue_tail_steps)
        ]
        self.assertGreater(
            curve[0], 0.20, "the room contributed nothing on the first silent step"
        )
        self.assertLess(curve[0], 1.0, "the offset step is no quieter than the source")
        for before, after in zip(curve, curve[1:]):
            self.assertLess(after, before, curve)
        self.assertAlmostEqual(
            ring_rows[ring_offset + window.cue_tail_steps - 1].measured_rms,
            ring_bed,
            places=5,
        )
        # ...and the two arms differ where the OLD readout could not tell them apart.
        self.assertGreater(
            window.post_offset_audible_steps,
            self.window.post_offset_audible_steps,
            "the ringing room and the anechoic delta produce the same audible-step "
            "count, so this file cannot tell a tail from a hard cut",
        )

    def test_the_render_still_happens_on_a_silent_step(self):
        """Smoke criterion 1 is ``n_renders_in_loop == n_loop_steps``, silence included.

        A "do not render while silent" optimisation would break the equality on an
        episode that is mostly silent, and ``audio_render_summary`` would then measure a
        workload that is not the one being claimed.
        """
        audit = self.result.audit
        silent = [row for row in audit.steps if row.step >= self.window.offset_step]
        self.assertGreater(len(silent), 20, "there is barely a silent phase to test")
        self.assertEqual(audit.metrics["n_renders_in_loop"], audit.metrics["n_loop_steps"])
        self.assertEqual(audit.n_render_steps, int(audit.metrics["n_loop_steps"]))
        self.assertTrue(all(row.audio_render_s is not None for row in silent))

    def test_the_evidence_the_open_duration_question_needs_is_recorded(self):
        """ADR-0017 leaves the duration policy open and this build does not close it.

        What it does is record the three numbers the choice has to be made against, so
        one sweep at the provisional default answers it: how long the agent took to hear
        the source, whether it heard the source or only its tail, and the shape of the
        two phases.
        """
        metrics = self.result.audit.metrics
        self.assertEqual(metrics["sounding_window_closed"], 1.0)
        self.assertEqual(metrics["offset_step"], float(self.window.offset_step))
        self.assertEqual(metrics["sounding_duration_steps"], float(self.cfg.sounding_steps))
        self.assertEqual(
            metrics["silent_phase_steps"],
            float(len(self.result.audit.steps) - self.window.offset_step),
        )
        # THE VALUE, not the key. Measured FROM THE WINDOW OPENING -- `onset_step -
        # t_anom` -- and on any episode with `t_anom > 0` a delay written as the bare
        # onset step is inflated by exactly `t_anom`, which biases the distribution the
        # window duration is chosen against and is invisible in a key-presence check.
        onset_step = self.result.audit.onset.onset_step
        self.assertGreater(self.cfg.t_anom, 0, "a t_anom of 0 cannot separate the two")
        self.assertEqual(
            metrics["onset_delay_steps"], float(onset_step - self.cfg.t_anom)
        )
        self.assertNotEqual(metrics["onset_delay_steps"], float(onset_step))
        # ...and the ramp is beside it. Since ADR-0019 it is the CLIP ramp and its
        # consumer has moved: it bounds the CLAP deferral rather than correcting the
        # delay, because the cue window is written whole by one fold.
        self.assertEqual(metrics["sounding_ramp_steps"], float(self.window.ramp_steps))
        # THE TWO BOUNDS THAT REPLACED THE RAMP AS THE DELAY'S CORRECTION, both on the
        # metrics bag because neither is re-derivable from it. The room's own build-up is
        # at most `sounding_cue_tail_steps - 1` steps and the loop's intermittency at most
        # `sounding_phase_folds - 1`; a reader with only the ramp would subtract the wrong
        # number, which is exactly the failure the ramp was recorded to prevent.
        self.assertEqual(
            metrics["sounding_cue_tail_steps"], float(self.window.cue_tail_steps)
        )
        self.assertEqual(metrics["sounding_phase_folds"], 5.0)
        self.assertNotEqual(
            metrics["sounding_cue_tail_steps"],
            float(self.window.tail_steps),
            "the two tails are the same number on this fixture, so nothing here could "
            "tell a criterion reading the wrong one",
        )
        self.assertEqual(metrics["heard_within_window"], 1.0)

    def test_the_audible_tail_is_MEASURED_off_the_trace_rather_than_derived(self):
        """``cue_tail_steps`` and ``post_offset_audible_steps`` answer different questions
        and only the second one is evidence.

        ``cue_tail_steps`` is arithmetic off the IR's width -- how long the room COULD
        outlive the offset step -- and ``tail_is_active`` returns True for a buffer that
        was correctly built, correctly handed a real IR, and carried no energy past the
        offset step at all. A ZERO here means the silence arrived as a hard cut: an SWS
        over such episodes is a number about the mechanism ADR-0017 replaced.

        **The delta fake measures ZERO since ADR-0019, and that is correct.** Its IR is a
        single tap at index 0 padded to 64 samples, so the room really does nothing and
        the cue readout says so. Under the clip readout the same fixture reported FOUR
        audible steps -- 0.01341 0.01126 0.00830 0.00356 -- which was the 5 s analysis
        window emptying and not the room, and is the defect this change corrects. The
        arithmetic tail stayed at 6 either way, which is exactly why the measured half has
        to exist.

        Both arms: the anechoic fake at zero, ``RingingHandle`` at two, off the same
        recount. Recomputed here from the per-step trace and the bed on the record, which
        is a different route to the number than the runner's own local.
        """
        def recount(result, cfg):
            bed = cfg.audio.bed_rms
            margin = bed * abs(cfg.audio.pre_onset_rms_tol)
            return sum(
                1
                for row in result.audit.steps
                if row.step >= result.audit.sounding_window.offset_step
                and abs(row.measured_rms - bed) > margin
            )

        recounted = recount(self.result, self.cfg)
        self.assertEqual(self.window.post_offset_audible_steps, recounted)
        self.assertEqual(
            recounted,
            0,
            "the anechoic delta fake reports an audible tail, so the reading still "
            "carries the analysis window rather than the room",
        )
        self.assertEqual(
            self.result.audit.metrics["post_offset_audible_steps"], float(recounted)
        )

        ringing, ring_cfg = _episode_with(RingingHandle)
        ring_counted = recount(ringing, ring_cfg)
        self.assertEqual(
            ringing.audit.sounding_window.post_offset_audible_steps, ring_counted
        )
        self.assertEqual(ring_counted, 2)
        self.assertLess(
            ring_counted,
            ringing.audit.sounding_window.cue_tail_steps,
            "the measured half must not simply restate the arithmetic one",
        )

    def test_the_record_round_trips_so_the_window_is_readable_a_year_later(self):
        from earshot.report.audit import EpisodeAudit

        restored = EpisodeAudit.from_dict(self.result.audit.as_dict())
        self.assertEqual(restored.sounding_window, self.window)
        self.assertEqual(restored.source_reached_step, self.result.audit.source_reached_step)


class TestTheContinuousControlArm(unittest.TestCase):
    """``WindowPolicy.CONTINUOUS`` reproduces the pre-ADR-0017 source, THROUGH the tail.

    **This is the control arm every windowed delta will be measured against**, and it is
    not a fallback. A windowed run crosses two changes at once — the offset step and the
    accumulating renderer — so a funnel delta against ``arrive-2`` or ``yield-2`` cannot
    say which one moved it. This repo's own rule: a claim that X broke because of a
    change needs the arm where the change is absent, and the hermeticity gate already
    paid once for ignoring it.
    """

    @classmethod
    def setUpClass(cls):
        cls.result, cls.cfg = _windowed_episode(sounding_policy=WindowPolicy.CONTINUOUS)

    def test_the_source_never_stops(self):
        window = self.result.audit.sounding_window
        self.assertIsNone(window.offset_step)
        self.assertEqual(window.policy, "continuous")
        for row in self.result.audit.steps:
            self.assertEqual(
                row.source_playing,
                row.step >= self.cfg.t_anom,
                "step {} disagrees with `step >= t_anom`".format(row.step),
            )

    def test_it_still_runs_through_the_accumulator_so_only_the_offset_step_differs(self):
        """The renderer is the SAME in both arms, which is what isolates the window.

        A control arm that also reverted the renderer would confound exactly the
        comparison it exists to make.
        """
        window = self.result.audit.sounding_window
        self.assertGreater(window.max_ir_samples, 0)
        self.assertEqual(window.hop_samples, 441)
        self.assertEqual(window.analysis_window_samples, len(CLIP))

    def test_no_sws_is_claimed_for_an_arm_with_no_offset_step(self):
        """No offset step, no silent phase, no eligibility — and an ABSENT key, not 0.0."""
        metrics = self.result.audit.metrics
        self.assertEqual(metrics["sounding_window_closed"], 0.0)
        self.assertNotIn("sws", metrics)
        self.assertNotIn("sws_eligible", metrics)
        self.assertNotIn("offset_step", metrics)


class TestAWindowThatClosesBeforeTheAgentIsInEarshot(unittest.TestCase):
    """THE FORCED-FAILURE ARM, and the mechanism's most dangerous property.

    A one-step window against a source 16 m away: the agent never gets close enough to
    cross ``onset_rms`` before the source stops, and after the tail the level is the bed,
    which ``calibrate_onset`` separates from the threshold by >= 6 dB by construction. So
    ``observe_step`` can never latch again.

    **The distance moved from 12 m to 16 m at ADR-0019, and the reason is the defect.**
    At 12 m one sounding fold used to read 0.00171 -- below the 0.003 threshold -- because
    the 5 s analysis window was 1/5 full. It now reads 0.00317 and CROSSES, because one
    fold writes the cue window whole. So this arm was passing partly for the wrong reason:
    the source was audible and the analysis window was hiding it. At 16 m one fold reads
    0.00248 and the episode is genuinely out of earshot, which is the fact the arm claims.

    **Nothing anywhere raises.** ``assert_provenance``'s three checks are all about steps
    before ``t_anom`` or about the recorded artefact, so they pass; the funnel caps at
    ``T_ANOM_REACHED``; CLAP is never handed a clip; the audit writes. The episode reads
    as ordinary §2.5 attrition, indistinguishable from a source the agent simply failed
    to find — which is why ``RunConfig.sounding_steps`` defaults GENEROUSLY (ADR-0014)
    and why this arm is pinned rather than argued about in a comment.
    """

    @classmethod
    def setUpClass(cls):
        world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0, wall=lambda p: p.z < -1.0)
        source = Xyz(0.0, 0.0, -16.0)
        handle = FakeAudioSensorHandle(world, source)
        anomaly_episode = make_anomaly_episode(
            source=source,
            t_anom=2,
            episode=make_episode(start_yaw=0.0, goals=[make_goal(Xyz(0.0, 0.0, -40.0))]),
        )
        cfg = make_config(max_steps=20, t_anom=2, sounding_steps=1)
        cls.result = run(world, handle, anomaly_episode, cfg, calibration=CALIBRATION)

    def test_the_single_sounding_fold_really_is_below_threshold(self):
        """The arm's own premise, measured, because it used to hold for the wrong reason.

        A forced failure that fails because the analysis window had not filled is not the
        failure this class names. The sounding step's reading is the SETTLED level for
        that pose since ADR-0019, so this assertion is now about the distance and nothing
        else.
        """
        sounding = [row for row in self.result.audit.steps if row.source_playing]
        self.assertEqual(len(sounding), 1)
        self.assertLess(sounding[0].measured_rms, CALIBRATION.onset_rms)

    def test_the_onset_never_fires_and_nothing_raises(self):
        # Reaching here at all is half the assertion: `run_episode` calls
        # `assert_provenance`, which RAISES, before it builds either artefact.
        self.assertIsNone(self.result.audit.onset.onset_step)
        self.assertTrue(self.result.audit.onset.provenance_asserted)
        self.assertIsNone(self.result.report.heard_at_step)

    def test_it_reads_as_ordinary_attrition_in_the_funnel(self):
        self.assertEqual(self.result.audit.funnel_stage, FunnelStage.T_ANOM_REACHED)
        self.assertIsNone(self.result.audit.source_reached_step)
        self.assertIsNone(self.result.report.anomaly_class)

    def test_the_audit_is_still_written_with_the_window_that_did_it(self):
        """The only place the cause is visible. Without the window on the record this
        episode is unattributable — it looks exactly like a source out of earshot."""
        window = self.result.audit.sounding_window
        self.assertEqual(window.offset_step, 3)
        self.assertEqual(window.opens_at, 2)
        self.assertEqual(self.result.audit.metrics["sounding_duration_steps"], 1.0)
        self.assertNotIn("heard_within_window", self.result.audit.metrics)


class TestSwsOnAnEpisodeThatActuallyReachedTheSource(unittest.TestCase):
    """All three of SWS's outcomes, produced by the loop rather than asserted on counts.

    ``TestTheFullLoop``'s geometry reaches the source at step 16 and ends at step 33, so
    moving the offset step across 16 moves the episode between the metric's two answers
    and moving it past 33 removes the episode from the denominator entirely. That is the
    whole definition, exercised: SWS counts reaching the SOURCE, and only after the
    source went silent.

    **Since ADR-0019, closing the last quarter-metre AFTER the offset step needs a ROOM,
    and that is the finding rather than a fixture inconvenience.** With the delta fake the
    cue readout is exactly the bed on the offset step, the climb has nothing to rise on,
    the stall branch casts and the agent never enters the 1 m ring: measured, every
    ``sounding_steps`` from 9 to 13 reaches NOTHING at all, and 14 upward reaches at step
    16 because step 15 was still sounding. So the anechoic arm can only ever produce
    ``reached == offset_step``, never ``reached > offset_step``.

    Under the clip readout it produced both, and it should not have: what carried the
    agent through those steps was the 5 s analysis window emptying, which an anechoic
    control reproduces to ~1 point. ``RingingHandle`` -- a decaying 900-sample IR -- is
    what buys the step honestly (``post_offset_audible_steps`` 3 against the delta fake's
    0), so the strictly-after arm runs on it and the delta fake is its forced failure.
    """

    def _run(self, sounding_steps, handle_cls=FakeAudioSensorHandle):
        world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
        source = Xyz(0.0, 0.0, -5.0)
        handle = handle_cls(world, source)
        anomaly_episode = make_anomaly_episode(
            source=source,
            t_anom=2,
            episode=make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))]),
        )
        cfg = make_config(max_steps=80, t_anom=2, sounding_steps=sounding_steps)
        return run(world, handle, anomaly_episode, cfg, calibration=CALIBRATION)

    def test_reaching_the_source_after_the_offset_step_scores(self):
        """BOTH ARMS. The room's own tail is what carries the agent past the offset step,
        and the anechoic control is what proves the tail is doing it."""
        audit = self._run(13, RingingHandle).audit
        self.assertEqual(audit.sounding_window.offset_step, 15)
        self.assertEqual(audit.source_reached_step, 16)
        self.assertGreater(audit.source_reached_step, audit.sounding_window.offset_step)
        self.assertEqual(audit.metrics["sws_eligible"], 1.0)
        self.assertEqual(audit.metrics["sws"], 1.0)
        self.assertAlmostEqual(silent_phase_tally([audit]).sws, 1.0)
        self.assertGreater(audit.sounding_window.post_offset_audible_steps, 0)

        # FORCED FAILURE -- the same geometry with a single-delta IR. Its cue readout is
        # the bed on the offset step, so the agent never closes the last 0.25 m and the
        # episode is eligible-and-zero rather than eligible-and-one.
        anechoic = self._run(13).audit
        self.assertEqual(anechoic.sounding_window.offset_step, 15)
        self.assertIsNone(anechoic.source_reached_step)
        self.assertEqual(anechoic.sounding_window.post_offset_audible_steps, 0)
        self.assertEqual(anechoic.metrics["sws_eligible"], 1.0)
        self.assertEqual(anechoic.metrics["sws"], 0.0)

    def test_reaching_it_on_the_offset_step_itself_counts_as_silent(self):
        """``[opens_at, offset_step)`` is the sounding phase, so the offset step is the
        FIRST silent step. The boundary is a decision, not an accident."""
        audit = self._run(14).audit
        self.assertEqual(audit.sounding_window.offset_step, 16)
        self.assertEqual(audit.source_reached_step, 16)
        self.assertEqual(audit.metrics["sws"], 1.0)

    def test_reaching_it_one_step_before_the_offset_is_eligible_and_scores_zero(self):
        """The sharp edge: the episode HAD a silent phase available and did not need it."""
        audit = self._run(15).audit
        self.assertEqual(audit.sounding_window.offset_step, 17)
        self.assertEqual(audit.source_reached_step, 16)
        self.assertEqual(audit.metrics["sws_eligible"], 1.0)
        self.assertEqual(audit.metrics["sws"], 0.0)
        self.assertAlmostEqual(silent_phase_tally([audit]).sws, 0.0)

    def test_an_episode_that_ended_before_its_own_offset_step_is_absent_not_zero(self):
        """The provisional 60-step default on this fixture. The episode ends at 33, the
        window would have closed at 62, and NOBODY ASKED — so there is no key at all."""
        audit = self._run(60).audit
        self.assertEqual(audit.sounding_window.offset_step, 62)
        self.assertLess(len(audit.steps), 62)
        self.assertEqual(audit.metrics["sws_eligible"], 0.0)
        self.assertNotIn("sws", audit.metrics)
        self.assertIsNone(silent_phase_tally([audit]).sws)


def _reaching_continuous_episode():
    """``TestTheFullLoop``'s geometry on the CONTINUOUS arm: the source IS reached.

    ``_windowed_episode``'s wall exists to buy a long silent phase and it costs the
    episode its reach, so a tally built from it has a zero everywhere -- which is how the
    JSON-level "never report SWS without SR" assertion came to compare zero against zero.
    """
    world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
    source = Xyz(0.0, 0.0, -5.0)
    handle = FakeAudioSensorHandle(world, source)
    anomaly_episode = make_anomaly_episode(
        source=source,
        t_anom=2,
        episode=make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))]),
    )
    cfg = make_config(max_steps=80, t_anom=2, sounding_policy=WindowPolicy.CONTINUOUS)
    return run(world, handle, anomaly_episode, cfg, calibration=CALIBRATION)


class TestTheRunLevelSilentPhaseTally(unittest.TestCase):
    """SWS over a run, with SR structurally beside it and no way to publish one alone.

    ``CONTEXT.md`` says *avoid reporting SWS without SR beside it*. A convention held by
    good intentions is one the next reader in a hurry breaks, so ``n_source_reached`` is
    a FIELD of the tally rather than something a caller is trusted to fetch.
    """

    def test_the_tally_counts_the_four_numbers_off_the_records(self):
        result, _cfg = _windowed_episode(sounding_steps=6)
        tally = silent_phase_tally([result.audit])
        self.assertEqual(tally.n_episodes, 1)
        self.assertEqual(tally.n_window_closed, 1)
        self.assertEqual(tally.n_reached_after_offset, 0)
        self.assertAlmostEqual(tally.sws, 0.0)
        self.assertAlmostEqual(tally.anomaly_response_sr, 0.0)

    def test_a_continuous_arm_run_reports_sws_as_not_run_rather_than_zero(self):
        """THE FORCED-FAILURE ARM for ADR-0014's rule, at the run level.

        The control arm has no offset step, so no episode is eligible and there is no
        SWS to report. 0.0 would say *the agent never succeeded in silence*; NOT_RUN says
        *nobody asked*, and the two incidents behind that rule are a probe that skipped
        and reported success and a canary that was never armed reading as a pass.
        """
        result, _cfg = _windowed_episode(sounding_policy=WindowPolicy.CONTINUOUS)
        tally = silent_phase_tally([result.audit])
        self.assertEqual(tally.n_window_closed, 0)
        self.assertIsNone(tally.sws)
        self.assertIsNot(tally.sws, 0.0)
        payload = tally.as_dict()
        self.assertIsNone(payload["sws"])
        self.assertEqual(payload["sws_status"], "not_run")

        # SR IS STILL THERE, and the episode above cannot show it: the wall stops the
        # agent, so `n_source_reached` is 0 and the assertion compared zero against zero.
        # `TestTheFullLoop`'s geometry on the continuous arm DOES reach the source, so the
        # SR beside the missing SWS is a number rather than an absence.
        reached = silent_phase_tally([_reaching_continuous_episode().audit])
        self.assertEqual(reached.n_source_reached, 1)
        self.assertEqual(reached.n_window_closed, 0)
        self.assertIsNone(reached.sws)
        payload = reached.as_dict()
        self.assertIsNone(payload["sws"])
        self.assertEqual(payload["sws_status"], "not_run")
        self.assertEqual(payload["n_source_reached"], 1)
        self.assertEqual(payload["anomaly_response_sr"], 1.0)

    def test_sws_and_sr_are_printed_on_one_line_or_neither_is(self):
        from earshot.task.runner import RunSummary

        measured = RunSummary(
            run_dir="runs/x", scene_label="FAKE", n_episodes=4,
            funnel={stage.name: 0 for stage in FunnelStage},
            silent_phase=SilentPhaseTally(
                n_episodes=4, n_window_closed=4, n_reached_after_offset=1,
                n_source_reached=2,
            ),
        )
        line = next(
            row for row in measured.summary().splitlines() if row.startswith("SWS")
        )
        self.assertIn("0.250", line)
        self.assertIn("SR: 2/4", line)

        not_run = RunSummary(
            run_dir="runs/x", scene_label="FAKE", n_episodes=4,
            funnel={stage.name: 0 for stage in FunnelStage},
            silent_phase=SilentPhaseTally(
                n_episodes=4, n_window_closed=0, n_reached_after_offset=0,
                n_source_reached=2,
            ),
        )
        line = next(
            row for row in not_run.summary().splitlines() if row.startswith("SWS")
        )
        self.assertIn("NOT_RUN", line)
        self.assertIn("SR: 2/4", line)

    def test_no_sws_is_produced_for_an_episode_whose_reverb_tail_did_not_run(self):
        """THE FORCED-FAILURE ARM for ADR-0017's own bar, and it is code rather than prose.

        ADR-0017 line 49: no sounding-window run may report an SWS before the
        accumulation buffer is in. Without the tail the silent phase arrives as a hard
        step to the bed, so an SWS measured on it is a number about an artefact.

        The evidence is ``max_ir_samples``, which is the widest IR the accumulator was
        actually HANDED and stays 0 until a sounding step folded one in -- the clause
        that separates "a tail was configured" from "a tail ran". Every path from an
        episode to an SWS numerator goes through this predicate, so the refusal is total:
        no partial rate over the episodes that did have one, and no 0.0.
        """
        result, _cfg = _windowed_episode(sounding_steps=6)
        no_tail = dataclasses.replace(
            result.audit,
            sounding_window=dataclasses.replace(
                result.audit.sounding_window, max_ir_samples=0
            ),
        )
        self.assertFalse(tail_is_active(no_tail.sounding_window))
        with self.assertRaises(TailNotActiveError):
            silent_phase_tally([no_tail])
        # the healthy arm, so the refusal above is not a function that always refuses
        self.assertTrue(tail_is_active(result.audit.sounding_window))
        self.assertIsNotNone(silent_phase_tally([result.audit]).sws)

    def test_a_record_written_before_the_window_existed_carries_no_tail_either(self):
        """Absent is unknown, and unknown is not evidence the accumulator was there.

        Such an episode is also never eligible -- it has no offset step -- so it lands in
        ``n_episodes`` and in SR and in neither half of SWS, which is the honest reading
        of a run that predates the mechanism.
        """
        from earshot.report.audit import EpisodeAudit

        old = EpisodeAudit(t_anom=2, funnel_stage=FunnelStage.SOURCE_REACHED)
        self.assertFalse(tail_is_active(old.sounding_window))
        tally = silent_phase_tally([old])
        self.assertEqual(tally.n_episodes, 1)
        self.assertEqual(tally.n_source_reached, 1)
        self.assertIsNone(tally.sws)

    def test_the_episode_itself_refuses_to_write_an_sws_the_tail_did_not_earn(self):
        """ADR-0017's bar is held in TWO places and only the tally's copy was exercised.

        ``run_episode`` checks it again before writing the per-episode ``sws`` key, and
        that copy is the one that matters: the tally reads records off disk, so an
        episode that wrote ``sws`` into its own ``metrics`` would have published the
        number before anything got the chance to refuse it. Deleting that ``raise`` left
        the whole suite green, because no in-loop configuration can reach it -- an
        eligible episode has sounding steps by construction, so its accumulator has
        always folded a render.

        Unreachable from the loop is exactly why it is forced here. The predicate is
        patched on the module, so what is being exercised is the guard's PLACEMENT --
        before the key is written, not after -- which is the property the comment claims
        and the only one that can go wrong.
        """
        import earshot.task.runner as runner

        healthy, _cfg = _windowed_episode(sounding_steps=6)
        self.assertEqual(healthy.audit.metrics["sws_eligible"], 1.0)
        self.assertIn("sws", healthy.audit.metrics)

        real = runner.tail_is_active
        runner.tail_is_active = lambda window: False
        try:
            with self.assertRaises(TailNotActiveError) as caught:
                _windowed_episode(sounding_steps=6)
        finally:
            runner.tail_is_active = real
        self.assertIn("offset step", str(caught.exception))
        # ...and the guard is not one that always refuses: same call, predicate restored.
        self.assertIn("sws", _windowed_episode(sounding_steps=6)[0].audit.metrics)


class TestTheCalibrationRanInTheDomainTheLoopReads(unittest.TestCase):
    """The threshold and the signal must come from the same path, and the drift is silent.

    ``onset_rms`` and the scatter ``climb_eps`` reads are both derived from the sweep and
    both applied to what ``tail.heard_step`` produces, and the accumulator's settled
    level sits above bare ``render_through_ir``. ``onset.py`` already names this failure
    — it "would silently move the domain the threshold was calibrated in" — and nothing
    raises on it, so the only real fix is that the sweep and the loop take the same code
    path. This spies on the hop actually passed.
    """

    @staticmethod
    def _spied_calibration(cfg):
        """``calibrate_episode`` with all three arms spied. Returns ``(result, calls)``.

        Since ADR-0019 the two ``Optional[hop]`` functions were SPLIT into four named
        ones, so this spy records WHICH function ran rather than which hop it was handed:
        an Optional that switches measurement domain is exactly the silent unit error the
        split removes, and a spy that could still express it would pin a surface that no
        longer exists.
        """
        import earshot.task.runner as runner

        seen = {"cue_sweep": [], "single": [], "loop": []}
        real_sweep = runner.sweep_cue_rms
        real_single = runner.sweep_render_scatter
        real_loop = runner.sweep_loop_scatter

        def spy_sweep(poses, render_at, clip, *, hop):
            seen["cue_sweep"].append(hop)
            return real_sweep(poses, render_at, clip, hop=hop)

        def spy_single(pose, render_at, clip, repeats=12):
            seen["single"].append(repeats)
            return real_single(pose, render_at, clip, repeats)

        def spy_loop(pose, render_at, clip, repeats=12, *, hop):
            seen["loop"].append(hop)
            return real_loop(pose, render_at, clip, repeats, hop=hop)

        world = FakeWorld()
        source = Xyz(3.0, 0.0, 0.0)
        handle = FakeAudioSensorHandle(world, source)
        runner.sweep_cue_rms = spy_sweep
        runner.sweep_render_scatter = spy_single
        runner.sweep_loop_scatter = spy_loop
        try:
            result, poses = calibrate_episode(world, handle, source, CLIP, cfg)
        finally:
            runner.sweep_cue_rms = real_sweep
            runner.sweep_render_scatter = real_single
            runner.sweep_loop_scatter = real_loop
        return result, seen, poses

    def test_the_sweep_is_handed_the_loops_own_hop(self):
        cfg = make_config()
        _result, seen, _poses = self._spied_calibration(cfg)
        expected = hop_samples(
            step_seconds=cfg.audio.step_seconds, sample_rate=cfg.audio.sample_rate
        )
        self.assertIsNotNone(expected)
        self.assertEqual(seen["cue_sweep"], [expected])

    def test_the_threshold_is_placed_on_the_CUE_LEVEL_and_it_does_not_move(self):
        """THE IDENTITY, THROUGH THE SWEEP. It is the one number that must NOT move.

        ``sweep_cue_rms`` measures the quadratic mean of the cue readout over the loop's
        ``phase_folds`` phases, and those windows are disjoint, consecutive and tile the
        settled period exactly -- so that mean EQUALS the clip readout's RMS. The sweep
        changed domain at ADR-0019 and ``onset_rms`` stayed where it was, which is what
        makes the whole change reviewable as one number that must not move beside several
        that must.

        Asserted here and not only in ``test_audio_tail.py`` because this is where the
        threshold is actually PLACED: a runner that took ``max(phases)`` instead would
        raise it by the crest factor and every historic threshold would be unpriceable.
        """
        from earshot.audio.clips import rms
        from earshot.audio.tail import steady_state_render

        cfg = make_config()
        hop = hop_samples(
            step_seconds=cfg.audio.step_seconds, sample_rate=cfg.audio.sample_rate
        )
        world = FakeWorld()
        source = Xyz(3.0, 0.0, 0.0)
        handle = FakeAudioSensorHandle(world, source)
        result, poses = calibrate_episode(world, handle, source, CLIP, cfg)

        def render_at(position):
            world.set_pose(position)
            observation, _guard = handle.observe()
            return handle.audio_of(observation)

        levels = [level for _distance, level in result.profile]
        self.assertEqual(len(levels), len(poses))
        for pose, level in zip(poses, levels):
            expected = rms(steady_state_render(render_at(pose), CLIP, hop=hop))
            self.assertAlmostEqual(level / expected, 1.0, places=9)
        # THE CONTROL, and without it the identity above is satisfied by any aggregation
        # of a constant sequence. The phases are NOT flat here -- measured 1.131 crest and
        # 0.514 min ratio at this fixture -- so the quadratic mean was a CHOICE, and the
        # rejected alternatives really would have moved the threshold: the maximum by
        # +13%, the minimum by -49%.
        self.assertGreater(result.cue_phase_crest, 1.10)
        self.assertLess(result.cue_phase_min_ratio, 0.60)

    def test_the_held_pose_is_measured_in_THREE_ARMS_and_climb_eps_reads_the_cue(self):
        """Three estimates of one word, and only one is the quantity ``is_rising``
        compares.

        ``cue_render_scatter`` is the spread of successive CUE readouts -- ``hop`` samples
        wide, which is what the agent reads since ADR-0019. ``clip_render_scatter`` is the
        ADR-0017 arm off the SAME folds, so it is free. ``single_render_scatter`` is the
        pre-ADR-0017 estimator, independent whole-clip renders, and the only one that
        costs extra renders; measured 1.91x above the clip arm at a held pose (3.55x under
        a second noise model, lag-1 autocorrelation 0.804 against 0.022).

        All three are kept because ``climb_eps``' input changed domain TWICE: every
        ``eps`` on disk is one of the two older numbers, and a change nobody can price
        against its own history is a change nobody can undo.
        """
        cfg = make_config()
        expected = hop_samples(
            step_seconds=cfg.audio.step_seconds, sample_rate=cfg.audio.sample_rate
        )
        result, seen, _poses = self._spied_calibration(cfg)
        self.assertEqual(seen["single"], [12], "the pre-ADR-0017 control arm did not run")
        self.assertEqual(seen["loop"], [expected])
        self.assertEqual(result.cue_scatter_repeats, 12)
        self.assertEqual(result.clip_scatter_repeats, 12)
        self.assertEqual(result.single_render_repeats, 12)
        self.assertIsNotNone(result.cue_render_scatter)
        self.assertIsNotNone(result.clip_render_scatter)
        self.assertIsNotNone(result.single_render_scatter)
        # The loop arm's two readouts come off ONE set of folds, so equal numbers would
        # mean the two readouts had collapsed into one.
        self.assertNotEqual(result.cue_render_scatter, result.clip_render_scatter)

    def test_the_loop_phases_reach_the_record_so_a_bursty_clip_is_identifiable(self):
        """``cue_phase_*`` is recorded and NOT gated, and this pins both halves.

        A clip whose energy sits inside one hop is loud on one fold and near-silent on the
        others, so the gate can pass on the quadratic mean while four folds in five read
        at the bed. That is honest -- the loop really is silent then -- and refusing on it
        would make four of ESC-50's five classes unusable, which is the argument
        ``tail_is_active``'s docstring already makes about the same clips. So it is
        measured, written down, and left to a later decision.
        """
        cfg = make_config()
        result, _seen, _poses = self._spied_calibration(cfg)
        hop = hop_samples(
            step_seconds=cfg.audio.step_seconds, sample_rate=cfg.audio.sample_rate
        )
        self.assertEqual(
            result.cue_phase_folds, len(CLIP) // math.gcd(len(CLIP), hop)
        )
        self.assertEqual(result.cue_phase_aggregation, "quadratic_mean_over_loop_phases")
        self.assertGreaterEqual(result.cue_phase_crest, 1.0)
        self.assertLessEqual(result.cue_phase_min_ratio, 1.0)
        self.assertTrue(result.passed, "the gate is not tightened by the phase block")

    def test_climb_eps_reads_the_CUE_arm_and_never_the_two_kept_ones(self):
        """The record carries three and the CONTROLLER must be handed exactly one.

        Handing it the clip arm would replay ADR-0017's epsilon against a reading that is
        no longer that quantity; handing it the single arm would be the pre-ADR-0017
        number. Neither is what ``is_rising`` compares, no artefact records which ``eps``
        was in force at the call site, and all three are distinct here on purpose -- so
        the binding is asserted at the call.
        """
        import earshot.task.runner as runner

        seen = []
        real_eps = runner.climb_eps
        runner.climb_eps = lambda scatter: seen.append(scatter) or real_eps(scatter)
        try:
            _windowed_episode(
                sounding_steps=6,
                calibration_override=dataclasses.replace(
                    CALIBRATION,
                    cue_render_scatter=2.5e-4,
                    cue_scatter_repeats=12,
                    clip_render_scatter=1.1e-4,
                    clip_scatter_repeats=12,
                    single_render_scatter=9.9e-4,
                    single_render_repeats=12,
                ),
            )
        finally:
            runner.climb_eps = real_eps
        self.assertTrue(seen)
        self.assertEqual(set(seen), {2.5e-4})

    def test_all_three_arms_survive_the_audit_round_trip(self):
        """A number that cannot be read back off ``audit.json`` cannot price anything."""
        from earshot.report.audit import EpisodeAudit

        result, _cfg = _windowed_episode(
            sounding_steps=6,
            calibration_override=dataclasses.replace(
                CALIBRATION,
                cue_render_scatter=2.5e-4,
                cue_scatter_repeats=12,
                clip_render_scatter=1.1e-4,
                clip_scatter_repeats=12,
                single_render_scatter=9.9e-4,
                single_render_repeats=12,
                cue_phase_folds=5,
                cue_phase_crest=1.13,
                cue_phase_min_ratio=0.52,
                cue_phase_aggregation="quadratic_mean_over_loop_phases",
            ),
        )
        record = result.audit.calibration
        self.assertIsNotNone(
            record.single_render_scatter,
            "the pre-ADR-0017 estimator is missing from the record, so this episode's "
            "eps cannot be priced against the runs detour-2 and eps-1 were tuned on",
        )
        self.assertIsNotNone(
            record.clip_render_scatter,
            "the ADR-0017 arm is missing, so this episode's eps cannot be priced against "
            "the era between the accumulator and the split readout",
        )
        self.assertAlmostEqual(record.cue_render_scatter, 2.5e-4)
        self.assertAlmostEqual(record.clip_render_scatter, 1.1e-4)
        self.assertAlmostEqual(record.single_render_scatter, 9.9e-4)
        self.assertEqual(record.single_render_repeats, 12)
        self.assertEqual(record.cue_phase_folds, 5)
        self.assertEqual(record.cue_phase_aggregation, "quadratic_mean_over_loop_phases")
        restored = EpisodeAudit.from_dict(result.audit.as_dict())
        self.assertEqual(restored.calibration, record)

    def test_a_record_written_before_the_control_arm_existed_reads_as_absent(self):
        """Absent is *not measured*, never 0.0: a scatter of zero says the renderer
        agrees with itself perfectly, which is a claim and not a gap."""
        from earshot.report.audit import CalibrationRecord

        old = CalibrationRecord.from_dict(
            {
                "onset_rms": 3e-3,
                "bed_rms": 1e-3,
                "separation_db": 18.0,
                "n_poses": 16,
                "global_volume": 1.0,
                "passed": True,
            }
        )
        self.assertIsNone(old.single_render_scatter)
        self.assertEqual(old.single_render_repeats, 0)


class WideIrHandle(FakeAudioSensorHandle):
    """The same fake with a wider impulse response, and it exists to defeat a constant.

    ``max_ir_samples`` was writable as a plausible literal with the whole suite green,
    which is the CLAP forced-failure scar exactly: a record that states a width the
    accumulator never measured. One IR width cannot tell a measurement from a constant,
    so there are two.
    """

    IR_LENGTH = 900


class PreallocatingHandle(FakeAudioSensorHandle):
    """A handle whose guard report carries the IR width its arming render measured.

    ``AudioContextReport()``'s default ``ir_shape`` is ``None`` -- the Mac fake, and the
    only shape the suite exercised -- so ``run_episode``'s preallocation hint could be
    replaced by a bare ``0`` and nothing noticed. On the box that costs every episode a
    2.3 MB reallocation on its first sounding step, inside the ``audio_render_s`` bracket
    whose 0.5 s ceiling criterion 7 has already breached once at 0.5335 s.
    """

    def __init__(self, world, source, gain=0.5, visible=True):
        super().__init__(world, source, gain=gain, visible=visible)
        self.report = AudioContextReport(ir_shape=(2, self.IR_LENGTH))


def _episode_with(handle_cls, clip=None, **cfg_overrides):
    """``_windowed_episode``'s geometry, with the handle under test.

    ``clip`` overrides the module's ``CLIP``. It exists for one caller -- the second
    geometry in ``TestTheWindowRecordIsTheAccumulatorsOwnMeasurement`` -- because a
    record field that is ``len(clip)`` cannot be told from a literal while every fixture
    in the file shares one clip.
    """
    world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0, wall=lambda p: p.z < -1.0)
    source = Xyz(0.0, 0.0, -5.0)
    handle = handle_cls(world, source)
    anomaly_episode = make_anomaly_episode(
        source=source,
        t_anom=2,
        episode=make_episode(start_yaw=0.0, goals=[make_goal(Xyz(0.0, 0.0, -40.0))]),
    )
    cfg = make_config(max_steps=40, t_anom=2, sounding_steps=6, **cfg_overrides)
    extra = {} if clip is None else {"clip": clip}
    return run(world, handle, anomaly_episode, cfg, calibration=CALIBRATION, **extra), cfg


class TestTheWindowRecordIsTheAccumulatorsOwnMeasurement(unittest.TestCase):
    """Every number on the record has to have been MEASURED, not restated.

    ``tail_is_active`` gates whether an SWS may be published at all, and the one field it
    turns on is ``max_ir_samples`` -- yet nothing compared the record's value against what
    the accumulator folded. Both existing arms sidestep the binding: one
    ``dataclasses.replace``s the field on an already-written record and the other patches
    the predicate out of the module. So a runner writing a plausible width while the
    buffer folded nothing published an SWS with a green suite, which is the CLAP
    forced-failure scar with a different field name.

    The same hole ran along the whole record: ``step_seconds`` and ``n_buffer_grows``
    were both replaceable by constants. ``n_buffer_grows`` is the one ``audit.py`` argues
    hardest for -- it is on the record *precisely* so a truncation is visible in the
    artefact instead of inferred from a level that looks a little low.

    **A SECOND GEOMETRY, because two IR widths only defeat a constant on the two fields
    the IR width feeds.** Every fixture in this file used one clip (2205 samples) and one
    ``step_seconds`` (0.01, a 441-sample hop), so ``hop_samples``, ``analysis_window_
    samples``, ``ramp_steps`` and ``sounding_phase_folds`` were each ONE number and each
    writable as a plausible literal with the whole suite green -- measured, four separate
    holes. Worse, 441 divides 2205, so ``clip_ramp_steps`` and ``phase_folds`` coincided
    at 5 and either could be swapped for the other.

    ``ALT`` is the second geometry and it is chosen so all four separate at once: a
    1764-sample clip at a 630-sample hop gives window 1764, hop 630, ``ramp_steps`` 3 and
    ``phase_folds`` 14 -- every one different from the shipped fixture's 2205 / 441 / 5 /
    5, and the ramp different from the period, which the shipped pair cannot show.
    """

    # 630 = round(44100/70) samples a step against a 1764-sample clip. 630 does NOT
    # divide 1764, which is the property that pulls `clip_ramp_steps` (3) apart from
    # `phase_folds` (14); on the shipped fixture 441 divides 2205 and the two coincide.
    ALT_STEP_SECONDS = 630.0 / 44100.0
    ALT_CLIP = synthetic_burst(44100, seconds=0.04)

    @classmethod
    def setUpClass(cls):
        cls.narrow, cls.cfg = _episode_with(FakeAudioSensorHandle)
        cls.wide, _ = _episode_with(WideIrHandle)
        cls.preallocated, _ = _episode_with(PreallocatingHandle)
        cls.hop = hop_samples(
            step_seconds=cls.cfg.audio.step_seconds,
            sample_rate=cls.cfg.audio.sample_rate,
        )
        cls.alt, cls.alt_cfg = _episode_with(
            FakeAudioSensorHandle,
            audio=AudioConfig(step_seconds=cls.ALT_STEP_SECONDS),
            clip=cls.ALT_CLIP,
        )
        cls.alt_hop = hop_samples(
            step_seconds=cls.alt_cfg.audio.step_seconds,
            sample_rate=cls.alt_cfg.audio.sample_rate,
        )

    def test_the_recorded_ir_width_is_the_one_the_accumulator_was_handed(self):
        """TWO widths, because one cannot separate a measurement from a literal."""
        self.assertEqual(
            self.narrow.audit.sounding_window.max_ir_samples,
            FakeAudioSensorHandle.IR_LENGTH,
        )
        self.assertEqual(
            self.wide.audit.sounding_window.max_ir_samples, WideIrHandle.IR_LENGTH
        )
        self.assertNotEqual(FakeAudioSensorHandle.IR_LENGTH, WideIrHandle.IR_LENGTH)

    def test_the_tail_length_is_derived_from_that_width_rather_than_pinned(self):
        """``clips.py`` forbids a fixed-width assumption about the IR: ticket 06's
        ``[2, 72300]`` is one scene's measurement and not a cap. A wider IR must produce
        a longer tail, which is the whole reason the width is on the record.
        """
        for result, ir_length in (
            (self.narrow, FakeAudioSensorHandle.IR_LENGTH),
            (self.wide, WideIrHandle.IR_LENGTH),
        ):
            window = result.audit.sounding_window
            self.assertEqual(
                window.tail_steps,
                int(math.ceil((len(CLIP) + ir_length - 1) / float(self.hop))),
                ir_length,
            )
        self.assertGreater(
            self.wide.audit.sounding_window.tail_steps,
            self.narrow.audit.sounding_window.tail_steps,
        )

    def test_the_unit_conversion_on_the_record_is_this_runs_own(self):
        """``step_seconds`` is the invented unit (``provenance: fake``) a window duration
        in STEPS has to be read through before it can be cross-quoted at all. A 1.0
        written onto a run configured at 0.01 makes every such reading 100x wrong."""
        window = self.narrow.audit.sounding_window
        self.assertEqual(window.step_seconds, self.cfg.audio.step_seconds)
        self.assertNotEqual(window.step_seconds, AudioConfig().step_seconds)
        self.assertEqual(window.hop_samples, self.hop)
        self.assertEqual(window.analysis_window_samples, len(CLIP))
        self.assertEqual(
            window.ramp_steps, int(math.ceil(len(CLIP) / float(self.hop)))
        )

    def test_the_buffers_geometry_separates_across_TWO_clips_and_TWO_hops(self):
        """TWO geometries, because one cannot tell any of these from a literal.

        ``hop_samples``, ``analysis_window_samples``, ``ramp_steps`` and
        ``sounding_phase_folds`` were each one number across every fixture in this file
        -- 441, 2205, 5, 5 -- and each was measured writable as that literal with the
        whole suite green. Four separate holes of the shape ``max_ir_samples`` already
        has two IR widths to close.

        The pair is also chosen so ``ramp_steps`` and ``phase_folds`` come APART. 441
        divides 2205, so on the shipped fixture ``ceil(N/hop)`` and ``N // gcd(N, hop)``
        are the same 5 and either expression could be swapped for the other; 630 does not
        divide 1764, and there they are 3 and 14.
        """
        shipped = self.narrow.audit.sounding_window
        alt = self.alt.audit.sounding_window
        for window, clip, hop in (
            (shipped, CLIP, self.hop),
            (alt, self.ALT_CLIP, self.alt_hop),
        ):
            self.assertEqual(window.hop_samples, hop)
            self.assertEqual(window.analysis_window_samples, len(clip))
            self.assertEqual(window.ramp_steps, int(math.ceil(len(clip) / float(hop))))
        for field in ("hop_samples", "analysis_window_samples", "ramp_steps"):
            self.assertNotEqual(
                getattr(shipped, field),
                getattr(alt, field),
                "{} is the same in both geometries, so a literal still passes".format(field),
            )
        # The loop's period, which only the metrics bag carries, and the one number that
        # bounds the onset's worst-case delay on a bursty clip.
        for result, clip, hop in (
            (self.narrow, CLIP, self.hop), (self.alt, self.ALT_CLIP, self.alt_hop)
        ):
            self.assertEqual(
                result.audit.metrics["sounding_phase_folds"],
                float(phase_folds(window=len(clip), hop=hop)),
            )
            # The metrics bag's copy of the ramp is a SECOND writable literal beside the
            # record's, and the two are meant to be one number.
            self.assertEqual(
                result.audit.metrics["sounding_ramp_steps"],
                float(math.ceil(len(clip) / float(hop))),
            )
            self.assertEqual(
                result.audit.metrics["sounding_ramp_steps"],
                float(result.audit.sounding_window.ramp_steps),
            )
        for key in ("sounding_phase_folds", "sounding_ramp_steps"):
            self.assertNotEqual(
                self.narrow.audit.metrics[key], self.alt.audit.metrics[key], key
            )
        # ...and the ramp is NOT the period, which the shipped fixture alone cannot say.
        self.assertNotEqual(
            float(alt.ramp_steps), self.alt.audit.metrics["sounding_phase_folds"]
        )

    def test_a_buffer_with_no_headroom_grows_once_and_the_record_counts_it(self):
        """``AudioContextReport().ir_shape`` is ``None`` on the Mac fake, so the buffer
        is opened at ``len(clip)`` and grows on the first sounding fold. Counted, because
        a truncating accumulator loses reverb silently and the signal stays plausible."""
        self.assertIsNone(FakeAudioSensorHandle(FakeWorld(), Xyz(0, 0, 0)).report.ir_shape)
        self.assertEqual(self.narrow.audit.sounding_window.n_buffer_grows, 1)
        self.assertEqual(self.wide.audit.sounding_window.n_buffer_grows, 1)

    def test_the_guards_measured_ir_width_preallocates_the_buffer(self):
        """THE OTHER ARM of the same field, and the only test of the hint at all.

        Handed the width the arming render already measured, the buffer is opened wide
        enough and never reallocates. Without the hint every episode on the box pays a
        2.3 MB allocation on its first sounding step, inside the bracket criterion 7's
        0.5 s ceiling audits.
        """
        window = self.preallocated.audit.sounding_window
        self.assertEqual((2, PreallocatingHandle.IR_LENGTH),
                         tuple(PreallocatingHandle(FakeWorld(), Xyz(0, 0, 0)).report.ir_shape))
        self.assertEqual(window.n_buffer_grows, 0)
        # ...and it is the same buffer otherwise: the hint may not change the signal.
        self.assertEqual(window.max_ir_samples, FakeAudioSensorHandle.IR_LENGTH)
        self.assertEqual(
            [row.measured_rms for row in self.preallocated.audit.steps],
            [row.measured_rms for row in self.narrow.audit.steps],
        )


class TestEveryClauseOfTheTailPredicateIsLoadBearing(unittest.TestCase):
    """``tail_is_active`` has four clauses and three of them asserted nothing.

    Deleting the ``hop_samples`` pair, the ``analysis_window_samples`` pair or the
    ``tail_steps`` pair each left the whole suite green; only ``max_ir_samples`` went
    red. A predicate whose clauses nobody exercises is a predicate that can be narrowed
    by accident, and this one is the single gate between an episode and an SWS.

    Each record below fails EXACTLY ONE clause, so a green here says the clause is
    load-bearing rather than merely present.
    """

    HEALTHY = SoundingWindowRecord(
        opens_at=2,
        offset_step=8,
        policy=WindowPolicy.FIXED_STEPS.value,
        step_seconds=0.01,
        hop_samples=441,
        analysis_window_samples=2205,
        max_ir_samples=64,
        n_buffer_grows=1,
        tail_steps=6,
        ramp_steps=5,
    )

    def test_the_healthy_record_answers_true(self):
        """Otherwise every refusal below is a function that always refuses."""
        self.assertTrue(tail_is_active(self.HEALTHY))

    def test_each_clause_alone_is_enough_to_answer_false(self):
        for field in (
            "hop_samples",
            "analysis_window_samples",
            "max_ir_samples",
            "tail_steps",
        ):
            for value in (0, None):
                bent = dataclasses.replace(self.HEALTHY, **{field: value})
                self.assertFalse(
                    tail_is_active(bent),
                    "{}={!r} left the tail reading as ACTIVE, so that clause asserts "
                    "nothing and an SWS can be published over a buffer that never "
                    "folded a render".format(field, value),
                )

    def test_an_absent_record_is_unknown_and_unknown_is_not_evidence(self):
        self.assertFalse(tail_is_active(None))


class TestTheOtherTwoPoliciesReachTheRunnerAtAll(unittest.TestCase):
    """``DRAWN`` and ``BUDGET_FRACTION`` were reachable in ``plan_window`` and nowhere else.

    No runner or smoke test set ``sounding_policy`` to either, so every argument the
    runner passes to ``plan_window`` could be replaced by a literal with the suite green:
    the episode index, the seed, the draw range and the budget fraction, four for four.

    **The index is the one with teeth.** ``tools/episode_diff.py`` pairs the SAME episode
    index across two sweeps and it is the only test this apparatus has that can resolve a
    delta of a dozen episodes; a run whose index was dropped draws ONE duration for every
    episode, every pair then compares two different tasks, and nothing downstream can
    tell. ``audio/window.py`` argues the draw must be a pure function of ``(seed, index)``
    for exactly that reason -- and until now nothing checked that the runner handed it
    either.
    """

    RANGE = (5, 25)

    def _drawn(self, index, **cfg_overrides):
        cfg_overrides.setdefault("sounding_draw_steps", self.RANGE)
        result, cfg = _windowed_episode(
            sounding_policy=WindowPolicy.DRAWN, index=index, **cfg_overrides
        )
        return result.audit.sounding_window, cfg

    def _expected(self, cfg, index):
        return plan_window(
            t_anom=int(cfg.t_anom),
            max_steps=int(cfg.max_steps),
            policy=cfg.sounding_policy,
            sounding_steps=int(cfg.sounding_steps),
            budget_fraction=float(cfg.sounding_budget_fraction),
            draw_steps_range=cfg.sounding_draw_steps,
            seed=int(cfg.seed),
            episode_index=int(index),
        )

    def test_a_drawn_run_records_the_window_the_planner_draws_for_that_episode(self):
        """All four arguments at once: same seed, same range, same index, same answer."""
        for index in (0, 1, 2, 3):
            window, cfg = self._drawn(index)
            expected = self._expected(cfg, index)
            self.assertEqual(window.policy, "drawn")
            self.assertEqual(
                window.offset_step,
                expected.offset_step,
                "episode {} recorded a window the planner does not draw for it, so a "
                "sweep paired by episode index is comparing different tasks".format(index),
            )

    def test_two_episode_indices_in_one_run_draw_different_durations(self):
        """The forced failure for the index specifically, and it is not the same test.

        The equality above still holds when the index is dropped IF every episode draws
        the same duration -- which is precisely the bug. This says the durations move.
        """
        durations = {
            index: self._drawn(index)[0].offset_step for index in range(6)
        }
        self.assertGreater(
            len(set(durations.values())),
            1,
            "every episode index drew the same window {}, which is FIXED_STEPS wearing "
            "a policy name and silently breaks tools/episode_diff.py".format(durations),
        )

    def test_the_draw_is_the_runs_own_seed_and_not_a_fixed_one(self):
        """A red run that cannot be reproduced is not evidence (``RunConfig.seed``)."""
        default, cfg = self._drawn(1)
        reseeded, other = self._drawn(1, seed=11)
        self.assertNotEqual(cfg.seed, other.seed)
        self.assertEqual(default.offset_step, self._expected(cfg, 1).offset_step)
        self.assertEqual(reseeded.offset_step, self._expected(other, 1).offset_step)
        self.assertNotEqual(default.offset_step, reseeded.offset_step)

    def test_the_drawn_duration_stays_inside_the_configured_range(self):
        """A range the runner never passed on would leave every duration inside some
        OTHER range, and the record would not say which."""
        for index in range(6):
            window, cfg = self._drawn(index)
            duration = int(window.offset_step) - int(window.opens_at)
            self.assertGreaterEqual(duration, self.RANGE[0], index)
            self.assertLessEqual(duration, self.RANGE[1], index)
        self.assertEqual(cfg.sounding_draw_steps, self.RANGE)

    def test_a_budget_fraction_run_records_floor_of_the_fraction_times_the_budget(self):
        """``floor(fraction * max_steps)``, read off the config rather than a default.

        The fraction is a RUN-level constant while ``t_anom`` is per episode, so this
        policy fixes the window's LENGTH while its opening still moves -- which is the
        whole reason it is a separate arm rather than FIXED_STEPS in disguise.
        """
        result, cfg = _windowed_episode(
            sounding_policy=WindowPolicy.BUDGET_FRACTION,
            sounding_budget_fraction=0.25,
        )
        window = result.audit.sounding_window
        self.assertEqual(cfg.max_steps, 40)
        self.assertEqual(window.policy, "budget_fraction")
        self.assertEqual(window.offset_step, cfg.t_anom + 10)
        self.assertEqual(result.audit.metrics["sounding_duration_steps"], 10.0)
        # ...and it is the CONFIGURED fraction, not the default one riding along.
        default = make_config().sounding_budget_fraction
        self.assertNotEqual(cfg.sounding_budget_fraction, default)
        self.assertNotEqual(
            window.offset_step, cfg.t_anom + int(default * cfg.max_steps)
        )


class TestAnOnsetThatFiredOnTheReverbTail(unittest.TestCase):
    """``heard_within_window`` had only ever been asserted TRUE, and 0.0 is the point.

    ``tail.py`` and ``runner.py`` both call ``onset_step > offset_step`` REACHABLE and
    load-bearing -- an agent can first cross threshold on the source's decaying tail
    rather than on the source -- and the key exists to separate the two in the record.
    With no episode producing the 0.0 case, ``<`` could be widened to ``<=`` or the
    metric written as a literal 1.0 and the whole suite stayed green, which would report
    every tail-only episode as having heard the source.

    **The fixture is built on measured levels, not on a guess, and ADR-0019 CHANGED WHAT
    MAKES IT POSSIBLE.** It used to rest on the fill ramp -- the 5 s analysis window was
    still filling when the window closed, so the reading climbed for one more step -- and
    that is the artefact the split readout removed. The cue readout is written whole by
    one fold, so a ramp-shaped fixture would now simply cross on the first sounding step.

    What replaces it is physical: the SOUND'S FLIGHT TIME. ``DelayedIrHandle`` puts the
    direct path at ``distance / 343 * sample_rate``, so a source 5 m away is 643 samples
    -- 1.46 hops at this file's 441-sample step -- behind its own emission. Sounding for
    ONE step from ``t_anom = 2``, measured:

        step 2  0.00100000   (the ONLY sounding step -- the bed, the sound is in flight)
        step 3  0.00422332   (the OFFSET step -- the arrival, and it is silent)
        step 4  0.00571858   (the maximum, two steps after the window closed)
        step 5  0.00100000   (the bed again; `cue_tail_steps` is 4)

    ``onset_rms`` is placed at 0.003, between the first two, so the first crossing lands
    exactly ON the offset step. That is the only fixture shape that separates ``<`` from
    ``<=``: an onset merely *after* the offset step answers 0.0 under both.
    """

    ONSET_RMS = TAIL_ONSET_RMS

    @classmethod
    def setUpClass(cls):
        cls.result, cls.cfg = _tail_onset_episode()

    def test_the_arrival_outlives_the_window_which_is_what_makes_this_possible(self):
        """The flight time outlives the window, so the loudest steps are silent ones.

        The last sounding step reads the BED -- not a partly-filled level, the bed -- and
        the arrival lands after the source was told to stop. Under the old clip readout
        this shape was unreachable: a 5 s analysis window swallows a 15 ms flight time
        whole.
        """
        rows = self.result.audit.steps
        offset = self.result.audit.sounding_window.offset_step
        self.assertEqual(offset, 3)
        self.assertFalse(rows[offset].source_playing)
        self.assertTrue(rows[offset - 1].source_playing)
        self.assertLess(rows[offset - 1].measured_rms, self.ONSET_RMS)
        self.assertGreaterEqual(rows[offset].measured_rms, self.ONSET_RMS)
        self.assertGreater(
            rows[offset + 1].measured_rms,
            rows[offset].measured_rms,
            "the arrival peaks on the offset step itself, so the flight time is inside "
            "one hop and this fixture is not measuring what it says it is",
        )

    def test_the_onset_fired_on_the_offset_step_and_the_record_says_it_was_not_the_source(self):
        onset = self.result.audit.onset
        offset = self.result.audit.sounding_window.offset_step
        self.assertEqual(onset.onset_step, offset)
        self.assertEqual(
            self.result.audit.metrics["heard_within_window"],
            0.0,
            "the offset step is the FIRST SILENT step, so an onset on it heard the tail "
            "and not the source",
        )

    def test_the_healthy_arm_answers_one_so_the_zero_is_not_a_metric_that_never_fires(self):
        healthy, _cfg = _windowed_episode(sounding_steps=6)
        window = healthy.audit.sounding_window
        self.assertLess(healthy.audit.onset.onset_step, window.offset_step)
        self.assertEqual(healthy.audit.metrics["heard_within_window"], 1.0)

    def test_the_delay_is_measured_from_the_window_opening_and_not_from_step_zero(self):
        """``t_anom`` is 2 and the onset is at 3, so the two readings differ.

        This fixture is the one that separates them: on ``TestTheFullLoop``'s geometry
        the onset lands on ``t_anom`` itself and the delay is 0, which agrees with the
        bare step index for every wrong reason.
        """
        metrics = self.result.audit.metrics
        self.assertEqual(self.result.audit.onset.onset_step, 3)
        self.assertEqual(self.cfg.t_anom, 2)
        self.assertEqual(metrics["onset_delay_steps"], 1.0)
        # THE TWO BOUNDS THE DELAY NOW CARRIES, in place of the fill ramp it used to.
        # There is no ramp bias left -- one fold writes the cue window whole -- so what is
        # left is the room's own build-up and the loop's period, and both are recorded so
        # an analyst can bound the residual instead of subtracting a constant.
        self.assertEqual(metrics["sounding_cue_tail_steps"], 4.0)
        self.assertEqual(metrics["sounding_phase_folds"], 5.0)

    def test_it_is_still_an_ordinary_episode_in_every_other_respect(self):
        """Hearing the tail is not an error state: nothing raises, the funnel climbs, the
        provenance assertion ran. Only this one key separates it."""
        audit = self.result.audit
        self.assertTrue(audit.onset.provenance_asserted)
        self.assertGreaterEqual(audit.funnel_stage, FunnelStage.ONSET_FIRED)
        self.assertEqual(audit.metrics["onset_delay_censored"], 0.0)


class TestTheSwsReachesTheRunSummary(unittest.TestCase):
    """The only path by which an SWS gets into ``summary.json``, and it was untested.

    ``run()`` imports ``sim.world`` so no Mac can execute it, and the two links either
    side of that -- ``silent_phase=silent_phase_tally(audits)`` at the call site, and the
    ``silent_phase`` block in ``as_dict`` -- could each be replaced with ``None`` and the
    whole suite stayed green. A box run would then finish with ``"silent_phase": null``
    in ``summary.json``, which is indistinguishable from a run in which no episode was
    eligible: the artefact says NOT_RUN and the reason is a dropped wire.

    The half a Mac can execute is executed here. ``tests/box/test_sounding_window_box.py``
    owns the other half, because a capability is exercised rather than proxied and this
    file cannot construct a ``World``.
    """

    @staticmethod
    def _tally():
        return SilentPhaseTally(
            n_episodes=4,
            n_window_closed=4,
            n_reached_after_offset=1,
            n_source_reached=2,
            n_tail_audible=3,
            n_tail_active=4,
        )

    def _summary(self, silent_phase):
        from earshot.task.runner import RunSummary

        return RunSummary(
            run_dir="runs/x",
            scene_label="FAKE",
            n_episodes=4,
            funnel={stage.name: 0 for stage in FunnelStage},
            silent_phase=silent_phase,
        )

    def test_the_tally_lands_in_the_artefact_whole(self):
        """Not a rounded scalar: the counts have to travel with the rate, because
        ``CONTEXT.md`` forbids reporting SWS without SR and a lone float cannot carry it.
        """
        tally = self._tally()
        payload = self._summary(tally).as_dict()
        self.assertIsNotNone(payload["silent_phase"])
        self.assertEqual(payload["silent_phase"], tally.as_dict())
        self.assertAlmostEqual(payload["silent_phase"]["sws"], 0.25)
        self.assertEqual(payload["silent_phase"]["sws_status"], "measured")
        self.assertEqual(payload["silent_phase"]["n_source_reached"], 2)
        self.assertAlmostEqual(payload["silent_phase"]["anomaly_response_sr"], 0.5)

    def test_the_artefact_is_json_and_the_null_means_no_tally_was_built(self):
        """A run that skipped the tally writes an explicit ``null`` and NOT a zero rate.

        ``run()`` takes this branch on the empty-dataset path, where no episode ran at
        all -- so the null is honest there and only there.
        """
        import json

        payload = self._summary(None).as_dict()
        self.assertIsNone(payload["silent_phase"])
        json.dumps(payload)  # raises on an Enum or a numpy scalar sneaking in
        json.dumps(self._summary(self._tally()).as_dict())

    def test_run_hands_the_tally_to_the_summary_it_writes(self):
        """The wire itself, read out of ``run()``'s ast because no Mac can call it.

        Same move ``TestTheRunnerAndTheRealCollaboratorsAgreeOnNames`` makes for
        ``sim/world.py``: the subject cannot be imported here, so it is read rather than
        executed. What this pins is that ``silent_phase`` is named on EVERY ``RunSummary``
        the function builds -- omitting it would default to ``None`` silently -- and that
        the one built after the episode loop is handed ``silent_phase_tally``.
        """
        import ast

        import _tree

        tree = _tree.parse(_tree.PACKAGE_ROOT / "task" / "runner.py")
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "RunSummary"
        ]
        self.assertTrue(calls, "run() builds no RunSummary at all")
        tallied = []
        for call in calls:
            keywords = {kw.arg: kw.value for kw in call.keywords}
            self.assertIn(
                "silent_phase",
                keywords,
                "a RunSummary in run() does not name silent_phase, so it defaults to "
                "None and summary.json says NOT_RUN for a run that measured it",
            )
            value = keywords["silent_phase"]
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                tallied.append(value.func.id)
        self.assertEqual(
            tallied,
            ["silent_phase_tally"],
            "exactly one RunSummary in run() may compute the tally -- the one after the "
            "episode loop; the empty-dataset path has no episodes and passes None",
        )


def _wall_episode():
    """``TestAWallTheClimbWalksInto``'s fixture, rebuilt so an arm can be swapped under it."""
    world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0, wall=lambda p: p.z < -1.0)
    source = Xyz(0.0, 0.0, -5.0)
    handle = FakeAudioSensorHandle(world, source)
    anomaly_episode = make_anomaly_episode(
        source=source, t_anom=0, episode=make_episode(start_yaw=0.0)
    )
    cfg = make_config(max_steps=40, t_anom=0)
    return run(world, handle, anomaly_episode, cfg, calibration=CALIBRATION)


def _blocked_forwards(audit):
    return [row for row in audit.steps if row.action == "move_forward" and row.collided]


def _first_turn_step(audit):
    return next(
        (row.step for row in audit.steps if row.action in ("turn_left", "turn_right")),
        None,
    )


def _rising_blocked_pairs(audit):
    """Consecutive blocked forwards whose reading went UP -- a rise the agent did not earn."""
    rows = audit.steps
    return sum(
        1
        for before, after in zip(rows, rows[1:])
        if before.action == "move_forward"
        and before.collided
        and after.action == "move_forward"
        and after.collided
        and after.measured_rms > before.measured_rms
    )


class TestTheWallAgainstThePreAdr0017Renderer(unittest.TestCase):
    """THREE ARMS, on one fixture, because two cannot attribute ADR-0019's change.

    ``audio/bed.py`` keeps ``heard_signal`` as the pre-ADR-0017 control, in as many words
    -- "retained, unchanged and un-called by the runner, as the continuous-source
    composition and the NAMED CONTROL the tail's Mac tests measure their decay against".
    That gives two arms. It is not enough: differencing the CUE readout against the
    whole-clip renderer confounds ADR-0019's split with ADR-0017's accumulator itself.
    So there is a middle arm that composes through the CLIP readout -- the runner's own
    behaviour between the two ADRs -- and the delta attributable to the split is the last
    column minus the middle one. This repo's own rule: a claim that X broke because of a
    change needs the arm where the change is absent.

    **Measured here, three arms, same wall, same 40 steps:**

        arm                             pre-ADR-0017   clip readout   cue readout
        blocked forwards, episode                 27             22            26
        first turn at step                        13             17             5
        blocked forwards before the turn           9             13             1
        rising blocked pairs                       0             14            14
        distinct readings, 5 held steps            1              5             5
        held-pose spread / max                  0.000         0.0038        0.548
        collided / forward                       0.871          0.846         0.867

    **THE FOUR-STEP STALL LAG DOES NOT SURVIVE -- IT INVERTS, AND BY EIGHT STEPS.** The
    clip readout turned four steps LATE because the 5 s window kept filling with the
    wall's own nearer pose. The cue readout turns EIGHT steps EARLY (step 5 against the
    pre-ADR-0017 arm's 13, twelve steps ahead of the clip readout's 17), and the reason is
    on the table above: the held-pose spread went from 0.38% of the level to 54.8%,
    because one fold in five of this fixture's burst clip carries little energy. The stall
    branch fires on the first quiet fold, one blocked forward after the wall. The
    manufactured rises are UNCHANGED at 14, so the buffer's memory is not what moved --
    the loop's intermittency is.

    **The lateral cue changed character too and nothing compensates.** It is now an ILD
    over one step's arrival instead of a five-pose average stale by up to four steps:
    measured, the cue arm's first nonzero ``lateral_sign`` is step 6 against the clip
    arm's step 18. Fresher when there is an arrival, ``LATERAL_AMBIGUOUS`` on a quiet fold
    because the bed is diotic. That rides in this commit unmeasured beyond this table.
    """

    @classmethod
    def setUpClass(cls):
        import earshot.task.runner as runner

        cls.cue = _wall_episode()

        # The pre-ADR-0017 composition, at the seam the runner actually calls. The
        # accumulator's state is threaded through untouched, so the CONTROL arm's record
        # carries `max_ir_samples` 0 -- which is correct: no fold ever happened. The
        # clip-length bed is rebuilt here from `bed_cue`'s own level rather than being
        # passed in, because since ADR-0019 the runner hands this seam the HOP-length bed
        # and `heard_signal` composes at the clip's length.
        def pre_adr_0017(state, *, ir, clip, bed_cue, sounding):
            return state, heard_signal(
                ir, clip, bed_signal(len(clip), rms(bed_cue)), playing=sounding
            )

        # The ADR-0017 arm: the same accumulator, read at the CLIP window's width, which
        # is exactly what `heard_step` returned before the split.
        def clip_readout_arm(state, *, ir, clip, bed_cue, sounding):
            nxt = advance_tail(state, ir=ir, clip=clip, sounding=sounding)
            return nxt, mix_bed(
                clip_readout(nxt), bed_signal(len(clip), rms(bed_cue))
            )

        real = runner.heard_step
        try:
            runner.heard_step = pre_adr_0017
            cls.whole_clip = _wall_episode()
            runner.heard_step = clip_readout_arm
            cls.clip = _wall_episode()
        finally:
            runner.heard_step = real

    def test_the_control_arm_repeats_its_reading_at_a_held_pose_and_this_one_does_not(self):
        """The property ``controller.py`` still asserts, true in exactly one of the arms."""
        control = [row.measured_rms for row in _blocked_forwards(self.whole_clip.audit)]
        clip = [row.measured_rms for row in _blocked_forwards(self.clip.audit)]
        cue = [row.measured_rms for row in _blocked_forwards(self.cue.audit)]
        self.assertGreater(len(control), 4)
        self.assertEqual(
            len(set(control[:5])),
            1,
            "the pre-ADR-0017 renderer is a pure function of pose and must repeat",
        )
        self.assertEqual(len(set(clip[4:9])), 5, clip)
        self.assertEqual(len(set(cue[4:9])), 5, cue)

    def test_the_held_pose_SPREAD_is_what_the_split_moved_and_it_moved_144x(self):
        """The number that explains every other row of the table.

        Measured over one loop period at a held pose, each arm settled by the readout it
        composes: the clip readout's five values span **0.38%** of the maximum, the cue
        readout's span **54.8%**, a ratio of 144. Same wall, same clip, same folds -- what
        differs is that the 5 s window averaged the loop's quiet fold away and a one-step
        window cannot.

        Each arm is settled by ITS OWN tail, which is the only fair comparison: the clip
        readout needs ``tail_steps`` folds at a pose before it stops carrying the previous
        one and the cue readout needs ``cue_tail_steps``. Settling the clip arm on the
        cue's tail reads 5.9% -- the ramp, not the loop -- which is the mistake this
        helper's argument exists to prevent.
        """
        def settled_spread(result, settle):
            rows = result.audit.steps
            start, length = _longest_held_run(rows)
            first = start + settle - 1
            self.assertGreaterEqual(
                length - (settle - 1), 5, "no settled loop period in the held run"
            )
            levels = [rows[first + k].measured_rms for k in range(5)]
            return (max(levels) - min(levels)) / max(levels)

        clip_spread = settled_spread(self.clip, self.clip.audit.sounding_window.tail_steps)
        cue_spread = settled_spread(self.cue, self.cue.audit.sounding_window.cue_tail_steps)
        self.assertLess(clip_spread, 0.01, clip_spread)
        self.assertGreater(cue_spread, 0.40, cue_spread)
        self.assertGreater(cue_spread / clip_spread, 100.0)

    def test_the_accumulator_manufactures_rises_the_control_arm_never_produces(self):
        """Zero against fourteen against fourteen. ``is_rising`` cannot tell any of them
        from a climb, and the split did NOT move this count -- which is what says the
        buffer's memory is not what changed the stall step."""
        self.assertEqual(_rising_blocked_pairs(self.whole_clip.audit), 0)
        self.assertEqual(_rising_blocked_pairs(self.clip.audit), 14)
        self.assertEqual(_rising_blocked_pairs(self.cue.audit), 14)

    def test_the_stall_lag_INVERTS_at_the_split_and_that_is_the_open_question(self):
        """Recorded rather than fixed. Compensating for the loop's phase would be a
        CONTROLLER change riding inside an audio commit, which is the confound this repo
        has already paid for once.

        The clip readout deferred the stall by four steps; the cue readout brings it
        forward by eight against the pre-ADR-0017 control and by twelve against the clip
        readout. Both deltas are the split's, and the middle arm is what attributes them.
        """
        self.assertEqual(_first_turn_step(self.whole_clip.audit), 13)
        self.assertEqual(_first_turn_step(self.clip.audit), 17)
        self.assertEqual(_first_turn_step(self.cue.audit), 5)
        self.assertEqual(
            _first_turn_step(self.clip.audit) - _first_turn_step(self.whole_clip.audit),
            4,
            "the ADR-0017 lag moved, so the middle arm is no longer the behaviour this "
            "change is being differenced against",
        )
        self.assertEqual(
            _first_turn_step(self.cue.audit) - _first_turn_step(self.clip.audit),
            -12,
            "the delta attributable to ADR-0019 moved",
        )

    def test_the_blocked_forward_counts_are_pinned_in_BOTH_scopes(self):
        """The two counts a docstring can print, and the reason both are asserted here.

        ``agent/controller.py`` published "9 / 13 blocked forwards" beside "14 rising
        blocked pairs". Those are different scopes -- the first pair counts only the
        forwards BEFORE the first turn, the second counts adjacent pairs over the whole
        40 steps -- and read together they are impossible, because 13 forwards cannot
        make 14 adjacent pairs. This class's own docstring meanwhile printed the
        whole-episode 22 / 27. Neither number was asserted anywhere, which is how they
        drifted; the same shape as ``bed.py``'s "once per run" against
        ``audio/config.py``'s "per step".
        """
        for label, audit, whole, before_turn in (
            ("cue readout", self.cue.audit, 26, 1),
            ("clip readout", self.clip.audit, 22, 13),
            ("pre-ADR-0017", self.whole_clip.audit, 27, 9),
        ):
            turn = _first_turn_step(audit)
            self.assertEqual(
                len(_blocked_forwards(audit)),
                whole,
                "{}: whole-episode blocked forwards moved".format(label),
            )
            self.assertEqual(
                len([row for row in _blocked_forwards(audit) if row.step < turn]),
                before_turn,
                "{}: blocked forwards before the first turn moved".format(label),
            )

    def test_the_lateral_cue_arrives_earlier_because_it_is_one_steps_arrival(self):
        """The risk this change carries into the controller, measured rather than argued.

        Before the split the sign was an ILD over a window holding folds rendered at up to
        ``clip_ramp_steps`` DIFFERENT poses -- a five-pose average stale by up to four
        steps. It is now this step's arrival: fresher when there is one, and
        ``LATERAL_AMBIGUOUS`` on a quiet fold, because the bed is diotic and contributes
        exactly zero ILD.
        """
        def first_lateral(result):
            return next(
                (row.step for row in result.audit.steps if row.lateral_sign != 0), None
            )

        self.assertEqual(first_lateral(self.clip), 18)
        self.assertEqual(first_lateral(self.cue), 6)

    def test_the_control_arm_folds_nothing_so_its_own_record_says_the_tail_never_ran(self):
        """The arms are separable from the artefact alone, which is what makes a delta
        attributable. ``tail_is_active`` reads False for the control and True for both
        accumulator arms, so a control-arm run could never publish an SWS by mistake."""
        self.assertFalse(tail_is_active(self.whole_clip.audit.sounding_window))
        self.assertTrue(tail_is_active(self.clip.audit.sounding_window))
        self.assertTrue(tail_is_active(self.cue.audit.sounding_window))
        self.assertEqual(self.whole_clip.audit.sounding_window.max_ir_samples, 0)
        self.assertEqual(
            self.cue.audit.sounding_window.max_ir_samples,
            FakeAudioSensorHandle.IR_LENGTH,
        )
        # The two accumulator arms are NOT separable from the record, and they must not
        # be: they fold the same buffer and differ only in which readout is composed. The
        # arm is the runner's code path, which is why the middle arm exists here rather
        # than being reconstructed from an audit.json.
        self.assertEqual(
            self.clip.audit.sounding_window.as_dict(),
            self.cue.audit.sounding_window.as_dict(),
        )


class _SpyClapEncoder(audio_fakes.FakeClapEncoder):
    """The fake encoder, plus the waveforms it was handed.

    ``FakeClapEncoder`` ignores its audio argument by design -- the cosines are the
    calibration's subject, not this file's -- but WHAT THE BUFFER CONTAINED at the moment
    of classification is exactly what ADR-0017 changed and what nothing recorded.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waveforms = []

    def encode_audio(self, waveform, sample_rate):
        self.waveforms.append(np.asarray(waveform))
        return super().encode_audio(waveform, sample_rate)


def _encoder_favouring(prompt):
    """A CLAP stand-in whose audio vector points at exactly one prompt.

    Every other prompt scores 0, so ``is_anomaly``'s calibrated pair decides the arm: a
    prompt from ``CLASS_TO_CLAP_PROMPT`` gives ``s_anom`` 1.0 and fires; a normal prompt
    leaves ``s_anom`` at 0, under the 0.0341 absolute floor, and does not.
    """
    from earshot.audio.clap import CLASS_TO_CLAP_PROMPT, NORMAL_PROMPTS

    vectors = {
        text: audio_fakes.one_hot(1, scale=0.0)
        for text in tuple(CLASS_TO_CLAP_PROMPT.values()) + tuple(NORMAL_PROMPTS)
    }
    vectors[prompt] = audio_fakes.one_hot(0, scale=1.0)
    return _SpyClapEncoder(audio_fakes.one_hot(0, scale=1.0), vectors)


def _encoder_scoring(scales):
    """A CLAP stand-in with a chosen cosine per prompt. `{prompt_text: scale}`.

    `_encoder_favouring` is the one-prompt case. This one exists because the gate and the
    testimony read DIFFERENT banks, so demonstrating that needs a clip scoring in both --
    one favoured prompt cannot show two banks disagreeing.
    """
    from earshot.audio.clap import CLASS_TO_CLAP_PROMPT, NORMAL_PROMPTS

    vectors = {
        text: audio_fakes.one_hot(1, scale=0.0)
        for text in tuple(CLASS_TO_CLAP_PROMPT.values()) + tuple(NORMAL_PROMPTS)
    }
    for text, scale in scales.items():
        vectors[text] = audio_fakes.one_hot(0, scale=scale)
    return _SpyClapEncoder(audio_fakes.one_hot(0, scale=1.0), vectors)


def _clap_episode(encoder, **cfg_overrides):
    """``TestTheFullLoop``'s geometry with an encoder wired in.

    The onset fires on ``t_anom`` itself here, one fold into a five-fold ramp, which is
    the case the deferral exists for.
    """
    world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
    source = Xyz(0.0, 0.0, -5.0)
    handle = FakeAudioSensorHandle(world, source)
    anomaly_episode = make_anomaly_episode(
        source=source,
        t_anom=2,
        episode=make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))]),
    )
    cfg_overrides.setdefault("max_steps", 80)
    cfg = make_config(t_anom=2, **cfg_overrides)
    return (
        run(
            world,
            handle,
            anomaly_episode,
            cfg,
            clap_encoder=encoder,
            calibration=CALIBRATION,
        ),
        cfg,
    )


class TestTheClassificationWaitsForTheReadWindowToFill(unittest.TestCase):
    """No test exercised the CLAP path through the loop at all, and ADR-0017 moved it.

    §4.3 classifies **the heard clip**, and since the accumulator that is the read window
    rather than a whole-clip ``render_through_ir``. It differs in three ways, none of
    which changes the RMS: the clip LOOPS, the readout is a cyclic ROTATION of one
    period, and the window can be PARTLY FULL. ADR-0018's bank of record and
    ``task/clap_gate.py``'s separation were both measured on the pre-ADR-0017 waveform,
    so the confound has to be visible in ``audit.json`` -- and until now the keys that
    make it visible were written by code nothing ran.

    Measured on this fixture: the onset fires at step 2 with the read window 1/5 full,
    the classification waits and fires at step 6 at fill 1.0, ``clap_deferred_steps`` 4,
    and ``is_anomaly`` is called ONCE with a full 2205-sample waveform.
    """

    @classmethod
    def setUpClass(cls):
        from earshot.audio.clap import CLASS_TO_CLAP_PROMPT

        cls.encoder = _encoder_favouring(CLASS_TO_CLAP_PROMPT["alarm"])
        cls.result, cls.cfg = _clap_episode(cls.encoder)

    def test_the_classification_is_deferred_until_the_window_holds_only_source(self):
        metrics = self.result.audit.metrics
        onset_step = self.result.audit.onset.onset_step
        self.assertEqual(onset_step, 2)
        self.assertEqual(metrics["clap_step"], 6.0)
        self.assertEqual(metrics["clap_deferred_steps"], float(6 - onset_step))
        self.assertEqual(metrics["clap_window_fill"], 1.0)
        self.assertEqual(
            metrics["clap_deferred_steps"],
            metrics["sounding_ramp_steps"] - 1.0,
            "the deferral is bounded by the ramp: ceil(N/hop) - 1 steps at most",
        )

    def test_clap_is_handed_the_whole_read_window_exactly_once(self):
        self.assertEqual(len(self.encoder.waveforms), 1)
        waveform = self.encoder.waveforms[0]
        self.assertEqual(waveform.shape, (len(CLIP),))
        self.assertEqual(self.encoder.seen_rates, [self.cfg.audio.sample_rate])
        # ...and it is not silence: a half-empty buffer classified in silence is the one
        # thing the deferral exists to prevent.
        self.assertGreater(
            float(np.sqrt(np.mean(waveform ** 2))), self.cfg.audio.bed_rms
        )

    def test_the_rotation_is_recorded_because_it_is_the_confound(self):
        """``clap_rotation_phase_samples`` is ``tail.phase`` -- the clip index the next
        hop starts at, which IS the rotation, with period ``N/hop``. Two episodes at the
        same pose whose onset step differs by one hand CLAP different waveforms, and
        without this key that difference is invisible in the artefact."""
        metrics = self.result.audit.metrics
        hop = hop_samples(
            step_seconds=self.cfg.audio.step_seconds,
            sample_rate=self.cfg.audio.sample_rate,
        )
        phase = metrics["clap_rotation_phase_samples"]
        self.assertEqual(phase % hop, 0.0)
        self.assertLess(phase, len(CLIP))
        # One hop per SOUNDING fold, wrapped at the clip's length: the classification
        # step is inclusive, so folds `t_anom` through `clap_step` have all advanced it.
        folds = int(metrics["clap_step"]) - self.cfg.t_anom + 1
        self.assertEqual(phase, float(folds * hop % len(CLIP)))

    def test_the_verdict_reached_the_report_and_the_detour_ran(self):
        self.assertEqual(self.result.report.anomaly_class, "alarm")
        self.assertGreaterEqual(
            self.result.audit.funnel_stage, FunnelStage.INVESTIGATE_ENTERED
        )
        self.assertEqual(self.result.audit.metrics["clap_after_offset"], 0.0)


class TestTheTestimonyNamesTheRunsOwnBank(unittest.TestCase):
    """The report said "alarm" on a `toilet_flush` episode, and this is both arms of it.

    `is_anomaly` defaults to `ANOMALY_CLASSES` -- `alarm`, `baby_cry`, `glass_break` --
    because `ANOMALY_GATE_DELTA` and `ANOMALY_GATE_TAU` were calibrated against exactly
    those prompts, and `clap.py` HAZARD 2 forbids quoting them for a wider bank. The
    runner then copied that gate's `best_class` into `report.anomaly_class`, which is an
    argmax over three emergency names whatever the episode's source actually was.

    ADR-0018's matrix runs `toilet_flush`, `snoring` and `keyboard_typing`, so under the
    old wiring every one of those episodes would testify to a sound that was not there.
    The gate keeps its bank; the testimony takes the run's own.

    The fixture makes the two banks disagree ON PURPOSE: the alarm prompt scores 0.5 and
    the flush prompt 1.0, so the gate fires calling it "alarm" and the testimony says
    "toilet_flush".
    """

    @classmethod
    def setUpClass(cls):
        from earshot.audio.clap import CLASS_TO_CLAP_PROMPT

        cls.encoder = _encoder_scoring({
            CLASS_TO_CLAP_PROMPT["alarm"]: 0.5,
            CLASS_TO_CLAP_PROMPT["toilet_flush"]: 1.0,
        })
        cls.result, cls.cfg = _clap_episode(cls.encoder, anomaly_class="toilet_flush")

    def test_the_report_names_the_class_that_was_heard(self):
        self.assertEqual(self.result.report.anomaly_class, "toilet_flush")

    def test_it_is_not_one_of_the_three_emergency_names(self):
        """The defect stated as itself, not as the absence of the fix."""
        from earshot.audio.clap import ANOMALY_CLASSES

        self.assertNotIn(self.result.report.anomaly_class, ANOMALY_CLASSES)

    def test_the_gate_still_fired_on_its_own_calibrated_bank(self):
        """Unchanged, and the detour still ran -- otherwise this would be a report fix
        that quietly disarmed the interrupt."""
        self.assertGreaterEqual(
            self.result.audit.funnel_stage, FunnelStage.INVESTIGATE_ENTERED
        )

    def test_one_forward_pass_serves_both_questions(self):
        """Two banks, ONE render. The audio encoder is 153.5 M params, and a gate and a
        testimony about different renders would agree only by luck."""
        self.assertEqual(len(self.encoder.waveforms), 1)


class TestTheCarriedBankStillWorks(unittest.TestCase):
    """The healthy arm: a run whose class IS one of the three is unchanged."""

    def test_an_alarm_run_still_testifies_to_alarm(self):
        from earshot.audio.clap import CLASS_TO_CLAP_PROMPT

        encoder = _encoder_scoring({
            CLASS_TO_CLAP_PROMPT["alarm"]: 1.0,
            CLASS_TO_CLAP_PROMPT["toilet_flush"]: 0.5,
        })
        result, _cfg = _clap_episode(encoder, anomaly_class="alarm")
        self.assertEqual(result.report.anomaly_class, "alarm")


class TestTheInterruptWaitsForTheVerdictItIsGatedOn(unittest.TestCase):
    """THE ONE BEHAVIOUR CHANGE in ADR-0017's runner, and its forced-failure arm.

    ``step_controller``'s SEARCH branch reads ``is_anomaly is None`` as *nothing
    conditioned this, so any onset interrupts* -- correct with no encoder, and a licence
    to divert on an UNDECIDED verdict with one. The detour latches (``investigated`` and
    ``investigate_aborted`` are both terminal, and ``is_anomaly`` is never read again
    once INVESTIGATE is entered), so a benign verdict arriving four steps later could not
    pull the agent back and ``n_benign_ignored`` would never count it: §4.3's gate would
    be spent on every mid-ramp onset.

    Measured with the benign encoder: the funnel caps at ONSET_FIRED and 26 benign steps
    are counted. With the withholding removed, the same episode diverts at the onset step
    and reaches INVESTIGATE_ENTERED regardless of the verdict.
    """

    @classmethod
    def setUpClass(cls):
        cls.benign = _encoder_favouring("people talking")
        cls.result, cls.cfg = _clap_episode(cls.benign)

    def test_a_benign_verdict_is_honoured_rather_than_arriving_after_the_detour(self):
        audit = self.result.audit
        self.assertEqual(
            audit.funnel_stage,
            FunnelStage.ONSET_FIRED,
            "the agent diverted on a sound CLAP called benign, so the gate bought "
            "nothing -- the detour latches and the verdict arrives too late",
        )
        self.assertGreater(self.result.report.n_benign_ignored, 0)
        self.assertFalse(self.result.report.resumed)
        self.assertFalse(self.result.report.investigate_aborted)

    def test_the_verdict_was_reached_at_all_so_the_cap_is_not_a_silent_no_op(self):
        """The healthy half: the classification DID run, at the deferred step, and said
        benign. A funnel capped at ONSET_FIRED because nothing was ever classified would
        pass the assertion above for the wrong reason."""
        metrics = self.result.audit.metrics
        self.assertEqual(metrics["clap_step"], 6.0)
        self.assertEqual(metrics["clap_window_fill"], 1.0)
        self.assertEqual(len(self.benign.waveforms), 1)
        self.assertNotIn("clap_deferred_unresolved", metrics)

    def test_a_run_with_no_encoder_keeps_the_pre_adr_0017_timing_to_the_step(self):
        """Nothing to wait for, so nothing waits -- which is what keeps this an audio
        change rather than a policy one, and why the whole Mac suite is unmoved."""
        no_encoder, _cfg = _clap_episode(None)
        self.assertEqual(no_encoder.audit.funnel_stage, FunnelStage.PRIMARY_RESUMED)
        self.assertIsNone(no_encoder.report.anomaly_class)
        # ...and the classification step is still recorded, so the confound keys exist
        # on a no-CLAP run too and an analyst can pool the two.
        self.assertEqual(no_encoder.audit.metrics["clap_step"], 6.0)


class TestAnEpisodeThatEndedMidDeferral(unittest.TestCase):
    """THE FORCED-FAILURE ARM of the deferral: the episode ran out first.

    Without ``clap_deferred_unresolved`` such an episode is indistinguishable from a run
    with no encoder at all -- ``anomaly_class`` is None on both, and one of them means
    *no encoder* while the other means *the agent heard it and stopped before we could
    say what it was*. Measured: the buffer is still filling at ``max_steps`` 4, 5 and 6,
    and the classification resolves at 7.
    """

    def test_the_unresolved_episode_says_so_rather_than_looking_like_a_no_clap_run(self):
        from earshot.audio.clap import CLASS_TO_CLAP_PROMPT

        for max_steps in (4, 5, 6):
            encoder = _encoder_favouring(CLASS_TO_CLAP_PROMPT["alarm"])
            result, _cfg = _clap_episode(encoder, max_steps=max_steps)
            metrics = result.audit.metrics
            self.assertEqual(metrics["clap_deferred_unresolved"], 1.0, max_steps)
            self.assertNotIn("clap_step", metrics)
            self.assertEqual(encoder.waveforms, [], "the encoder was called anyway")
            self.assertIsNone(result.report.anomaly_class)
            self.assertEqual(result.audit.funnel_stage, FunnelStage.ONSET_FIRED)

    def test_one_more_step_resolves_it_so_the_key_is_not_always_written(self):
        from earshot.audio.clap import CLASS_TO_CLAP_PROMPT

        encoder = _encoder_favouring(CLASS_TO_CLAP_PROMPT["alarm"])
        result, _cfg = _clap_episode(encoder, max_steps=7)
        self.assertNotIn("clap_deferred_unresolved", result.audit.metrics)
        self.assertEqual(result.audit.metrics["clap_step"], 6.0)
        self.assertEqual(len(encoder.waveforms), 1)


class TestAClassificationOnTheTailSaysSo(unittest.TestCase):
    """The other half of the deferral: once the window has CLOSED, waiting makes it worse.

    The buffer only empties from there, so whatever is in it gets classified and the
    record says it was a tail. Same fixture as ``TestAnOnsetThatFiredOnTheReverbTail``,
    with an encoder: the onset lands on the offset step, the CLIP window is 1/5 full of
    source and the rest is decay, and ``clap_after_offset`` is the flag that makes that
    legible in ``audit.json`` rather than a reviewer's inference.

    **The fill and the deferral are the CLIP readout's, and ADR-0019 left both alone.**
    That is the point of the split: the controller crossed threshold off a one-step cue
    while the classification still reasons about the 5 s waveform ADR-0018's bank of
    record was measured on.
    """

    @classmethod
    def setUpClass(cls):
        from earshot.audio.clap import CLASS_TO_CLAP_PROMPT

        cls.encoder = _encoder_favouring(CLASS_TO_CLAP_PROMPT["alarm"])
        cls.result, cls.cfg = _tail_onset_episode(clap_encoder=cls.encoder)

    def test_it_classifies_immediately_rather_than_waiting_for_a_fill_that_cannot_come(self):
        metrics = self.result.audit.metrics
        self.assertEqual(metrics["clap_step"], 3.0)
        self.assertEqual(metrics["clap_deferred_steps"], 0.0)
        self.assertLess(metrics["clap_window_fill"], 1.0)
        self.assertEqual(len(self.encoder.waveforms), 1)

    def test_the_record_says_the_buffer_was_a_decaying_tail(self):
        metrics = self.result.audit.metrics
        self.assertEqual(metrics["clap_after_offset"], 1.0)
        self.assertEqual(
            metrics["clap_step"], float(self.result.audit.sounding_window.offset_step)
        )
        self.assertEqual(metrics["heard_within_window"], 0.0)


class _UnroutableSourceWorld(FakeWorld):
    """A world with no navmesh route to the SOURCE, and an ordinary one to the goal.

    23 of ``yield-2``'s 365 episodes are this: the source sits on a disconnected navmesh
    island and the pathfinder answers ``None``. The straight-line ``xz`` distance is still
    measurable, which is exactly the trap -- a route recorded as 0.0 would put a phantom
    sample at the source in the very distribution the window's duration is chosen from.
    """

    def __init__(self, *args, **kwargs):
        self.unroutable = kwargs.pop("unroutable")
        super().__init__(*args, **kwargs)

    def geodesic_distance(self, start, ends):
        if list(ends) == [self.unroutable]:
            return None
        return super().geodesic_distance(start, ends)


def _out_of_earshot_episode(world=None, gain=0.1, **cfg_overrides):
    """A source 12 m away behind a wall, quiet enough to never cross ``onset_rms``.

    **``gain`` is what makes it out of earshot, and before ADR-0019 the RAMP was.** The
    default gain used to be ``FakeAudioSensorHandle``'s 0.5, which puts the settled
    received level at 5.769e-03 against a fixture threshold of 3e-03 -- i.e. the source
    was plainly AUDIBLE and the episode was censored only because the clip readout took
    ``clip_ramp_steps`` folds to fill: one sounding fold read 1.326e-03 and three were
    needed to cross. The cue readout is written whole by one fold
    (``tail.CUE_RAMP_STEPS``), so that fixture now fires on its single sounding step
    (2.964e-03 of source, 3.128e-03 once the diotic bed is mixed in) and stops being a
    censored episode at all.

    That is the fill-ramp bias ADR-0019 removed, caught in a fixture rather than in a
    run. The repair is to make the censoring a LEVEL fact, which is the only kind the cue
    readout can express: at ``gain = 0.1`` the settled level is 1.154e-03, its loudest
    loop phase 1.303e-03, and 1.635e-03 with the bed -- 1.8x under the threshold, and no
    number of sounding steps changes that.

    ``test_the_censored_arm_is_out_of_earshot_BY_LEVEL`` asserts exactly that, so the
    dependence cannot come back silently.
    """
    source = Xyz(0.0, 0.0, -12.0)
    if world is None:
        world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0, wall=lambda p: p.z < -1.0)
    handle = FakeAudioSensorHandle(world, source, gain=gain)
    anomaly_episode = make_anomaly_episode(
        source=source,
        t_anom=2,
        episode=make_episode(start_yaw=0.0, goals=[make_goal(Xyz(0.0, 0.0, -40.0))]),
    )
    cfg_overrides.setdefault("sounding_steps", 1)
    cfg = make_config(max_steps=20, t_anom=2, **cfg_overrides)
    return run(world, handle, anomaly_episode, cfg, calibration=CALIBRATION), cfg


class TestTheCensoredEpisodesAreVisible(unittest.TestCase):
    """``onset_delay_steps`` is RIGHT-CENSORED by construction and nothing said so.

    It exists only where the onset fired, so every value in the sample is smaller than
    the window that produced it: a run whose window was too short reports a comfortable
    median and hides the episodes it truncated. And the censored set is not recoverable
    from the funnel -- a ``T_ANOM_REACHED`` with no onset reads identically to ordinary
    §2.5 attrition, which is precisely what ``TestAWindowThatClosesBeforeTheAgentIsInEarshot``
    is.

    A censored episode is not empty. It carries a censoring TIME -- the sounding steps it
    actually got, so the delay is known to exceed it -- and a distance, which says whether
    *a bit longer* would have been enough or whether the agent was in another room the
    whole time. Together they are a survival sample rather than a gap, and this is the
    only test that asks for either.

    **The two arms differ by LEVEL since ADR-0019, not by window length.** They used to
    differ by ``sounding_steps`` alone, and that worked only because the clip readout's
    fill ramp made one sounding fold read a fifth of a plainly audible source. The cue
    readout has no fill ramp, so window length no longer decides audibility for an agent
    that cannot walk any closer -- and this fixture's wall is what stops it walking. See
    ``_out_of_earshot_episode``. The censoring KEYS are what this class is about, and they
    are unchanged.
    """

    @classmethod
    def setUpClass(cls):
        cls.censored, cls.cfg = _out_of_earshot_episode()
        cls.heard, _ = _out_of_earshot_episode(sounding_steps=5, gain=0.5)

    def test_the_censored_arm_is_out_of_earshot_BY_LEVEL_and_not_by_the_fill_ramp(self):
        """The premise of every other test here, asserted instead of assumed.

        Before ADR-0019 this fixture was censored because the 5 s analysis window filled
        over ``clip_ramp_steps`` folds, not because the source was quiet: its settled
        level was 1.9x the threshold. A fixture whose premise is an artefact of the
        readout is a fixture that goes green for the wrong reason, so the level itself is
        pinned here -- the loudest loop phase, mixed with the bed exactly as the runner
        composes it, must still sit under ``onset_rms``.
        """
        window = self.censored.audit.sounding_window
        source = self.censored.audit.source_xyz
        world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
        handle = FakeAudioSensorHandle(world, source, gain=0.1)
        phases = steady_state_cue_rms(
            handle.audio_of({}), CLIP, hop=int(window.hop_samples)
        )
        loudest = max(phases)
        with_bed = math.sqrt(loudest ** 2 + CALIBRATION.bed_rms ** 2)
        self.assertLess(with_bed, CALIBRATION.onset_rms, phases)
        # ...and the LOUD arm is over it on its own first fold, which is what makes the
        # pair a comparison rather than two quiet episodes.
        loud = FakeAudioSensorHandle(world, source, gain=0.5)
        loud_phases = steady_state_cue_rms(
            loud.audio_of({}), CLIP, hop=int(window.hop_samples)
        )
        self.assertGreater(
            math.sqrt(max(loud_phases) ** 2 + CALIBRATION.bed_rms ** 2),
            CALIBRATION.onset_rms,
        )

    def test_the_flag_is_written_on_both_arms_so_a_reader_can_take_a_rate(self):
        """ALWAYS present. A key that appears only on the censored episodes cannot be
        summed into a denominator, and the absence would then mean two things."""
        self.assertEqual(self.censored.audit.metrics["onset_delay_censored"], 1.0)
        self.assertEqual(self.heard.audit.metrics["onset_delay_censored"], 0.0)
        self.assertEqual(self.censored.audit.metrics["onset_fired"], 0.0)
        self.assertEqual(self.heard.audit.metrics["onset_fired"], 1.0)

    def test_the_censoring_time_is_the_sounding_steps_the_episode_actually_got(self):
        """Not the window's planned length: an episode can end mid-window, and then the
        delay is known to exceed what it RAN rather than what it was promised."""
        metrics = self.censored.audit.metrics
        sounding = [row.step for row in self.censored.audit.steps if row.source_playing]
        self.assertEqual(metrics["onset_delay_censored_at_steps"], float(len(sounding)))
        self.assertEqual(metrics["onset_delay_censored_at_steps"], 1.0)
        self.assertNotIn("onset_delay_steps", metrics)

    def test_the_distance_is_recorded_in_both_route_and_straight_line_metres(self):
        """*Would a longer window have been enough* is a question about WALKING, so the
        route is the axis. The two come apart exactly where it matters: at 5-12 m the
        agent is usually in another room, where ``xz`` shrinks while the walk does not."""
        metrics = self.censored.audit.metrics
        self.assertEqual(metrics["min_d2source_in_window_m"], 11.5)
        self.assertEqual(metrics["min_route_to_source_in_window_m"], 11.5)
        # ...and it is the minimum over the SOUNDING steps only, not over the episode:
        # how close the agent got before the source stopped is the whole question.
        self.assertLess(
            min(
                row.position.horizontal_distance_to(self.censored.audit.source_xyz)
                for row in self.censored.audit.steps
            ),
            metrics["min_d2source_in_window_m"],
        )

    def test_an_unroutable_source_records_no_route_rather_than_a_zero(self):
        """THE FORCED-FAILURE ARM. 23 of ``yield-2``'s 365 episodes have no navmesh route
        to their source at all, and a 0.0 there would read as *the agent was standing on
        it* in the distribution the window duration is chosen from."""
        world = _UnroutableSourceWorld(
            start=Xyz(0.0, 0.0, 0.0),
            yaw=0.0,
            wall=lambda p: p.z < -1.0,
            unroutable=Xyz(0.0, 0.0, -12.0),
        )
        result, _cfg = _out_of_earshot_episode(world=world)
        metrics = result.audit.metrics
        self.assertEqual(metrics["onset_delay_censored"], 1.0)
        self.assertEqual(metrics["min_d2source_in_window_m"], 11.5)
        self.assertNotIn("min_route_to_source_in_window_m", metrics)
        self.assertTrue(
            all(row.geodesic_to_source is None for row in result.audit.steps)
        )

    def test_a_heard_episode_carries_the_delay_and_not_the_censoring_keys(self):
        """The two are mutually exclusive by construction, which is what lets a reader
        pool them: ``sum(heard_within_window) / count(sounding_window_closed)`` is the
        rate, and this key names the episodes missing from the numerator."""
        metrics = self.heard.audit.metrics
        self.assertIn("onset_delay_steps", metrics)
        self.assertNotIn("onset_delay_censored_at_steps", metrics)
        self.assertNotIn("min_d2source_in_window_m", metrics)
        self.assertNotIn("min_route_to_source_in_window_m", metrics)


class TestTheTallyRefusesToStateFewerActiveTailsThanItCounts(unittest.TestCase):
    """ADR-0017's bar carried as a FIELD, and the difference between a rule and a habit.

    The bar used to live at the two call sites that happen to exist today --
    ``silent_phase_tally`` and ``run_episode`` -- while the type's constructor took bare
    counts and asked nothing. A later cross-run aggregator walking ``audit.json`` files
    and building one of these would have published an SWS having never asked whether a
    single accumulator folded a render, and nothing would have stopped it.

    ``None`` is deliberately NOT refused: the value is also built by hand -- by the
    summary printer, by a reader who has counts and no records -- and forcing a fabricated
    number there buys a false claim rather than a check. So an UNVERIFIED SWS is legible
    as unverified and an UNVERIFIABLE one raises.
    """

    def _tally(self, **overrides):
        base = dict(
            n_episodes=4,
            n_window_closed=4,
            n_reached_after_offset=1,
            n_source_reached=2,
        )
        base.update(overrides)
        return SilentPhaseTally(**base)

    def test_a_stated_shortfall_cannot_be_constructed_at_all(self):
        """Refused in ``__post_init__`` rather than in ``sws``, because a value that
        cannot be published is a value that should not exist -- ``as_dict``, ``summary``
        and every future reader go through this constructor, and a property-side check
        would have to be repeated in each of them."""
        with self.assertRaises(TailNotActiveError) as caught:
            self._tally(n_tail_active=3)
        self.assertIn("1 of the 4", str(caught.exception))
        self.assertIn("hard cut", str(caught.exception))
        # ...and the healthy arm, so this is not a constructor that always refuses.
        self.assertAlmostEqual(self._tally(n_tail_active=4).sws, 0.25)

    def test_a_verified_tally_says_measured_and_an_unverified_one_says_so(self):
        verified = self._tally(n_tail_active=4).as_dict()
        self.assertEqual(verified["sws_status"], "measured")
        self.assertEqual(verified["n_tail_active"], 4)

        bare = self._tally().as_dict()
        self.assertAlmostEqual(bare["sws"], 0.25)
        self.assertEqual(bare["sws_status"], "measured_tail_unverified")
        self.assertIsNone(bare["n_tail_active"])

    def test_the_printed_line_carries_the_same_words_as_the_artefact(self):
        """An operator reading the console and a reader parsing ``summary.json`` must not
        be told different things about the same run."""
        from earshot.task.runner import RunSummary

        def line(tally):
            summary = RunSummary(
                run_dir="runs/x",
                scene_label="FAKE",
                n_episodes=4,
                funnel={stage.name: 0 for stage in FunnelStage},
                silent_phase=tally,
            )
            return next(
                row for row in summary.summary().splitlines() if row.startswith("SWS")
            )

        self.assertIn("(tail unverified)", line(self._tally()))
        self.assertNotIn("(tail unverified)", line(self._tally(n_tail_active=4)))

    def test_the_run_level_tally_always_fills_it_because_it_is_the_thing_that_asked(self):
        """``silent_phase_tally`` reads the records, so its answer is never unverified.
        A tally assembled anywhere else says the opposite, which is the point."""
        result, _cfg = _windowed_episode(sounding_steps=6)
        tally = silent_phase_tally([result.audit])
        self.assertTrue(tally.tail_verified)
        self.assertEqual(tally.n_tail_active, tally.n_window_closed)
        self.assertEqual(tally.as_dict()["sws_status"], "measured")


class _StoppedAfterThePreflight(RuntimeError):
    """The stub ``run()`` trips over once the accumulator's preflight is behind it.

    ``write_env_report`` is the first thing ``run()`` does after the preflight and the
    last thing before ``from earshot.sim.world import World``, so raising there stops the
    function at a known line rather than at whatever a Mac's missing habitat-sim happens
    to raise -- which is a different exception on a machine that has one.
    """


def _run_to_the_preflight(cfg, clip, clip_path, said, **run_kwargs):
    """Drive ``run()`` as far as the accumulator's preflight, with nothing real behind it.

    Everything ``run()`` reaches for BEFORE the preflight is stubbed and nothing after it
    is: the point of the preflight is that a config typo costs seconds rather than a
    scene load, a CLAP load and 16 calibration renders per episode, and a test that had
    to build any of those would not be able to say so.

    ``run_kwargs`` forwards straight to ``runner.run`` -- ``memory_condition`` and
    ``memory_prior_stores`` are constructed and validated BEFORE ``build_anomaly_episodes``
    is even called, so this same stub reaches that code too.
    """
    import earshot.task.runner as runner

    class _Env(object):
        def summary(self):
            return "env: stubbed"

        def as_dict(self):
            return {"probes": []}

    class _Dataset(object):
        scene_label = "FAKE"
        scene_path = "/nonexistent/FAKE.basis.glb"

    class _Build(object):
        episodes = ()
        skipped = ()

        def summary(self):
            return "build: stubbed"

    def _stop(*args, **kwargs):
        raise _StoppedAfterThePreflight("the preflight is behind us")

    patched = {
        "assert_env": lambda clap=False: _Env(),
        "find_split_dir": lambda split, root=None: "/nonexistent/split",
        "find_scenes_dir": lambda root=None: "/nonexistent/scenes",
        "_pick_scene": lambda split_dir, scenes_dir, scene: _Dataset(),
        "build_anomaly_episodes": lambda *args, **kwargs: _Build(),
        "resolve_anomaly_clip": lambda *args, **kwargs: clip_path,
        "load_anomaly_clip": lambda *args, **kwargs: clip,
        "write_env_report": _stop,
    }
    saved = {name: getattr(runner, name) for name in patched}
    for name, stub in patched.items():
        setattr(runner, name, stub)
    try:
        runner.run(cfg, progress=said.append, **run_kwargs)
    finally:
        for name, real in saved.items():
            setattr(runner, name, real)


class TestTheAccumulatorsOneConfigurationRefusal(unittest.TestCase):
    """``run()``'s preflight -- the only part of that function a Mac can execute at all.

    Left to ``open_tail`` inside ``run_episode`` this raised PER EPISODE, after habitat,
    CLAP, the scene and 16 calibration renders had all been paid for, and it raised in the
    ACCUMULATOR's words -- samples, a read window, a sensor -- about a mistake made in
    ``AudioConfig``. A config typo should cost seconds and name the field it is in.
    """

    CLIP_PATH = "/data/anomaly_audio/alarm.wav"

    def test_a_step_that_outruns_the_clip_is_refused_in_seconds_and_names_the_field(self):
        """THE FORCED FAILURE. ``step_seconds`` 6 against a 5 s clip: consecutive read
        windows share no samples, so everything between two steps is dropped -- a
        different sensor, which has to be asked for rather than fallen into."""
        cfg = make_config(audio=AudioConfig(step_seconds=6.0))
        clip = synthetic_burst(cfg.audio.sample_rate, seconds=5.0)
        said = []
        with self.assertRaises(ValueError) as caught:
            _run_to_the_preflight(cfg, clip, self.CLIP_PATH, said)
        message = str(caught.exception)
        for fragment in (
            "AudioConfig.step_seconds",
            "264600",
            "220500",
            self.CLIP_PATH,
            "Set step_seconds below",
        ):
            self.assertIn(fragment, message)
        # ...in SECONDS, which is the domain the mistake was made in. The accumulator's
        # own refusal names neither the field nor a second.
        self.assertIn("6 s", message)
        self.assertIn("5 s", message)
        self.assertNotIn("reverb tail:", "".join(said))

    def test_a_workable_configuration_gets_past_it_and_builds_the_buffer(self):
        """THE HEALTHY ARM. A refusal nobody can get past reads as a broken run.

        The bound is stated by the comparison and PROVED by the construction: ADR-0014's
        rule is that a capability is exercised, so ``open_tail`` is called anyway and the
        three numbers an operator needs to read a window duration in STEPS are printed.
        """
        import earshot.task.runner as runner

        cfg = make_config(audio=AudioConfig(step_seconds=1.0))
        clip = synthetic_burst(cfg.audio.sample_rate, seconds=5.0)
        said = []
        built = []
        real_open = runner.open_tail

        def spy_open_tail(*args, **kwargs):
            built.append(kwargs)
            return real_open(*args, **kwargs)

        runner.open_tail = spy_open_tail
        try:
            with self.assertRaises(_StoppedAfterThePreflight):
                _run_to_the_preflight(cfg, clip, self.CLIP_PATH, said)
        finally:
            runner.open_tail = real_open
        self.assertEqual(
            built,
            [{"window": len(clip), "hop": 44100}],
            "the buffer was described and not built, which is the shape of a probe that "
            "skipped and reported success",
        )
        line = next(row for row in said if row.startswith("reverb tail:"))
        self.assertIn("hop 44100 samples", line)
        self.assertIn("read window 220500", line)
        self.assertIn("ramp 5 steps", line)

    def test_the_boundary_is_the_clip_length_itself_and_it_is_exclusive(self):
        """A hop of exactly the read window already makes consecutive readouts disjoint,
        so the refusal is ``>=`` and not ``>``."""
        clip = synthetic_burst(44100, seconds=5.0)
        exact = make_config(audio=AudioConfig(step_seconds=5.0))
        with self.assertRaises(ValueError):
            _run_to_the_preflight(exact, clip, self.CLIP_PATH, [])
        under = make_config(audio=AudioConfig(step_seconds=4.9))
        with self.assertRaises(_StoppedAfterThePreflight):
            _run_to_the_preflight(under, clip, self.CLIP_PATH, [])


def _semantic_entry(sound_class, category, embedding=(1.0, 0.0)):
    return SemanticEntry(
        sound_class=sound_class,
        room="bedroom",
        category=category,
        embedding=list(embedding),
        donor_scene="donor",
    )


class TestTheMatrixCellIsBuiltBeforeTheSimulator(unittest.TestCase):
    """``run()``'s new memory wiring, on the same seam as the preflight tests above:
    ``memory_condition``/``memory_prior_stores`` are validated and the cell's
    ``MemoryContext`` is built right after ``_pick_scene`` -- before
    ``build_anomaly_episodes``, before the simulator -- so a Mac can exercise all of it.
    """

    CLIP_PATH = "/data/anomaly_audio/alarm.wav"

    def _cfg_and_clip(self):
        cfg = make_config(anomaly_class="alarm")
        clip = synthetic_burst(cfg.audio.sample_rate, seconds=5.0)
        return cfg, clip

    def test_a_condition_with_no_stores_raises_before_anything_is_built(self):
        """THE FORCED FAILURE. A cell selected with nothing to realise it must not run
        silently under `MemoryCondition.NONE` -- that is a matrix cell that looks
        populated and measures nothing, the exact shape this repo has paid for twice."""
        cfg, clip = self._cfg_and_clip()
        with self.assertRaises(ValueError) as caught:
            _run_to_the_preflight(
                cfg, clip, self.CLIP_PATH, [],
                memory_condition=MemoryCondition.HEARD_SEEN,
            )
        message = str(caught.exception)
        self.assertIn("memory_prior_stores", message)
        self.assertIn("heard_seen", message)

    def test_a_condition_with_stores_reaches_the_identical_stopping_point(self):
        """THE HEALTHY ARM. Memory construction must not itself change where the
        function stops -- it reaches the same preflight the no-memory tests reach."""
        cfg, clip = self._cfg_and_clip()
        stores = (SemanticStore(entries=(_semantic_entry("alarm", "bed"),)), EpisodicStore())
        with self.assertRaises(_StoppedAfterThePreflight):
            _run_to_the_preflight(
                cfg, clip, self.CLIP_PATH, [],
                memory_condition=MemoryCondition.HEARD_SEEN,
                memory_prior_stores=stores,
            )

    def test_the_condition_filters_the_semantic_store_before_the_context_is_built(self):
        """`stores_for_cell` runs INSIDE `run()`, not left to the caller: a
        `NOT_HEARD_*` condition must not still carry the run's own class."""
        import earshot.task.runner as runner_module

        cfg, clip = self._cfg_and_clip()
        semantic = SemanticStore(entries=(
            _semantic_entry("alarm", "bed", (1.0, 0.0)),
            _semantic_entry("snoring", "bed", (0.0, 1.0)),
        ))
        captured = {}
        real_context = runner_module.MemoryContext

        def spy_context(*args, **kwargs):
            captured["semantic"] = kwargs.get("semantic")
            return real_context(*args, **kwargs)

        runner_module.MemoryContext = spy_context
        try:
            with self.assertRaises(_StoppedAfterThePreflight):
                _run_to_the_preflight(
                    cfg, clip, self.CLIP_PATH, [],
                    memory_condition=MemoryCondition.NOT_HEARD_SEEN,
                    memory_prior_stores=(semantic, EpisodicStore()),
                )
        finally:
            runner_module.MemoryContext = real_context
        self.assertEqual(
            sorted(entry.sound_class for entry in captured["semantic"].entries),
            ["snoring"],
        )

    def test_a_toured_categorys_own_point_reaches_the_context_over_ground_truth(self):
        """THE SEEN AXIS ITSELF. Without `points_by_category_for_cell` this test cannot
        distinguish `HEARD_SEEN` from `HEARD_UNSEEN` at all -- see that module's own
        tests for the isolated claim; this asserts `run()` actually calls it."""
        import earshot.task.runner as runner_module

        cfg, clip = self._cfg_and_clip()
        semantic = SemanticStore(entries=(_semantic_entry("alarm", "bed"),))
        episodic = EpisodicStore(entries=(
            EpisodicEntry(scene="FAKE", room="bedroom", category="bed",
                          point=Xyz(1.0, 0.0, 0.0)),
        ))
        captured = {}
        real_context = runner_module.MemoryContext

        def spy_context(*args, **kwargs):
            captured["points_by_category"] = kwargs.get("points_by_category")
            return real_context(*args, **kwargs)

        runner_module.MemoryContext = spy_context
        try:
            with self.assertRaises(_StoppedAfterThePreflight):
                _run_to_the_preflight(
                    cfg, clip, self.CLIP_PATH, [],
                    memory_condition=MemoryCondition.HEARD_SEEN,
                    memory_prior_stores=(semantic, episodic),
                )
        finally:
            runner_module.MemoryContext = real_context
        self.assertEqual(captured["points_by_category"]["bed"], (Xyz(1.0, 0.0, 0.0),))

    def test_the_disclosure_is_printed_when_the_arm_is_live(self):
        cfg, clip = self._cfg_and_clip()
        said = []
        stores = (SemanticStore(entries=(_semantic_entry("alarm", "bed"),)), EpisodicStore())
        with self.assertRaises(_StoppedAfterThePreflight):
            _run_to_the_preflight(
                cfg, clip, self.CLIP_PATH, said,
                memory_condition=MemoryCondition.HEARD_SEEN,
                memory_prior_stores=stores,
            )
        self.assertTrue(any(RUN_DISCLOSURE in line for line in said))

    def test_no_condition_at_all_is_byte_identical_to_the_pre_matrix_behaviour(self):
        """Every non-matrix caller passes nothing at all, and that path is untouched:
        the same `said` transcript either way."""
        cfg, clip = self._cfg_and_clip()
        said_bare, said_explicit_none = [], []
        with self.assertRaises(_StoppedAfterThePreflight):
            _run_to_the_preflight(cfg, clip, self.CLIP_PATH, said_bare)
        with self.assertRaises(_StoppedAfterThePreflight):
            _run_to_the_preflight(
                cfg, clip, self.CLIP_PATH, said_explicit_none, memory_condition=None
            )
        self.assertEqual(said_bare, said_explicit_none)


def _detour_episode(world=None, **cfg_overrides):
    """The `TestTheFullLoop` fixture as a function: source 5 m ahead, goal 9 m ahead.

    Returned rather than shared on a class, because every arm below has to run the SAME
    task under a DIFFERENT config, and a `setUpClass` result cannot be re-run.
    """
    source = Xyz(0.0, 0.0, -5.0)
    if world is None:
        world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
    handle = FakeAudioSensorHandle(world, source)
    anomaly_episode = make_anomaly_episode(
        source=source,
        episode=make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))]),
        t_anom=2,
    )
    cfg = make_config(**cfg_overrides)
    return run(world, handle, anomaly_episode, cfg, calibration=CALIBRATION)


class TestTheSourceSideMetrics(unittest.TestCase):
    """SPL and distance-to-goal taken against the SOURCE, which nothing carried.

    Both arms per ADR-0014: an episode that had a route to its source writes the keys,
    and an episode with no route writes NOTHING rather than a zero. The second is the
    one that matters -- 23 of `yield-2`'s 365 episodes are unwinnable, and scoring them
    0.0 puts them in the same bucket as an episode that had a route and failed to walk
    it.
    """

    @classmethod
    def setUpClass(cls):
        cls.result = _detour_episode()

    def test_the_reach_is_scored_against_the_source_and_not_the_primary_stop(self):
        metrics = self.result.audit.metrics
        self.assertEqual(metrics["source_find_sr_1m"], 1.0)
        self.assertAlmostEqual(metrics["source_spl"], 1.0)
        # `compute_benchmark_spl` could not have produced this: the primary `stopped`
        # flag is what gates it, and the source is reached mid-episode with the primary
        # task still running.
        self.assertIsNotNone(self.result.audit.source_reached_step)
        print("source SPL {:.3f}, Find-SR@1m {:.0f}, reached at step {}".format(
            metrics["source_spl"], metrics["source_find_sr_1m"],
            self.result.audit.source_reached_step))

    def test_the_final_distance_is_a_different_number_from_the_closest_approach(self):
        """The conflation this key exists to prevent, as an assertion.

        `min_d2source_m` is a MINIMUM over the episode; `dtg_source_final_m` is the
        distance at its end. This episode walks to the source and then resumes toward a
        goal 4 m past it, so the two differ by metres.
        """
        metrics = self.result.audit.metrics
        self.assertIn("dtg_source_final_m", metrics)
        self.assertIn("min_d2source_m", metrics)
        self.assertGreater(metrics["dtg_source_final_m"], metrics["min_d2source_m"])
        print("closest approach {:.2f} m, final route to source {:.2f} m".format(
            metrics["min_d2source_m"], metrics["dtg_source_final_m"]))

    def test_the_source_path_length_is_the_one_at_the_reach_and_not_the_episodes(self):
        """`L_taken` stops at the reach. If the whole episode's `path_len_m` were used,
        the SPL would fall as the primary search walked on after the source was already
        found -- which would make the number a function of the OTHER task."""
        metrics = self.result.audit.metrics
        # The agent kept walking after the reach, so the two lengths differ.
        self.assertGreater(metrics["path_len_m"], 0.0)
        # SPL 1.0 is only reachable if `L_taken` was the length at the reach: the whole
        # episode's path is longer than the start-to-source route.
        self.assertAlmostEqual(metrics["source_spl"], 1.0)

    def test_an_episode_with_no_route_to_its_source_writes_no_spl_and_no_zero(self):
        """The forced-failure arm. ABSENT, and absent is not 0.0 and not a failed find."""
        world = _UnroutableSourceWorld(
            start=Xyz(0.0, 0.0, 0.0), yaw=0.0, unroutable=Xyz(0.0, 0.0, -5.0)
        )
        metrics = _detour_episode(world=world).audit.metrics

        self.assertNotIn("source_spl", metrics)
        self.assertNotIn("source_find_sr_1m", metrics)
        self.assertNotIn("dtg_source_final_m", metrics)
        # The control, on the same task with a routable source: the keys ARE written, so
        # their absence above is the missing route and not a broken call site.
        self.assertIn("source_spl", self.result.audit.metrics)
        print("unroutable source: source_spl / source_find_sr_1m / dtg_source_final_m "
              "all ABSENT, and the routable control has all three")


class TestTheAblationArmsReachTheLoop(unittest.TestCase):
    """ADR-0018's four arms, threaded from `RunConfig` into the loop.

    Each is asserted on the OUTCOME and not on the stored flag: an arm that is recorded
    and never read is the shape of failure this tree keeps paying for.
    """

    def test_the_defaults_are_recorded_on_every_episode(self):
        """`summary.json` holds the run config once per RUN and every comparison here is
        per EPISODE, so the arm has to be on the audit or a paired diff cannot check it
        was comparing like with like."""
        audit = _detour_episode().audit
        self.assertEqual(
            (audit.climb_rule, audit.lateral_cue, audit.cast_policy, audit.ir_policy),
            ("live", "live", "cast", "full"),
        )

    def test_a_selected_arm_is_recorded_as_selected(self):
        audit = _detour_episode(
            climb_rule=ClimbRule.OFF,
            lateral_cue=LateralCue.OFF,
            cast_policy=CastPolicy.SCAN_ONLY,
            ir_policy=IrPolicy.ANECHOIC,
        ).audit
        self.assertEqual(
            (audit.climb_rule, audit.lateral_cue, audit.cast_policy, audit.ir_policy),
            ("off", "off", "scan_only", "anechoic"),
        )

    def test_the_controller_arms_off_actually_change_what_the_episode_reaches(self):
        """The OFF arm has to change the OUTCOME, not just the record. With the climb,
        the lateral cue and the cast all dead the agent has nothing left to localize
        with, so the detour is entered and never converts."""
        live = _detour_episode()
        off = _detour_episode(
            climb_rule=ClimbRule.OFF,
            lateral_cue=LateralCue.OFF,
            cast_policy=CastPolicy.SCAN_ONLY,
        )

        self.assertEqual(live.audit.funnel_stage, FunnelStage.PRIMARY_RESUMED)
        self.assertIsNotNone(live.audit.source_reached_step)
        self.assertEqual(off.audit.funnel_stage, FunnelStage.INVESTIGATE_ENTERED)
        self.assertIsNone(off.audit.source_reached_step)
        # It had a route the whole time, so this zero is the real, reportable kind.
        self.assertEqual(off.audit.metrics["source_spl"], 0.0)
        self.assertEqual(off.audit.metrics["source_find_sr_1m"], 0.0)
        print("arms LIVE: reached at step {}   arms OFF: never reached (funnel {})".format(
            live.audit.source_reached_step, off.audit.funnel_stage.name))

    def test_the_anechoic_policy_reaches_the_render_and_shortens_the_room(self):
        """`IrPolicy` is applied where the IR leaves the sensor, so what changes is the
        AUDIO and not a config value. The accumulator's own measurement of how long the
        room stays audible past a step is the readout: a `(2, 1)` IR has no tail to
        outlive one."""
        full = _detour_episode().audit.sounding_window
        anechoic = _detour_episode(ir_policy=IrPolicy.ANECHOIC).audit.sounding_window

        self.assertGreater(full.cue_tail_steps, 1)
        self.assertEqual(anechoic.cue_tail_steps, 1)
        self.assertLess(anechoic.max_ir_samples, full.max_ir_samples)
        print("cue_tail_steps FULL {} -> ANECHOIC {}   max_ir_samples {} -> {}".format(
            full.cue_tail_steps, anechoic.cue_tail_steps,
            full.max_ir_samples, anechoic.max_ir_samples))

    def test_the_calibration_sweep_takes_the_same_ir_path_as_the_loop(self):
        """ADR-0017's rule, on the arm that could break it. `calibrate_episode` is run
        here rather than injected, so the sweep renders through `ir_under_policy` too --
        an `onset_rms` derived through the real IR and applied to an anechoic reading
        would be a threshold in a domain the loop does not run in."""
        source = Xyz(0.0, 0.0, -5.0)

        def sweep(cfg):
            # A FRESH world per sweep. `FakeWorld.random_navigable_point` is seeded by
            # its own call count, so a second sweep against the same world draws a
            # different band of poses and the two arms would differ for that reason
            # rather than for the IR's.
            world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
            handle = FakeAudioSensorHandle(world, source)
            return calibrate_episode(world, handle, source, CLIP, cfg)[0]

        swept = sweep(make_config(ir_policy=IrPolicy.ANECHOIC))
        control = sweep(make_config())

        self.assertNotAlmostEqual(swept.onset_rms, control.onset_rms)
        print("onset_rms swept through ANECHOIC {:.6g} vs FULL {:.6g}".format(
            swept.onset_rms, control.onset_rms))


if __name__ == "__main__":
    unittest.main(verbosity=2)
