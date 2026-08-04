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

import math
import unittest

from _interpreter import assert_interpreter  # noqa: F401
from _task_fakes import (
    FakeAudioSensorHandle,
    FakeWorld,
    make_anomaly_episode,
    make_episode,
    make_goal,
)

from earshot.agent.config import ControllerConfig
from earshot.agent.proposers import SOURCE_INVESTIGATE
from earshot.agent.reachability import EmptyPoolError
from earshot.audio.bed import bed_signal
from earshot.audio.calibration import CalibrationResult
from earshot.audio.clips import synthetic_burst
from earshot.audio.lateral import LATERAL_AMBIGUOUS, bearing_lateral_sign
from earshot.config import Detector, Localization, RunConfig
from earshot.report.agent import SCHEMA_FIELDS
from earshot.report.audit import FunnelStage
from earshot.task.runner import (
    DIVERT_CANDIDATE_ID,
    _divert_candidate,
    _funnel_stage,
    calibration_poses,
    make_detector,
    run_episode,
)
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
    )
    base.update(overrides)
    return RunConfig(**base)


def run(world, handle, anomaly_episode, cfg, **kwargs):
    kwargs.setdefault("detector", make_detector(cfg, world, anomaly_episode))
    return run_episode(world, handle, anomaly_episode, cfg, clip=CLIP, **kwargs)


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
        """
        self.assertIsNotNone(self.result.audit.dist_at_stop)
        self.assertLess(self.result.audit.dist_at_stop, 1.0)
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
        """
        pre_onset = [row for row in self.result.audit.steps if not row.source_playing]
        self.assertEqual(len(pre_onset), self.anomaly_episode.t_anom)
        for row in pre_onset:
            self.assertAlmostEqual(row.measured_rms, self.cfg.audio.bed_rms, places=6)

    def test_the_onset_fired_at_or_after_t_anom_and_the_provenance_was_asserted(self):
        onset = self.result.audit.onset
        self.assertIsNotNone(onset.onset_step)
        self.assertGreaterEqual(onset.onset_step, self.anomaly_episode.t_anom)
        self.assertEqual(onset.n_pre_onset_readings, self.anomaly_episode.t_anom)
        # Ticket 16's discipline: the record says the assertion RAN, because an artefact
        # that exists at all looks like proof it passed.
        self.assertTrue(onset.provenance_asserted)
        self.assertEqual(self.result.report.heard_at_step, onset.onset_step)

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
        """The whole point of the cue: a climb that starts backwards recovers."""
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
