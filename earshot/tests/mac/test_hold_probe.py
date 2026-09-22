"""The hold probe's logic, against a fake world whose renderer the test controls.

What a Mac can settle: that the poses are the scans ``by_scan`` reads, one per episode;
that the heading rebuild agrees with the frame ``audio/lateral.py`` states; that each
sequence renders what its docstring says through the run's own ``heard_step``, in a
World of its own; and that the readout and the branch read what the module docstring
pre-registers. What it cannot settle is how the real renderer behaves, which is the whole
question, and whether habitat moves the agent the way the fake does: the box checks every
replayed position against the record.

ADR-0014's two arms throughout. A renderer that fades must read FALLS and a fixed one
must not; a replay that leaves the recorded path must not be read; the FORCED checks must
see their own fall on a renderer that holds.
"""

import contextlib
import dataclasses
import json
import math
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

import numpy as np
from _interpreter import assert_interpreter  # noqa: F401
from test_leg_replay import FOLDS, PRE, RING, THREE_LEGS, episode

from earshot.agent.controller import ACT_FORWARD, ACT_TURN_LEFT, ACT_TURN_RIGHT
from earshot.audio.bed import bed_signal
from earshot.audio.config import AudioConfig
from earshot.config import RunConfig
from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_env_report, write_episode, write_run_summary
from earshot.tools.hold_probe import (
    ARM_ORDER,
    EXCLUDED_DUPLICATE,
    EXCLUDED_NO_HEADING,
    EXCLUDED_SHORT,
    FALLS,
    FLAT,
    FORCED_DECAY,
    HEADING,
    MIN_POSES,
    MIXED,
    NOT_RUN,
    PIPELINE,
    PRESET,
    RENDERER,
    RISES,
    SELECTION,
    SEQUENCE_KINDS,
    UNREAD,
    PoseResult,
    ProbeIO,
    ProbePose,
    RenderConfig,
    arm_header,
    audio_config_of,
    decide,
    format_readout,
    main,
    plan_pose,
    probe_scene,
    read_tag,
    readout,
    scene_payload,
    select_poses,
    trace_hold,
    trace_rescan,
    trace_teleport,
    trend,
    yaw_rotation,
)
from earshot.tools.leg_replay import SCAN_READINGS, episode_legs, same_phase_change
from earshot.types import Xyz

HOP = 100
CLIP = np.random.default_rng(7).standard_normal(5 * HOP).astype(np.float32)  # 5 folds
BED = bed_signal(HOP, 1e-3)
SHAPE = np.exp(-np.arange(60) / 12.0).astype(np.float32)
IR = np.stack([SHAPE, 0.8 * SHAPE])
WALK_IN = PRE  # the fixture's three SEARCH steps before the detour
HOLD = 20
LOOP_FALL = 1.0 - FORCED_DECAY ** FOLDS
RENDER_CONFIG = RenderConfig(audio=dataclasses.asdict(AudioConfig()), anomaly_class="alarm",
                             anomaly_clip=None, split="val")


class FakeWorld:
    """Habitat's motion model and a renderer the test writes.

    ``gain(n, yaw)`` scales one fixed IR at the ``n``-th render at heading ``yaw``. A
    forward moves ``step`` metres along ``(-sin yaw, 0, -cos yaw)`` and a turn is 30
    degrees, left positive, which is the frame ``audio/lateral.py`` states.
    """

    def __init__(self, gain=lambda n, yaw: 1.0, step=0.25):
        self.gain, self.step = gain, step
        self.position, self.yaw, self.renders = Xyz(0.0, 0.0, 0.0), 0.0, 0
        self.seats, self.sources = [], []

    def seat(self, position, yaw):
        self.position, self.yaw = position, yaw
        self.seats.append((position, yaw))

    def render(self):
        self.renders += 1
        return IR * np.float32(self.gain(self.renders, self.yaw))

    def act(self, action):
        if action == ACT_FORWARD:
            self.position = Xyz(self.position.x - self.step * math.sin(self.yaw),
                                self.position.y,
                                self.position.z - self.step * math.cos(self.yaw))
        elif action == ACT_TURN_LEFT:
            self.yaw += math.radians(30.0)
        elif action == ACT_TURN_RIGHT:
            self.yaw -= math.radians(30.0)

    def io(self):
        return ProbeIO(seat=self.seat, render=self.render, act=self.act,
                       where=lambda: self.position, place_source=self.sources.append)


def factory(make_world=FakeWorld, opened=None):
    """``probe_scene``'s ``fresh``: a new world per call, each recorded in ``opened``."""
    @contextlib.contextmanager
    def fresh():
        world = make_world()
        if opened is not None:
            opened.append(world)
        yield world.io()
    return fresh


def walked_audit(index=0, *, collided=(False, False, False), turn_first=False):
    """leg_replay's scan fixture, with a walk-in the agent took to reach the scan.

    The fixture's scan stands at x = 4 from its fourth step on. The walk-in is its three
    SEARCH steps: three clean forwards along -z (yaw 0) that end there, or with
    ``turn_first`` a left turn and then two forwards at yaw +30 degrees. The detour's
    actions are what the rule answered, as the runner records when the follower obeys.
    """
    audit = episode(THREE_LEGS, index=index, offset_at=30, loop=RING, folds=FOLDS)
    rows = list(audit.steps)
    scan_at = rows[PRE].position
    if turn_first:
        heading = math.radians(30.0)
        dx, dz = -0.25 * math.sin(heading), -0.25 * math.cos(heading)
        walk = [Xyz(scan_at.x - 2 * dx, 0.0, scan_at.z - 2 * dz)] * 2 + [
            Xyz(scan_at.x - dx, 0.0, scan_at.z - dz)]
        actions = [ACT_TURN_LEFT, ACT_FORWARD, ACT_FORWARD]
    else:
        walk = [Xyz(scan_at.x, 0.0, scan_at.z + 0.25 * (PRE - k)) for k in range(PRE)]
        actions = [ACT_FORWARD] * PRE
    for k in range(PRE):
        rows[k] = replace(rows[k], position=walk[k], action=actions[k],
                          collided=collided[k], displacement_m=0.25)
    for k in range(PRE, len(rows)):
        rows[k] = replace(rows[k], action=rows[k].realizable_action, collided=False)
    return replace(audit, steps=tuple(rows), source_class="alarm")


def first_scan(audit):
    return episode_legs(audit, run="tag/full", scene="AAAscene").scans[0]


def a_pose(**overrides):
    pose, why = plan_pose(walked_audit(), first_scan(walked_audit()), walk_in=WALK_IN)
    assert pose is not None, why
    return replace(pose, **overrides) if overrides else pose


def sequences(make_world=FakeWorld, pose=None, opened=None):
    (result,) = probe_scene([pose or a_pose()], factory(make_world, opened), clip=CLIP,
                            bed_cue=BED, hop=HOP, hold=HOLD, progress=lambda line: None)
    return result


def change(trace, length, series="cue"):
    values = list(getattr(trace, series))[trace.read_from:trace.read_from + length]
    return same_phase_change(values, lag=FOLDS)


class TestThePoseIsTheRecordedScan(unittest.TestCase):
    def test_the_pose_is_the_first_scan_by_scan_reads(self):
        scan = first_scan(walked_audit())
        self.assertTrue(scan.first and scan.complete and scan.static)
        pose = a_pose()
        self.assertEqual((pose.start_step, pose.period, pose.walk_in), (PRE, FOLDS, PRE))
        self.assertEqual(len(pose.positions), PRE + SCAN_READINGS)
        self.assertEqual(len(pose.actions), PRE + SCAN_READINGS - 1)
        self.assertAlmostEqual(pose.recorded_change, scan.change)
        self.assertEqual(pose.source_class, "alarm")

    def test_a_clean_forward_along_minus_z_is_yaw_zero(self):
        pose = a_pose()
        self.assertAlmostEqual(pose.seat_yaw, 0.0)
        self.assertAlmostEqual(pose.scan_yaw, 0.0)

    def test_a_turn_before_the_anchor_is_taken_off_the_seat(self):
        audit = walked_audit(turn_first=True)
        pose, _ = plan_pose(audit, first_scan(audit), walk_in=WALK_IN)
        self.assertAlmostEqual(pose.seat_yaw, 0.0)
        self.assertAlmostEqual(pose.scan_yaw, math.radians(30.0))

    def test_a_collided_forward_is_not_an_anchor(self):
        """Habitat slides a collided agent along the wall, so its step is no heading."""
        audit = walked_audit(collided=(True, False, False))
        pose, _ = plan_pose(audit, first_scan(audit), walk_in=WALK_IN)
        self.assertAlmostEqual(pose.seat_yaw, 0.0)
        audit = walked_audit(collided=(True, True, True))
        pose, why = plan_pose(audit, first_scan(audit), walk_in=WALK_IN)
        self.assertIsNone(pose)
        self.assertEqual(why, EXCLUDED_NO_HEADING)

    def test_a_scan_too_early_for_the_walk_in_is_excluded(self):
        audit = walked_audit()
        pose, why = plan_pose(audit, first_scan(audit), walk_in=WALK_IN + 1)
        self.assertIsNone(pose)
        self.assertEqual(why, EXCLUDED_SHORT)

    def test_the_pose_survives_a_round_trip(self):
        pose = a_pose()
        self.assertEqual(ProbePose.from_dict(json.loads(json.dumps(pose.as_dict()))), pose)


def write_arm(root, audits, *, tag="tag", run_config=None, env=True):
    arm = pathlib.Path(root) / tag / "full"
    scene = arm / "AAAscene"
    scene.mkdir(parents=True)
    for index, audit in enumerate(audits):
        write_episode(str(scene), index, AgentReport(resumed=True), audit)
    if env:
        config = run_config or RunConfig(run_dir=str(scene)).as_dict()
        write_env_report(str(scene), {"run_config": config})
    return str(arm)


class TestTheSelection(unittest.TestCase):
    def test_every_read_first_scan_is_a_pose_and_the_rest_are_counted(self):
        with tempfile.TemporaryDirectory() as root:
            arm = write_arm(root, [walked_audit(0), walked_audit(1, collided=(True,) * 3)])
            selection = select_poses([arm], walk_in=WALK_IN)
        self.assertEqual(selection.refusals, ())
        self.assertEqual(selection.n_first_scans, 2)
        self.assertEqual([p.episode for p in selection.poses], [0])
        self.assertEqual(dict(selection.excluded), {EXCLUDED_NO_HEADING: 1})
        self.assertEqual(selection.config.anomaly_class, "alarm")
        self.assertEqual(selection.config.split, "val")

    def test_the_same_episode_in_a_second_run_is_one_pose(self):
        """Two runs of one behaviour share their episodes; a pose is the sampling unit
        the sign test counts, so the second run's scan of a posed episode is left out."""
        with tempfile.TemporaryDirectory() as root:
            first = write_arm(root, [walked_audit(0)], tag="a")
            second = write_arm(root, [walked_audit(0)], tag="b")
            selection = select_poses([first, second], walk_in=WALK_IN)
        self.assertEqual([p.run for p in selection.poses], ["a/full"])
        self.assertEqual(dict(selection.excluded), {EXCLUDED_DUPLICATE: 1})

    def test_an_arm_that_is_not_full_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            arm = write_arm(root, [replace(walked_audit(), cast_policy="scan")])
            selection = select_poses([arm], walk_in=WALK_IN)
        self.assertEqual(len(selection.refusals), 1)
        self.assertIn("cast_policy", selection.refusals[0])

    def test_a_scene_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            arm = write_arm(root, [walked_audit()])
            selection = select_poses([arm + "/AAAscene"], walk_in=WALK_IN)
        self.assertIn("scene directory", selection.refusals[0])

    def test_a_run_with_no_recorded_configuration_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            selection = select_poses([write_arm(root, [walked_audit()], env=False)],
                                     walk_in=WALK_IN)
        self.assertIn("env_report.json is missing", selection.refusals[0])

    def test_a_zero_yield_scene_is_named_and_skipped(self):
        """`run()` writes a summary of 0 episodes and raises before `env_report.json`.
        hold-1 was refused on `mL8ThkuaVTM` for that missing file, in both runs."""
        with tempfile.TemporaryDirectory() as root:
            arm = write_arm(root, [walked_audit()])
            write_run_summary(str(pathlib.Path(arm) / "ZZZbarren"), {"n_episodes": 0})
            selection = select_poses([arm], walk_in=WALK_IN)
            out = StringIO()
            with redirect_stdout(out):
                status = main(["select", arm, "--walk-in", str(WALK_IN)])
        self.assertEqual(selection.refusals, ())
        self.assertEqual(len(selection.poses), 1)
        self.assertEqual(selection.zero_yield, ("tag/full/ZZZbarren",))
        self.assertEqual(status, 0)
        self.assertIn("zero-yield, no episode to probe: tag/full/ZZZbarren", out.getvalue())

    def test_a_summary_with_episodes_does_not_excuse_a_missing_configuration(self):
        with tempfile.TemporaryDirectory() as root:
            arm = write_arm(root, [walked_audit()], env=False)
            write_run_summary(str(pathlib.Path(arm) / "AAAscene"), {"n_episodes": 1})
            selection = select_poses([arm], walk_in=WALK_IN)
        self.assertEqual(selection.zero_yield, ())
        self.assertIn("env_report.json is missing", selection.refusals[0])

    def test_an_anechoic_run_is_refused(self):
        config = dict(RunConfig(run_dir="x").as_dict(), ir_policy="anechoic")
        with tempfile.TemporaryDirectory() as root:
            selection = select_poses([write_arm(root, [walked_audit()], run_config=config)],
                                     walk_in=WALK_IN)
        self.assertIn("ir_policy", selection.refusals[0])

    def test_two_configurations_are_refused(self):
        other = RunConfig(run_dir="x").as_dict()
        other["audio"] = dict(other["audio"], indirect_ray_count=2500)
        with tempfile.TemporaryDirectory() as root:
            first = write_arm(root, [walked_audit(0)], tag="a")
            second = write_arm(root, [walked_audit(0)], tag="b", run_config=other)
            selection = select_poses([first, second], walk_in=WALK_IN)
        self.assertEqual(len(selection.refusals), 1)
        self.assertIn("2 different configurations", selection.refusals[0])

    def test_select_prints_the_poses_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as root:
            arm = write_arm(root, [walked_audit()])
            out = StringIO()
            with redirect_stdout(out):
                status = main(["select", arm, "--walk-in", str(WALK_IN)])
        self.assertEqual(status, 0)
        self.assertIn("poses: 1", out.getvalue())
        self.assertIn("clip alarm", out.getvalue())

    def test_a_walk_in_too_short_to_settle_is_refused(self):
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["select", "runs/none/full", "--walk-in", "2"]), 2)


class TestTheRecordedConfiguration(unittest.TestCase):
    def test_the_recorded_config_comes_back_with_only_coherence_changed(self):
        recorded = json.loads(json.dumps(dataclasses.asdict(AudioConfig(
            indirect_ray_count=900))))
        for wanted in (True, False):
            config = audio_config_of(recorded, temporal_coherence=wanted)
            self.assertEqual(config, AudioConfig(indirect_ray_count=900,
                                                 temporal_coherence=wanted))

    def test_a_record_from_before_the_knob_is_read(self):
        recorded = dataclasses.asdict(AudioConfig())
        del recorded["temporal_coherence"]
        self.assertFalse(audio_config_of(recorded, temporal_coherence=False)
                         .temporal_coherence)

    def test_a_field_the_tree_no_longer_has_is_refused(self):
        recorded = dict(dataclasses.asdict(AudioConfig()), ir_time=4.0)
        with self.assertRaises(ValueError):
            audio_config_of(recorded, temporal_coherence=True)


class TestTheSequences(unittest.TestCase):
    def test_a_fixed_renderer_holds_and_the_forced_checks_fall(self):
        """The healthy path and its forced failure, off the same renders."""
        result = sequences()
        self.assertEqual(change(result.hold, SCAN_READINGS), 0.0)
        self.assertEqual(change(result.hold, HOLD, "frozen"), 0.0)
        self.assertEqual(change(result.teleport, SCAN_READINGS), 0.0)
        self.assertEqual(change(result.rescan, SCAN_READINGS), 0.0)
        forced = change(result.hold, SCAN_READINGS, "forced")
        self.assertAlmostEqual(forced, -LOOP_FALL, delta=0.02)
        self.assertAlmostEqual(change(result.rescan, SCAN_READINGS, "forced"), forced,
                               places=6)
        print("fixed renderer: hold 0.0, frozen 0.0, forced {:+.4f} per loop "
              "(FORCED_DECAY {} over {} folds is {:+.4f})".format(
                  forced, FORCED_DECAY, FOLDS, -LOOP_FALL))

    def test_a_fading_renderer_falls_and_the_frozen_arm_does_not(self):
        """A renderer whose IR loses 2% a render. FROZEN folds the hold's first render
        only, so it holds while the real hold and the IR level fall."""
        result = sequences(lambda: FakeWorld(gain=lambda n, yaw: 0.98 ** n))
        self.assertLess(change(result.hold, SCAN_READINGS), -0.05)
        self.assertLess(change(result.hold, SCAN_READINGS, "ir_level"), -0.05)
        self.assertLess(change(result.teleport, SCAN_READINGS), -0.05)
        self.assertLess(abs(change(result.hold, HOLD, "frozen")), 1e-3)

    def test_heading_moves_the_rescan_and_not_the_hold(self):
        """The level is louder facing yaw 0, and the scan turns left seven times."""
        result = sequences(lambda: FakeWorld(gain=lambda n, yaw: 1.5 + math.cos(yaw)))
        self.assertLess(change(result.rescan, SCAN_READINGS), -0.3)
        self.assertEqual(change(result.hold, SCAN_READINGS), 0.0)

    def test_every_kind_renders_in_a_world_of_its_own(self):
        """In one world the hold would follow the rescan at the same pose, and the
        teleport the hold. Each world here sees one kind, and every pose's source."""
        opened = []
        poses = [a_pose(), a_pose(episode=1, source=Xyz(9.0, 0.0, 9.0))]
        probe_scene(poses, factory(opened=opened), clip=CLIP, bed_cue=BED, hop=HOP,
                    hold=HOLD, progress=lambda line: None)
        self.assertEqual(len(opened), len(SEQUENCE_KINDS))
        rescan, hold, teleport = opened
        self.assertEqual([p for p, _ in rescan.seats], [poses[0].positions[0]] * 2)
        self.assertEqual([p for p, _ in hold.seats], [poses[0].positions[0]] * 2)
        self.assertEqual([p for p, _ in teleport.seats], [poses[0].positions[WALK_IN]] * 2)
        for world in opened:
            self.assertEqual(world.sources, [poses[0].source, poses[1].source])
        self.assertEqual(teleport.renders, 2 * HOLD)

    def test_the_rescan_takes_the_recorded_turns_in_place(self):
        world = FakeWorld()
        trace = trace_rescan(a_pose(), world.io(), clip=CLIP, bed_cue=BED, hop=HOP)
        self.assertIsNone(trace.diverged_at)
        self.assertEqual(len(trace.cue), WALK_IN + SCAN_READINGS)
        self.assertEqual(world.renders, WALK_IN + SCAN_READINGS)
        self.assertAlmostEqual(world.yaw, math.radians(30.0) * (SCAN_READINGS - 1))

    def test_a_replay_that_leaves_the_record_is_marked_and_stops(self):
        """The forced failure for the position check: a forward that moves too far."""
        world = FakeWorld(step=0.3)
        trace = trace_hold(a_pose(), world.io(), clip=CLIP, bed_cue=BED, hop=HOP,
                           hold=HOLD)
        self.assertEqual(trace.diverged_at, 1)
        self.assertTrue(sequences(lambda: FakeWorld(step=0.3)).diverged)

    def test_a_wrong_seat_heading_diverges_at_the_first_forward(self):
        """The rebuild is tested by the anchor forward landing where the record says."""
        pose = a_pose(seat_yaw=math.radians(30.0))
        trace = trace_rescan(pose, FakeWorld().io(), clip=CLIP, bed_cue=BED, hop=HOP)
        self.assertEqual(trace.diverged_at, 1)

    def test_the_teleport_seats_at_the_scan_with_no_walk_in(self):
        world = FakeWorld()
        pose = a_pose()
        trace = trace_teleport(pose, world.io(), clip=CLIP, bed_cue=BED, hop=HOP, hold=HOLD)
        self.assertEqual(world.seats, [(pose.positions[WALK_IN], pose.scan_yaw)])
        self.assertEqual(world.renders, HOLD)
        self.assertEqual(len(trace.cue), HOLD)

    def test_a_result_survives_a_round_trip(self):
        result = sequences()
        self.assertEqual(PoseResult.from_dict(json.loads(json.dumps(result.as_dict()))),
                         result)

    def test_the_seat_rotation_is_the_yaw_the_world_reads(self):
        """``World.set_pose`` takes [x, y, z, w]; ``yaw_from_quaternion`` reads it back."""
        for yaw in (0.0, 0.4, -2.0, math.pi / 2):
            x, y, z, w = yaw_rotation(yaw)
            read = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))
            self.assertAlmostEqual(read, yaw)


class TestTheTrend(unittest.TestCase):
    def test_a_consistent_fall_falls_and_a_rise_rises(self):
        self.assertEqual(trend([-0.1] * MIN_POSES)["verdict"], FALLS)
        self.assertEqual(trend([0.1] * MIN_POSES)["verdict"], RISES)

    def test_a_tiny_consistent_fall_is_flat(self):
        """Significant and too small to explain anything."""
        self.assertEqual(trend([-0.01] * 100)["verdict"], FLAT)

    def test_a_large_median_on_a_split_sign_is_flat(self):
        self.assertEqual(trend([-0.2] * 12 + [0.05] * 10)["verdict"], FLAT)

    def test_ties_count_neither_way_and_too_few_poses_is_unread(self):
        self.assertEqual(trend([0.0] * 50)["verdict"], FLAT)
        self.assertEqual(trend([0.0] * 50)["p"], None)
        self.assertEqual(trend([-0.1] * (MIN_POSES - 1))["verdict"], UNREAD)


def verdicts(recorded=FALLS, rescan=(FALLS, FALLS), hold=(FLAT, FLAT),
             rescan_forced=(FALLS, FALLS), hold_forced=(FALLS, FALLS), frozen=(FLAT, FLAT)):
    return {"recorded": {"run": recorded},
            **{row: dict(zip(ARM_ORDER, cells)) for row, cells in (
                ("rescan", rescan), ("hold", hold), ("rescan_forced", rescan_forced),
                ("hold_forced", hold_forced), ("frozen", frozen))}}


class TestTheBranch(unittest.TestCase):
    def test_every_branch_the_docstring_names(self):
        cases = [
            (verdicts(hold=(FALLS, FLAT)), PRESET),
            (verdicts(hold=(FALLS, FALLS)), RENDERER),
            (verdicts(hold=(FLAT, FALLS)), MIXED),
            (verdicts(hold=(RISES, RISES)), MIXED),
            (verdicts(), HEADING),
            (verdicts(rescan=(FALLS, FLAT)), PRESET),
            (verdicts(rescan=(FLAT, FALLS)), MIXED),
            (verdicts(rescan=(FLAT, FLAT)), SELECTION),
            (verdicts(rescan=(RISES, FLAT)), SELECTION),
            (verdicts(frozen=(FLAT, RISES)), PIPELINE),
            (verdicts(recorded=FLAT), NOT_RUN),
            (verdicts(hold_forced=(FALLS, FLAT)), NOT_RUN),
            (verdicts(rescan_forced=(FLAT, FALLS)), NOT_RUN),
            (verdicts(hold=(UNREAD, FLAT)), NOT_RUN),
        ]
        for given, wanted in cases:
            self.assertEqual(decide(given)[0], wanted, given)

    def test_a_fall_at_a_fixed_pose_is_never_called_selection(self):
        """The verifier's case: the turning replay does not fall and the hold does. The
        hold is at a fixed pose and heading, so it is read first."""
        for rescan in ((FLAT, FLAT), (RISES, RISES)):
            self.assertEqual(decide(verdicts(rescan=rescan, hold=(FALLS, FALLS)))[0],
                             RENDERER)
            self.assertEqual(decide(verdicts(rescan=rescan, hold=(FALLS, FLAT)))[0],
                             PRESET)

    def test_a_rising_sequence_excuses_its_own_forced_check(self):
        """A 10% fall on a level that rises faster need not read as a fall."""
        self.assertEqual(decide(verdicts(rescan=(RISES, RISES), rescan_forced=(FLAT, FLAT)))[0],
                         SELECTION)

    def test_the_pipeline_is_read_before_the_hold(self):
        self.assertEqual(decide(verdicts(hold=(FALLS, FALLS), frozen=(FALLS, FLAT)))[0],
                         PIPELINE)


def arm_results(make_world, n=MIN_POSES + 5):
    """``n`` poses rendered in one arm, each kind in a fresh world from ``make_world``."""
    base = a_pose(recorded_change=-0.1)
    poses = [replace(base, episode=k) for k in range(n)]
    results = probe_scene(poses, factory(make_world), clip=CLIP, bed_cue=BED, hop=HOP,
                          hold=HOLD, progress=lambda line: None)
    return {result.pose.key: result for result in results}


def fading(n, yaw):
    return 0.98 ** n


class TestTheReadout(unittest.TestCase):
    def test_a_fall_only_with_coherence_on_is_the_preset(self):
        result = readout({"tc1": arm_results(lambda: FakeWorld(gain=fading)),
                          "tc0": arm_results(FakeWorld)}, hold=HOLD)
        self.assertEqual(result["branch"], PRESET, result["why"])
        self.assertEqual(result["n_read"], MIN_POSES + 5)
        self.assertEqual(result["rows"]["frozen"]["tc1"]["verdict"], FLAT)

    def test_a_fall_only_while_turning_is_heading(self):
        heading = lambda: FakeWorld(gain=lambda n, yaw: 1.5 + math.cos(yaw))  # noqa: E731
        result = readout({"tc1": arm_results(heading), "tc0": arm_results(heading)},
                         hold=HOLD)
        self.assertEqual(result["branch"], HEADING, result["why"])

    def test_nothing_falling_again_is_selection(self):
        result = readout({"tc1": arm_results(FakeWorld), "tc0": arm_results(FakeWorld)},
                         hold=HOLD)
        self.assertEqual(result["branch"], SELECTION, result["why"])

    def test_diverged_poses_are_not_read(self):
        tc1 = arm_results(FakeWorld)
        tc0 = arm_results(lambda: FakeWorld(step=0.3))
        result = readout({"tc1": tc1, "tc0": tc0}, hold=HOLD)
        self.assertEqual((result["n_read"], result["diverged"]["tc0"]), (0, len(tc0)))
        self.assertEqual(result["branch"], NOT_RUN)


class TestTheTagOnDisk(unittest.TestCase):
    def _tag(self, root, worlds):
        for arm, make_world in worlds.items():
            arm_dir = pathlib.Path(root) / arm
            arm_dir.mkdir(parents=True)
            header = arm_header(
                arm, applied={"temporalCoherence": int(arm == "tc1")}, config=RENDER_CONFIG,
                clip="clip.wav", clip_samples=CLIP.size, hop=HOP, period=FOLDS, hold=HOLD,
                n_poses=MIN_POSES + 5)
            (arm_dir / "arm.json").write_text(json.dumps(header))
            payload = scene_payload("AAAscene", None, list(arm_results(make_world).values()))
            (arm_dir / "AAAscene.json").write_text(json.dumps(payload))
            (arm_dir / "BBBscene.json").write_text(
                json.dumps(scene_payload("BBBscene", "RuntimeError: boom", [])))

    def test_read_prints_the_branch_and_the_scene_error(self):
        with tempfile.TemporaryDirectory() as root:
            self._tag(root, {"tc1": lambda: FakeWorld(gain=fading), "tc0": FakeWorld})
            out = StringIO()
            with redirect_stdout(out):
                status = main(["read", root])
            arms, headers, errors = read_tag(root)
        text = out.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("BRANCH: PRESET", text)
        self.assertIn("SCENE ERROR tc1 BBBscene: RuntimeError: boom", text)
        self.assertEqual(headers["hold"], HOLD)
        self.assertEqual(len(errors), 2)

    def test_a_header_that_disagrees_with_its_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            self._tag(root, {"tc1": FakeWorld, "tc0": FakeWorld})
            path = pathlib.Path(root) / "tc0" / "arm.json"
            header = json.loads(path.read_text())
            path.write_text(json.dumps(dict(header, temporal_coherence=True)))
            with self.assertRaises(ValueError):
                read_tag(root)

    def test_not_run_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as root:
            self._tag(root, {"tc1": FakeWorld, "tc0": lambda: FakeWorld(step=0.3)})
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["read", root]), 2)

    def test_the_readout_names_every_row(self):
        result = readout({"tc1": arm_results(FakeWorld), "tc0": arm_results(FakeWorld)},
                         hold=HOLD)
        text = format_readout(result, tag="t", errors=[])
        for label in ("recorded first scan", "rescan", "hold: fixed heading, first 8",
                      "teleport hold", "CHECK forced", "CHECK frozen", "IR level"):
            self.assertIn(label, text)


if __name__ == "__main__":
    unittest.main()
