"""The leg probe's logic, against the fake world the hold probe's tests already use.

What a Mac can settle: that a pose is the leg ``leg_replay`` graded, one per episode;
that the walk replays the recorded actions and is refused when it leaves the recorded
path; that the readout reads the level the way the record was read; and that the branch
is the one the module docstring pre-registers. What it cannot settle is how the real
renderer behaves while the agent walks, which is the whole question.

ADR-0014's two arms throughout: a renderer that fades must read FALLS and a fixed one
must not, the FORCED check must see its own fall on a renderer that holds, and a replay
that leaves the recorded path must not be read.
"""

import json
import math
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

from _interpreter import assert_interpreter  # noqa: F401
from test_hold_probe import (
    BED,
    CLIP,
    HOP,
    RENDER_CONFIG,
    FakeWorld,
    factory,
    write_arm,
)
from test_leg_replay import FOLDS, PRE, RING, THREE_LEGS, episode

from earshot.agent.controller import ACT_FORWARD, ACT_TURN_LEFT, ACT_TURN_RIGHT
from earshot.tools.hold_probe import (
    EXCLUDED_DUPLICATE,
    EXCLUDED_SHORT,
    FALLS,
    FLAT,
    MIN_POSES,
    MIXED,
    NOT_RUN,
    PRESET,
    RENDERER,
    RISES,
    SELECTION,
    UNREAD,
    arm_header,
    scene_payload,
)
from earshot.tools.leg_probe import (
    LegResult,
    decide,
    format_readout,
    main,
    plan_leg,
    probe_legs,
    read_walk,
    readout,
    select_legs,
    sounding_legs,
)
from earshot.tools.leg_replay import COMPLETED, LEG_PERIOD, SOUNDING, episode_legs
from earshot.types import Xyz

WALK_IN = 10  # far enough back to reach the search forwards the heading is rebuilt from
STEP_M = 0.25
TURN_RAD = math.radians(30.0)


def legged_audit(index=0, **kwargs):
    """The leg fixture, with the positions habitat's model would have produced.

    ``episode`` writes a position per step off the same ``x`` its levels come from, so
    its walk-in runs along one axis and its legs along another. A record the runner wrote
    has ONE motion model throughout, and the probe replays actions through that model, so
    the positions here are rebuilt from the recorded actions. The levels are left alone:
    what the probe renders is the test's own field, and the recorded levels are only what
    the recorded change is read from.
    """
    for key, value in (("offset_at", 30), ("loop", RING), ("folds", FOLDS)):
        kwargs.setdefault(key, value)
    audit = episode(THREE_LEGS, index=index, **kwargs)
    rows = list(audit.steps)
    position, yaw = Xyz(0.0, 0.0, 0.0), 0.0
    for k, row in enumerate(rows):
        action = ACT_FORWARD if k < PRE else row.realizable_action
        rows[k] = replace(row, position=position, action=action, collided=False,
                          displacement_m=STEP_M if action == ACT_FORWARD else 0.0)
        if action == ACT_FORWARD:
            position = Xyz(position.x - STEP_M * math.sin(yaw), position.y,
                           position.z - STEP_M * math.cos(yaw))
        elif action == ACT_TURN_LEFT:
            yaw += TURN_RAD
        elif action == ACT_TURN_RIGHT:
            yaw -= TURN_RAD
    return replace(audit, steps=tuple(rows), source_class="alarm")


def first_leg(audit):
    return episode_legs(audit, run="tag/full", scene="AAAscene").legs[0]


def a_pose(audit=None, **overrides):
    audit = audit or legged_audit()
    pose, why = plan_leg(audit, first_leg(audit), walk_in=WALK_IN)
    assert pose is not None, why
    return replace(pose, **overrides) if overrides else pose


def walks(make_world=FakeWorld, poses=None):
    return probe_legs(poses or [a_pose()], factory(make_world), clip=CLIP, bed_cue=BED,
                      hop=HOP, progress=lambda line: None)


class TestThePoseIsTheRecordedLeg(unittest.TestCase):
    def test_the_pose_reads_the_leg_leg_replay_graded(self):
        audit = legged_audit()
        leg = first_leg(audit)
        self.assertEqual((leg.outcome, leg.sounding), (COMPLETED, SOUNDING))
        pose = a_pose(audit)
        self.assertEqual(pose.start_step, leg.start_step)
        self.assertEqual(len(pose.positions), WALK_IN + LEG_PERIOD)
        self.assertEqual(len(pose.recorded_rms), LEG_PERIOD)
        self.assertEqual(pose.walk_in, WALK_IN)
        self.assertEqual(pose.period, FOLDS)
        self.assertAlmostEqual(pose.recorded_change, leg.change_per_loop)

    def test_the_readings_start_one_step_after_the_turn_that_opened_the_leg(self):
        audit = legged_audit()
        leg = first_leg(audit)
        by_step = {int(r.step): r for r in audit.steps}
        pose = a_pose(audit)
        self.assertAlmostEqual(
            pose.recorded_rms[0], by_step[int(leg.start_step) + 1].measured_rms)

    def test_the_heading_is_rebuilt_from_the_walk_ins_first_clean_forward(self):
        self.assertAlmostEqual(a_pose().seat_yaw, 0.0)

    def test_a_leg_too_early_for_the_walk_in_is_excluded(self):
        audit = legged_audit()
        pose, why = plan_leg(audit, first_leg(audit), walk_in=WALK_IN + 20)
        self.assertIsNone(pose)
        self.assertEqual(why, EXCLUDED_SHORT)

    def test_the_pose_survives_a_round_trip(self):
        pose = a_pose()
        result = LegResult(pose=pose, walk=walks()[0].walk)
        self.assertEqual(LegResult.from_dict(json.loads(json.dumps(result.as_dict()))),
                         result)


class TestTheSelection(unittest.TestCase):
    def test_one_leg_an_episode_and_it_is_the_first(self):
        audit = legged_audit()
        replay = episode_legs(audit, run="tag/full", scene="AAAscene")
        self.assertGreater(sum(1 for leg in replay.legs if leg.outcome == COMPLETED), 1)
        planned = list(sounding_legs(audit, replay, walk_in=WALK_IN))
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0][0].start_step, replay.legs[0].start_step)

    def test_a_leg_the_source_stopped_under_is_no_candidate(self):
        audit = legged_audit(offset_at=4)
        replay = episode_legs(audit, run="tag/full", scene="AAAscene")
        self.assertEqual(list(sounding_legs(audit, replay, walk_in=WALK_IN)), [])

    def test_the_same_episode_in_a_second_run_is_one_pose(self):
        with tempfile.TemporaryDirectory() as root:
            first = write_arm(root, [legged_audit(0)], tag="a")
            second = write_arm(root, [legged_audit(0)], tag="b")
            selection = select_legs([first, second], walk_in=WALK_IN)
        self.assertEqual(selection.refusals, ())
        self.assertEqual([p.run for p in selection.poses], ["a/full"])
        self.assertEqual(dict(selection.excluded), {EXCLUDED_DUPLICATE: 1})

    def test_select_prints_the_legs_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as root:
            arm = write_arm(root, [legged_audit(0), legged_audit(1)])
            out = StringIO()
            with redirect_stdout(out):
                status = main(["select", arm, "--walk-in", str(WALK_IN)])
        self.assertEqual(status, 0)
        self.assertIn("legs offered by the runs: 2   poses: 2", out.getvalue())

    def test_a_walk_in_too_short_to_settle_is_refused(self):
        with redirect_stdout(StringIO()):
            self.assertEqual(main(["select", "runs/none/full", "--walk-in", "2"]), 2)


class TestTheWalk(unittest.TestCase):
    def test_the_walk_follows_the_recorded_path_and_reads_the_leg(self):
        (result,) = walks()
        self.assertIsNone(result.walk.diverged_at)
        self.assertEqual(len(result.walk.cue), WALK_IN + LEG_PERIOD)
        self.assertEqual(result.walk.read_from, WALK_IN)

    def test_a_world_that_moves_differently_diverges_and_is_not_read(self):
        """The forced failure: a heading rebuilt wrong walks somewhere else, and a
        sequence that left the recorded path is not a replay of it."""
        (result,) = walks(lambda: FakeWorld(step=0.30))
        self.assertIsNotNone(result.walk.diverged_at)
        self.assertTrue(result.diverged)

    def test_one_world_serves_every_pose_in_the_scene(self):
        opened = []
        poses = [a_pose(), replace(a_pose(), episode=1)]
        results = probe_legs(poses, factory(FakeWorld, opened), clip=CLIP, bed_cue=BED,
                             hop=HOP, progress=lambda line: None)
        self.assertEqual(len(opened), 1)
        self.assertEqual(len(opened[0].sources), 2)
        self.assertEqual(len(results), 2)


def fading(n, yaw):
    return 0.98 ** n


def arm_results(make_world, n=MIN_POSES + 5, change=-0.1):
    """``n`` legs walked in one arm, in one world from ``make_world``."""
    base = a_pose(recorded_change=change)
    poses = [replace(base, episode=k) for k in range(n)]
    return {r.pose.key: r for r in probe_legs(
        poses, factory(make_world), clip=CLIP, bed_cue=BED, hop=HOP,
        progress=lambda line: None)}


def verdicts(recorded=FALLS, walk=(FLAT, FLAT), walk_forced=(FALLS, FALLS),
             ir_walk=(FLAT, FLAT)):
    rows = {"walk": walk, "walk_forced": walk_forced, "ir_walk": ir_walk}
    out = {"recorded": {"run": recorded}}
    out.update({row: dict(zip(("tc1", "tc0"), value)) for row, value in rows.items()})
    return out


class TestTheBranch(unittest.TestCase):
    def test_a_fall_in_both_arms_is_the_renderer(self):
        self.assertEqual(decide(verdicts(walk=(FALLS, FALLS)))[0], RENDERER)

    def test_a_fall_with_coherence_on_alone_is_the_preset(self):
        branch, why = decide(verdicts(walk=(FALLS, FLAT)))
        self.assertEqual(branch, PRESET)
        self.assertIn("sweep", why)

    def test_a_fall_with_coherence_off_alone_is_mixed(self):
        self.assertEqual(decide(verdicts(walk=(FLAT, FALLS)))[0], MIXED)

    def test_a_rise_in_either_arm_is_mixed(self):
        self.assertEqual(decide(verdicts(walk=(RISES, FLAT)))[0], MIXED)

    def test_flat_in_both_arms_is_selection_and_names_the_next_check(self):
        branch, why = decide(verdicts())
        self.assertEqual(branch, SELECTION)
        self.assertIn("by_cut_scan", why)

    def test_an_unread_population_is_not_run(self):
        self.assertEqual(decide(verdicts(walk=(UNREAD, FLAT)))[0], NOT_RUN)

    def test_recorded_legs_that_do_not_fall_are_not_run(self):
        branch, why = decide(verdicts(recorded=FLAT))
        self.assertEqual(branch, NOT_RUN)
        self.assertIn("not the legs the chain is about", why)

    def test_a_forced_fall_the_readout_missed_is_not_run(self):
        self.assertEqual(decide(verdicts(walk_forced=(FALLS, FLAT)))[0], NOT_RUN)

    def test_a_rising_walk_excuses_its_own_forced_check(self):
        """A 2% fall per reading on a level that rises faster need not read as a fall."""
        self.assertEqual(
            decide(verdicts(walk=(RISES, RISES), walk_forced=(FLAT, FLAT)))[0], MIXED)


class TestTheReadout(unittest.TestCase):
    def test_a_fall_only_with_coherence_on_is_the_preset(self):
        result = readout({"tc1": arm_results(lambda: FakeWorld(gain=fading)),
                          "tc0": arm_results(FakeWorld)})
        self.assertEqual(result["branch"], PRESET, result["why"])
        self.assertEqual(result["n_read"], MIN_POSES + 5)
        self.assertEqual(result["rows"]["recorded"]["run"]["verdict"], FALLS)

    def test_a_fall_in_neither_arm_is_selection(self):
        result = readout({"tc1": arm_results(FakeWorld), "tc0": arm_results(FakeWorld)})
        self.assertEqual(result["branch"], SELECTION, result["why"])
        self.assertEqual(result["rows"]["walk_forced"]["tc1"]["verdict"], FALLS)

    def test_a_fall_in_both_arms_is_the_renderer(self):
        result = readout({"tc1": arm_results(lambda: FakeWorld(gain=fading)),
                          "tc0": arm_results(lambda: FakeWorld(gain=fading))})
        self.assertEqual(result["branch"], RENDERER, result["why"])

    def test_diverged_legs_are_not_read(self):
        tc0 = arm_results(lambda: FakeWorld(step=0.3))
        result = readout({"tc1": arm_results(FakeWorld), "tc0": tc0})
        self.assertEqual((result["n_read"], result["diverged"]["tc0"]), (0, len(tc0)))
        self.assertEqual(result["branch"], NOT_RUN)


class TestTheTagOnDisk(unittest.TestCase):
    def _tag(self, root, worlds):
        for arm, make_world in worlds.items():
            arm_dir = pathlib.Path(root) / arm
            arm_dir.mkdir(parents=True)
            (arm_dir / "arm.json").write_text(json.dumps(arm_header(
                arm, applied={"temporalCoherence": int(arm == "tc1")},
                config=RENDER_CONFIG, clip="clip.wav", clip_samples=CLIP.size, hop=HOP,
                period=FOLDS, hold=0, n_poses=MIN_POSES + 5)))
            (arm_dir / "AAAscene.json").write_text(json.dumps(scene_payload(
                "AAAscene", None, list(arm_results(make_world).values()))))
            (arm_dir / "BBBscene.json").write_text(json.dumps(
                scene_payload("BBBscene", "RuntimeError: boom", [])))

    def test_read_prints_the_branch_and_the_scene_error(self):
        with tempfile.TemporaryDirectory() as root:
            self._tag(root, {"tc1": lambda: FakeWorld(gain=fading), "tc0": FakeWorld})
            out = StringIO()
            with redirect_stdout(out):
                status = main(["read", root])
            arms, _headers, errors = read_walk(root)
        text = out.getvalue()
        self.assertEqual(status, 0)
        self.assertIn("leg probe", text)
        self.assertIn("BRANCH: PRESET", text)
        self.assertIn("SCENE ERROR tc1 BBBscene: RuntimeError: boom", text)
        self.assertEqual(len(errors), 2)
        self.assertEqual(len(arms["tc1"]), MIN_POSES + 5)

    def test_the_readout_names_the_check_rows_and_the_pre_registration(self):
        result = readout({"tc1": arm_results(FakeWorld), "tc0": arm_results(FakeWorld)})
        text = format_readout(result, tag="runs/walk-1", errors=[])
        self.assertIn("CHECK forced fall, the walk", text)
        self.assertIn("recorded leg, the run's own renders", text)
        self.assertIn("Pre-registered in leg_probe.py before the first box run", text)

    def test_not_run_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as root:
            self._tag(root, {"tc1": FakeWorld, "tc0": lambda: FakeWorld(step=0.3)})
            with redirect_stdout(StringIO()):
                self.assertEqual(main(["read", root]), 2)


if __name__ == "__main__":
    unittest.main()
