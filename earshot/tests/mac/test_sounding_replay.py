"""The sounding-half reader, against walks whose true heading the fixture knows.

The fixtures move a pose the way the fake world does (30 degree turns, 0.25 m forwards
along ``forward_xz``) and record only what the runner records: positions and actions, no
heading. So a reader that rebuilds the heading right recovers the fixture's own, and one
that rebuilds it wrong is caught here before it grades a single real step.

ADR-0014's two arms: a turn the record missed must make its anchor pair disagree, and an
ear that hears the wrong side must grade at 0%.
"""

import json
import math
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.occupancy import forward_xz
from earshot.audio.lateral import bearing_lateral_sign
from earshot.report.agent import AgentReport
from earshot.report.artifacts import ENV_REPORT_NAME, write_episode
from earshot.report.audit import (
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    SoundingWindowRecord,
    StepRecord,
)
from earshot.tools.sounding_replay import (
    NO_DETOUR,
    READ,
    TURN_RAD,
    episode_sounding,
    headings,
    load_run,
    main,
    summarize,
)
from earshot.types import Pose, Xyz

SOURCE = Xyz(0.0, 0.0, 0.0)
PRE = 3
FULL = {"localization_arm": "realizable", "climb_rule": "live", "lateral_cue": "live",
        "cast_policy": "cast"}
F, L, R = "move_forward", "turn_left", "turn_right"
# Turns and forwards in several headings, so every anchor pair spans turns.
WALK = [L, L, F, F, F, R, R, R, F, F, F, L, F, F, R, F, F, F, L, L, F, F] * 2


def simulate(start, yaw, actions):
    """The true pose before each action, as the fake world moves it."""
    poses, position = [], start
    for action in actions:
        poses.append(Pose(position=position, yaw_rad=yaw))
        if action == F:
            dx, dz = forward_xz(yaw)
            position = Xyz(position.x + 0.25 * dx, position.y, position.z + 0.25 * dz)
        elif action == L:
            yaw += TURN_RAD
        elif action == R:
            yaw -= TURN_RAD
    return poses


def walk_episode(actions=WALK, *, start=Xyz(1.0, 0.0, 6.0), yaw=0.3, offset_at=30,
                 hear=lambda truth: truth, route_scale=1.0, reached_at=None,
                 missed_turn_at=None, index=0):
    """An episode whose detour walks ``actions``. ``hear`` maps the true side to the
    recorded sign. ``missed_turn_at`` records a turn at that detour step as a forward
    that did not move, which is a record that lost a turn."""
    poses = simulate(start, yaw, actions)
    rows = [StepRecord(step=i, measured_rms=0.02, position=start) for i in range(PRE)]
    offset = PRE + offset_at
    for k, (pose, action) in enumerate(zip(poses, actions)):
        truth = bearing_lateral_sign(pose, SOURCE)
        line = math.hypot(pose.position.x, pose.position.z)
        recorded = F if k == missed_turn_at else action
        rows.append(StepRecord(
            step=PRE + k, measured_rms=0.1, lateral_sign=hear(truth),
            position=pose.position, source_playing=PRE + k < offset,
            geodesic_to_source=route_scale * line,
            realizable_action=recorded, action=recorded))
    return EpisodeAudit(
        episode_index=index, source_xyz=SOURCE,
        funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
        source_reached_step=None if reached_at is None else PRE + reached_at,
        onset=OnsetRecord(onset_step=PRE, pre_onset_rms=0.02),
        sounding_window=SoundingWindowRecord(
            opens_at=PRE, offset_step=offset, policy="fixed_steps", cue_tail_steps=2),
        steps=tuple(rows), **FULL)


def read(audit):
    return episode_sounding(audit, run="tag/full", scene="AAAscene")


class TestTheHeadingIsRebuiltFromTheRecord(unittest.TestCase):
    def test_every_confirmed_heading_is_the_true_one(self):
        poses = simulate(Xyz(1.0, 0.0, 6.0), 0.3, WALK)
        yaws, pairs, disagree = headings([p.position for p in poses], WALK)
        self.assertGreater(pairs, 5)
        self.assertEqual(disagree, 0)
        confirmed = [(y, p.yaw_rad) for y, p in zip(yaws, poses) if y is not None]
        self.assertGreater(len(confirmed), len(WALK) // 2)
        for rebuilt, true in confirmed:
            self.assertAlmostEqual(math.sin(rebuilt - true), 0.0, places=9)
            self.assertGreater(math.cos(rebuilt - true), 0.0)

    def test_steps_before_the_first_forward_are_not_confirmed(self):
        poses = simulate(Xyz(1.0, 0.0, 6.0), 0.3, WALK)
        yaws, _, _ = headings([p.position for p in poses], WALK)
        self.assertEqual(yaws[:2], [None, None])  # two turns before the first forward

    def test_a_turn_the_record_lost_makes_its_pair_disagree(self):
        audit = walk_episode(missed_turn_at=5)  # the first right turn, recorded as a forward
        s = read(audit)
        self.assertEqual(s.anchor_disagree, 1)
        positions = [r.position for r in audit.steps[PRE:]]
        actions = [r.action for r in audit.steps[PRE:]]
        yaws, _, _ = headings(positions, actions)
        # The steps inside the pair lose their heading. The anchors at either end keep
        # theirs: each is a forward that moved, and it is confirmed by its other neighbour.
        self.assertTrue(all(y is None for y in yaws[5:8]))
        self.assertIsNotNone(yaws[4])
        self.assertIsNotNone(yaws[8])

    def test_malformed_input_is_refused(self):
        with self.assertRaises(ValueError):
            headings([SOURCE, SOURCE], [F])


class TestTheEar(unittest.TestCase):
    def test_an_ear_that_hears_the_true_side_grades_right(self):
        s = read(walk_episode())
        self.assertEqual(s.status, READ)
        self.assertTrue(s.ear)
        self.assertTrue(all(e.heard == e.truth for e in s.ear))

    def test_an_ear_that_hears_the_wrong_side_grades_wrong(self):
        summary = summarize([_run([walk_episode(hear=lambda t: -t)])])
        self.assertEqual(summary["pooled"]["ear"]["right_of_heard"], 0.0)

    def test_an_ear_that_hears_nothing_is_counted_and_not_graded(self):
        summary = summarize([_run([walk_episode(hear=lambda t: 0)])])
        ear = summary["pooled"]["ear"]
        self.assertEqual(ear["heard_nothing"], ear["n"])
        self.assertIsNone(ear["right_of_heard"])

    def test_only_steps_while_the_source_sounds_are_graded(self):
        late = read(walk_episode(offset_at=10))
        early = read(walk_episode(offset_at=40))
        self.assertLess(len(late.ear), len(early.ear))

    def test_a_route_much_longer_than_the_line_is_around_a_wall(self):
        summary = summarize([_run([walk_episode(route_scale=2.0)])])
        self.assertEqual(summary["ear_open"]["n"], 0)
        self.assertEqual(summary["ear_walled"]["n"], summary["pooled"]["ear"]["n"])


class TestTheClock(unittest.TestCase):
    def test_a_far_source_is_out_of_time(self):
        s = read(walk_episode(start=Xyz(0.0, 0.0, 20.0), offset_at=30))
        self.assertFalse(s.in_time)  # (20 - 1) / 0.25 = 76 forwards, 30 steps sounded

    def test_a_near_source_is_in_time(self):
        s = read(walk_episode(start=Xyz(0.0, 0.0, 6.0), offset_at=30))
        self.assertTrue(s.in_time)  # (6 - 1) / 0.25 = 20 forwards
        self.assertEqual(s.sounding_steps, 30)

    def test_a_reach_counts_only_before_the_offset_step(self):
        self.assertTrue(read(walk_episode(reached_at=20)).reached_sounding)
        self.assertFalse(read(walk_episode(reached_at=30)).reached_sounding)

    def test_an_episode_with_no_detour_is_named(self):
        audit = walk_episode()
        bare = EpisodeAudit(episode_index=0, source_xyz=SOURCE,
                            funnel_stage=FunnelStage.T_ANOM_REACHED,
                            steps=audit.steps[:PRE], **FULL)
        self.assertEqual(read(bare).status, NO_DETOUR)


def _run(audits):
    from earshot.tools.sounding_replay import SoundingRun
    episodes = tuple(read(a) for a in audits)
    return SoundingRun("tag/full", "/runs/tag/full", len(audits), episodes, ())


class TestTheRunsOnDisk(unittest.TestCase):
    def _arm(self, audits, *, audio=None):
        arm = pathlib.Path(tempfile.mkdtemp()) / "legs-1" / "full"
        scene = arm / "AAAscene"
        scene.mkdir(parents=True)
        for index, audit in enumerate(audits):
            write_episode(str(scene), index, AgentReport(resumed=True), audit)
        recorded = {"temporal_coherence": False, "pre_onset_rms_tol": 0.05}
        recorded.update(audio or {})
        (scene / ENV_REPORT_NAME).write_text(
            json.dumps({"run_config": {"audio": recorded}}), encoding="utf-8")
        return arm

    def _main(self, *argv):
        out = StringIO()
        with redirect_stdout(out):
            code = main([str(a) for a in argv])
        return code, out.getvalue()

    def test_a_run_prints_every_section(self):
        arm = self._arm([walk_episode(index=0), walk_episode(index=1, reached_at=20)])
        self.assertEqual(load_run(str(arm)).refusals, ())
        code, text = self._main(arm)
        self.assertEqual(code, 0)
        for heading in ("1. THE CLOCK", "2. THE EAR", "3. THE HEADING REBUILD",
                        "WHAT EACH ANSWER MEANS", "NOT A GATE"):
            self.assertIn(heading, text)
        self.assertNotIn("RED:", text)

    def test_json_carries_the_summary(self):
        code, text = self._main(self._arm([walk_episode()]), "--json")
        self.assertEqual(code, 0)
        payload = json.loads(text)
        self.assertEqual(payload["summary"]["pooled"]["n_read"], 1)

    def test_a_render_with_the_preset_key_on_is_refused(self):
        code, text = self._main(self._arm([walk_episode()],
                                          audio={"temporal_coherence": True}))
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", text)

    def test_a_run_with_no_confirmed_heading_is_red(self):
        turning = [L, R] * 20  # never a forward, so nothing anchors a heading
        code, text = self._main(self._arm([walk_episode(actions=turning)]))
        self.assertEqual(code, 2)
        self.assertIn("RED:", text)


if __name__ == "__main__":
    unittest.main()
