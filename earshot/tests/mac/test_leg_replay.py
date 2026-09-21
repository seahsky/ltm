"""ADR-0029's gate, against episodes the controller itself wrote.

The replay's value is that it grades the verdict `READ_LEGS` would act on, on the legs
`full` actually walked. So the fixtures here do not hand-write `realizable_action`: they
drive `realizable_investigate_step` and `next_plateau_steps` tick by tick, the way
`step_controller` does, and record what the rule answered. A replay that segments those
correctly segments the rule, not a model of it.

ADR-0014's two arms throughout: the healthy path, and the forced failure that must fire.
A record that disagrees with the rule must make its leg UNVERIFIED; a reader with the sign
wrong must be graded wrong; a gate that passes pooled must still fail on one bad run.
"""

import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_STOP,
    ACT_TURN_LEFT,
    CAST_STEPS,
    SCAN_STEPS,
    climb_eps,
    next_plateau_steps,
    realizable_investigate_step,
)
from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_episode
from earshot.report.audit import (
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    StepRecord,
)
from earshot.tools.leg_replay import (
    BUILD,
    COMPLETED,
    CUT_BY_END,
    CUT_BY_STOP,
    CUT_BY_SURGE,
    ONE_BRANCH,
    STOP,
    UNVERIFIED,
    Leg,
    episode_legs,
    evaluate_gate,
    format_report,
    load_run,
    main,
    score,
)
from earshot.types import Xyz

SOURCE = Xyz(0.0, 0.0, 0.0)
PRE = 3  # SEARCH steps before the detour opens
# The climb's bar. Far above any leg's trend here, so nothing surges unless a test asks.
EPS = 1.0
NOISE = (0.004, -0.003, 0.002, -0.004, 0.003, -0.002, 0.004, -0.003, -0.001, 0.001)
PERIOD = 1 + CAST_STEPS
TOWARD, AWAY = -1, +1
FULL = {"localization_arm": "realizable", "climb_rule": "live", "lateral_cue": "live",
        "cast_policy": "cast"}


def cast_xs(x0, headings, n):
    """Where the agent stands at each of ``n`` detour readings when nothing surges.

    Still through the scan and through each leg's turn, then 0.25 m per forward along
    that leg's heading on the x axis, where the source sits at the origin. A reading is
    taken at the pose BEFORE the step's action, which is what the runner records.
    """
    xs, x = [], float(x0)
    for k in range(n):
        xs.append(x)
        if k >= SCAN_STEPS and (k - SCAN_STEPS) % PERIOD:
            x += headings[(k - SCAN_STEPS) // PERIOD] * 0.25
    return xs


def episode(xs, *, gain=0.01, index=0, surge_at=None, stop_at=None, tamper_at=None,
            routed=True, scatter=EPS, arm=None):
    """An audit whose detour `realizable_action` the controller wrote, tick by tick.

    ``gain`` is how much louder the cue gets per metre nearer; negative is a field that
    gets quieter on the approach. ``surge_at`` steps the level up from that detour step
    on, so the rule surges. ``stop_at`` is the detour step the confirm fires on.
    ``tamper_at`` records the wrong action at that detour step, and nothing else changes.
    """
    levels, rows, plateau = [], [], 0
    eps = climb_eps(scatter)  # what the runner hands the rule, fallback included
    for i, x in enumerate([xs[0]] * PRE + list(xs)):
        k = i - PRE
        level = 0.1 - gain * x + NOISE[i % len(NOISE)]
        if surge_at is not None and k >= surge_at:
            level += 20.0
        levels.append(level)
        action = None
        if k >= 0:
            action = realizable_investigate_step(
                levels, -1, k == stop_at, eps=eps, plateau_steps=plateau)
            plateau = next_plateau_steps(levels, eps=eps, plateau_steps=plateau)
        recorded = action
        if k == tamper_at:
            recorded = ACT_TURN_LEFT if action == ACT_FORWARD else ACT_FORWARD
        rows.append(StepRecord(
            step=i, measured_rms=level, lateral_sign=-1, position=Xyz(x, 0.0, 0.0),
            displacement_m=0.25, geodesic_to_source=abs(x) if routed else None,
            realizable_action=recorded))
        if action == ACT_STOP:
            break
    fields = dict(FULL, **(arm or {}))
    return EpisodeAudit(
        episode_index=index, source_xyz=SOURCE,
        funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
        onset=OnsetRecord(onset_step=PRE),
        calibration=None if scatter is None else CalibrationRecord(
            onset_rms=0.01, bed_rms=0.001, separation_db=40.0, n_poses=16,
            global_volume=1.0, cue_render_scatter=scatter, cue_scatter_repeats=12),
        steps=tuple(rows), **fields)


# Three legs: toward the source, away from it, toward it again, over 30 detour steps.
# The third needs step 33 to finish, so it is cut by the detour's end.
THREE_LEGS = cast_xs(4.0, [TOWARD, AWAY, TOWARD], 30)


def replay(audit):
    return episode_legs(audit, run="tag/full", scene="AAAscene")


class TestTheLegsAreTheRulesOwn(unittest.TestCase):
    def test_two_legs_complete_and_the_third_is_cut_by_the_end(self):
        result = replay(episode(THREE_LEGS))
        self.assertEqual([leg.outcome for leg in result.legs],
                         [COMPLETED, COMPLETED, CUT_BY_END])
        self.assertEqual(result.n_steps_checked, 30)
        self.assertEqual(result.n_steps_agree, 30)

    def test_a_leg_opens_after_the_scan_and_every_period_after(self):
        starts = [leg.start_step for leg in replay(episode(THREE_LEGS)).legs]
        first = PRE + SCAN_STEPS
        self.assertEqual(starts, [first, first + PERIOD, first + 2 * PERIOD])

    def test_the_readings_are_the_ones_held_at_the_next_turn(self):
        """From the pose after the leg's turn to the pose after its last forward."""
        leg = replay(episode(THREE_LEGS)).legs[0]
        first, last = SCAN_STEPS + 1, SCAN_STEPS + PERIOD
        self.assertAlmostEqual(leg.route_start_m, THREE_LEGS[first])
        self.assertAlmostEqual(leg.delta_route_m, THREE_LEGS[last] - THREE_LEGS[first])
        self.assertAlmostEqual(leg.delta_route_m, -2.0)
        self.assertAlmostEqual(leg.length_m, 2.0)

    def test_a_leg_toward_the_source_reads_louder_and_one_away_quieter(self):
        toward, away, _ = replay(episode(THREE_LEGS)).legs
        self.assertGreater(toward.t, 2.5)
        self.assertLess(away.t, -2.5)
        row = score([toward, away], 2.5)
        self.assertEqual((row["louder_right"], row["louder_fired"]), (1, 1))
        self.assertEqual((row["quieter_right"], row["quieter_fired"]), (1, 1))
        self.assertEqual((row["n_approached"], row["n_receded"]), (1, 1))

    def test_a_record_that_disagrees_makes_its_leg_unverified(self):
        """The forced failure. The reconstruction is unchanged, the record is not, and
        the leg the record disagrees inside must not be graded."""
        result = replay(episode(THREE_LEGS, tamper_at=SCAN_STEPS + 3))
        self.assertEqual([leg.outcome for leg in result.legs],
                         [UNVERIFIED, COMPLETED, CUT_BY_END])
        self.assertEqual(result.n_steps_agree, result.n_steps_checked - 1)
        self.assertEqual(score(result.legs, 2.5)["n_informative"], 1)

    def test_a_surge_mid_leg_cuts_it(self):
        legs = replay(episode(THREE_LEGS, surge_at=SCAN_STEPS + 3)).legs
        self.assertEqual(legs[0].outcome, CUT_BY_SURGE)
        self.assertIsNone(legs[0].t, "a leg cut short is never measured")

    def test_a_surge_on_the_next_turn_still_preempts_the_verdict(self):
        """All eight forwards ran, and the rule surged at the step it would have turned
        on. The reader is never asked, so the leg gets no verdict."""
        result = replay(episode(THREE_LEGS, surge_at=SCAN_STEPS + PERIOD))
        self.assertEqual(result.legs[0].outcome, CUT_BY_SURGE)

    def test_a_stop_mid_leg_cuts_it(self):
        legs = replay(episode(THREE_LEGS, stop_at=SCAN_STEPS + 3)).legs
        self.assertEqual([leg.outcome for leg in legs], [CUT_BY_STOP])

    def test_a_stop_where_a_leg_would_open_is_no_leg(self):
        self.assertEqual(replay(episode(THREE_LEGS, stop_at=SCAN_STEPS)).legs, ())

    def test_an_unrouted_leg_is_measured_but_not_graded(self):
        legs = replay(episode(THREE_LEGS, routed=False)).legs
        self.assertEqual(legs[0].outcome, COMPLETED)
        self.assertIsNone(legs[0].delta_route_m)
        self.assertEqual(score(legs, 2.5)["n_informative"], 0)

    def test_a_field_that_gets_quieter_on_the_approach_is_graded_wrong(self):
        """The grader's forced failure: a verdict with the wrong sign scores zero."""
        legs = replay(episode(THREE_LEGS, gain=-0.01)).legs
        row = score(legs, 2.5)
        self.assertEqual((row["louder_right"], row["louder_fired"]), (0, 1))
        self.assertEqual((row["quieter_right"], row["quieter_fired"]), (0, 1))

    def test_a_short_leg_is_uninformative_and_feeds_the_false_decisive_rate(self):
        """A leg across the source's bearing has no right answer. A verdict on it is a
        false positive, and that is the rate a critical value cannot state."""
        leg = Leg("r", "s", 0, 0, COMPLETED, t=5.0, length_m=2.0, route_start_m=3.0,
                  delta_route_m=0.2)
        row = score([leg], 2.5)
        self.assertEqual(row["n_informative"], 0)
        self.assertEqual((row["uninformative_decisive"], row["n_uninformative"]), (1, 1))

    def test_an_episode_that_never_diverted_has_no_legs(self):
        audit = episode(THREE_LEGS)
        bare = EpisodeAudit(
            episode_index=0, source_xyz=SOURCE, funnel_stage=FunnelStage.RUN,
            steps=tuple(StepRecord(step=r.step, measured_rms=r.measured_rms)
                        for r in audit.steps))
        result = replay(bare)
        self.assertEqual((result.legs, result.has_detour), ((), False))


def legs(run, pairs):
    """Completed legs from ``(t, delta_route_m)`` pairs."""
    return [Leg(run, "s", i, 0, COMPLETED, t=t, length_m=2.0, route_start_m=3.0,
                delta_route_m=delta) for i, (t, delta) in enumerate(pairs)]


RIGHT_LOUDER, RIGHT_QUIETER = (5.0, -2.0), (-5.0, 2.0)
WRONG_LOUDER, WRONG_QUIETER = (5.0, 2.0), (-5.0, -2.0)


class TestTheGateIsTheOneWrittenDown(unittest.TestCase):
    def _three(self, pairs):
        return {name: legs(name, pairs) for name in ("a/full", "b/full", "c/full")}

    def test_both_branches_right_is_build(self):
        gate = evaluate_gate(self._three(
            [RIGHT_LOUDER] * 10 + [RIGHT_QUIETER] * 10 + [WRONG_QUIETER]))
        self.assertEqual(gate["verdict"], BUILD)
        self.assertTrue(gate["preregistered"])

    def test_one_coin_flip_branch_is_one_branch_and_not_build(self):
        """Pooled, LOUDER's 100% would carry QUIETER's 50% over 75%. The branches are
        judged one at a time, so the coin flip is not shipped inside a good average."""
        gate = evaluate_gate(self._three(
            [RIGHT_LOUDER] * 12 + [RIGHT_QUIETER] * 4 + [WRONG_QUIETER] * 4))
        self.assertEqual((gate["verdict"], gate["branch"]), (ONE_BRANCH, "louder"))

    def test_one_bad_run_stops_a_gate_that_passes_pooled(self):
        """The forced failure of the per-run clause: 26 of 30 pooled is 87%, and one
        render at 60% says the verdict is not a property of the field."""
        good = [RIGHT_LOUDER] * 10 + [RIGHT_QUIETER] * 10
        bad = [RIGHT_LOUDER] * 6 + [WRONG_LOUDER] * 4 + [RIGHT_QUIETER] * 10
        gate = evaluate_gate({"a/full": legs("a/full", good), "b/full": legs("b/full", good),
                              "c/full": legs("c/full", bad)})
        row = gate["rows"][0]
        self.assertGreater(row["branches"]["louder"]["accuracy"], 0.75)
        self.assertAlmostEqual(row["branches"]["louder"]["worst_run_accuracy"], 0.6)
        self.assertEqual(gate["verdict"], ONE_BRANCH)
        self.assertEqual(gate["branch"], "quieter")

    def test_a_run_where_a_branch_never_fired_fails_that_branch(self):
        """An accuracy that could not be measured is never green."""
        runs = self._three([RIGHT_LOUDER] * 10 + [RIGHT_QUIETER] * 10)
        runs["c/full"] = legs("c/full", [RIGHT_LOUDER] * 10 + [(0.0, 2.0)] * 10)
        gate = evaluate_gate(runs)
        self.assertIsNone(gate["rows"][0]["branches"]["quieter"]["worst_run_accuracy"])
        self.assertEqual((gate["verdict"], gate["branch"]), (ONE_BRANCH, "louder"))

    def test_an_accurate_reader_that_rarely_fires_is_stop(self):
        gate = evaluate_gate(self._three([RIGHT_LOUDER] * 2 + [(0.5, -2.0)] * 18))
        self.assertEqual(gate["verdict"], STOP)
        self.assertIn("STOP.", self._text(gate))

    def test_the_most_decisive_passing_value_is_chosen(self):
        """At 1.0 the weak wrong LOUDERs sink the branch to 71%. From 1.5 up it is right
        every time, and 1.5 and 2.0 are decisive on the most legs, so 1.5 is chosen."""
        pairs = ([RIGHT_LOUDER] * 6 + [(2.2, -2.0)] * 4 + [(1.2, 2.0)] * 4
                 + [RIGHT_QUIETER] * 10)
        gate = evaluate_gate(self._three(pairs))
        self.assertEqual((gate["verdict"], gate["t_leg"]), (BUILD, 1.5))

    def test_fewer_runs_than_the_adr_names_says_so(self):
        gate = evaluate_gate({"a/full": legs("a/full", [RIGHT_LOUDER] * 10)})
        self.assertFalse(gate["preregistered"])
        self.assertIn("NOT THE PRE-REGISTERED GATE", self._text(gate))

    def _text(self, gate):
        return format_report([], gate, display_t_leg=2.5, display_why="test")


class TestTheRunsOnDisk(unittest.TestCase):
    """Through the real writers, because that seam is where a replay that is right over
    injected legs finds nothing on disk and reports it as a finding."""

    def _arm(self, root, name, audits_by_scene):
        arm = pathlib.Path(root) / name / "full"
        for scene, audits in audits_by_scene.items():
            (arm / scene).mkdir(parents=True)
            for index, audit in enumerate(audits):
                write_episode(str(arm / scene), index, AgentReport(resumed=True), audit)
        return arm

    def _healthy(self, root, name):
        return self._arm(root, name, {
            "AAAscene": [episode(THREE_LEGS, index=0), episode(THREE_LEGS, index=1)],
            "BBBscene": [episode(cast_xs(5.0, [TOWARD, AWAY, AWAY, TOWARD], 40), index=0)],
        })

    def _main(self, *argv):
        out = StringIO()
        with redirect_stdout(out):
            code = main([str(a) for a in argv])
        return code, out.getvalue()

    def test_a_run_loads_every_scene(self):
        run = load_run(str(self._healthy(tempfile.mkdtemp(), "abl-2")))
        self.assertEqual(run.label, "abl-2/full")
        self.assertEqual((run.n_episodes, run.n_detours), (3, 3))
        self.assertEqual(run.refusals, ())
        self.assertEqual(run.n_steps_agree, run.n_steps_checked)
        self.assertEqual({leg.scene for leg in run.legs}, {"AAAscene", "BBBscene"})

    def test_three_runs_print_the_gate(self):
        root = tempfile.mkdtemp()
        code, text = self._main(*(self._healthy(root, n) for n in ("r1", "r2", "r3")))
        self.assertEqual(code, 0)
        self.assertIn("THE GATE (ADR-0029, pre-registered)", text)
        self.assertNotIn("NOT THE PRE-REGISTERED GATE", text)
        self.assertIn("WHAT THE CURRENT READER MISSED", text)
        self.assertIn("THE FIELD, per scene", text)
        self.assertIn("AAAscene", text)

    def test_json_carries_the_gate_and_every_leg(self):
        root = tempfile.mkdtemp()
        code, text = self._main(self._healthy(root, "r1"), "--json", "--no-field")
        self.assertEqual(code, 0)
        payload = json.loads(text)
        self.assertIn(payload["gate"]["verdict"], (BUILD, ONE_BRANCH, STOP))
        self.assertEqual(len(payload["legs"]), 3 + 3 + 4)
        self.assertIsNone(payload["field"])

    def test_an_oracle_arm_is_refused_by_name(self):
        """No `realizable_action` by construction, and not `full`'s rule."""
        audit = episode(THREE_LEGS, arm={"localization_arm": "oracle_matched"})
        bare = replace(audit, steps=tuple(
            replace(r, realizable_action=None) for r in audit.steps))
        arm = self._arm(tempfile.mkdtemp(), "oracle-2", {"AAAscene": [bare]})
        code, text = self._main(arm)
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", text)
        self.assertIn("localization_arm is 'oracle_matched'", text)
        self.assertIn("realizable_action", text)

    def test_another_arm_s_rule_is_refused_by_name(self):
        arm = self._arm(tempfile.mkdtemp(), "abl-2", {
            "AAAscene": [episode(THREE_LEGS, arm={"climb_rule": "off"})]})
        run = load_run(str(arm))
        self.assertTrue(any("climb_rule is 'off'" in r for r in run.refusals))

    def test_a_run_before_the_cue_scatter_is_refused_by_name(self):
        arm = self._arm(tempfile.mkdtemp(), "old", {
            "AAAscene": [episode(THREE_LEGS, scatter=None)]})
        run = load_run(str(arm))
        self.assertTrue(any("cue_render_scatter" in r for r in run.refusals))

    def test_a_run_before_the_route_is_refused_by_name(self):
        arm = self._arm(tempfile.mkdtemp(), "old", {
            "AAAscene": [episode(THREE_LEGS, routed=False)]})
        run = load_run(str(arm))
        self.assertTrue(any("geodesic_to_source" in r for r in run.refusals))

    def test_a_scene_directory_is_refused_rather_than_read_as_empty(self):
        arm = self._healthy(tempfile.mkdtemp(), "abl-2")
        run = load_run(str(arm / "AAAscene"))
        self.assertTrue(any("scene directory" in r for r in run.refusals))

    def test_the_same_run_twice_is_refused(self):
        """Its legs would count twice, and one render would pass the per-run test twice."""
        arm = self._healthy(tempfile.mkdtemp(), "abl-2")
        code, text = self._main(arm, arm)
        self.assertEqual(code, 2)
        self.assertIn("twice", text)

    def test_a_missing_directory_is_exit_two(self):
        code, _ = self._main("/nonexistent/leg-replay")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
