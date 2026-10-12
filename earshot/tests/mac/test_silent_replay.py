"""ADR-0031's gate, against episodes the controller itself wrote.

The fixtures drive `realizable_investigate_step` tick by tick, as `test_leg_replay` does,
so the detour the replay reads is the rule's and `episode_legs` can check it. The source
sits at the origin and the agent casts along x, so the route is |x|.

ADR-0014's two arms throughout: the healthy path, and the forced failure that must fire.
A detector that fires on a sounding source must be counted as a false fire; a run whose
loudest place is farther must read STOP; a gate missing a render must read NOT_RUN.
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

from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_STOP,
    ACT_TURN_LEFT,
    CAST_STEPS,
    SCAN_STEPS,
    back_at_bed,
    climb_eps,
    loudest_place,
    next_plateau_steps,
    realizable_investigate_step,
)
from earshot.report.agent import AgentReport
from earshot.report.artifacts import ENV_REPORT_NAME, write_episode
from earshot.report.audit import (
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    SoundingWindowRecord,
    StepRecord,
)
from earshot.tools import silent_replay
from earshot.tools.silent_replay import (
    BUILD,
    ELIGIBLE,
    NEVER_FIRED,
    NO_BED_LEVEL,
    NO_DETOUR,
    NO_LOUDEST,
    NOT_RUN,
    REACHED_FIRST,
    STOP,
    UNROUTED_START,
    UNVERIFIED,
    Silence,
    SilentRun,
    episode_silence,
    evaluate,
    fit_reach_curve,
    load_run,
    main,
    reach_probability,
)
from earshot.types import Xyz

SOURCE = Xyz(0.0, 0.0, 0.0)
PRE = 3       # SEARCH steps before the detour opens, at the bed
EPS = 1.0     # the climb's bar: far above any trend here, so nothing surges
BED = 0.02
LOOP = 5
TOL = 0.05
PERIOD = 1 + CAST_STEPS
AWAY = +1
FULL = {"localization_arm": "realizable", "climb_rule": "live", "lateral_cue": "live",
        "cast_policy": "cast"}


def cast_xs(x0, headings, n):
    """Where the agent stands at each of ``n`` detour readings when nothing surges.

    Still through the scan and each leg's turn, then 0.25 m per forward along that leg's
    heading. A reading is taken at the pose BEFORE the step's action.
    """
    xs, x = [], float(x0)
    for k in range(n):
        xs.append(x)
        if k >= SCAN_STEPS and (k - SCAN_STEPS) % PERIOD:
            x += headings[(k - SCAN_STEPS) // PERIOD] * 0.25
    return xs


def silent_episode(xs, *, offset_at, index=0, heard_until=None, stop_at=None,
                   tamper_at=None, routed=True, pre_onset=BED, folds=LOOP, arm=None,
                   stage=FunnelStage.INVESTIGATE_ENTERED):
    """An audit whose detour the controller wrote, with a source that stops.

    The source plays from the detour's first step until detour step ``offset_at``. While
    it is heard the level is the bed plus a cue that falls 0.01 per metre from the
    source; ``heard_until`` makes it fall to the bed earlier while it still plays, which
    is a source too faint to hear. ``stop_at`` is the detour step the confirm fires on.

    An arrival is recorded as the runner records it: the controller hands back a CHECK
    with no ``realizable_action`` on that tick, and ``source_reached_step`` is that tick.
    So the arrival is the first step after the detour, never a detour step.
    """
    offset = PRE + offset_at
    heard = PRE + (offset_at if heard_until is None else heard_until)
    levels, rows, plateau = [], [], 0
    eps = climb_eps(EPS)
    for i, x in enumerate([xs[0]] * PRE + list(xs)):
        k = i - PRE
        playing = k >= 0 and i < offset
        level = BED + (0.08 - 0.01 * abs(x) if playing and i < heard else 0.0)
        levels.append(level)
        action = None
        if k >= 0:
            action = realizable_investigate_step(
                levels, -1, k == stop_at, eps=eps, plateau_steps=plateau)
            plateau = next_plateau_steps(levels, eps=eps, plateau_steps=plateau)
        recorded = action
        if k == tamper_at:
            recorded = ACT_TURN_LEFT if action == ACT_FORWARD else ACT_FORWARD
        arrived = action == ACT_STOP
        rows.append(StepRecord(
            step=i, measured_rms=level, lateral_sign=-1, position=Xyz(x, 0.0, 0.0),
            source_playing=playing, displacement_m=0.25,
            geodesic_to_source=abs(x) if routed else None,
            realizable_action=None if arrived else recorded,
            action=None if arrived else action))
        if action == ACT_STOP:
            break
    reached = None if stop_at is None else PRE + stop_at
    return EpisodeAudit(
        episode_index=index, source_xyz=SOURCE,
        funnel_stage=FunnelStage.SOURCE_REACHED if reached is not None else stage,
        source_reached_step=reached,
        onset=OnsetRecord(onset_step=PRE, pre_onset_rms=pre_onset),
        sounding_window=SoundingWindowRecord(
            opens_at=PRE, offset_step=offset, policy="fixed_steps", cue_tail_steps=1),
        calibration=CalibrationRecord(
            onset_rms=0.045, bed_rms=BED, separation_db=20.0, n_poses=16,
            global_volume=1.0, cue_render_scatter=EPS, cue_scatter_repeats=12,
            cue_phase_folds=folds),
        steps=tuple(rows),
        **dict(FULL, **(arm or {})))


# Near the source through the scan, then four legs walking away. The source stops on
# detour step 30, when the agent stands at 6.75 m.
AWAY_WALK = cast_xs(1.5, [AWAY] * 4, 40)


def read(audit):
    return episode_silence(audit, run="tag/full", scene="AAAscene", tolerance=TOL)


class TestTheSilentHalfIsTheArmsOwn(unittest.TestCase):
    def test_the_replay_runs_the_controllers_functions(self):
        self.assertIs(silent_replay.back_at_bed, back_at_bed)
        self.assertIs(silent_replay.loudest_place, loudest_place)

    def test_a_walk_away_from_a_source_that_stops_is_eligible(self):
        s = read(silent_episode(AWAY_WALK, offset_at=30))
        self.assertEqual(s.status, ELIGIBLE)
        # One whole loop of bed readings: detour steps 30..34.
        self.assertEqual(s.fire_step, PRE + 34)
        self.assertEqual(s.delay_steps, 4)
        self.assertFalse(s.false_fire)
        # The scan stood at 1.5 m, and the first whole loop there is kept: a tie never
        # replaces it.
        self.assertEqual(s.loudest_step, PRE + 2)
        self.assertAlmostEqual(s.d_loudest, 1.5)
        self.assertAlmostEqual(s.d_off, 7.5)
        self.assertAlmostEqual(s.d_min, 1.5)
        self.assertAlmostEqual(s.d_end, AWAY_WALK[-1])
        self.assertTrue(s.would_return)
        self.assertAlmostEqual(s.return_m, 6.0)
        self.assertEqual(s.forwards_back, 24)
        self.assertFalse(s.reached_after)

    def test_a_reach_after_the_detector_fired_is_a_silent_reach(self):
        s = read(silent_episode(AWAY_WALK, offset_at=30, stop_at=37))
        self.assertEqual(s.status, ELIGIBLE)
        self.assertTrue(s.reached_after)

    def test_the_arrival_is_the_step_after_the_detour(self):
        """As the runner writes it, so a detector that fired on the detour's last step
        still fired before the reach."""
        audit = silent_episode(AWAY_WALK, offset_at=30, stop_at=35)
        self.assertIsNone(audit.steps[-1].realizable_action)
        self.assertEqual(audit.source_reached_step, audit.steps[-1].step)
        s = read(audit)
        self.assertEqual((s.status, s.fire_step), (ELIGIBLE, PRE + 34))
        self.assertTrue(s.reached_after)

    def test_a_record_that_reached_before_the_detector_fired_is_not_read(self):
        """No arrival the runner writes takes this shape. The status is the guard for a
        record that does."""
        audit = replace(silent_episode(AWAY_WALK, offset_at=30), source_reached_step=PRE + 34)
        self.assertEqual(read(audit).status, REACHED_FIRST)

    def test_a_faint_source_that_still_plays_is_a_false_fire(self):
        s = read(silent_episode(AWAY_WALK, offset_at=36, heard_until=20))
        self.assertEqual(s.status, ELIGIBLE)
        self.assertTrue(s.false_fire)
        self.assertLess(s.delay_steps, 0)

    def test_a_detector_that_fires_before_one_loop_is_kept_has_no_place(self):
        s = read(silent_episode(AWAY_WALK, offset_at=36, heard_until=0))
        self.assertEqual(s.status, NO_LOUDEST)
        self.assertTrue(s.false_fire)

    def test_a_source_that_outlives_the_detour_never_fires(self):
        self.assertEqual(read(silent_episode(AWAY_WALK, offset_at=60)).status, NEVER_FIRED)

    def test_an_agent_already_at_its_loudest_place_does_not_return(self):
        s = read(silent_episode([3.0] * 40, offset_at=30))
        self.assertEqual(s.status, ELIGIBLE)
        self.assertFalse(s.would_return)
        self.assertAlmostEqual(s.d_loudest, s.d_off)

    def test_a_record_that_disagrees_with_the_rule_is_never_read(self):
        self.assertEqual(read(silent_episode(AWAY_WALK, offset_at=30, tamper_at=10)).status,
                         UNVERIFIED)

    def test_the_conditions_are_named(self):
        self.assertEqual(read(silent_episode(
            AWAY_WALK, offset_at=30, routed=False)).status, UNROUTED_START)
        self.assertEqual(read(silent_episode(
            AWAY_WALK, offset_at=30, pre_onset=None)).status, NO_BED_LEVEL)
        bare = silent_episode(AWAY_WALK, offset_at=30)
        bare = replace(bare, funnel_stage=FunnelStage.T_ANOM_REACHED, steps=tuple(
            replace(r, realizable_action=None) for r in bare.steps))
        self.assertEqual(read(bare).status, NO_DETOUR)


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


def synth_run(label, *, shift=3.0, eligible=150, false_fires=0, reach=True,
              n_episodes=282, slope=-0.6, intercept=2.0):
    """A run's eligible episodes, written directly. Reach falls with distance as a real
    curve would, decided by a fixed sequence so the test has no randomness."""
    episodes = []
    for i in range(eligible):
        d_off = 2.0 + (i % 40) * 0.2
        reached = reach and ((i * 37) % 100) / 100.0 < _sigmoid(intercept + slope * d_off)
        d_loudest = max(1.1, d_off - shift)
        episodes.append(Silence(
            run=label, scene="S", episode=i, status=ELIGIBLE, fire_step=100,
            false_fire=i < false_fires, delay_steps=4, loudest_step=50, d_off=d_off,
            d_loudest=d_loudest, d_min=min(d_loudest, d_off), d_end=d_off,
            reached_after=reached, return_m=abs(shift), would_return=abs(shift) > 1.0,
            forwards_back=10))
    return SilentRun(label, "/runs/" + label, n_episodes, tuple(episodes), ())


def five(**kw):
    return [synth_run("r{}/full".format(i), **kw) for i in range(5)]


class TestTheReachCurve(unittest.TestCase):
    def test_a_falling_reach_rate_fits_a_falling_curve(self):
        points = [(s.d_off, s.reached_after) for s in synth_run("r/full").episodes]
        curve = fit_reach_curve(points)
        self.assertIsNotNone(curve)
        self.assertLess(curve[1], 0.0)
        self.assertGreater(reach_probability(curve, 2.0), reach_probability(curve, 8.0))

    def test_no_curve_without_both_outcomes(self):
        self.assertIsNone(fit_reach_curve([(2.0, False), (5.0, False)]))
        self.assertIsNone(fit_reach_curve([(2.0, True), (5.0, True)]))

    def test_separated_data_has_no_curve(self):
        """Every near episode reached and no far one: the likelihood has no maximum."""
        points = [(d / 10.0, d < 30) for d in range(10, 90)]
        self.assertIsNone(fit_reach_curve(points))


class TestTheGateIsTheOneWrittenDown(unittest.TestCase):
    def test_a_nearer_place_that_prices_over_the_mde_is_build(self):
        result = evaluate(five())
        self.assertEqual(result["verdict"], BUILD, result["reasons"])
        self.assertGreaterEqual(result["pooled_price_points"], 7.30)
        self.assertEqual(result["pooled_accuracy"], 1.0)

    def test_a_short_return_prices_under_the_mde_and_stops(self):
        result = evaluate(five(shift=1.2))
        self.assertEqual(result["verdict"], STOP)
        self.assertTrue(any("pooled price" in r for r in result["reasons"]))

    def test_a_farther_place_stops_on_accuracy_and_price(self):
        result = evaluate(five(shift=-3.0))
        self.assertEqual(result["verdict"], STOP)
        self.assertLess(result["pooled_price_points"], 0.0)
        self.assertTrue(any("informative episodes pooled" in r for r in result["reasons"]))

    def test_one_run_that_prices_at_a_loss_stops_a_gate_that_passes_pooled(self):
        runs = five()[:4] + [synth_run("bad/full", shift=-0.5)]
        result = evaluate(runs)
        self.assertEqual(result["verdict"], STOP)
        self.assertTrue(any(r.startswith("bad/full prices at") for r in result["reasons"]))

    def test_a_run_with_too_few_informative_episodes_is_not_measured(self):
        result = evaluate(five(shift=0.5))
        self.assertEqual(result["verdict"], STOP)
        self.assertTrue(any("not measured" in r for r in result["reasons"]))

    def test_a_curve_that_does_not_fall_is_stop_whatever_the_price(self):
        """ADR-0031: a reach rate that rises with distance means no return to a nearer
        place can pay. Here the loudest place is FARTHER, so the price comes out a gain
        and the accuracy test is the only other thing that would stop it."""
        result = evaluate(five(shift=-3.0, slope=0.6, intercept=-4.0))
        self.assertEqual(result["verdict"], STOP)
        self.assertGreater(result["curve"]["b"], 0.0)
        self.assertTrue(any("does not fall" in r for r in result["reasons"]))

    def test_too_few_eligible_episodes_is_not_run(self):
        self.assertEqual(evaluate(five(eligible=90))["verdict"], NOT_RUN)

    def test_a_detector_that_fires_on_sound_is_not_run(self):
        result = evaluate(five(false_fires=20))  # 20 of 150, over 10%
        self.assertEqual(result["verdict"], NOT_RUN)
        self.assertTrue(any("while the source sounded" in r for r in result["reasons"]))

    def test_no_silent_reach_is_not_run(self):
        result = evaluate(five(reach=False))
        self.assertEqual(result["verdict"], NOT_RUN)
        self.assertIsNone(result["curve"])

    def test_fewer_renders_than_the_adr_names_is_not_run(self):
        result = evaluate(five()[:4])
        self.assertEqual(result["verdict"], NOT_RUN)
        self.assertTrue(any("names 5 renders" in r for r in result["reasons"]))

    def test_a_false_fire_is_priced_but_not_fitted(self):
        result = evaluate(five(false_fires=10))
        self.assertEqual(result["n_fit"], 5 * 140)
        self.assertEqual(result["n_eligible"], 5 * 150)


class TestTheRunsOnDisk(unittest.TestCase):
    """Through the real writers, because that seam is where a replay that is right over
    injected episodes finds nothing on disk and reports it as a finding."""

    def _arm(self, root, name, audits, *, audio=None, report=True):
        arm = pathlib.Path(root) / name / "full"
        scene = arm / "AAAscene"
        scene.mkdir(parents=True)
        for index, audit in enumerate(audits):
            write_episode(str(scene), index, AgentReport(resumed=True), audit)
        if report:
            recorded = {"temporal_coherence": False, "pre_onset_rms_tol": TOL}
            recorded.update(audio or {})
            (scene / ENV_REPORT_NAME).write_text(
                json.dumps({"run_config": {"audio": recorded}}), encoding="utf-8")
        return arm

    def _healthy(self, root, name):
        return self._arm(root, name, [
            silent_episode(AWAY_WALK, offset_at=30, index=0),
            silent_episode(AWAY_WALK, offset_at=30, index=1, stop_at=37),
        ])

    def _main(self, *argv):
        out = StringIO()
        with redirect_stdout(out):
            code = main([str(a) for a in argv])
        return code, out.getvalue()

    def test_a_run_loads_every_episode(self):
        run = load_run(str(self._healthy(tempfile.mkdtemp(), "legs-1")))
        self.assertEqual(run.label, "legs-1/full")
        self.assertEqual(run.refusals, ())
        self.assertEqual([s.status for s in run.episodes], [ELIGIBLE, ELIGIBLE])
        self.assertTrue(run.episodes[1].reached_after)

    def test_five_small_runs_print_every_section_and_read_not_run(self):
        root = tempfile.mkdtemp()
        code, text = self._main(*(self._healthy(root, "r{}".format(i)) for i in range(5)))
        self.assertEqual(code, 2)
        for heading in ("WHERE SILENCE FINDS THE AGENT", "THE SPLIT",
                        "WHAT THE LOUDEST PLACE KNOWS", "LOUDEST AGAINST NEAREST",
                        "THE OFFSET DETECTOR", "THE RETURN", "THE PRICE",
                        "BRANCH: NOT_RUN"):
            self.assertIn(heading, text)
        self.assertIn("eligible episode(s), under 100", text)

    def test_json_carries_the_gate_and_every_episode(self):
        root = tempfile.mkdtemp()
        code, text = self._main(self._healthy(root, "r1"), "--json")
        self.assertEqual(code, 2)
        payload = json.loads(text)
        self.assertEqual(payload["gate"]["verdict"], NOT_RUN)
        self.assertEqual(len(payload["episodes"]), 2)

    def test_a_render_with_the_preset_key_on_is_refused(self):
        arm = self._arm(tempfile.mkdtemp(), "old", [silent_episode(AWAY_WALK, offset_at=30)],
                        audio={"temporal_coherence": None})
        code, text = self._main(arm)
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", text)
        self.assertIn("temporal_coherence None", text)
        self.assertIn("BRANCH: NOT_RUN", text)

    def test_another_arms_rule_is_refused_by_name(self):
        arm = self._arm(tempfile.mkdtemp(), "x", [
            silent_episode(AWAY_WALK, offset_at=30, arm={"cast_policy": "read_legs"})])
        run = load_run(str(arm))
        self.assertTrue(any("cast_policy is 'read_legs'" in r for r in run.refusals))

    def test_a_run_with_no_loop_length_is_refused(self):
        arm = self._arm(tempfile.mkdtemp(), "x", [
            silent_episode(AWAY_WALK, offset_at=30, folds=0)])
        run = load_run(str(arm))
        self.assertTrue(any("cue_phase_folds" in r for r in run.refusals))

    def test_a_scene_with_no_env_report_is_refused(self):
        arm = self._arm(tempfile.mkdtemp(), "x", [silent_episode(AWAY_WALK, offset_at=30)],
                        report=False)
        run = load_run(str(arm))
        self.assertTrue(any(ENV_REPORT_NAME in r for r in run.refusals))

    def test_the_same_run_twice_is_refused(self):
        arm = self._healthy(tempfile.mkdtemp(), "r1")
        code, text = self._main(arm, arm)
        self.assertEqual(code, 2)
        self.assertIn("given twice", text)


if __name__ == "__main__":
    unittest.main()
