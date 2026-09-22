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
    leg_t,
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
    SoundingWindowRecord,
    StepRecord,
)
from earshot.tools.leg_replay import (
    BUILD,
    COMPLETED,
    CUT_BY_END,
    CUT_BY_STOP,
    CUT_BY_SURGE,
    ONE_BRANCH,
    SILENT,
    SOUNDING,
    SPANS_OFFSET,
    STOP,
    UNVERIFIED,
    Leg,
    SCAN_READINGS,
    RunReplay,
    by_cut_scan,
    by_line,
    by_scan,
    by_sounding,
    episode_legs,
    evaluate_gate,
    format_report,
    load_run,
    loop_removed,
    main,
    same_phase_change,
    same_phase_t,
    score,
    sounding_state,
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
            routed=True, scatter=EPS, arm=None, offset_at=None, tail=3, loop=(),
            folds=None, route=abs, drift=0.0, walk_loss=0.0):
    """An audit whose detour `realizable_action` the controller wrote, tick by tick.

    ``gain`` is how much louder the cue gets per metre nearer; negative is a field that
    gets quieter on the approach. ``surge_at`` steps the level up from that detour step
    on, so the rule surges. ``stop_at`` is the detour step the confirm fires on.
    ``tamper_at`` records the wrong action at that detour step, and nothing else changes.
    ``offset_at`` is the detour step the source stops on: from there the cue is a flat
    bed with no gradient, and the audit carries the window that says so.
    ``loop`` is added to the level while the source sounds, one entry per step from the
    detour's first, repeating: the clip's loop. ``folds`` is the period the record states.
    ``route`` maps a position's x to its route length. The default is the straight line,
    and anything else is a house whose walls make the walk differ from the line.
    ``drift`` moves the level per detour step whatever the agent does: a trend in time.
    ``walk_loss`` takes it down per forward walked and never while standing: a fall
    that motion causes.
    """
    offset = None if offset_at is None else PRE + offset_at
    levels, rows, plateau = [], [], 0
    eps = climb_eps(scatter)  # what the runner hands the rule, fallback included
    walked = [0]  # forwards taken before each detour reading
    for a, b in zip(xs, xs[1:]):
        walked.append(walked[-1] + int(round(abs(b - a) / 0.25)))
    for i, x in enumerate([xs[0]] * PRE + list(xs)):
        k = i - PRE
        level = 0.1 - gain * x + NOISE[i % len(NOISE)]
        if loop and k >= 0:
            level += loop[k % len(loop)]
        if k >= 0:
            level += drift * k - walk_loss * walked[k]
        if offset is not None and i >= offset:
            level = 0.02 + NOISE[i % len(NOISE)]  # the bed: no source, no gradient
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
            source_playing=i >= PRE and (offset is None or i < offset),
            displacement_m=0.25, geodesic_to_source=route(x) if routed else None,
            realizable_action=recorded))
        if action == ACT_STOP:
            break
    fields = dict(FULL, **(arm or {}))
    return EpisodeAudit(
        episode_index=index, source_xyz=SOURCE,
        funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
        onset=OnsetRecord(onset_step=PRE),
        sounding_window=None if offset is None else SoundingWindowRecord(
            opens_at=PRE, offset_step=offset, policy="fixed_steps", cue_tail_steps=tail),
        calibration=None if scatter is None else CalibrationRecord(
            onset_rms=0.01, bed_rms=0.001, separation_db=40.0, n_poses=16,
            global_volume=1.0, cue_render_scatter=scatter, cue_scatter_repeats=12),
        steps=tuple(rows),
        metrics={} if folds is None else {"sounding_phase_folds": float(folds)},
        **fields)


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


WINDOW = SoundingWindowRecord(opens_at=0, offset_step=20, policy="fixed_steps",
                              cue_tail_steps=3)  # the cue is exactly the bed from step 22


class TestTheSoundingSplit(unittest.TestCase):
    """Is the pull toward QUIETER the leg, or the source stopping under it?

    The fence posts are smoke criterion 4's: `offset_step` is the first silent step, and
    the cue carries the room's tail until `offset_step + cue_tail_steps - 1`."""

    def test_a_leg_read_before_the_offset_is_sounding(self):
        self.assertEqual(sounding_state(10, 19, WINDOW), SOUNDING)

    def test_a_leg_whose_last_reading_is_the_offset_step_spans_it(self):
        self.assertEqual(sounding_state(10, 20, WINDOW), SPANS_OFFSET)

    def test_a_leg_that_opens_inside_the_cue_tail_spans_it(self):
        """No step of it had the source playing, and the room was still falling."""
        self.assertEqual(sounding_state(21, 29, WINDOW), SPANS_OFFSET)

    def test_a_leg_that_opens_on_the_bed_is_silent(self):
        self.assertEqual(sounding_state(22, 30, WINDOW), SILENT)

    def test_a_continuous_source_never_stops(self):
        continuous = SoundingWindowRecord(opens_at=0, policy="continuous")
        self.assertEqual(sounding_state(100, 200, continuous), SOUNDING)

    def test_a_record_that_cannot_say_is_unknown_rather_than_sounding(self):
        """No window, or no cue tail (before ADR-0019): guessing SOUNDING would put
        legs read after the offset into the state that is meant to exclude them."""
        self.assertIsNone(sounding_state(10, 19, None))
        self.assertIsNone(sounding_state(10, 19, SoundingWindowRecord(
            opens_at=0, offset_step=20, policy="fixed_steps")))

    def test_the_legs_of_one_episode_fall_on_both_sides(self):
        """Offset at detour step 20: the leg read over 7-15 sounds, 16-24 spans it,
        and 25-33 opens after the tail, at 22."""
        xs = cast_xs(6.0, [TOWARD] * 4, 40)
        legs = [leg for leg in replay(episode(xs, offset_at=20)).legs
                if leg.outcome == COMPLETED]
        self.assertEqual([leg.sounding for leg in legs], [SOUNDING, SPANS_OFFSET, SILENT])

    def test_the_split_puts_a_stopping_source_where_it_lives(self):
        """Every leg walks TOWARD the source. The one the source stops under reads
        quieter anyway, and the split shows it there and not in the sounding state.
        Pooled, the two cancel into a median that says nothing about either."""
        xs = cast_xs(6.0, [TOWARD] * 4, 40)
        legs = replay(episode(xs, offset_at=20)).legs
        sounding, spans = (
            next(leg for leg in legs if leg.sounding == state)
            for state in (SOUNDING, SPANS_OFFSET))
        self.assertGreater(sounding.t, 2.5)
        self.assertLess(spans.t, -2.5, "the source stopping must read as quieter")
        run = RunReplay("r/full", "r/full", 1, 1, legs, 0, 0, 0, ())
        states = by_sounding([run])
        self.assertGreater(states[SOUNDING]["median_t_approached"], 0)
        self.assertLess(states[SPANS_OFFSET]["median_t_approached"], 0)
        self.assertEqual(states["unknown"]["n_completed"], 0)

    def test_the_section_says_it_is_not_a_gate(self):
        xs = cast_xs(6.0, [TOWARD] * 4, 40)
        run = RunReplay("r/full", "r/full", 1, 1, replay(episode(xs, offset_at=20)).legs,
                        0, 0, 0, ())
        gate = evaluate_gate({run.label: run.legs})
        text = format_report([run], gate, display_t_leg=2.5, display_why="test")
        self.assertIn("BY SOUNDING STATE. NOT A GATE", text)
        self.assertIn("the grid, sounding legs only", text)
        self.assertNotIn("  unknown", text, "no unknown row when every leg is known")


LEG_XS = [0.25 * j for j in range(PERIOD)]  # one clear leg's nine displacements
FOLDS = 5  # the box's loop period
RING = (2.0, 0.0, 0.0, 0.0, 0.0)  # a bursty clip: one fold in five rings
SPIKES = [RING[j % FOLDS] for j in range(PERIOD)]  # rings at readings 0 and 5


class TestTheLoopRemoved(unittest.TestCase):
    """Is the pull on sounding legs the clip's loop? The same-phase reader pairs each
    reading with the one a whole loop later, where the source emits the same fold."""

    def _trend(self, gain, loop=()):
        return [0.1 + gain * x + NOISE[j] + (loop[j] if loop else 0.0)
                for j, x in enumerate(LEG_XS)]

    def test_a_level_that_repeats_with_the_loop_cancels(self):
        clean = same_phase_t(LEG_XS, self._trend(0.01), lag=FOLDS)
        looped = same_phase_t(LEG_XS, self._trend(0.01, SPIKES), lag=FOLDS)
        self.assertGreater(clean, 2.5)
        self.assertAlmostEqual(looped, clean, places=6)

    def test_the_same_loop_reverses_the_nine_reading_fit(self):
        """The forced failure the reader exists for. The leg got louder, and the fit
        reads it quieter because the loop rang on its first reading and its sixth."""
        self.assertGreater(leg_t(LEG_XS, self._trend(0.01)), 2.5)
        self.assertLess(leg_t(LEG_XS, self._trend(0.01, SPIKES)), 0.0)

    def test_a_leg_that_got_quieter_still_reads_down(self):
        self.assertLess(same_phase_t(LEG_XS, self._trend(-0.01, SPIKES), lag=FOLDS), -2.5)

    def test_a_loop_alone_reads_nothing(self):
        self.assertIsNone(same_phase_t(LEG_XS, [0.1 + s for s in SPIKES], lag=FOLDS))

    def test_a_loop_too_long_for_the_leg_reads_nothing(self):
        """Nine readings at a period of 8 leave one pair, and one pair has no residual."""
        self.assertIsNone(same_phase_t(LEG_XS, self._trend(0.01), lag=8))

    def test_malformed_input_is_refused(self):
        with self.assertRaises(ValueError):
            same_phase_t(LEG_XS, self._trend(0.01)[:-1], lag=FOLDS)
        with self.assertRaises(ValueError):
            same_phase_t(LEG_XS, self._trend(0.01), lag=0)

    def _legs(self, **kwargs):
        xs = cast_xs(6.0, [TOWARD] * 4, 40)
        return replay(episode(xs, offset_at=20, loop=RING, **kwargs)).legs

    def test_only_a_sounding_leg_gets_the_same_phase_reading(self):
        legs = [l for l in self._legs(folds=FOLDS) if l.outcome == COMPLETED]
        self.assertEqual([l.sounding for l in legs], [SOUNDING, SPANS_OFFSET, SILENT])
        self.assertEqual({l.phase_folds for l in legs}, {FOLDS})
        self.assertGreater(legs[0].t_same_phase, 2.5)
        self.assertIsNone(legs[1].t_same_phase, "the loop stopped with the source")
        self.assertIsNone(legs[2].t_same_phase)

    def test_a_record_with_no_period_is_counted_as_unrecorded(self):
        """Never read as a loop of some length: the section names it."""
        run = RunReplay("r/full", "r/full", 1, 1, self._legs(), 0, 0, 0, ())
        entry = loop_removed([run])
        self.assertEqual(entry["periods"], {"unrecorded": 1})
        self.assertEqual(entry["n_paired"], 0)
        self.assertIn("loop period: unrecorded on 1", self._text(run))

    def test_a_period_that_is_not_whole_steps_is_a_writer_fault(self):
        audit = episode(THREE_LEGS)
        broken = replace(audit, metrics={"sounding_phase_folds": 4.5})
        with self.assertRaises(ValueError):
            replay(broken)

    def test_both_readers_are_read_over_the_same_legs(self):
        run = RunReplay("r/full", "r/full", 1, 1, self._legs(folds=FOLDS), 0, 0, 0, ())
        entry = loop_removed([run])
        self.assertEqual((entry["n_informative"], entry["n_paired"]), (1, 1))
        self.assertEqual(entry["periods"], {"5": 1})
        same = entry["readers"]["same_phase"]
        self.assertEqual(same["approached_read_up"], 1.0)
        self.assertIsNone(same["receded_read_down"], "no receding leg is not 0%")
        text = self._text(run)
        self.assertIn("THE LOOP, REMOVED. NOT A GATE", text)
        self.assertIn("loop period: 5 steps on 1", text)

    def _text(self, run):
        gate = evaluate_gate({run.label: run.legs})
        return format_report([run], gate, display_t_leg=2.5, display_why="test")


def around_a_wall(x):
    """A route that lengthens as the straight line shortens: the door is behind."""
    return 20.0 - x


class TestTheStraightLine(unittest.TestCase):
    """Is the pull the grader's axis? The same legs, graded on the horizontal straight
    line to the source in place of the route."""

    def _run(self, **kwargs):
        xs = cast_xs(6.0, [TOWARD, AWAY, TOWARD, AWAY], 40)
        audit = episode(xs, offset_at=30, loop=RING, folds=FOLDS, **kwargs)
        return RunReplay("r/full", "r/full", 1, 1, replay(audit).legs, 0, 0, 0, ())

    def test_a_leg_carries_its_line_distance(self):
        leg = replay(episode(THREE_LEGS)).legs[0]
        self.assertAlmostEqual(leg.line_start_m, THREE_LEGS[SCAN_STEPS + 1])
        self.assertAlmostEqual(leg.delta_line_m, -2.0)

    def test_a_record_with_no_source_has_no_line(self):
        audit = replace(episode(THREE_LEGS), source_xyz=None)
        leg = replay(audit).legs[0]
        self.assertIsNone(leg.delta_line_m)
        self.assertIsNotNone(leg.delta_route_m)

    def test_a_field_on_the_line_reads_right_there_and_wrong_on_the_route(self):
        """Both arms on one house. The cue follows the line, and the route runs the
        other way round a wall. Graded on the route, every leg reads wrong; graded on
        the line, every leg reads right."""
        run = self._run(route=around_a_wall)
        line = by_line([run])
        self.assertEqual((line["n_approached"], line["n_receded"]), (1, 1))
        self.assertEqual((line["n_both"], line["n_agree"]), (2, 0))
        on_line = line["readers"]["same_phase"]
        self.assertEqual((on_line["approached_read_up"], on_line["receded_read_down"]),
                         (1.0, 1.0))
        on_route = loop_removed([run])["readers"]["same_phase"]
        self.assertEqual((on_route["approached_read_up"], on_route["receded_read_down"]),
                         (0.0, 0.0))

    def test_where_route_and_line_agree_the_two_gradings_agree(self):
        run = self._run()
        line, route = by_line([run]), loop_removed([run])
        self.assertEqual((line["n_both"], line["n_agree"]), (2, 2))
        self.assertEqual(line["readers"], route["readers"])

    def test_a_leg_with_no_route_is_still_graded_on_the_line(self):
        run = self._run(routed=False)
        self.assertEqual(loop_removed([run])["n_informative"], 0)
        self.assertEqual(by_line([run])["n_informative"], 2)
        self.assertEqual(by_line([run])["n_both"], 0)

    def test_the_section_says_it_is_not_a_gate(self):
        run = self._run(route=around_a_wall)
        gate = evaluate_gate({run.label: run.legs})
        text = format_report([run], gate, display_t_leg=2.5, display_why="test")
        self.assertIn("BY STRAIGHT LINE. NOT A GATE", text)
        self.assertIn("route and line agree on 0 of the 2 legs", text)


class TestTheScanStanding(unittest.TestCase):
    """Does the level fall with time, or only while the agent walks? The scan's readings
    are all taken in one place, so the change over one loop there is time alone."""

    def test_the_change_over_a_loop_is_read_through_the_loop(self):
        levels = [1.0 + 0.01 * j + RING[j % FOLDS] for j in range(SCAN_READINGS)]
        self.assertAlmostEqual(same_phase_change(levels, lag=FOLDS),
                               0.05 / (sum(levels) / len(levels)))

    def test_a_loop_alone_changes_nothing(self):
        levels = [0.1 + RING[j % FOLDS] for j in range(SCAN_READINGS)]
        self.assertEqual(same_phase_change(levels, lag=FOLDS), 0.0)

    def test_no_pair_is_none_and_no_period_is_refused(self):
        self.assertIsNone(same_phase_change([0.1] * SCAN_READINGS, lag=SCAN_READINGS))
        with self.assertRaises(ValueError):
            same_phase_change([0.1] * SCAN_READINGS, lag=0)

    def _scans(self, **kwargs):
        for key, value in (("offset_at", 30), ("loop", RING), ("folds", FOLDS)):
            kwargs.setdefault(key, value)
        return replay(episode(THREE_LEGS, **kwargs))

    def test_the_scan_is_the_readings_before_the_first_forward(self):
        (scan,) = self._scans().scans
        self.assertEqual((scan.start_step, scan.first, scan.complete), (PRE, True, True))
        self.assertEqual((scan.static, scan.sounding, scan.phase_folds),
                         (True, SOUNDING, FOLDS))
        self.assertIsNotNone(scan.change)

    def _run(self, **kwargs):
        # A flat field with no loop, so the one trend under test is the whole change.
        result = self._scans(gain=0.0, loop=(), **kwargs)
        return by_scan([RunReplay("r/full", "r/full", 1, 1, result.legs, 0, 0, 0, (),
                                  result.scans)])

    def test_a_level_that_falls_with_time_falls_standing_and_walking(self):
        rows = self._run(drift=-0.002)["rows"]
        self.assertEqual((rows["first_scan"]["n"], rows["first_scan"]["fell"]), (1, 1.0))
        self.assertEqual(rows["walking"]["fell"], 1.0)

    def test_a_level_that_falls_with_motion_falls_walking_only(self):
        """The forced failure the check exists for: every leg falls, and the scan,
        where the agent stood, reads no more than the noise."""
        rows = self._run(walk_loss=0.002)["rows"]
        self.assertEqual(rows["walking"]["fell"], 1.0)
        self.assertLess(rows["walking"]["median_change"], -0.1)
        self.assertLess(abs(rows["first_scan"]["median_change"]), 0.05)

    def test_a_scan_that_moved_is_not_read(self):
        audit = episode(THREE_LEGS, offset_at=30, loop=RING, folds=FOLDS)
        rows = list(audit.steps)
        rows[PRE + 3] = replace(rows[PRE + 3], position=Xyz(THREE_LEGS[3] + 0.25, 0.0, 0.0))
        (scan,) = replay(replace(audit, steps=tuple(rows))).scans
        self.assertEqual((scan.complete, scan.static, scan.change), (True, False, None))

    def test_a_scan_the_source_stopped_under_is_not_read(self):
        (scan,) = self._scans(offset_at=4).scans
        self.assertEqual(scan.sounding, SPANS_OFFSET)
        self.assertIsNone(scan.change)

    def test_a_surge_cuts_the_scan_and_the_next_one_is_a_later_scan(self):
        scans = self._scans(surge_at=3).scans
        self.assertEqual((scans[0].first, scans[0].complete), (True, False))
        self.assertEqual((scans[1].first, scans[1].complete), (False, True))

    def test_the_section_says_it_is_not_a_gate(self):
        result = self._scans()
        run = RunReplay("r/full", "r/full", 1, 1, result.legs, 0, 0, 0, (), result.scans)
        gate = evaluate_gate({run.label: run.legs})
        text = format_report([run], gate, display_t_leg=2.5, display_why="test")
        self.assertIn("THE SCAN, STANDING. NOT A GATE", text)
        self.assertIn("standing still: 1", text)
        self.assertIn("loop period: 5 steps on 1", text)


class TestTheScansTheSurgeCut(unittest.TestCase):
    """What the standing rows never saw. A scan completes only if no reading through its
    turns read as rising, so `by_scan` grades a population selected against rises."""

    def _scans(self, **kwargs):
        for key, value in (("offset_at", 30), ("loop", RING), ("folds", FOLDS)):
            kwargs.setdefault(key, value)
        return replay(episode(THREE_LEGS, **kwargs))

    def _runs(self, *results):
        return [RunReplay("r{}/full".format(i), "r/full", 1, 1, r.legs, 0, 0, 0, (),
                          r.scans)
                for i, r in enumerate(results)]

    def test_a_complete_scan_names_itself_and_keeps_its_change(self):
        (scan,) = self._scans().scans
        self.assertEqual((scan.outcome, scan.n_readings), (COMPLETED, SCAN_READINGS))
        self.assertIsNotNone(scan.change)
        self.assertIsNone(scan.cut_change)

    def test_a_surge_cuts_the_scan_and_the_reading_it_cut_on_counts(self):
        """The record is written at render time and the action follows it, so the agent
        was still standing in the scan when the cutting reading was taken."""
        first, later = self._scans(surge_at=6).scans[:2]
        self.assertEqual((first.complete, first.outcome, first.n_readings),
                         (False, CUT_BY_SURGE, 7))
        self.assertIsNone(first.change)
        self.assertIsNotNone(first.cut_change)
        self.assertGreater(first.cut_change, 0.0)  # a surge is a rise, and it reads as one
        self.assertEqual((later.first, later.complete), (False, True))

    def test_a_scan_cut_inside_its_first_loop_has_no_pair_and_is_not_invented(self):
        # The level steps up at detour step 3 and the 5-against-5 window reads it at 4,
        # so the scan keeps five readings: one short of the pair a loop needs.
        (first, *_rest) = self._scans(surge_at=3).scans
        self.assertEqual((first.outcome, first.n_readings), (CUT_BY_SURGE, 5))
        self.assertIsNone(first.cut_change)

    def test_a_confirm_is_a_stop_and_a_tampered_record_is_unverified(self):
        self.assertEqual(self._scans(stop_at=2).scans[0].outcome, CUT_BY_STOP)
        self.assertEqual(self._scans(tamper_at=2).scans[0].outcome, UNVERIFIED)

    def test_a_scan_the_detour_ran_out_under_is_cut_by_the_end(self):
        audit = episode(THREE_LEGS, offset_at=30, loop=RING, folds=FOLDS)
        (first, *_rest) = replay(replace(audit, steps=audit.steps[:PRE + 4])).scans
        self.assertEqual((first.outcome, first.n_readings), (CUT_BY_END, 4))

    def test_the_section_counts_every_cut_and_grades_the_surges(self):
        entry = by_cut_scan(self._runs(self._scans(), self._scans(surge_at=6)))
        counts = entry["outcomes"]["first"]
        self.assertEqual((counts[COMPLETED], counts[CUT_BY_SURGE]), (1, 1))
        self.assertEqual(sum(counts.values()), 2)
        self.assertEqual(entry["rows"]["first_complete"]["n"], 1)
        self.assertEqual(entry["rows"]["first_surge"]["n"], 1)
        self.assertEqual(entry["rows"]["first_surge"]["rose"], 1.0)
        self.assertEqual(entry["n_too_short"]["first"], 0)
        self.assertEqual(entry["median_readings"]["first"], 7)

    def test_a_scan_cut_too_early_is_counted_and_never_graded(self):
        """The forced failure: a cut with no whole loop in it must not reach a row."""
        entry = by_cut_scan(self._runs(self._scans(surge_at=3)))
        self.assertEqual(entry["outcomes"]["first"][CUT_BY_SURGE], 1)
        self.assertEqual(entry["rows"]["first_surge"]["n"], 0)
        self.assertIsNone(entry["rows"]["first_surge"]["median_change"])
        self.assertEqual(entry["n_too_short"]["first"], 1)
        self.assertIsNone(entry["median_readings"]["first"])

    def test_the_section_says_it_is_not_a_gate_and_prints_both_populations(self):
        runs = self._runs(self._scans(), self._scans(surge_at=6))
        gate = evaluate_gate({run.label: run.legs for run in runs})
        text = format_report(runs, gate, display_t_leg=2.5, display_why="test")
        self.assertIn("THE SCANS THE SURGE CUT. NOT A GATE", text)
        self.assertIn("first, cut by a surge", text)
        self.assertIn("too short to grade: first 0", text)
        self.assertIn("surge 1", text)


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
        # These fixtures carry no window, so every completed leg is unknown, not sounding.
        self.assertEqual(payload["by_sounding"]["unknown"]["n_completed"], 2 + 2 + 3)
        self.assertEqual(payload["by_sounding"]["sounding"]["n_completed"], 0)
        self.assertEqual(payload["loop_removed"]["n_informative"], 0)
        self.assertEqual(payload["by_line"]["n_informative"], 0)
        self.assertIn("delta_line_m", payload["legs"][0])
        # One scan per episode, opened on the detour's first step.
        self.assertEqual(payload["by_scan"]["n_started"], 3)
        self.assertEqual(len(payload["scans"]), 3)

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
