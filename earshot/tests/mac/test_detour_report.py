"""The detour arithmetic, against injected records — the runs need a GPU, the maths does not.

`trace_one` and `aggregate` turn per-step audits into the one measurement that separates
yield-1's two candidate diagnoses: *were the twelve abandoned detours walking at the
source when the budget cut them off, or were they wandering?* Ticket 19's third row,
applied to a diagnosis rather than to a gate — given records that say the agent walked
twenty metres and closed one, does the report say so, or does the number come out looking
like a converging climb that ran out of steps.

The two arms are the point. A synthetic "abandoned" trace proves nothing on its own; the
reached ones are the control, and this asserts they are reported side by side (CLAUDE.md:
a claim that X failed because of Y needs the arm where Y is absent).
"""

import inspect
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.controller import (
    ACT_FORWARD,
    ACT_STOP,
    ACT_TURN_LEFT,
    ACT_TURN_RIGHT,
    RISING_WINDOW,
    next_plateau_steps,
    realizable_investigate_step,
)
from earshot.report.audit import (
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    StepRecord,
)
from earshot.tools.detour_report import (
    ABANDONED,
    ARRIVAL_RING_M,
    NO_DETOUR,
    REACHED,
    RISING_EPS,
    aggregate,
    band_rows,
    fit_slope,
    format_report,
    plateau_index,
    plateau_windows,
    rising_flags,
    rule_action,
    trace_one,
)
from earshot.types import Xyz

SOURCE = Xyz(0.0, 0.0, 0.0)
ONSET_STEP = 5


def steps(distances, *, onset=ONSET_STEP, displacement=0.25, collided=False):
    """One step per distance, laid out along +x so `horizontal_distance_to` is the number.

    `displacement` is what the agent actually moved that step, which is deliberately NOT
    derived from the distance series: an agent can walk a metre and end up no closer, and
    that gap between path length and progress is exactly what the report measures.
    """
    return tuple(
        StepRecord(step=onset + i, measured_rms=0.05, position=Xyz(float(d), 0.0, 0.0),
                   displacement_m=displacement, collided=collided)
        for i, d in enumerate(distances)
    )


def audit(distances, *, stage, index=0, **kwargs):
    return EpisodeAudit(
        episode_index=index,
        source_xyz=SOURCE,
        funnel_stage=stage,
        onset=OnsetRecord(onset_step=ONSET_STEP),
        steps=steps(distances, **kwargs),
    )


class TestOneDetour(unittest.TestCase):
    def test_a_climb_that_walked_straight_at_the_source_reads_near_one(self):
        """8 m closed over 8 m walked: short of steps, not lost."""
        row = trace_one(audit([10.0, 8.0, 6.0, 4.0, 2.0], stage=FunnelStage.INVESTIGATE_ENTERED,
                              displacement=2.0))
        self.assertEqual(row["outcome"], ABANDONED)
        self.assertAlmostEqual(row["gap_closed_m"], 8.0)
        self.assertAlmostEqual(row["walked_m"], 10.0)
        self.assertAlmostEqual(row["walked_per_metre_closed"], 1.25)

    def test_a_wander_reads_an_order_of_magnitude_higher(self):
        """Twenty steps of real movement, half a metre closed. No threshold needed to see it."""
        row = trace_one(audit([6.0] * 10 + [5.5] * 10, stage=FunnelStage.INVESTIGATE_ENTERED,
                              displacement=1.0))
        self.assertAlmostEqual(row["gap_closed_m"], 0.5)
        self.assertAlmostEqual(row["walked_m"], 20.0)
        self.assertAlmostEqual(row["walked_per_metre_closed"], 40.0)

    def test_a_stuck_agent_shows_it_in_walked_and_in_collisions(self):
        row = trace_one(audit([6.0] * 20, stage=FunnelStage.INVESTIGATE_ENTERED,
                              displacement=0.0, collided=True))
        self.assertEqual(row["walked_m"], 0.0)
        self.assertEqual(row["gap_closed_m"], 0.0)
        self.assertEqual((row["n_collided"], row["n_moves"]), (20, 20))
        self.assertIsNone(row["walked_per_metre_closed"],
                          "a gap of zero has no metres-per-metre; a number here would be "
                          "a division that invented progress")

    def test_reaching_the_source_is_the_other_arm(self):
        row = trace_one(audit([6.0, 3.0, 0.4], stage=FunnelStage.PRIMARY_RESUMED))
        self.assertEqual(row["outcome"], REACHED)
        self.assertTrue(row["walked_is_upper_bound"],
                        "the window overshoots into the resumed primary search, and the "
                        "report has to say so rather than quietly over-count the detour")

    def test_an_episode_that_never_diverted_has_no_detour(self):
        row = trace_one(audit([6.0], stage=FunnelStage.ONSET_FIRED))
        self.assertEqual(row["outcome"], NO_DETOUR)
        self.assertNotIn("walked_m", row)

    def test_the_budget_clips_the_window(self):
        """An abandoned detour ends at the budget by definition; steps after it are the
        resumed primary search and are not the detour's cost."""
        row = trace_one(audit([9.0, 8.0, 7.0, 6.0, 5.0], stage=FunnelStage.INVESTIGATE_ENTERED),
                        budget=2)
        self.assertEqual(row["detour_steps"], 3)  # onset, +1, +2
        self.assertAlmostEqual(row["d_end_m"], 7.0)

    def test_records_without_a_position_read_absent_rather_than_zero(self):
        """Every audit written before StepRecord.position. A distance of n/a and a
        distance of zero are different claims and only one is a measurement."""
        bare = EpisodeAudit(
            source_xyz=SOURCE,
            funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
            onset=OnsetRecord(onset_step=ONSET_STEP),
            steps=tuple(StepRecord(step=ONSET_STEP + i, measured_rms=0.05,
                                   displacement_m=0.25) for i in range(5)),
        )
        row = trace_one(bare)
        self.assertIsNone(row["d_onset_m"])
        self.assertIsNone(row["gap_closed_m"])
        self.assertAlmostEqual(row["walked_m"], 1.25)  # still knows it moved


class TestTheTwoArms(unittest.TestCase):
    def test_the_arms_are_counted_and_reported_separately(self):
        traces = [
            trace_one(audit([10.0, 9.5], stage=FunnelStage.INVESTIGATE_ENTERED, index=i,
                            displacement=3.0))
            for i in range(12)
        ] + [
            trace_one(audit([6.0, 3.0, 0.4], stage=FunnelStage.PRIMARY_RESUMED, index=12 + i))
            for i in range(8)
        ]
        agg = aggregate(traces)
        self.assertEqual(agg["n_episodes"], 20)
        self.assertEqual(agg["arms"][ABANDONED]["n"], 12)
        self.assertEqual(agg["arms"][REACHED]["n"], 8)
        self.assertGreater(agg["arms"][ABANDONED]["walked_per_metre_closed"],
                           agg["arms"][REACHED]["walked_per_metre_closed"])

    def test_an_empty_run_aggregates_to_nothing_rather_than_crashing(self):
        agg = aggregate([])
        self.assertEqual(agg["n_episodes"], 0)
        self.assertIsNone(agg["arms"][ABANDONED]["walked_m"])

    def test_the_report_names_both_arms_and_flags_missing_positions(self):
        bare = EpisodeAudit(source_xyz=SOURCE, funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
                            onset=OnsetRecord(onset_step=ONSET_STEP),
                            steps=(StepRecord(step=ONSET_STEP, measured_rms=0.05),))
        text = format_report(aggregate([
            trace_one(audit([10.0, 5.0], stage=FunnelStage.INVESTIGATE_ENTERED)),
            trace_one(audit([6.0, 0.4], stage=FunnelStage.PRIMARY_RESUMED, index=1)),
            trace_one(bare),
        ]))
        self.assertIn(ABANDONED, text)
        self.assertIn(REACHED, text)
        self.assertIn("carry no per-step position", text)


def rms_steps(pairs, *, onset=ONSET_STEP, action=None, realizable=None, lateral=0):
    """Steps from explicit ``(distance, measured_rms)`` pairs.

    The helper above pins ``measured_rms`` at a constant because the questions it serves
    are about distance. The plateau questions are about the *energy*, so this one lets a
    test say what the render did at each pose — which is the whole input to `rising`.

    ``realizable`` takes one action for every step, or a sequence for one per step. The
    per-step form is what the reconstruction check needs: the FIRST detour step has no
    predecessor in the window and so is always rising, and a test that hands it a turn is
    asserting against the rule rather than against the tool.
    """
    per_step = realizable if isinstance(realizable, (list, tuple)) else [realizable] * len(pairs)
    return tuple(
        StepRecord(step=onset + i, measured_rms=float(rms), position=Xyz(float(d), 0.0, 0.0),
                   displacement_m=0.25, lateral_sign=lateral, action=action,
                   realizable_action=per_step[i])
        for i, (d, rms) in enumerate(pairs)
    )


def rms_audit(pairs, *, stage=FunnelStage.INVESTIGATE_ENTERED, index=0, **kwargs):
    return EpisodeAudit(
        episode_index=index,
        source_xyz=SOURCE,
        funnel_stage=stage,
        onset=OnsetRecord(onset_step=ONSET_STEP),
        steps=rms_steps(pairs, **kwargs),
    )


class TestTheRulesOwnPredicate(unittest.TestCase):
    """`rising` and the turn it implies, carried rather than re-invented."""

    def test_the_epsilon_is_the_rules_own_default(self):
        """The drift guard. Two copies of this constant is two controllers."""
        signature = inspect.signature(realizable_investigate_step)
        self.assertEqual(RISING_EPS, signature.parameters["eps"].default)

    def test_the_first_reading_is_rising_and_a_repeat_is_not(self):
        self.assertEqual(rising_flags([0.05, 0.05]), [True, False])

    def test_a_rise_inside_epsilon_does_not_count(self):
        """The rule needs `current > previous + eps`, not merely `>`."""
        self.assertEqual(rising_flags([0.05, 0.05 + RISING_EPS / 2]), [True, False])
        self.assertEqual(rising_flags([0.05, 0.05 + RISING_EPS * 10]), [True, True])

    def test_the_rule_forwards_while_rising_and_turns_by_the_lateral_sign(self):
        self.assertEqual(rule_action(True, -1), ACT_FORWARD)
        self.assertEqual(rule_action(False, 1), ACT_TURN_RIGHT)
        self.assertEqual(rule_action(False, -1), ACT_TURN_LEFT)

    def test_an_ambiguous_or_absent_sign_scans_left_like_the_rule_does(self):
        self.assertEqual(rule_action(False, 0), ACT_TURN_LEFT)
        self.assertEqual(rule_action(False, None), ACT_TURN_LEFT)

    def test_the_reconstruction_matches_the_rule_it_reconstructs(self):
        """Both halves against the real function, not against a copy of its body."""
        history = [0.05, 0.05, 0.06]
        for i in range(1, len(history) + 1):
            window = history[:i]
            flag = rising_flags(window)[-1]
            self.assertEqual(
                rule_action(flag, -1),
                realizable_investigate_step(window, -1, False),
            )

    def test_the_reconstruction_follows_the_cast_through_a_whole_plateau(self):
        """The replay must model the un-cued branch, which is no longer just a turn.

        A reconstruction that still answered TURN on every dead step would disagree with
        the agent on most of every plateau — and this file's whole value is that the
        recomputed action can be checked against the recorded one.
        """
        flat = [0.0100, 0.0102, 0.0098, 0.0101, 0.0099, 0.0103] * 4
        flags = rising_flags(flat, eps=2.8e-3)
        indices = plateau_index(flags)
        for i, (flag, index) in enumerate(zip(flags, indices)):
            self.assertEqual(
                rule_action(flag, 1, plateau_steps=index),
                realizable_investigate_step(
                    flat[: i + 1], 1, False, eps=2.8e-3, plateau_steps=index),
                "step {} disagrees with the rule".format(i))
        self.assertIn(ACT_FORWARD,
                      [rule_action(f, 1, plateau_steps=i)
                       for f, i in zip(flags, indices)],
                      "a plateau this long must contain a cast")

    def test_the_replays_counter_is_the_one_the_agent_kept(self):
        """`plateau_index` against `next_plateau_steps` iterated, which is what the runner
        threads through `ControllerState`. Two ways of counting the same thing, and if
        they ever disagree the reconstruction check silently measures nothing."""
        series = [0.010, 0.011, 0.011, 0.011, 0.030, 0.030, 0.030, 0.031]
        flags = rising_flags(series, eps=2.8e-3)
        derived = plateau_index(flags)
        threaded = []
        running = 0
        for i in range(len(series)):
            threaded.append(running)
            running = next_plateau_steps(
                series[: i + 1], eps=2.8e-3, plateau_steps=running)
        self.assertEqual(derived, threaded)


class TestTheReplayFollowsTheRuleThatRan(unittest.TestCase):
    """The drift this file did not catch the first time.

    `rising_flags` used to carry the rule's body by hand. When `3f26572` replaced the
    single-step comparison with a median-of-`RISING_WINDOW` baseline, the copy here did
    not move, and the guard above still passed because it only checks that the *constant*
    is not re-spelled. A replay of that run would have reconstructed windows for a
    controller that never ran and read as evidence the fix did nothing.

    Both arms per ADR-0014: the series where the two rules DISAGREE (so the assertion can
    fail if the replay reverts), and the short series where they agree by construction.
    """

    # A live climb with one unlucky render at index 5 — the case the fix exists for.
    # Single-step: 0.0535 < 0.0540, not rising, turn. Windowed: the median of the five
    # before it is 0.0520, so the climb holds.
    DIPPED = [0.0500, 0.0510, 0.0520, 0.0530, 0.0540, 0.0535, 0.0555]

    def test_the_replay_holds_the_climb_where_the_single_step_rule_turned(self):
        flags = rising_flags(self.DIPPED)
        single_step = [
            i == 0 or value > self.DIPPED[i - 1] + RISING_EPS
            for i, value in enumerate(self.DIPPED)
        ]
        self.assertTrue(flags[5], "the windowed rule holds a climb through one bad render")
        self.assertFalse(single_step[5], "and the rule it replaced did not — the arms differ")
        self.assertNotEqual(flags, single_step)

    def test_the_replay_equals_the_rule_step_for_step(self):
        """Against the real function over the whole series, not a copy of its body."""
        flags = rising_flags(self.DIPPED)
        for i in range(1, len(self.DIPPED) + 1):
            self.assertEqual(
                rule_action(flags[i - 1], -1),
                realizable_investigate_step(self.DIPPED[:i], -1, False),
                "step {} disagrees with the rule".format(i - 1),
            )

    def test_the_runners_history_is_long_enough_for_the_window(self):
        """The rule reads `2 * window` entries; the runner trims to `ENERGY_HISTORY`.

        If the reach ever outgrows the trim, the agent judges its rise against a baseline
        shorter than the one this replay uses — the replay sees the whole series and the
        agent saw a truncated one — and every plateau ever reported goes wrong silently.
        It has already happened once: `ENERGY_HISTORY` was 8 against a rule that needed 6,
        and the two-sided rule needs 10. Held here rather than in prose.
        """
        from earshot.task.runner import ENERGY_HISTORY

        self.assertLessEqual(2 * RISING_WINDOW, ENERGY_HISTORY)


class TestTheThresholdComesFromTheEpisode(unittest.TestCase):
    """`eps` is the episode's measured scatter, not a constant — and the replay reads it.

    A rise of 1e-3 per step is a live climb under the unmeasured fallback and noise under
    a renderer that scatters 2.8e-3. Same trace, two thresholds, opposite readings: which
    one a replay uses is not a detail.
    """

    RISES = [(6.0, 0.0500), (5.75, 0.0510), (5.5, 0.0520), (5.25, 0.0530)]

    def _audit(self, scatter):
        return EpisodeAudit(
            episode_index=0,
            source_xyz=SOURCE,
            funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
            onset=OnsetRecord(onset_step=ONSET_STEP),
            steps=rms_steps(self.RISES),
            calibration=None if scatter is None else CalibrationRecord(
                onset_rms=0.01, bed_rms=0.001, separation_db=40.0, n_poses=16,
                global_volume=1.0, render_scatter=scatter, scatter_repeats=12),
        )

    def test_a_measured_floor_is_used_and_reported(self):
        row = trace_one(self._audit(2.8e-3))
        self.assertEqual(row["rising_eps"], 2.8e-3)
        self.assertTrue(row["eps_measured"])
        # 1e-3 a step cannot clear a 2.8e-3 floor: the whole detour plateaus.
        self.assertEqual(row["plateau_steps"], len(self.RISES) - 1)

    def test_an_unmeasured_episode_falls_back_and_says_so(self):
        row = trace_one(self._audit(None))
        self.assertEqual(row["rising_eps"], RISING_EPS)
        self.assertFalse(row["eps_measured"])
        # The same trace at the pre-detour-2 threshold is a climb with no plateau at all.
        self.assertEqual(row["plateau_steps"], 0)

    def test_the_report_discloses_which_threshold_was_in_force(self):
        text = format_report(aggregate([
            trace_one(self._audit(2.8e-3)),
            trace_one(self._audit(None)),
        ]))
        self.assertIn("1 of 2 measured", text)
        self.assertIn("UNMEASURED constant", text)


class TestTheDistanceAxis(unittest.TestCase):
    """Which distance the field was read against, chosen explicitly and printed.

    `eps-1` read an INVERTED gradient beyond 5 m on the horizontal axis. At that range the
    agent is usually in another room, where `xz` shrinks and the walk does not, so the
    record could not separate a real inversion from the axis failing. Both arms here: the
    route used where it exists, the horizontal fallback labelled and caveated where it
    does not.
    """

    PAIRS = [(6.0, 0.030), (5.0, 0.036), (4.0, 0.045), (3.0, 0.060)]

    def _audit(self, routes):
        steps = tuple(
            StepRecord(step=ONSET_STEP + i, measured_rms=rms,
                       position=Xyz(float(d), 0.0, 0.0), displacement_m=0.25,
                       geodesic_to_source=route)
            for i, ((d, rms), route) in enumerate(zip(self.PAIRS, routes)))
        return EpisodeAudit(
            episode_index=0, source_xyz=SOURCE,
            funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
            onset=OnsetRecord(onset_step=ONSET_STEP), steps=steps)

    def test_the_route_is_used_where_the_record_carries_it(self):
        """A pose 3 m away in xz and 11 m away by navmesh is 11 m from the source."""
        row = trace_one(self._audit([14.0, 13.0, 12.0, 11.0]))
        self.assertEqual(row["distance_axis"], "geodesic")
        self.assertEqual(row["d_min_m"], 11.0)

    def test_an_older_record_falls_back_and_is_labelled(self):
        row = trace_one(self._audit([None, None, None, None]))
        self.assertEqual(row["distance_axis"], "horizontal")
        self.assertEqual(row["d_min_m"], 3.0)

    def test_the_report_carries_the_caveat_only_on_the_horizontal_axis(self):
        horizontal = format_report(aggregate([trace_one(self._audit([None] * 4))]))
        self.assertIn("axis: horizontal", horizontal)
        self.assertIn("READ THE FAR BANDS WITH CARE", horizontal)

        routed = format_report(aggregate([trace_one(self._audit([14.0, 13.0, 12.0, 11.0]))]))
        self.assertIn("axis: geodesic", routed)
        self.assertNotIn("READ THE FAR BANDS WITH CARE", routed)

    def test_a_run_holding_both_kinds_of_record_says_so(self):
        """Pooling two axes into one band table would average two different measurements."""
        text = format_report(aggregate([
            trace_one(self._audit([14.0, 13.0, 12.0, 11.0])),
            trace_one(self._audit([None] * 4)),
        ]))
        self.assertIn("TWO AXES IN ONE REPORT", text)


class TestTheArrivalsThatWereRefused(unittest.TestCase):
    """An abandoned episode that stood inside the ring reached the source and was not
    counted, and the record proves it without needing `visual_confirm` recorded.

    The confirm is a pure function of distance — the oracle fires at `oracle_radius_m`
    geodesic to the anomaly object's view points, and the source position IS one of them.
    So an in-ring step HAD the confirm. The rule STOPs on confirm-and-not-rising and an
    abandoned episode never STOPped, so `rising` was true at every in-ring step: the
    climb's memory of the approach vetoing an arrival already made. `cast-1` put seven of
    fifteen abandoned episodes in `DYehNKdT76V` in exactly that position.
    """

    def _audit(self, routes, *, stage, rms=None, index=0):
        series = rms or [0.05] * len(routes)
        steps = tuple(
            StepRecord(step=ONSET_STEP + i, measured_rms=value,
                       position=Xyz(float(d), 0.0, 0.0), displacement_m=0.25,
                       geodesic_to_source=d)
            for i, (d, value) in enumerate(zip(routes, series)))
        return EpisodeAudit(
            episode_index=index, source_xyz=SOURCE, funnel_stage=stage,
            onset=OnsetRecord(onset_step=ONSET_STEP), steps=steps)

    def test_an_abandoned_episode_inside_the_ring_is_a_refused_arrival(self):
        row = trace_one(self._audit([4.0, 2.0, 0.9, 0.4],
                                    stage=FunnelStage.INVESTIGATE_ENTERED))
        self.assertTrue(row["arrival_refused"])
        self.assertEqual(row["n_steps_in_ring"], 2)

    def test_a_reached_episode_inside_the_ring_is_not_refused(self):
        """The control arm. Standing in the ring is only a finding when it was not counted."""
        row = trace_one(self._audit([4.0, 2.0, 0.9, 0.4],
                                    stage=FunnelStage.PRIMARY_RESUMED))
        self.assertFalse(row["arrival_refused"])
        self.assertEqual(row["n_steps_in_ring"], 2)

    def test_an_episode_that_never_entered_the_ring_is_not_refused(self):
        row = trace_one(self._audit([6.0, 5.0, 4.0, 3.0],
                                    stage=FunnelStage.INVESTIGATE_ENTERED))
        self.assertFalse(row["arrival_refused"])
        self.assertEqual(row["n_steps_in_ring"], 0)

    def test_the_in_ring_steps_that_read_rising_are_counted(self):
        """The mechanism, not just the count: a climbing series keeps `rising` true
        through the ring, which is the only way an in-ring step can refuse to STOP."""
        climbing = [0.010, 0.014, 0.018, 0.022, 0.026, 0.030]
        row = trace_one(self._audit([5.0, 4.0, 3.0, 2.0, 0.8, 0.3],
                                    stage=FunnelStage.INVESTIGATE_ENTERED,
                                    rms=climbing))
        self.assertEqual(row["n_steps_in_ring"], 2)
        self.assertEqual(row["n_in_ring_rising"], 2)

    def test_the_report_names_the_count_and_the_ring(self):
        text = format_report(aggregate([
            trace_one(self._audit([4.0, 0.9], stage=FunnelStage.INVESTIGATE_ENTERED)),
            trace_one(self._audit([4.0, 3.0], stage=FunnelStage.INVESTIGATE_ENTERED,
                                  index=1)),
        ]))
        self.assertIn("arrivals refused: 1 of 2", text)
        self.assertIn("{:.1f} m ring".format(ARRIVAL_RING_M), text)
        self.assertIn("LOWER BOUND", text)

    def test_a_horizontal_axis_record_is_caveated_rather_than_quoted(self):
        """The ring is geodesic. Counting against the derived horizontal distance measures
        a different thing, and the report has to say so before the number is used."""
        bare = EpisodeAudit(
            episode_index=0, source_xyz=SOURCE,
            funnel_stage=FunnelStage.INVESTIGATE_ENTERED,
            onset=OnsetRecord(onset_step=ONSET_STEP),
            steps=tuple(StepRecord(step=ONSET_STEP + i, measured_rms=0.05,
                                   position=Xyz(float(d), 0.0, 0.0), displacement_m=0.25)
                        for i, d in enumerate([4.0, 0.9])))
        text = format_report(aggregate([trace_one(bare)]))
        self.assertIn("arrivals refused: 1 of 1", text)
        self.assertIn("WRONG AXIS", text)


class TestTheFieldItself(unittest.TestCase):
    """Whether a forward step was worth anything, before any question about the rule.

    Both arms: a live gradient the threshold can see, and a flat field where no threshold
    setting recovers a climb because the field delivers nothing to clear it.
    """

    def test_a_live_gradient_reads_well_above_its_threshold(self):
        # 0.01 RMS per metre closed; a 0.25 m step buys 2.5e-3 against a 1e-3 floor.
        pairs = [(d, 0.10 - 0.01 * d) for d in (5.0, 4.5, 4.0, 3.5, 3.0)]
        row = [r for r in band_rows([d for d, _ in pairs], [v for _, v in pairs], eps=1e-3)
               if r["band"] == "3-5"][0]
        self.assertAlmostEqual(row["rise_over_eps"], 2.5, places=6)
        self.assertLess(row["slope_per_m"], 0.0, "negative slope is a live gradient")

    def test_a_flat_field_reads_at_zero_however_the_threshold_is_set(self):
        pairs = [(d, 0.05) for d in (5.0, 4.5, 4.0, 3.5, 3.0)]
        row = [r for r in band_rows([d for d, _ in pairs], [v for _, v in pairs], eps=1e-3)
               if r["band"] == "3-5"][0]
        self.assertAlmostEqual(row["rise_over_eps"], 0.0, places=9)

    def test_a_band_that_gets_quieter_on_approach_reads_negative_not_strong(self):
        """The sign arm. |slope| would report a null or an occluder as a powerful cue and
        send the next lever at the estimator."""
        pairs = [(d, 0.02 + 0.01 * d) for d in (5.0, 4.5, 4.0, 3.5, 3.0)]
        row = [r for r in band_rows([d for d, _ in pairs], [v for _, v in pairs], eps=1e-3)
               if r["band"] == "3-5"][0]
        self.assertAlmostEqual(row["rise_over_eps"], -2.5, places=6)

    def test_a_band_the_agent_never_entered_is_absent_not_zero(self):
        rows = {r["band"]: r for r in band_rows([5.0, 4.0], [0.05, 0.06], eps=1e-3)}
        self.assertEqual(rows["0-1"]["n_steps"], 0)
        self.assertIsNone(rows["0-1"]["rise_over_eps"])

    def test_the_bands_reach_the_report(self):
        pairs = [(d, 0.10 - 0.01 * d) for d in (5.0, 4.5, 4.0, 3.5, 3.0, 2.5)]
        text = format_report(aggregate([trace_one(rms_audit(pairs))]))
        self.assertIn("rise/eps", text)
        self.assertIn("2-3", text)


class TestPlateauWindows(unittest.TestCase):
    def test_maximal_runs_of_not_rising_are_the_windows(self):
        self.assertEqual(plateau_windows([True, False, False, True, False]),
                         [(1, 3), (4, 5)])

    def test_a_run_that_reaches_the_end_is_closed(self):
        self.assertEqual(plateau_windows([True, False, False]), [(1, 3)])

    def test_an_always_rising_climb_has_no_plateau(self):
        self.assertEqual(plateau_windows([True, True, True]), [])

    def test_a_flat_trace_is_one_window_not_many(self):
        self.assertEqual(plateau_windows([False, False, False]), [(0, 3)])

    def test_rising_is_computed_over_the_episode_not_the_detour(self):
        """The specific bug: a window-local recompute calls the onset step rising.

        The detour starts at ONSET_STEP + 1 here and the reading has not changed since the
        step before it, so the agent was already plateaued when it diverted. Slicing first
        and recomputing second would invent a FORWARD the agent never took.
        """
        row = trace_one(rms_audit([(3.0, 0.05), (3.0, 0.05), (3.0, 0.05)]))
        self.assertEqual(row["n_plateaus"], 1)
        self.assertEqual(row["plateau_steps"], 2)
        self.assertEqual(row["plateaus"][0]["start_step"], ONSET_STEP + 1)


class TestFitSlope(unittest.TestCase):
    def test_a_clean_line_recovers_its_slope_and_scatters_nothing(self):
        slope, resid = fit_slope([1.0, 2.0, 3.0, 4.0], [0.10, 0.08, 0.06, 0.04])
        self.assertAlmostEqual(slope, -0.02)
        self.assertAlmostEqual(resid, 0.0)

    def test_two_points_fit_a_line_but_report_no_scatter(self):
        """A line through two points leaves nothing to scatter; None, never 0.0, which
        would read as a noiseless measurement rather than an unmeasurable one."""
        slope, resid = fit_slope([1.0, 2.0], [0.10, 0.08])
        self.assertAlmostEqual(slope, -0.02)
        self.assertIsNone(resid)

    def test_an_agent_that_never_moved_has_no_slope(self):
        self.assertEqual(fit_slope([2.0, 2.0, 2.0], [0.05, 0.09, 0.04]), (None, None))

    def test_one_point_is_not_a_fit(self):
        self.assertEqual(fit_slope([2.0], [0.05]), (None, None))


class TestTheTwoDiagnoses(unittest.TestCase):
    """The measurement this half exists for: is the cue exhausted, or missed?"""

    def test_a_real_plateau_reads_flat_while_the_agent_keeps_moving(self):
        """The agent closes 1.25 m of gap and the render does not budge. Cue exhausted.

        The window is the six steps AFTER the onset: the first detour step has nothing
        before it inside the episode and so is rising by the rule's own definition.
        """
        row = trace_one(rms_audit([(3.0, 0.05)] + [(3.0 - 0.25 * i, 0.05) for i in range(1, 7)]))
        window = row["plateaus"][0]
        self.assertEqual(window["n_steps"], 6)
        self.assertFalse(window["static"])
        self.assertAlmostEqual(window["slope_per_m"], 0.0)
        self.assertAlmostEqual(window["d_span_m"], 1.25)

    def test_a_spurious_plateau_keeps_a_negative_slope_under_the_jitter(self):
        """A live gradient the single-step test cannot see.

        The agent is being carried AWAY from the source and the render tracks it: quieter
        at every step, so `rising` is false at every step, so the rule answers a turn at
        every step and the whole stretch is one window. But regressed against distance the
        cue is unmistakable — louder near, quieter far, a clean negative slope well clear
        of its own scatter. This is the (ii) signature: the field was informative and the
        one-step comparison could not use it.
        """
        pairs = [(1.9, 0.0720),
                 (2.0, 0.0700), (2.2, 0.0679), (2.4, 0.0662),
                 (2.6, 0.0639), (2.8, 0.0621), (3.0, 0.0600)]
        row = trace_one(rms_audit(pairs))
        window = row["plateaus"][0]
        self.assertEqual(window["n_steps"], 6)
        self.assertLess(window["slope_per_m"], 0.0)
        self.assertAlmostEqual(window["d_span_m"], 1.0)
        # The rays-1 gate: the cue clears its own noise by a wide margin here, so a
        # windowed estimator would have found it and the ray count is not the lever.
        self.assertGreater(window["signal_to_scatter"], 1.0)

    def test_an_agent_turning_in_place_is_static_and_its_slope_is_withheld(self):
        """The ill-conditioned case, reported rather than regressed: no translation means
        no test of the field, and a slope fitted here would be noise wearing a number."""
        row = trace_one(rms_audit([(2.3, 0.05), (2.3, 0.06), (2.3, 0.04), (2.3, 0.07)]))
        window = row["plateaus"][0]
        self.assertTrue(window["static"])
        self.assertIsNone(window["slope_per_m"])
        self.assertIsNone(window["signal_to_scatter"])
        self.assertEqual(row["n_static_plateaus"], 1)

    def test_static_windows_are_counted_in_the_aggregate_not_dropped(self):
        agg = aggregate([trace_one(rms_audit([(2.3, 0.05)] * 4, index=0))])
        arm = agg["plateaus"][ABANDONED]
        self.assertEqual(arm["n_windows"], 1)
        self.assertEqual(arm["n_static"], 1)
        self.assertEqual(arm["n_fitted"], 0)


class TestTheReconstructionCheck(unittest.TestCase):
    """ADR-0014's both arms: the check passing, and the check firing."""

    def test_a_record_that_agrees_reads_as_fully_checked(self):
        """Rising on the first detour step, plateaued after: FORWARD then two turns."""
        row = trace_one(rms_audit(
            [(3.0, 0.05), (2.8, 0.05), (2.6, 0.05)],
            realizable=[ACT_FORWARD, ACT_TURN_LEFT, ACT_TURN_LEFT], lateral=-1))
        self.assertEqual(row["n_rule_checked"], 3)
        self.assertEqual(row["n_rule_agree"], 3)
        self.assertEqual(aggregate([row])["rule_check"]["agreement"], 1.0)

    def test_a_record_that_disagrees_is_caught_and_named(self):
        """Forced failure: the trace is plateaued so the rule must answer a turn, and the
        record says the cue answered FORWARD. Silence here would let every plateau above
        rest on a model of the controller no one had tested."""
        row = trace_one(rms_audit([(3.0, 0.05), (2.8, 0.05), (2.6, 0.05)],
                                  realizable=ACT_FORWARD, lateral=-1))
        self.assertEqual(row["n_rule_checked"], 3)
        self.assertEqual(row["n_rule_agree"], 1)  # the rising first step, and nothing else
        text = format_report(aggregate([row]))
        self.assertIn("DISAGREEMENT IS THE FINDING", text)

    def test_a_stop_is_excluded_by_name_rather_than_counted_as_a_disagreement(self):
        """The rule's STOP needs `visual_confirm`, which no record carries."""
        row = trace_one(rms_audit([(3.0, 0.05), (0.4, 0.05)],
                                  realizable=ACT_STOP, lateral=-1))
        self.assertEqual(row["n_rule_checked"], 0)
        self.assertEqual(row["n_rule_stop"], 2)

    def test_a_pre_field_record_reads_unvalidated_and_says_so(self):
        """yield-2's shape. 0/0 must never render as agreement."""
        agg = aggregate([trace_one(rms_audit([(3.0, 0.05), (2.8, 0.05)]))])
        self.assertEqual(agg["rule_check"]["n_checked"], 0)
        self.assertIsNone(agg["rule_check"]["agreement"])
        self.assertIn("RECONSTRUCTION UNVALIDATED", format_report(agg))

    def test_the_plateau_table_names_both_arms(self):
        text = format_report(aggregate([
            trace_one(rms_audit([(3.0, 0.05)] * 4, index=0)),
            trace_one(rms_audit([(3.0, 0.05), (0.4, 0.05)], index=1,
                                stage=FunnelStage.PRIMARY_RESUMED)),
        ]))
        self.assertIn("plateau windows", text)
        self.assertIn("sig/sc", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
