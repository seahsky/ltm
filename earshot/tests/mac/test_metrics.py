"""Benchmark-standard ObjectNav SPL — the pure math behind the cross-quotable number.

Carried from ``embodied_memory/scripts/test_metrics.py``, which ticket 10 phase 3
deletes along with the rest of that tree. The eight cases are unchanged; only the
harness is, from hand-rolled ``assert`` functions to ``unittest``, so the one Mac suite
discovers it. Without this the reset would have silently dropped the only coverage on
correctness-critical scoring math.

ADR-0005: R1 (Table 1) must report a number cross-quotable to VLFM's SPL 0.304 and
VLingNav's 0.429. The harness's native ``spl`` is scored at the 0.1 m ring
(localization-bound) and ``success_1m`` is a STOP-independent reach diagnostic —
neither is the benchmark. Benchmark SPL is:

    success = the agent CALLED STOP within ``success_radius`` geodesic of a goal
              viewpoint (STOP-gated, unlike ``success_1m``).
    spl     = success * L_opt / max(L_taken, L_opt).
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.metrics import (
    compute_benchmark_spl,
    compute_soft_spl,
    compute_sws,
    sws_episode,
)


class TestBenchmarkSpl(unittest.TestCase):
    def test_perfect_path_is_spl_one(self):
        ok, spl = compute_benchmark_spl(
            stopped=True, dist_at_stop=0.4, geodesic_optimal=10.0, path_len_taken=10.0
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(spl, 1.0)

    def test_double_path_halves_spl(self):
        ok, spl = compute_benchmark_spl(
            stopped=True, dist_at_stop=0.9, geodesic_optimal=10.0, path_len_taken=20.0
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(spl, 0.5)

    def test_stop_beyond_the_radius_fails(self):
        """Within 1.0 m is the ring; a STOP at 1.5 m is not a benchmark success."""
        ok, spl = compute_benchmark_spl(
            stopped=True, dist_at_stop=1.5, geodesic_optimal=10.0, path_len_taken=10.0
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(spl, 0.0)

    def test_timeout_without_stop_fails_even_if_close(self):
        """The whole point of STOP-gating: near the goal but never called STOP."""
        ok, spl = compute_benchmark_spl(
            stopped=False, dist_at_stop=0.2, geodesic_optimal=10.0, path_len_taken=12.0
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(spl, 0.0)

    def test_unknown_distance_fails(self):
        ok, spl = compute_benchmark_spl(
            stopped=True, dist_at_stop=None, geodesic_optimal=10.0, path_len_taken=10.0
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(spl, 0.0)

    def test_start_on_the_goal_is_spl_one(self):
        """L_opt == 0 — started on the goal viewpoint and stopped there."""
        ok, spl = compute_benchmark_spl(
            stopped=True, dist_at_stop=0.05, geodesic_optimal=0.0, path_len_taken=0.0
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(spl, 1.0)

    def test_the_ratio_is_capped_at_one(self):
        """Defensive: a path shorter than the precomputed optimal cannot exceed 1.0."""
        ok, spl = compute_benchmark_spl(
            stopped=True, dist_at_stop=0.3, geodesic_optimal=10.0, path_len_taken=8.0
        )
        self.assertTrue(ok)
        self.assertAlmostEqual(spl, 1.0)

    def test_a_custom_radius_tightens_the_ring(self):
        """Same STOP distance, a 0.1 m ring, now a failure.

        The two rings are quoted together deliberately: binary SPL at 0.1 m is
        localization-bound, and the benchmark ring is 1.0 m.
        """
        ok, spl = compute_benchmark_spl(
            stopped=True,
            dist_at_stop=0.5,
            geodesic_optimal=10.0,
            path_len_taken=10.0,
            success_radius=0.1,
        )
        self.assertFalse(ok)
        self.assertAlmostEqual(spl, 0.0)


class TestSoftSpl(unittest.TestCase):
    """habitat-lab's ``SoftSPL``, re-derived because habitat-lab is not a dependency.

    §6 requires soft-SPL be computed and not headlined. The old tree read it off
    habitat-lab's own measure, so these cases pin the re-derivation against the formula
    in ``SoftSPL.update_metric`` rather than against a remembered shape.
    """

    def test_reaching_the_goal_on_the_optimal_path_is_one(self):
        self.assertAlmostEqual(
            compute_soft_spl(
                dist_to_goal_final=0.0, start_end_distance=10.0, path_len_taken=10.0
            ),
            1.0,
        )

    def test_halfway_there_on_the_optimal_path_is_half(self):
        self.assertAlmostEqual(
            compute_soft_spl(
                dist_to_goal_final=5.0, start_end_distance=10.0, path_len_taken=10.0
            ),
            0.5,
        )

    def test_the_efficiency_term_penalises_a_long_path(self):
        """Same progress, twice the walking: half the score."""
        self.assertAlmostEqual(
            compute_soft_spl(
                dist_to_goal_final=5.0, start_end_distance=10.0, path_len_taken=20.0
            ),
            0.25,
        )

    def test_it_does_not_require_a_stop(self):
        """The whole difference from ``compute_benchmark_spl``, stated as a test.

        The same episode scores 0.0 on the benchmark (no STOP was called) and a real
        number here, which is why §6 keeps both and headlines neither from this map.
        """
        _ok, benchmark = compute_benchmark_spl(
            stopped=False, dist_at_stop=0.2, geodesic_optimal=10.0, path_len_taken=10.0
        )
        self.assertAlmostEqual(benchmark, 0.0)
        self.assertGreater(
            compute_soft_spl(
                dist_to_goal_final=0.2, start_end_distance=10.0, path_len_taken=10.0
            ),
            0.9,
        )

    def test_moving_away_from_the_goal_floors_at_zero(self):
        """habitat-lab's ``max(0, ...)``: a negative progress term is not a negative score."""
        self.assertAlmostEqual(
            compute_soft_spl(
                dist_to_goal_final=25.0, start_end_distance=10.0, path_len_taken=30.0
            ),
            0.0,
        )

    def test_an_unreachable_goal_scores_zero_rather_than_raising(self):
        """``None`` is ticket 21's boundary conversion, and it must not reach arithmetic."""
        self.assertAlmostEqual(
            compute_soft_spl(
                dist_to_goal_final=None, start_end_distance=10.0, path_len_taken=5.0
            ),
            0.0,
        )

    def test_starting_on_the_goal_is_the_one_case_habitat_lab_divides_by_zero_on(self):
        """The disclosed divergence. Still there scores 1.0; having left scores 0.0."""
        self.assertAlmostEqual(
            compute_soft_spl(
                dist_to_goal_final=0.0, start_end_distance=0.0, path_len_taken=0.0
            ),
            1.0,
        )
        self.assertAlmostEqual(
            compute_soft_spl(
                dist_to_goal_final=3.0, start_end_distance=0.0, path_len_taken=3.0
            ),
            0.0,
        )


class TestSuccessWhenSilent(unittest.TestCase):
    """SWS (Chen et al., CVPR 2021 §5), adopted verbatim to stay cross-quotable.

    Two decisions are being pinned here rather than the arithmetic, because both are
    real and both are silently reversible. SWS counts reaching the SOUND SOURCE, not the
    primary ObjectNav goal — there are two successes in this record and ADR-0017 makes
    the source the find-task. And the denominator is episodes that ran past their OWN
    offset step, not every episode: one that ended first never had a silent phase.
    """

    def test_an_episode_that_ended_before_its_window_closed_is_not_eligible(self):
        """It never had a silent phase, so it cannot answer the question SWS asks.

        Counting it in the denominator would score every short episode as a failure to
        succeed in silence, which is a claim about the step budget rather than about the
        agent.
        """
        eligible, reached = sws_episode(
            offset_step=60, n_loop_steps=40, source_reached_step=30
        )
        self.assertFalse(eligible)
        self.assertFalse(reached)

    def test_an_episode_that_ended_ON_its_offset_step_is_not_eligible_either(self):
        """THE DENOMINATOR'S BOUNDARY, and the fixtures above sit nowhere near it.

        60 against 40 and 60 against 200 both answer under any comparison; the real edge
        is ``offset_step == n_loop_steps``. An episode with steps 0..59 has
        ``n_loop_steps`` 60 and its LAST step is 59 -- the step before the source was due
        to stop -- so it never ran a silent step at all. Widening ``<`` to ``<=`` here
        puts it in SWS's denominator as a failure to succeed in silence, which is a claim
        about the step budget rather than about the agent, and the numerator's own
        boundary check cannot see it.
        """
        eligible, reached = sws_episode(
            offset_step=60, n_loop_steps=60, source_reached_step=59
        )
        self.assertFalse(eligible)
        self.assertFalse(reached)
        # ...and one more step IS the whole difference: step 60 is the first silent one.
        eligible, reached = sws_episode(
            offset_step=60, n_loop_steps=61, source_reached_step=60
        )
        self.assertTrue(eligible)
        self.assertTrue(reached)

    def test_reaching_the_source_before_the_offset_step_is_eligible_and_not_a_success(self):
        """The definition's sharp edge, and the whole content of the metric.

        The episode ran past its offset step, so it IS in the denominator — the agent
        had a silent phase available to it. It reached the source while the source was
        still sounding, so it is not in the numerator: it did not complete when silent.
        """
        eligible, reached = sws_episode(
            offset_step=60, n_loop_steps=200, source_reached_step=42
        )
        self.assertTrue(eligible)
        self.assertFalse(reached)

    def test_reaching_it_after_the_offset_step_is_the_metric(self):
        eligible, reached = sws_episode(
            offset_step=60, n_loop_steps=200, source_reached_step=131
        )
        self.assertTrue(eligible)
        self.assertTrue(reached)

    def test_the_offset_step_itself_counts_as_silent(self):
        """``[opens_at, offset_step)`` is the sounding phase, so the offset step is the
        FIRST silent step and a reach on it is a reach in silence."""
        eligible, reached = sws_episode(
            offset_step=60, n_loop_steps=200, source_reached_step=60
        )
        self.assertTrue(eligible)
        self.assertTrue(reached)

    def test_never_reaching_the_source_is_eligible_and_not_a_success(self):
        eligible, reached = sws_episode(
            offset_step=60, n_loop_steps=200, source_reached_step=None
        )
        self.assertTrue(eligible)
        self.assertFalse(reached)

    def test_a_continuous_arm_episode_is_never_eligible(self):
        """No offset step means no silent phase, whatever else the episode did.

        ``WindowPolicy.CONTINUOUS`` is the control arm every windowed delta is measured
        against, and an SWS quietly computed over it would be a number about an arm in
        which the metric is undefined.
        """
        eligible, reached = sws_episode(
            offset_step=None, n_loop_steps=500, source_reached_step=310
        )
        self.assertFalse(eligible)
        self.assertFalse(reached)

    def test_sws_over_no_eligible_episodes_is_not_run_rather_than_zero(self):
        """THE FORCED-FAILURE ARM (ADR-0014). ``None``, and explicitly not ``0.0``.

        Two incidents are behind the rule: a probe that skipped and reported success, and
        a canary that was never armed reading as a pass. 0.0 says *the agent never
        succeeded in silence*; NOT_RUN says *nobody asked*, and a reader cannot tell
        those apart from a float. The healthy arm below is the same call with a
        denominator.
        """
        result = compute_sws(n_eligible=0, n_reached_after_offset=0)
        self.assertIsNone(result)
        self.assertIsNot(result, 0.0)
        # the healthy arm, so the None above is not just a function that never works
        self.assertAlmostEqual(compute_sws(n_eligible=8, n_reached_after_offset=0), 0.0)
        self.assertAlmostEqual(compute_sws(n_eligible=8, n_reached_after_offset=3), 0.375)

    def test_an_sws_over_episodes_whose_tail_never_ran_is_refused(self):
        """ADR-0017's bar carried INTO the primitive, and it is the whole of line 49.

        An episode whose accumulation buffer folded no render had a silent phase that
        arrived as a HARD CUT to the bed, so an SWS counting it is a number about the
        mechanism ADR-0017 replaced rather than about the agent. ``n_tail_active`` is
        Optional only because this function takes two ints and cannot fetch the records
        itself -- a caller that HAS the evidence must pass it, and a caller that does not
        is publishing an unverified rate.
        """
        with self.assertRaises(ValueError) as caught:
            compute_sws(n_eligible=8, n_reached_after_offset=3, n_tail_active=5)
        self.assertIn("3 of the 8", str(caught.exception))
        self.assertIn("hard cut", str(caught.exception))
        # the healthy arm: every eligible episode carried one, so the rate stands
        self.assertAlmostEqual(
            compute_sws(n_eligible=8, n_reached_after_offset=3, n_tail_active=8), 0.375
        )
        # ...and omitting it is permitted-but-unverified rather than refused, because a
        # caller with counts and no records must not be forced to fabricate the number.
        self.assertAlmostEqual(
            compute_sws(n_eligible=8, n_reached_after_offset=3), 0.375
        )

    def test_impossible_counts_raise(self):
        """The quiet fix — ``min(numerator, denominator)`` — would publish 1.0 for a bug."""
        with self.assertRaises(ValueError):
            compute_sws(n_eligible=3, n_reached_after_offset=4)
        with self.assertRaises(ValueError):
            compute_sws(n_eligible=-1, n_reached_after_offset=0)
        with self.assertRaises(ValueError):
            compute_sws(n_eligible=3, n_reached_after_offset=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
