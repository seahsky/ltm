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

from earshot.metrics import compute_benchmark_spl, compute_soft_spl


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
