"""The frontier proposer: extraction, clustering, the two score kernels, and the cadence.

All four surviving pieces of ADR-0008's rewrite, plus the two properties that are
invariants rather than arithmetic: the pool is never empty, and frontier candidates and
compass candidates are never mixed.

The grids here are hand-built rather than splatted, so a failure names the piece under
test. ``tests/mac/test_agent_occupancy.py`` is where a real depth frame becomes a map.
"""

import math
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.config import PlannerConfig
from earshot.agent.occupancy import (
    CELL_FREE,
    CELL_OCCUPIED,
    CELL_UNKNOWN,
    OccupancyGrid,
    forward_xz,
    new_grid,
)
from earshot.agent.proposers import (
    SOURCE_COMPASS,
    SOURCE_FRONTIER,
    Candidate,
    FrontierProposer,
    cluster_cells,
    compass_score,
    frontier_cells,
    frontier_score,
    ray_occupancy_fractions,
)
from earshot.types import Pose, Xyz

CFG = PlannerConfig(grid_size_m=4.0, grid_res_m=0.1, n_candidates=3, splat_samples=8)


def pose_at(yaw=0.0, x=0.0, z=0.0):
    return Pose(position=Xyz(x, 0.0, z), yaw_rad=yaw)


def grid_with(states, cfg=CFG):
    """A grid centred on the origin with ``states`` applied as ``{(row, col): state}``."""
    grid = new_grid(Xyz(0.0, 0.0, 0.0), cfg)
    cells = grid.cells.copy()
    for (row, col), state in states.items():
        cells[row, col] = state
    return OccupancyGrid(
        resolution_m=grid.resolution_m, origin_xy=grid.origin_xy, cells=cells
    )


class TestFrontierExtraction(unittest.TestCase):
    def test_a_free_cell_beside_unknown_is_a_frontier(self):
        grid = grid_with({(10, 10): CELL_FREE})
        self.assertIn((10, 10), frontier_cells(grid))

    def test_a_free_cell_surrounded_by_free_is_not(self):
        states = {(10, 10): CELL_FREE}
        for row, col in ((9, 10), (11, 10), (10, 9), (10, 11)):
            states[(row, col)] = CELL_FREE
        self.assertNotIn((10, 10), frontier_cells(grid_with(states)))

    def test_unknown_and_occupied_cells_are_never_frontiers(self):
        grid = grid_with({(5, 5): CELL_OCCUPIED, (20, 20): CELL_UNKNOWN})
        cells = frontier_cells(grid)
        self.assertNotIn((5, 5), cells)
        self.assertNotIn((20, 20), cells)

    def test_an_all_unknown_map_has_no_frontier(self):
        self.assertEqual(frontier_cells(new_grid(Xyz(0.0, 0.0, 0.0), CFG)), [])


class TestClustering(unittest.TestCase):
    def test_nearby_cells_join_and_distant_ones_do_not(self):
        cells = [(0, 0), (0, 1), (1, 1), (30, 30)]
        clusters = cluster_cells(cells, max_clusters=4, radius_cells=3)
        self.assertEqual(sorted(len(c) for c in clusters), [1, 3])

    def test_clusters_come_back_largest_first(self):
        cells = [(0, 0), (0, 1), (0, 2), (20, 20), (40, 40), (40, 41)]
        sizes = [len(c) for c in cluster_cells(cells, max_clusters=3, radius_cells=3)]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_the_search_may_find_more_clusters_than_it_returns(self):
        """Truncating the search would return whichever were seeded first, not the biggest.

        Five separated singletons plus one triple, asking for one cluster: the triple has
        to win, which is only possible if the search did not stop at one.
        """
        cells = [(0, 0), (0, 1), (0, 2)] + [(10 * i, 30) for i in range(1, 6)]
        clusters = cluster_cells(cells, max_clusters=1, radius_cells=3)
        self.assertEqual([len(c) for c in clusters], [3])

    def test_no_cells_no_clusters(self):
        self.assertEqual(cluster_cells([], max_clusters=3, radius_cells=3), [])

    def test_every_cell_lands_in_exactly_one_cluster(self):
        cells = [(r, c) for r in range(0, 12, 2) for c in range(0, 12, 2)]
        clusters = cluster_cells(cells, max_clusters=8, radius_cells=3)
        assigned = [cell for cluster in clusters for cell in cluster]
        self.assertEqual(len(assigned), len(set(assigned)))


class TestScoreKernels(unittest.TestCase):
    def test_a_bigger_cluster_scores_higher_at_the_same_range(self):
        self.assertGreater(frontier_score(30, 2.5), frontier_score(3, 2.5))

    def test_the_distance_kernel_peaks_at_two_and_a_half_metres(self):
        peak = frontier_score(10, 2.5)
        self.assertGreater(peak, frontier_score(10, 0.5))
        self.assertGreater(peak, frontier_score(10, 8.0))

    def test_the_compass_score_spans_its_documented_range(self):
        self.assertAlmostEqual(compass_score(1.0, 0.0), 1.0, places=9)
        self.assertAlmostEqual(compass_score(0.0, 0.0), 0.7, places=9)
        self.assertAlmostEqual(compass_score(0.0, 1.0), 0.2, places=9)

    def test_the_compass_score_is_clipped(self):
        self.assertEqual(compass_score(5.0, 0.0), 1.0)
        self.assertEqual(compass_score(0.0, 5.0), 0.0)


class TestRayOccupancy(unittest.TestCase):
    def test_the_scan_reads_the_direction_the_splat_wrote(self):
        """Same ``forward_xz``, or the scan would score a direction nothing observed."""
        cfg = PlannerConfig(grid_size_m=4.0, grid_res_m=0.1)
        grid = new_grid(Xyz(0.0, 0.0, 0.0), cfg)
        cells = grid.cells.copy()
        for i in range(1, 15):
            fx, fz = forward_xz(0.0)
            row, col = grid.world_to_grid(fx * i * 0.1, fz * i * 0.1)
            cells[row, col] = CELL_FREE
        grid = OccupancyGrid(grid.resolution_m, grid.origin_xy, cells)
        ahead, _ = ray_occupancy_fractions(grid, Xyz(0.0, 0.0, 0.0), 0.0, 1.0)
        behind, _ = ray_occupancy_fractions(grid, Xyz(0.0, 0.0, 0.0), math.pi, 1.0)
        self.assertGreater(ahead, 0.9)
        self.assertEqual(behind, 0.0)

    def test_an_out_of_bounds_ray_reports_nothing_rather_than_zero_free(self):
        grid = new_grid(Xyz(0.0, 0.0, 0.0), PlannerConfig(grid_size_m=0.4, grid_res_m=0.1))
        self.assertEqual(ray_occupancy_fractions(grid, Xyz(50.0, 0.0, 50.0), 0.0, 2.0), (0.0, 0.0))


class TestTheProposer(unittest.TestCase):
    def _proposer(self, cfg=CFG):
        proposer = FrontierProposer(cfg=cfg)
        proposer.reset(pose_at())
        return proposer

    def test_proposing_before_reset_raises(self):
        """The grid must be centred on the agent, and a default centre is not a default."""
        with self.assertRaises(RuntimeError):
            FrontierProposer(cfg=CFG).propose(pose_at())

    def test_the_pool_is_never_empty_on_a_blank_map(self):
        pool = self._proposer().propose(pose_at())
        self.assertEqual(len(pool), CFG.n_candidates)
        self.assertTrue(all(c.source == SOURCE_COMPASS for c in pool))

    def test_the_compass_fan_puts_picks_behind_the_agent(self):
        """Two of three, which is what escapes a start-wall stall."""
        pool = self._proposer().propose(pose_at())
        behind = [c for c in pool if abs(c.bearing_rad) > math.pi / 2]
        self.assertEqual(len(behind), 2)

    def test_real_frontiers_displace_the_fan_entirely(self):
        """Never mixed: a fan pick at a convenient bearing must not outrank a real cluster."""
        proposer = self._proposer()
        cells = proposer.grid.cells.copy()
        cells[8:12, 8:12] = CELL_FREE
        proposer.grid = OccupancyGrid(
            proposer.grid.resolution_m, proposer.grid.origin_xy, cells
        )
        pool = proposer.propose(pose_at())
        self.assertTrue(pool)
        self.assertTrue(all(c.source == SOURCE_FRONTIER for c in pool))

    def test_a_candidate_carries_the_agents_own_height(self):
        """So the navmesh snap cannot land a storey down (ADR-0010's same-floor policy)."""
        proposer = self._proposer()
        pool = proposer.propose(Pose(position=Xyz(0.0, 1.75, 0.0), yaw_rad=0.0))
        self.assertTrue(all(c.position.y == 1.75 for c in pool))

    def test_candidate_ids_are_unique_and_increasing(self):
        proposer = self._proposer()
        ids = [c.candidate_id for c in proposer.propose(pose_at()) + proposer.propose(pose_at())]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(ids), len(set(ids)))

    def test_the_pool_is_sorted_by_intrinsic_score(self):
        proposer = self._proposer()
        cells = proposer.grid.cells.copy()
        cells[8:12, 8:12] = CELL_FREE   # a big cluster
        cells[30, 30] = CELL_FREE       # a lone cell
        proposer.grid = OccupancyGrid(
            proposer.grid.resolution_m, proposer.grid.origin_xy, cells
        )
        scores = [c.raw_score for c in proposer.propose(pose_at())]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_the_pool_is_capped_at_n_candidates(self):
        proposer = self._proposer()
        cells = proposer.grid.cells.copy()
        for i in range(6):
            cells[5 * i + 2, 5 * i + 2] = CELL_FREE
        proposer.grid = OccupancyGrid(
            proposer.grid.resolution_m, proposer.grid.origin_xy, cells
        )
        self.assertLessEqual(len(proposer.propose(pose_at())), CFG.n_candidates)


class TestCadence(unittest.TestCase):
    def _proposer(self):
        proposer = FrontierProposer(cfg=PlannerConfig(decision_period=10, stuck_window=4))
        proposer.reset(pose_at())
        return proposer

    def test_the_first_step_is_always_a_decision(self):
        self.assertTrue(self._proposer().is_decision_step())

    def test_the_scheduled_cadence_fires_on_the_period(self):
        proposer = self._proposer()
        fired = []
        for step in range(1, 21):
            proposer.observe(None, pose_at(x=float(step)))
            fired.append(proposer.is_decision_step())
        self.assertEqual([i + 1 for i, f in enumerate(fired) if f], [10, 20])

    def test_a_replan_request_is_consumed_once(self):
        proposer = self._proposer()
        proposer.observe(None, pose_at(x=1.0))
        proposer.request_replan()
        self.assertTrue(proposer.is_decision_step())
        proposer.observe(None, pose_at(x=2.0))
        self.assertFalse(proposer.is_decision_step())
        self.assertEqual(proposer.stats()["replan_requested"], 1)

    def test_a_stationary_agent_is_stuck_and_re_proposes(self):
        proposer = self._proposer()
        for _ in range(4):
            proposer.observe(None, pose_at())
        self.assertTrue(proposer.is_stuck())
        self.assertTrue(proposer.is_decision_step())
        self.assertEqual(proposer.stats()["replan_stuck"], 1)

    def test_a_moving_agent_is_not_stuck(self):
        proposer = self._proposer()
        for step in range(1, 5):
            proposer.observe(None, pose_at(x=float(step)))
        self.assertFalse(proposer.is_stuck())

    def test_stuck_needs_a_full_window_of_evidence(self):
        proposer = self._proposer()
        proposer.observe(None, pose_at())
        self.assertFalse(proposer.is_stuck())

    def test_the_position_history_is_bounded(self):
        proposer = self._proposer()
        for step in range(200):
            proposer.observe(None, pose_at(x=float(step)))
        self.assertLessEqual(len(proposer._positions), 32)

    def test_reset_clears_the_counters_and_the_map(self):
        proposer = self._proposer()
        for _ in range(4):
            proposer.observe(None, pose_at())
        proposer.is_decision_step()
        proposer.reset(pose_at(x=9.0))
        self.assertEqual(proposer.n_steps, 0)
        self.assertEqual(proposer.stats()["replan_stuck"], 0)
        self.assertEqual(
            int(np.count_nonzero(proposer.grid.cells != CELL_UNKNOWN)), 0
        )


class TestStats(unittest.TestCase):
    def test_the_census_counts_the_whole_grid(self):
        proposer = FrontierProposer(cfg=CFG)
        proposer.reset(pose_at())
        stats = proposer.stats()
        total = stats["cells_free"] + stats["cells_occupied"] + stats["cells_unknown"]
        self.assertEqual(total, proposer.grid.n ** 2)
        self.assertEqual(stats["frontier_cells"], 0)


class TestCandidateType(unittest.TestCase):
    def test_a_candidate_is_frozen(self):
        candidate = Candidate(
            candidate_id=1,
            position=Xyz(0.0, 0.0, 0.0),
            source=SOURCE_FRONTIER,
            distance_m=1.0,
            bearing_rad=0.0,
            raw_score=0.5,
        )
        with self.assertRaises(Exception):
            candidate.raw_score = 0.9  # type: ignore[misc]

    def test_the_geodesic_is_unset_until_the_navmesh_answers(self):
        candidate = Candidate(
            candidate_id=1,
            position=Xyz(0.0, 0.0, 0.0),
            source=SOURCE_FRONTIER,
            distance_m=1.0,
            bearing_rad=0.0,
            raw_score=0.5,
        )
        self.assertIsNone(candidate.geodesic_m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
