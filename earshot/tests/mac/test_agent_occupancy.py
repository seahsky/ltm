"""The occupancy grid and the depth splat, against hand-built depth frames.

A fake here licenses a lot, because there is nothing simulator-shaped in the subject: the
splat is arithmetic over an ``(H, W)`` array and a pose. What it cannot license is that
habitat's depth arrives in the units this assumes — ticket 21 read that off the binding
(no ``normalize_depth`` on ``CameraSensorSpec``, so depth is raw and metric) and
``tests/box/test_world_box.py`` renders one.

The cases that earn their place are the ones that were bugs: the grid must be centred on
the agent (a world-origin grid read out of bounds for every HM3D start and silently
discarded the episode), obstacles must be sticky (a floor ray overshooting a wall must not
erase it), and the height gate must measure from the floor the agent stands on rather than
from an offset that happened to cancel another offset.
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
    cell_counts,
    forward_xz,
    integrate_depth,
    new_grid,
)
from earshot.types import Pose, Xyz

# A small grid keeps the assertions readable: 4 m square at 0.1 m is 40x40.
CFG = PlannerConfig(grid_size_m=4.0, grid_res_m=0.1, splat_samples=8, max_depth_m=5.0)


def pose_at(yaw=0.0, x=0.0, z=0.0, y=0.0):
    return Pose(position=Xyz(x, y, z), yaw_rad=yaw)


def flat_depth(value, height=8, width=8):
    return np.full((height, width), float(value), dtype=np.float32)


class TestGridGeometry(unittest.TestCase):
    def test_the_grid_is_centred_on_the_agent(self):
        grid = new_grid(Xyz(-12.0, 0.0, 17.5), CFG)
        row, col = grid.world_to_grid(-12.0, 17.5)
        self.assertEqual((row, col), (grid.n // 2, grid.n // 2))

    def test_a_start_far_from_the_world_origin_is_in_bounds(self):
        """The bug this fixes: HM3D starts are routinely 15-20 m from the origin."""
        grid = new_grid(Xyz(0.0, 0.0, -17.77), CFG)
        self.assertTrue(grid.in_bounds(*grid.world_to_grid(0.0, -17.77)))

    def test_world_and_grid_round_trip_within_half_a_cell(self):
        grid = new_grid(Xyz(1.0, 0.0, 2.0), CFG)
        for x, z in ((1.0, 2.0), (2.44, 0.13), (-0.5, 3.2)):
            row, col = grid.world_to_grid(x, z)
            bx, bz = grid.grid_to_world(row, col)
            self.assertLessEqual(abs(bx - x), CFG.grid_res_m)
            self.assertLessEqual(abs(bz - z), CFG.grid_res_m)

    def test_a_point_outside_the_low_edge_floors_to_minus_one(self):
        """``int()`` truncates toward zero, which the old grid did (``:79-83``).

        The probe has to straddle the **grid's origin**, not the world's: a grid centred at
        (0, 0) has its origin at -2.0, so both sides of the world origin give positive
        quotients and truncation and flooring agree. The first version of this test probed
        (+-0.05, +-0.05) and so could not tell them apart.

        Under truncation this returns (0, 0) — aliasing a point outside the map onto the
        edge cell, which is how a discarded ray endpoint would get written into the grid.
        """
        grid = new_grid(Xyz(0.0, 0.0, 0.0), CFG)
        self.assertEqual(grid.origin_xy, (-2.0, -2.0))
        self.assertEqual(grid.world_to_grid(-2.05, -2.05), (-1, -1))
        self.assertFalse(grid.in_bounds(-1, -1))

    def test_out_of_bounds_reads_as_unknown(self):
        grid = new_grid(Xyz(0.0, 0.0, 0.0), CFG)
        self.assertEqual(grid.state_at(-1, 0), CELL_UNKNOWN)
        self.assertEqual(grid.state_at(grid.n, 0), CELL_UNKNOWN)

    def test_a_fresh_grid_is_entirely_unknown(self):
        counts = cell_counts(new_grid(Xyz(0.0, 0.0, 0.0), CFG))
        self.assertEqual(counts["cells_free"], 0)
        self.assertEqual(counts["cells_occupied"], 0)
        self.assertEqual(counts["cells_unknown"], 40 * 40)


class TestTheSplat(unittest.TestCase):
    def test_the_input_grid_is_not_mutated(self):
        """``integrate_depth`` returns a new grid; a stale reference cannot be edited."""
        grid = new_grid(Xyz(0.0, 0.0, 0.0), CFG)
        before = grid.cells.copy()
        updated = integrate_depth(grid, flat_depth(2.0), pose_at(), CFG)
        np.testing.assert_array_equal(grid.cells, before)
        self.assertGreater(int(np.count_nonzero(updated.cells != CELL_UNKNOWN)), 0)

    def test_free_space_is_carved_in_front_of_the_agent(self):
        """In FRONT: at yaw 0 that is ``-z``, which the old tree had as ``+z``."""
        grid = integrate_depth(
            new_grid(Xyz(0.0, 0.0, 0.0), CFG), flat_depth(1.5), pose_at(), CFG
        )
        ahead = grid.state_at(*grid.world_to_grid(0.0, -0.8))
        behind = grid.state_at(*grid.world_to_grid(0.0, 0.8))
        self.assertEqual(ahead, CELL_FREE)
        self.assertEqual(behind, CELL_UNKNOWN)

    def test_the_carved_cone_follows_the_yaw(self):
        for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
            grid = integrate_depth(
                new_grid(Xyz(0.0, 0.0, 0.0), CFG), flat_depth(1.5), pose_at(yaw), CFG
            )
            fx, fz = forward_xz(yaw)
            self.assertEqual(
                grid.state_at(*grid.world_to_grid(fx * 0.8, fz * 0.8)),
                CELL_FREE,
                "yaw {:.3f}: nothing carved along the heading".format(yaw),
            )

    def test_the_height_gate_decides_obstacle_versus_floor(self):
        """The same frame, two thresholds. An 8-wide frame has an on-axis pixel at u=4.

        Its endpoint sits at eye height, 0.88 m above the floor: an obstacle against the
        0.3 m gate, floor against a 3 m one. Nothing else about the frame changes, so the
        gate is the only thing under test.
        """
        level = flat_depth(2.0)
        endpoint = (0.0, -2.0)

        tight = PlannerConfig(grid_size_m=4.0, grid_res_m=0.1, splat_samples=8, obstacle_min_h=0.3)
        grid = integrate_depth(new_grid(Xyz(0.0, 0.0, 0.0), tight), level, pose_at(), tight)
        self.assertEqual(grid.state_at(*grid.world_to_grid(*endpoint)), CELL_OCCUPIED)

        loose = PlannerConfig(grid_size_m=4.0, grid_res_m=0.1, splat_samples=8, obstacle_min_h=3.0)
        grid = integrate_depth(new_grid(Xyz(0.0, 0.0, 0.0), loose), level, pose_at(), loose)
        self.assertEqual(grid.state_at(*grid.world_to_grid(*endpoint)), CELL_FREE)

    def test_an_obstacle_is_not_erased_by_a_later_free_ray(self):
        """A wall at 2 m, then a 3 m ray marching free straight through the same cell."""
        grid = integrate_depth(
            new_grid(Xyz(0.0, 0.0, 0.0), CFG), flat_depth(2.0), pose_at(), CFG
        )
        occupied = np.argwhere(grid.cells == CELL_OCCUPIED)
        self.assertGreater(len(occupied), 0, "the 2 m frame marked no obstacle at all")
        grid = integrate_depth(grid, flat_depth(3.0), pose_at(), CFG)
        for row, col in occupied:
            self.assertEqual(
                grid.state_at(int(row), int(col)),
                CELL_OCCUPIED,
                "cell ({}, {}) was erased by a longer ray".format(row, col),
            )

    def test_invalid_depths_are_skipped_rather_than_projected(self):
        depth = np.array([[float("nan"), 0.0, float("inf"), 0.01]], dtype=np.float32)
        grid = integrate_depth(new_grid(Xyz(0.0, 0.0, 0.0), CFG), depth, pose_at(), CFG)
        self.assertEqual(cell_counts(grid)["cells_unknown"], 40 * 40)

    def test_depth_beyond_the_range_is_clamped_not_dropped(self):
        far = flat_depth(50.0)
        cfg = PlannerConfig(grid_size_m=20.0, grid_res_m=0.1, splat_samples=8, max_depth_m=5.0)
        grid = integrate_depth(new_grid(Xyz(0.0, 0.0, 0.0), cfg), far, pose_at(), cfg)
        self.assertEqual(grid.state_at(*grid.world_to_grid(0.0, -4.0)), CELL_FREE)
        # Nothing is written past the clamp.
        self.assertEqual(grid.state_at(*grid.world_to_grid(0.0, -8.0)), CELL_UNKNOWN)

    def test_an_hw1_frame_is_accepted(self):
        grid = integrate_depth(
            new_grid(Xyz(0.0, 0.0, 0.0), CFG), flat_depth(1.5).reshape(8, 8, 1), pose_at(), CFG
        )
        self.assertGreater(cell_counts(grid)["cells_free"], 0)

    def test_an_empty_frame_leaves_the_map_alone(self):
        grid = new_grid(Xyz(0.0, 0.0, 0.0), CFG)
        self.assertIs(integrate_depth(grid, np.zeros((0, 0)), pose_at(), CFG), grid)

    def test_a_frame_of_the_wrong_rank_raises(self):
        with self.assertRaises(ValueError):
            integrate_depth(
                new_grid(Xyz(0.0, 0.0, 0.0), CFG), np.zeros((2, 2, 3)), pose_at(), CFG
            )

    def test_a_cone_that_leaves_the_grid_writes_only_what_is_inside(self):
        """A 1 m window and a 4 m ray: most of the cone is out of bounds by design.

        Two properties, and the first version of this test asserted neither: nothing
        raises, the in-bounds part of the ray IS written, and nothing outside is (which is
        what ``world_to_grid``'s flooring underwrites — under truncation the discarded
        endpoint would land on the edge cell).
        """
        cfg = PlannerConfig(grid_size_m=1.0, grid_res_m=0.1, splat_samples=8, max_depth_m=5.0)
        grid = integrate_depth(
            new_grid(Xyz(0.0, 0.0, 0.0), cfg), flat_depth(4.0), pose_at(), cfg
        )
        self.assertEqual(grid.n, 10)
        counts = cell_counts(grid)
        self.assertGreater(counts["cells_free"], 0)
        # The 4 m endpoint is far outside a 1 m window, so no cell may be occupied.
        self.assertEqual(counts["cells_occupied"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
