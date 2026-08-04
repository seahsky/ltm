"""ADR-0008's invariant: the pool is never empty, and every member is snapped and reachable.

The navmesh arrives as two injected callables (ADR-0013), so the filter is fully
Mac-testable against two lambdas — which is the whole reason the invariant lives in its own
module rather than inside the proposer. What a Mac cannot tell you is what habitat-sim's
``snap_point`` actually returns for a point inside geometry; ticket 21 converted its NaN
failure into ``None`` at the boundary and ``tests/box/test_world_box.py`` exercises it.

The counters matter as much as the filtering. The raise has to say *which* stage ate the
candidates, because "off-navmesh", "another floor" and "no route" are three different
broken episodes and the old tree's answer to all three was a straight-line fallback into
the wall.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.config import PlannerConfig
from earshot.agent.proposers import SOURCE_FRONTIER, Candidate
from earshot.agent.reachability import (
    EmptyPoolError,
    assert_pool,
    reachable_pool,
)
from earshot.types import Pose, Xyz

POSE = Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)


def candidate(cid=1, x=0.0, y=0.0, z=-3.0):
    return Candidate(
        candidate_id=cid,
        position=Xyz(x, y, z),
        source=SOURCE_FRONTIER,
        distance_m=abs(z),
        bearing_rad=0.0,
        raw_score=0.5,
        cluster_size=7,
    )


def identity_snap(point):
    return point


def straight_line(start, end):
    return start.horizontal_distance_to(end)


class TestFiltering(unittest.TestCase):
    def test_a_reachable_candidate_survives(self):
        report = reachable_pool(
            [candidate()], POSE, snap_point=identity_snap, geodesic=straight_line
        )
        self.assertEqual(report.n_kept, 1)
        self.assertEqual(report.n_proposed, 1)

    def test_an_off_navmesh_candidate_is_dropped(self):
        report = reachable_pool(
            [candidate()], POSE, snap_point=lambda p: None, geodesic=straight_line
        )
        self.assertEqual(report.n_kept, 0)
        self.assertEqual(report.n_off_navmesh, 1)

    def test_an_unreachable_candidate_is_dropped(self):
        report = reachable_pool(
            [candidate()], POSE, snap_point=identity_snap, geodesic=lambda a, b: None
        )
        self.assertEqual(report.n_kept, 0)
        self.assertEqual(report.n_unreachable, 1)

    def test_a_snap_to_another_floor_is_dropped(self):
        """The one snap the filter refuses: ADR-0010's ``|dy| < 1.0 m``."""
        report = reachable_pool(
            [candidate()],
            POSE,
            snap_point=lambda p: Xyz(p.x, p.y - 2.8, p.z),
            geodesic=straight_line,
            cfg=PlannerConfig(same_floor_m=1.0),
        )
        self.assertEqual(report.n_kept, 0)
        self.assertEqual(report.n_wrong_floor, 1)

    def test_the_floor_tolerance_comes_from_the_config(self):
        """One home for the number, so the value in force is the one the run record has.

        An earlier revision had a literal in ``reachable_pool`` as well as in
        ``PlannerConfig``: two values that can drift, with the effective one invisible in
        ``env_report.json``.
        """
        snap = lambda point: Xyz(point.x, point.y - 0.5, point.z)  # noqa: E731
        strict = reachable_pool(
            [candidate()], POSE, snap_point=snap, geodesic=straight_line,
            cfg=PlannerConfig(same_floor_m=0.25),
        )
        lax = reachable_pool(
            [candidate()], POSE, snap_point=snap, geodesic=straight_line,
            cfg=PlannerConfig(same_floor_m=2.0),
        )
        self.assertEqual((strict.n_kept, strict.n_wrong_floor), (0, 1))
        self.assertEqual((lax.n_kept, lax.n_wrong_floor), (1, 0))

    def test_a_small_vertical_correction_is_kept(self):
        """A navmesh sitting a few centimetres off the candidate's y is not a storey."""
        report = reachable_pool(
            [candidate()],
            POSE,
            snap_point=lambda p: Xyz(p.x, p.y + 0.04, p.z),
            geodesic=straight_line,
        )
        self.assertEqual(report.n_kept, 1)

    def test_a_large_horizontal_snap_is_kept_deliberately(self):
        """No horizontal cap: ``snap_point`` returns the *nearest* navigable point, so a
        correction is the filter working. The detector's 0.5 m gate answers a different
        question — see ``reachability.py``'s docstring."""
        report = reachable_pool(
            [candidate()],
            POSE,
            snap_point=lambda p: Xyz(p.x + 1.9, p.y, p.z),
            geodesic=straight_line,
        )
        self.assertEqual(report.n_kept, 1)

    def test_the_stages_are_counted_separately(self):
        pool = [candidate(1, z=-3.0), candidate(2, z=-4.0), candidate(3, z=-5.0)]
        report = reachable_pool(
            pool,
            POSE,
            snap_point=lambda p: None if p.z == -4.0 else p,
            geodesic=lambda a, b: None if b.z == -5.0 else straight_line(a, b),
        )
        self.assertEqual(
            (report.n_kept, report.n_off_navmesh, report.n_unreachable), (1, 1, 1)
        )
        self.assertEqual(report.counters()["pool_proposed"], 3)

    def test_an_empty_input_is_an_empty_report_not_a_raise(self):
        report = reachable_pool([], POSE, snap_point=identity_snap, geodesic=straight_line)
        self.assertEqual(report.n_kept, 0)
        self.assertEqual(report.n_proposed, 0)


class TestTheSurvivorsGeometry(unittest.TestCase):
    def test_the_candidate_moves_to_its_snapped_position(self):
        report = reachable_pool(
            [candidate(z=-3.0)],
            POSE,
            snap_point=lambda p: Xyz(p.x, p.y, p.z + 1.0),
            geodesic=straight_line,
        )
        self.assertEqual(report.candidates[0].position.z, -2.0)

    def test_distance_and_bearing_are_recomputed_at_the_snapped_point(self):
        """Otherwise the scorer ranks the pool on a waypoint that does not exist."""
        report = reachable_pool(
            [candidate(z=-3.0)],
            POSE,
            snap_point=lambda p: Xyz(4.0, p.y, 0.0),
            geodesic=straight_line,
        )
        kept = report.candidates[0]
        self.assertAlmostEqual(kept.distance_m, 4.0, places=9)
        # (4, 0) is to the agent's right at yaw 0, so the bearing is negative.
        self.assertLess(kept.bearing_rad, 0.0)

    def test_the_geodesic_distance_is_recorded(self):
        report = reachable_pool(
            [candidate(z=-3.0)],
            POSE,
            snap_point=identity_snap,
            geodesic=lambda a, b: 7.5,
        )
        self.assertEqual(report.candidates[0].geodesic_m, 7.5)

    def test_everything_else_about_the_candidate_is_preserved(self):
        report = reachable_pool(
            [candidate(cid=42)], POSE, snap_point=identity_snap, geodesic=straight_line
        )
        kept = report.candidates[0]
        self.assertEqual((kept.candidate_id, kept.source, kept.cluster_size), (42, SOURCE_FRONTIER, 7))


class TestTheInvariant(unittest.TestCase):
    def test_a_non_empty_pool_passes_through(self):
        report = reachable_pool(
            [candidate()], POSE, snap_point=identity_snap, geodesic=straight_line
        )
        self.assertEqual(assert_pool(report, stage="frontier"), report.candidates)

    def test_an_empty_pool_raises(self):
        report = reachable_pool(
            [candidate()], POSE, snap_point=lambda p: None, geodesic=straight_line
        )
        with self.assertRaises(EmptyPoolError):
            assert_pool(report, stage="frontier")

    def test_the_raise_names_the_stage_and_the_counts(self):
        report = reachable_pool(
            [candidate(1), candidate(2, z=-4.0)],
            POSE,
            snap_point=identity_snap,
            geodesic=lambda a, b: None,
        )
        with self.assertRaises(EmptyPoolError) as caught:
            assert_pool(report, stage="compass fan after the frontier pool emptied")
        message = str(caught.exception)
        self.assertIn("compass fan after the frontier pool emptied", message)
        self.assertIn("2 unreachable", message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
