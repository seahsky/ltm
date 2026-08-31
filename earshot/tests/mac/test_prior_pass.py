"""The scripted prior pass: the plan is deterministic, and a partial tour cannot pass as whole.

ADR-0018 left the prior pass open; 2026-08-21 answered it with a scripted tour rather than an
agent-driven episode, on the grounds that the matrix carries a measured 16.2% flip rate and
cannot afford another variance source. That argument only holds if the route really is
deterministic and if an incomplete tour is visibly incomplete, so both are tested here rather
than asserted in a docstring.

The walk is exercised against a fake world. The real one needs habitat-sim and belongs in
`tests/box`; what is checkable here is the control flow that decides reached from abandoned,
and that is where the failure this repo keeps finding would live.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.task.prior_pass import candidate_stops, plan_tour, walk_tour
from earshot.types import Pose, Xyz

ROOMS = {
    "toilet": "bathroom",
    "bed": "bedroom",
    "sofa": "living_room",
    "chair": "living_room",
}
START = Xyz(0.0, 0.0, 0.0)


def euclidean(a, b):
    return a.horizontal_distance_to(b)


def blocked(*points):
    """A geodesic that has no route to any of `points`."""
    dead = {p.as_tuple() for p in points}

    def measure(a, b):
        if b.as_tuple() in dead:
            return None
        return euclidean(a, b)

    return measure


class TestCandidateStops(unittest.TestCase):
    def test_a_category_with_no_room_is_skipped_not_raised(self):
        """`plant` has no room by design, and HM3D scenes publish plant goals."""
        stops = candidate_stops(
            {"toilet": [Xyz(1.0, 0.0, 0.0)], "plant": [Xyz(2.0, 0.0, 0.0)]}, ROOMS
        )
        self.assertEqual([stop.category for stop in stops], ["toilet"])

    def test_every_instance_becomes_a_candidate(self):
        stops = candidate_stops({"chair": [Xyz(1.0, 0.0, 0.0), Xyz(2.0, 0.0, 0.0)]}, ROOMS)
        self.assertEqual(len(stops), 2)
        self.assertEqual({stop.room for stop in stops}, {"living_room"})


class TestPlanTour(unittest.TestCase):
    def _candidates(self):
        return candidate_stops(
            {
                "toilet": [Xyz(3.0, 0.0, 0.0)],
                "bed": [Xyz(1.0, 0.0, 0.0)],
                "sofa": [Xyz(9.0, 0.0, 0.0)],
                "chair": [Xyz(5.0, 0.0, 0.0)],
            },
            ROOMS,
        )

    def test_one_stop_per_room(self):
        """A second sofa teaches the room-level store nothing the first did not."""
        plan = plan_tour(self._candidates(), START, euclidean)
        self.assertEqual(len(plan.stops), len(set(plan.rooms)))
        self.assertEqual(set(plan.rooms), {"bathroom", "bedroom", "living_room"})

    def test_the_living_room_is_reached_by_its_nearer_object(self):
        plan = plan_tour(self._candidates(), START, euclidean)
        living = [stop for stop in plan.stops if stop.room == "living_room"][0]
        self.assertEqual(living.category, "chair")

    def test_the_route_is_nearest_first(self):
        plan = plan_tour(self._candidates(), START, euclidean)
        self.assertEqual(list(plan.rooms), ["bedroom", "bathroom", "living_room"])

    def test_the_plan_is_identical_across_repeated_calls(self):
        """Determinism is the entire argument for scripting rather than driving an agent."""
        first = plan_tour(self._candidates(), START, euclidean)
        second = plan_tour(self._candidates(), START, euclidean)
        self.assertEqual(first.as_dict(), second.as_dict())

    def test_input_order_does_not_change_the_route(self):
        forward = plan_tour(self._candidates(), START, euclidean)
        backward = plan_tour(tuple(reversed(self._candidates())), START, euclidean)
        self.assertEqual(forward.rooms, backward.rooms)

    def test_an_unroutable_candidate_is_recorded_with_a_reason(self):
        """23 of 365 episodes had no route to their source and nothing counted them."""
        candidates = self._candidates()
        plan = plan_tour(candidates, START, blocked(Xyz(3.0, 0.0, 0.0)))
        self.assertNotIn("bathroom", plan.rooms)
        reasons = {stop.category: reason for stop, reason in plan.unreachable}
        self.assertIn("toilet", reasons)
        self.assertIn("no navmesh route", reasons["toilet"])

    def test_a_room_survives_when_only_one_of_its_objects_is_unroutable(self):
        candidates = candidate_stops(
            {"sofa": [Xyz(9.0, 0.0, 0.0)], "chair": [Xyz(5.0, 0.0, 0.0)]}, ROOMS
        )
        plan = plan_tour(candidates, START, blocked(Xyz(5.0, 0.0, 0.0)))
        self.assertEqual(plan.rooms, ("living_room",))
        self.assertEqual(plan.stops[0].category, "sofa")

    def test_no_candidates_gives_an_empty_plan_rather_than_raising(self):
        plan = plan_tour((), START, euclidean)
        self.assertEqual(plan.stops, ())


class FakeWorld:
    """Steps toward the target and arrives after `steps_needed`, or never."""

    def __init__(self, steps_needed, *, gap=0.0, refuse=False):
        self._needed = dict(steps_needed)
        self._taken = {}
        self._target = None
        self._gap = gap
        self._refuse = refuse
        self.actions = []

    def follower(self, _goal_radius):
        def next_action(target):
            if self._refuse:
                raise RuntimeError("no navmesh route")
            key = target.as_tuple()
            self._target = key
            if self._taken.get(key, 0) >= self._needed.get(key, 0):
                return None
            return "move_forward"

        return next_action

    def step(self, action):
        self.actions.append(action)
        self._taken[self._target] = self._taken.get(self._target, 0) + 1
        return True

    def pose(self):
        return Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)

    def geodesic_distance(self, _start, _ends):
        return self._gap


class TestWalkTour(unittest.TestCase):
    def _plan(self):
        return plan_tour(
            candidate_stops(
                {"toilet": [Xyz(3.0, 0.0, 0.0)], "bed": [Xyz(1.0, 0.0, 0.0)]}, ROOMS
            ),
            START,
            euclidean,
        )

    def test_a_tour_that_arrives_everywhere_is_complete(self):
        plan = self._plan()
        world = FakeWorld({stop.point.as_tuple(): 3 for stop in plan.stops})
        record = walk_tour(world, plan, scene="S")
        self.assertTrue(record.complete)
        self.assertEqual(set(record.rooms_reached), {"bathroom", "bedroom"})
        self.assertTrue(all(leg.reason == "arrived" for leg in record.legs))

    def test_a_leg_over_budget_is_abandoned_and_the_tour_is_not_complete(self):
        """A partial tour reintroduces exactly the coverage variance scripting removes."""
        plan = self._plan()
        far = plan.stops[-1].point.as_tuple()
        world = FakeWorld({far: 10_000}, gap=4.2)
        record = walk_tour(world, plan, scene="S", leg_budget=5)
        self.assertFalse(record.complete)
        abandoned = [leg for leg in record.legs if not leg.reached]
        self.assertEqual(len(abandoned), 1)
        self.assertEqual(abandoned[0].steps, 5)
        self.assertAlmostEqual(abandoned[0].final_gap_m, 4.2, places=6)
        self.assertIn("budget", abandoned[0].reason)

    def test_an_abandoned_leg_does_not_stop_the_remaining_legs(self):
        """Throwing away the rooms that DID work would waste the pass."""
        plan = self._plan()
        first = plan.stops[0].point.as_tuple()
        world = FakeWorld({first: 10_000})
        record = walk_tour(world, plan, scene="S", leg_budget=4)
        self.assertEqual(len(record.legs), len(plan.stops))
        self.assertFalse(record.legs[0].reached)
        self.assertTrue(record.legs[1].reached)

    def test_a_follower_that_refuses_is_recorded_not_raised(self):
        plan = self._plan()
        record = walk_tour(FakeWorld({}, refuse=True), plan, scene="S")
        self.assertFalse(record.complete)
        self.assertTrue(all("refused" in leg.reason for leg in record.legs))
        self.assertTrue(all(leg.steps == 0 for leg in record.legs))

    def test_observe_fires_once_per_ARRIVED_stop_only(self):
        """An observation from a stop the agent never reached is fabricated audio."""
        plan = self._plan()
        first = plan.stops[0].point.as_tuple()
        world = FakeWorld({first: 10_000})
        seen = []
        record = walk_tour(
            world,
            plan,
            scene="S",
            leg_budget=3,
            observe=lambda stop: seen.append(stop.room) or {"room": stop.room},
        )
        self.assertEqual(seen, ["bathroom"])
        self.assertEqual(len(record.observations), 1)

    def test_an_empty_plan_is_not_complete(self):
        """Nothing toured is NOT_RUN, and NOT_RUN is red."""
        record = walk_tour(FakeWorld({}), plan_tour((), START, euclidean), scene="S")
        self.assertFalse(record.complete)

    def test_the_record_round_trips_to_a_dict(self):
        plan = self._plan()
        world = FakeWorld({stop.point.as_tuple(): 1 for stop in plan.stops})
        payload = walk_tour(world, plan, scene="S").as_dict()
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["scene"], "S")
        self.assertEqual(len(payload["legs"]), 2)


if __name__ == "__main__":
    unittest.main()
