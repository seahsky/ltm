"""`task/prior_driver.py`'s Mac-testable half: which stops, in what order, for what class.

Everything that renders real audio is box-only by construction (ADR-0013): the seam is
`embed`, an injected callback, so a `FakeWorld` and a stub embedding function exercise the
planning and the observe-wiring without a simulator. What is checked here is the invariant
the whole design rests on: `plan_scene_tour` restricts every stop to a category `classes`
can sound from, so `walk_scene` never asks `observation_for` a question it answers `None`
to. The box half (`render_embedding_at_stop`, `run_prior_pass`) needs `tests/box`.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.vocabulary import ROOM_OF_ANCHOR
from earshot.task.prior_driver import (
    SceneTourOutcome,
    merge_scene_records,
    pass_provenance,
    plan_scene_tour,
    walk_scene,
)
from earshot.task.prior_pass import LegOutcome, TourRecord, TourStop
from earshot.types import Pose, Xyz

# The matrix's own room-balanced assignment (PR #77's measured result), reused rather than
# invented: `toilet_flush`->toilet, `snoring`->bed, `keyboard_typing`->chair. Real names
# from `audio.vocabulary`, because `class_at_category`/`categories_with_a_sound` read the
# real table and cannot be stubbed.
MATRIX_CLASSES = ("toilet_flush", "snoring", "keyboard_typing")


class _Goal:
    def __init__(self, position):
        self.position = position


class _Episode:
    def __init__(self, object_category, goals):
        self.object_category = object_category
        self.goals = tuple(goals)


class _Dataset:
    def __init__(self, episodes):
        self.episodes = tuple(episodes)


def _euclidean(a, b):
    return a.horizontal_distance_to(b)


class TestPlanSceneTour(unittest.TestCase):
    """THE INVARIANT: no stop this function plans can ever answer `None` under `classes`."""

    def test_a_category_none_of_classes_anchors_at_is_excluded(self):
        """A living-room `sofa` or `tv_monitor` contributes nothing to
        `keyboard_typing`'s bank -- `chair` is the only living-room anchor any of the
        three matrix classes reaches."""
        dataset = _Dataset([
            _Episode("chair", [_Goal(Xyz(1.0, 0.0, 0.0))]),
            _Episode("sofa", [_Goal(Xyz(2.0, 0.0, 0.0))]),
            _Episode("tv_monitor", [_Goal(Xyz(3.0, 0.0, 0.0))]),
        ])
        plan = plan_scene_tour(
            dataset, ROOM_OF_ANCHOR, MATRIX_CLASSES, Xyz(0.0, 0.0, 0.0), _euclidean
        )
        self.assertEqual([stop.category for stop in plan.stops], ["chair"])

    def test_every_category_the_matrix_bank_reaches_is_included(self):
        dataset = _Dataset([
            _Episode("chair", [_Goal(Xyz(1.0, 0.0, 0.0))]),
            _Episode("bed", [_Goal(Xyz(0.0, 0.0, 5.0))]),
            _Episode("toilet", [_Goal(Xyz(0.0, 5.0, 0.0))]),
        ])
        plan = plan_scene_tour(
            dataset, ROOM_OF_ANCHOR, MATRIX_CLASSES, Xyz(0.0, 0.0, 0.0), _euclidean
        )
        self.assertEqual(set(plan.rooms), {"living_room", "bedroom", "bathroom"})

    def test_a_narrower_class_bank_narrows_the_plan(self):
        """The same scene, planned for a bank that reaches only ONE of its rooms."""
        dataset = _Dataset([
            _Episode("chair", [_Goal(Xyz(1.0, 0.0, 0.0))]),
            _Episode("bed", [_Goal(Xyz(0.0, 0.0, 5.0))]),
        ])
        plan = plan_scene_tour(
            dataset, ROOM_OF_ANCHOR, ("snoring",), Xyz(0.0, 0.0, 0.0), _euclidean
        )
        self.assertEqual(plan.rooms, ("bedroom",))

    def test_plant_is_excluded_even_though_no_class_bank_could_ever_include_it(self):
        """`plant` has no room at all (ADR-0018's amendment): excluded by
        `ROOM_OF_ANCHOR` alone, before `classes` is even consulted."""
        dataset = _Dataset([_Episode("plant", [_Goal(Xyz(9.0, 0.0, 0.0))])])
        plan = plan_scene_tour(
            dataset, ROOM_OF_ANCHOR, MATRIX_CLASSES, Xyz(0.0, 0.0, 0.0), _euclidean
        )
        self.assertEqual(plan.stops, ())

    def test_a_scene_with_no_episodes_plans_an_empty_tour_not_a_raise(self):
        plan = plan_scene_tour(
            _Dataset([]), ROOM_OF_ANCHOR, MATRIX_CLASSES, Xyz(0.0, 0.0, 0.0), _euclidean
        )
        self.assertEqual(plan.stops, ())


class FakeWorld:
    """`test_prior_pass.FakeWorld`'s shape, reproduced: arrives after `steps_needed`."""

    def __init__(self, steps_needed):
        self._needed = dict(steps_needed)
        self._taken = {}
        self._target = None

    def follower(self, _goal_radius):
        def next_action(target):
            key = target.as_tuple()
            self._target = key
            if self._taken.get(key, 0) >= self._needed.get(key, 0):
                return None
            return "move_forward"

        return next_action

    def step(self, action):
        self._taken[self._target] = self._taken.get(self._target, 0) + 1

    def pose(self):
        return Pose(position=Xyz(0.0, 0.0, 0.0), yaw_rad=0.0)

    def geodesic_distance(self, _start, _ends):
        return 0.0


class TestWalkScene(unittest.TestCase):
    """The observe seam: `observation_for` wired in, and the invariant enforced."""

    def _plan(self):
        dataset = _Dataset([
            _Episode("toilet", [_Goal(Xyz(3.0, 0.0, 0.0))]),
            _Episode("bed", [_Goal(Xyz(1.0, 0.0, 0.0))]),
        ])
        return plan_scene_tour(
            dataset, ROOM_OF_ANCHOR, MATRIX_CLASSES, Xyz(0.0, 0.0, 0.0), _euclidean
        )

    def test_reached_stops_produce_rows_named_by_the_correct_class(self):
        plan = self._plan()
        world = FakeWorld({stop.point.as_tuple(): 0 for stop in plan.stops})
        record = walk_scene(
            world, plan, scene="S", classes=MATRIX_CLASSES,
            embed=lambda stop: [1.0, 0.0],
        )
        self.assertEqual(len(record.observations), len(plan.stops))
        classes_heard = {row["sound_class"] for row in record.observations}
        self.assertEqual(classes_heard, {"toilet_flush", "snoring"})

    def test_embed_is_called_only_for_a_reached_stop(self):
        plan = self._plan()
        far = plan.stops[-1].point.as_tuple()
        world = FakeWorld({far: 10_000})
        calls = []
        walk_scene(
            world, plan, scene="S", classes=MATRIX_CLASSES, leg_budget=3,
            embed=lambda stop: calls.append(stop.category) or [1.0, 0.0],
        )
        self.assertEqual(len(calls), len(plan.stops) - 1)

    def test_a_plan_not_restricted_first_trips_the_invariant_rather_than_writing_none(self):
        """THE FORCED FAILURE: a plan built from every category, unrestricted, reaches
        a stop `classes` cannot sound from -- and this raises rather than silently
        writing `None` into `TourRecord.observations`."""
        from earshot.task.prior_pass import candidate_stops, plan_tour

        candidates = candidate_stops({"sofa": [Xyz(9.0, 0.0, 0.0)]}, ROOM_OF_ANCHOR)
        unrestricted_plan = plan_tour(candidates, Xyz(0.0, 0.0, 0.0), _euclidean)
        world = FakeWorld({stop.point.as_tuple(): 0 for stop in unrestricted_plan.stops})
        with self.assertRaises(AssertionError) as caught:
            walk_scene(
                world, unrestricted_plan, scene="S", classes=MATRIX_CLASSES,
                embed=lambda stop: [1.0, 0.0],
            )
        self.assertIn("sofa", str(caught.exception))


class TestMergeSceneRecords(unittest.TestCase):
    def test_it_merges_the_observations_of_every_complete_record(self):
        record_a = TourRecord(
            scene="A",
            legs=(),
            observations=({
                "sound_class": "snoring", "room": "bedroom", "category": "bed",
                "embedding": [1.0, 0.0],
            },),
        )
        record_b = TourRecord(
            scene="B",
            legs=(),
            observations=({
                "sound_class": "toilet_flush", "room": "bathroom", "category": "toilet",
                "embedding": [0.0, 1.0],
            },),
        )
        semantic, episodic = merge_scene_records([record_a, record_b])
        self.assertEqual(
            sorted(entry.sound_class for entry in semantic.entries),
            ["snoring", "toilet_flush"],
        )
        self.assertEqual(len(episodic), 0)


class TestPassProvenance(unittest.TestCase):
    """Every requested scene lands in exactly one of THREE provenance lists.

    The matrix-1 review's D3: the original provenance block had only `scenes_complete`
    and `scenes_failed`, so a scene that LOADED but whose tour left a leg unreached was
    in neither -- excluded from the merge, invisible on the record, and a sweep gating
    on the store file's existence ran its seen cells byte-identical to its unseen cells
    with no error anywhere. Both arms below, per ADR-0014: the healthy pass, and the
    incomplete tour that used to vanish.
    """

    @staticmethod
    def _observation():
        return {
            "sound_class": "snoring", "room": "bedroom", "category": "bed",
            "embedding": [1.0, 0.0],
        }

    @staticmethod
    def _leg(reached):
        return LegOutcome(
            stop=TourStop(room="bedroom", category="bed", point=Xyz(0.0, 0.0, 0.0)),
            reached=reached,
            steps=3,
            final_gap_m=None if reached else 4.0,
            reason="reached" if reached else "budget",
        )

    def test_a_complete_tour_and_a_failed_load_land_in_their_own_lists(self):
        complete = TourRecord(
            scene="A", legs=(self._leg(True),), observations=(self._observation(),)
        )
        outcomes = [
            SceneTourOutcome(scene="A", record=complete),
            SceneTourOutcome(scene="B", record=None, error="no mesh"),
        ]
        provenance = pass_provenance(
            outcomes, split="val", classes=["snoring"], seed=7, scenes=["A", "B"]
        )
        self.assertEqual(provenance["scenes_complete"], ["A"])
        self.assertEqual(provenance["scenes_incomplete"], [])
        self.assertEqual([f["scene"] for f in provenance["scenes_failed"]], ["B"])

    def test_an_incomplete_tour_is_on_the_record_not_silently_absent(self):
        """The forced-failure arm: a loaded scene with one unreached leg must appear in
        `scenes_incomplete` -- and NOT in `scenes_complete` -- so that every requested
        scene is accounted for somewhere."""
        partial = TourRecord(
            scene="C",
            legs=(self._leg(True), self._leg(False)),
            observations=(self._observation(),),
        )
        outcomes = [SceneTourOutcome(scene="C", record=partial)]
        provenance = pass_provenance(
            outcomes, split="val", classes=["snoring"], seed=7, scenes=["C"]
        )
        self.assertEqual(provenance["scenes_complete"], [])
        self.assertEqual(
            [entry["scene"] for entry in provenance["scenes_incomplete"]], ["C"]
        )
        self.assertFalse(provenance["scenes_incomplete"][0]["complete"])
        accounted = (
            set(provenance["scenes_complete"])
            | {entry["scene"] for entry in provenance["scenes_incomplete"]}
            | {entry["scene"] for entry in provenance["scenes_failed"]}
        )
        self.assertEqual(accounted, set(provenance["scenes_requested"]))


class TestSceneTourOutcome(unittest.TestCase):
    def test_a_failed_scene_carries_its_reason_and_no_record(self):
        outcome = SceneTourOutcome(scene="X", record=None, error="no mesh")
        self.assertFalse(outcome.ok)
        payload = outcome.as_dict()
        self.assertEqual(payload["error"], "no mesh")
        self.assertIsNone(payload["complete"])

    def test_a_successful_scene_carries_its_record_summary(self):
        record = TourRecord(
            scene="X",
            legs=(),
            observations=({
                "sound_class": "snoring", "room": "bedroom", "category": "bed",
                "embedding": [1.0],
            },),
        )
        outcome = SceneTourOutcome(scene="X", record=record)
        self.assertTrue(outcome.ok)
        payload = outcome.as_dict()
        self.assertEqual(payload["n_observations"], 1)
        self.assertIsNone(payload["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
