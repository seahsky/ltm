"""`task/memory_prior.py`: the place the memory names once the room has gone quiet.

Every path here is decidable without a simulator, which is the point of the module's shape:
the caller supplies the scene's category table and a distance function, so a wrong recall, a
category the house does not have, and an instance on a disconnected island are three
separate assertions rather than three ways of getting `None`.

ADR-0014's rule runs through the file: each capability is exercised with its forced failure
beside it, and the three misses are asserted apart because collapsing them would make "the
memory is empty" and "the memory was right and the house has no stove" the same measurement.
"""

import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.store import EpisodicEntry, EpisodicStore, SemanticEntry, SemanticStore
from earshot.task.memory_prior import (
    RUN_DISCLOSURE,
    MemoryPrior,
    PriorMiss,
    category_points,
    points_by_category_for_cell,
    resolve_prior,
)
from earshot.types import Xyz


def _entry(sound_class, room, category, embedding, donor_scene="donor_1"):
    return SemanticEntry(
        sound_class=sound_class,
        room=room,
        category=category,
        embedding=np.asarray(embedding, dtype=np.float32),
        donor_scene=donor_scene,
    )


def _episodic(scene, room, category, point):
    return EpisodicEntry(scene=scene, room=room, category=category, point=Xyz(*point))


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


def _euclidean(origin):
    return lambda point: origin.horizontal_distance_to(point)


class TestCategoryPoints(unittest.TestCase):
    """The adapter from the scene's own annotations. Duck-typed, so no dataset is built."""

    def test_it_groups_every_goal_by_its_category(self):
        table = category_points(
            _Dataset(
                [
                    _Episode("stove", [_Goal(Xyz(1.0, 0.0, 0.0))]),
                    _Episode("toilet", [_Goal(Xyz(0.0, 0.0, 5.0))]),
                ]
            )
        )
        self.assertEqual(sorted(table), ["stove", "toilet"])
        self.assertEqual(table["stove"], (Xyz(1.0, 0.0, 0.0),))

    def test_the_same_instance_hoisted_across_episodes_is_counted_once(self):
        # `goals_by_category` hoists one copy per category across every episode of the
        # scene, so without the dedup a single stove would report n_instances=3.
        table = category_points(
            _Dataset([_Episode("stove", [_Goal(Xyz(1.0, 0.0, 0.0))]) for _ in range(3)])
        )
        self.assertEqual(len(table["stove"]), 1)

    def test_two_real_instances_are_kept_apart(self):
        table = category_points(
            _Dataset(
                [
                    _Episode(
                        "stove",
                        [_Goal(Xyz(1.0, 0.0, 0.0)), _Goal(Xyz(9.0, 0.0, 0.0))],
                    )
                ]
            )
        )
        self.assertEqual(len(table["stove"]), 2)

    def test_a_scene_with_no_episodes_is_an_empty_table_not_a_raise(self):
        self.assertEqual(category_points(_Dataset([])), {})


class TestPointsByCategoryForCell(unittest.TestCase):
    """THE FIX: what makes `HEARD_SEEN` and `HEARD_UNSEEN` capable of disagreeing at all.

    `resolve_prior` votes on the semantic store, which `stores_for_cell` never filters
    by scene -- `HEARD_SEEN` and `HEARD_UNSEEN` recall the identical category from the
    identical store. `category_points` alone resolves that category through the SAME
    scene ground truth in both cells, so a `MemoryContext` built from it directly is one
    the two cells cannot be told apart by. This class is the forced-failure arm made an
    assertion: build the mapping without the episodic recall this function adds, and the
    two cells collapse to one number.
    """

    def setUp(self):
        self.ground_truth = _Dataset(
            [_Episode("stove", [_Goal(Xyz(9.0, 0.0, 0.0))])]
        )

    def test_a_toured_categorys_own_recalled_point_wins_over_ground_truth(self):
        episodic = EpisodicStore(entries=(
            _episodic("scene_a", "kitchen", "stove", (1.0, 0.0, 0.0)),
        ))
        table = points_by_category_for_cell(self.ground_truth, episodic, "scene_a")
        self.assertEqual(table["stove"], (Xyz(1.0, 0.0, 0.0),))

    def test_an_untoured_scene_falls_back_to_ground_truth(self):
        """The `HEARD_UNSEEN` cell: `without_scene` has already emptied `episodic` before
        this is called, so every category falls through to `category_points`."""
        table = points_by_category_for_cell(self.ground_truth, EpisodicStore(), "scene_a")
        self.assertEqual(table["stove"], (Xyz(9.0, 0.0, 0.0),))

    def test_a_category_the_tour_never_reached_also_falls_back(self):
        """The tour visited a different room; this category is untouched by it."""
        episodic = EpisodicStore(entries=(
            _episodic("scene_a", "bathroom", "toilet", (2.0, 0.0, 0.0)),
        ))
        table = points_by_category_for_cell(self.ground_truth, episodic, "scene_a")
        self.assertEqual(table["stove"], (Xyz(9.0, 0.0, 0.0),))

    def test_a_recall_from_a_different_scene_does_not_leak_in(self):
        episodic = EpisodicStore(entries=(
            _episodic("scene_b", "kitchen", "stove", (1.0, 0.0, 0.0)),
        ))
        table = points_by_category_for_cell(self.ground_truth, episodic, "scene_a")
        self.assertEqual(table["stove"], (Xyz(9.0, 0.0, 0.0),))

    def test_seen_and_unseen_now_resolve_to_different_places(self):
        """The headline: two cells, same store, same vote, different answer -- because
        this function, not `category_points` alone, is what the driver must call."""
        episodic = EpisodicStore(entries=(
            _episodic("scene_a", "kitchen", "stove", (1.0, 0.0, 0.0)),
        ))
        seen = points_by_category_for_cell(self.ground_truth, episodic, "scene_a")
        unseen = points_by_category_for_cell(self.ground_truth, EpisodicStore(), "scene_a")
        self.assertNotEqual(seen["stove"], unseen["stove"])

    def test_it_mutates_neither_the_dataset_nor_the_episodic_store(self):
        episodic = EpisodicStore(entries=(
            _episodic("scene_a", "kitchen", "stove", (1.0, 0.0, 0.0)),
        ))
        before = episodic.entries
        points_by_category_for_cell(self.ground_truth, episodic, "scene_a")
        self.assertIs(episodic.entries, before)


class TestTheHealthyArm(unittest.TestCase):
    """A learned class-to-category association, resolved to a place in a scene."""

    def setUp(self):
        # Heard at a stove near [1,0,0] on two prior tours, at a toilet near [0,1,0] once.
        self.store = SemanticStore(
            entries=(
                _entry("alarm", "kitchen", "stove", [1.0, 0.0, 0.0]),
                _entry("alarm", "kitchen", "stove", [0.9, 0.1, 0.0]),
                _entry("flush", "bathroom", "toilet", [0.0, 1.0, 0.0]),
            )
        )
        self.scene = {
            "stove": (Xyz(3.0, 0.0, 0.0), Xyz(12.0, 0.0, 0.0)),
            "toilet": (Xyz(0.0, 0.0, 20.0),),
        }

    def _resolve(self, query, origin=Xyz(0.0, 0.0, 0.0), **kwargs):
        return resolve_prior(
            self.store,
            np.asarray(query, dtype=np.float32),
            k=kwargs.pop("k", 2),
            points_by_category=kwargs.pop("points_by_category", self.scene),
            distance_to=kwargs.pop("distance_to", _euclidean(origin)),
        )

    def test_it_names_the_nearest_instance_of_the_recalled_category(self):
        prior, miss = self._resolve([1.0, 0.0, 0.0])
        self.assertIsNone(miss)
        print(
            "\n  [prior] recalled {!r} at cosine {:.4f}, {} instance(s), nearest at "
            "{:.2f} m".format(
                prior.category, prior.confidence, prior.n_instances, prior.distance_m
            ),
            flush=True,
        )
        self.assertEqual(prior.category, "stove")
        self.assertEqual(prior.target, Xyz(3.0, 0.0, 0.0))
        self.assertEqual(prior.n_instances, 2)
        self.assertAlmostEqual(prior.distance_m, 3.0, places=6)

    def test_a_different_class_lands_on_a_different_object(self):
        # The forced arm of the recall itself: if the vote returned a constant, this passes
        # only by accident.
        prior, miss = self._resolve([0.0, 1.0, 0.0], k=1)
        self.assertIsNone(miss)
        self.assertEqual(prior.category, "toilet")
        self.assertEqual(prior.target, Xyz(0.0, 0.0, 20.0))

    def test_the_nearest_is_by_the_callers_metric_not_by_list_order(self):
        # Stand at the far stove: the answer must move to it.
        prior, _ = self._resolve([1.0, 0.0, 0.0], origin=Xyz(11.0, 0.0, 0.0))
        self.assertEqual(prior.target, Xyz(12.0, 0.0, 0.0))

    def test_the_confidence_is_the_votes_own_cosine(self):
        prior, _ = self._resolve([1.0, 0.0, 0.0])
        expected_second = 0.9 / float(np.linalg.norm([0.9, 0.1, 0.0]))
        self.assertAlmostEqual(prior.confidence, (1.0 + expected_second) / 2.0, places=6)

    def test_it_mutates_neither_the_store_nor_the_scene_table(self):
        before_store = [(e.category, e.sound_class) for e in self.store.entries]
        before_scene = {key: tuple(value) for key, value in self.scene.items()}
        self._resolve([1.0, 0.0, 0.0])
        self.assertEqual(
            before_store, [(e.category, e.sound_class) for e in self.store.entries]
        )
        self.assertEqual(before_scene, {k: tuple(v) for k, v in self.scene.items()})


class TestTheThreeMissesAreThreeDifferentFacts(unittest.TestCase):
    """Collapsing these into one `None` is the absence bug this repo has paid for twice."""

    def setUp(self):
        self.store = SemanticStore(
            entries=(_entry("alarm", "kitchen", "stove", [1.0, 0.0, 0.0]),)
        )
        self.query = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    def _resolve(self, store, scene, distance_to=None):
        return resolve_prior(
            store,
            self.query,
            k=1,
            points_by_category=scene,
            distance_to=distance_to or _euclidean(Xyz(0.0, 0.0, 0.0)),
        )

    def test_an_empty_store_is_no_prediction(self):
        # The `not_heard` cells' expected value. It must not read as a wrong recall.
        prior, miss = self._resolve(SemanticStore(), {"stove": (Xyz(1.0, 0.0, 0.0),)})
        self.assertIsNone(prior)
        self.assertIs(miss, PriorMiss.NO_PREDICTION)

    def test_a_degenerate_query_is_no_prediction_not_a_low_score(self):
        prior, miss = resolve_prior(
            self.store,
            np.zeros(3, dtype=np.float32),
            k=1,
            points_by_category={"stove": (Xyz(1.0, 0.0, 0.0),)},
            distance_to=_euclidean(Xyz(0.0, 0.0, 0.0)),
        )
        self.assertIsNone(prior)
        self.assertIs(miss, PriorMiss.NO_PREDICTION)

    def test_a_house_with_no_such_object_is_category_absent(self):
        # The memory answered. The scene did not have the thing. A generalization failure
        # of a different kind from a wrong recall.
        prior, miss = self._resolve(self.store, {"toilet": (Xyz(1.0, 0.0, 0.0),)})
        self.assertIsNone(prior)
        self.assertIs(miss, PriorMiss.CATEGORY_ABSENT)

    def test_an_empty_scene_table_is_category_absent(self):
        prior, miss = self._resolve(self.store, {})
        self.assertIsNone(prior)
        self.assertIs(miss, PriorMiss.CATEGORY_ABSENT)

    def test_instances_with_no_route_are_unreachable_not_absent(self):
        prior, miss = self._resolve(
            self.store, {"stove": (Xyz(1.0, 0.0, 0.0),)}, distance_to=lambda _p: None
        )
        self.assertIsNone(prior)
        self.assertIs(miss, PriorMiss.UNREACHABLE)

    def test_one_routable_instance_among_unroutable_ones_still_answers(self):
        # The forced arm of UNREACHABLE: excluding a routeless point must not exclude the
        # whole category.
        reachable = Xyz(7.0, 0.0, 0.0)
        prior, miss = self._resolve(
            self.store,
            {"stove": (Xyz(1.0, 0.0, 0.0), reachable)},
            distance_to=lambda point: None if point != reachable else 7.0,
        )
        self.assertIsNone(miss)
        self.assertEqual(prior.target, reachable)
        # `n_instances` counts what the scene HAS, not what routed: the two numbers are
        # different claims and the audit wants the first.
        self.assertEqual(prior.n_instances, 2)

    def test_every_miss_has_a_distinct_value(self):
        values = [m.value for m in PriorMiss]
        self.assertEqual(len(values), len(set(values)))
        self.assertEqual(len(values), 3)


class TestTheAuditSurface(unittest.TestCase):
    def test_the_metrics_are_all_floats_and_carry_no_string(self):
        prior = MemoryPrior(
            category="stove",
            confidence=0.87,
            target=Xyz(1.0, 0.0, 2.0),
            distance_m=3.5,
            n_instances=2,
        )
        metrics = prior.as_metrics()
        self.assertTrue(all(isinstance(v, float) for v in metrics.values()))
        self.assertNotIn("stove", metrics.values())
        self.assertEqual(
            sorted(metrics),
            [
                "memory_prior_confidence",
                "memory_prior_distance_m",
                "memory_prior_instances",
            ],
        )

    def test_the_disclosure_says_what_is_annotated_and_what_is_learned(self):
        # §8's rule: a run that takes a privilege says so in words a reader cannot miss.
        self.assertIn("ObjectNav annotations", RUN_DISCLOSURE)
        self.assertIn("learned", RUN_DISCLOSURE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
