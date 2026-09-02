"""`task/prior_build.py`: which sound is heard where, and tours as one merged store.

The most important thing in this file is `TestTheMechanismCannotLearnAnythingYet`. It does
not test a bug; it PINS A FACT ABOUT THE TASK that decides whether the memory arm can work
at all, so that the fact is a failing assertion the day someone changes it rather than a
paragraph in a docstring nobody re-reads. Both halves of it are things this module found
while being built, and neither is fixable here.
"""

import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.clips import ANOMALY_CLASSES
from earshot.audio.vocabulary import HM3D_GOAL_CATEGORIES
from earshot.memory.store import EpisodicStore, SemanticStore
from earshot.task.memory_build import MemoryBuildError
from earshot.task.prior_build import (
    anchor_of_run_class,
    categories_with_a_sound,
    class_at_category,
    merge_stores,
    observation_for,
    stores_from_records,
)
from earshot.task.prior_pass import LegOutcome, TourRecord, TourStop
from earshot.types import Xyz


def _stop(room="bedroom", category="bed", point=(1.0, 0.0, 2.0)):
    return TourStop(room=room, category=category, point=Xyz(*point))


def _record(scene, observations, stops=None):
    stops = stops or [_stop()]
    return TourRecord(
        scene=scene,
        legs=tuple(
            LegOutcome(stop=stop, reached=True, steps=3, final_gap_m=0.4, reason="reached")
            for stop in stops
        ),
        observations=tuple(observations),
    )


class TestTheClassNamespaces(unittest.TestCase):
    """The tree has two names for one sound and a store must hold only the run's."""

    def test_the_run_name_bridges_to_the_vocabulary_name(self):
        # `alarm` (what `--anomaly-class` takes) -> `clock_alarm` (ESC-50, what the
        # vocabulary calls it) -> `bed` (its anchor). If this chain breaks, the store fills
        # with rows keyed on a name `without_class` will never filter.
        self.assertEqual(anchor_of_run_class("alarm"), "bed")

    def test_a_class_the_vocabulary_has_no_row_for_has_no_anchor(self):
        # Real state, not a bug: `glass_break` is one of the three locked emergency classes
        # and has no ESC-50 vocabulary entry.
        self.assertIsNone(anchor_of_run_class("glass_break"))

    def test_everything_returned_is_in_the_runs_namespace(self):
        # The bug this guards: a store holding `clock_alarm` while a sweep filters `alarm`
        # would filter nothing, and hand the not-heard cells a full store. Four identical
        # cells, and no test downstream of the store could see it.
        for category, name in categories_with_a_sound(HM3D_GOAL_CATEGORIES).items():
            self.assertIn(name, ANOMALY_CLASSES, category)

    def test_the_choice_is_deterministic_when_classes_share_an_anchor(self):
        # `alarm` and `baby_cry` both anchor at `bed`; the name that sorts first wins, so
        # two prior passes over one scene teach the same association.
        self.assertEqual(class_at_category("bed"), "alarm")
        self.assertEqual(class_at_category("bed"), class_at_category("bed"))

    def test_restricting_to_a_runs_own_class_moves_the_answer(self):
        self.assertEqual(class_at_category("bed", classes=["baby_cry"]), "baby_cry")
        self.assertIsNone(class_at_category("bed", classes=["glass_break"]))


class TestTheMechanismCannotLearnAnythingYet(unittest.TestCase):
    """Two facts about the TASK that make the tour-learned category unlearnable today.

    Neither is fixable in `prior_build.py`. Both are pinned here so that the day the task
    changes, these fail and say what to re-read -- rather than a sweep quietly running four
    cells that could never have differed.
    """

    def test_the_run_class_set_maps_onto_a_single_category(self):
        """A predictor with one answer is not a predictor.

        `clips.ANOMALY_CLASSES` is three names. `alarm` and `baby_cry` both anchor at
        `bed`; `glass_break` has no vocabulary row. Over HM3D's six goal categories that
        leaves exactly one category any sound is anchored at, so `predict_category` would
        answer `bed` to every query it ever got.

        FIXING IT means widening the sounding class set from the three emergency names to
        the vocabulary's seventeen, which is a task decision and re-baselines `abl-1`.
        """
        table = categories_with_a_sound(HM3D_GOAL_CATEGORIES)
        print(
            "\n  [prior] {} run class(es) over {} goal categories -> {} category/ies "
            "that teach anything: {}".format(
                len(ANOMALY_CLASSES), len(HM3D_GOAL_CATEGORIES), len(table), table
            ),
            flush=True,
        )
        self.assertEqual(
            table,
            {"bed": "alarm"},
            "the run-class-to-category table changed; re-read prior_build's header and "
            "decide whether the memory arm is now learnable",
        )

    def test_the_episode_builder_still_ignores_the_sound_class_when_placing(self):
        """The source is placed by GEOMETRY, so a learned category has nothing to predict.

        `place_anomaly_source` ranks candidates by `(same_category, separation, category)`
        and nothing in `task/` calls `vocabulary.anchor_object`. In every episode this repo
        has run, `abl-1` included, the alarm sits at whatever object cleared the separation
        rules. A store that learned "alarm at bed" is predicting a rule the task does not
        follow.

        FIXING IT means anchoring the placement to the class, which changes the task and
        re-baselines `abl-1`. This asserts the CURRENT state so the change is visible.
        """
        import inspect

        from earshot.task import dataset

        source = inspect.getsource(dataset)
        self.assertNotIn(
            "anchor_object",
            source,
            "task/dataset.py now reads the anchor table — if placement is anchored to the "
            "sound class, prior_build's header is out of date and the memory arm may now "
            "be learnable",
        )
        self.assertIn("qualifying.sort(key=lambda row: (row[1], row[0], row[2]))", source)


class TestObservationFor(unittest.TestCase):
    def test_a_stop_whose_category_hosts_the_runs_class_becomes_a_row(self):
        payload = observation_for(_stop(), [1.0, 0.0], classes=["alarm"])
        self.assertEqual(payload["sound_class"], "alarm")
        self.assertEqual(payload["category"], "bed")
        self.assertEqual(payload["room"], "bedroom")

    def test_a_stop_with_no_sound_at_its_category_is_none_not_a_raise(self):
        # A tour legitimately walks past a sofa when the run's class is `alarm`.
        self.assertIsNone(observation_for(_stop(category="sofa"), [1.0, 0.0]))

    def test_a_malformed_embedding_still_raises(self):
        # The two failures are different and only this one is a bug.
        with self.assertRaises(MemoryBuildError):
            observation_for(_stop(), [], classes=["alarm"])


class TestMerging(unittest.TestCase):
    def test_two_scenes_merge_into_one_store_of_both(self):
        a = _record("scene_a", [observation_for(_stop(), [1.0, 0.0], classes=["alarm"])])
        b = _record("scene_b", [observation_for(_stop(), [0.0, 1.0], classes=["alarm"])])
        semantic, episodic = stores_from_records([a, b])
        print(
            "\n  [prior] merged {} semantic row(s) over donor scene(s) {}".format(
                len(semantic), semantic.donor_scenes
            ),
            flush=True,
        )
        self.assertEqual(len(semantic), 2)
        self.assertEqual(semantic.donor_scenes, ("scene_a", "scene_b"))
        self.assertEqual(episodic.scenes, ("scene_a", "scene_b"))

    def test_an_empty_merge_is_two_empty_stores_not_a_raise(self):
        semantic, episodic = merge_stores([])
        self.assertEqual(len(semantic), 0)
        self.assertEqual(len(episodic), 0)

    def test_mixed_encoder_widths_raise_at_the_merge_not_at_a_later_cosine(self):
        a = _record("scene_a", [observation_for(_stop(), [1.0, 0.0], classes=["alarm"])])
        b = _record("scene_b", [observation_for(_stop(), [1.0, 0.0, 0.0], classes=["alarm"])])
        with self.assertRaises(ValueError):
            stores_from_records([a, b])

    def test_a_malformed_record_names_the_scene_it_came_from(self):
        good = _record("scene_a", [observation_for(_stop(), [1.0], classes=["alarm"])])
        bad = _record("scene_b", [{"sound_class": "alarm", "room": "bedroom"}])
        with self.assertRaises(MemoryBuildError) as caught:
            stores_from_records([good, bad])
        self.assertIn("scene_b", str(caught.exception))

    def test_the_merge_preserves_the_callers_order(self):
        a = _record("scene_a", [observation_for(_stop(), [1.0], classes=["alarm"])])
        b = _record("scene_b", [observation_for(_stop(), [2.0], classes=["alarm"])])
        semantic, _ = stores_from_records([b, a])
        self.assertEqual(
            [entry.donor_scene for entry in semantic.entries], ["scene_b", "scene_a"]
        )

    def test_merging_does_not_mutate_its_inputs(self):
        left = (SemanticStore(), EpisodicStore())
        right = (SemanticStore(), EpisodicStore())
        merge_stores([left, right])
        self.assertEqual(left[0].entries, ())
        self.assertEqual(right[1].entries, ())


class TestTheStoreIsQueryableEndToEnd(unittest.TestCase):
    """A tour, a store, and the query the unseen cell would run over it."""

    def test_a_tour_becomes_a_store_that_answers_the_category(self):
        record = _record(
            "scene_a", [observation_for(_stop(), [1.0, 0.0], classes=["alarm"])]
        )
        semantic, _ = stores_from_records([record])
        answer = semantic.predict_category(np.array([1.0, 0.0], dtype=np.float32), k=1)
        self.assertEqual(answer[0], "bed")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
