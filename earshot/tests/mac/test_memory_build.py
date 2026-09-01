"""The join between a walked tour and the two stores, and the carving of the four cells.

Three properties carry the design and each has a class here.

**The four cells come out of ONE built store.** ADR-0018 amendment (a). If they did not,
the same episode could not run in all four and `episode_diff` would have nothing to pair,
which costs the design roughly 3 points of MDE at the same rendering cost. `TestCells`
asserts the carve is a pure filter over shared inputs and that the inputs survive it.

**Nothing is dropped quietly.** A tour observation with no embedding is a stop where the
render or the encoder failed, and a store one row short votes differently for a reason
nothing on disk records. It raises. `TestMalformedObservations` holds every branch.

**The JSON round trip is exact.** The prior pass and the run that consults its store are
different processes, so the file IS the contract. A cosine that moves in the eighth digit
between the tour and the run is a store answering a different question from the one it
learned, and it would never show up as an error.
"""

import json
import pathlib
import tempfile
import unittest

import numpy as np
from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.store import EpisodicStore, MemoryCondition, SemanticStore
from earshot.task.memory_build import (
    STORE_FORMAT_VERSION,
    MemoryBuildError,
    dump_stores,
    episodic_from_tour,
    load_stores,
    semantic_from_tour,
    stores_for_cell,
    tour_observation,
)
from earshot.task.prior_pass import LegOutcome, TourRecord, TourStop
from earshot.types import Xyz


def _stop(room, category, point=(0.0, 0.0, 0.0)):
    return TourStop(room=room, category=category, point=Xyz.from_sequence(point))


def _leg(stop, reached=True):
    return LegOutcome(
        stop=stop,
        reached=reached,
        steps=12,
        final_gap_m=0.4 if reached else 3.1,
        reason="arrived" if reached else "budget of 200 steps exhausted",
    )


def _record(scene="sceneA", legs=(), observations=()):
    return TourRecord(scene=scene, legs=tuple(legs), observations=tuple(observations))


def _tour(scene="sceneA"):
    """A two-room tour that reached both, heard `alarm` at both, plus one abandoned leg."""
    bath = _stop("bathroom", "toilet", (1.0, 0.0, 2.0))
    bed = _stop("bedroom", "bed", (5.0, 0.0, 6.0))
    lost = _stop("living_room", "sofa", (9.0, 0.0, 9.0))
    return _record(
        scene=scene,
        legs=(_leg(bath), _leg(bed), _leg(lost, reached=False)),
        observations=(
            tour_observation(bath, sound_class="alarm", embedding=[1.0, 0.0, 0.0]),
            tour_observation(bed, sound_class="alarm", embedding=[0.0, 1.0, 0.0]),
        ),
    )


class TestEpisodicFromTour(unittest.TestCase):
    def test_only_reached_legs_become_rows(self):
        store = episodic_from_tour(_tour())
        self.assertEqual(len(store), 2)
        self.assertEqual({entry.room for entry in store.entries}, {"bathroom", "bedroom"})

    def test_an_abandoned_leg_leaves_no_row_and_no_point(self):
        """The point of an abandoned leg was never confirmed navigable-to from that route."""
        store = episodic_from_tour(_tour())
        self.assertEqual(store.points_for_room("sceneA", "living_room"), ())

    def test_the_scene_is_carried_from_the_record_not_from_the_stop(self):
        store = episodic_from_tour(_tour(scene="sceneB"))
        self.assertEqual(store.scenes, ("sceneB",))

    def test_a_tour_that_reached_nothing_is_an_empty_store_not_a_raise(self):
        record = _record(legs=(_leg(_stop("bathroom", "toilet"), reached=False),))
        self.assertEqual(len(episodic_from_tour(record)), 0)


class TestSemanticFromTour(unittest.TestCase):
    def test_every_observation_becomes_a_row_tagged_with_the_tour_scene(self):
        store = semantic_from_tour(_tour(scene="sceneB"))
        self.assertEqual(len(store), 2)
        self.assertEqual(store.donor_scenes, ("sceneB",))
        self.assertEqual(store.sound_classes, ("alarm",))

    def test_the_store_answers_the_room_it_heard_the_class_in(self):
        store = semantic_from_tour(_tour())
        room, cosine = store.predict_room(np.array([1.0, 0.05, 0.0]), k=1)
        self.assertEqual(room, "bathroom")
        self.assertAlmostEqual(cosine, 1.0, places=2)
        self.assertEqual(
            store.predict_room(np.array([0.05, 1.0, 0.0]), k=1)[0], "bedroom"
        )

    def test_the_width_is_the_encoders_and_is_not_hard_coded(self):
        store = semantic_from_tour(_tour())
        self.assertEqual(store.dim, 3)


class TestMalformedObservations(unittest.TestCase):
    """Each branch raises, and the message names the tour. Nothing is skipped."""

    def test_a_missing_key_raises_and_names_it(self):
        record = _record(observations=({"sound_class": "alarm", "room": "bathroom"},))
        with self.assertRaises(MemoryBuildError) as caught:
            semantic_from_tour(record)
        self.assertIn("category", str(caught.exception))
        self.assertIn("embedding", str(caught.exception))
        self.assertIn("sceneA", str(caught.exception))

    def test_an_embedding_of_the_wrong_type_raises_and_names_the_type(self):
        record = _record(
            observations=(
                {
                    "sound_class": "alarm",
                    "room": "bathroom",
                    "category": "toilet",
                    "embedding": "not a vector",
                },
            )
        )
        with self.assertRaises(MemoryBuildError) as caught:
            semantic_from_tour(record)
        self.assertIn("str", str(caught.exception))

    def test_an_empty_embedding_raises_at_the_constructor(self):
        with self.assertRaises(MemoryBuildError):
            tour_observation(_stop("bathroom", "toilet"), sound_class="alarm", embedding=[])

    def test_an_empty_embedding_that_arrives_anyway_still_raises(self):
        """`tour_observation` is the front door, not the only one — the payload is a Mapping."""
        record = _record(
            observations=(
                {
                    "sound_class": "alarm",
                    "room": "bathroom",
                    "category": "toilet",
                    "embedding": [],
                },
            )
        )
        with self.assertRaises(MemoryBuildError):
            semantic_from_tour(record)

    def test_two_encoders_in_one_tour_raise_at_the_store(self):
        record = _record(
            observations=(
                tour_observation(_stop("bathroom", "toilet"), sound_class="a", embedding=[1.0]),
                tour_observation(
                    _stop("bedroom", "bed"), sound_class="a", embedding=[1.0, 0.0]
                ),
            )
        )
        with self.assertRaises(ValueError):
            semantic_from_tour(record)


class TestCells(unittest.TestCase):
    """ADR-0018's 2x2, carved from one built store each so all four run the same episode."""

    def setUp(self):
        self.semantic = semantic_from_tour(_tour(scene="sceneA"))
        # A second scene in the episodic store, so `without_scene` has something to keep.
        other = _record(
            scene="sceneB",
            legs=(_leg(_stop("kitchen", "sink", (2.0, 0.0, 2.0))),),
        )
        self.episodic = EpisodicStore(
            entries=episodic_from_tour(_tour("sceneA")).entries
            + episodic_from_tour(other).entries
        )

    def _cell(self, condition):
        return stores_for_cell(
            self.semantic,
            self.episodic,
            condition,
            sound_class="alarm",
            scene="sceneA",
        )

    def test_heard_seen_keeps_both(self):
        semantic, episodic = self._cell(MemoryCondition.HEARD_SEEN)
        self.assertEqual(semantic.sound_classes, ("alarm",))
        self.assertIn("sceneA", episodic.scenes)

    def test_heard_unseen_keeps_the_class_and_drops_the_scene(self):
        semantic, episodic = self._cell(MemoryCondition.HEARD_UNSEEN)
        self.assertEqual(semantic.sound_classes, ("alarm",))
        self.assertNotIn("sceneA", episodic.scenes)
        # And it drops ONLY that scene: the co-primary contrast needs the store to still
        # be a store, not to be emptied.
        self.assertIn("sceneB", episodic.scenes)

    def test_not_heard_seen_drops_the_class_and_keeps_the_scene(self):
        semantic, episodic = self._cell(MemoryCondition.NOT_HEARD_SEEN)
        self.assertEqual(len(semantic), 0)
        self.assertIn("sceneA", episodic.scenes)

    def test_not_heard_unseen_is_the_baseline_and_holds_neither(self):
        semantic, episodic = self._cell(MemoryCondition.NOT_HEARD_UNSEEN)
        self.assertEqual(len(semantic), 0)
        self.assertNotIn("sceneA", episodic.scenes)

    def test_none_is_empty_and_is_not_the_full_store(self):
        """`NONE` means "ran outside the matrix". Handing it both stores would make that
        indistinguishable from `HEARD_SEEN` in every artefact."""
        semantic, episodic = self._cell(MemoryCondition.NONE)
        self.assertEqual(len(semantic), 0)
        self.assertEqual(len(episodic), 0)

    def test_every_condition_is_handled(self):
        for condition in MemoryCondition:
            self.assertIsNotNone(self._cell(condition))

    def test_the_inputs_are_untouched_by_any_carve(self):
        """The property the whole paired design rests on: four cells, one built store."""
        before = (len(self.semantic), len(self.episodic))
        for condition in MemoryCondition:
            self._cell(condition)
        self.assertEqual((len(self.semantic), len(self.episodic)), before)

    def test_a_class_the_store_never_heard_carves_to_the_same_store(self):
        """Total, not partial: an absent class is a no-op copy and never a raise."""
        semantic, _ = stores_for_cell(
            self.semantic,
            self.episodic,
            MemoryCondition.NOT_HEARD_SEEN,
            sound_class="a_class_no_tour_ever_heard",
            scene="sceneA",
        )
        self.assertEqual(len(semantic), len(self.semantic))


class TestTheJsonRoundTripIsExact(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = str(pathlib.Path(self.dir.name) / "stores" / "memory.json")

    def tearDown(self):
        self.dir.cleanup()

    def test_both_stores_and_the_provenance_survive(self):
        semantic = semantic_from_tour(_tour())
        episodic = episodic_from_tour(_tour())
        dump_stores(self.path, semantic, episodic, provenance={"commit": "abc123"})
        back_semantic, back_episodic, provenance = load_stores(self.path)

        self.assertEqual(len(back_semantic), len(semantic))
        self.assertEqual(len(back_episodic), len(episodic))
        self.assertEqual(provenance, {"commit": "abc123"})

    def test_the_embeddings_come_back_bit_for_bit(self):
        """Not `almost equal`. A k-NN vote turns on digits a tolerance would forgive."""
        rng = np.random.default_rng(0)
        awkward = np.asarray(rng.normal(size=17) * 1e-3, dtype=np.float32)
        store = SemanticStore(
            entries=semantic_from_tour(
                _record(
                    observations=(
                        tour_observation(
                            _stop("bathroom", "toilet"),
                            sound_class="alarm",
                            embedding=awkward.tolist(),
                        ),
                    )
                )
            ).entries
        )
        dump_stores(self.path, store, episodic_from_tour(_tour()))
        back, _episodic, _prov = load_stores(self.path)
        np.testing.assert_array_equal(back.entries[0].embedding, store.entries[0].embedding)

    def test_a_stored_point_comes_back_as_the_same_coordinates(self):
        episodic = episodic_from_tour(_tour())
        dump_stores(self.path, SemanticStore(), episodic)
        _semantic, back, _prov = load_stores(self.path)
        self.assertEqual(
            [entry.point.as_tuple() for entry in back.entries],
            [entry.point.as_tuple() for entry in episodic.entries],
        )

    def test_a_reloaded_store_answers_the_same_query_the_same_way(self):
        """The property that matters at the far end: the store learned one thing and the
        run that consults it must get that thing."""
        semantic = semantic_from_tour(_tour())
        dump_stores(self.path, semantic, episodic_from_tour(_tour()))
        back, _episodic, _prov = load_stores(self.path)
        query = np.asarray([0.9, 0.1, 0.0], dtype=np.float32)
        self.assertEqual(back.predict_room(query, k=2), semantic.predict_room(query, k=2))

    def test_the_parent_directory_is_created(self):
        dump_stores(self.path, SemanticStore(), episodic_from_tour(_tour()))
        self.assertTrue(pathlib.Path(self.path).is_file())

    def test_no_partial_file_is_left_behind(self):
        dump_stores(self.path, SemanticStore(), episodic_from_tour(_tour()))
        siblings = [p.name for p in pathlib.Path(self.path).parent.iterdir()]
        self.assertEqual(siblings, ["memory.json"])

    def test_an_unknown_format_version_is_refused_rather_than_read(self):
        dump_stores(self.path, semantic_from_tour(_tour()), episodic_from_tour(_tour()))
        payload = json.loads(pathlib.Path(self.path).read_text())
        payload["format_version"] = STORE_FORMAT_VERSION + 1
        pathlib.Path(self.path).write_text(json.dumps(payload))
        with self.assertRaises(MemoryBuildError) as caught:
            load_stores(self.path)
        self.assertIn("format version", str(caught.exception))

    def test_a_file_with_no_version_at_all_is_refused(self):
        pathlib.Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(self.path).write_text(json.dumps({"semantic": [], "episodic": []}))
        with self.assertRaises(MemoryBuildError):
            load_stores(self.path)


if __name__ == "__main__":
    unittest.main()
