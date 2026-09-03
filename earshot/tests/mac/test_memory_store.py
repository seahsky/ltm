"""The memory stack: both stores, both filters, and the fence around the placement table.

ADR-0014's rule for every behaviour switch applies to `without_class` and `without_scene`
as much as to any config arm: each gets the healthy path (a query the removed row would
have answered) AND the forced-failure path (the same query after the row is gone,
asserted on the returned VALUE, never on `len()` alone — a store that silently returned
the wrong room would still have the right length).

The fence tests are the second line of defense on top of `test_audio_vocabulary.py`'s
`TestAnchorFence`, scoped to just the two files this module owns. Two mechanisms, same
shape `test_analyst_only.py` already uses for `sourceIsVisible`: a structural import
check (this file's AST, not a running interpreter's `sys.modules` — the whole discovered
suite shares one process, and another test module importing `earshot.audio.vocabulary`
first would make a `sys.modules` check pass by accident of test ORDER rather than by the
property actually holding) and an AST name scan that excludes docstrings, because a
module that must not READ the placement table is still allowed to NAME it in prose
explaining why not — `store.py`'s own docstring does exactly this, the same shape
`test_analyst_only.py`'s docstring names `sourceIsVisible`.
"""

import unittest

import numpy as np

import _tree
from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.store import (
    EpisodicEntry,
    EpisodicStore,
    MemoryCondition,
    SemanticEntry,
    SemanticStore,
    without_class,
    without_scene,
)
from earshot.types import Xyz


def _entry(sound_class, room, embedding, donor_scene="scene_a", category=None):
    # `category` defaults to one derived from the room so the existing room tests read
    # unchanged, and the category tests below pass it explicitly. It is a REQUIRED field
    # on the entry itself -- the default lives here, in the test factory, not on the
    # dataclass, where it would let an unusable row through.
    return SemanticEntry(
        sound_class=sound_class,
        room=room,
        category=("{}_object".format(room) if category is None else category),
        embedding=np.asarray(embedding, dtype=np.float32),
        donor_scene=donor_scene,
    )


def _episodic(scene, room, category, point=(0.0, 0.0, 0.0)):
    return EpisodicEntry(scene=scene, room=room, category=category, point=Xyz(*point))


class TestSemanticEntry(unittest.TestCase):
    def test_the_embedding_is_recopied_and_frozen(self):
        """Purity in both directions: the caller's array and the stored one diverge."""
        source = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        entry = _entry("alarm", "kitchen", source)

        source[0] = 999.0  # mutate the caller's array after construction
        self.assertEqual(
            entry.embedding.tolist(), [1.0, 2.0, 3.0],
            "the stored embedding moved when the caller's array did — not a copy",
        )
        with self.assertRaises(ValueError):
            entry.embedding[0] = 1.0  # the copy itself must be read-only

    def test_the_embedding_is_coerced_to_1d_float32(self):
        entry = _entry("alarm", "kitchen", [[1, 2, 3]])  # a (1, 3) row, coerced to (3,)
        self.assertEqual(entry.embedding.shape, (3,))
        self.assertEqual(entry.embedding.dtype, np.float32)

    def test_an_empty_embedding_raises_at_construction(self):
        with self.assertRaises(ValueError):
            _entry("alarm", "kitchen", [])

    def test_entries_carry_identity_equality_not_array_equality(self):
        """`eq=False`: two entries built from equal arrays must not compare `==` and
        must not raise trying. `agent/occupancy.py`'s `OccupancyGrid` sets the same flag
        for the identical reason."""
        a = _entry("alarm", "kitchen", [1.0, 0.0])
        b = _entry("alarm", "kitchen", [1.0, 0.0])
        self.assertNotEqual(a, b)  # identity, not value
        self.assertEqual(a, a)


class TestSemanticStorePredictRoom(unittest.TestCase):
    def setUp(self):
        # Two kitchen observations near [1, 0, 0], one bathroom observation near
        # [0, 1, 0] — far enough apart that a correct k-NN vote is unambiguous.
        self.store = SemanticStore(
            entries=(
                _entry("alarm", "kitchen", [1.0, 0.0, 0.0], donor_scene="donor_1"),
                _entry("alarm", "kitchen", [0.9, 0.1, 0.0], donor_scene="donor_1"),
                _entry("alarm", "bathroom", [0.0, 1.0, 0.0], donor_scene="donor_2"),
            )
        )

    def test_dim_and_the_derived_properties(self):
        self.assertEqual(self.store.dim, 3)
        self.assertEqual(self.store.sound_classes, ("alarm",))
        self.assertEqual(self.store.donor_scenes, ("donor_1", "donor_2"))
        self.assertEqual(len(self.store), 3)

    def test_empty_store_dim_is_none_not_zero(self):
        """NOT_RUN is red: an empty store has no width, and 0 would read as a real one."""
        self.assertIsNone(SemanticStore().dim)

    def test_the_healthy_arm_predicts_the_planted_room(self):
        room, score = self.store.predict_room(np.array([1.0, 0.0, 0.0]), k=2)
        print(
            "\n  [memory] predict_room(k=2) on the planted store -> {!r} at mean cosine "
            "{:.6f}".format(room, score),
            flush=True,
        )
        self.assertEqual(room, "kitchen")
        # cos([1,0,0], [1,0,0]) = 1.0 exactly; cos([1,0,0], [0.9,0.1,0]) = 0.9/|0.9,0.1,0|
        expected_second = 0.9 / float(np.linalg.norm([0.9, 0.1, 0.0]))
        self.assertAlmostEqual(score, (1.0 + expected_second) / 2.0, places=6)

    def test_k_larger_than_the_store_uses_every_entry(self):
        """"or all entries if len(self) < k" — must not raise or silently truncate to
        nothing."""
        room, _score = self.store.predict_room(np.array([1.0, 0.0, 0.0]), k=99)
        self.assertEqual(room, "kitchen")

    def test_ties_are_broken_by_room_name_ascending(self):
        """Two rooms at an EXACT tie: alphabetical, not insertion order.

        Same-direction vectors give an identical cosine regardless of magnitude, so
        `[2, 0]` and `[1, 0]` both score 1.0 against a `[1, 0]` query — a tie constructed
        on purpose rather than hoped for.
        """
        tied = SemanticStore(
            entries=(
                _entry("x", "zeta", [1.0, 0.0], donor_scene="s"),
                _entry("x", "alpha", [2.0, 0.0], donor_scene="s"),
            )
        )
        room, score = tied.predict_room(np.array([1.0, 0.0]), k=2)
        print(
            "\n  [memory] tie at cosine {:.6f}: alpha vs zeta -> {!r}".format(score, room),
            flush=True,
        )
        self.assertEqual(room, "alpha")
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_empty_store_returns_none_never_a_zero_score(self):
        self.assertIsNone(SemanticStore().predict_room(np.array([1.0, 0.0, 0.0]), k=1))

    def test_zero_norm_query_returns_none(self):
        self.assertIsNone(self.store.predict_room(np.zeros(3), k=1))

    def test_store_of_zero_norm_entries_returns_none(self):
        """Every candidate skipped, not scored as a cosine of 0.0 against a real query."""
        degenerate = SemanticStore(entries=(_entry("x", "kitchen", [0.0, 0.0, 0.0]),))
        self.assertIsNone(degenerate.predict_room(np.array([1.0, 0.0, 0.0]), k=1))

    def test_a_mix_of_zero_and_real_norm_entries_scores_only_the_real_one(self):
        mixed = SemanticStore(
            entries=(
                _entry("x", "silent", [0.0, 0.0, 0.0]),
                _entry("x", "loud", [1.0, 0.0, 0.0]),
            )
        )
        room, score = mixed.predict_room(np.array([1.0, 0.0, 0.0]), k=2)
        self.assertEqual(room, "loud")
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_k_below_one_raises(self):
        with self.assertRaises(ValueError):
            self.store.predict_room(np.array([1.0, 0.0, 0.0]), k=0)

    def test_wrong_dimension_query_raises_not_a_low_score(self):
        with self.assertRaises(ValueError):
            self.store.predict_room(np.array([1.0, 0.0]), k=1)  # store is dim 3

    def test_mismatched_entry_dimensions_raise_at_construction(self):
        with self.assertRaises(ValueError):
            SemanticStore(
                entries=(
                    _entry("x", "a", [1.0, 0.0, 0.0]),
                    _entry("x", "b", [1.0, 0.0]),
                )
            )

    def test_predict_room_never_mutates_the_store_or_the_query(self):
        before = self.store.entries
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        query_before = query.copy()
        self.store.predict_room(query, k=2)
        self.assertIs(self.store.entries, before)
        self.assertTrue(np.array_equal(query, query_before))


class TestSemanticStorePredictCategory(unittest.TestCase):
    """The query the unseen-and-heard cell runs, and the reason it exists.

    In an unseen scene `without_scene` has emptied the episodic store, so `points_for_room`
    returns `()` and a room name points at no coordinate. The category the class was heard
    AT transfers, because the scene under test has its own instances of it. This class holds
    that `predict_category` answers the category rather than the room, and that it inherits
    every contract `predict_room` already had rather than re-deriving them loosely.
    """

    def setUp(self):
        # The SAME rows answer two different questions: the two near [1,0,0] were heard at
        # a stove in a kitchen, the far one at a toilet in a bathroom. A vote that returned
        # the room here would be the bug this test exists to catch.
        self.store = SemanticStore(
            entries=(
                _entry("alarm", "kitchen", [1.0, 0.0, 0.0], category="stove"),
                _entry("alarm", "kitchen", [0.9, 0.1, 0.0], category="stove"),
                _entry("alarm", "bathroom", [0.0, 1.0, 0.0], category="toilet"),
            )
        )

    def test_the_healthy_arm_predicts_the_planted_category(self):
        category, score = self.store.predict_category(np.array([1.0, 0.0, 0.0]), k=2)
        print(
            "\n  [memory] predict_category(k=2) -> {!r} at mean cosine {:.6f}".format(
                category, score
            ),
            flush=True,
        )
        self.assertEqual(category, "stove")
        expected_second = 0.9 / float(np.linalg.norm([0.9, 0.1, 0.0]))
        self.assertAlmostEqual(score, (1.0 + expected_second) / 2.0, places=6)

    def test_the_forced_arm_predicts_the_other_category(self):
        # ADR-0014: the healthy path passing is half a detector. Query the far cluster and
        # the answer must move, or the vote is returning a constant.
        category, _ = self.store.predict_category(np.array([0.0, 1.0, 0.0]), k=1)
        self.assertEqual(category, "toilet")

    def test_room_and_category_disagree_on_the_same_query(self):
        """The whole point: one store, two answers, and they are not the same string."""
        query = np.array([1.0, 0.0, 0.0])
        self.assertEqual(self.store.predict_room(query, k=2)[0], "kitchen")
        self.assertEqual(self.store.predict_category(query, k=2)[0], "stove")

    def test_it_inherits_every_contract_predict_room_has(self):
        query = np.array([1.0, 0.0, 0.0])
        with self.assertRaises(ValueError) as caught:
            self.store.predict_category(query, k=0)
        # The message names the query the caller actually made, not the shared helper.
        self.assertIn("predict_category", str(caught.exception))
        with self.assertRaises(ValueError):
            self.store.predict_category(np.array([1.0, 0.0]), k=1)
        self.assertIsNone(SemanticStore().predict_category(query, k=1))
        self.assertIsNone(self.store.predict_category(np.zeros(3), k=1))

    def test_ties_are_broken_by_category_name_ascending(self):
        store = SemanticStore(
            entries=(
                _entry("alarm", "kitchen", [1.0, 0.0, 0.0], category="stove"),
                _entry("alarm", "kitchen", [1.0, 0.0, 0.0], category="counter"),
            )
        )
        category, score = store.predict_category(np.array([1.0, 0.0, 0.0]), k=2)
        self.assertEqual(category, "counter")
        self.assertAlmostEqual(score, 1.0, places=6)

    def test_it_never_mutates_the_store_or_the_query(self):
        query = np.array([1.0, 0.0, 0.0])
        before = [e.category for e in self.store.entries]
        self.store.predict_category(query, k=3)
        self.assertEqual(before, [e.category for e in self.store.entries])
        np.testing.assert_array_equal(query, np.array([1.0, 0.0, 0.0]))

    def test_the_category_survives_the_unseen_filter(self):
        """`without_scene` is episodic-only, so the semantic answer is untouched by it."""
        episodic = EpisodicStore(entries=(_episodic("scene_x", "kitchen", "stove"),))
        self.assertEqual(without_scene(episodic, "scene_x").entries, ())
        # Same store, same query, same answer: this is the mechanism the cell rests on.
        self.assertEqual(
            self.store.predict_category(np.array([1.0, 0.0, 0.0]), k=2)[0], "stove"
        )


class TestWithoutClass(unittest.TestCase):
    def setUp(self):
        # `alarm_kitchen` is the exact query direction; `glass_bath` is close but not
        # exact, and a different room -- so removing the `alarm` class does not merely
        # blank the verdict, it hands the query to a genuinely different second-best.
        self.alarm_kitchen = _entry("alarm", "kitchen", [1.0, 0.0], donor_scene="d1")
        self.glass_bath = _entry("glass_break", "bathroom", [0.9, 0.1], donor_scene="d2")
        self.store = SemanticStore(entries=(self.alarm_kitchen, self.glass_bath))

    def test_removes_every_entry_of_the_named_class(self):
        pruned = without_class(self.store, "alarm")
        self.assertEqual(pruned.entries, (self.glass_bath,))

    def test_leaves_every_other_entry_byte_identical(self):
        """`assertIs`, not `==`: the SAME object survives, not an equal copy of it."""
        pruned = without_class(self.store, "alarm")
        self.assertIs(pruned.entries[0], self.glass_bath)

    def test_the_healthy_and_forced_failure_arms_change_the_verdict(self):
        """ADR-0014, in `predict_room` terms: the class removal must change what a
        subsequent query answers, not merely what `len()` reports -- so this asserts
        the returned ROOM, not the store's size."""
        query = np.array([1.0, 0.0])
        before = self.store.predict_room(query, k=1)
        after = without_class(self.store, "alarm").predict_room(query, k=1)
        print(
            "\n  [memory] without_class('alarm'): predict_room before={} after={}".format(
                before, after
            ),
            flush=True,
        )
        self.assertEqual(before[0], "kitchen")
        self.assertEqual(after[0], "bathroom")
        self.assertNotEqual(before[0], after[0])

    def test_a_class_absent_from_the_store_is_a_no_op(self):
        same = without_class(self.store, "not_a_class")
        self.assertEqual(same.entries, self.store.entries)

    def test_the_original_store_is_unchanged(self):
        before = self.store.entries
        without_class(self.store, "alarm")
        self.assertIs(self.store.entries, before)
        self.assertEqual(len(self.store), 2)


class TestWithoutScene(unittest.TestCase):
    def setUp(self):
        self.kitchen_a = _episodic("scene_a", "kitchen", "chair", (1.0, 0.0, 1.0))
        self.kitchen_b = _episodic("scene_b", "kitchen", "chair", (2.0, 0.0, 2.0))
        self.store = EpisodicStore(entries=(self.kitchen_a, self.kitchen_b))

    def test_points_for_room_finds_the_planted_point(self):
        self.assertEqual(
            self.store.points_for_room("scene_a", "kitchen"), (self.kitchen_a.point,)
        )

    def test_points_for_room_is_an_empty_tuple_not_none_when_nothing_matches(self):
        result = self.store.points_for_room("scene_a", "bathroom")
        self.assertEqual(result, ())
        self.assertIsNotNone(result)

    def test_healthy_and_forced_failure_arms_of_without_scene(self):
        dropped = without_scene(self.store, "scene_a")
        print(
            "\n  [memory] without_scene('scene_a'): scene_a room count {} -> {}, "
            "scene_b room count {} -> {}".format(
                len(self.store.points_for_room("scene_a", "kitchen")),
                len(dropped.points_for_room("scene_a", "kitchen")),
                len(self.store.points_for_room("scene_b", "kitchen")),
                len(dropped.points_for_room("scene_b", "kitchen")),
            ),
            flush=True,
        )
        self.assertEqual(dropped.points_for_room("scene_a", "kitchen"), ())
        self.assertEqual(
            dropped.points_for_room("scene_b", "kitchen"), (self.kitchen_b.point,)
        )


class TestPointsForCategory(unittest.TestCase):
    """`resolve_prior` votes at the category grain, not the room's -- this is the
    accessor a point resolver actually calls. Same fixture as `TestWithoutScene`,
    because the two accessors answer the same question at two different grains."""

    def setUp(self):
        self.kitchen_a = _episodic("scene_a", "kitchen", "chair", (1.0, 0.0, 1.0))
        self.kitchen_b = _episodic("scene_b", "kitchen", "chair", (2.0, 0.0, 2.0))
        self.store = EpisodicStore(entries=(self.kitchen_a, self.kitchen_b))

    def test_it_finds_the_planted_point(self):
        self.assertEqual(
            self.store.points_for_category("scene_a", "chair"), (self.kitchen_a.point,)
        )

    def test_it_is_an_empty_tuple_not_none_when_nothing_matches(self):
        result = self.store.points_for_category("scene_a", "toilet")
        self.assertEqual(result, ())
        self.assertIsNotNone(result)

    def test_it_does_not_cross_scenes(self):
        self.assertEqual(
            self.store.points_for_category("scene_a", "chair"), (self.kitchen_a.point,)
        )
        self.assertEqual(
            self.store.points_for_category("scene_b", "chair"), (self.kitchen_b.point,)
        )

    def test_without_scene_empties_it_the_same_way_it_empties_points_for_room(self):
        dropped = without_scene(self.store, "scene_a")
        self.assertEqual(dropped.points_for_category("scene_a", "chair"), ())
        self.assertEqual(
            dropped.points_for_category("scene_b", "chair"), (self.kitchen_b.point,)
        )

    def test_a_scene_absent_from_the_store_is_a_no_op(self):
        same = without_scene(self.store, "no_such_scene")
        self.assertEqual(same.entries, self.store.entries)

    def test_the_original_store_is_unchanged(self):
        before = self.store.entries
        without_scene(self.store, "scene_a")
        self.assertIs(self.store.entries, before)
        self.assertEqual(self.store.scenes, ("scene_a", "scene_b"))


class TestMemoryCondition(unittest.TestCase):
    def test_the_five_cells_have_the_spec_values(self):
        self.assertEqual(MemoryCondition.NONE.value, "none")
        self.assertEqual(MemoryCondition.HEARD_SEEN.value, "heard_seen")
        self.assertEqual(MemoryCondition.HEARD_UNSEEN.value, "heard_unseen")
        self.assertEqual(MemoryCondition.NOT_HEARD_SEEN.value, "not_heard_seen")
        self.assertEqual(MemoryCondition.NOT_HEARD_UNSEEN.value, "not_heard_unseen")


class TestTheStoreCannotSeeGroundTruth(unittest.TestCase):
    """The fence, scoped to just this package's two files — belt to
    `test_audio_vocabulary.py`'s braces, not a replacement for it."""

    def test_memory_does_not_import_the_vocabulary_module(self):
        """Structural, not a `sys.modules` probe.

        `unittest discover` loads every test module in the suite into one process
        before any test body runs, and several of them import
        `earshot.audio.vocabulary` directly — so by the time this test runs,
        `'earshot.audio.vocabulary' in sys.modules` is already True regardless of
        whether `earshot.memory.store` itself ever imports it. Parsing the source is
        the only check that answers the actual question.
        """
        for name in ("__init__", "store"):
            path = _tree.PACKAGE_ROOT / "memory" / (name + ".py")
            tree = _tree.parse(path)
            self.assertFalse(
                _tree.imports_module(tree, "earshot.audio.vocabulary"),
                "{} imports earshot.audio.vocabulary".format(path),
            )
            self.assertFalse(
                _tree.imports_module(tree, "earshot.audio"),
                "{} imports earshot.audio".format(path),
            )

    def test_the_layer_graph_does_not_grant_memory_an_audio_edge(self):
        allowed = _tree.LAYER_IMPORTS["memory"]
        self.assertFalse(
            _tree.edge_allowed("audio.vocabulary", allowed),
            "LAYER_IMPORTS['memory'] = {!r} would let memory/ reach the placement "
            "table".format(allowed),
        )

    def test_no_memory_source_names_the_placement_table_outside_a_docstring(self):
        """The AST-level half: a bare string or attribute reach, docstrings excluded.

        Docstrings are excluded on purpose, not by oversight — `store.py`'s own module
        docstring NAMES `ROOM_OF_ANCHOR` and `anchor_object` to explain why the module
        must not read them, the same citation shape `test_analyst_only.py`'s docstring
        uses for `sourceIsVisible`. A whole-file substring scan would flag that
        citation as a violation, which is the wrong failure: naming a forbidden symbol
        in prose is not reading it.
        """
        forbidden = ("ROOM_OF_ANCHOR", "anchor_object", "room_of")
        offenders = []
        for name in ("__init__", "store"):
            path = _tree.PACKAGE_ROOT / "memory" / (name + ".py")
            tree = _tree.parse(path)
            for lineno, value in _tree.code_string_constants(tree):
                if value in forbidden:
                    offenders.append("{}:{} {!r}".format(path.name, lineno, value))
            for lineno, attr in _tree.attribute_names(tree):
                if attr in forbidden:
                    offenders.append("{}:{} .{}".format(path.name, lineno, attr))
        self.assertEqual(offenders, [], "\n".join([""] + offenders))


if __name__ == "__main__":
    unittest.main(verbosity=2)
