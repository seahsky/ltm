"""DREAM §III.E's three-level long-term memory (eq. 14-18), and what `G` had to decide.

PR 4 of 6 building DREAM as `ICRA2027_Memory` specifies it.

Eq. 15 and 17 list four fields each and define two of the eight. The tests that matter are
the ones a plausible WRONG implementation would fail:

  * `h^traj` built from coordinates -- the obvious implementation, which §III.A's own
    sentence rules out. `TestTrajectoryDescriptor` holds the arm: the same walk shape in
    two different places must give the same descriptor, and two different shapes in the
    same place must not (`test_the_descriptor_is_the_shape_and_not_the_place`);
  * `G` abstracting from failures as well as successes, which the paper explicitly
    forbids at this stage and which a `C_j`-shaped reading would have allowed;
  * `M^P` coming out in insertion order, which makes two runs over the same experiences
    produce two different memories and a retrieval irreproducible;
  * `novelty_vectors` including `M^K`, whose keys are half the width -- eq. 12 reads as
    one max over all of `M^L` and cannot be.

No torch, no simulator. `memory/` is still fenced off from `audio` and `agent`; the only
new edge is `memory` to itself, for the `SemanticStore` that IS `M^K`.
"""

import math
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.consolidate import novelty
from earshot.memory.longterm import (
    TRAJECTORY_COMPONENTS,
    ExperienceEntry,
    ExperienceStore,
    LongTermMemory,
    Outcome,
    Pattern,
    PatternStore,
    Strategy,
    abstract,
    trajectory_descriptor,
)
from earshot.memory.store import SemanticEntry, SemanticStore
from earshot.types import Xyz


def _points(*pairs):
    return [Xyz(float(x), 0.0, float(z)) for x, z in pairs]


def _experience(
    context=(1.0, 0.0),
    sound="toilet_flush",
    obj="toilet",
    room=None,
    reached=True,
    gap=0.4,
    walk=None,
):
    return ExperienceEntry(
        context=np.asarray(context, dtype=np.float32),
        trajectory=trajectory_descriptor(walk or _points((0, 0), (1, 0), (2, 0))),
        target_concept=obj,
        outcome=Outcome(reached=reached, final_gap_m=gap),
        sound_concept=sound,
        room_concept=room,
    )


class TestOutcome(unittest.TestCase):
    """`y_i` (eq. 15). Two fields, because the tree measures two things."""

    def test_it_keeps_both_the_predicate_and_the_distance(self):
        outcome = Outcome(reached=False, final_gap_m=7.25)
        self.assertFalse(outcome.reached)
        self.assertAlmostEqual(outcome.final_gap_m, 7.25)

    def test_a_negative_gap_raises(self):
        """A negative distance to the source is a sign error upstream, not an outcome."""
        with self.assertRaises(ValueError) as caught:
            Outcome(reached=True, final_gap_m=-0.1)
        self.assertIn("must be >= 0", str(caught.exception))


class TestTrajectoryDescriptor(unittest.TestCase):
    """`h^traj_i` (eq. 15). The paper does not define it and §III.A rules out the obvious
    implementation in so many words."""

    def test_it_has_the_named_components_in_order(self):
        descriptor = trajectory_descriptor(_points((0, 0), (3, 0)))
        self.assertEqual(descriptor.shape, (len(TRAJECTORY_COMPONENTS),))
        self.assertAlmostEqual(float(descriptor[0]), 3.0, places=5)
        self.assertAlmostEqual(float(descriptor[1]), 3.0, places=5)
        self.assertAlmostEqual(float(descriptor[2]), 1.0, places=5)
        self.assertAlmostEqual(float(descriptor[3]), 0.0, places=5)

    def test_the_descriptor_is_the_shape_and_not_the_place(self):
        """**THE ARM THAT KEEPS `M^E` TRANSFERABLE.** §III.A: `M^L` must capture
        transferable knowledge "rather than memorizing environment-specific trajectories
        or target locations". The same walk 100 m away must be the same experience, and a
        descriptor built from coordinates fails this on the first assertion."""
        here = trajectory_descriptor(_points((0, 0), (1, 0), (1, 1)))
        far = trajectory_descriptor(_points((100, 100), (101, 100), (101, 101)))
        np.testing.assert_allclose(here, far, atol=1e-5)
        different_shape = trajectory_descriptor(_points((0, 0), (1, 0), (2, 0)))
        self.assertFalse(
            np.allclose(here, different_shape, atol=1e-5),
            "an L-shaped walk and a straight one produced the same descriptor, so h^traj "
            "encodes nothing about the trajectory",
        )

    def test_straightness_separates_walking_to_it_from_wandering_into_it(self):
        """The component no single distance can supply: two walks ending the same
        distance away, one direct and one a loop."""
        direct = trajectory_descriptor(_points((0, 0), (1, 0), (2, 0)))
        looping = trajectory_descriptor(_points((0, 0), (1, 3), (2, 0)))
        self.assertAlmostEqual(float(direct[1]), float(looping[1]), places=5)
        self.assertGreater(float(direct[2]), float(looping[2]))

    def test_a_walk_that_returned_to_its_start_is_not_straight(self):
        descriptor = trajectory_descriptor(_points((0, 0), (2, 0), (0, 0)))
        self.assertAlmostEqual(float(descriptor[1]), 0.0, places=5)
        self.assertAlmostEqual(float(descriptor[2]), 0.0, places=5)

    def test_the_turn_is_unsigned_so_it_asserts_no_frame(self):
        """`types.py` holds no bearing helper because the lateral sign silently inverted
        between world and agent frame (ticket 09). A SIGNED turn here would assert the
        same convention this tree refuses to assert outside `audio/lateral.py`. Mirror
        images must therefore score identically."""
        left = trajectory_descriptor(_points((0, 0), (1, 0), (1, 1)))
        right = trajectory_descriptor(_points((0, 0), (1, 0), (1, -1)))
        np.testing.assert_allclose(left, right, atol=1e-5)
        self.assertAlmostEqual(float(left[3]), math.pi / 2.0, places=4)

    def test_height_is_ignored(self):
        """A descriptor that charged a walk for a staircase would make the same route two
        different experiences on two floors."""
        flat = trajectory_descriptor(_points((0, 0), (1, 0), (2, 0)))
        climbing = trajectory_descriptor([
            Xyz(0.0, 0.0, 0.0), Xyz(1.0, 4.0, 0.0), Xyz(2.0, 8.0, 0.0)
        ])
        np.testing.assert_allclose(flat, climbing, atol=1e-5)

    def test_a_single_position_is_a_zero_length_walk_not_a_crash(self):
        descriptor = trajectory_descriptor(_points((3, 4)))
        self.assertAlmostEqual(float(descriptor[0]), 0.0, places=5)
        self.assertAlmostEqual(float(descriptor[2]), 0.0, places=5)

    def test_no_positions_raises(self):
        """A zero descriptor would read as a perfectly straight walk of zero length,
        which is a claim about a walk that did not happen."""
        with self.assertRaises(ValueError) as caught:
            trajectory_descriptor([])
        self.assertIn("no positions", str(caught.exception))


class TestExperienceEntry(unittest.TestCase):
    """`m^E_i` (eq. 15), plus the two fields eq. 17 needs and eq. 15 omits."""

    def test_it_copies_its_arrays(self):
        context = np.array([1.0, 0.0], dtype=np.float32)
        entry = _experience(context=context)
        context[0] = 99.0
        self.assertAlmostEqual(float(entry.context[0]), 1.0)

    def test_the_stored_arrays_are_read_only(self):
        entry = _experience()
        with self.assertRaises(ValueError):
            entry.context[0] = 5.0
        with self.assertRaises(ValueError):
            entry.trajectory[0] = 5.0

    def test_an_empty_context_raises(self):
        with self.assertRaises(ValueError) as caught:
            _experience(context=())
        self.assertIn("empty h^av", str(caught.exception))

    def test_a_non_finite_context_raises(self):
        """A NaN key is stored and can never be retrieved: it loses every comparison in
        eq. 12 and eq. 20 silently."""
        with self.assertRaises(ValueError) as caught:
            _experience(context=(1.0, float("nan")))
        self.assertIn("non-finite", str(caught.exception))

    def test_a_trajectory_of_the_wrong_width_raises(self):
        with self.assertRaises(ValueError) as caught:
            ExperienceEntry(
                context=np.asarray([1.0, 0.0], dtype=np.float32),
                trajectory=np.asarray([1.0, 2.0], dtype=np.float32),
                target_concept="toilet",
                outcome=Outcome(reached=True, final_gap_m=0.2),
                sound_concept="toilet_flush",
                room_concept=None,
            )
        self.assertIn("TRAJECTORY_COMPONENTS", str(caught.exception))

    def test_an_empty_concept_raises_on_either_field(self):
        """An unnamed concept groups with every other unnamed concept in `G` and produces
        a pattern about nothing."""
        for kwargs in ({"obj": ""}, {"sound": "   "}):
            with self.assertRaises(ValueError) as caught:
                _experience(**kwargs)
            self.assertIn("empty", str(caught.exception))

    def test_an_absent_room_is_allowed_and_is_the_ordinary_case(self):
        """`NullRoomLabeler` always abstains, so `None` is what runs actually produce.
        Making it visible in the row is what keeps the scene axis of `G` visibly flat
        rather than quietly flat."""
        self.assertIsNone(_experience().room_concept)


class TestExperienceStore(unittest.TestCase):
    """`M^E` (eq. 14)."""

    def test_extend_returns_a_new_store(self):
        """The audit holds the memory as it stood at the start of each episode, which is
        the only state a retrieval can be reproduced against afterwards."""
        before = ExperienceStore().extend([_experience()])
        after = before.extend([_experience()])
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 2)
        self.assertIsNot(before, after)

    def test_a_store_mixing_key_widths_raises(self):
        with self.assertRaises(ValueError) as caught:
            ExperienceStore(entries=(
                _experience(context=(1.0, 0.0)), _experience(context=(1.0, 0.0, 0.0))
            ))
        self.assertIn("mismatched width", str(caught.exception))

    def test_dim_is_none_on_an_empty_store_and_never_zero(self):
        self.assertIsNone(ExperienceStore().dim)
        self.assertEqual(ExperienceStore().extend([_experience()]).dim, 2)

    def test_keys_are_the_contexts_in_order(self):
        store = ExperienceStore().extend([
            _experience(context=(1.0, 0.0)), _experience(context=(0.0, 1.0))
        ])
        np.testing.assert_allclose(np.stack(store.keys), [[1.0, 0.0], [0.0, 1.0]])


class TestG(unittest.TestCase):
    """`M^P = G(M^E)` (eq. 16). The paper names `G` and does not define it."""

    def test_experiences_sharing_a_concept_triple_become_one_pattern(self):
        store = ExperienceStore().extend([_experience(), _experience(), _experience()])
        patterns = abstract(store, min_support=2)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns.patterns[0].concepts, ("toilet_flush", "toilet", None))
        self.assertEqual(patterns.patterns[0].strategy.support, 3)

    def test_different_triples_stay_different_patterns(self):
        store = ExperienceStore().extend([
            _experience(sound="toilet_flush", obj="toilet"),
            _experience(sound="toilet_flush", obj="toilet"),
            _experience(sound="snoring", obj="bed"),
            _experience(sound="snoring", obj="bed"),
        ])
        self.assertEqual(len(abstract(store, min_support=2)), 2)

    def test_the_sound_concept_alone_separates_two_patterns(self):
        """THE ARM THAT ISOLATES THE ACOUSTIC AXIS. The test above varies the sound and
        the object TOGETHER, so a `G` keyed on the object alone passes it -- the
        forced-failure run is what showed that. Two sounds that anchor at the SAME object
        is the real case (`anchor_yield` puts `chair`, `sofa` and `tv_monitor` all in the
        living room), and collapsing them would build one pattern claiming a single fused
        signature means "go to the sofa" across two acoustically different sounds."""
        store = ExperienceStore().extend([
            _experience(sound="alarm", obj="sofa", context=(1.0, 0.0)),
            _experience(sound="alarm", obj="sofa", context=(1.0, 0.0)),
            _experience(sound="glass_break", obj="sofa", context=(0.0, 1.0)),
            _experience(sound="glass_break", obj="sofa", context=(0.0, 1.0)),
        ])
        patterns = abstract(store, min_support=2)
        self.assertEqual(
            [p.sound_concept for p in patterns.patterns], ["alarm", "glass_break"],
            "two sounds anchoring at the same object collapsed into one pattern, so G "
            "is not keyed on the acoustic concept eq. 17 requires",
        )

    def test_failures_are_not_abstracted(self):
        """**THE ARM THE PAPER IS EXPLICIT ABOUT.** §III.E.2: "abstracts recurring
        regularities from related SUCCESSFUL experiences". `consolidate.contribution`
        REFUSED a success gate on the same word, because there the gate would have
        discarded two thirds of all experience before novelty saw it. Here the gate is at
        abstraction rather than retention, so `M^E` still keeps the failures."""
        store = ExperienceStore().extend([
            _experience(reached=True), _experience(reached=False), _experience(reached=False)
        ])
        self.assertEqual(len(store), 3, "M^E must keep the failures")
        self.assertEqual(len(abstract(store, min_support=2)), 0)
        self.assertEqual(abstract(store, min_support=1).patterns[0].strategy.support, 1)

    def test_a_group_below_min_support_is_not_a_regularity(self):
        store = ExperienceStore().extend([
            _experience(sound="toilet_flush"), _experience(sound="toilet_flush"),
            _experience(sound="snoring", obj="bed"),
        ])
        patterns = abstract(store, min_support=2)
        self.assertEqual([p.sound_concept for p in patterns.patterns], ["toilet_flush"])

    def test_min_support_has_no_default(self):
        """The number of times something has to recur before it counts is the whole
        content of the phrase "recurring regularity"."""
        with self.assertRaises(TypeError):
            abstract(ExperienceStore())

    def test_a_min_support_below_one_raises(self):
        with self.assertRaises(ValueError) as caught:
            abstract(ExperienceStore(), min_support=0)
        self.assertIn("at least one experience", str(caught.exception))

    def test_the_output_order_is_deterministic_and_not_insertion_order(self):
        """Insertion order would make two runs over the same experiences produce two
        different `M^P` and every retrieval over them irreproducible."""
        forward = [
            _experience(sound="snoring", obj="bed"), _experience(sound="snoring", obj="bed"),
            _experience(sound="alarm", obj="fireplace"),
            _experience(sound="alarm", obj="fireplace"),
        ]
        one = abstract(ExperienceStore().extend(forward), min_support=2)
        two = abstract(ExperienceStore().extend(list(reversed(forward))), min_support=2)
        self.assertEqual(
            [p.concepts for p in one.patterns], [p.concepts for p in two.patterns]
        )
        self.assertEqual([p.sound_concept for p in one.patterns], ["alarm", "snoring"])

    def test_an_absent_room_sorts_last_rather_than_raising(self):
        """`None` cannot be compared with `str` under Python 3, so a naive sort key would
        raise here rather than order anything."""
        store = ExperienceStore().extend([
            _experience(room=None), _experience(room=None),
            _experience(room="bathroom"), _experience(room="bathroom"),
        ])
        patterns = abstract(store, min_support=2)
        self.assertEqual([p.room_concept for p in patterns.patterns], ["bathroom", None])

    def test_the_signature_is_the_unit_mean_of_the_groups_contexts(self):
        store = ExperienceStore().extend([
            _experience(context=(1.0, 0.0)), _experience(context=(0.0, 1.0))
        ])
        signature = abstract(store, min_support=2).patterns[0].strategy.signature
        np.testing.assert_allclose(
            signature, np.array([1.0, 1.0]) / math.sqrt(2.0), atol=1e-6
        )

    def test_the_strategy_averages_the_walk_and_the_gap(self):
        store = ExperienceStore().extend([
            _experience(gap=0.2, walk=_points((0, 0), (2, 0))),
            _experience(gap=0.8, walk=_points((0, 0), (4, 0))),
        ])
        strategy = abstract(store, min_support=2).patterns[0].strategy
        self.assertAlmostEqual(strategy.mean_final_gap_m, 0.5, places=5)
        self.assertAlmostEqual(float(strategy.trajectory[0]), 3.0, places=5)

    def test_a_group_whose_contexts_cancel_raises(self):
        """A key that cannot be normalised would be stored and never retrieved. Raising
        names the group; skipping it would drop a pattern with nothing saying so."""
        store = ExperienceStore().extend([
            _experience(context=(1.0, 0.0)), _experience(context=(-1.0, 0.0))
        ])
        with self.assertRaises(ValueError) as caught:
            abstract(store, min_support=2)
        self.assertIn("zero-norm", str(caught.exception))

    def test_an_empty_store_abstracts_to_an_empty_pattern_store(self):
        self.assertEqual(len(abstract(ExperienceStore(), min_support=1)), 0)


class TestStrategy(unittest.TestCase):
    """`rho_k` (eq. 17)."""

    def test_support_below_one_raises(self):
        with self.assertRaises(ValueError) as caught:
            Strategy(
                support=0, signature=np.ones(2, dtype=np.float32),
                trajectory=np.zeros(4, dtype=np.float32), mean_final_gap_m=1.0,
            )
        self.assertIn("not an abstraction", str(caught.exception))

    def test_the_arrays_are_read_only(self):
        strategy = Strategy(
            support=1, signature=np.ones(2, dtype=np.float32),
            trajectory=np.zeros(4, dtype=np.float32), mean_final_gap_m=1.0,
        )
        with self.assertRaises(ValueError):
            strategy.signature[0] = 3.0


class TestPatternStore(unittest.TestCase):
    def test_keys_are_the_signatures_in_order(self):
        store = ExperienceStore().extend([
            _experience(sound="alarm", obj="fireplace", context=(1.0, 0.0)),
            _experience(sound="alarm", obj="fireplace", context=(1.0, 0.0)),
            _experience(sound="snoring", obj="bed", context=(0.0, 1.0)),
            _experience(sound="snoring", obj="bed", context=(0.0, 1.0)),
        ])
        keys = abstract(store, min_support=2).keys
        self.assertEqual(len(keys), 2)
        np.testing.assert_allclose(np.stack(keys), [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)

    def test_an_empty_store_has_no_keys(self):
        self.assertEqual(PatternStore().keys, ())


class TestLongTermMemory(unittest.TestCase):
    """`M^L = {M^E, M^P, M^K}` (eq. 14), and what eq. 12 can actually range over."""

    def _memory(self, knowledge=None):
        experience = ExperienceStore().extend([_experience(), _experience()])
        return LongTermMemory(
            experience=experience,
            pattern=abstract(experience, min_support=2),
            knowledge=knowledge if knowledge is not None else SemanticStore(),
        )

    def test_it_holds_the_semantic_store_rather_than_a_second_table(self):
        """Eq. 18 lands on `SemanticEntry` almost exactly, and that store is already
        written by `prior_pass` and queried by `runner` through ONE embedding path.
        Forking it would be two paths answering the same question with no symptom."""
        knowledge = SemanticStore(entries=(SemanticEntry(
            sound_class="toilet_flush", room="bathroom", category="toilet",
            embedding=np.ones(4, dtype=np.float32), donor_scene="scene_a",
        ),))
        memory = self._memory(knowledge=knowledge)
        self.assertIs(memory.knowledge, knowledge)
        self.assertEqual(len(memory.knowledge), 1)

    def test_novelty_vectors_are_the_two_levels_that_share_the_fused_space(self):
        memory = self._memory()
        self.assertEqual(len(memory.novelty_vectors()), len(memory.experience) + len(memory.pattern))

    def test_m_k_is_excluded_because_its_keys_are_a_different_width(self):
        """**THE ARM THAT SAYS EQ. 12 CANNOT MEAN WHAT IT LOOKS LIKE.** `M^E`/`M^P` are
        keyed by fused `z^av`; an `m^K` is keyed by a CLAP vector of half the width. A
        `novelty_vectors` that included `M^K` would make `consolidate.novelty` raise on a
        width mismatch -- which is the check that catches a wrong encoder, so dropping it
        is not an option either. Eq. 20-22's separate per-level retrieval is the paper
        conceding the same fact."""
        knowledge = SemanticStore(entries=(SemanticEntry(
            sound_class="toilet_flush", room="bathroom", category="toilet",
            embedding=np.ones(4, dtype=np.float32), donor_scene="scene_a",
        ),))
        memory = self._memory(knowledge=knowledge)
        widths = {int(vector.size) for vector in memory.novelty_vectors()}
        self.assertEqual(widths, {2}, "an M^K key of width 4 reached the novelty set")
        # And the consequence, exercised rather than asserted in prose: including it
        # would raise inside eq. 12 itself.
        with self.assertRaises(ValueError):
            novelty(
                np.asarray([1.0, 0.0], dtype=np.float32),
                list(memory.novelty_vectors()) + [np.ones(4, dtype=np.float32)],
            )

    def test_novelty_over_a_real_memory_scores_a_stored_row_at_zero(self):
        """Eq. 12 against `M^L` as this module actually assembles it, rather than against
        a list built in the test."""
        memory = self._memory()
        self.assertAlmostEqual(
            novelty(memory.experience.keys[0], memory.novelty_vectors()), 0.0, places=5
        )
        self.assertAlmostEqual(
            novelty(np.asarray([0.0, 1.0], dtype=np.float32), memory.novelty_vectors()),
            1.0, places=5,
        )

    def test_an_empty_memory_reports_three_empty_levels(self):
        memory = LongTermMemory(
            experience=ExperienceStore(), pattern=PatternStore(), knowledge=SemanticStore()
        )
        self.assertEqual(memory.novelty_vectors(), ())
        self.assertIn("M^E=0 rows", repr(memory))
        self.assertIn("M^K=0 rows", repr(memory))

    def test_the_pattern_level_is_a_strict_abstraction_of_the_experience_level(self):
        """`M^P = G(M^E)` end to end: three successes and two failures over two triples."""
        experience = ExperienceStore().extend([
            _experience(sound="toilet_flush", obj="toilet"),
            _experience(sound="toilet_flush", obj="toilet"),
            _experience(sound="toilet_flush", obj="toilet", reached=False),
            _experience(sound="snoring", obj="bed"),
            _experience(sound="snoring", obj="bed", reached=False),
        ])
        memory = LongTermMemory(
            experience=experience,
            pattern=abstract(experience, min_support=2),
            knowledge=SemanticStore(),
        )
        self.assertEqual(len(memory.experience), 5)
        self.assertEqual(len(memory.pattern), 1)
        self.assertEqual(memory.pattern.patterns[0].strategy.support, 2)


class TestPattern(unittest.TestCase):
    def test_the_concept_triple_is_what_makes_two_patterns_the_same(self):
        strategy = Strategy(
            support=1, signature=np.ones(2, dtype=np.float32),
            trajectory=np.zeros(4, dtype=np.float32), mean_final_gap_m=1.0,
        )
        pattern = Pattern(
            sound_concept="alarm", object_concept="fireplace",
            room_concept="living_room", strategy=strategy,
        )
        self.assertEqual(pattern.concepts, ("alarm", "fireplace", "living_room"))


if __name__ == "__main__":
    unittest.main()
