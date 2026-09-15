"""DREAM §III.C's short-term memory (eq. 4-7), and the two properties it is only safe with.

PR 2 of 6 building DREAM as `ICRA2027_Memory` specifies it.

`M^S_t` is the one memory in the paper that is RESET BETWEEN EPISODES, and that reset is
what makes it safe: nothing here crosses an episode boundary, so no amount of it can carry
one episode's answer into the next. The tests below treat that as a property to be broken,
not a comment to be trusted.

The second property is IMMUTABILITY. `push` and `reset` return new memories, so a caller
holding step 5's memory still holds step 5's history at step 40. That is what makes a
retrieval reproducible after the fact, and it is the arm a mutating implementation fails:
such an implementation passes every "the memory has the right entries now" assertion and
silently rewrites the past.

No torch. `agent/stm.py` holds vectors and never encoders (ADR-0013:
`"agent": ("agent", "vlm", "types")`), which is the whole reason eq. 4-7 are testable on a
Mac at all.
"""

import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.agent.stm import ShortTermMemory, StmEntry, fuse
from earshot.types import Pose, Xyz


def _pose(x=0.0):
    return Pose(position=Xyz(float(x), 0.0, 0.0), yaw_rad=0.0)


def _entry(visual=(1.0, 0.0), audio=(0.0, 1.0), x=0.0, prev_action=None):
    return StmEntry(
        visual=np.asarray(visual, dtype=np.float32),
        audio=np.asarray(audio, dtype=np.float32),
        pose=_pose(x),
        prev_action=prev_action,
    )


class TestFuse(unittest.TestCase):
    """`f_fuse` (eq. 7). The paper does not say what it is; this is the choice and why."""

    def test_the_fused_vector_is_unit_norm(self):
        out = fuse([3.0, 4.0], [0.0, 5.0])
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)

    def test_both_halves_survive_intact_and_in_order(self):
        """Concatenation rather than a sum, so `M^K`'s sound-to-object table can query
        with the audio half alone instead of trying to unmix it."""
        out = fuse([1.0, 0.0], [0.0, 1.0])
        self.assertEqual(out.shape, (4,))
        np.testing.assert_allclose(out[:2], [1.0 / np.sqrt(2.0), 0.0], atol=1e-6)
        np.testing.assert_allclose(out[2:], [0.0, 1.0 / np.sqrt(2.0)], atol=1e-6)

    def test_a_loud_half_cannot_dominate_a_quiet_one(self):
        """THE ARM THE BALANCE ARGUMENT RESTS ON. Each half is normalised BEFORE the
        concatenation, so an encoder that happens to return a larger scale this step does
        not tilt `q_t` toward its modality. Without the pre-normalisation the 1000x visual
        half below would carry the whole vector."""
        balanced = fuse([1.0, 0.0], [0.0, 1.0])
        lopsided = fuse([1000.0, 0.0], [0.0, 1.0])
        np.testing.assert_allclose(balanced, lopsided, atol=1e-6)

    def test_an_empty_half_raises_rather_than_halving_the_width(self):
        """Both directions, because either encoder can be the one that failed."""
        with self.assertRaises(ValueError) as caught:
            fuse([], [1.0, 2.0])
        self.assertIn("both halves", str(caught.exception))
        with self.assertRaises(ValueError):
            fuse([1.0, 2.0], [])

    def test_the_width_is_the_sum_of_the_two_encoders(self):
        self.assertEqual(fuse(np.ones(512), np.ones(512)).shape, (1024,))


class TestStmEntry(unittest.TestCase):
    def test_the_entry_copies_its_vectors(self):
        """The forced-failure arm for aliasing. The runner reuses its embedding buffers
        between steps; an entry holding a reference would have every step of the episode
        reading the LAST step's observation, and `M^E` would record it."""
        visual = np.array([1.0, 0.0], dtype=np.float32)
        entry = _entry(visual=visual)
        visual[0] = 99.0
        self.assertAlmostEqual(float(entry.visual[0]), 1.0)

    def test_the_stored_vectors_are_read_only(self):
        entry = _entry()
        with self.assertRaises(ValueError):
            entry.visual[0] = 5.0

    def test_an_empty_embedding_raises_on_either_half(self):
        with self.assertRaises(ValueError) as caught:
            _entry(visual=())
        self.assertIn("visual", str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            _entry(audio=())
        self.assertIn("audio", str(caught.exception))

    def test_prev_action_is_none_only_before_the_first_action(self):
        """`a_{t-1}` does not exist at t=0. A sentinel string would make "had not acted
        yet" indistinguishable from an action named that."""
        self.assertIsNone(_entry().prev_action)
        self.assertEqual(_entry(prev_action="move_forward").prev_action, "move_forward")

    def test_fused_is_derived_from_the_halves_it_holds(self):
        entry = _entry(visual=(3.0, 4.0), audio=(0.0, 1.0))
        np.testing.assert_allclose(entry.fused, fuse((3.0, 4.0), (0.0, 1.0)), atol=1e-7)


class TestTheHorizon(unittest.TestCase):
    def test_entries_are_oldest_first(self):
        memory = ShortTermMemory(horizon=4)
        for step in range(3):
            memory = memory.push(_entry(x=step))
        self.assertEqual([e.pose.position.x for e in memory.entries], [0.0, 1.0, 2.0])

    def test_the_oldest_entry_is_evicted_past_the_horizon(self):
        memory = ShortTermMemory(horizon=2)
        for step in range(5):
            memory = memory.push(_entry(x=step))
        self.assertEqual(len(memory), 2)
        self.assertEqual([e.pose.position.x for e in memory.entries], [3.0, 4.0])

    def test_a_memory_constructed_over_its_horizon_is_trimmed_not_trusted(self):
        """The constructor is public and `replace` goes through it, so the invariant has
        to hold on construction rather than only on `push`."""
        memory = ShortTermMemory(horizon=2, entries=tuple(_entry(x=i) for i in range(5)))
        self.assertEqual([e.pose.position.x for e in memory.entries], [3.0, 4.0])

    def test_a_horizon_below_one_raises(self):
        """A zero horizon drops every entry as it is pushed, making `q_t` a function of
        the current observation alone with nothing on disk saying K was set to nothing."""
        for horizon in (0, -1):
            with self.assertRaises(ValueError) as caught:
                ShortTermMemory(horizon=horizon)
            self.assertIn("at least 1", str(caught.exception))


class TestImmutability(unittest.TestCase):
    """THE PROPERTY A MUTATING IMPLEMENTATION PASSES EVERY OTHER TEST WITHOUT."""

    def test_push_leaves_the_receiver_alone(self):
        before = ShortTermMemory(horizon=4).push(_entry(x=0))
        after = before.push(_entry(x=1))
        self.assertEqual(len(before), 1)
        self.assertEqual(len(after), 2)
        self.assertIsNot(before, after)

    def test_reset_leaves_the_receiver_alone(self):
        before = ShortTermMemory(horizon=4).push(_entry(x=0))
        self.assertEqual(len(before.reset()), 0)
        self.assertEqual(
            len(before), 1,
            "reset emptied the memory in place, so a caller holding the old episode's "
            "history silently lost it",
        )

    def test_reset_keeps_the_horizon(self):
        self.assertEqual(ShortTermMemory(horizon=7).push(_entry()).reset().horizon, 7)

    def test_an_episode_boundary_carries_nothing_across(self):
        """The paper's one structural guarantee for `M^S`, asserted rather than assumed."""
        memory = ShortTermMemory(horizon=8)
        for step in range(8):
            memory = memory.push(_entry(x=step))
        fresh = memory.reset()
        self.assertEqual(fresh.entries, ())
        self.assertIsNone(fresh.latest)
        self.assertIsNone(fresh.context(decay=0.9))


class TestTheQueryContext(unittest.TestCase):
    """The STM half of `q_t = f_q(z^av_t, M^S_t)` (eq. 19)."""

    def test_an_empty_memory_has_no_context(self):
        """`None`, not a zero vector: a zero query would match the store's own arithmetic
        rather than anything about the episode, and every retrieval would tie."""
        self.assertIsNone(ShortTermMemory(horizon=3).context(decay=1.0))

    def test_the_context_is_unit_norm(self):
        memory = ShortTermMemory(horizon=3)
        for step in range(3):
            memory = memory.push(_entry(visual=(1.0, float(step)), audio=(0.0, 1.0)))
        self.assertAlmostEqual(
            float(np.linalg.norm(memory.context(decay=0.7))), 1.0, places=5
        )

    def test_a_flat_decay_is_the_plain_mean_over_the_window(self):
        memory = ShortTermMemory(horizon=2)
        memory = memory.push(_entry(visual=(1.0, 0.0))).push(_entry(visual=(0.0, 1.0)))
        expected = fuse((1.0, 0.0), (0.0, 1.0)) + fuse((0.0, 1.0), (0.0, 1.0))
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(memory.context(decay=1.0), expected, atol=1e-6)

    def test_a_small_decay_leans_on_the_present(self):
        """The arm that proves the weighting is applied in the right direction. With a
        decay of 0.01 the latest entry should nearly be the context; a reversed weighting
        would return the oldest instead."""
        memory = ShortTermMemory(horizon=2)
        memory = memory.push(_entry(visual=(1.0, 0.0))).push(_entry(visual=(0.0, 1.0)))
        context = memory.context(decay=0.01)
        latest = memory.latest.fused
        oldest = memory.entries[0].fused
        self.assertGreater(
            float(np.dot(context, latest)), float(np.dot(context, oldest)),
            "the context is closer to the OLDEST entry, so the recency weighting runs "
            "backwards and q_t would ask about where the agent has been rather than "
            "where it is",
        )
        self.assertGreater(float(np.dot(context, latest)), 0.99)

    def test_a_decay_outside_zero_to_one_raises(self):
        """Both sides. Above 1 the oldest entry dominates, which inverts the recency this
        exists for; at or below 0 the weights are not a mean at all."""
        memory = ShortTermMemory(horizon=2).push(_entry())
        for decay in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError) as caught:
                memory.context(decay=decay)
            self.assertIn("(0, 1]", str(caught.exception))

    def test_decay_has_no_default(self):
        """`resolve_prior`'s `k` has none for the same reason: a knob with a default is a
        knob that reaches no artefact."""
        with self.assertRaises(TypeError):
            ShortTermMemory(horizon=2).push(_entry()).context()


if __name__ == "__main__":
    unittest.main()
