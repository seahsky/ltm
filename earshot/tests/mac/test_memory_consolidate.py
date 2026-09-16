"""DREAM §III.D's consolidation (eq. 8-13), and the four choices the paper leaves open.

PR 3 of 6 building DREAM as `ICRA2027_Memory` specifies it.

The paper gives eq. 10, 12 and 13 literally and leaves `f_seg`, the segmentation of eq. 9,
`C_j` and `U_j` as prose clauses. Those four are choices, and the tests that matter here
are the ones that would still pass under the WRONG choice if they were written loosely:

  * the slow-drift corridor, which a pairwise coherence rule files as one segment and the
    running-mean rule cuts (`TestTheSegmentationChoice`);
  * an empty long-term memory scoring 1.0 rather than 2.0, which is the difference between
    a store that keeps learning and a store filled by episode one (`TestNovelty`);
  * a belief APPEARING not counting as a belief changing, which otherwise puts every
    episode's maximum surprise on whichever segment held the onset (`TestSurprise`);
  * `f_seg` being flat, which a recency-weighted mean passes every norm assertion without
    (`TestFSeg`).

No torch, no simulator. `memory/consolidate.py` is a leaf beside `store.py`
(ADR-0013: `"memory": ("types",)`) — eq. 12 ranges over three row types whose only shared
surface is a vector, so `novelty` takes vectors and the layer table did not have to move.
"""

import math
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.memory.consolidate import (
    ImportanceWeights,
    Segment,
    TrajectoryStep,
    contribution,
    importance,
    novelty,
    retain,
    segment_trajectory,
    surprise,
)
from earshot.types import Xyz


def _step(observation=(1.0, 0.0), x=0.0, belief=None):
    return TrajectoryStep(
        observation=np.asarray(observation, dtype=np.float32),
        position=Xyz(float(x), 0.0, 0.0),
        belief=belief,
    )


def _turned(radians):
    """A 2-D unit observation at `radians` from the x axis."""
    return (math.cos(radians), math.sin(radians))


def _drift(count, per_step):
    """A corridor: every consecutive pair nearly identical, the two ends far apart."""
    return [_step(observation=_turned(index * per_step), x=index) for index in range(count)]


class TestTrajectoryStep(unittest.TestCase):
    def test_the_step_copies_its_observation(self):
        """The runner reuses its embedding buffers between steps; a step holding a
        reference would file the LAST observation of the episode under every segment."""
        observation = np.array([1.0, 0.0], dtype=np.float32)
        step = _step(observation=observation)
        observation[0] = 99.0
        self.assertAlmostEqual(float(step.observation[0]), 1.0)

    def test_the_stored_observation_is_read_only(self):
        with self.assertRaises(ValueError):
            _step().observation[0] = 5.0

    def test_an_empty_observation_raises(self):
        with self.assertRaises(ValueError) as caught:
            _step(observation=())
        self.assertIn("empty observation", str(caught.exception))

    def test_a_non_finite_observation_raises(self):
        """A NaN turns every comparison in eq. 12 into a silent False, so `max` returns
        the first row it saw and novelty becomes a function of insertion order."""
        with self.assertRaises(ValueError) as caught:
            _step(observation=(1.0, float("nan")))
        self.assertIn("non-finite", str(caught.exception))

    def test_a_missing_belief_is_none_and_allowed(self):
        """Before the anomaly fires the agent has no source estimate. That is the
        ordinary state, not a fault."""
        self.assertIsNone(_step().belief)


class TestFSeg(unittest.TestCase):
    """`h_j = f_seg(delta_j)` (eq. 11). The paper names it and does not define it."""

    def test_the_representation_is_unit_norm(self):
        segment = Segment(steps=tuple(_step(observation=_turned(i * 0.1)) for i in range(4)))
        self.assertAlmostEqual(float(np.linalg.norm(segment.representation)), 1.0, places=5)

    def test_the_representation_is_the_mean_direction(self):
        segment = Segment(steps=(_step(observation=(1.0, 0.0)), _step(observation=(0.0, 1.0))))
        expected = np.array([1.0, 1.0], dtype=np.float32) / math.sqrt(2.0)
        np.testing.assert_allclose(segment.representation, expected, atol=1e-6)

    def test_f_seg_is_flat_and_not_recency_weighted(self):
        """THE ARM A RECENCY-WEIGHTED `f_seg` FAILS. `ShortTermMemory.context` leans on
        the present because a query has a "now"; a segment being FILED does not, and a
        weighted `h_j` would describe the boundary that ended the segment rather than the
        run it summarises. Reversing the steps must not move the representation."""
        steps = tuple(_step(observation=_turned(i * 0.2)) for i in range(5))
        forward = Segment(steps=steps).representation
        backward = Segment(steps=tuple(reversed(steps))).representation
        np.testing.assert_allclose(forward, backward, atol=1e-6)

    def test_a_segment_whose_observations_cancel_raises(self):
        """Two exactly opposed observations mean to a zero vector. Returning it would
        score 0.0 against every memory row and so read as MAXIMUM novelty."""
        segment = Segment(steps=(_step(observation=(1.0, 0.0)), _step(observation=(-1.0, 0.0))))
        with self.assertRaises(ValueError) as caught:
            segment.representation  # noqa: B018
        self.assertIn("zero-norm", str(caught.exception))

    def test_an_empty_segment_raises(self):
        with self.assertRaises(ValueError):
            Segment(steps=())

    def test_a_segment_mixing_widths_raises(self):
        with self.assertRaises(ValueError) as caught:
            Segment(steps=(_step(observation=(1.0, 0.0)), _step(observation=(1.0, 0.0, 0.0))))
        self.assertIn("mixes observation widths", str(caught.exception))


class TestTheSegmentationChoice(unittest.TestCase):
    """`tau = {delta_1, ..., delta_J}` (eq. 9). "Temporally coherent" is all the paper says."""

    def test_the_slow_drift_corridor_is_cut(self):
        """**THE ARM THE CHOICE RESTS ON.** 20 steps, each 0.998 cosine of the one before
        it, the two ends at 0.36. A pairwise rule never cuts this and files one segment
        whose `h_j` describes neither end. The running-mean rule cuts it."""
        steps = _drift(20, 0.06)
        consecutive = min(
            float(np.dot(steps[i].observation, steps[i + 1].observation))
            for i in range(len(steps) - 1)
        )
        ends = float(np.dot(steps[0].observation, steps[-1].observation))
        self.assertGreater(consecutive, 0.95, "the fixture is not a slow drift")
        self.assertLess(ends, 0.95, "the fixture's ends are not far apart")
        segments = segment_trajectory(steps, coherence=0.95, min_length=2, max_length=50)
        self.assertGreater(
            len(segments), 1,
            "the whole corridor was filed as one segment, so coherence is being measured "
            "against the previous step rather than against the segment's own h_j",
        )

    def test_a_partition_covers_every_step_in_order(self):
        steps = _drift(20, 0.06)
        segments = segment_trajectory(steps, coherence=0.95, min_length=2, max_length=7)
        flattened = [step for segment in segments for step in segment.steps]
        self.assertEqual(len(flattened), len(steps))
        for original, seen in zip(steps, flattened):
            self.assertIs(original, seen)

    def test_max_length_cuts_a_corridor_no_similarity_rule_would(self):
        """Fifty identical observations. Cosine is 1.0 at every pair, so only the cap
        ends a segment — without it `h_j` becomes an average of the whole episode."""
        segments = segment_trajectory(
            [_step(x=i) for i in range(50)], coherence=0.9, min_length=1, max_length=10
        )
        self.assertEqual([len(segment) for segment in segments], [10, 10, 10, 10, 10])

    def test_min_length_stops_one_outlier_opening_a_sliver(self):
        """A single orthogonal frame in the middle of an otherwise identical run.

        What `min_length` actually buys is narrower than it first looks, and the test says
        the narrower thing: at 1 the outlier cuts twice and leaves two one-step segments,
        each of whose `h_j` is a single frame. At 3 it cannot, though the outlier still
        drags the running mean far enough to cut at the NEXT step -- `min_length` bounds
        the segments it opens, it does not un-see the frame."""
        steps = [_step(x=0), _step(observation=(0.0, 1.0), x=1), _step(x=2), _step(x=3)]
        loose = segment_trajectory(steps, coherence=0.9, min_length=1, max_length=50)
        self.assertEqual([len(segment) for segment in loose], [1, 1, 2])
        firm = segment_trajectory(steps, coherence=0.9, min_length=3, max_length=50)
        self.assertTrue(
            all(len(segment) >= 3 for segment in firm[:-1]),
            "a segment below min_length opened before the tail",
        )

    def test_the_two_bounds_hold_over_a_whole_corridor(self):
        """The invariant both knobs exist for, asserted over a trajectory long enough to
        produce several segments rather than over a three-step fixture."""
        segments = segment_trajectory(_drift(40, 0.06), coherence=0.95, min_length=4, max_length=9)
        self.assertGreater(len(segments), 2)
        self.assertTrue(all(len(segment) <= 9 for segment in segments))
        self.assertTrue(all(len(segment) >= 4 for segment in segments[:-1]))

    def test_the_short_tail_is_its_own_segment(self):
        """A partition has to cover `tau`. Folding a 2-step tail into the segment before
        it would silently breach `max_length`."""
        segments = segment_trajectory(
            [_step(x=i) for i in range(12)], coherence=0.9, min_length=5, max_length=5
        )
        self.assertEqual([len(segment) for segment in segments], [5, 5, 2])

    def test_an_empty_trajectory_is_no_segments(self):
        """`()`, never a tuple holding an empty segment."""
        self.assertEqual(segment_trajectory([], coherence=0.9, min_length=1, max_length=5), ())

    def test_a_single_step_is_one_segment(self):
        self.assertEqual(
            len(segment_trajectory([_step()], coherence=0.9, min_length=4, max_length=5)), 1
        )

    def test_a_coherence_outside_zero_to_one_raises(self):
        for coherence in (0.0, -0.5, 1.5):
            with self.assertRaises(ValueError) as caught:
                segment_trajectory([_step()], coherence=coherence, min_length=1, max_length=2)
            self.assertIn("(0, 1]", str(caught.exception))

    def test_a_max_length_below_min_length_raises(self):
        with self.assertRaises(ValueError) as caught:
            segment_trajectory([_step()], coherence=0.9, min_length=5, max_length=4)
        self.assertIn("no cut rule can satisfy both", str(caught.exception))

    def test_a_min_length_below_one_raises(self):
        with self.assertRaises(ValueError):
            segment_trajectory([_step()], coherence=0.9, min_length=0, max_length=4)

    def test_every_knob_is_keyword_only_and_required(self):
        """The reason `resolve_prior`'s `k` has no default: a knob with a default is a
        knob that reaches no artefact."""
        with self.assertRaises(TypeError):
            segment_trajectory([_step()])
        with self.assertRaises(TypeError):
            segment_trajectory([_step()], 0.9, 1, 2)

    def test_a_trajectory_changing_width_raises(self):
        with self.assertRaises(ValueError) as caught:
            segment_trajectory(
                [_step(observation=(1.0, 0.0)), _step(observation=(1.0, 0.0, 0.0))],
                coherence=0.9, min_length=1, max_length=9,
            )
        self.assertIn("two encoders", str(caught.exception))


class TestContribution(unittest.TestCase):
    """`C_j` (eq. 10). The clause is "the contribution of the segment to successful
    navigation"; the choice is the distance the episode actually closed, as a MULTIPLE of
    the mean segment's (ADR-0024 -- a share carried a hidden 1/J that `N_j` does not)."""

    def _segments(self, positions):
        return segment_trajectory(
            [_step(x=x) for x in positions], coherence=0.9, min_length=1, max_length=2
        )

    def test_the_mean_is_one_whatever_the_segment_count(self):
        """THE PROPERTY THE 1/J BUG BROKE, pinned at three different `J`.

        `mean_j(C_j) = 1.0` exactly, so `I_j` is commensurate with `N_j` (a cosine) at
        every episode length. Under the old share form the mean was `1/J`, which is what
        made eq. 10 degenerate to eq. 12 and cost `dream-2` its retention. ADR-0024.
        """
        for positions in ([0.0, 1.0, 2.0, 5.0],
                          [0.0, 1.0, 2.0, 3.0, 4.0, 7.0],
                          [float(i) for i in range(12)]):
            segments = self._segments(positions)
            scores = contribution(segments, target=Xyz(100.0, 0.0, 0.0))
            self.assertAlmostEqual(sum(scores) / len(scores), 1.0, 5)

    def test_the_scale_does_not_move_with_the_segment_count(self):
        """A longer episode must not make every segment look less important. Two
        trajectories closing the same distance over different `J` keep the same mean."""
        short = contribution(self._segments([0.0, 2.0, 4.0]), target=Xyz(100.0, 0.0, 0.0))
        long = contribution(
            self._segments([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
            target=Xyz(100.0, 0.0, 0.0),
        )
        self.assertAlmostEqual(sum(short) / len(short), sum(long) / len(long), 5)

    def test_the_segment_that_closed_more_scores_more(self):
        segments = self._segments([0.0, 1.0, 2.0, 8.0])
        shares = contribution(segments, target=Xyz(10.0, 0.0, 0.0))
        self.assertEqual(len(shares), 2)
        self.assertGreater(shares[1], shares[0])

    def test_the_transitions_tile_the_episode_exactly(self):
        """One rule for both `C_j` and `U_j`: a transition belongs to the segment holding
        its LATER step. Every transition lands in exactly one bucket, so the scores are
        measured against the episode's own total and not against an arbitrary subset.

        The tiling is recovered by scaling back through the mean (ADR-0024: the divisor
        is the mean segment, so the raw amount is `score * total / J`)."""
        positions = [0.0, 1.0, 3.0, 6.0, 7.0, 9.0]
        segments = self._segments(positions)
        target = Xyz(20.0, 0.0, 0.0)
        scores = contribution(segments, target=target)
        closed = positions[-1] - positions[0]
        raw = [score * closed / len(scores) for score in scores]
        self.assertAlmostEqual(sum(raw), closed, places=4)

    def test_walking_away_scores_zero_and_never_negative(self):
        """Scoring a detour negative would make `alpha` a penalty scale as well as a
        weight, which eq. 10 gives no room for.

        The exact shares are asserted, not just their sign. An earlier version of this
        test used a trajectory whose net progress was exactly zero, so the unclamped
        implementation ALSO returned all zeros and the arm passed vacuously -- the
        forced-failure run is what caught that."""
        segments = self._segments([0.0, 3.0, 2.5, 2.0])
        scores = contribution(segments, target=Xyz(10.0, 0.0, 0.0))
        # The whole of the episode's progress sits in one of two segments, so that one
        # carries 2x the mean and the other carries none. Under the old share form this
        # read [1.0, 0.0]; the ZERO is the arm that matters and it is unchanged.
        np.testing.assert_allclose(scores, [2.0, 0.0], atol=1e-6)

    def test_an_episode_that_closed_nothing_scores_every_segment_zero(self):
        """All zeros rather than an equal split: no segment contributed, and an equal
        split would say each contributed its share of nothing."""
        segments = self._segments([3.0, 3.0, 3.0, 3.0])
        self.assertEqual(
            set(contribution(segments, target=Xyz(10.0, 0.0, 0.0))), {0.0}
        )

    def test_height_is_ignored(self):
        """`Xyz.horizontal_distance_to`, the tree's convention: a 3D distance would charge
        a segment for a staircase. The same walk at two heights must score identically."""
        def walk(height_per_step):
            steps = [
                TrajectoryStep(
                    observation=np.asarray((1.0, 0.0), dtype=np.float32),
                    position=Xyz(float(index), height_per_step * index, 0.0),
                    belief=None,
                )
                for index in range(4)
            ]
            segments = segment_trajectory(steps, coherence=0.9, min_length=1, max_length=2)
            return contribution(segments, target=Xyz(10.0, 0.0, 0.0))

        np.testing.assert_allclose(walk(5.0), walk(0.0), atol=1e-6)

    def test_target_is_keyword_only(self):
        with self.assertRaises(TypeError):
            contribution(self._segments([0.0, 1.0]), Xyz(0.0, 0.0, 0.0))


class TestSurprise(unittest.TestCase):
    """`U_j` (eq. 10): "unexpected changes in the agent's predictions"."""

    def _segments(self, beliefs):
        steps = [_step(x=index, belief=belief) for index, belief in enumerate(beliefs)]
        return segment_trajectory(steps, coherence=0.9, min_length=1, max_length=2)

    def test_the_segment_whose_belief_moved_further_scores_more(self):
        beliefs = [
            Xyz(0.0, 0.0, 0.0), Xyz(0.1, 0.0, 0.0), Xyz(0.2, 0.0, 0.0), Xyz(9.0, 0.0, 0.0)
        ]
        scores = surprise(self._segments(beliefs))
        self.assertEqual(len(scores), 2)
        self.assertGreater(scores[1], scores[0])
        self.assertAlmostEqual(sum(scores) / len(scores), 1.0, places=5)

    def test_a_belief_appearing_is_not_a_belief_changing(self):
        """**THE ARM THAT KEEPS `U_j` ABOUT THE AGENT.** `None -> Xyz` is the anomaly
        firing, and counting it as a revision would put the maximum surprise of EVERY
        episode on whichever segment held the onset -- a measurement of the task's
        structure, not of the agent's predictions."""
        beliefs = [None, None, Xyz(50.0, 0.0, 0.0), Xyz(50.5, 0.0, 0.0)]
        scores = surprise(self._segments(beliefs))
        self.assertEqual(scores[0], 0.0)
        # Every revision the episode had is in the second of two segments, so it carries
        # 2x the mean. The onset itself contributing NOTHING is the arm under test.
        self.assertAlmostEqual(scores[1], 2.0, places=5)

    def test_a_belief_disappearing_is_not_a_revision_either(self):
        beliefs = [Xyz(1.0, 0.0, 0.0), Xyz(1.0, 0.0, 0.0), None, None]
        self.assertEqual(set(surprise(self._segments(beliefs))), {0.0})

    def test_an_episode_that_never_had_a_belief_scores_zero(self):
        self.assertEqual(set(surprise(self._segments([None] * 4))), {0.0})

    def test_a_belief_that_never_moved_scores_zero(self):
        steady = [Xyz(2.0, 0.0, 0.0)] * 4
        self.assertEqual(set(surprise(self._segments(steady))), {0.0})


class TestNovelty(unittest.TestCase):
    """`N_j = 1 - max sim(h_j, m)` (eq. 12), the one scorer the paper writes out."""

    def test_a_row_the_memory_already_holds_is_not_novel(self):
        h = np.array([1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(novelty(h, [np.array([1.0, 0.0], dtype=np.float32)]), 0.0, 5)

    def test_an_orthogonal_memory_scores_one(self):
        h = np.array([1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(novelty(h, [np.array([0.0, 1.0], dtype=np.float32)]), 1.0, 5)

    def test_the_nearest_row_wins_not_the_last(self):
        h = np.array([1.0, 0.0], dtype=np.float32)
        memory = [
            np.array([0.0, 1.0], dtype=np.float32),
            np.array([0.99, 0.14], dtype=np.float32),
            np.array([-1.0, 0.0], dtype=np.float32),
        ]
        self.assertLess(novelty(h, memory), 0.05)

    def test_an_empty_memory_scores_one_and_not_two(self):
        """**THE ARM THAT DECIDES WHETHER THE STORE KEEPS LEARNING.** `max` over nothing
        is undefined and the paper does not say. At 2.0 the first episode's segments
        outscore every later segment on novelty purely because the store was empty, fill
        `M^E`, and then threshold out everything that follows."""
        self.assertEqual(novelty(np.array([1.0, 0.0], dtype=np.float32), []), 1.0)

    def test_a_zero_norm_memory_row_is_skipped_not_scored_zero(self):
        """Scoring it 0.0 would cap every `N_j` at 1.0 the moment one degenerate row
        entered the store. `store.py::_vote` skips for the same reason."""
        h = np.array([1.0, 0.0], dtype=np.float32)
        memory = [np.zeros(2, dtype=np.float32), np.array([-1.0, 0.0], dtype=np.float32)]
        self.assertAlmostEqual(novelty(h, memory), 2.0, places=5)

    def test_an_all_degenerate_memory_reads_as_empty(self):
        self.assertEqual(
            novelty(np.array([1.0, 0.0], dtype=np.float32), [np.zeros(2, dtype=np.float32)]),
            1.0,
        )

    def test_a_width_mismatch_raises_rather_than_reading_as_distant(self):
        with self.assertRaises(ValueError) as caught:
            novelty(np.array([1.0, 0.0], dtype=np.float32), [np.ones(3, dtype=np.float32)])
        self.assertIn("two embedding spaces", str(caught.exception))

    def test_a_zero_norm_query_raises(self):
        with self.assertRaises(ValueError):
            novelty(np.zeros(2, dtype=np.float32), [np.array([1.0, 0.0], dtype=np.float32)])


class TestImportanceWeights(unittest.TestCase):
    def test_a_negative_weight_raises(self):
        """A negative `gamma` retains what memory already holds and discards what is new,
        which is eq. 12 run backwards rather than down-weighted."""
        for kwargs in ({"alpha": -1.0}, {"beta": -1.0}, {"gamma": -1.0}):
            fields = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
            fields.update(kwargs)
            with self.assertRaises(ValueError) as caught:
                ImportanceWeights(**fields)
            self.assertIn("must be >= 0", str(caught.exception))

    def test_an_all_zero_triple_raises(self):
        """Every `I_j` is exactly 0, so eq. 13 keeps everything or nothing on the sign of
        `eta` alone and nothing on disk says consolidation stopped discriminating."""
        with self.assertRaises(ValueError) as caught:
            ImportanceWeights(alpha=0.0, beta=0.0, gamma=0.0)
        self.assertIn("stopped discriminating", str(caught.exception))

    def test_zeroing_two_of_three_is_allowed(self):
        """A single-term ablation of eq. 10 is a legitimate arm, not a misconfiguration."""
        self.assertEqual(ImportanceWeights(alpha=0.0, beta=0.0, gamma=1.0).gamma, 1.0)

    def test_the_weights_have_no_defaults(self):
        with self.assertRaises(TypeError):
            ImportanceWeights()


class TestImportance(unittest.TestCase):
    """`I_j = alpha*C_j + beta*U_j + gamma*N_j` (eq. 10)."""

    def _segments(self):
        steps = [
            _step(observation=_turned(index * 0.5), x=float(index), belief=Xyz(float(index), 0, 0))
            for index in range(4)
        ]
        return segment_trajectory(steps, coherence=0.9, min_length=1, max_length=2)

    def test_one_score_per_segment_in_order(self):
        segments = self._segments()
        scores = importance(
            segments,
            target=Xyz(10.0, 0.0, 0.0),
            memory=[],
            weights=ImportanceWeights(alpha=1.0, beta=1.0, gamma=1.0),
        )
        self.assertEqual(len(scores), len(segments))

    def test_each_weight_moves_the_score_it_weights(self):
        """Three single-term arms. Under an implementation that transposed two terms, at
        least one of these lands on the wrong segment."""
        segments = self._segments()
        target, memory = Xyz(10.0, 0.0, 0.0), []
        only_c = importance(
            segments, target=target, memory=memory,
            weights=ImportanceWeights(alpha=1.0, beta=0.0, gamma=0.0),
        )
        only_u = importance(
            segments, target=target, memory=memory,
            weights=ImportanceWeights(alpha=0.0, beta=1.0, gamma=0.0),
        )
        only_n = importance(
            segments, target=target, memory=memory,
            weights=ImportanceWeights(alpha=0.0, beta=0.0, gamma=1.0),
        )
        np.testing.assert_allclose(only_c, contribution(segments, target=target), atol=1e-6)
        np.testing.assert_allclose(only_u, surprise(segments), atol=1e-6)
        np.testing.assert_allclose(only_n, [1.0] * len(segments), atol=1e-6)

    def test_the_score_is_the_weighted_sum_of_the_three(self):
        segments = self._segments()
        target = Xyz(10.0, 0.0, 0.0)
        memory = [np.asarray(_turned(0.1), dtype=np.float32)]
        weights = ImportanceWeights(alpha=2.0, beta=3.0, gamma=5.0)
        expected = [
            2.0 * c + 3.0 * u + 5.0 * novelty(segment.representation, memory)
            for segment, c, u in zip(
                segments, contribution(segments, target=target), surprise(segments)
            )
        ]
        np.testing.assert_allclose(
            importance(segments, target=target, memory=memory, weights=weights),
            expected,
            atol=1e-6,
        )


class TestRetain(unittest.TestCase):
    """`D* = {delta_j | I_j > eta}` (eq. 13)."""

    def _three(self):
        return segment_trajectory(
            [_step(x=float(index)) for index in range(3)],
            coherence=0.9, min_length=1, max_length=1,
        )

    def test_only_segments_above_the_threshold_survive(self):
        segments = self._three()
        kept = retain(segments, [0.1, 0.9, 0.5], eta=0.4, max_kept=99)
        self.assertEqual([segment.steps[0].position.x for segment in kept], [1.0, 2.0])

    def test_the_threshold_is_strict(self):
        """`> eta`, as written. A segment exactly at the threshold is dropped."""
        segments = self._three()
        self.assertEqual(len(retain(segments, [0.5, 0.5, 0.5], eta=0.5, max_kept=99)), 0)

    def test_order_is_preserved(self):
        segments = self._three()
        kept = retain(segments, [1.0, 1.0, 1.0], eta=0.0, max_kept=99)
        self.assertEqual([segment.steps[0].position.x for segment in kept], [0.0, 1.0, 2.0])

    def test_a_length_mismatch_raises_rather_than_truncating(self):
        """Zipping to the shorter of the two would drop `D*`'s tail with nothing saying
        so, which is the silent-failure shape this repo keeps removing."""
        with self.assertRaises(ValueError) as caught:
            retain(self._three(), [0.1, 0.9], eta=0.0, max_kept=99)
        self.assertIn("not these segments' scores", str(caught.exception))

    def test_eta_is_keyword_only_and_required(self):
        with self.assertRaises(TypeError):
            retain(self._three(), [1.0, 1.0, 1.0])

    def test_retain_does_not_rescore(self):
        """Separate from `importance` on purpose: the scores are what the audit records
        and what tells a later sweep whether `eta` was set anywhere sensible."""
        segments = self._three()
        self.assertEqual(len(retain(segments, [9.0, 9.0, 9.0], eta=8.0, max_kept=99)), 3)
        self.assertEqual(len(retain(segments, [1.0, 1.0, 1.0], eta=8.0, max_kept=99)), 0)

    def test_the_cap_is_inert_when_fewer_clear_eta_than_it_allows(self):
        """BOTH ARMS OF THE CAP, and this is the one that says it is not always on.

        A deviation from eq. 13 that fired on every episode would BE the retention rule.
        This pins that it is absent whenever the gate is doing its own work.
        """
        segments = self._three()
        kept = retain(segments, [0.1, 0.9, 0.5], eta=0.4, max_kept=2)
        self.assertEqual([segment.steps[0].position.x for segment in kept], [1.0, 2.0])

    def test_the_cap_keeps_the_highest_scoring_and_holds_order(self):
        """The other arm: more clear `eta` than the cap allows, so it bites. The kept set
        is the top `max_kept` BY SCORE, returned in trajectory order, not score order."""
        segments = self._three()
        kept = retain(segments, [0.9, 0.2, 0.7], eta=0.1, max_kept=2)
        self.assertEqual([segment.steps[0].position.x for segment in kept], [0.0, 2.0])

    def test_the_cap_bounds_the_empty_memory_flood(self):
        """THE CASE THE CAP EXISTS FOR. `novelty` hands an empty `M^E` `N_j = 1.0` for
        every segment, so episode one of a chain clears any `eta` below 1 outright --
        `dream-2` wrote 38 of its 45 rows from one walk that way. ADR-0024."""
        segments = self._three()
        every_segment_clears = [1.4, 1.2, 1.3]
        self.assertEqual(len(retain(segments, every_segment_clears, eta=0.5, max_kept=99)), 3)
        self.assertEqual(len(retain(segments, every_segment_clears, eta=0.5, max_kept=1)), 1)

    def test_ties_at_the_cap_boundary_go_to_the_earlier_segment(self):
        """Deterministic in its inputs, not in a sort's stability."""
        segments = self._three()
        kept = retain(segments, [0.8, 0.8, 0.8], eta=0.1, max_kept=2)
        self.assertEqual([segment.steps[0].position.x for segment in kept], [0.0, 1.0])

    def test_a_cap_below_one_raises(self):
        """A cap of 0 writes nothing ever, which reads on an audit exactly like the eta
        collapse it was added to prevent. Refused at the door instead."""
        with self.assertRaises(ValueError) as caught:
            retain(self._three(), [1.0, 1.0, 1.0], eta=0.0, max_kept=0)
        self.assertIn("max_kept=0", str(caught.exception))

    def test_max_kept_is_keyword_only_and_required(self):
        with self.assertRaises(TypeError):
            retain(self._three(), [1.0, 1.0, 1.0], eta=0.0)


class TestTheWholePipeline(unittest.TestCase):
    """Eq. 8 through 13 end to end, on one trajectory."""

    def test_a_trajectory_goes_through_to_a_retained_set(self):
        steps = [
            _step(
                observation=_turned(index * 0.08),
                x=float(index) * 0.4,
                belief=Xyz(20.0 - index * 0.1, 0.0, 0.0),
            )
            for index in range(24)
        ]
        segments = segment_trajectory(steps, coherence=0.95, min_length=3, max_length=12)
        self.assertGreater(len(segments), 1)
        scores = importance(
            segments,
            target=Xyz(20.0, 0.0, 0.0),
            memory=[],
            weights=ImportanceWeights(alpha=1.0, beta=1.0, gamma=1.0),
        )
        self.assertEqual(len(scores), len(segments))
        # Nothing in memory, so every N_j is 1.0 and every score clears a threshold below
        # gamma. The interesting case -- a threshold that drops something -- is above.
        self.assertEqual(len(retain(segments, scores, eta=0.99, max_kept=99)), len(segments))
        self.assertEqual(len(retain(segments, scores, eta=99.0, max_kept=99)), 0)

    def test_consolidation_never_mutates_its_input(self):
        """Every function here is pure. `task/` will call these on the episode's own
        trajectory and must get the same trajectory back."""
        steps = [_step(observation=_turned(i * 0.3), x=float(i), belief=Xyz(1.0, 0, 0))
                 for i in range(8)]
        before = [np.array(step.observation, copy=True) for step in steps]
        segments = segment_trajectory(steps, coherence=0.9, min_length=2, max_length=4)
        importance(
            segments, target=Xyz(9.0, 0.0, 0.0), memory=[],
            weights=ImportanceWeights(alpha=1.0, beta=1.0, gamma=1.0),
        )
        for original, step in zip(before, steps):
            np.testing.assert_array_equal(original, step.observation)


if __name__ == "__main__":
    unittest.main()
