"""The 5:3:2 split: exact counts, stable assignment, and the unit it is taken at.

Three things could go wrong quietly and all three are tested rather than commented.

**Counts that do not sum.** Repeated flooring loses up to two units on a three-way split, and
a silently dropped scene is a denominator nobody counted. Largest remainder is exact.

**An assignment that moves when a scene is added.** Sorting by `sha256` of the label means a
new label takes its own slot; a seeded shuffle would reshuffle everything and quietly move
scenes across the test boundary between one run and the next.

**A split taken at the episode level.** Episodes inside a scene share a room, a source and a
renderer, so an episode-level split puts the test block in rooms development already tuned on.
The API only accepts scene labels, which is the enforcement.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.dataset_split import (
    ESC50_RECORDINGS_PER_CLASS,
    ROLES,
    split_counts,
    split_recordings,
    split_scenes,
)

# The real val labels are opaque IDs; these stand in with the same shape.
SCENES = tuple("scene{:02d}".format(i) for i in range(20))


class TestSplitCounts(unittest.TestCase):
    def test_twenty_scenes_split_ten_six_four(self):
        self.assertEqual(split_counts(20), (10, 6, 4))

    def test_forty_recordings_split_twenty_twelve_eight(self):
        self.assertEqual(split_counts(ESC50_RECORDINGS_PER_CLASS), (20, 12, 8))

    def test_the_counts_always_sum_to_n(self):
        """Largest remainder, not repeated flooring. This is the whole reason it exists."""
        for n in range(0, 200):
            self.assertEqual(sum(split_counts(n)), n, "lost a unit at n={}".format(n))

    def test_an_awkward_n_still_sums(self):
        self.assertEqual(sum(split_counts(7)), 7)
        self.assertEqual(sum(split_counts(11)), 11)

    def test_a_ratio_with_the_wrong_number_of_parts_raises(self):
        with self.assertRaises(ValueError):
            split_counts(20, (5, 3))
        with self.assertRaises(ValueError):
            split_counts(20, (5, 3, 2, 1))

    def test_a_degenerate_ratio_raises(self):
        with self.assertRaises(ValueError):
            split_counts(20, (0, 0, 0))
        with self.assertRaises(ValueError):
            split_counts(20, (5, -3, 2))

    def test_a_negative_n_raises(self):
        with self.assertRaises(ValueError):
            split_counts(-1)


class TestSplitScenes(unittest.TestCase):
    def test_every_scene_lands_in_exactly_one_role(self):
        blocks = split_scenes(SCENES)
        assigned = [label for block in blocks for label in block.members]
        self.assertEqual(sorted(assigned), sorted(SCENES))
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_the_roles_are_in_ratio_order(self):
        blocks = split_scenes(SCENES)
        self.assertEqual(tuple(b.role for b in blocks), ROLES)
        self.assertEqual(tuple(b.n for b in blocks), (10, 6, 4))

    def test_the_assignment_is_identical_across_calls(self):
        self.assertEqual(
            [b.members for b in split_scenes(SCENES)],
            [b.members for b in split_scenes(SCENES)],
        )

    def test_input_order_does_not_change_the_assignment(self):
        forward = [b.members for b in split_scenes(SCENES)]
        backward = [b.members for b in split_scenes(tuple(reversed(SCENES)))]
        self.assertEqual(forward, backward)

    def test_adding_a_scene_does_not_reshuffle_the_others(self):
        """A seeded shuffle would permute everything. Hashing inserts one label and stops.

        Block sizes change when n changes, so not every scene can keep its role. What must
        hold is that the underlying ORDER is stable: strike the new label out and the
        remaining twenty are in the same sequence they were in before.
        """
        def order(labels):
            return [label for block in split_scenes(labels) for label in block.members]

        before = order(SCENES)
        after = [label for label in order(SCENES + ("scene99",)) if label != "scene99"]
        self.assertEqual(before, after)

    def test_a_seeded_shuffle_would_have_failed_the_test_above(self):
        """The control. Without it, the stability test could be passing for a weaker reason."""
        import random

        shuffled_before = sorted(SCENES, key=lambda x: random.Random(7).random())
        rng = random.Random(7)
        shuffled_after = sorted(SCENES + ("scene99",), key=lambda x: rng.random())
        self.assertNotEqual(
            shuffled_before,
            [label for label in shuffled_after if label != "scene99"],
            "the control did not diverge, so it proves nothing about hashing",
        )

    def test_a_duplicate_label_raises(self):
        """Two entries for one scene would inflate a block and double-count its episodes."""
        with self.assertRaises(ValueError):
            split_scenes(SCENES + ("scene00",))

    def test_no_scenes_gives_empty_blocks_rather_than_raising(self):
        blocks = split_scenes(())
        self.assertEqual(tuple(b.n for b in blocks), (0, 0, 0))


class TestSplitRecordings(unittest.TestCase):
    def test_the_blocks_are_contiguous_and_cover_everything(self):
        """Contiguous because `--clip-start` takes a start offset, not a set."""
        blocks = split_recordings()
        self.assertEqual([b.span for b in blocks], [(0, 19), (20, 31), (32, 39)])

    def test_the_blocks_do_not_overlap(self):
        spans = [b.span for b in split_recordings()]
        for (a_lo, a_hi), (b_lo, _b_hi) in zip(spans, spans[1:]):
            self.assertLess(a_hi, b_lo)
            self.assertEqual(b_lo, a_hi + 1)

    def test_the_clips_already_spent_are_inside_development(self):
        """clapgate-2 staged 0..7 and clapheld-1 staged 8..15. Both must be development.

        If either had crossed into verification, the bank of record would be contaminated
        and no clean number would be available without re-staging.
        """
        development = split_recordings()[0]
        self.assertEqual(development.span, (0, 19))
        for spent in range(0, 16):
            self.assertIn(str(spent), development.members)

    def test_a_zero_recording_count_raises(self):
        with self.assertRaises(ValueError):
            split_recordings(0)


if __name__ == "__main__":
    unittest.main()
