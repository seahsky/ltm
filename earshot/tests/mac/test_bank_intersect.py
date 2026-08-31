"""The bank of record, and the disjointness check that is the whole argument for it.

`clapgate-2` and `clapheld-1` agreed on the aggregate to 0.013 anchor top-1 and on the open-set
EER to three decimals, and then disagreed on which twelfth class to keep. `water_drops` scored
0.998 anchor recall on ESC-50 clips 0-7 and 0.449 on clips 8-15. Eight recordings do not pin a
class, so the intersection is the bank and a disputed class is cut.

The load-bearing part is that the two runs staged DISJOINT recordings. If they overlap, a class
clearing the bar in both is one observation counted twice and the intersection proves nothing,
so that is a raise rather than a warning.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.bank_intersect import assert_disjoint, clip_range


class TestClipRange(unittest.TestCase):
    def test_the_range_is_inclusive_of_both_ends(self):
        self.assertEqual(clip_range({"clip_start": "8", "n_per_class": "8"}), (8, 15))

    def test_a_run_from_before_the_flag_existed_defaults_to_zero(self):
        """Those runs staged from 0 by construction. That is a fact, not a guess."""
        self.assertEqual(clip_range({"n_per_class": "8"}), (0, 7))

    def test_a_zero_recording_count_raises(self):
        with self.assertRaises(ValueError):
            clip_range({"clip_start": "0", "n_per_class": "0"})


class TestDisjointness(unittest.TestCase):
    def test_the_real_pair_is_disjoint(self):
        assert_disjoint([("clapgate-2", (0, 7)), ("clapheld-1", (8, 15))])

    def test_an_overlap_raises(self):
        """A class clearing the bar on shared audio is one observation counted twice."""
        with self.assertRaises(ValueError) as caught:
            assert_disjoint([("a", (0, 7)), ("b", (4, 11))])
        self.assertIn("OVERLAP", str(caught.exception))

    def test_an_identical_range_raises(self):
        with self.assertRaises(ValueError):
            assert_disjoint([("a", (0, 7)), ("b", (0, 7))])

    def test_touching_but_not_overlapping_is_fine(self):
        assert_disjoint([("a", (0, 7)), ("b", (8, 8))])

    def test_the_check_is_pairwise_over_more_than_two(self):
        with self.assertRaises(ValueError):
            assert_disjoint([("a", (0, 7)), ("b", (8, 15)), ("c", (14, 20))])


if __name__ == "__main__":
    unittest.main()
