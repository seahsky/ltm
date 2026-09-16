"""The guard that stops a resumed sweep from silently shortening the DREAM chain.

`--resume` skips a scene directory that already holds a finished run. The `dream` arm
also appends `M^E` to one file, and those two facts can disagree: a scene whose episodes
completed but whose memory dump never landed leaves a finished directory and a memory
file that never saw it. Skipping it drops a house from `M^E` for the rest of the sweep.

`matrix-1` lost two of nineteen scenes exactly this way and nothing on disk said so until
a review found it afterwards. Both arms here, per ADR-0014: the agreeing case passes and
every way of disagreeing is required to fail, because a guard that cannot refuse is
decoration.
"""

import json
import pathlib
import shutil
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.chain_check import compare, main


class TestCompare(unittest.TestCase):
    def test_the_same_scenes_in_the_same_order_agree(self):
        agree, why = compare(("a", "b", "c"), ("a", "b", "c"))
        self.assertTrue(agree)
        print(why)

    def test_an_empty_chain_matches_nothing_finished(self):
        """The first invocation of a sweep: no memory file, no finished scene."""
        agree, _why = compare((), ())
        self.assertTrue(agree)

    def test_a_scene_the_memory_never_saw_is_refused(self):
        """**THE `matrix-1` FAILURE.** Three scenes finished, two reached the memory."""
        agree, why = compare(("a", "b"), ("a", "b", "c"))
        self.assertFalse(agree)
        self.assertIn("silently drop", why)
        print(why)

    def test_a_memory_ahead_of_the_directories_is_refused(self):
        """The other direction, and it is not benign either: the memory holds a scene
        whose directory is gone or unfinished, so the rows and the record disagree."""
        agree, why = compare(("a", "b", "c"), ("a", "b"))
        self.assertFalse(agree)
        print(why)

    def test_finished_scenes_with_NO_provenance_at_all_are_refused(self):
        """A missing scene list is the shape a pre-provenance memory file has, and
        resuming onto it would restart M^E from empty with nothing saying so."""
        agree, why = compare((), ("a", "b"))
        self.assertFalse(agree)
        self.assertIn("restart M^E from empty", why)
        print(why)

    def test_the_same_scenes_in_a_DIFFERENT_order_are_refused(self):
        """Scene order is part of the chain's result: episode k of scene n was scored
        against scenes 0..n-1. Two orders are two memories."""
        agree, why = compare(("a", "c", "b"), ("a", "b", "c"))
        self.assertFalse(agree)
        self.assertIn("position 1", why)
        print(why)


class TestTheCommandLine(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def memory(self, scenes):
        path = pathlib.Path(self.root) / "memory.json"
        path.write_text(json.dumps({"scenes": list(scenes), "experience": []}))
        return str(path)

    def test_agreement_exits_zero(self):
        self.assertEqual(main([self.memory(["a", "b"]), "--expect", "a b"]), 0)

    def test_disagreement_exits_two(self):
        self.assertEqual(main([self.memory(["a"]), "--expect", "a b"]), 2)

    def test_a_missing_file_with_nothing_finished_exits_zero(self):
        """The first scene of a chain has no memory file and that is not an error."""
        missing = str(pathlib.Path(self.root) / "not-yet.json")
        self.assertEqual(main([missing, "--expect", ""]), 0)

    def test_a_missing_file_with_finished_scenes_exits_two(self):
        missing = str(pathlib.Path(self.root) / "not-yet.json")
        self.assertEqual(main([missing, "--expect", "a"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
