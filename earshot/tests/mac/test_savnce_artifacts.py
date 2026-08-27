"""The artefact manifest tells "changed under us" apart from "new and unrecorded".

Q9 of the design session bought a checksum manifest because the episode dataset and the
released checkpoints come from a Google Drive folder that carries no version. The risk
it guards is a silent swap: the same path, different bytes, a different number, and
nothing in the run that says so.

The reason severity exists at all is the failure mode of the obvious design. A manifest
that blocks on *any* discrepancy blocks every first run, and a gate that blocks the
first run is a gate that gets bypassed. So a changed or missing recorded artefact is
blocking, an unrecorded one is printed and continues, and both are always printed.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401  (raises on the wrong Python)

from earshot.tools.savnce_artifacts import HARD, SOFT, compare

RECORDED = "a" * 64
CHANGED = "b" * 64


def manifest(sha256):
    return {"sha256": sha256, "sizes": {}}


def observed(sha256):
    return {"sha256": sha256, "sizes": {}}


class TestBlockingDiscrepancies(unittest.TestCase):
    def test_a_recorded_artefact_with_different_bytes_blocks(self):
        found = compare(manifest({"ckpt.pth": RECORDED}), observed({"ckpt.pth": CHANGED}))
        self.assertEqual([item.severity for item in found], [HARD])
        self.assertEqual(found[0].expected, RECORDED)
        self.assertEqual(found[0].actual, CHANGED)

    def test_a_recorded_artefact_that_vanished_blocks(self):
        found = compare(manifest({"ckpt.pth": RECORDED}), observed({}))
        self.assertEqual([item.severity for item in found], [HARD])
        self.assertEqual(found[0].actual, "absent")


class TestNonBlockingDiscrepancies(unittest.TestCase):
    def test_an_unrecorded_artefact_is_reported_but_does_not_block(self):
        found = compare(manifest({}), observed({"ckpt.pth": RECORDED}))
        self.assertEqual([item.severity for item in found], [SOFT])
        self.assertFalse(any(item.is_hard for item in found))

    def test_a_first_run_against_an_empty_manifest_blocks_nothing(self):
        """The design constraint: an empty manifest must never stop a run."""
        found = compare(manifest({}), observed({"a.pth": RECORDED, "b.json.gz": CHANGED}))
        self.assertEqual(len(found), 2)
        self.assertFalse(any(item.is_hard for item in found))

    def test_an_explicit_null_reads_as_not_yet_recorded(self):
        found = compare(manifest({"ckpt.pth": None}), observed({"ckpt.pth": RECORDED}))
        self.assertEqual([item.severity for item in found], [SOFT])
        self.assertIn("not yet recorded", found[0].expected)


class TestTheCleanCase(unittest.TestCase):
    def test_identical_sets_produce_nothing(self):
        same = {"ckpt.pth": RECORDED, "test.json.gz": CHANGED}
        self.assertEqual(compare(manifest(same), observed(dict(same))), [])

    def test_a_malformed_manifest_is_blocking_rather_than_ignored(self):
        found = compare({"sha256": "not-an-object"}, observed({"ckpt.pth": RECORDED}))
        self.assertTrue(found and found[0].is_hard)


if __name__ == "__main__":
    unittest.main()
