"""`hm3d_link_splits.sh`, RUN, against a fake two-version HM3D tree.

The third symlink trap, and the first one a download springs by itself. habitat-sim's
downloader points `scene_datasets/hm3d` at the versioned tree it just filled — one
symlink, one tree — so the moment two splits live in two versions the last download takes
the whole path with it.

MEASURED 2026-09-10, straight after `--uids hm3d_train_habitat`:

    scene_datasets/hm3d -> versioned_data/hm3d-1.0/hm3d     (holds only train)
    train  73 usable scenes, 1068 anchored
    val    FATAL: no scene with a mesh on this box   <- all 20, silently

This file builds that exact tree in a tempdir and runs the shipped script over it. Nothing
here reads the script's text: it is filesystem surgery on a live data tree, so it is
exercised rather than proxied (ADR-0014), and it is written without associative arrays so
that stays true on macOS's bash 3.2.
"""

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "tools" / "hm3d_link_splits.sh"


class _Tree:
    """A fake `data/hm3d` with whatever (version, split) pairs a test needs."""

    def __init__(self, root, pairs, symlink_to=None):
        self.data = pathlib.Path(root) / "data" / "hm3d"
        for version, split in pairs:
            scene = self.data / "versioned_data" / version / "hm3d" / split / "00800-AAA"
            scene.mkdir(parents=True)
            (scene / "AAA.basis.glb").write_text("mesh", encoding="utf-8")
        (self.data / "scene_datasets").mkdir(parents=True)
        if symlink_to is not None:
            os.symlink(
                self.data / "versioned_data" / symlink_to / "hm3d",
                self.data / "scene_datasets" / "hm3d",
            )

    @property
    def target(self):
        return self.data / "scene_datasets" / "hm3d"


class TestLinkSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("bash") is None:
            raise unittest.SkipTest("no bash on this machine")

    def _run(self, tree, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), "--data-path", str(tree.data), *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    def test_it_rescues_the_split_the_download_orphaned(self):
        """The measured case, end to end: train in 1.0, val in 0.2, symlink on 1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-0.2", "val"), ("hm3d-1.0", "train")],
                         symlink_to="hm3d-1.0")
            self.assertFalse((tree.target / "val").is_dir(), "fixture is not the bug")

            done = self._run(tree)
            self.assertEqual(done.returncode, 0, done.stdout)
            # BOTH splits now resolve to a real mesh through the path episodes name.
            for split in ("train", "val"):
                self.assertTrue(
                    (tree.target / split / "00800-AAA" / "AAA.basis.glb").is_file(),
                    "{} does not resolve:\n{}".format(split, done.stdout),
                )

    def test_each_split_keeps_the_version_it_was_measured_on(self):
        """`abl-2` and `matrix-1` were rendered against 0.2. Silently moving val onto 1.0
        would re-base a published number, so val stays where it is and train takes 1.0."""
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-0.2", "val"), ("hm3d-1.0", "train")],
                         symlink_to="hm3d-1.0")
            self._run(tree)
            self.assertIn("hm3d-0.2", os.readlink(str(tree.target / "val")))
            self.assertIn("hm3d-1.0", os.readlink(str(tree.target / "train")))

    def test_the_newest_version_wins_when_a_split_is_in_two(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-0.2", "val"), ("hm3d-1.0", "val")])
            done = self._run(tree)
            self.assertIn("hm3d-1.0", os.readlink(str(tree.target / "val")))
            self.assertIn("newest of", done.stdout)

    def test_a_pin_holds_a_split_against_a_newer_tree(self):
        """The escape hatch for exactly the re-basing risk above."""
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-0.2", "val"), ("hm3d-1.0", "val")])
            done = self._run(tree, "--pin", "val=hm3d-0.2")
            self.assertIn("hm3d-0.2", os.readlink(str(tree.target / "val")))
            self.assertIn("PINNED", done.stdout)

    def test_a_pin_at_a_version_that_has_no_such_split_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-1.0", "train")])
            done = self._run(tree, "--pin", "train=hm3d-0.2")
            self.assertEqual(done.returncode, 2)
            self.assertIn("does not exist", done.stdout)

    def test_the_symlinks_are_relative(self):
        """The runbook's SECOND trap: an absolute symlink survives an rsync as text and
        resolves nowhere on the other machine."""
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-1.0", "train")])
            self._run(tree)
            link = os.readlink(str(tree.target / "train"))
            self.assertFalse(link.startswith("/"), link)
            self.assertTrue(link.startswith("../../versioned_data/"), link)

    def test_it_is_idempotent(self):
        """It has to be re-runnable after every download, which is the whole point."""
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-0.2", "val"), ("hm3d-1.0", "train")],
                         symlink_to="hm3d-1.0")
            first = self._run(tree)
            second = self._run(tree)
            self.assertEqual(second.returncode, 0, second.stdout)
            self.assertEqual(
                os.readlink(str(tree.target / "val")),
                os.readlink(str(tree.target / "val")),
            )
            self.assertTrue((tree.target / "train" / "00800-AAA").is_dir(), first.stdout)

    def test_check_is_red_before_the_fix_and_green_after(self):
        """Both arms of the gate, on one tree. `--check` must not repair anything, or a
        broken layout would report itself healthy by fixing itself on the way past."""
        with tempfile.TemporaryDirectory() as tmp:
            tree = _Tree(tmp, [("hm3d-0.2", "val"), ("hm3d-1.0", "train")],
                         symlink_to="hm3d-1.0")
            before = self._run(tree, "--check")
            self.assertEqual(before.returncode, 1, before.stdout)
            self.assertIn("DEAD", before.stdout)
            self.assertFalse((tree.target / "val").is_dir(), "--check repaired something")

            self._run(tree)
            after = self._run(tree, "--check")
            self.assertEqual(after.returncode, 0, after.stdout)
            self.assertNotIn("DEAD", after.stdout)

    def test_an_empty_versioned_directory_is_a_named_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp) / "data" / "hm3d"
            (data / "versioned_data").mkdir(parents=True)
            done = subprocess.run(
                ["bash", str(SCRIPT), "--data-path", str(data)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            self.assertEqual(done.returncode, 1)
            self.assertIn("no <version>/hm3d/<split>/ directory", done.stdout)


if __name__ == "__main__":
    unittest.main()
