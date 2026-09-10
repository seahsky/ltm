"""`matrix_sweep.sh --resume`: the predicate that decides a cell is already done.

Without it a killed sweep re-runs every cell. That was survivable at 19 scenes x 4
conditions; the train pool is 73 scenes, so a run is 292 cells over about eight hours and
a crash in the last hour costs the other seven.

The predicate is EXTRACTED FROM THE SCRIPT and run by bash, the same way
`test_ablation_sweep_driver.py` handles `is_zero_yield` and for the same reason: a copy
pasted into this file would pass forever after someone edited the driver.

Both arms throughout (ADR-0014). A finished cell must be recognised, and an unfinished one
must NOT be -- a predicate that answered "skip" to everything would turn a resumed run
green by never running anything at all, which is the exact failure this file exists to
make impossible.
"""

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

DRIVER = pathlib.Path(__file__).resolve().parents[2] / "tools" / "matrix_sweep.sh"


def extract_function(source: str, name: str) -> str:
    """The shell text of `name`, from its opening line to the closing brace."""
    lines = source.splitlines()
    opener = "{}() {{".format(name)
    for start, line in enumerate(lines):
        if line.startswith(opener):
            for end in range(start + 1, len(lines)):
                if lines[end] == "}":
                    return "\n".join(lines[start:end + 1])
            raise AssertionError("{} in {} is never closed".format(name, DRIVER))
    raise AssertionError("{} is not defined in {}".format(name, DRIVER))


class TestIsFinishedCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which("bash")
        if cls.bash is None:
            raise unittest.SkipTest("no bash on this machine")
        cls.body = extract_function(DRIVER.read_text(), "is_finished_cell")
        cls.python = shutil.which("python") or shutil.which("python3")
        if cls.python is None:
            raise unittest.SkipTest("no python on PATH for the predicate to call")

    def _ask(self, run_dir):
        """Run the SHIPPED predicate against a directory. True means 'already finished'."""
        script = "{}\nPATH={}:$PATH\nis_finished_cell \"$1\"\n".format(
            self.body, os.path.dirname(self.python)
        )
        done = subprocess.run(
            [self.bash, "-c", script, "_", str(run_dir)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return done.returncode == 0

    def _cell(self, tmp, payload=None, name="cell"):
        run_dir = pathlib.Path(tmp) / name
        run_dir.mkdir()
        if payload is not None:
            (run_dir / "summary.json").write_text(payload, encoding="utf-8")
        return run_dir

    def test_a_finished_cell_is_recognised(self):
        with tempfile.TemporaryDirectory() as tmp:
            cell = self._cell(tmp, json.dumps({"n_episodes": 15, "scene": "s"}))
            self.assertTrue(self._ask(cell))

    def test_a_zero_yield_cell_counts_as_finished(self):
        """It wrote a real summary saying the scene placed no episode. That is a measured
        fact about HM3D, not work left to redo -- and re-running it would produce the
        same nothing at the same cost."""
        with tempfile.TemporaryDirectory() as tmp:
            cell = self._cell(tmp, json.dumps({"n_episodes": 0}))
            self.assertTrue(self._ask(cell))

    def test_a_cell_that_never_ran_is_not_finished(self):
        """The arm that keeps a resumed run from skipping everything."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self._ask(self._cell(tmp)))

    def test_a_missing_directory_is_not_finished(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self._ask(pathlib.Path(tmp) / "never-created"))

    def test_a_truncated_summary_is_not_finished(self):
        """`report/` writes atomically, so this should not occur -- and if the guarantee
        ever slips, a half-written file must read as work to redo rather than as a cell
        that finished."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self._ask(self._cell(tmp, '{"n_episodes": 1')))

    def test_a_summary_without_the_episode_count_is_not_finished(self):
        """Valid JSON is not the test. A summary that parses but carries no `n_episodes`
        is not a summary this sweep wrote."""
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(self._ask(self._cell(tmp, json.dumps({"scene": "s"}))))


class TestTheResumeFlagIsWiredIn(unittest.TestCase):
    """The predicate is useless if the loop never asks it, or if a re-exec drops the flag."""

    @classmethod
    def setUpClass(cls):
        cls.source = DRIVER.read_text()

    def test_the_cell_loop_asks_the_predicate(self):
        self.assertIn('if [ "$RESUME" = 1 ] && is_finished_cell "$run_dir"; then',
                      self.source)

    def test_resume_implies_force_so_the_directory_is_reusable(self):
        """`--resume` on its own would trip "one directory is one run" before the first
        cell, which is the check it exists to work with rather than around."""
        self.assertIn("--resume)         RESUME=1; FORCE=1;", self.source)

    def test_the_flag_survives_the_self_update_re_exec(self):
        start = self.source.index('exec bash "$0"')
        end = self.source.index("\nelse", start)
        self.assertIn("--resume", self.source[start:end])

    def test_the_count_reaches_the_operator_and_the_provenance(self):
        """A resumed directory's episodes come from more than one invocation, and a reader
        who does not know that reads one wall clock for all of them."""
        self.assertIn('echo "resumed_cells:  $RESUMED"', self.source)
        self.assertIn("RESUMED: $RESUMED cell(s) were already finished", self.source)


if __name__ == "__main__":
    unittest.main()
