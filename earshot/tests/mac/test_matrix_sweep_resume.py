"""`matrix_sweep.sh --resume`: what it skips, and what it must not rebuild.

Two halves, and a resume needs both. The CELL predicate decides a cell is already done;
without it a killed sweep re-runs every cell. That was survivable at 19 scenes x 4
conditions; the train pool is 73 scenes, so a run is 292 cells over about eight hours and
a crash in the last hour costs the other seven.

The STORE guard (`TestResumeReusesTheStore`, below) decides whether step 4 runs at all.
`matrix-2` is why it exists: the predicate worked perfectly and the run still never
reached a cell, because step 4 re-toured all 73 scenes first and died on the write.

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
import re
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


class TestResumeReusesTheStore(unittest.TestCase):
    """A resume must skip the tour, not redo it.

    `matrix-2` finished with one crashed cell (`not_heard_seen/LcAd9dhvVwh`), leaving that
    arm at n=288 over 72 scenes against the other three arms' 292 over 73. `--resume` is
    the documented recovery, and it never reached a cell: `--resume` implies `--force`, so
    it cleared "one directory is one run" at the top, then ran step 4 unconditionally,
    toured all 73 scenes for 10m 34s, printed `merged: 184 semantic row(s), 184 episodic
    row(s), 73 of 73 scene(s) complete`, and died in `write_pass_store`:

        FileExistsError: runs/matrix-2/prior/store.json already exists. One directory is
        one run: pass a fresh --run-dir, or --overwrite if replacing it is the intent.

    Exit 1, ten minutes for nothing, and the cell the run existed for never started.

    `--overwrite` is NOT the fix. It would delete the store the 291 finished cells
    consumed and leave an unchecked one in its place. A same-seed rebuild is LIKELY
    identical -- `prior_driver` calls `world.seed_navmesh(seed)` before every draw -- but
    likely is not the standard here: step 1 git-pulls and re-execs, so the code rebuilding
    the store is by construction not guaranteed to be the code that built it, and the
    climb loop's early exits decide how many draws the seeded sequence consumes. Reuse is
    provable, rebuild is hope.

    The condition is READ OUT OF THE SCRIPT and evaluated by bash over all four
    (resume, store-on-disk) states, for the same reason the predicate above is: a copy
    pasted here would pass forever after someone edited the driver.
    """

    GUARD = re.compile(r'^if (\[ "\$RESUME" = 1 \].*); then$', re.M)

    @classmethod
    def setUpClass(cls):
        cls.bash = shutil.which("bash")
        if cls.bash is None:
            raise unittest.SkipTest("no bash on this machine")
        cls.source = DRIVER.read_text()
        found = cls.GUARD.search(cls.source)
        if found is None:
            raise AssertionError(
                "the --resume guard on the prior pass is not in {}, so a resume re-tours "
                "every scene before it reaches a cell".format(DRIVER)
            )
        cls.condition = found.group(1)

    def _branch(self, *, resume, store):
        """Run the SHIPPED condition. REUSE means step 4 is skipped."""
        script = 'RESUME="$1"; STORE="$2"; if {}; then echo REUSE; else echo TOUR; fi'
        done = subprocess.run(
            [self.bash, "-c", script.format(self.condition), "_", resume, store],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        return done.stdout.decode().strip()

    def test_a_resume_over_an_existing_store_reuses_it(self):
        """The arm `matrix-2` failed: ten minutes of touring, then a FileExistsError."""
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "store.json"
            store.write_text('{"semantic": []}', encoding="utf-8")
            self.assertEqual(self._branch(resume="1", store=str(store)), "REUSE")

    def test_a_resume_with_no_store_still_tours(self):
        """The arm that keeps the guard from turning every resume into a no-op: with no
        store there is nothing to reuse, and the cells would run against a missing file."""
        with tempfile.TemporaryDirectory() as tmp:
            missing = pathlib.Path(tmp) / "store.json"
            self.assertEqual(self._branch(resume="1", store=str(missing)), "TOUR")

    def test_an_empty_store_file_is_not_a_store(self):
        """A zero-byte `store.json` is a crashed write, not a tour worth keeping."""
        with tempfile.TemporaryDirectory() as tmp:
            empty = pathlib.Path(tmp) / "store.json"
            empty.write_text("", encoding="utf-8")
            self.assertEqual(self._branch(resume="1", store=str(empty)), "TOUR")

    def test_a_normal_run_tours_even_with_a_store_present(self):
        """Without `--resume` a store on disk belongs to another run. Reusing it would
        pair a fresh tag's cells with a tour nothing asked for."""
        with tempfile.TemporaryDirectory() as tmp:
            store = pathlib.Path(tmp) / "store.json"
            store.write_text('{"semantic": []}', encoding="utf-8")
            self.assertEqual(self._branch(resume="0", store=str(store)), "TOUR")

    def test_the_tour_is_the_only_thing_the_guard_skips(self):
        """`prior_driver` moves inside the else. The coverage gate must NOT.

        matrix-1's D3: a store EXISTING is not a store COVERING the assignment, and a
        reused store has had less scrutiny than a fresh one, not more. NOT_RUN is never
        green, so the gate runs on both branches.
        """
        start = self.GUARD.search(self.source).start()
        block = self.source[start:self.source.index("\nfi\n", start)]
        self.assertIn(
            "python -m earshot.task.prior_driver", block,
            "the tour is not inside the guard, so a resume still re-tours",
        )
        self.assertNotIn(
            "matrix_audit --store", block,
            "the coverage gate is inside the guard, so a reused store would reach the "
            "cells unchecked",
        )

    def test_the_store_path_is_set_before_the_guard_reads_it(self):
        """`$STORE` was assigned AFTER the tour. Left there, the guard tests an empty
        string, never fires, and this whole class passes while the bug stands."""
        self.assertLess(
            self.source.index('STORE="$OUT_DIR/prior/store.json"'),
            self.GUARD.search(self.source).start(),
        )

    def test_the_gate_runs_after_both_branches(self):
        start = self.GUARD.search(self.source).start()
        self.assertGreater(
            self.source.index("python -m earshot.tools.matrix_audit --store"),
            self.source.index("\nfi\n", start),
        )


if __name__ == "__main__":
    unittest.main()
