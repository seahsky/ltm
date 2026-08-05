"""The phase-3 delete list is checked against the tree it describes.

`earshot/tools/reset_manifest.py` exists because ticket 27 names the delete set in prose,
and the deletion commit is irreversible. Prose cannot be checked. These assertions are the
reason the list is code:

  * every entry still exists — an entry for an absent path is an inert pin, which is
    ticket 20's own correction to the walker's exclusion list, in the one place where an
    inert pin means a gate covering less than it says;
  * the tracked-file counts still match, so the reset is re-audited by a human when the
    tree moves rather than widening silently;
  * the named survivors are outside every delete entry, because the failure mode of a
    mis-typed delete path is not a red test, it is `docs/` going with `dialogue_memory/`.

And one seam: `task/smoke.py` duplicates the record's name and schema string because
`task/` may not import `tools/`. Duplicated constants are how two languages drift, so the
duplication is pinned here rather than trusted.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.task.smoke import (
    HERMETICITY_NAME,
    HERMETICITY_SCHEMA,
    CriterionStatus,
)
from earshot.task.smoke import _hermeticity  # the criterion under test, directly
from earshot.tools.reset_manifest import (
    DELETE_SET,
    KEEP_PINS,
    RECORD_NAME,
    build_record,
    delete_paths,
    verify,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _git(*args):
    return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True,
                          text=True)


class TestManifestDescribesThisTree(unittest.TestCase):
    def test_git_is_available(self):
        """Absence is red, not skipped — a skipped audit reads exactly like a passing one."""
        proc = _git("rev-parse", "--is-inside-work-tree")
        self.assertEqual(proc.returncode, 0,
                         "no git here, so the counts below cannot be audited")

    def test_every_entry_exists(self):
        missing = [e.path for e in DELETE_SET if not (REPO_ROOT / e.path).exists()]
        self.assertEqual(
            missing, [],
            "the manifest names paths that are already gone: {}. Either the reset has "
            "partly happened, or the list is describing a tree that no longer exists."
            .format(missing),
        )

    def test_tracked_file_counts_still_match(self):
        drifted = []
        for entry in DELETE_SET:
            proc = _git("ls-files", "--", entry.path)
            count = len([ln for ln in proc.stdout.splitlines() if ln.strip()])
            if count != entry.tracked_files:
                drifted.append((entry.path, entry.tracked_files, count))
        self.assertEqual(
            drifted, [],
            "tracked-file counts moved since the audit (path, audited, now): {}. This is "
            "not a test to update in passing — the deletion is irreversible, so look at "
            "what arrived or left first.".format(drifted),
        )

    def test_survivors_are_outside_every_delete_entry(self):
        doomed = [pathlib.Path(p) for p in delete_paths()]
        for keep in KEEP_PINS:
            self.assertTrue((REPO_ROOT / keep).exists(), "keep pin is missing: " + keep)
            kept = pathlib.Path(keep)
            for d in doomed:
                self.assertFalse(
                    kept == d or d in kept.parents,
                    "{} would be deleted with {}".format(keep, d),
                )

    def test_the_carried_notify_trio_is_not_in_the_delete_set(self):
        """`scripts/` goes wholesale; the trio survives only because it was copied out.

        Ticket 27's phrasing ("except the three notify files already carried") reads as an
        exception to the delete if you are skimming. It is not — it names why nothing is
        lost. This asserts the copy that survives is the one at the new location.
        """
        self.assertIn("scripts", delete_paths())
        for name in ("notify-run.sh", "notify_email.py", "test_notify_email.py"):
            self.assertTrue((REPO_ROOT / "earshot/tools/notify" / name).is_file(), name)


class TestVerify(unittest.TestCase):
    def test_absent_everywhere_is_complete(self):
        with tempfile.TemporaryDirectory() as td:
            evidence = verify(td, when="before")
        self.assertTrue(evidence["complete"])
        self.assertEqual(evidence["still_present"], [])
        self.assertEqual(list(evidence["checked"]), list(delete_paths()))

    def test_one_survivor_is_named_and_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            (pathlib.Path(td) / "scripts").mkdir()
            evidence = verify(td, when="after")
        self.assertFalse(evidence["complete"])
        self.assertEqual(evidence["still_present"], ["scripts"])

    def test_the_real_repo_is_not_hermetic_yet(self):
        """The delete set is present right now, so a `--verify-absent` here must fail.

        Pins the direction of the check. A verifier that returned "complete" against the
        live tree would make the gate pass without moving anything.
        """
        self.assertFalse(verify(REPO_ROOT, when="now")["complete"])


class TestRecord(unittest.TestCase):
    def _blob(self, complete=True, still=()):
        return {"when": "before", "at": 0.0, "checked": list(delete_paths()),
                "still_present": list(still), "complete": complete}

    def test_complete_needs_both_halves(self):
        both = build_record(run_dir="runs/x", before=self._blob(), after=self._blob())
        self.assertTrue(both["complete"])
        half = build_record(run_dir="runs/x", before=self._blob(),
                            after=self._blob(complete=False, still=["scripts"]))
        self.assertFalse(half["complete"])

    def test_schema_and_name_agree_with_the_judge(self):
        """The deliberate duplication across the `task/` <- JSON -> `tools/` boundary."""
        self.assertEqual(RECORD_NAME, HERMETICITY_NAME)
        record = build_record(run_dir="runs/x", before=self._blob(), after=self._blob())
        self.assertEqual(record["schema"], HERMETICITY_SCHEMA)


class TestCriterionNine(unittest.TestCase):
    """Ticket 19's third row: given a bad record, does the gate go red?"""

    def _record(self, **over):
        blob = {"when": "before", "at": 0.0, "checked": list(delete_paths()),
                "still_present": [], "complete": True}
        record = build_record(run_dir="runs/hermetic-1", before=dict(blob),
                              after=dict(blob))
        record.update(over)
        return record

    def test_absent_record_is_not_run_not_pass(self):
        c = _hermeticity(None, "runs/ordinary")
        self.assertEqual(c.status, CriterionStatus.NOT_RUN)

    def test_a_good_record_passes(self):
        c = _hermeticity(self._record(), "runs/hermetic-1")
        self.assertEqual(c.status, CriterionStatus.PASS, c.detail)

    def test_a_record_from_another_run_fails(self):
        c = _hermeticity(self._record(), "runs/some-other-run")
        self.assertEqual(c.status, CriterionStatus.FAIL)

    def test_a_hand_set_complete_flag_does_not_win(self):
        """The top-level flag is recomputed from the halves; it is one edit away."""
        record = self._record()
        record["before"]["still_present"] = ["scripts"]
        record["before"]["complete"] = False
        record["complete"] = True
        c = _hermeticity(record, "runs/hermetic-1")
        self.assertEqual(c.status, CriterionStatus.FAIL)
        self.assertIn("scripts", c.detail)

    def test_verifying_a_subset_fails(self):
        """Moving out one path and recording that is the obvious way to fake this."""
        record = self._record()
        record["before"]["checked"] = ["data/msc"]
        c = _hermeticity(record, "runs/hermetic-1")
        self.assertEqual(c.status, CriterionStatus.FAIL)
        self.assertIn("never checked", c.detail)

    def test_an_empty_manifest_fails(self):
        c = _hermeticity(self._record(entries=[]), "runs/hermetic-1")
        self.assertEqual(c.status, CriterionStatus.FAIL)

    def test_an_unknown_schema_fails(self):
        c = _hermeticity(self._record(schema="something/2"), "runs/hermetic-1")
        self.assertEqual(c.status, CriterionStatus.FAIL)

    def test_a_missing_half_fails(self):
        record = self._record()
        record.pop("after")
        c = _hermeticity(record, "runs/hermetic-1")
        self.assertEqual(c.status, CriterionStatus.FAIL)
        self.assertIn("after", c.detail)


class TestCliRoundTrip(unittest.TestCase):
    """The two halves the gate script actually calls, end to end, without a box."""

    def _cli(self, *args, cwd=None):
        return subprocess.run(
            ["python3", "-m", "earshot.tools.reset_manifest", *args],
            capture_output=True, text=True, cwd=str(cwd or REPO_ROOT),
            env={"PYTHONPATH": str(REPO_ROOT), "PATH": __import__("os").environ["PATH"]},
        )

    def test_verify_then_write_then_judge(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            before = self._cli("--verify-absent", "--when", "before", "--root", str(td))
            self.assertEqual(before.returncode, 0, before.stderr)
            (td / "before.json").write_text(before.stdout, encoding="utf-8")
            after = self._cli("--verify-absent", "--when", "after", "--root", str(td))
            (td / "after.json").write_text(after.stdout, encoding="utf-8")

            run_dir = td / "runs" / "hermetic-1"
            wrote = self._cli("--write-record", "--run-dir", str(run_dir),
                              "--before", str(td / "before.json"),
                              "--after", str(td / "after.json"))
            self.assertEqual(wrote.returncode, 0, wrote.stderr)

            record = json.loads((run_dir / RECORD_NAME).read_text(encoding="utf-8"))
            c = _hermeticity(record, str(run_dir))
        self.assertEqual(c.status, CriterionStatus.PASS, c.detail)

    def test_verify_absent_exits_nonzero_against_the_live_repo(self):
        proc = self._cli("--verify-absent", "--root", str(REPO_ROOT))
        self.assertEqual(proc.returncode, 1)
        self.assertIn("STILL PRESENT", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
