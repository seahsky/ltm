"""A leak is green-then-red, and nothing else is.

The gate's first box run on the RACE V100 stopped with "the box suite is red without the
old trees — that is a leak". Two of 45 tests had failed on `clap_instantiable`, which
loads `laion/clap-htsat-unfused` through transformers and touches no path the reset
deletes. The run could not have known that, because it had one arm: a red without the old
trees says nothing until you know the colour with them.

`compare()` is the fix and it is pure, so every case below is Mac-testable — including
the two the box actually produced, which is the point of writing them down rather than
trusting the next reader to remember which failure was which.
"""

import json
import pathlib
import subprocess
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.task.smoke import CriterionStatus, _hermeticity
from earshot.tools.reset_manifest import build_record, delete_paths
from earshot.tools.suite_result import compare, run_suite

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def result(**outcomes):
    return {"outcomes": dict(outcomes), "n_tests": len(outcomes)}


class TestCompare(unittest.TestCase):
    def test_green_then_red_is_a_leak(self):
        v = compare(result(a="pass", b="pass"), result(a="pass", b="fail"))
        self.assertEqual(v["leaks"], ["b"])
        self.assertEqual(v["pre_existing"], [])

    def test_red_on_both_sides_is_not_a_leak(self):
        """The box's actual case: clap_instantiable was already failing."""
        v = compare(result(a="pass", clap="fail"), result(a="pass", clap="fail"))
        self.assertEqual(v["leaks"], [])
        self.assertEqual(v["pre_existing"], ["clap"])

    def test_an_error_counts_as_red_on_both_sides(self):
        """The box produced one `fail` and one `error` from the same cause."""
        v = compare(result(a="error"), result(a="error"))
        self.assertEqual(v["pre_existing"], ["a"])
        self.assertEqual(v["leaks"], [])

    def test_an_import_failure_only_without_the_trees_is_a_leak(self):
        """Discovery turns an unimportable module into a `_FailedTest` error."""
        v = compare(result(mod="pass"), result(mod="error"))
        self.assertEqual(v["leaks"], ["mod"])

    def test_red_then_green_is_reported_not_ignored(self):
        v = compare(result(a="fail"), result(a="pass"))
        self.assertEqual(v["recovered"], ["a"])
        self.assertEqual(v["leaks"], [])

    def test_a_skip_is_not_a_failure(self):
        v = compare(result(a="pass"), result(a="skip"))
        self.assertEqual(v["leaks"], [])

    def test_different_test_sets_are_not_comparable(self):
        v = compare(result(a="pass"), result(b="pass"))
        self.assertFalse(v["comparable"])
        self.assertEqual(v["vanished"], ["a"])
        self.assertEqual(v["appeared"], ["b"])

    def test_empty_is_not_comparable(self):
        """A suite that collected nothing must not read as 'no leaks'."""
        self.assertFalse(compare(result(), result())["comparable"])


class TestRunSuite(unittest.TestCase):
    """`run_suite` against a throwaway suite, so the recorder is exercised not assumed."""

    def test_outcomes_are_recorded_by_id(self):
        with tempfile.TemporaryDirectory() as td:
            pathlib.Path(td, "test_sample.py").write_text(
                "import unittest\n"
                "class T(unittest.TestCase):\n"
                "    def test_ok(self): pass\n"
                "    def test_bad(self): self.fail('no')\n"
                "    def test_boom(self): raise RuntimeError('x')\n"
                "    @unittest.skip('why') \n"
                "    def test_skipped(self): pass\n",
                encoding="utf-8",
            )
            payload = run_suite(td)
        by_name = {k.rsplit(".", 1)[-1]: v for k, v in payload["outcomes"].items()}
        self.assertEqual(by_name["test_ok"], "pass")
        self.assertEqual(by_name["test_bad"], "fail")
        self.assertEqual(by_name["test_boom"], "error")
        self.assertEqual(by_name["test_skipped"], "skip")
        self.assertFalse(payload["green"])

    def test_an_unimportable_module_lands_as_an_error(self):
        """The leak signal itself: a module that only imports with the old trees there."""
        with tempfile.TemporaryDirectory() as td:
            pathlib.Path(td, "test_broken.py").write_text(
                "import a_module_that_is_not_here\n", encoding="utf-8")
            payload = run_suite(td)
        self.assertEqual(payload["n_tests"], 1)
        self.assertEqual(list(payload["outcomes"].values()), ["error"])


class TestCriterionNineReadsTheComparison(unittest.TestCase):
    def _record(self, box):
        blob = {"when": "before", "at": 0.0, "checked": list(delete_paths()),
                "still_present": [], "complete": True}
        return build_record(run_dir="runs/h", before=dict(blob), after=dict(blob),
                            box_compare=box)

    def test_a_leak_fails_the_criterion(self):
        c = _hermeticity(self._record(compare(result(a="pass"), result(a="fail"))),
                         "runs/h")
        self.assertEqual(c.status, CriterionStatus.FAIL)
        self.assertIn("pass with the old trees", c.detail)

    def test_a_pre_existing_failure_passes_but_says_so(self):
        """The box's case. It is a sick environment, not a hermeticity failure."""
        c = _hermeticity(self._record(compare(result(clap="fail"), result(clap="fail"))),
                         "runs/h")
        self.assertEqual(c.status, CriterionStatus.PASS, c.detail)
        self.assertIn("pre-existing", c.detail)
        self.assertIn("clap", c.detail)

    def test_an_incomparable_pair_fails(self):
        c = _hermeticity(self._record(compare(result(a="pass"), result(b="pass"))),
                         "runs/h")
        self.assertEqual(c.status, CriterionStatus.FAIL)
        self.assertIn("absence of evidence", c.detail)

    def test_no_box_comparison_still_passes_on_absence_alone(self):
        """--skip-box-tests is allowed; the criterion's subject is the delete set."""
        c = _hermeticity(self._record(None), "runs/h")
        self.assertEqual(c.status, CriterionStatus.PASS, c.detail)


class TestCli(unittest.TestCase):
    def _cli(self, *args, cwd=None):
        import os
        return subprocess.run(
            ["python3", "-m", "earshot.tools.suite_result", *args],
            capture_output=True, text=True, cwd=str(cwd or REPO_ROOT),
            env={"PYTHONPATH": str(REPO_ROOT), "PATH": os.environ["PATH"]},
        )

    def test_compare_exit_codes_are_the_three_verdicts(self):
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            def write(name, payload):
                td.joinpath(name).write_text(json.dumps(payload), encoding="utf-8")
                return str(td / name)

            green = write("green.json", result(a="pass"))
            leaked = write("leak.json", result(a="fail"))
            other = write("other.json", result(b="pass"))

            self.assertEqual(self._cli("--compare", green, green).returncode, 0)
            self.assertEqual(self._cli("--compare", green, leaked).returncode, 1)
            self.assertEqual(self._cli("--compare", green, other).returncode, 2)
            # red on both sides is exit 0, and says why
            both = write("both.json", result(a="fail"))
            proc = self._cli("--compare", leaked, both)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("pre-existing", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
