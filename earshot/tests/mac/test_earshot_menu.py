"""`earshot.sh` as a sourceable file, and its one coupling to the Python it reads.

The menu is meant to be `source`d — that is the only way "set the env up" can end with the
operator actually in `ss2`, since no child process can activate anything in its parent.
Sourcing puts every line of it in the caller's interactive shell, which turns two ordinary
shell habits into session-killers, and notify-run.sh's comment block records paying for
the first one already:

* **an `exit`** closes the operator's session instead of leaving the menu;
* **a `set -u`** turns their next typo'd variable into a closed session too.

Neither is visible when the file is executed, which is how it will usually be tried.

The third test is a drift guard on the seam: bash reads `format_plan()`'s output by
matching on `KEY`, so renaming a key in Python silently empties a field in the kill
confirmation — the prompt that says which run is about to be stopped.
"""

import pathlib
import re
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.nrun_tasks import NrunTask, format_plan

MENU = pathlib.Path(__file__).resolve().parents[3] / "earshot.sh"


class TestSourceSafety(unittest.TestCase):
    def setUp(self):
        self.text = MENU.read_text(encoding="utf-8")
        # Comments and quoted strings are stripped before the scan: the file *talks* about
        # `exit` and `set -u` at length, and a checker that cannot tell a mention from a
        # command would force the comments explaining the rule to be deleted to satisfy it.
        self.lines = [
            re.sub(r'"[^"]*"|\'[^\']*\'', '""', line)
            for line in self.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    def test_the_file_exists_and_is_executable(self):
        self.assertTrue(MENU.is_file(), "{} is missing".format(MENU))
        self.assertTrue(MENU.stat().st_mode & 0o111, "earshot.sh is not executable")

    def test_the_only_exit_is_the_one_guarded_by_not_being_sourced(self):
        """A sourced `exit` closes the operator's session. There may be exactly one, and
        it must sit in the executed branch at the bottom."""
        exits = [line.strip() for line in self.lines if re.search(r"\bexit\b", line)]
        self.assertEqual(
            exits, ["exit $?"],
            "every refusal must `return`; the sole `exit` is the executed-mode tail. "
            "Found:\n  " + "\n  ".join(exits),
        )

    def test_shell_options_are_only_set_when_not_sourced(self):
        """`set -u` leaked into an interactive shell makes the next typo fatal."""
        offenders = [
            line.strip() for line in self.lines
            if re.match(r"\s*set\s+-[uo]", line) and "_earshot_sourced ||" not in line
        ]
        self.assertEqual(
            offenders, [],
            "top-level `set` must be guarded by `_earshot_sourced ||`. Found:\n  "
            + "\n  ".join(offenders),
        )

    def test_sourcing_is_detected_the_way_notify_run_does_it(self):
        self.assertIn('[ "${BASH_SOURCE[0]}" != "${0}" ]', self.text)


class TestPlanSeam(unittest.TestCase):
    def test_every_key_the_plan_emits_is_read_by_the_menu(self):
        """Rename a key in Python and the kill confirmation silently loses a field."""
        task = NrunTask(pid=1, pgid=1, etimes=1, started="s", command="c", children=())
        keys = [line.split(" ", 1)[0] for line in format_plan(task).splitlines()]
        text = MENU.read_text(encoding="utf-8")
        missing = [k for k in keys if "{}) PLAN_".format(k) not in text]
        self.assertEqual(
            missing, [],
            "earshot.sh does not read these --plan keys: {}".format(missing),
        )
