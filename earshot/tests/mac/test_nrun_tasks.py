"""The nrun task scan, against captured `ps` output — the box runs it, the parsing doesn't.

Two failure modes pay for this file, and both are kill-shaped rather than display-shaped:

1. **A group kill that reaps the caller.** `nrun` does not `setsid`, so a task launched
   from a non-interactive shell shares its caller's process group. `kill -- -PGID` there
   is a kill on your login shell. The forced-failure arm below is exactly that row, and it
   must come back `tree`, never `group`.
2. **A kill that reports success and frees nothing.** Signalling the wrapper alone orphans
   the python child that holds the GPU, and the orphan then vanishes from the listing —
   the scan looks clean because the evidence left with it. So the tree fallback has to
   carry the descendants, and that is asserted rather than assumed.

The `ps` text here is the real shape (`PS_ARGS`), including the two-space day padding
`lstart` uses and a command line with spaces in it, because both are what the arity-based
split is for.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.nrun_tasks import (
    NrunTask,
    build_tasks,
    descendants,
    format_plan,
    group_kill_is_safe,
    kill_targets,
    parse_ps,
    resolve_log_path,
    scrape_exit_code,
    summarize_log,
    wrapped_command,
    wrapper_rows,
)

# pid ppid pgid etimes lstart(5 tokens) args
PS = """
    1     0     1 999999 Thu Aug  7 09:00:00 2026 /sbin/init
 4100  1000  4100   7600 Thu Aug  7 16:00:00 2026 bash /home/u/ltm/earshot/tools/notify/notify-run.sh python -m earshot --run-dir runs/t --n-episodes 1
 4101  4100  4100   7600 Thu Aug  7 16:00:00 2026 tee /home/u/ltm/runs/notify-earshot-20260807-160000.log
 4102  4100  4100   7600 Thu Aug  7 16:00:00 2026 python -m earshot --run-dir runs/t --n-episodes 1
 4180  4102  4100   7599 Thu Aug  7 16:00:01 2026 /usr/bin/nvidia-smi
 5200  5100  5100     42 Thu Aug  7 18:00:00 2026 bash /home/u/ltm/earshot/tools/notify/notify-run.sh bash earshot/tools/box_gate.sh
 5201  5200  5100     42 Thu Aug  7 18:00:00 2026 bash earshot/tools/box_gate.sh
 6000  1000  6000      3 Thu Aug  7 18:02:00 2026 bash ./earshot.sh
 6001  6000  6000      3 Thu Aug  7 18:02:00 2026 grep -F notify-run.sh
"""


class TestParsing(unittest.TestCase):
    def test_the_command_line_survives_its_spaces(self):
        rows = {r.pid: r for r in parse_ps(PS)}
        self.assertEqual(
            rows[4100].args,
            "bash /home/u/ltm/earshot/tools/notify/notify-run.sh python -m earshot "
            "--run-dir runs/t --n-episodes 1",
        )
        # `lstart` pads single-digit days to two columns ("Aug  7"); the arity split
        # re-joins on single spaces, so the stored value is normalised. Pinned as such —
        # it is displayed verbatim in the kill confirmation.
        self.assertEqual(rows[4100].started, "Thu Aug 7 16:00:00 2026")
        self.assertEqual((rows[4100].ppid, rows[4100].pgid, rows[4100].etimes),
                         (1000, 4100, 7600))

    def test_a_short_or_junk_line_is_skipped_not_guessed_at(self):
        self.assertEqual(parse_ps("garbage\n1 2 3\n\n"), [])

    def test_only_real_wrappers_match(self):
        """The menu and the grep that finds it both carry the string; neither is a task."""
        pids = [r.pid for r in wrapper_rows(parse_ps(PS))]
        self.assertEqual(pids, [4100, 5200])

    def test_a_bare_notify_run_with_no_command_is_not_a_task(self):
        """That is the usage-error path — nothing to report on and nothing to kill."""
        line = " 7000  1  7000  5 Thu Aug  7 18:00:00 2026 bash /x/notify-run.sh\n"
        self.assertEqual(wrapper_rows(parse_ps(line)), [])

    def test_the_wrapper_prefix_is_stripped_off_the_command(self):
        row = {r.pid: r for r in parse_ps(PS)}[4100]
        self.assertEqual(wrapped_command(row),
                         "python -m earshot --run-dir runs/t --n-episodes 1")

    def test_descendants_are_transitive(self):
        self.assertEqual(descendants(parse_ps(PS), 4100), (4101, 4102, 4180))


class TestTasks(unittest.TestCase):
    def tasks(self):
        return build_tasks(parse_ps(PS), resolve_log=lambda pid: "/runs/nrun-{}.out".format(pid))

    def test_oldest_first_because_that_is_finishing_order(self):
        self.assertEqual([t.pid for t in self.tasks()], [4100, 5200])

    def test_the_log_comes_from_the_injected_resolver(self):
        self.assertEqual(self.tasks()[0].log, "/runs/nrun-4100.out")

    def test_a_group_leader_is_killed_as_a_group(self):
        task = self.tasks()[0]
        self.assertTrue(group_kill_is_safe(task))
        self.assertEqual(kill_targets(task), ("group", (4100,)))

    def test_a_task_sharing_its_callers_group_is_never_group_killed(self):
        """The forced-failure arm: `nrun` from a non-interactive shell.

        pgid 5100 belongs to the CALLER, not to this wrapper. `kill -- -5100` would take
        that caller down with the task. Returning `group` here is the bug this test exists
        to catch, and it is invisible in any interactive-shell fixture.
        """
        task = self.tasks()[1]
        self.assertEqual((task.pid, task.pgid), (5200, 5100))
        self.assertFalse(group_kill_is_safe(task))
        mode, targets = kill_targets(task)
        self.assertEqual(mode, "tree")
        self.assertEqual(targets, (5200, 5201))

    def test_the_tree_fallback_carries_the_children_not_just_the_wrapper(self):
        """Killing the wrapper alone orphans the child holding the GPU — and hides it."""
        task = NrunTask(pid=9, pgid=8, etimes=1, started="s", command="c",
                        children=(10, 11))
        self.assertEqual(kill_targets(task), ("tree", (9, 10, 11)))

    def test_the_plan_is_line_oriented_and_keeps_the_command_intact(self):
        plan = dict(line.split(" ", 1) for line in format_plan(self.tasks()[0]).splitlines())
        self.assertEqual(plan["COMMAND"],
                         "python -m earshot --run-dir runs/t --n-episodes 1")
        self.assertEqual((plan["MODE"], plan["TARGETS"], plan["PID"]), ("group", "4100", "4100"))
        self.assertEqual(plan["ELAPSED"], "2h06m")


class TestLogResolution(unittest.TestCase):
    def test_a_real_file_is_the_log(self):
        self.assertEqual(
            resolve_log_path(1, readlink=lambda p: "/home/u/ltm/runs/nrun-x.out"),
            "/home/u/ltm/runs/nrun-x.out",
        )

    def test_a_pipe_a_tty_or_an_unreadable_proc_entry_is_not(self):
        def boom(_):
            raise OSError("no such process")

        self.assertIsNone(resolve_log_path(1, readlink=lambda p: "pipe:[12345]"))
        self.assertIsNone(resolve_log_path(1, readlink=lambda p: "/dev/pts/3"))
        self.assertIsNone(resolve_log_path(1, readlink=boom))


class TestFinishedLogs(unittest.TestCase):
    SENT = ("[notify] email sent to a@b.com "
            "(✅ [ltm] box_gate — exit 0 (2m 11s))")
    FAILED = ("[notify] email sent to a@b.com "
              "(❌ [ltm] yield_sweep — exit 1 (41m 2s))")

    def test_the_exit_code_is_read_off_the_notifier_line(self):
        self.assertEqual(scrape_exit_code("noise\n" + self.SENT), 0)
        self.assertEqual(scrape_exit_code("noise\n" + self.FAILED), 1)

    def test_an_unrecorded_status_is_unknown_and_never_ok(self):
        """notify-run.sh never prints the exit code; only the emailer does.

        So an unconfigured emailer leaves no status in the .out file, and the repo rule is
        that a thing which could not be evaluated is not green.
        """
        row = summarize_log("runs/nrun-x.out", 1.0, "ran some stuff\nand finished\n")
        self.assertIsNone(row["exit_code"])
        self.assertEqual(row["status"], "?")
        self.assertEqual(row["last_line"], "and finished")

    def test_the_last_notifier_line_wins(self):
        """A retry logs more than one; the final one is the run's verdict."""
        self.assertEqual(scrape_exit_code(self.FAILED + "\n" + self.SENT), 0)

    def test_a_bare_word_exit_in_program_output_is_not_mistaken_for_the_verdict(self):
        self.assertIsNone(scrape_exit_code("Traceback ...\nSystemExit: exit 3\n"))
