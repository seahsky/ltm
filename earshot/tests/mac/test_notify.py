"""The carried notify trio actually dispatches, and actually reaches its notifier.

Ticket 10 carried `notify-run.sh` / `notify_email.py` / `test_notify_email.py` from
`scripts/` "as-is", and verbatim is what broke them. All three self-references were
written repo-relative — `$repo_root/scripts/notify-run.sh`, `$REPO_ROOT/scripts/
notify_email.py`, `Path(__file__).parent.parent` for `.env` — and the files moved from
one level deep to three. Every one of them then pointed at `earshot/tools/scripts/`,
which has never existed:

  * `nrun` dispatched at a missing file, printed `[nrun] detached (pid …)`, and the
    failure landed in a detached `.out` nobody reads. `box_gate.sh:50` documents
    exactly that invocation, so the box trip ticket 27 needed had no working launcher.
  * the emailer's absence was swallowed by the `|| true` that exists to keep a
    notifier failure from changing the wrapped exit code.
  * `.env` was read from a directory that has never held one, so a fully configured
    box declined to email and said only "not configured".

Three silent successes. Nothing caught them because `earshot/tools/notify/
test_notify_email.py` is a standalone assert script (`python …/test_notify_email.py`)
that `unittest discover earshot/tests/mac` does not collect — the trio was carried
with its tests and its tests were carried out of the suite. Its own wrapper smoke was
green throughout, because it asserts the wrapped exit code and the log file, which are
precisely the two things `|| true` protects.

So this is deliberately BEHAVIOURAL, not a grep for `scripts/`. A path assertion in
the shape "this string does not appear" would pass the day someone writes the same
bug with a different string; these run the thing and read what it did. Ticket 27.
"""

import os
import pathlib
import subprocess
import tempfile
import time
import unittest

from _interpreter import assert_interpreter  # noqa: F401

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
NOTIFY_DIR = REPO_ROOT / "earshot" / "tools" / "notify"
WRAPPER = NOTIFY_DIR / "notify-run.sh"
NOTIFIER = NOTIFY_DIR / "notify_email.py"

# The wrapper's `|| true` means a dead notifier shows up only as an interpreter
# complaint on stderr. That string IS the bug's signature.
MISSING_FILE_MARKERS = ("No such file or directory", "can't open file")


def _run_wrapper(*command, log_dir):
    env = dict(os.environ)
    env["NOTIFY_DISABLE"] = "1"  # no network from a unit test
    env["NOTIFY_RUN_LOG_DIR"] = str(log_dir)
    return subprocess.run(
        ["bash", str(WRAPPER), *command],
        capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
    )


class TestWrapperReachesItsNotifier(unittest.TestCase):
    def test_the_files_are_where_the_suite_thinks(self):
        """A red below means nothing if these moved and this test kept passing."""
        self.assertTrue(WRAPPER.is_file(), WRAPPER)
        self.assertTrue(NOTIFIER.is_file(), NOTIFIER)

    def test_execute_mode_does_not_reference_a_missing_file(self):
        with tempfile.TemporaryDirectory() as td:
            proc = _run_wrapper("echo", "ok-marker", log_dir=td)
        combined = proc.stdout + proc.stderr
        for marker in MISSING_FILE_MARKERS:
            self.assertNotIn(
                marker, combined,
                "the wrapper executed a path that does not exist — this is the "
                "carried-verbatim bug. Output:\n{}".format(combined),
            )

    def test_execute_mode_actually_reaches_the_notifier(self):
        """Absence of an error is not presence of the notifier.

        `NOTIFY_DISABLE=1` is honoured *inside* notify_email.py, so its skip line is
        proof the interpreter got there — which a missing file could never produce.
        """
        with tempfile.TemporaryDirectory() as td:
            proc = _run_wrapper("echo", "ok-marker", log_dir=td)
        self.assertIn("NOTIFY_DISABLE", proc.stdout + proc.stderr,
                      "notify_email.py never ran")

    def test_a_missing_notifier_is_loud_rather_than_swallowed(self):
        """The `|| true` protects the exit code; it must not protect a broken checkout.

        Planted, because the whole finding is that this case used to be silent.
        """
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            broken = td / "notify"
            broken.mkdir()
            # the wrapper but not its sibling notifier
            broken.joinpath("notify-run.sh").write_text(
                WRAPPER.read_text(encoding="utf-8"), encoding="utf-8")
            env = dict(os.environ)
            env["NOTIFY_RUN_LOG_DIR"] = str(td)
            proc = subprocess.run(
                ["bash", str(broken / "notify-run.sh"), "echo", "hi"],
                capture_output=True, text=True, env=env, cwd=str(REPO_ROOT),
            )
        self.assertEqual(proc.returncode, 0, "a notifier failure changed the exit code")
        self.assertIn("no notifier at", proc.stdout + proc.stderr,
                      "a missing notifier passed silently")

    def test_the_wrapped_exit_code_still_survives(self):
        """The property the `|| true` exists for, pinned beside the fix to it."""
        with tempfile.TemporaryDirectory() as td:
            proc = _run_wrapper("false", log_dir=td)
        self.assertEqual(proc.returncode, 1)


class TestNrunDispatches(unittest.TestCase):
    """`nrun` is how every long box run is launched, and it is fire-and-forget.

    Its failure mode is the worst shape available: it prints a pid, returns 0, and
    leaves the reason in a file the operator is not watching. So the assertion is on
    the detached output, not on nrun's own exit code.
    """

    def test_nrun_dispatches_to_a_file_that_exists(self):
        with tempfile.TemporaryDirectory() as td:
            script = pathlib.Path(td) / "drive.sh"
            script.write_text(
                '. "{wrapper}"\n'
                'nrun echo dispatch-marker\n'.format(wrapper=WRAPPER),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["NOTIFY_DISABLE"] = "1"
            env["NOTIFY_RUN_LOG_DIR"] = td
            subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env=env, cwd=str(REPO_ROOT), timeout=60)
            # `disown` takes the job out of the table, so the driver cannot `wait` on
            # it — polling is the shape of the thing being tested, not a workaround.
            deadline = time.time() + 30.0
            detached = ""
            while time.time() < deadline:
                outs = sorted(pathlib.Path(td).glob("nrun-*.out"))
                if outs:
                    detached = outs[0].read_text(encoding="utf-8")
                    if detached.strip():
                        break
                time.sleep(0.1)
            self.assertTrue(outs, "nrun wrote no detached log")
        for marker in MISSING_FILE_MARKERS:
            self.assertNotIn(
                marker, detached,
                "nrun dispatched at a path that does not exist — the detached log is "
                "the only place this shows. Contents:\n{}".format(detached),
            )
        self.assertIn("dispatch-marker", detached, "the command never ran")


class TestHermeticCopy(unittest.TestCase):
    """The tests above have a blind spot, and it is ticket 10 phase 2's whole argument.

    Written first, they were GREEN against a planted `$repo_root/scripts/notify-run.sh`
    — because `scripts/` has not been deleted yet, so the stale reference resolved to a
    working file. A reference to a doomed path is correct today and broken the instant
    the deletion commit lands, which no test run inside the present tree can see.

    So this runs the carried trio inside a skeleton holding ONLY what survives phase 3:
    the notify directory at its real depth, and nothing else. It is the hermeticity gate
    in miniature, on the one carried tool that is not Python and therefore invisible to
    the layering and import checks. Cheap enough to run on every Mac suite, which the box
    gate is not.
    """

    def _skeleton(self, td):
        root = pathlib.Path(td) / "hermetic_repo"
        dest = root / "earshot" / "tools" / "notify"
        dest.mkdir(parents=True)
        for name in ("notify-run.sh", "notify_email.py"):
            dest.joinpath(name).write_text(
                NOTIFY_DIR.joinpath(name).read_text(encoding="utf-8"), encoding="utf-8")
        root.joinpath(".env.example").write_text("", encoding="utf-8")
        return root

    def test_the_wrapper_runs_with_the_deleted_trees_absent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._skeleton(td)
            env = dict(os.environ)
            env["NOTIFY_DISABLE"] = "1"
            env["NOTIFY_RUN_LOG_DIR"] = str(pathlib.Path(td) / "logs")
            env.pop("REPO_ROOT", None)
            proc = subprocess.run(
                ["bash", str(root / "earshot/tools/notify/notify-run.sh"),
                 "echo", "ok-marker"],
                capture_output=True, text=True, env=env, cwd=str(root),
            )
        combined = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 0, combined)
        for marker in MISSING_FILE_MARKERS:
            self.assertNotIn(
                marker, combined,
                "the wrapper reaches outside what survives the reset. Output:\n{}"
                .format(combined),
            )
        self.assertIn("NOTIFY_DISABLE", combined, "notify_email.py never ran")

    def test_nrun_dispatches_with_the_deleted_trees_absent(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._skeleton(td)
            logs = pathlib.Path(td) / "logs"
            script = pathlib.Path(td) / "drive.sh"
            script.write_text(
                '. "{w}"\nnrun echo dispatch-marker\n'.format(
                    w=root / "earshot/tools/notify/notify-run.sh"),
                encoding="utf-8")
            env = dict(os.environ)
            env["NOTIFY_DISABLE"] = "1"
            env["NOTIFY_RUN_LOG_DIR"] = str(logs)
            env.pop("REPO_ROOT", None)
            subprocess.run(["bash", str(script)], capture_output=True, text=True,
                           env=env, cwd=str(root), timeout=60)
            deadline = time.time() + 30.0
            detached, outs = "", []
            while time.time() < deadline:
                outs = sorted(logs.glob("nrun-*.out")) if logs.is_dir() else []
                if outs:
                    detached = outs[0].read_text(encoding="utf-8")
                    if detached.strip():
                        break
                time.sleep(0.1)
            self.assertTrue(outs, "nrun wrote no detached log")
        self.assertIn(
            "dispatch-marker", detached,
            "nrun dispatched at a path that does not survive the reset:\n{}"
            .format(detached),
        )


class TestNotifierFindsTheRepoRoot(unittest.TestCase):
    """The third instance: `.env` lives at the repo root, not beside the notifier."""

    def test_default_repo_root_is_the_repo_root(self):
        proc = subprocess.run(
            ["python3", "-c",
             "import sys; sys.argv=['x','--exit-code','0','--log','/dev/null',"
             "'--command','c','--start-ts','0'];"
             "import importlib.util as u;"
             "s=u.spec_from_file_location('ne', {!r});"
             "m=u.module_from_spec(s); s.loader.exec_module(m);"
             "import argparse;"
             "print(m.__file__)".format(str(NOTIFIER))],
            capture_output=True, text=True, cwd=str(REPO_ROOT),
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        # The default is computed from __file__; assert the arithmetic directly, and
        # against a repo marker rather than a hardcoded depth, so a future move of the
        # notify directory fails here instead of silently reading a stranger's .env.
        default_root = NOTIFIER.resolve().parents[3]
        self.assertEqual(default_root, REPO_ROOT)
        self.assertTrue((default_root / ".env.example").is_file(),
                        "the notifier's default --repo-root holds no .env.example, so "
                        "it is not the repo root and .env will never be found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
