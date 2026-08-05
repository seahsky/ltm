"""The gate puts the repo back — including when it fails with the repo taken apart.

`hermeticity_gate.sh` moves 163 + 30 + 51 + 8 tracked files out of the repository and
back. The run in the middle needs the box; **the move and the restore do not**, and
ADR-0014 is explicit that an our-logic assertion which reaches for the real artefact is a
seam defect rather than a box test. So the script grew `--dry-run` (move, verify, restore,
stop) and these drive it against a scratch git repo.

The test that matters is the second one. A restore that works on the happy path is worth
little: the failure mode is a gate that dies half way — a red box suite, a Ctrl-C, a
smoke that raises — and leaves the tree missing a directory the operator then commits
around. `--self-test-abort` is the forced-failure arm ADR-0014 requires: it exits at the
one moment the repo is disassembled, and the assertion is that the EXIT trap put
everything back and `git status` is clean.

The happy path is also checked for **vacuity**: everything being back at the end is what
a script that moved nothing would also produce, so the pre-run verification blob is read
back and required to name the whole delete set as absent. That the files went away is the
claim; that they returned is only the safety property.
"""

import json
import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.tools.reset_manifest import delete_paths

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
GATE = REPO_ROOT / "earshot" / "tools" / "hermeticity_gate.sh"


class GateHarness:
    """A throwaway git repo shaped like this one, with the gate script inside it."""

    def __init__(self, tmp):
        self.tmp = pathlib.Path(tmp)
        self.repo = self.tmp / "repo"
        self.hold = self.tmp / "hold"
        (self.repo / "earshot" / "tools").mkdir(parents=True)
        shutil.copy(str(GATE), str(self.repo / "earshot" / "tools" / GATE.name))
        for rel in delete_paths():
            target = self.repo / rel
            if "." in target.name and not target.name.endswith(("_memory", "msc")):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("doomed\n", encoding="utf-8")
            else:
                target.mkdir(parents=True, exist_ok=True)
                (target / "content.py").write_text("doomed = True\n", encoding="utf-8")
        (self.repo / "survivor.md").write_text("kept\n", encoding="utf-8")
        self._git("init", "-q")
        self._git("add", "-A")
        self._git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")

    def _git(self, *args):
        return subprocess.run(["git", *args], cwd=str(self.repo), capture_output=True,
                              text=True)

    def run(self, *flags):
        env = dict(os.environ)
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["PYTHON"] = "python3"
        env["HERMETICITY_HOLD_DIR"] = str(self.hold)
        env["HOME"] = str(self.tmp)
        return subprocess.run(
            ["bash", str(self.repo / "earshot/tools/hermeticity_gate.sh"),
             "--no-pull", *flags],
            capture_output=True, text=True, env=env, cwd=str(self.repo), timeout=300,
        )

    def porcelain(self):
        return self._git("status", "--porcelain").stdout.strip()

    def missing(self):
        return [p for p in delete_paths() if not (self.repo / p).exists()]


class TestRestore(unittest.TestCase):
    def test_dry_run_moves_everything_out_and_puts_it_back(self):
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            proc = h.run("--dry-run")
            missing, dirt = h.missing(), h.porcelain()
            before = json.loads((h.hold / "verify-before.json").read_text(encoding="utf-8"))
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        # Not vacuous: this is the claim. Everything-is-back is only the safety property.
        self.assertTrue(before["complete"])
        self.assertEqual(sorted(before["checked"]), sorted(delete_paths()))
        self.assertEqual(before["still_present"], [])
        self.assertEqual(missing, [], "the gate did not restore these")
        self.assertEqual(dirt, "", "the tree is not back the way it was")

    def test_an_abort_with_the_repo_taken_apart_still_restores(self):
        """The arm that matters. Exits 42 mid-disassembly; the EXIT trap must repair it."""
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            proc = h.run("--dry-run", "--self-test-abort")
            missing, dirt = h.missing(), h.porcelain()
            out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 42, out)
        self.assertIn("moved", out)  # it really did take the repo apart first
        self.assertEqual(missing, [], "an aborted gate left the repo missing files")
        self.assertEqual(dirt, "", "an aborted gate left the tree dirty")

    def test_a_dirty_tree_is_refused_before_anything_moves(self):
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            (h.repo / "survivor.md").write_text("edited\n", encoding="utf-8")
            proc = h.run("--dry-run")
            missing = h.missing()
            still_edited = (h.repo / "survivor.md").read_text(encoding="utf-8")
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("dirty", proc.stdout + proc.stderr)
        self.assertEqual(missing, [], "it moved files despite refusing to run")
        self.assertEqual(still_edited, "edited\n", "the refusal cost an uncommitted edit")

    def test_a_repo_left_taken_apart_is_named_not_called_dirty(self):
        """The wreckage a SIGKILLed run leaves, and the one message that must not fire.

        A dead pod or a closed session takes the process without running the EXIT trap,
        so the repo stays disassembled: tracked files missing, nothing else wrong. The
        generic dirty-tree refusal would tell the operator to commit or stash, and
        stashing here records the deletion of every moved file.
        """
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            shutil.rmtree(h.repo / "embodied_memory")
            (h.repo / "README_MSC_EVAL.md").unlink()
            proc = h.run("--dry-run")
            out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, out)
        self.assertIn("tracked file(s) are missing", out)
        self.assertIn("DO NOT stash", out)
        self.assertIn("ls-files --deleted", out)
        self.assertNotIn("Commit or stash first", out)

    def test_a_second_gate_is_refused_while_one_is_running(self):
        """Two at once move the same paths twice and restore into each other."""
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            # This test process is alive, so its pid is a live lock by construction.
            pathlib.Path(td, ".earshot-hermeticity.lock").write_text(
                str(os.getpid()), encoding="utf-8")
            proc = h.run("--dry-run")
            missing = h.missing()
            out = proc.stdout + proc.stderr
        self.assertEqual(proc.returncode, 1, out)
        self.assertIn("already running", out)
        self.assertEqual(missing, [], "the refused run moved files anyway")

    def test_a_stale_lock_alone_does_not_block(self):
        """A dead pid's lock is a note, not a refusal — the tree is the real subject."""
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            pathlib.Path(td, ".earshot-hermeticity.lock").write_text(
                "999999", encoding="utf-8")
            proc = h.run("--dry-run")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("stale lock", proc.stdout)

    def test_a_dry_run_does_not_claim_a_verdict(self):
        """The trap judges only a run that happened; --dry-run runs no episode.

        The first green box run exited 0 having never produced the nine-point verdict,
        so the trap judges now. This pins the other direction: it must not judge when
        there is nothing to judge, which would be a verdict about a previous run.
        """
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            out = h.run("--dry-run").stdout
        self.assertNotIn("the nine criteria", out)

    def test_the_recovery_command_is_in_the_log_before_the_move(self):
        """Layer 3 of the restore: a SIGKILL leaves no trap, only what was printed."""
        with tempfile.TemporaryDirectory() as td:
            h = GateHarness(td)
            proc = h.run("--dry-run")
            out = proc.stdout
        self.assertIn("git checkout --", out)
        for path in delete_paths():
            self.assertIn(path, out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
