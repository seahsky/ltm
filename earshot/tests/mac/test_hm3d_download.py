"""`hm3d_download.sh`: the two things that make the shipped downloader unusable here.

`python -m habitat_sim.utils.datasets_download --help` does not print help on the box. It
prints `free(): invalid pointer` and dumps core, because it imports `habitat_sim` with no
`import torch` before it -- `test_habitat_import_order.py`'s measured case 1, exit 134.
It is an ABORT: nothing is raised, nothing catches it, and there is no Python-level
diagnostic to read.

Two invariants therefore have to hold in this wrapper's own text, and neither is the kind
a reader would notice going missing:

  * torch is imported BEFORE habitat_sim, in the same process;
  * `--no-replace` is always passed, because without it the downloader reads stdin and a
    detached `nrun` process has none.

Asserted against the script AS SHIPPED, and the ordering one is asserted positionally
rather than by presence -- an `import torch` that lands after the `runpy` call satisfies
a grep and aborts on the box exactly as before.
"""

import pathlib
import subprocess
import shutil
import unittest

from _interpreter import assert_interpreter  # noqa: F401

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "tools" / "hm3d_download.sh"


class TestTheImportOrderIsHeldInTheScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text()

    def test_torch_is_imported_before_habitat_sim(self):
        # Anchored to the start of a line, so the header's own prose about `import torch`
        # cannot satisfy this. The first draft of this test matched the comment and passed
        # with the real import moved BELOW the runpy call -- the exact bug it guards.
        torch_at = self.source.index("\nimport torch")
        habitat_at = self.source.index("habitat_sim.utils.datasets_download\", run_name")
        self.assertLess(
            torch_at, habitat_at,
            "torch must be imported before habitat_sim is loaded, or the process aborts "
            "with free(): invalid pointer and no Python-level diagnostic",
        )

    def test_the_torch_import_is_marked_load_bearing(self):
        """`ruff`'s F401 exists to delete an unused import, and this one must survive it.
        The `noqa` is the only thing standing between the box and exit 134."""
        line = next(
            row for row in self.source.splitlines() if row.strip().startswith("import torch")
        )
        self.assertIn("noqa", line)
        self.assertIn("habitat_sim", line)

    def test_it_runs_the_shipped_downloader_rather_than_reimplementing_it(self):
        """A hand-rolled fetch would drift from habitat-sim's own uid list and checksums.
        This wraps the module and lets its argparse and entry point be the ones that run."""
        self.assertIn('runpy.run_module("habitat_sim.utils.datasets_download"', self.source)
        self.assertIn('run_name="__main__"', self.source)

    def test_no_replace_is_always_passed(self):
        """Without it the downloader asks "Replace versioned data?" on stdin, and a
        detached process has none: OSError [Errno 9], about 21 seconds in."""
        self.assertIn("--no-replace", self.source)
        self.assertNotIn("${NO_REPLACE", self.source)


class TestTheCredentialGate(unittest.TestCase):
    """`--list` and `--help` must work before the Matterport agreement is signed; a real
    download must not start without a token and then fail deep inside habitat-sim."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("bash") is None:
            raise unittest.SkipTest("no bash on this machine")

    def _run(self, *args):
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        )

    def test_a_download_without_credentials_stops_before_python(self):
        done = self._run("--uids", "hm3d_train_full")
        self.assertEqual(done.returncode, 2)
        self.assertIn(b"MATTERPORT_TOKEN_ID", done.stdout)
        # The agreement is the actual blocker, so the message says where to get one.
        self.assertIn(b"matterport.com", done.stdout)


if __name__ == "__main__":
    unittest.main()
