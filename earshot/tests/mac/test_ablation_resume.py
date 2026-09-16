"""`ablation_sweep.sh --resume`, and the `rm -rf` inside it.

A nineteen-scene two-arm sweep is 4h45m in one process tree, and a box that drops it at
hour four has cost a night. The sweep was always chunked -- one `python -m earshot` per
(arm, scene) -- so what `--resume` adds is only the ability not to redo the finished ones.

Two things here are worth a test rather than a reading. The first is `clear_scene_dir`,
which is an `rm -rf` built from an operator-supplied variable. The second is that
`--resume` survives the script's own re-exec: the sweep re-execs itself when a `git pull`
changes it, and a resume that turned back into a bare `--force` there would re-run
finished scenes into directories `write_episode` refuses to overwrite, after the pull,
with nobody awake.

The functions are extracted and run under real bash rather than read as text. The script
cannot be sourced -- it runs a sweep on sight -- which is why they are lifted out.
"""

import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

SWEEP = pathlib.Path(__file__).resolve().parents[2] / "tools" / "ablation_sweep.sh"


def extract(name):
    """One shell function's source, by name."""
    text = SWEEP.read_text()
    match = re.search(
        r"^{}\(\) \{{\n(?:.*?)^\}}".format(re.escape(name)), text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(
            "{}() is not in ablation_sweep.sh; the test cannot exercise a function that "
            "is not there, and a silently-skipped check is what ADR-0014 refuses".format(
                name
            )
        )
    return match.group(0)


def run_bash(script):
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True
    )


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)


class TestIsFinishedScene(Fixture):
    """The definition of "done": a summary.json that parses and carries n_episodes. It is
    the LAST artefact `run()` writes, so its presence means the episodes and the memory
    dump both landed."""

    def check(self, body):
        scene = pathlib.Path(self.root) / "dream" / "sceneA"
        scene.mkdir(parents=True)
        if body is not None:
            (scene / "summary.json").write_text(body)
        return run_bash(
            "{}\nis_finished_scene '{}'".format(extract("is_finished_scene"), scene)
        ).returncode

    def test_a_finished_scene_is_finished(self):
        self.assertEqual(self.check('{"n_episodes": 15}'), 0)

    def test_a_ZERO_YIELD_scene_counts_as_finished(self):
        """It cannot pose the task. Re-running it every resume would retry it forever."""
        self.assertEqual(self.check('{"n_episodes": 0}'), 0)

    def test_a_scene_with_no_summary_is_not_finished(self):
        self.assertNotEqual(self.check(None), 0)

    def test_a_TRUNCATED_summary_is_not_finished(self):
        """A process killed mid-write leaves half a JSON file, and half a record must not
        read as a whole one."""
        self.assertNotEqual(self.check('{"n_epi'), 0)

    def test_a_summary_without_n_episodes_is_not_finished(self):
        self.assertNotEqual(self.check('{"scene": "sceneA"}'), 0)


class TestClearSceneDirRefusesWhatItShould(Fixture):
    """**THE `rm -rf` ARM.** Every refusal is asserted, because the failure mode of this
    function is not a wrong answer."""

    def run_clear(self, target, out_dir=None):
        out = out_dir if out_dir is not None else self.root
        return run_bash(
            'OUT_DIR="{}"\n{}\nclear_scene_dir "{}"'.format(
                out, extract("clear_scene_dir"), target)
        )

    def test_it_clears_an_arm_scene_directory_under_the_run(self):
        scene = pathlib.Path(self.root) / "dream" / "sceneA"
        (scene / "episodes").mkdir(parents=True)
        (scene / "episodes" / "ep0000.audit.json").write_text("{}")

        result = self.run_clear(str(scene))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(scene.exists())
        print(result.stdout.strip())

    def test_it_refuses_a_path_containing_dotdot(self):
        scene = pathlib.Path(self.root) / "dream" / "sceneA"
        scene.mkdir(parents=True)

        result = self.run_clear("{}/dream/../dream/sceneA".format(self.root))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to clear", result.stdout + result.stderr)
        self.assertTrue(scene.exists(), "it deleted the directory anyway")

    def test_it_refuses_a_path_outside_the_run_directory(self):
        outside = pathlib.Path(self.root) / "elsewhere" / "precious"
        outside.mkdir(parents=True)

        result = self.run_clear(str(outside), out_dir="{}/runs".format(self.root))

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(outside.exists(), "it deleted a directory outside OUT_DIR")

    def test_it_refuses_the_run_directory_ITSELF(self):
        """`$OUT_DIR` is two levels up from a scene. Clearing it would delete the sweep."""
        out = pathlib.Path(self.root) / "runs"
        (out / "dream" / "sceneA").mkdir(parents=True)

        result = self.run_clear(str(out), out_dir=str(out))

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(out.exists())

    def test_it_refuses_an_ARM_directory(self):
        """One level short of a scene: clearing it would drop every scene in the arm AND
        the chained memory.json that sits beside them."""
        out = pathlib.Path(self.root) / "runs"
        arm = out / "dream"
        (arm / "sceneA").mkdir(parents=True)
        (arm / "memory.json").write_text('{"scenes": ["sceneA"]}')

        result = self.run_clear(str(arm), out_dir=str(out))

        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((arm / "memory.json").exists())

    def test_it_refuses_an_empty_path(self):
        result = self.run_clear("")
        self.assertNotEqual(result.returncode, 0)

    def test_a_directory_that_is_not_there_is_not_an_error(self):
        """A scene that never started is the common case on the first resume."""
        result = self.run_clear("{}/dream/never-ran".format(self.root))
        self.assertEqual(result.returncode, 0, result.stderr)


class TestAResumeCannotChangeTheKnobs(Fixture):
    """**THE SILENT TWO-CONFIGURATION SWEEP.** `--dream-eta` is a flag so tonight's run
    can use the value `eta_pass.sh` priced, which means it can be forgotten on the resume
    invocation. The scenes already on disk ran at the recorded eta; finishing the rest at
    the default produces one tag holding two configurations, detectable afterwards only
    by reading section F of a report nobody runs at 3am.
    """

    def run_check(self, recorded, eta, cap="12"):
        out = pathlib.Path(self.root) / "runs"
        out.mkdir()
        if recorded is not None:
            (out / "provenance.txt").write_text(recorded)
        return run_bash(
            'OUT_DIR="{}"\nDREAM_ETA="{}"\nDREAM_MAX_RETAINED="{}"\n{}\n'
            "check_resumed_knobs".format(out, eta, cap, extract("check_resumed_knobs"))
        )

    RECORD = "seed:           20260805\ndream_eta:      4.5\ndream_max_ret:  12\n"

    def test_matching_knobs_pass(self):
        result = self.run_check(self.RECORD, "4.5")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        print(result.stdout.strip())

    def test_a_changed_eta_is_refused(self):
        result = self.run_check(self.RECORD, "2.0")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("would change the DREAM knobs mid-sweep", result.stdout)
        print(result.stdout.strip().splitlines()[0])

    def test_a_changed_cap_is_refused(self):
        result = self.run_check(self.RECORD, "4.5", cap="24")
        self.assertNotEqual(result.returncode, 0)

    def test_a_tag_predating_the_record_is_allowed_through(self):
        """ABSENT IS NOT A MISMATCH. A directory from before these lines were written
        carries no eta, and refusing it would make old tags unresumable for no gain."""
        result = self.run_check("seed:           20260805\n", "4.5")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_provenance_at_all_is_allowed_through(self):
        result = self.run_check(None, "4.5")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class TestTheDriverWiresItUp(unittest.TestCase):
    def setUp(self):
        self.text = SWEEP.read_text()

    def test_resume_implies_force(self):
        """It reuses a non-empty directory ON PURPOSE, so it must not trip the guard."""
        self.assertIn("--resume)          RESUME=1; FORCE=1;", self.text)

    def test_resume_survives_the_re_exec(self):
        """The script re-execs itself when a git pull changes it. A resume that became a
        bare --force there would re-run finished scenes into directories `write_episode`
        refuses to overwrite."""
        self.assertIn('_resume_flag="--resume"', self.text)
        self.assertIn("${_resume_flag:+--resume}", self.text)

    def test_the_chain_is_checked_before_the_arm_runs(self):
        """Before the first episode of the resumed arm, not after the night."""
        guard = self.text.index("earshot.tools.chain_check")
        loop = self.text.index('for scene in "${SCENE_LIST[@]}"; do\n    run_dir=')
        self.assertLess(guard, loop)

    def test_a_failed_chain_check_stops_the_sweep(self):
        after = self.text[self.text.index("earshot.tools.chain_check"):][:400]
        self.assertIn("exit 1", after)

    def test_the_knob_guard_runs_before_the_provenance_is_overwritten(self):
        """The guard reads the record that the run is about to replace."""
        guard = self.text.index('[ "$RESUME" = 1 ] && check_resumed_knobs')
        write = self.text.index('} > "$OUT_DIR/provenance.txt"')
        self.assertLess(guard, write)

    def test_a_resumed_run_says_so_in_its_own_summary(self):
        """The tag's records then come from more than one invocation, and a reader
        quoting 'one run' has to be told."""
        self.assertIn("RESUMED: $RESUMED scene/arm cell(s)", self.text)
        self.assertIn("resume:         $RESUME", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
