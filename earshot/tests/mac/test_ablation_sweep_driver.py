"""`ablation_sweep.sh`'s zero-yield predicate, run as shipped.

`abl-1` finished five complete arms, 1410 episodes, and reported RED. Nothing had gone
wrong with the run: `mL8ThkuaVTM` places no episode in any sweep this repo has run, the
run loop recorded that correctly and skipped it, and then the readout loop judged the five
empty directories anyway. `smoke` returned 2, which is right -- a gate with nothing to
judge is NOT_RUN and NOT_RUN is red -- so the driver was asking the wrong question in the
wrong place.

The fix is one predicate asked in both loops, and this file holds it to both arms
(ADR-0014): the zero-yield directory must be recognised, and a directory with episodes in
it must NOT be, because a predicate that answered "skip" to everything would have turned
the same run green by never judging anything at all.

The function text is EXTRACTED FROM THE SCRIPT and run by bash. A copy of the predicate
pasted into this file would pass forever after someone edited the driver.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

DRIVER = pathlib.Path(__file__).resolve().parents[2] / "tools" / "ablation_sweep.sh"


def extract_function(source: str, name: str) -> str:
    """Return the shell text of `name`, from its opening line to the closing brace."""
    lines = source.splitlines()
    opener = "{}() {{".format(name)
    for start, line in enumerate(lines):
        if line.startswith(opener):
            for end in range(start + 1, len(lines)):
                if lines[end] == "}":
                    return "\n".join(lines[start:end + 1])
            raise AssertionError("{} in {} is never closed".format(name, DRIVER))
    raise AssertionError("{} is not defined in {}".format(name, DRIVER))


class TestTheZeroYieldPredicate(unittest.TestCase):
    """The one question `abl-1` asked in only one of the two places that needed it."""

    @classmethod
    def setUpClass(cls):
        if shutil.which("bash") is None:
            raise unittest.SkipTest("no bash on PATH")
        cls.function = extract_function(DRIVER.read_text(), "is_zero_yield")

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # The predicate calls bare `python`, which on the box is the `ss2` env's. Here it
        # has to be this interpreter, so give bash a PATH where `python` is exactly that.
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        shim = self.bin / "python"
        shim.write_text('#!/bin/sh\nexec "{}" "$@"\n'.format(sys.executable))
        shim.chmod(0o755)

    def ask(self, run_dir: pathlib.Path) -> int:
        """Exit status of `is_zero_yield <run_dir>`; 0 means "skip this cell"."""
        env = dict(os.environ, PATH="{}:{}".format(self.bin, os.environ.get("PATH", "")))
        return subprocess.call(
            ["bash", "-c", '{}\nis_zero_yield "$1"'.format(self.function), "_", str(run_dir)],
            env=env,
        )

    def _cell(self, name: str, summary):
        run_dir = self.tmp / name
        run_dir.mkdir()
        if summary is not None:
            (run_dir / "summary.json").write_text(json.dumps(summary))
        return run_dir

    def test_the_scene_that_placed_no_episode_is_recognised(self):
        # This is `mL8ThkuaVTM`: the summary is written, and it says zero.
        self.assertEqual(0, self.ask(self._cell("empty", {"n_episodes": 0})))

    def test_a_cell_with_episodes_is_not_recognised_and_must_still_be_judged(self):
        # The arm of the detector that matters most. If this returned 0 the driver would
        # skip every gate and report green over a sweep it never checked.
        self.assertEqual(1, self.ask(self._cell("full", {"n_episodes": 15})))

    def test_a_crashed_cell_that_wrote_no_summary_is_not_excused(self):
        # No summary means the runner did not reach the point where it writes one. That is
        # a broken run, and it stays red.
        self.assertEqual(1, self.ask(self._cell("crashed", None)))

    def test_an_unreadable_summary_is_not_excused(self):
        run_dir = self.tmp / "truncated"
        run_dir.mkdir()
        (run_dir / "summary.json").write_text("{not json")
        self.assertEqual(1, self.ask(run_dir))

    def test_a_summary_without_the_key_is_not_excused(self):
        self.assertEqual(1, self.ask(self._cell("nokey", {"scene": "mL8ThkuaVTM"})))


class TestTheVacuousArmFloorSitsAtTheArmNotTheScene(unittest.TestCase):
    """`abl-1`'s other false red, and the one that mattered more.

    Criterion 5 is a rate: green iff at least one episode closed the loop, and `tally`
    fails it at 0/n to catch ADR-0014's vacuous arm. Ten scene/arm cells hit that floor in
    `abl-1` and every one was in an arm built to cripple the controller -- `scan-only` in
    seven scenes, one each in `no-climb`, `no-cue` and `anechoic`, none in `full`. The
    baseline closed the loop in all nineteen scenes on the same episodes, so those zeros
    are the ablation working. A gate that reds on them reds hardest on the strongest
    result in the table.

    The floor therefore moves to the arm. These tests hold the three things that keeps:
    the baseline still judged per scene, the vacuous ARM still caught, and a failure that
    is not criterion-5-alone still red wherever it happens.
    """

    PASS, MEASUREMENT, RED = 0, 1, 2

    # The three shapes `smoke` actually emits, ending as its `summary()` ends.
    GREEN = "task spec §8 — acceptance criteria over 15 episode(s):\n  ...\nGREEN"
    ONLY_5 = "task spec §8 — acceptance criteria over 15 episode(s):\n  ...\nRED — criteria 5"
    FIVE_AND_SEVEN = ONLY_5 + ", 7"
    NOT_FIVE = "task spec §8 — acceptance criteria over 15 episode(s):\n  ...\nRED — criteria 1, 3"

    @classmethod
    def setUpClass(cls):
        if shutil.which("bash") is None:
            raise unittest.SkipTest("no bash on PATH")
        cls.function = extract_function(DRIVER.read_text(), "gate_verdict")

    def verdict(self, arm: str, rc: int, out: str) -> int:
        return subprocess.call(
            ["bash", "-c",
             'set -uo pipefail\n{}\ngate_verdict "$1" "$2" "$3"'.format(self.function),
             "_", arm, str(rc), out],
        )

    def test_the_ablation_scene_that_closed_the_loop_zero_times_is_a_measurement(self):
        # This is `scan-only/6s7QHgap2fW` and the nine others `abl-1` reported as RED.
        self.assertEqual(self.MEASUREMENT, self.verdict("scan-only", 1, self.ONLY_5))

    def test_the_baseline_is_denied_the_allowance(self):
        # `full` is the baseline of record (ADR-0021). A scene where it never once closed
        # the loop is a bug there, and it had none in `abl-1`.
        self.assertEqual(self.RED, self.verdict("full", 1, self.ONLY_5))

    def test_a_second_failing_criterion_stays_red_in_an_ablation_arm(self):
        # The arm of this detector that matters. Criterion 7 is the audio wall-clock
        # ceiling and has gone red on a real run before; the allowance must not carry it.
        self.assertEqual(self.RED, self.verdict("scan-only", 1, self.FIVE_AND_SEVEN))

    def test_a_failure_that_is_not_criterion_5_stays_red_in_an_ablation_arm(self):
        self.assertEqual(self.RED, self.verdict("anechoic", 1, self.NOT_FIVE))

    def test_a_zero_return_is_a_pass_whatever_the_arm(self):
        for arm in ("full", "scan-only"):
            self.assertEqual(self.PASS, self.verdict(arm, 0, self.GREEN), arm)

    def test_a_nothing_to_judge_return_is_never_excused(self):
        # `smoke` returns 2 for NOT_RUN. That is the zero-yield path, handled before the
        # gate runs at all; if one ever reaches here it is red, not a measurement.
        self.assertEqual(
            self.RED, self.verdict("scan-only", 2, "no episode records under x — nothing to judge"))


class TestTheVacuousArmIsStillCaught(unittest.TestCase):
    """Moving the floor to the arm must not delete it (ADR-0014's vacuous arm)."""

    def setUp(self):
        self.source = DRIVER.read_text()

    def test_an_arm_green_in_no_scene_is_red(self):
        self.assertIn('if [ "$arm_green" -eq 0 ]; then', self.source)
        vacuous = self.source.index('"$arm_green" -eq 0')
        self.assertIn("GATE_FAILED=1", self.source[vacuous:vacuous + 400])

    def test_the_readout_loop_routes_every_gate_through_the_predicate(self):
        gate = self.source.index("python -m earshot.task.smoke")
        self.assertIn('gate_verdict "$arm" "$gate_rc" "$gate_out"', self.source[gate:gate + 400])

    def test_no_piped_grep_in_any_executable_line(self):
        # Line 182's footgun: under pipefail a matching `grep -q` exits early, SIGPIPEs
        # its producer, and turns found-it into a pipeline failure. Comments are excluded
        # because two of them warn about exactly this and would match themselves.
        offenders = [
            line for line in self.source.splitlines()
            if "| grep -q" in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual([], offenders)

    def test_the_count_is_printed_because_it_is_a_measurement(self):
        # CLAUDE.md: box tests print their measurements. The per-arm count is a second
        # ordering of the ablation table, so it goes in the banner rather than a log.
        self.assertIn("VACUOUS_BY_ARM", self.source)
        self.assertIn('echo "  scenes where an ablation arm closed the loop ZERO times',
                      self.source)


class TestBothLoopsAskIt(unittest.TestCase):
    """The bug was not the predicate. It was that only one of two loops asked."""

    def setUp(self):
        self.source = DRIVER.read_text()

    def test_the_predicate_is_defined_once(self):
        self.assertEqual(1, self.source.count("is_zero_yield() {"))

    def test_the_run_loop_and_the_readout_loop_both_call_it(self):
        # Two call sites, and no third copy of the json probe anywhere else: `abl-1` cost
        # a red banner because the readout loop had no guard at all.
        self.assertEqual(2, self.source.count('is_zero_yield "'))

    def test_the_readout_loop_guards_the_gate_it_runs(self):
        gate = self.source.index("python -m earshot.task.smoke")
        guard = self.source.rindex('is_zero_yield "', 0, gate)
        self.assertIn("SKIPPED, zero yield", self.source[guard:gate])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
