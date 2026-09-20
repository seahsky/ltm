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

class TestTheDreamArmChainsItsMemory(unittest.TestCase):
    """The per-scene reset `dream_report` section D measured, removed in the driver.

    The chain is shell, so it is asserted against the script text and then RUN as shell
    below -- a copy of the logic pasted into this file would pass forever after someone
    edited the driver, which is the rule this module's docstring already states.
    """

    def setUp(self):
        self.source = DRIVER.read_text()

    def test_only_the_dream_arm_gets_a_memory_file(self):
        """No other arm has an M^L to chain, and giving one a path would be a lie."""
        self.assertIn('if [ "$arm" = "dream" ]; then', self.source)
        self.assertIn('MEMORY_FILE="$OUT_DIR/$arm/memory.json"', self.source)

    def test_the_flags_reach_the_runner_invocation(self):
        launch = self.source.index("python -m earshot \\")
        window = self.source[launch:launch + 900]
        self.assertIn("${MEMORY_FLAGS}", window)

    def test_the_in_flag_is_withheld_until_the_file_exists(self):
        """`run()` treats a missing in-path as an error, so the first scene must not
        pass one. A driver that always passed it would fail every sweep's first scene."""
        self.assertIn('if [ -f "$MEMORY_FILE" ]; then', self.source)
        guard = self.source.index('if [ -f "$MEMORY_FILE" ]; then')
        self.assertIn("--dream-memory-in", self.source[guard:guard + 200])

    def test_the_scene_order_caveat_is_written_down(self):
        """Episode k of the last scene now depends on every scene before it."""
        self.assertIn("SCENE ORDER IS NOW PART OF THE RESULT", self.source)

    def _flags_for(self, arm, exists):
        """Run the driver's own flag logic in bash, for one arm and one file state."""
        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        memory_file = pathlib.Path(root) / "memory.json"
        if exists:
            memory_file.write_text("{}", encoding="utf-8")
        script = """
        arm="{arm}"
        OUT_DIR="{root}"
        MEMORY_FILE=""
        if [ "$arm" = "dream" ]; then
          MEMORY_FILE="{memory}"
        fi
        MEMORY_FLAGS=""
        if [ -n "$MEMORY_FILE" ]; then
          MEMORY_FLAGS="--dream-memory-out $MEMORY_FILE"
          if [ -f "$MEMORY_FILE" ]; then
            MEMORY_FLAGS="--dream-memory-in $MEMORY_FILE $MEMORY_FLAGS"
          fi
        fi
        echo "$MEMORY_FLAGS"
        """.format(arm=arm, root=root, memory=memory_file)
        done = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        return done.stdout.strip()

    def test_the_first_dream_scene_writes_and_does_not_read(self):
        flags = self._flags_for("dream", exists=False)
        self.assertIn("--dream-memory-out", flags)
        self.assertNotIn("--dream-memory-in", flags)
        print("first dream scene: {}".format(flags))

    def test_every_later_dream_scene_reads_and_writes(self):
        flags = self._flags_for("dream", exists=True)
        self.assertIn("--dream-memory-in", flags)
        self.assertIn("--dream-memory-out", flags)
        print("later dream scene: {}".format(flags))

    def test_the_baseline_arm_gets_no_chain_flags_at_all(self):
        """THE CONTROL. `full` must reach the runner byte-identical to before."""
        for exists in (False, True):
            self.assertEqual("", self._flags_for("full", exists=exists))
        print("full arm: no chain flags, with or without a memory file present")


class TestTheOracleArmActuallyOverridesTheHardcodedFlag(unittest.TestCase):
    """`oracle-loc` is the one arm whose flag is ALREADY on the command line.

    Every other arm names a flag the invocation does not pass. This one collides:
    `--localization realizable` is passed explicitly at `ablation_sweep.sh`, and the arm
    wins only because `${ARM_FLAGS[$i]}` is word-split AFTER it and argparse takes the
    last occurrence of a `store` action.

    **Both halves of that are assumptions until something asserts them**, and each fails
    silently in the same direction: the arm runs REALIZABLE, the run finishes green, and
    the sweep reports a ceiling arm that is a duplicate of the baseline. That is exactly
    the shape of `dream-2` -- a control byte-identical to its treatment, differenced
    against itself, reported as a clean null.

    So the ordering is read out of the shipped script, and the override is put to the
    REAL parser rather than to a belief about argparse.
    """

    @classmethod
    def setUpClass(cls):
        cls.text = DRIVER.read_text()

    def test_the_arm_is_declared_with_the_flag_it_needs(self):
        self.assertIn("oracle-loc", self.text)
        self.assertIn('"--localization oracle"', self.text)

    def test_the_arm_flags_are_word_split_after_the_explicit_localization(self):
        """THE ORDERING, which is the whole mechanism. Reordering these two lines runs
        the ceiling arm as the baseline and nothing anywhere would say so."""
        explicit = self.text.index("--localization realizable")
        arm_flags = self.text.index("${ARM_FLAGS[$i]}")
        self.assertLess(
            explicit, arm_flags,
            "ARM_FLAGS must be word-split AFTER --localization realizable, or the "
            "oracle-loc arm silently runs realizable and reports a null",
        )
        print("ordering: --localization realizable at {}, ARM_FLAGS at {}".format(
            explicit, arm_flags))

    def test_the_real_parser_takes_the_last_localization(self):
        """Put to `earshot.__main__.build_parser`, not to a copy of argparse's rules."""
        from earshot.__main__ import build_parser

        args = build_parser().parse_args([
            "--run-dir", "runs/x",
            "--localization", "realizable",
            "--localization", "oracle",
        ])
        self.assertEqual(args.localization, "oracle")
        print("last-wins confirmed on the shipped parser: {}".format(args.localization))

    def test_the_control_arm_is_unaffected_by_the_same_ordering(self):
        """ADR-0014's other arm. `full` carries an EMPTY flag string, so the explicit
        `realizable` is the only one the parser sees and the baseline cannot drift."""
        from earshot.__main__ import build_parser

        args = build_parser().parse_args([
            "--run-dir", "runs/x", "--localization", "realizable",
        ])
        self.assertEqual(args.localization, "realizable")

    def test_the_three_arm_arrays_stay_the_same_length(self):
        """A name with no flags, or flags with no reason, pairs an arm with another
        arm's command line. Counted off the shipped text rather than trusted."""
        names = self.text[self.text.index("ARM_NAMES=("):]
        names = names[len("ARM_NAMES=("):names.index(")")].split()
        flags = _count_array_entries(self.text, "ARM_FLAGS=(")
        why = _count_array_entries(self.text, "ARM_WHY=(")
        self.assertEqual(len(names), flags, "ARM_NAMES and ARM_FLAGS disagree")
        self.assertEqual(len(names), why, "ARM_NAMES and ARM_WHY disagree")
        print("{} arm(s), all three arrays agree: {}".format(len(names), " ".join(names)))


class TestTheMatchedOracleArmIsSelectableAndIsNotThePrefixArm(unittest.TestCase):
    """ADR-0028's arm, and the hazard its NAME creates.

    `oracle-loc-matched` has `oracle-loc` as a prefix, and the two arms are scored on
    different criteria -- `oracle-1` read 94.3% source-reached out of one and 0.7%
    Find-SR@1m out of the same episodes. A selector that prefix-matched would run BOTH
    under one name, or the wrong one under either, and the sweep would finish green
    while the readout differenced a ceiling against a ceiling on another criterion.

    So the selection is RUN IN BASH off the shipped arrays rather than reasoned about,
    and the flag string is asserted against the enum's own value rather than a literal
    typed twice.
    """

    @classmethod
    def setUpClass(cls):
        if shutil.which("bash") is None:
            raise unittest.SkipTest("no bash on PATH")
        cls.text = DRIVER.read_text()

    def _select(self, wanted: str):
        """The driver's own `--arms` block, over its own arrays, for one selection."""
        arrays = self.text[self.text.index("ARM_NAMES=("):self.text.index("N_SCENES=")]
        script = 'WANTED_ARMS="{}"\n{}\nfor i in "${{!ARM_NAMES[@]}}"; do\n'.format(
            wanted, arrays
        ) + '  printf "%s\\t%s\\n" "${ARM_NAMES[$i]}" "${ARM_FLAGS[$i]}"\ndone\n'
        done = subprocess.run(
            ["bash", "-c", script], capture_output=True, text=True, check=True
        )
        return [
            tuple(line.split("\t")) for line in done.stdout.splitlines() if line.strip()
        ]

    def test_the_arm_carries_exactly_the_enums_own_value(self):
        """A renamed enum member must break HERE and not at 11pm on the box, where the
        only symptom is argparse's `invalid choice` after the scene list has loaded."""
        from earshot.config import Localization

        self.assertIn(
            '"--localization {}"'.format(Localization.ORACLE_MATCHED.value), self.text
        )

    def test_selecting_it_gives_the_matched_flag_and_nothing_else(self):
        self.assertEqual(
            self._select("oracle-loc-matched"),
            [("oracle-loc-matched", "--localization oracle_matched")],
        )

    def test_selecting_the_prefix_arm_does_not_drag_the_matched_one_in(self):
        """THE PREFIX HAZARD, put to bash. Exact `=` is what makes this pass; a `case`
        or a glob would not, and the difference is invisible in a green run."""
        selected = self._select("oracle-loc")
        self.assertEqual(selected, [("oracle-loc", "--localization oracle")])

    def test_the_decisive_pair_selects_in_order_with_full_first(self):
        """The run this arm exists for. `full` is the IN-RUN control, because
        `repeat-1` measured 16.2% of outcomes flipping on byte-identical reruns, and
        `window_report` quotes a subset without `full` against its FIRST arm."""
        selected = self._select("full oracle-loc-matched")
        self.assertEqual(
            selected,
            [("full", ""), ("oracle-loc-matched", "--localization oracle_matched")],
        )
        print("decisive pair resolves to: {}".format(selected))

    def test_the_two_ceiling_arms_do_not_share_a_flag(self):
        """If they ever did, one of them is a duplicate of the other and the sweep pays
        a night to difference an arm against itself -- `dream-2`'s shape exactly."""
        both = dict(self._select("oracle-loc oracle-loc-matched"))
        self.assertEqual(len(set(both.values())), 2, both)


def _count_array_entries(text: str, opener: str) -> int:
    """Quoted entries of a bash array literal, ignoring comment lines inside it."""
    body = text[text.index(opener) + len(opener):]
    body = body[:body.index("\n)")]
    return sum(
        1 for line in body.splitlines()
        if line.strip().startswith('"') or line.strip() == '""'
    )



if __name__ == "__main__":  # pragma: no cover
    unittest.main()
