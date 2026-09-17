"""`propose_sweep.sh` — ADR-0026's contrast, and the four ways it could silently not be one.

The sweep cannot be sourced (it runs a night's work on sight) and cannot be run on a Mac,
so what is checkable here is its TEXT and the invariants that make it a one-knob contrast.
That is worth a test rather than a reading: `dream-2` was reported as a memory null and was
actually a two-variable contrast, and nothing in the tree could have caught it.

Four things have to hold.

1. The arms differ in `--memory-proposes` and in nothing else. Every other flag on the
   `python -m earshot` invocation is outside the branch that sets it.
2. Both arms are DREAM arms. `--memory-proposes` without `--dream` is a no-op (PR #133), so
   a sweep that forgot the knobs would run two identical arms and report a clean null.
3. The DREAM knobs are `ablation_sweep.sh`'s, verbatim. A knob that drifts between the two
   drivers makes this run non-comparable with `dream-3` and `dream-4` for a reason nobody
   would see.
4. The re-exec carries every knob. The script git-pulls and re-execs itself, and a flag
   dropped there is a run that silently used a default after the pull.
"""

import pathlib
import re
import unittest

from _interpreter import assert_interpreter  # noqa: F401

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
SWEEP = TOOLS / "propose_sweep.sh"
ABLATION = TOOLS / "ablation_sweep.sh"


def dream_knobs(text):
    """The `--dream ...` block of a driver's `DREAM_KNOBS`, as a flat token list."""
    match = re.search(r'^DREAM_KNOBS="(.*?)"$', text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError("no DREAM_KNOBS assignment found")
    return tuple(match.group(1).replace("\\\n", " ").split())


class TestItIsAOneKnobContrast(unittest.TestCase):
    def setUp(self):
        self.text = SWEEP.read_text()

    def test_the_only_thing_the_arm_name_decides_is_memory_proposes(self):
        """`dream-2` is why. A second difference hiding in the branch would make this
        contrast uninterpretable in exactly the way ADR-0024 documented."""
        branch = self.text[self.text.index('  case "$arm" in'):]
        branch = branch[:branch.index("esac")]
        self.assertIn('propose-*) PROPOSE_FLAG="--memory-proposes"', branch)
        self.assertNotIn("--dream", branch)
        self.assertNotIn("--memory-condition", branch)
        self.assertNotIn("--seed", branch)
        self.assertNotIn("--n-episodes", branch)

    def test_an_arm_that_names_neither_side_is_refused(self):
        """Nothing else decides which side an arm is on, so a typo would run a control
        under a treatment name and the sweep would report a null it never tested."""
        self.assertIn("is neither propose-* nor replace-*", self.text)

    def test_the_condition_is_passed_from_one_variable_not_from_the_arm(self):
        self.assertIn('--memory-condition "$CONDITION"', self.text)
        self.assertNotIn('--memory-condition "$arm"', self.text)

    def test_both_arms_are_dream_arms(self):
        """--memory-proposes without --dream is a no-op, so a sweep missing the knobs
        would run two identical arms and report a clean, meaningless null."""
        invocation = self.text[self.text.index("    python -m earshot \\"):]
        invocation = invocation[:invocation.index("    status=$?")]
        self.assertIn("$DREAM_KNOBS", invocation)
        self.assertIn("$PROPOSE_FLAG", invocation)
        self.assertIn("--clap", invocation)
        self.assertIn('--memory-store "$STORE"', invocation)


class TestTheDreamKnobsMatchTheOtherDriver(unittest.TestCase):
    """`eta_pass.sh` already holds this invariant against `ablation_sweep.sh` and it is
    the same reason here: a knob that drifts makes the runs non-comparable silently."""

    def test_every_dream_knob_is_the_ablation_sweeps_own(self):
        mine = dream_knobs(SWEEP.read_text())
        theirs = dream_knobs(ABLATION.read_text())
        self.assertEqual(
            mine, theirs,
            "propose_sweep.sh's DREAM knobs have drifted from ablation_sweep.sh's, so "
            "this run cannot be quoted against dream-3 or dream-4",
        )

    def test_eta_is_the_value_the_runs_of_record_used(self):
        """`ablation_sweep.sh` still defaults to 2.0, where `eta-1` measured the cap
        binding on 15 of 15 episodes -- top-k wearing a threshold's name. `eta-2` priced
        4.5 and both `dream-3` and `dream-4` ran at it, so that is the comparable value
        and this driver carries it rather than inheriting the stale default."""
        self.assertIn("\nDREAM_ETA=4.5\n", SWEEP.read_text())

    def test_the_memory_term_is_actually_on(self):
        """`--dream-lambda-memory 0.0` is `dream-nomem`, and with it eq. 26 has no memory
        term, so neither arm could prefer the proposal and the contrast is void."""
        knobs = dream_knobs(SWEEP.read_text())
        self.assertIn("--dream-lambda-memory", knobs)
        self.assertNotEqual(knobs[knobs.index("--dream-lambda-memory") + 1], "0.0")

    def test_eta_and_the_cap_survive_the_re_exec(self):
        """They are shell variables now, so a re-exec that dropped them would revert to
        this file's defaults after a pull rather than keeping what was asked for."""
        text = SWEEP.read_text()
        line = text[text.index('exec bash "$0"'):]
        line = line[:line.index("\n  fi")]
        self.assertIn('--dream-eta "$DREAM_ETA"', line)
        self.assertIn('--dream-max-retained "$DREAM_MAX_RETAINED"', line)


class TestTheReExecCarriesEveryKnob(unittest.TestCase):
    """The script git-pulls and re-execs itself, so a flag dropped here is a run that
    quietly used a default AFTER the pull, with nobody awake."""

    def setUp(self):
        text = SWEEP.read_text()
        self.exec_line = text[text.index('exec bash "$0"'):]
        self.exec_line = self.exec_line[:self.exec_line.index("\n  fi")]

    def test_the_arms_survive(self):
        self.assertIn('--arms "$ARMS"', self.exec_line)

    def test_the_condition_survives(self):
        self.assertIn('--condition "$CONDITION"', self.exec_line)

    def test_the_safety_rail_survives(self):
        """ADR-0026 makes the rail part of the decision, so a re-exec that reverted it to
        the default would change the experiment mid-run."""
        self.assertIn('--propose-max-offset "$PROPOSE_MAX_OFFSET"', self.exec_line)

    def test_resume_survives(self):
        self.assertIn("_resume_flag", self.exec_line)


class TestTheReadoutAnswersTheQuestionItWasRunFor(unittest.TestCase):
    def setUp(self):
        self.text = SWEEP.read_text()

    def test_the_primary_outcome_is_the_conditional_rate(self):
        """ADR-0026 pre-registers stage 4 -> stage 5 conversion, NOT Find-SR."""
        self.assertIn("--given-stage INVESTIGATE_ENTERED", self.text)

    def test_find_sr_is_printed_beside_it(self):
        """ADR-0016: both tests, always."""
        self.assertIn("Find-SR@1m over every paired episode", self.text)

    def test_the_within_arm_repeats_are_read_back(self):
        """THE LESSON OF dream-3 AND dream-4. The same command twice is the floor the
        contrast has to clear, and measuring it in the same run is the whole point of
        having four arms instead of two."""
        self.assertIn('pair "within REPLACE" replace-a replace-b', self.text)
        self.assertIn('pair "within PROPOSE" propose-a propose-b', self.text)

    def test_the_mechanism_check_runs_before_the_contrast_is_believed(self):
        """ADR-0026's fourth branch: under 5% of eligible steps and the run is not a
        result about memory at all."""
        self.assertIn("DID THE MECHANISM RUN AT ALL", self.text)
        self.assertIn("memory_propose_ranked_first", self.text)
        self.assertIn("UNDER 5%", self.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
