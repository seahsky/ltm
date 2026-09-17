"""Which arm the sweep's readout pairs everything against, and when that is RED.

`dream-4` is why this file exists. It ran `--arms "dream dream-nomem"` -- the two-arm
repeat pair ADR-0025 pre-registered by name -- for 5h40m, wrote all 564 episodes, passed
every one of its 38 smoke gates, and exited 1. The readout looked for a directory called
`full`, did not find one, and reported "no baseline to quote against". Nothing was wrong
with the run: the operator had not asked for `full`, and the contrast that sweep wanted
was its two arms against each other, which had to be run by hand afterwards.

So the rule is split in two. A sweep that ASKED for `full` and does not have it on disk is
a run that did not happen, and that is still red. A sweep that never asked for it is a
contrast between the arms it did ask for, and the readout says which one it chose.

The function is extracted and run under real bash, for `test_ablation_resume`'s reason:
the script cannot be sourced, because it runs a sweep on sight.
"""

import pathlib
import unittest

from _interpreter import assert_interpreter  # noqa: F401
from test_ablation_resume import extract, run_bash

SWEEP = pathlib.Path(__file__).resolve().parents[2] / "tools" / "ablation_sweep.sh"


class TestReferenceArm(unittest.TestCase):
    """`full` when it is there, the first arm asked for when it is not."""

    def choose(self, *arms):
        result = run_bash(
            "{}\nreference_arm {}".format(
                extract("reference_arm"), " ".join("'%s'" % a for a in arms)
            )
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_the_full_sweep_quotes_against_full(self):
        self.assertEqual(
            self.choose("full", "no-climb", "no-cue", "scan-only", "anechoic"), "full"
        )

    def test_full_wins_wherever_it_sits_in_the_list(self):
        """Its position in --arms is an operator's typing order, not a decision."""
        self.assertEqual(self.choose("dream", "full", "dream-nomem"), "full")
        self.assertEqual(self.choose("no-cue", "anechoic", "full"), "full")

    def test_the_repeat_pair_quotes_against_its_first_arm(self):
        """dream-4's own invocation. `dream -> dream-nomem` is the pre-registered
        contrast, and it is the one `episode_diff` was run by hand to get."""
        self.assertEqual(self.choose("dream", "dream-nomem"), "dream")

    def test_the_order_asked_for_decides_when_full_is_absent(self):
        self.assertEqual(self.choose("dream-nomem", "dream"), "dream-nomem")

    def test_one_arm_is_its_own_reference(self):
        self.assertEqual(self.choose("dream"), "dream")


class TestTheReadoutIsRedOnlyForAnArmThatDidNotRun(unittest.TestCase):
    """The distinction dream-4 cost 5h40m to find. Asserted against the script's text,
    because the branch it guards is 700 lines into a driver that runs a sweep on sight."""

    def setUp(self):
        self.text = SWEEP.read_text()

    def test_the_readout_pairs_against_the_chosen_reference_not_a_hardcoded_full(self):
        self.assertIn('REFERENCE_ARM="$(reference_arm "${ARM_NAMES[@]}")"', self.text)
        self.assertNotIn('BASELINE_DIR="$OUT_DIR/full"', self.text)

    def test_a_missing_reference_arm_is_still_red(self):
        """`--arms "full dream"` with no full/ on disk is a run that did not happen."""
        block = self.text[self.text.index('if [ ! -d "$REFERENCE_DIR" ]; then'):][:600]
        self.assertIn("READ_FAILED=1", block)

    def test_a_single_arm_sweep_is_not_red(self):
        """It has nothing to pair against, which is what was asked for."""
        start = self.text.index('elif [ "${#ARM_NAMES[@]}" -lt 2 ]; then')
        block = self.text[start:self.text.index("else", start)]
        self.assertIn("SKIPPED: one arm", block)
        self.assertNotIn("READ_FAILED=1", block)

    def test_the_header_names_the_arm_it_chose(self):
        """Nothing is ever quoted against an arm the reader did not expect."""
        self.assertIn(
            'echo "  --- each arm against $REFERENCE_ARM, PAIRED BY EPISODE ---"',
            self.text,
        )
        self.assertIn('echo "  === $REFERENCE_ARM -> $arm ==="', self.text)

    def test_the_green_footer_does_not_name_a_fixed_number_of_arms(self):
        """It said "the other four" and "contains NO memory arm" through three arms
        being added, both of which stopped being true."""
        footer = self.text[self.text.index("GREEN — every arm ran"):]
        self.assertIn("${ARM_NAMES[*]}", footer)
        self.assertNotIn("the other four", footer)
        self.assertNotIn("contains NO", footer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
