"""`tools/eta_pass.sh` — ADR-0024 step 1, and the fence that keeps it honest.

The driver duplicates `ablation_sweep.sh`'s `DREAM_KNOBS` rather than sourcing them,
because that script runs a sweep on sight and cannot be sourced for its constants. A
duplicate is a divergence waiting to happen, and a divergence here is the worst kind:
the run would price `eta` against a configuration no arm uses, print a number, and look
exactly like a run that priced it correctly.

**SO THE TWO LISTS ARE HELD AGAINST EACH OTHER HERE.** A knob added to the sweep and not
to this driver fails this file rather than reaching a box slot. The two knobs the driver
exists to vary -- `eta` and `max-retained` -- are the only ones allowed to differ, and
they are required to differ by being flags rather than literals.

No simulator: this reads the two shell files as text. `test_no_env_flags.py` is the
precedent for a mac test that holds a shell script to a rule.
"""

import pathlib
import re
import shlex
import shutil
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
ETA_PASS = TOOLS / "eta_pass.sh"
ABLATION_SWEEP = TOOLS / "ablation_sweep.sh"

# The two the driver prices. Everything else must match the sweep exactly.
PRICED = ("dream-eta", "dream-max-retained")


def _knob_block(text, marker):
    """The `DREAM_KNOBS="..."` assignment, as one line-joined string."""
    match = re.search(
        r'^DREAM_KNOBS="(.*?)"$', text, flags=re.MULTILINE | re.DOTALL
    )
    if match is None:
        raise AssertionError(
            "no DREAM_KNOBS assignment found in {}; the fence cannot compare what it "
            "cannot find, and a silently-skipped fence is the shape ADR-0014 refuses"
            .format(marker)
        )
    return " ".join(match.group(1).replace("\\\n", " ").split())


def _pairs(block):
    """`{flag: value}` over a `--flag value` list, with bare flags mapping to None."""
    tokens = block.split()
    pairs = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            raise AssertionError("expected a flag, got {!r} in {!r}".format(token, block))
        name = token[2:]
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            pairs[name] = tokens[index + 1]
            index += 2
        else:
            pairs[name] = None
            index += 1
    return pairs


class TestTheDriverExists(unittest.TestCase):
    def test_the_script_is_on_disk_and_executable_as_bash(self):
        """`eta-1` exited 127 against a script that did not exist. It does now."""
        self.assertTrue(ETA_PASS.is_file(), "{} is missing".format(ETA_PASS))
        self.assertTrue(ETA_PASS.read_text().startswith("#!/bin/bash"))

    def test_it_refuses_to_be_sourced(self):
        """The house rule: a sourced driver's `exit` kills the operator's shell."""
        self.assertIn("don't source it", ETA_PASS.read_text())

    def test_the_tag_is_required_and_one_directory_is_one_run(self):
        text = ETA_PASS.read_text()
        self.assertIn("--tag is required", text)
        self.assertIn("One directory is one run", text)


class TestTheKnobsMatchTheSweep(unittest.TestCase):
    """**THE FENCE.** A knob that reaches the sweep and not this driver prices `eta`
    against a configuration no arm runs."""

    def setUp(self):
        self.driver = _pairs(_knob_block(ETA_PASS.read_text(), "eta_pass.sh"))
        self.sweep = _pairs(_knob_block(ABLATION_SWEEP.read_text(), "ablation_sweep.sh"))

    def test_the_same_knobs_are_named_in_both(self):
        self.assertEqual(
            sorted(self.driver), sorted(self.sweep),
            "the two DREAM knob lists name different flags; a knob added to one and not "
            "the other makes this driver measure a configuration no arm uses",
        )

    def test_every_knob_the_driver_does_not_price_matches_the_sweep_exactly(self):
        for name, value in sorted(self.sweep.items()):
            if name in PRICED:
                continue
            self.assertEqual(
                self.driver[name], value,
                "--{} is {} in eta_pass.sh and {} in ablation_sweep.sh".format(
                    name, self.driver[name], value),
            )

    def test_the_two_priced_knobs_are_flags_and_not_literals(self):
        """They have to VARY: a literal here would make `--eta` silently inert, which is
        the shape of every silent-failure incident in this repo's convention list."""
        for name in PRICED:
            self.assertIn(name, self.driver, "--{} is not in the driver's knobs".format(name))
            self.assertTrue(
                str(self.driver[name]).startswith("$"),
                "--{} is the literal {!r} in eta_pass.sh; it must be a shell variable or "
                "the flag that sets it does nothing".format(name, self.driver[name]),
            )

    def test_dream_is_asked_for_by_name_in_both(self):
        self.assertIsNone(self.driver["dream"])
        self.assertIsNone(self.sweep["dream"])


class TestTheFenceFires(unittest.TestCase):
    """**THE FORCED-FAILURE ARM (ADR-0014).** The tests above pass on a healthy pair, and
    a comparison that cannot fail is not a fence. These feed it divergences and require
    each one to be caught, through the same helpers the real comparison uses."""

    HEALTHY = 'DREAM_KNOBS="--dream --dream-eta $ETA --dream-alpha 1.0"\n'

    def test_a_knob_in_one_list_and_not_the_other_is_caught(self):
        driver = _pairs(_knob_block(self.HEALTHY, "driver"))
        sweep = _pairs(_knob_block(
            'DREAM_KNOBS="--dream --dream-eta 0.5 --dream-alpha 1.0 --dream-beta 1.0"\n',
            "sweep",
        ))
        self.assertNotEqual(sorted(driver), sorted(sweep))

    def test_a_knob_whose_VALUE_drifted_is_caught(self):
        driver = _pairs(_knob_block(self.HEALTHY, "driver"))
        sweep = _pairs(_knob_block(
            'DREAM_KNOBS="--dream --dream-eta 0.5 --dream-alpha 9.0"\n', "sweep"))
        self.assertNotEqual(driver["dream-alpha"], sweep["dream-alpha"])

    def test_a_priced_knob_frozen_to_a_literal_is_caught(self):
        """The defect that would make `--eta` accept a value and ignore it."""
        frozen = _pairs(_knob_block('DREAM_KNOBS="--dream --dream-eta 2.0"\n', "frozen"))
        self.assertFalse(str(frozen["dream-eta"]).startswith("$"))

    def test_a_missing_knob_block_raises_rather_than_passing_vacuously(self):
        """A fence that finds nothing must not read as a fence that found no problem."""
        with self.assertRaises(AssertionError) as caught:
            _knob_block("nothing to see here\n", "empty.sh")
        self.assertIn("cannot compare what it cannot find", str(caught.exception))


class TestTheFlagsReachTheRunner(unittest.TestCase):
    """The driver's OWN flag list, through the real parser and the real builder.

    `eta-1` exited 127 on a script that did not exist. The next failure of that shape is
    a script that exists and emits a flag the runner rejects, 40 steps into a box slot.
    This is that arm, and it costs no simulator: `dream_kwargs_from_args` is pure.
    """

    def _argv(self, eta="2.0", max_retained="12"):
        block = _knob_block(ETA_PASS.read_text(), "eta_pass.sh")
        block = block.replace("$ETA", eta).replace("$MAX_RETAINED", max_retained)
        return shlex.split(block)

    def test_the_knob_list_parses_and_builds_dream_knobs(self):
        from earshot.__main__ import build_parser, dream_kwargs_from_args

        args = build_parser().parse_args(
            ["--run-dir", "runs/x", "--clap"] + self._argv())
        knobs = dream_kwargs_from_args(args)["dream_knobs"]
        self.assertAlmostEqual(knobs.eta, 2.0)
        self.assertEqual(knobs.max_retained, 12)

    def test_the_two_priced_flags_actually_move_what_reaches_the_run(self):
        """Substituting the shell variables must change the built knobs, or the driver's
        `--eta` is decoration."""
        from earshot.__main__ import build_parser, dream_kwargs_from_args

        args = build_parser().parse_args(
            ["--run-dir", "runs/x", "--clap"] + self._argv(eta="3.5", max_retained="7"))
        knobs = dream_kwargs_from_args(args)["dream_knobs"]
        self.assertAlmostEqual(knobs.eta, 3.5)
        self.assertEqual(knobs.max_retained, 7)

    def test_the_list_is_complete_so_dream_does_not_refuse_it(self):
        """`--dream` names every missing knob at once rather than one failed launch at a
        time. This asserts the driver leaves none of them for it to name."""
        from earshot.__main__ import build_parser, dream_kwargs_from_args

        args = build_parser().parse_args(
            ["--run-dir", "runs/x", "--clap"] + self._argv())
        try:
            dream_kwargs_from_args(args)
        except SystemExit as exit_:  # pragma: no cover - the failure is the message
            self.fail("the driver's knob list is incomplete: {}".format(exit_))


class TestTheControlArmDiffersInExactlyOneKnob(unittest.TestCase):
    """`dream-nomem` is the control ADR-0024 asked for, and its whole value is that it is
    identical to `dream` apart from `lambda_memory`.

    A control that drifted in a second knob would difference two things at once, which is
    precisely the defect it exists to remove from `dream-2`. A control that drifted in NO
    knob is worse: `ablation_sweep.sh` builds it by substituting into `DREAM_KNOBS`, and a
    substitution that matched nothing would run the control as a byte-identical copy of
    the treatment and report a clean null over an arm differenced with itself.

    Both are checked here, and the script checks the second at run time as well.
    """

    def setUp(self):
        self.text = ABLATION_SWEEP.read_text()

    def test_the_substitution_is_anchored_to_the_value_actually_in_the_knobs(self):
        knobs = _pairs(_knob_block(self.text, "ablation_sweep.sh"))
        self.assertIn("dream-lambda-memory", knobs)
        self.assertIn(
            "--dream-lambda-memory {}/".format(knobs["dream-lambda-memory"]),
            self.text,
            "the substitution that builds dream-nomem does not match the "
            "--dream-lambda-memory value in DREAM_KNOBS, so it would be a silent no-op",
        )

    def test_a_no_op_substitution_is_refused_at_run_time(self):
        self.assertIn("byte-identical to the treatment arm", self.text)

    def test_the_control_turns_the_memory_term_OFF(self):
        self.assertIn("--dream-lambda-memory 0.0}", self.text)

    def test_both_arms_are_named_and_carry_the_same_knob_string_otherwise(self):
        self.assertIn("dream dream-nomem)", self.text)
        self.assertIn('"--clap $DREAM_KNOBS"', self.text)
        self.assertIn('"--clap $DREAM_KNOBS_NOMEM"', self.text)

    def test_the_priced_knobs_reach_the_sweep_as_flags_too(self):
        """Tonight's sweep has to run at the eta `eta_pass.sh` priced, and editing a
        driver by hand at 11pm is how a sweep runs at a value nobody recorded."""
        knobs = _pairs(_knob_block(self.text, "ablation_sweep.sh"))
        self.assertTrue(str(knobs["dream-eta"]).startswith("$"))
        self.assertTrue(str(knobs["dream-max-retained"]).startswith("$"))
        self.assertIn("--dream-eta)       need_value", self.text)
        self.assertIn('--dream-eta "$DREAM_ETA"', self.text)


class TestTheReadoutCanReadWhatTheRunWrote(unittest.TestCase):
    """**THE `eta-1` FAILURE, FENCED.** The run worked: 15 episodes, 7m 36s, every
    funnel stage recorded. Then step 4 printed "NO EPISODES ON DISK" and the driver
    exited 2, because the runner writes `<run-dir>/episodes/` and `dream_report` walked
    only the sweep's `<arm>/<scene>/episodes/`.

    Two halves, and both are asserted: the driver hands the SAME directory to the runner
    and to the readout, and that directory's layout is one the readout reads.
    """

    def _argument_after(self, flag):
        """The token following `flag` in the script, as written."""
        tokens = ETA_PASS.read_text().split()
        self.assertIn(flag, tokens, "{} is not in eta_pass.sh".format(flag))
        return tokens[tokens.index(flag) + 1]

    def test_the_runner_and_the_readout_are_pointed_at_one_directory(self):
        run_dir = self._argument_after("--run-dir")
        text = ETA_PASS.read_text()
        report_line = [
            line for line in text.splitlines()
            if "earshot.tools.dream_report" in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(report_line), 1, "expected one dream_report invocation")
        self.assertIn(run_dir.strip('"'), report_line[0])

    def test_the_layout_the_driver_writes_is_one_the_readout_reads(self):
        """The half a text comparison cannot make. Writes the driver's layout with the
        REAL writer and reads it with the REAL reader, no simulator involved."""
        from earshot.report.agent import AgentReport
        from earshot.report.artifacts import write_episode
        from earshot.report.audit import EpisodeAudit, FunnelStage, StepRecord
        from earshot.tools.dream_report import main as dream_report_main

        root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, root, True)
        # `--run-dir runs/$TAG`, exactly as the driver passes it: no scene directory.
        run_dir = pathlib.Path(root) / "eta-1"
        run_dir.mkdir()
        for index in range(2):
            write_episode(
                str(run_dir), index, AgentReport(resumed=True),
                EpisodeAudit(
                    episode_index=index,
                    scene_id="4ok3usBNeis",
                    funnel_stage=FunnelStage.PRIMARY_RESUMED,
                    steps=(StepRecord(step=0, measured_rms=0.1, audio_render_s=0.05),),
                    metrics={
                        "dream_informed_steps": 20.0,
                        "dream_rows_added": 12.0,
                        "dream_segments_over_eta": 21.0,
                        "dream_experience_rows": 12.0,
                    },
                ),
            )

        self.assertEqual(
            dream_report_main([str(run_dir)]), 0,
            "the readout cannot read the layout the driver writes; that is the eta-1 "
            "failure and it exits 2 after the box time is already spent",
        )


class TestItReadsWhatItRan(unittest.TestCase):
    """A driver that leaves the readout to a second command is a driver whose numbers
    reach nobody. `dream-1` wrote its central quantity onto 282 episodes and no reader
    printed it."""

    def test_the_report_runs_in_the_same_invocation(self):
        self.assertIn("earshot.tools.dream_report", ETA_PASS.read_text())

    def test_the_branch_the_numbers_decide_is_printed(self):
        """The cap binding on most episodes is a STOP, not a cap to raise quietly."""
        text = ETA_PASS.read_text()
        self.assertIn("dream_segments_over_eta", text)
        self.assertIn("CAP IS THE RETENTION RULE", text)

    def test_it_disclaims_being_a_measurement_of_dream(self):
        """One scene, one run, against a 16.2% flip rate. ADR-0016's rule, in the output
        rather than only in a doc nobody reads at 3am."""
        self.assertIn("16.2%", ETA_PASS.read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)
