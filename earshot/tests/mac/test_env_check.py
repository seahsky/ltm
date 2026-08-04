"""``env_check`` — the two halves that need no box.

Ticket 19 split this module three ways, not two:

1. metadata comparison → **here**
2. capability probes → ``tests/box/test_env_check_box.py``
3. *given a failing probe result, does ``assert_env()`` raise* → **here, with injected
   results**, and it is the highest-value assertion in the module

Row 3 is ticket 13's exact bug. The old gate's torch layer computed the right answer and
then skipped, so the run reported success from a position where nothing had been proven.
Making ``judge()`` pure is what lets that be tested at all, and every case below is one
way a probe result can fail to be a pass.
"""

from __future__ import annotations

import contextlib
import io
import pathlib
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.env_check import (
    CLAP_PROBE,
    REQUIRED_PROBES,
    EnvCheckError,
    EnvReport,
    Probe,
    ProbeStatus,
    assert_green,
    compare_resolved_against_constraints,
    expected_probes,
    judge,
    main,
    parse_pins,
    parse_resolved,
)


def _passing(names=None):
    return [Probe(name, ProbeStatus.PASS, "ok") for name in sorted(names or REQUIRED_PROBES)]


class TestTheJudgeIsWhereTicket13sBugWouldHaveDied(unittest.TestCase):
    """Row 3: injected results, no box, no imports of torch or habitat-sim."""

    def test_all_passing_is_green(self):
        self.assertTrue(judge(_passing(), REQUIRED_PROBES).green)

    def test_a_failing_probe_is_red(self):
        probes = _passing()
        probes[0] = Probe(probes[0].name, ProbeStatus.FAIL, "allocation raised")
        report = judge(probes, REQUIRED_PROBES)
        self.assertFalse(report.green)
        self.assertIn(probes[0].name, report.failed)

    def test_a_probe_that_did_not_run_is_not_a_pass(self):
        """The bug itself. ``NOT_RUN`` is a third state precisely so that a layer which
        could not complete cannot be read as one that completed successfully."""
        probes = _passing()
        probes[1] = Probe(probes[1].name, ProbeStatus.NOT_RUN, "torch did not import")
        report = judge(probes, REQUIRED_PROBES)
        self.assertFalse(report.green)
        self.assertIn(probes[1].name, report.failed)

    def test_a_probe_that_stopped_being_emitted_cannot_pass_by_absence(self):
        """The other half of the same bug, one level up: silence is not consent.

        A probe deleted from ``run_probes`` would otherwise make the report greener,
        which is the shape of every inert pin this map has found — ticket 17's
        constraint on a never-installed package, ticket 20's exclusion of an absent
        directory.
        """
        probes = _passing()[1:]
        report = judge(probes, REQUIRED_PROBES)
        self.assertFalse(report.green)
        self.assertEqual(len(report.missing), 1)
        self.assertIn(report.missing[0], REQUIRED_PROBES)

    def test_an_empty_probe_list_is_red_rather_than_vacuously_green(self):
        report = judge([], REQUIRED_PROBES)
        self.assertFalse(report.green)
        self.assertEqual(set(report.missing), set(REQUIRED_PROBES))

    def test_two_probes_under_one_name_raise_rather_than_shadowing(self):
        probes = _passing() + [Probe(sorted(REQUIRED_PROBES)[0], ProbeStatus.FAIL, "…")]
        with self.assertRaises(ValueError):
            judge(probes, REQUIRED_PROBES)

    def test_clap_is_expected_only_when_requested(self):
        self.assertEqual(expected_probes(), REQUIRED_PROBES)
        self.assertIn(CLAP_PROBE, expected_probes(clap=True))

    def test_requesting_clap_and_not_emitting_it_is_red(self):
        """Opting in has to mean the probe ran, not that it was intended."""
        report = judge(_passing(), expected_probes(clap=True))
        self.assertFalse(report.green)
        self.assertEqual(report.missing, (CLAP_PROBE,))

    def test_the_summary_names_what_went_wrong(self):
        probes = _passing()[1:]
        probes[0] = Probe(probes[0].name, ProbeStatus.FAIL, "allocation raised")
        summary = judge(probes, REQUIRED_PROBES).summary()
        self.assertIn("RED", summary)
        self.assertIn("allocation raised", summary)
        self.assertIn("missing", summary)

    def test_the_report_serialises_for_env_report_json(self):
        payload = judge(_passing(), REQUIRED_PROBES).as_dict()
        self.assertTrue(payload["green"])
        self.assertEqual(len(payload["probes"]), len(REQUIRED_PROBES))
        self.assertIn("python", payload["environment"])


class TestAssertEnvRaisesOnAnythingButGreen(unittest.TestCase):
    """``assert_env`` is ``judge`` plus a raise, so the raise is what is left to pin.

    The raise is ``assert_green``, split out precisely so it can be called here with an
    injected report. Staging a ``raise`` inside the test instead would pass whether or
    not the code raised at all.
    """

    def test_a_red_report_raises_with_the_summary_attached(self):
        report = EnvReport(probes=(Probe("x", ProbeStatus.FAIL, "broke"),), missing=("y",))
        with self.assertRaises(EnvCheckError) as caught:
            assert_green(report)
        self.assertIn("broke", str(caught.exception))
        self.assertIn("y", str(caught.exception))

    def test_a_not_run_probe_raises_just_like_a_failure(self):
        """Ticket 13's bug, at the last line where it could still be caught."""
        report = judge([Probe("x", ProbeStatus.NOT_RUN, "torch did not import")], {"x"})
        with self.assertRaises(EnvCheckError):
            assert_green(report)

    def test_a_green_report_is_returned_so_the_caller_can_record_it(self):
        report = judge(_passing(), REQUIRED_PROBES)
        self.assertIs(assert_green(report), report)

    def test_assert_env_actually_calls_the_raise(self):
        """The composition, checked statically because running it needs a box.

        Without this, ``assert_green`` could be perfect and unreferenced: ``assert_env``
        would compute the right answer and hand it back green. That is not a
        hypothetical failure mode — it is exactly what the old gate's torch layer did,
        and it is the reason this module exists in its present shape.
        """
        import ast
        import pathlib

        source = (
            pathlib.Path(__file__).resolve().parents[2] / "env_check.py"
        ).read_text(encoding="utf-8")
        tree = ast.parse(source)
        body = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "assert_env"
        )
        called = {
            node.func.id
            for node in ast.walk(body)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("assert_green", called)
        self.assertIn("run_probes", called)


class TestTheProvenanceComparison(unittest.TestCase):
    """The bootstrap-time half. A capability probe cannot see any of these."""

    def test_a_constraint_on_a_package_nothing_installs_is_inert(self):
        """Ticket 17's named footgun: a misspelled pin reports success while enforcing
        nothing, and only this comparison can see it."""
        result = compare_resolved_against_constraints(
            {"soundfle": "0.13.1", "numpy": "1.23.5"}, {"numpy": "1.23.5"}
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.inert, ("soundfle",))
        self.assertEqual(result.skew, ())

    def test_a_resolver_that_ignored_a_pin_is_skew(self):
        result = compare_resolved_against_constraints(
            {"transformers": "4.57.6"}, {"transformers": "4.58.0"}
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.skew, (("transformers", "4.57.6", "4.58.0"),))

    def test_a_local_version_suffix_is_not_skew(self):
        """PEP 440: ``torch==2.2.2`` matches a resolved ``2.2.2+cu118``. Comparing the
        strings raw would report skew on the one package this env is most careful
        about."""
        result = compare_resolved_against_constraints({"torch": "2.2.2"}, {"torch": "2.2.2+cu118"})
        self.assertTrue(result.ok)

    def test_names_compare_on_the_pep_503_canonical_form(self):
        """pip spells distribution names inconsistently across versions, so
        ``numpy_quaternion`` and ``numpy-quaternion`` are one package."""
        pins = parse_pins("numpy_quaternion==2023.0.4\n")
        resolved = parse_resolved("numpy-quaternion==2023.0.4\n")
        self.assertTrue(compare_resolved_against_constraints(pins, resolved).ok)

    def test_comments_and_blank_lines_are_not_pins(self):
        pins = parse_pins("# torch==9.9.9 in a comment\n\nnumpy==1.23.5  # trailing\n")
        self.assertEqual(pins, {"numpy": "1.23.5"})

    def test_a_source_install_line_is_dropped_rather_than_compared(self):
        """``pip freeze`` records habitat-sim as ``habitat-sim @ file:///root/…``, which
        has no version to compare and cannot be reinstalled from."""
        resolved = parse_resolved(
            "habitat-sim @ file:///root/ss2-build/habitat-sim\nnumpy==1.23.5\n"
        )
        self.assertEqual(resolved, {"numpy": "1.23.5"})

    def test_the_summary_says_which_pin_did_what(self):
        summary = compare_resolved_against_constraints(
            {"soundfle": "0.13.1", "transformers": "4.57.6"}, {"transformers": "4.58.0"}
        ).summary()
        self.assertIn("INERT PIN", summary)
        self.assertIn("SKEW", summary)


class TestTheRealConstraintsFileParses(unittest.TestCase):
    """The pin the bootstrap actually passes as ``-c``, read rather than fabricated.

    A parser tested only against hand-written fixtures is a parser tested against its
    own author's assumptions. This is the one file it has to handle.
    """

    def test_the_nine_pins_are_all_read(self):
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[2] / "tools" / "ss2-constraints.txt"
        )
        pins = parse_pins(path.read_text(encoding="utf-8"))
        self.assertEqual(len(pins), 9, "ticket 17 pins nine packages; parsed {}".format(sorted(pins)))
        self.assertEqual(pins["numpy"], "1.23.5")
        self.assertEqual(pins["torch"], "2.2.2")

    def test_the_real_pins_pass_against_a_matching_freeze(self):
        import pathlib

        path = (
            pathlib.Path(__file__).resolve().parents[2] / "tools" / "ss2-constraints.txt"
        )
        pins = parse_pins(path.read_text(encoding="utf-8"))
        freeze = "\n".join("{}=={}".format(name, version) for name, version in pins.items())
        self.assertTrue(compare_resolved_against_constraints(pins, parse_resolved(freeze)).ok)


class TestTheCli(unittest.TestCase):
    """``python -m earshot.env_check`` is what ``bootstrap_ss2.sh`` runs.

    Output is captured rather than let through. Box tests print their measurements
    (ADR-0014); Mac tests do not, and an argparse usage block in the middle of a green
    suite is the kind of noise that trains a reader to skim past a real one.
    """

    def _run(self, argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            try:
                return main(argv), buffer.getvalue()
            except SystemExit as exit_code:  # argparse's own exit path
                return exit_code, buffer.getvalue()

    def test_provenance_needs_both_paths(self):
        code, _ = self._run(["--provenance", "--constraints", "x"])
        self.assertIsInstance(code, SystemExit)

    def test_an_unreadable_freeze_is_unverified_and_that_is_not_green(self):
        """The bootstrap's own comment: "provenance unverified, which is not the same as
        verified". Under ``--strict`` that has to be a non-zero exit."""
        code, output = self._run(
            ["--provenance", "--strict", "--constraints", "/nope", "--freeze", "/nope"]
        )
        self.assertEqual(code, 1)
        self.assertIn("UNVERIFIED", output)

    def test_without_strict_an_unreadable_freeze_does_not_kill_the_build(self):
        """The recipe already succeeded by that point, and killing a 40-minute build
        over an unjudged skew would destroy the evidence — the printed diff is the
        deliverable."""
        code, _ = self._run(["--provenance", "--constraints", "/nope", "--freeze", "/nope"])
        self.assertEqual(code, 0)

    def test_a_real_comparison_runs_end_to_end_through_the_cli(self):
        """The path the bootstrap takes, exercised rather than assumed."""
        with tempfile.TemporaryDirectory() as tmp:
            constraints = pathlib.Path(tmp) / "c.txt"
            freeze = pathlib.Path(tmp) / "f.txt"
            constraints.write_text("numpy==1.23.5\nsoundfle==0.13.1\n", encoding="utf-8")
            freeze.write_text("numpy==1.23.5\n", encoding="utf-8")
            code, output = self._run(
                ["--provenance", "--strict", "--constraints", str(constraints),
                 "--freeze", str(freeze)]
            )
            self.assertEqual(code, 1)
            self.assertIn("INERT PIN", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
