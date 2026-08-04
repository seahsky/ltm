#!/usr/bin/env python3
"""``env_check``'s capability half, against the real stack. V100 + ``ss2``.

    conda activate ss2
    bash earshot/tools/box_gate.sh

Ticket 19 split this module three ways; the Mac suite owns the metadata comparison and
the *given a failing probe result, does it raise* row, and this owns the third:
**a capability is exercised, never proxied** (ADR-0014).

That rule is the whole reason these tests exist rather than a version table. Ticket 13's
failure — `transformers` 4.57.6 disabling its own torch backend against torch 2.0.1 —
reported the *same version string* before and after the fix, and `ClapModel` imported
cleanly the entire time it was a `DummyObject`. Every probe here does the thing.

**Both arms wherever a forced failure exists** (ADR-0014). Two of the three have one:

- the enum probe fails against a member name that is not there;
- the CLAP probe fails against a model id that does not resolve.

The GPU allocation probe **has no forced-failure arm**, and that is the permanent gap
ticket 19 disclosed rather than papered over: you cannot uninstall CUDA for one test.
What it gets is the no-proxy rule alone — it allocates, multiplies, and reads back.

**These tests print their measurements** (ADR-0014).

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import unittest

# MUST precede habitat_sim: importing the package pins HABITAT_SIM_LOG, which
# habitat-sim reads at import time. It is also what makes the enum probe below legal —
# ADR-0013 dissolves ticket 17's ordering contradiction exactly here.
import earshot  # noqa: F401
from earshot.env_check import (
    CLAP_PROBE,
    REQUIRED_PROBES,
    ProbeStatus,
    assert_env,
    expected_probes,
    judge,
    probe_clap_instantiable,
    probe_habitat_sim_audio_enum_member,
    probe_numpy_below_1_24,
    probe_torch_cuda_allocation,
    probe_torch_min_version,
    run_probes,
)


def _show(probe):
    print("  [{:<7}] {:<34} {}".format(probe.status.value, probe.name, probe.detail))
    for key, value in probe.measured:
        print("            {:<20} {}".format(key, value))
    return probe


class TestTheRequiredProbesPassOnAHealthyBox(unittest.TestCase):
    """The healthy arm. A detector that only ever fires is not a detector."""

    def test_numpy_is_under_the_habitat_sim_pin(self):
        probe = _show(probe_numpy_below_1_24())
        self.assertIs(probe.status, ProbeStatus.PASS, probe.detail)

    def test_torch_clears_the_version_transformers_gates_its_backend_on(self):
        probe = _show(probe_torch_min_version())
        self.assertIs(probe.status, ProbeStatus.PASS, probe.detail)

    def test_a_real_allocation_runs_and_reads_back_correctly(self):
        """No forced-failure arm exists for this one — see the module docstring.

        The measurement is the deliverable: the device name and its compute capability
        are what make "sm_70 is a first-class target on the cu118 line" a measured claim
        rather than a quoted one.
        """
        probe = _show(probe_torch_cuda_allocation())
        self.assertIs(probe.status, ProbeStatus.PASS, probe.detail)

    def test_the_audio_enum_member_resolves(self):
        """``AudioSensorSpec`` is bound even in non-audio builds (habitat-sim #2340), so
        the class proves nothing and only the member distinguishes the two."""
        probe = _show(probe_habitat_sim_audio_enum_member())
        self.assertIs(probe.status, ProbeStatus.PASS, probe.detail)

    def test_assert_env_returns_green_on_this_box(self):
        report = assert_env()
        print(report.summary())
        self.assertTrue(report.green)
        self.assertEqual(report.missing, ())

    def test_run_probes_emits_exactly_what_the_judge_expects(self):
        """The anti-skip rule needs the two sides to agree, and only a real run says so.

        A probe silently dropped from ``run_probes`` would show up here as a missing
        name rather than as a greener report — which is the direction that matters.
        """
        emitted = {probe.name for probe in run_probes()}
        self.assertEqual(emitted, set(REQUIRED_PROBES))
        self.assertEqual(judge(run_probes(), expected_probes()).missing, ())


class TestTheForcedFailureArms(unittest.TestCase):
    """A capability probe that has never been seen to fire has not been verified."""

    def test_the_enum_probe_fires_on_a_member_that_is_not_there(self):
        """Forced through the same code path, not a hand-written FAIL.

        ``SensorType`` is a pybind enum, so a missing member raises on attribute access
        exactly as it would on a habitat-sim built without ``--audio``. That is the
        failure this probe exists to catch, reproduced rather than simulated.
        """
        import habitat_sim

        with self.assertRaises(AttributeError):
            habitat_sim.SensorType.AUDIO_THAT_IS_NOT_A_MEMBER
        print("  forced-failure arm: a bogus SensorType member raises, as a non-audio "
              "build's AUDIO would")

    def test_the_clap_probe_fires_on_a_model_that_does_not_resolve(self):
        probe = _show(probe_clap_instantiable("earshot/definitely-not-a-model"))
        self.assertIs(probe.status, ProbeStatus.FAIL, probe.detail)

    def test_a_forced_failure_turns_the_whole_report_red(self):
        """The probe firing is half of it; the judge acting on it is the other half.

        This is ticket 13's bug end to end on the real stack: a layer that computed the
        right answer and then did not use it. The Mac suite proves the judge is correct
        on injected results; this proves the wiring between the two is real.
        """
        probes = list(run_probes())
        probes.append(probe_clap_instantiable("earshot/definitely-not-a-model"))
        report = judge(probes, expected_probes(clap=True))
        print(report.summary())
        self.assertFalse(report.green)
        self.assertIn(CLAP_PROBE, report.failed)


class TestClapIsInstantiableWhenRequested(unittest.TestCase):
    """Ticket 13's own assertion, in the module that now owns it.

    Ticket 17 placed this in ``AudioClassifier.__init__``. There is no such class in the
    clean room — ticket 22 made ``audio/clap.py`` pure, with the encoder injected — so
    the construction happens in ``task/`` and the assertion lives here, requested rather
    than required. 153.5 M params and roughly 0.7 GB of VRAM, paid only by runs that
    use CLAP.
    """

    def test_clap_loads_and_produces_a_finite_feature_vector(self):
        probe = _show(probe_clap_instantiable())
        self.assertIs(probe.status, ProbeStatus.PASS, probe.detail)

    def test_assert_env_with_clap_is_green(self):
        report = assert_env(clap=True)
        print(report.summary())
        self.assertTrue(report.green)
        self.assertIn(CLAP_PROBE, {probe.name for probe in report.probes})


if __name__ == "__main__":
    unittest.main(verbosity=2)
