"""``audio/onset.py`` — one-shot detection, and §3.1's invariants raising.

The provenance assertions are the point. The `anommxv` matrix ran to completion with
its interrupt firing on the wrong sound, and the check that would have caught it was a
diagnostic read afterwards. These tests are what makes it a stop.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.onset import OnsetState, ProvenanceError, assert_provenance, observe_step

BED = 1e-3
THRESHOLD = 5e-3
TOL = 0.05
T_ANOM = 10


def step(state, index, measured, t_anom=T_ANOM, bed=BED, threshold=THRESHOLD, tol=TOL):
    return observe_step(
        state,
        step=index,
        measured_rms=measured,
        t_anom=t_anom,
        onset_rms=threshold,
        bed_rms=bed,
        tolerance=tol,
    )


class TestDetection(unittest.TestCase):
    def test_it_fires_at_the_first_crossing_and_never_again(self):
        state = OnsetState()
        for index in range(T_ANOM):
            state = step(state, index, BED)
        self.assertFalse(state.fired)
        state = step(state, T_ANOM, BED)  # playing but still quiet
        self.assertFalse(state.fired)
        state = step(state, T_ANOM + 3, 0.02)
        self.assertTrue(state.fired)
        self.assertEqual(state.onset_step, T_ANOM + 3)
        state = step(state, T_ANOM + 4, 0.5)
        self.assertEqual(state.onset_step, T_ANOM + 3)
        self.assertAlmostEqual(state.onset_rms_measured, 0.02)

    def test_the_state_is_frozen_so_a_leak_cannot_carry_an_onset_across_episodes(self):
        state = step(OnsetState(), T_ANOM, 0.5)
        with self.assertRaises(Exception):
            state.onset_step = 3

    def test_pre_onset_readings_are_recorded_for_the_audit(self):
        state = OnsetState()
        for index in range(4):
            state = step(state, index, BED)
        self.assertEqual(state.n_pre_onset_readings, 4)
        self.assertAlmostEqual(state.pre_onset_rms, BED)


class TestProvenanceDuringTheEpisode(unittest.TestCase):
    def test_a_pre_onset_reading_above_the_bed_stops_the_run(self):
        with self.assertRaises(ProvenanceError) as caught:
            step(OnsetState(), 3, BED * 4)
        self.assertIn("pre-onset RMS", str(caught.exception))
        self.assertIn("t_anom", str(caught.exception))

    def test_a_pre_onset_reading_below_the_bed_also_stops_the_run(self):
        """A decayed bed is as much a fabrication as an early source."""
        with self.assertRaises(ProvenanceError):
            step(OnsetState(), 3, BED / 4)

    def test_drift_inside_the_tolerance_passes(self):
        state = step(OnsetState(), 3, BED * (1.0 + TOL / 2.0))
        self.assertEqual(state.n_pre_onset_readings, 1)

    def test_an_early_loud_reading_raises_rather_than_firing(self):
        """This is the `anommxv` break: the interrupt firing before the source plays."""
        with self.assertRaises(ProvenanceError):
            step(OnsetState(), 2, 1.0)

    def test_a_zero_bed_uses_an_absolute_tolerance(self):
        step(OnsetState(), 1, 0.0, bed=0.0, tol=1e-6)
        with self.assertRaises(ProvenanceError):
            step(OnsetState(), 1, 0.5, bed=0.0, tol=1e-6)


class TestProvenanceOnTheRecordedState(unittest.TestCase):
    def test_a_healthy_state_passes(self):
        state = OnsetState(onset_step=12, pre_onset_rms=BED, n_pre_onset_readings=10)
        assert_provenance(state, t_anom=T_ANOM, bed_rms=BED, tolerance=TOL)

    def test_an_impossible_onset_step_raises(self):
        state = OnsetState(onset_step=2, pre_onset_rms=BED, n_pre_onset_readings=2)
        with self.assertRaises(ProvenanceError) as caught:
            assert_provenance(state, t_anom=T_ANOM, bed_rms=BED, tolerance=TOL)
        self.assertIn("ADR-0009", str(caught.exception))

    def test_a_recorded_pre_onset_reading_that_no_longer_matches_the_bed_raises(self):
        state = OnsetState(onset_step=12, pre_onset_rms=BED * 10, n_pre_onset_readings=4)
        with self.assertRaises(ProvenanceError):
            assert_provenance(state, t_anom=T_ANOM, bed_rms=BED, tolerance=TOL)

    def test_an_unexercised_invariant_reports_itself_as_unverified(self):
        """Ticket 16's log canary in a different costume: no evidence is not green."""
        state = OnsetState(onset_step=12, n_pre_onset_readings=0)
        with self.assertRaises(ProvenanceError) as caught:
            assert_provenance(state, t_anom=T_ANOM, bed_rms=BED, tolerance=TOL)
        self.assertIn("unverified, not satisfied", str(caught.exception))

    def test_t_anom_zero_needs_no_pre_onset_reading(self):
        assert_provenance(OnsetState(), t_anom=0, bed_rms=BED, tolerance=TOL)


if __name__ == "__main__":
    unittest.main(verbosity=2)
