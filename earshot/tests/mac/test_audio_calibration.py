"""``audio/calibration.py`` — where ``onset_rms`` comes from, and when it refuses.

The arithmetic is Mac-testable because the sweep injects its renderer. What is *not*
settled here is whether a real scene separates at all: that is the gate's job on the
box, and the number it prints is the deliverable.
"""

import math
import unittest

import numpy as np

import _audio_fakes as fakes
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.calibration import (
    CUE_PHASE_AGGREGATION,
    CalibrationError,
    band_poses,
    calibrate_onset,
    render_scatter_of,
    sweep_anomaly_rms,
    sweep_cue_rms,
    sweep_loop_scatter,
    sweep_render_scatter,
)
from earshot.audio.clips import render_through_ir, rms
from earshot.audio.tail import cue_crest, cue_level, cue_min_ratio, steady_state_render


class TestCalibrate(unittest.TestCase):
    def test_the_threshold_sits_strictly_between_the_two_distributions(self):
        result = calibrate_onset(0.001, [0.05, 0.08, 0.2, 0.11])
        self.assertGreater(result.onset_rms, result.bed_rms)
        self.assertLess(result.onset_rms, result.anomaly_low)

    def test_the_placement_is_geometric_so_the_margin_is_symmetric_in_db(self):
        """The arithmetic midpoint of 0.001 and 0.1 sits 34 dB above the bed and 6 dB
        below the anomaly, which is not "between" in the sense that matters."""
        result = calibrate_onset(0.001, [0.1] * 8)
        self.assertAlmostEqual(result.onset_rms, math.sqrt(0.001 * 0.1), places=9)
        up = 20.0 * math.log10(result.onset_rms / result.bed_rms)
        down = 20.0 * math.log10(result.anomaly_low / result.onset_rms)
        self.assertAlmostEqual(up, down, places=6)

    def test_the_gate_number_is_the_separation_in_db(self):
        result = calibrate_onset(0.001, [0.01] * 5)
        self.assertAlmostEqual(result.separation_db, 20.0, places=6)
        self.assertTrue(result.passed)
        self.assertEqual(result.n_poses, 5)

    def test_the_low_percentile_is_used_not_the_minimum(self):
        """One pose behind a closed door is §2.5's attrition, not the threshold's job."""
        samples = [0.0011] + [0.2] * 19
        result = calibrate_onset(0.001, samples)
        self.assertGreater(result.anomaly_low, 0.0011)
        self.assertEqual(result.anomaly_min, 0.0011)

    def test_overlap_fails_the_gate_and_names_the_correction(self):
        with self.assertRaises(CalibrationError) as caught:
            calibrate_onset(0.05, [0.01, 0.02, 0.04])
        message = str(caught.exception)
        self.assertIn("OVERLAP", message)
        self.assertIn("globalVolume", message)
        self.assertNotIn("lower the threshold", message)

    def test_too_little_daylight_fails_even_though_a_threshold_would_fit(self):
        """There is room for a threshold but not for a margin, so a small drift
        re-crosses it."""
        with self.assertRaises(CalibrationError) as caught:
            calibrate_onset(0.01, [0.015] * 6)
        self.assertIn("dB", str(caught.exception))
        self.assertIn("globalVolume", str(caught.exception))

    def test_a_failed_gate_raises_rather_than_returning_a_result(self):
        """A caller who can carry on past a failed gate is a caller who will."""
        with self.assertRaises(CalibrationError):
            calibrate_onset(0.05, [0.05])

    def test_an_empty_sweep_is_an_error_not_a_default(self):
        with self.assertRaises(CalibrationError):
            calibrate_onset(0.001, [])

    def test_a_zero_bed_is_rejected(self):
        with self.assertRaises(CalibrationError) as caught:
            calibrate_onset(0.0, [0.1] * 4)
        self.assertIn("bed_rms", str(caught.exception))

    def test_the_result_serialises_for_the_audit_record(self):
        record = calibrate_onset(0.001, [0.1] * 4).as_dict()
        self.assertEqual(record["n_poses"], 4)
        self.assertIn("separation_db", record)
        self.assertIn("global_volume", record)


class TestSweep(unittest.TestCase):
    def test_one_render_per_pose_measured_on_the_received_signal(self):
        """Not on the IR's own energy: the threshold is applied to what the agent hears,
        and the two differ by the clip's level."""
        rendered = []

        def render_at(pose):
            rendered.append(pose)
            return fakes.synthetic_ir(left=pose, right=pose)

        clip = np.ones(128, dtype=np.float32) * 0.1
        levels = sweep_anomaly_rms([1.0, 0.5, 0.25], render_at, clip)
        self.assertEqual(rendered, [1.0, 0.5, 0.25])
        self.assertEqual(len(levels), 3)
        self.assertGreater(levels[0], levels[1])
        self.assertGreater(levels[1], levels[2])

    def test_the_sweep_measures_the_anomaly_alone(self):
        """The bed is the other side of the comparison, so mixing it in here would
        compare the bed against itself."""
        clip = np.ones(64, dtype=np.float32)
        level = sweep_anomaly_rms([1.0], lambda _: fakes.synthetic_ir(), clip)[0]
        self.assertGreater(level, 0.0)
        self.assertNotAlmostEqual(level, rms(np.zeros(64)))


class TestTheControlArms(unittest.TestCase):
    """``sweep_anomaly_rms`` and ``sweep_render_scatter`` are the pre-ADR-0017 arms.

    They lost their ``Optional[hop]`` at ADR-0019 rather than gaining a third domain: an
    Optional that switches measurement domain is the silent unit error this module warns
    about, and a control is only a control while its body is unchanged. Every number on
    disk from before ADR-0017 is one of these two.
    """

    def setUp(self):
        self.clip = (
            np.random.default_rng(3).standard_normal(800) * 0.1
        ).astype(np.float32)

    def test_the_sweep_still_measures_the_bare_whole_clip_render(self):
        """THE CONTROL. Every historic number has to still mean what it meant."""
        irs = [fakes.synthetic_ir(left=g, right=g) for g in (1.0, 0.5, 0.25)]
        expected = [rms(render_through_ir(ir, self.clip)) for ir in irs]
        measured = sweep_anomaly_rms([0, 1, 2], lambda p: irs[p], self.clip)
        self.assertEqual(measured, expected)

        scatter = sweep_render_scatter("p", lambda _: irs[0], self.clip, repeats=3)
        self.assertEqual(scatter, [expected[0]] * 3)

    def test_neither_control_arm_can_be_asked_for_another_domain(self):
        """The split is the point: a caller that wants the loop calls the loop function.

        Passing ``hop`` used to move the measurement to a different signal entirely, and
        the two call sites that did so read identically at the diff.
        """
        with self.assertRaises(TypeError):
            sweep_anomaly_rms([0], lambda _: fakes.synthetic_ir(), self.clip, hop=100)
        with self.assertRaises(TypeError):
            sweep_render_scatter(
                "p", lambda _: fakes.synthetic_ir(), self.clip, repeats=3, hop=100
            )


class TestTheCueArm(unittest.TestCase):
    """ADR-0019: the sweep must place ``onset_rms`` inside the distribution the agent
    will ACTUALLY read, and since the split readout that is ``hop`` samples per step, not
    ``len(clip)``. Calibrating in one domain and thresholding in another is the silent
    unit error ``calibration.py`` warns about one layer down."""

    def setUp(self):
        self.clip = (
            np.random.default_rng(3).standard_normal(800) * 0.1
        ).astype(np.float32)
        self.hop = 100
        self.phase_folds = 8  # 800 // gcd(800, 100)
        self.clip_tail_steps = 14  # ceil((800 + 511) / 100)

    def test_the_cue_sweep_costs_one_live_render_per_pose(self):
        """The habitat bill is what it always was; only the numpy time moved."""
        rendered = []

        def render_at(pose):
            rendered.append(pose)
            return fakes.synthetic_ir(left=pose, right=pose)

        samples = sweep_cue_rms([1.0, 0.5, 0.25], render_at, self.clip, hop=self.hop)
        self.assertEqual(rendered, [1.0, 0.5, 0.25])
        self.assertEqual(len(samples), 3)
        self.assertGreater(samples[0].level, samples[1].level)
        self.assertGreater(samples[1].level, samples[2].level)

    def test_the_cue_level_is_the_clip_readouts_rms_exactly(self):
        """**THE IDENTITY, through the sweep.** This is why ``onset_rms`` did not move.

        The ``phase_folds`` cue windows are disjoint, consecutive and tile the settled
        period an integer number of times, so their quadratic mean equals what
        ``steady_state_render`` returns -- the number the pre-split sweep placed the
        threshold against. Measured ratio 1.000000000000 at this fixture.
        """
        irs = [fakes.synthetic_ir(left=g, right=g) for g in (1.0, 0.5, 0.25)]
        samples = sweep_cue_rms([0, 1, 2], lambda p: irs[p], self.clip, hop=self.hop)
        for sample, ir in zip(samples, irs):
            reference = rms(steady_state_render(ir, self.clip, hop=self.hop))
            self.assertAlmostEqual(sample.level / reference, 1.0, places=9)
            self.assertEqual(len(sample.phases), self.phase_folds)
            self.assertAlmostEqual(cue_level(sample.phases), sample.level, places=12)

    def test_the_phases_are_the_loops_and_carry_the_intermittency(self):
        """A near-flat clip and a burst, both arms, so the crest is a measurement.

        Measured at this fixture: white-ish noise gives crest 1.1850 and min_ratio 0.9030,
        a 60-sample burst on an 800-sample loop gives crest 2.7451 and min_ratio 0.000000.
        """
        burst = np.zeros(800, dtype=np.float32)
        burst[:60] = 1.0
        noisy = sweep_cue_rms(
            [0], lambda _: fakes.synthetic_ir(), self.clip, hop=self.hop
        )[0]
        bursty = sweep_cue_rms(
            [0], lambda _: fakes.synthetic_ir(), burst, hop=self.hop
        )[0]
        print(
            "\n  [calibration] cue crest / min_ratio: noise {:.4f}/{:.4f}  "
            "burst {:.4f}/{:.6f}".format(
                cue_crest(noisy.phases), cue_min_ratio(noisy.phases),
                cue_crest(bursty.phases), cue_min_ratio(bursty.phases)),
            flush=True,
        )
        self.assertLess(cue_crest(noisy.phases), 1.5)
        self.assertGreater(cue_crest(bursty.phases), 2.0)
        self.assertLess(cue_min_ratio(bursty.phases), 0.05)

    def test_the_loop_scatter_re_renders_every_fold_and_returns_both_readouts(self):
        """The scatter must be the spread of the reading THE CLIMB COMPARES.

        ``steady_state_render`` folds one render ``clip_tail_steps + 1`` times, so its
        spread across repeats is ONE render's spread; the runner's loop folds independent
        renders into every reading. So the renders go per FOLD, ``clip_tail_steps +
        repeats`` of them -- the CLIP tail, because it is the longer of the two and
        settling on the shorter would leave the clip arm measuring its own ramp.

        Both readouts come off the SAME folds, which is what makes the clip arm free.
        """
        calls = []

        def render_at(pose):
            calls.append(pose)
            return fakes.synthetic_ir()

        loop = sweep_loop_scatter(
            "one-pose", render_at, self.clip, repeats=4, hop=self.hop
        )
        self.assertEqual(calls, ["one-pose"] * (self.clip_tail_steps + 4))
        self.assertEqual(len(loop.cue), 4)
        self.assertEqual(len(loop.clip), 4)

        # ...and the clip arm is the accumulator's own settled reading, not one render's
        expected = rms(steady_state_render(fakes.synthetic_ir(), self.clip, hop=self.hop))
        for sample in loop.clip:
            self.assertAlmostEqual(sample, expected, places=6)

    def test_a_deterministic_renderer_leaves_the_clip_arm_flat_and_the_cue_arm_CYCLING(
        self
    ):
        """**THE CONTROL, and it does not land where the ADR-0017 arm's control landed.**

        Handed the same IR every fold, the CLIP arm agrees to float dust -- measured
        1.198e-16 relative, the last bits of a float32 buffer summed in a different
        rotation each step. The CUE arm does NOT: measured 1.066e-01 relative here and
        2.33 at the box's numbers with a 0.6 s transient, because the cue readout CYCLES
        WITH THE CLIP'S OWN ENVELOPE and consecutive folds are different loop phases.

        So ``cue_render_scatter`` is renderer non-determinism PLUS the loop phase, and for
        a bursty clip the second term dominates entirely. It is still an honest answer to
        "the spread of the reading the climb compares" -- ``is_rising`` really does
        compare readings that cycle -- but it is not what the word "render" suggests, and
        ``cue_phase_crest`` is on the record so the two terms stay separable.

        The isolating arm is the flat clip below: with the envelope removed, both arms go
        to zero.
        """
        loop = sweep_loop_scatter(
            "one-pose", lambda _: fakes.synthetic_ir(), self.clip, repeats=6,
            hop=self.hop,
        )
        clip_level = sum(loop.clip) / len(loop.clip)
        cue_lvl = sum(loop.cue) / len(loop.cue)
        print(
            "\n  [calibration] deterministic renderer, noise clip: "
            "clip SD/level {:.3e}   cue SD/level {:.3e}".format(
                render_scatter_of(loop.clip) / clip_level,
                render_scatter_of(loop.cue) / cue_lvl),
            flush=True,
        )
        self.assertLess(render_scatter_of(loop.clip) / clip_level, 1e-12, loop.clip)
        self.assertGreater(render_scatter_of(loop.cue) / cue_lvl, 0.05, loop.cue)

        # THE ISOLATING ARM: a clip whose energy is flat over the hop has no phase to
        # cycle through, so the cue arm collapses to the clip arm's float dust.
        flat = np.full(800, 0.1, dtype=np.float32)
        flat_loop = sweep_loop_scatter(
            "one-pose", lambda _: fakes.synthetic_ir(), flat, repeats=6, hop=self.hop
        )
        self.assertEqual(render_scatter_of(flat_loop.cue), 0.0)
        self.assertEqual(render_scatter_of(flat_loop.clip), 0.0)

    def test_a_renderer_that_disagrees_with_itself_shows_up_in_BOTH_arms(self):
        """THE HEALTHY ARM, measured on the FLAT clip so the loop phase is not in it.

        With the envelope removed the only moving part is the renderer, and the cue arm's
        spread is strictly the larger: measured SD 6.700e-03 against the clip arm's
        8.364e-04, a ratio of 8.011. The direction is the averaging prediction -- the clip
        readout folds more independent renders into every reading than the cue readout
        does -- but the SIZE is not: an independence argument at ``clip_ramp_steps`` = 8
        predicts about ``sqrt(8)`` = 2.83, and the renderer here is a deterministic
        six-gain cycle rather than a draw, so the excess is not attributable from this
        fixture. The real ordering is the box's to measure, and
        ``CalibrationResult``'s docstring pre-registers it.
        """
        flat = np.full(800, 0.1, dtype=np.float32)
        gains = iter([1.0, 1.02, 0.98, 1.03, 0.97, 1.01] * 20)

        def jittery(_pose):
            return fakes.synthetic_ir(left=next(gains), right=next(gains))

        loop = sweep_loop_scatter(
            "one-pose", jittery, flat, repeats=6, hop=self.hop
        )
        cue_sd = render_scatter_of(loop.cue)
        clip_sd = render_scatter_of(loop.clip)
        print(
            "\n  [calibration] jittery renderer, flat clip: cue SD {:.6e}  "
            "clip SD {:.6e}  cue/clip {:.3f}".format(cue_sd, clip_sd, cue_sd / clip_sd),
            flush=True,
        )
        self.assertGreater(clip_sd, 0.0)
        self.assertGreater(cue_sd, clip_sd)

    def test_fewer_than_two_repeats_is_refused_on_the_loop_arm_too(self):
        with self.assertRaises(CalibrationError):
            sweep_loop_scatter(
                "p", lambda _: fakes.synthetic_ir(), self.clip, repeats=1, hop=self.hop
            )

    def test_the_phases_reach_the_record_without_moving_the_threshold(self):
        """``cue_phases`` rides along the way ``profile`` does: recorded, never gating.

        Called twice, with and without, and ``onset_rms`` is identical -- which is the
        whole review claim of ADR-0019 reduced to one assertion.
        """
        phases = [(0.9, 1.1, 1.0, 1.0, 1.0), (0.8, 1.2, 1.0, 1.0, 1.0)]
        without = calibrate_onset(0.001, [0.01, 0.02, 0.03])
        with_phases = calibrate_onset(
            0.001, [0.01, 0.02, 0.03], cue_phases=phases
        )
        self.assertEqual(without.onset_rms, with_phases.onset_rms)
        self.assertEqual(without.separation_db, with_phases.separation_db)

        self.assertEqual(without.cue_phase_folds, 0)
        self.assertIsNone(without.cue_phase_crest)
        self.assertIsNone(without.cue_phase_min_ratio)
        self.assertIsNone(without.cue_phase_aggregation)

        self.assertEqual(with_phases.cue_phase_folds, 5)
        self.assertEqual(with_phases.cue_phase_aggregation, CUE_PHASE_AGGREGATION)
        # TWO fold counts, because one is writable as the literal 5 -- measured, with the
        # whole suite green. The period is `N // gcd(N, hop)` and the tree already ships
        # configurations at 5, 7, 8 and 14 of them.
        three = calibrate_onset(
            0.001, [0.01, 0.02, 0.03], cue_phases=[(1.0, 2.0, 1.0), (1.0, 1.0, 1.0)]
        )
        self.assertEqual(three.cue_phase_folds, 3)
        self.assertNotEqual(three.cue_phase_folds, with_phases.cue_phase_folds)
        # the MEDIAN over poses of each, not the mean and not the first pose's
        crests = sorted(cue_crest(p) for p in phases)
        self.assertAlmostEqual(
            with_phases.cue_phase_crest, sum(crests) / 2.0, places=12
        )
        mins = sorted(cue_min_ratio(p) for p in phases)
        self.assertAlmostEqual(
            with_phases.cue_phase_min_ratio, sum(mins) / 2.0, places=12
        )

    def test_the_three_scatter_arms_are_three_named_fields(self):
        """A silent redefinition is what the rename exists to prevent.

        ``render_scatter``'s written definition -- "the spread of the reading the climb
        compares" -- stayed true across the split while the reading changed length, which
        is exactly what would have let the domain move under a stable name.
        """
        result = calibrate_onset(
            0.001, [0.01, 0.02, 0.03],
            cue_scatter_samples=[0.009, 0.010, 0.011],
            clip_scatter_samples=[0.0098, 0.0100, 0.0102],
            single_render_samples=[0.008, 0.010, 0.012],
        )
        self.assertAlmostEqual(result.cue_render_scatter, 1e-3, places=12)
        self.assertAlmostEqual(result.clip_render_scatter, 2e-4, places=12)
        self.assertAlmostEqual(result.single_render_scatter, 2e-3, places=12)
        self.assertEqual(result.cue_scatter_repeats, 3)
        self.assertEqual(result.clip_scatter_repeats, 3)
        self.assertEqual(result.single_render_repeats, 3)

        record = result.as_dict()
        self.assertNotIn("render_scatter", record)
        self.assertNotIn("scatter_repeats", record)
        for key in (
            "cue_render_scatter", "clip_render_scatter", "single_render_scatter",
            "cue_scatter_repeats", "clip_scatter_repeats", "single_render_repeats",
            "cue_phase_folds", "cue_phase_crest", "cue_phase_min_ratio",
            "cue_phase_aggregation",
        ):
            self.assertIn(key, record)


class TestBandPoses(unittest.TestCase):
    def test_the_band_is_log_spaced_between_the_endpoints(self):
        distances = band_poses((1.0, 8.0), 4)
        self.assertAlmostEqual(distances[0], 1.0)
        self.assertAlmostEqual(distances[-1], 8.0)
        ratios = [b / a for a, b in zip(distances, distances[1:])]
        for ratio in ratios[1:]:
            self.assertAlmostEqual(ratio, ratios[0], places=9)

    def test_linear_spacing_would_crowd_the_quiet_end(self):
        """The reason log spacing is not a stylistic choice."""
        distances = band_poses((1.0, 8.0), 4)
        self.assertLess(distances[1], (1.0 + 8.0) / 2.0)

    def test_a_bad_band_is_rejected(self):
        with self.assertRaises(ValueError):
            band_poses((8.0, 1.0), 4)
        with self.assertRaises(ValueError):
            band_poses((0.0, 8.0), 4)

    def test_a_distribution_needs_more_than_one_pose(self):
        with self.assertRaises(ValueError):
            band_poses((1.0, 8.0), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
