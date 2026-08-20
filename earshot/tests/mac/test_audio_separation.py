"""The CLAP separation gate's arithmetic, and the batched render it depends on.

Two things are under test and they fail in different ways.

`audio/separation.py` is the gate's verdict. The failure it must not have is the one
`CONTEXT.md` names against the `anommxv` arc: reporting a number when only one arm ran. So
the tests below check the RAISES as hard as they check the arithmetic -- a summariser that
returns 1.0 accuracy over a run with no negatives is worse than one that crashes.

`task/clap_gate.render_batch_through_ir` is a duplicate of `clips.render_through_ir` that
exists purely for speed. A duplicate of a load-bearing signal function is a liability unless
its identity to the original is asserted, so that identity is a test rather than a comment.
"""

import unittest

import numpy as np
from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.clips import render_through_ir
from earshot.audio.separation import (
    GateRow,
    decision_score_of,
    equal_error_rate,
    prune,
    summarise,
    top1_of,
    true_margin_of,
)
from earshot.task.clap_gate import render_batch_through_ir

CLASSES = ("toilet_flush", "clock_alarm", "keyboard_typing")


def row(true_class, scores, *, distance=3.0, in_vocabulary=True, normal=0.0, recording=0):
    return GateRow(
        true_class=true_class,
        in_vocabulary=in_vocabulary,
        distance_m=distance,
        scene="TESTSCENE",
        recording_index=recording,
        scores=dict(scores),
        normal_cosine=normal,
    )


def confident(true_class, *, distance=3.0, margin=0.3, normal=0.0, recording=0):
    """A row whose true class wins by `margin` over a flat field."""
    scores = {name: 0.1 for name in CLASSES}
    scores[true_class] = 0.1 + margin
    return row(true_class, scores, distance=distance, normal=normal, recording=recording)


def absent_row(name, *, distance=3.0, best=0.1, normal=0.0):
    """A forced-failure row: an absent class, never in the prompt bank."""
    scores = {item: best for item in CLASSES}
    return row(name, scores, distance=distance, in_vocabulary=False, normal=normal)


class TestRowInvariants(unittest.TestCase):
    def test_a_row_with_no_scores_raises(self):
        with self.assertRaises(ValueError):
            row("toilet_flush", {})

    def test_an_in_vocabulary_row_must_have_been_scored(self):
        with self.assertRaises(ValueError):
            row("not_scored", {"toilet_flush": 0.2})

    def test_an_absent_row_that_was_scored_raises(self):
        """The vacuous forced-failure arm, caught at the row rather than in the report."""
        with self.assertRaises(ValueError):
            row("toilet_flush", {"toilet_flush": 0.2}, in_vocabulary=False)

    def test_a_non_finite_distance_raises(self):
        with self.assertRaises(ValueError):
            row("toilet_flush", {name: 0.1 for name in CLASSES}, distance=float("inf"))


class TestScoring(unittest.TestCase):
    def test_top1_is_the_argmax_and_ties_break_deterministically(self):
        self.assertEqual(top1_of(confident("clock_alarm")), "clock_alarm")
        tied = row("clock_alarm", {name: 0.5 for name in CLASSES})
        self.assertEqual(top1_of(tied), sorted(CLASSES)[0])

    def test_true_margin_is_signed(self):
        self.assertAlmostEqual(true_margin_of(confident("clock_alarm", margin=0.3)), 0.3, places=6)
        wrong = row("clock_alarm", {"clock_alarm": 0.1, "toilet_flush": 0.4, "keyboard_typing": 0.2})
        self.assertAlmostEqual(true_margin_of(wrong), -0.3, places=6)

    def test_true_margin_refuses_an_absent_row(self):
        """Returning 0.0 would fold a forced-failure row into the closed-set mean."""
        with self.assertRaises(ValueError):
            true_margin_of(absent_row("chainsaw"))

    def test_decision_score_is_measured_against_the_normal_bank(self):
        self.assertAlmostEqual(
            decision_score_of(confident("clock_alarm", margin=0.3, normal=0.05)),
            0.4 - 0.05,
            places=6,
        )


class TestEqualErrorRate(unittest.TestCase):
    def test_cleanly_separated_arms_give_a_low_eer(self):
        eer, _threshold = equal_error_rate([0.8, 0.9, 0.85], [0.1, 0.05, 0.2])
        self.assertLess(eer, 0.01)

    def test_identical_arms_give_a_chance_eer(self):
        """This is the shape of a gate that discriminates nothing, and it must READ as one."""
        values = [0.3, 0.4, 0.5, 0.6]
        eer, _threshold = equal_error_rate(values, list(values))
        self.assertGreater(eer, 0.4)

    def test_one_empty_arm_raises(self):
        with self.assertRaises(ValueError):
            equal_error_rate([0.5], [])
        with self.assertRaises(ValueError):
            equal_error_rate([], [0.5])


class TestSummarise(unittest.TestCase):
    def _healthy_rows(self):
        rows = []
        for name in CLASSES:
            for recording in range(4):
                rows.append(confident(name, distance=1.5, normal=0.0, recording=recording))
                rows.append(confident(name, distance=7.0, normal=0.0, recording=recording))
        rows += [absent_row("chainsaw", normal=0.3), absent_row("helicopter", normal=0.3)]
        return rows

    def test_a_healthy_run_reports_both_arms(self):
        report = summarise(self._healthy_rows(), n_bands=2)
        self.assertEqual(report.top1_accuracy, 1.0)
        self.assertAlmostEqual(report.chance_accuracy, 1.0 / 3.0, places=6)
        self.assertEqual(report.rejection.n_absent, 2)
        self.assertLess(report.rejection.eer, 0.01)

    def test_no_rows_at_all_raises(self):
        with self.assertRaises(ValueError):
            summarise([])

    def test_a_run_with_no_forced_failure_arm_raises(self):
        """ADR-0014's rule, enforced: a detector ships both arms or it is not green."""
        healthy = [confident(name) for name in CLASSES]
        with self.assertRaises(ValueError) as caught:
            summarise(healthy)
        self.assertIn("forced-failure", str(caught.exception))

    def test_a_run_with_no_healthy_arm_raises(self):
        with self.assertRaises(ValueError):
            summarise([absent_row("chainsaw"), absent_row("helicopter")])

    def test_the_band_curve_separates_near_from_far(self):
        """The whole point of banding: one scalar cannot say where CLAP stopped working."""
        rows = []
        for name in CLASSES:
            for recording in range(4):
                rows.append(confident(name, distance=1.2, margin=0.4, recording=recording))
                # Far rows are WRONG: the true class loses to a neighbour.
                scores = {item: 0.4 for item in CLASSES}
                scores[name] = 0.1
                rows.append(row(name, scores, distance=7.5, recording=recording))
        rows += [absent_row("chainsaw", normal=0.3)]
        report = summarise(rows, n_bands=2)
        self.assertEqual(len(report.per_band), 2)
        near, far = report.per_band[0], report.per_band[-1]
        self.assertGreater(near.top1_accuracy, far.top1_accuracy)
        self.assertEqual(near.top1_accuracy, 1.0)
        self.assertEqual(far.top1_accuracy, 0.0)

    def test_every_row_lands_in_exactly_one_band(self):
        """The farthest row must not fall off the end of the last half-open band."""
        rows = [
            confident(name, distance=distance, recording=index)
            for name in CLASSES
            for index, distance in enumerate((1.0, 2.5, 5.0, 8.0))
        ]
        rows += [absent_row("chainsaw", normal=0.3)]
        report = summarise(rows, n_bands=4)
        self.assertEqual(sum(band.n for band in report.per_band), len(CLASSES) * 4)

    def test_the_confusion_matrix_records_what_it_was_mistaken_for(self):
        rows = []
        for recording in range(4):
            scores = {name: 0.1 for name in CLASSES}
            scores["keyboard_typing"] = 0.5
            rows.append(row("clock_alarm", scores, recording=recording))
        rows += [confident("toilet_flush"), absent_row("chainsaw", normal=0.3)]
        report = summarise(rows)
        by_name = {item.name: item for item in report.per_class}
        self.assertEqual(by_name["clock_alarm"].recall, 0.0)
        self.assertEqual(by_name["clock_alarm"].top_confusion, ("keyboard_typing", 4))

    def test_affinity_is_carried_through_to_the_per_class_table(self):
        rows = [confident(name, recording=index) for index, name in enumerate(CLASSES)]
        rows += [absent_row("chainsaw", normal=0.3)]
        report = summarise(rows, affinities={"toilet_flush": "strong"})
        by_name = {item.name: item.affinity for item in report.per_class}
        self.assertEqual(by_name["toilet_flush"], "strong")
        self.assertEqual(by_name["clock_alarm"], "unknown")


class TestPrune(unittest.TestCase):
    def test_a_class_below_the_bar_is_cut_and_one_above_is_kept(self):
        rows = []
        for recording in range(8):
            rows.append(confident("toilet_flush", recording=recording))
            wrong = {name: 0.1 for name in CLASSES}
            wrong["toilet_flush"] = 0.5
            rows.append(row("clock_alarm", wrong, recording=recording))
        rows += [absent_row("chainsaw", normal=0.3)]
        report = summarise(rows)
        kept, cut = prune(report, min_recall=0.5, recall_level="class", min_n=4)
        self.assertIn("toilet_flush", kept)
        self.assertIn("clock_alarm", cut)

    def test_a_class_with_too_few_rows_is_cut_however_good_its_recall(self):
        """Dropped for want of data, not for want of separation. The caller prints which."""
        rows = [confident("toilet_flush", recording=index) for index in range(3)]
        rows += [confident("clock_alarm", recording=index) for index in range(8)]
        rows += [absent_row("chainsaw", normal=0.3)]
        report = summarise(rows)
        kept, cut = prune(report, min_recall=0.5, recall_level="class", min_n=8)
        self.assertIn("toilet_flush", cut)
        self.assertIn("clock_alarm", kept)


class TestBatchedRender(unittest.TestCase):
    """The gate's fast path must be the slow path, exactly."""

    def _ir(self, length=257, seed=7):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((2, length)).astype(np.float32)

    def test_batched_equals_one_at_a_time(self):
        ir = self._ir()
        rng = np.random.default_rng(11)
        clips = [rng.standard_normal(600).astype(np.float32) for _ in range(5)]
        batched = render_batch_through_ir(ir, clips)
        for clip, got in zip(clips, batched):
            expected = render_through_ir(ir, clip)
            self.assertEqual(got.shape, expected.shape)
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_clips_of_different_lengths_are_grouped_not_assumed(self):
        """A fixed-width assumption is what ticket 06 measured to be wrong about the IR."""
        ir = self._ir(length=193)
        rng = np.random.default_rng(3)
        clips = [rng.standard_normal(size).astype(np.float32) for size in (400, 700, 400)]
        batched = render_batch_through_ir(ir, clips)
        self.assertEqual([item.shape[1] for item in batched], [400, 700, 400])
        for clip, got in zip(clips, batched):
            np.testing.assert_allclose(got, render_through_ir(ir, clip), rtol=1e-4, atol=1e-5)

    def test_an_empty_clip_raises(self):
        with self.assertRaises(ValueError):
            render_batch_through_ir(self._ir(), [np.zeros(0, dtype=np.float32)])

    def test_a_mono_ir_raises_rather_than_being_averaged(self):
        """`as_binaural`'s rule, inherited: a mono IR means the channel layout did not take."""
        with self.assertRaises(ValueError):
            render_batch_through_ir(np.zeros((1, 64), dtype=np.float32), [np.ones(32, dtype=np.float32)])


if __name__ == "__main__":
    unittest.main()
