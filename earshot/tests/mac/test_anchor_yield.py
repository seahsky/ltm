"""`anchor_yield`, over the REAL builder.

The tool answers one question before a night is spent on it: if the sweep drew its sound class
per scene instead of running `alarm` in all 282 episodes, how many sources would land at a
category a memory prior could recall? `abl-2` measured the fixed-class answer at 134 of 282,
which is the number this exists to try to beat.

Its one way to be dangerously wrong is to model the placement instead of running it, so
`cell_yield` calls `build_anomaly_episodes` and the fixtures here are real `EpisodeDataset`s.
Both arms ship (ADR-0014): a scene that HAS the anchor and a scene that does not, a class with
an `anchor_object` row and a class without one, and a scene that can build nothing at all.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401
from _task_fakes import make_episode, make_goal

from earshot.audio.clips import ANOMALY_CLASSES, SOUNDING_CLASSES
from earshot.task.episodes import EpisodeDataset
from earshot.task.prior_build import anchor_of_run_class
from earshot.types import Xyz
from earshot.tools.anchor_yield import (
    ABL2_ALARM_ANCHORED,
    ABL2_ALARM_BUILT,
    SWEEP_N_EPISODES,
    CellYield,
    best_class_per_scene,
    cell_yield,
    fold_by_class,
    format_report,
    main,
)

# `alarm` anchors here; `_a_class_with_no_anchor` below asserts the other arm exists.
ANCHOR_OF_ALARM = "bed"


def scene(*episodes):
    return EpisodeDataset(
        scene_label="FAKE", scene_path="/nonexistent/FAKE.basis.glb", episodes=tuple(episodes)
    )


def goals(category, *positions):
    return [make_goal(position, category=category) for position in positions]


def a_scene_with_a_bed():
    """Two episodes, one goal each, which is the shape the published content files have.

    A category reaches the source ranking through `goal_table`, gathered ACROSS episodes.
    Putting the bed in the chair episode's own goal list instead makes it a primary anchor,
    and then the separation rules reject it as too near its own goal: the first draft of this
    file did that and built nothing.

    It anchors on exactly one of the two: the chair episode can use the bed, and the bed
    episode has only the chair left. So one scene exercises both branches.
    """
    return scene(
        make_episode(episode_id="a", category="chair",
                     goals=goals("chair", Xyz(0.0, 0.0, -9.0))),
        make_episode(episode_id="b", category="bed",
                     goals=goals("bed", Xyz(12.0, 0.0, -9.0))),
    )


def a_scene_with_no_bed():
    return scene(
        make_episode(episode_id="a", category="chair",
                     goals=goals("chair", Xyz(0.0, 0.0, -9.0))),
        make_episode(episode_id="b", category="sofa",
                     goals=goals("sofa", Xyz(12.0, 0.0, -9.0))),
    )


def a_scene_that_builds_nothing():
    """One goal and nothing else, so no candidate clears the separation rules."""
    return scene(make_episode(category="chair", goals=goals("chair", Xyz(0.0, 0.0, 0.0))))


class TestTheLookupHasBothArms(unittest.TestCase):
    """The premise the whole tool rests on: some classes anchor and some do not."""

    def test_alarm_anchors_at_a_category(self):
        self.assertEqual(anchor_of_run_class("alarm"), ANCHOR_OF_ALARM)

    def test_at_least_one_shipped_class_has_no_anchor_row(self):
        """`glass_break` is the one today. If that ever changes, the NONE column in the
        report becomes dead code and the reader should stop claiming to explain a 0%."""
        without = [
            name
            for name in tuple(SOUNDING_CLASSES) + tuple(ANOMALY_CLASSES)
            if anchor_of_run_class(name) is None
        ]
        self.assertTrue(without, "no class lacks an anchor; the NONE branch is unreachable")


class TestCellYieldRunsTheRealBuilder(unittest.TestCase):
    def test_a_scene_with_the_anchor_anchors(self):
        cell = cell_yield(
            a_scene_with_a_bed(), scene="S", anomaly_class="alarm", n_episodes=5
        )
        self.assertEqual(cell.anchor_category, ANCHOR_OF_ALARM)
        self.assertEqual((cell.n_built, cell.n_anchored), (2, 1))
        self.assertEqual(cell.rate, 0.5)

    def test_a_scene_without_the_anchor_still_builds_and_records_zero(self):
        """The forced-failure arm. ADR-0022's preference falls through rather than refusing,
        so yield cannot drop and the cell is a fraction rather than a pass/fail."""
        cell = cell_yield(
            a_scene_with_no_bed(), scene="S", anomaly_class="alarm", n_episodes=5
        )
        self.assertEqual((cell.n_built, cell.n_anchored), (2, 0))
        self.assertEqual(cell.rate, 0.0)

    def test_the_anchor_changes_the_count_and_nothing_else(self):
        """The yield is a property of the scene; the ANCHORED yield is what the class buys.
        Both cells build the same two episodes and only the anchored column moves."""
        with_bed = cell_yield(
            a_scene_with_a_bed(), scene="S", anomaly_class="alarm", n_episodes=5
        )
        without = cell_yield(
            a_scene_with_no_bed(), scene="S", anomaly_class="alarm", n_episodes=5
        )
        self.assertEqual(with_bed.n_built, without.n_built)
        self.assertGreater(with_bed.n_anchored, without.n_anchored)

    def test_a_class_with_no_anchor_row_anchors_nowhere(self):
        cell = cell_yield(
            a_scene_with_a_bed(), scene="S", anomaly_class="glass_break", n_episodes=5
        )
        self.assertIsNone(cell.anchor_category)
        self.assertEqual((cell.n_built, cell.n_anchored), (2, 0))

    def test_a_zero_yield_scene_is_a_cell_and_not_a_raise(self):
        """`mL8ThkuaVTM` builds nothing in any sweep this repo has run, and
        `EmptyDatasetError` carries its whole build precisely so nobody has to lose it."""
        cell = cell_yield(
            a_scene_that_builds_nothing(), scene="S", anomaly_class="alarm", n_episodes=5
        )
        self.assertEqual(cell.n_built, 0)
        self.assertEqual(cell.n_anchored, 0)
        self.assertIsNone(cell.error)

    def test_a_scene_that_built_nothing_has_no_rate_rather_than_zero(self):
        """0 of 0 is not 0.0%, and a reader that printed it would rank a scene that cannot
        pose the task alongside one that posed it and never anchored."""
        cell = cell_yield(
            a_scene_that_builds_nothing(), scene="S", anomaly_class="alarm", n_episodes=5
        )
        self.assertIsNone(cell.rate)


class TestFolding(unittest.TestCase):
    CELLS = (
        CellYield("s1", "alarm", "bed", n_built=10, n_anchored=6, n_skipped=0),
        CellYield("s2", "alarm", "bed", n_built=10, n_anchored=0, n_skipped=0),
        CellYield("s1", "toilet_flush", "toilet", n_built=10, n_anchored=2, n_skipped=0),
        CellYield("s2", "toilet_flush", "toilet", n_built=10, n_anchored=9, n_skipped=0),
    )

    def test_a_class_folds_over_every_scene(self):
        folded = {entry.anomaly_class: entry for entry in fold_by_class(self.CELLS)}
        self.assertEqual(folded["alarm"].n_anchored, 6)
        self.assertEqual(folded["alarm"].n_built, 20)
        self.assertEqual(folded["alarm"].n_scenes_with_any, 1)
        self.assertEqual(folded["alarm"].n_scenes_built, 2)

    def test_the_best_class_is_chosen_per_scene_and_not_globally(self):
        """The matrix may pick its class per scene, so the ceiling is the sum of the best
        per scene and never any one class's total. Here that is 6 + 9 = 15, and no single
        class reaches it."""
        best = best_class_per_scene(self.CELLS)
        self.assertEqual(best["s1"].anomaly_class, "alarm")
        self.assertEqual(best["s2"].anomaly_class, "toilet_flush")
        self.assertEqual(sum(cell.n_anchored for cell in best.values()), 15)

    def test_ties_break_on_class_name_so_the_answer_is_stable(self):
        tied = (
            CellYield("s1", "b_class", "bed", n_built=10, n_anchored=4, n_skipped=0),
            CellYield("s1", "a_class", "bed", n_built=10, n_anchored=4, n_skipped=0),
        )
        self.assertEqual(best_class_per_scene(tied)["s1"].anomaly_class, "b_class")
        self.assertEqual(
            best_class_per_scene(tuple(reversed(tied)))["s1"].anomaly_class, "b_class"
        )


class TestTheReproductionCheck(unittest.TestCase):
    """A build with no seed and no simulator must reproduce a finished run exactly."""

    def _report(self, anchored, built, *, n_episodes=SWEEP_N_EPISODES, split="val"):
        cells = (CellYield("s1", "alarm", "bed", n_built=built, n_anchored=anchored,
                           n_skipped=0),)
        return format_report(cells, scenes=["s1"], n_episodes=n_episodes, split=split)

    def test_matching_abl_2_says_it_agrees(self):
        text = self._report(ABL2_ALARM_ANCHORED, ABL2_ALARM_BUILT)
        self.assertIn("AGREES", text)

    def test_missing_abl_2_says_the_defect_is_here(self):
        """The wrong reading is "abl-2 was wrong". A finished run is evidence; this is not."""
        text = self._report(ABL2_ALARM_ANCHORED + 1, ABL2_ALARM_BUILT)
        self.assertIn("DISAGREES", text)
        self.assertIn("NOT a second opinion", text)

    def test_other_settings_skip_rather_than_compare_two_questions(self):
        text = self._report(1, 2, n_episodes=3)
        self.assertIn("SKIPPED", text)
        self.assertNotIn("DISAGREES", text)

    def test_another_split_also_skips(self):
        text = self._report(1, 2, split="train")
        self.assertIn("SKIPPED", text)


class TestTheReport(unittest.TestCase):
    def test_a_class_with_no_anchor_prints_NONE_and_is_explained(self):
        cells = (CellYield("s1", "glass_break", None, n_built=10, n_anchored=0, n_skipped=0),)
        text = format_report(cells, scenes=["s1"], n_episodes=15, split="val")
        self.assertIn("NONE", text)
        self.assertIn("is the design", text)

    def test_a_barren_scene_is_named_as_measured_not_dropped(self):
        cells = (CellYield("s1", "alarm", "bed", n_built=0, n_anchored=0, n_skipped=9),)
        text = format_report(cells, scenes=["s1"], n_episodes=15, split="val")
        self.assertIn("zero yield, measured", text)

    def test_the_ceiling_disclaimer_is_always_printed(self):
        cells = (CellYield("s1", "alarm", "bed", n_built=10, n_anchored=5, n_skipped=0),)
        text = format_report(cells, scenes=["s1"], n_episodes=15, split="val")
        self.assertIn("CEILING, NOT A RESULT", text)


class TestTheExitCode(unittest.TestCase):
    def test_a_bad_episode_count_is_two(self):
        self.assertEqual(main(["--n-episodes", "0"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
