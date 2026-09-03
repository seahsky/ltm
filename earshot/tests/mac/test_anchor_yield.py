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

import collections
import os
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401
from _task_fakes import make_episode, make_goal

from earshot.audio.clips import ANOMALY_CLASSES, SOUNDING_CLASSES
from earshot.task.episodes import EpisodeDataset
from earshot.task.prior_build import anchor_of_run_class
from earshot.types import Xyz
from earshot.audio.vocabulary import ROOM_OF_ANCHOR
from earshot.tools.anchor_yield import (
    ABL2_ALARM_ANCHORED,
    ABL2_ALARM_BUILT,
    SWEEP_N_EPISODES,
    CellYield,
    anchors_by_scene,
    anchors_without_a_room,
    balanced_assignment,
    best_class_per_scene,
    cell_yield,
    constant_predictor_share,
    fold_by_class,
    format_report,
    main,
    room_assignment_detail,
    rooms_by_scene,
    write_assignment_tsv,
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


class TestTheClassWithinAnAnchorIsFree(unittest.TestCase):
    """Placement reads the anchor CATEGORY and nothing else about the class.

    Load-bearing for ADR-0018, not a curiosity: it means the heard/not-heard axis can swap
    the class while holding the episode fixed, so the delta is the association rather than a
    different task. `anchors_by_scene` collapses classes onto anchors on this basis, and a
    day when it stops being true is a day that collapse becomes a silent averaging.
    """

    def test_two_classes_at_one_anchor_build_identically(self):
        scene_with_bed = a_scene_with_a_bed()
        first = cell_yield(scene_with_bed, scene="S", anomaly_class="alarm", n_episodes=5)
        second = cell_yield(scene_with_bed, scene="S", anomaly_class="snoring", n_episodes=5)
        self.assertEqual(first.anchor_category, second.anchor_category)
        self.assertEqual(
            (first.n_built, first.n_anchored), (second.n_built, second.n_anchored)
        )

    def test_collapsing_onto_anchors_keeps_the_count(self):
        cells = (
            CellYield("s1", "alarm", "bed", n_built=10, n_anchored=6, n_skipped=0),
            CellYield("s1", "snoring", "bed", n_built=10, n_anchored=6, n_skipped=0),
            CellYield("s1", "toilet_flush", "toilet", n_built=10, n_anchored=2, n_skipped=0),
        )
        self.assertEqual(anchors_by_scene(cells), {"s1": {"bed": 6, "toilet": 2}})

    def test_a_class_with_no_anchor_contributes_no_column(self):
        cells = (CellYield("s1", "glass_break", None, n_built=10, n_anchored=0, n_skipped=0),)
        self.assertEqual(anchors_by_scene(cells), {})


class TestTheConstantPredictor(unittest.TestCase):
    """The number that makes a greedy assignment a trap rather than a win."""

    def test_it_names_the_commonest_anchor_and_its_share(self):
        assignment = {"s1": ("chair", 90), "s2": ("chair", 80), "s3": ("bed", 30)}
        anchor, share, overall = constant_predictor_share(assignment)
        self.assertEqual((anchor, share, overall), ("chair", 170, 200))

    def test_an_even_assignment_leaves_the_null_hypothesis_less(self):
        """The whole point. Same tool, two designs, and the weaker design scores HIGHER."""
        lopsided = {"s1": ("chair", 90), "s2": ("chair", 80), "s3": ("bed", 30)}
        even = {"s1": ("chair", 50), "s2": ("bed", 50), "s3": ("toilet", 50)}
        self.assertGreater(
            constant_predictor_share(lopsided)[1] / constant_predictor_share(lopsided)[2],
            constant_predictor_share(even)[1] / constant_predictor_share(even)[2],
        )

    def test_an_empty_assignment_is_zero_of_zero_and_not_a_crash(self):
        self.assertEqual(constant_predictor_share({}), ("", 0, 0))


class TestTheBalancedAssignment(unittest.TestCase):
    ANCHORS = ("bed", "chair", "toilet", "tv_monitor")

    def test_every_anchor_gets_its_quota(self):
        per_scene = {
            "s{}".format(i): {anchor: 10 for anchor in self.ANCHORS} for i in range(8)
        }
        assignment = balanced_assignment(per_scene, self.ANCHORS)
        counts = collections.Counter(anchor for anchor, _ in assignment.values())
        self.assertEqual(sorted(counts.values()), [2, 2, 2, 2])

    def test_an_uneven_scene_count_splits_within_one(self):
        per_scene = {
            "s{}".format(i): {anchor: 10 for anchor in self.ANCHORS} for i in range(19)
        }
        assignment = balanced_assignment(per_scene, self.ANCHORS)
        counts = collections.Counter(anchor for anchor, _ in assignment.values())
        self.assertEqual(len(assignment), 19)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_within_the_quota_it_maximises_anchored_episodes(self):
        """Balance is the constraint; among balanced answers more episodes still wins."""
        per_scene = {
            "s0": {"bed": 9, "chair": 1},
            "s1": {"bed": 1, "chair": 9},
        }
        assignment = balanced_assignment(per_scene, ("bed", "chair"))
        self.assertEqual(assignment["s0"], ("bed", 9))
        self.assertEqual(assignment["s1"], ("chair", 9))

    def test_it_takes_a_worse_scene_rather_than_break_the_quota(self):
        """The forced-failure arm for the balance rule: greedy would put both on `chair`."""
        per_scene = {
            "s0": {"bed": 2, "chair": 9},
            "s1": {"bed": 3, "chair": 9},
        }
        assignment = balanced_assignment(per_scene, ("bed", "chair"))
        self.assertEqual(sorted(anchor for anchor, _ in assignment.values()),
                         ["bed", "chair"])
        # One scene must take `bed` and lose episodes for it. Which one is the optimiser's
        # call: s0->chair, s1->bed keeps 12 and the other way keeps 11, so it takes the 12.
        self.assertEqual(sum(anchored for _, anchored in assignment.values()), 12)

    def test_a_scene_that_anchors_nothing_anywhere_is_left_out(self):
        per_scene = {"s0": {"bed": 5}, "barren": {"bed": 0, "chair": 0}}
        assignment = balanced_assignment(per_scene, ("bed", "chair"))
        self.assertNotIn("barren", assignment)

    def test_no_scenes_is_an_empty_answer_and_not_a_crash(self):
        self.assertEqual(balanced_assignment({}, self.ANCHORS), {})


class TestTheDiscriminationSection(unittest.TestCase):
    def test_both_assignments_are_printed_with_their_null_scores(self):
        cells = (
            CellYield("s1", "keyboard_typing", "chair", n_built=15, n_anchored=14,
                      n_skipped=0),
            CellYield("s1", "snoring", "bed", n_built=15, n_anchored=3, n_skipped=0),
            CellYield("s2", "keyboard_typing", "chair", n_built=15, n_anchored=13,
                      n_skipped=0),
            CellYield("s2", "snoring", "bed", n_built=15, n_anchored=4, n_skipped=0),
        )
        text = format_report(cells, scenes=["s1", "s2"], n_episodes=15, split="val")
        self.assertIn("GREEDY", text)
        self.assertIn("BALANCED", text)
        self.assertIn("ALWAYS-", text)
        self.assertIn("learned NOTHING", text)

    def test_the_greedy_assignment_hands_the_null_a_perfect_score(self):
        """Two scenes both won by `chair` means an always-chair predictor is never wrong,
        and the balanced answer must not read the same."""
        cells = (
            CellYield("s1", "keyboard_typing", "chair", n_built=15, n_anchored=14,
                      n_skipped=0),
            CellYield("s1", "snoring", "bed", n_built=15, n_anchored=3, n_skipped=0),
            CellYield("s2", "keyboard_typing", "chair", n_built=15, n_anchored=13,
                      n_skipped=0),
            CellYield("s2", "snoring", "bed", n_built=15, n_anchored=4, n_skipped=0),
        )
        greedy = {scene: (cell.anchor_category, cell.n_anchored)
                  for scene, cell in best_class_per_scene(cells).items()}
        balanced = balanced_assignment(anchors_by_scene(cells), ("bed", "chair"))
        self.assertEqual(constant_predictor_share(greedy)[1:],
                         (27, 27))
        self.assertLess(constant_predictor_share(balanced)[1],
                        constant_predictor_share(balanced)[2])


# Six scenes, four anchor objects, THREE rooms. `chair` and `tv_monitor` are both the living
# room, which is the whole subject of the class below: an assignment balanced over four
# objects gives the living room four of the six scenes, and its own object-level null cannot
# see that.
_TWO_ROOM_OBJECTS_CELLS = tuple(
    cell
    for scene in ("s1", "s2", "s3", "s4", "s5", "s6")
    for cell in (
        CellYield(scene, "keyboard_typing", "chair", n_built=15, n_anchored=10, n_skipped=0),
        CellYield(scene, "clapping", "tv_monitor", n_built=15, n_anchored=9, n_skipped=0),
        CellYield(scene, "snoring", "bed", n_built=15, n_anchored=3, n_skipped=0),
        CellYield(scene, "toilet_flush", "toilet", n_built=15, n_anchored=3, n_skipped=0),
    )
)
_SIX_SCENE_OBJECTS = ("bed", "chair", "toilet", "tv_monitor")


def _room_level_share(assignment):
    """Re-key an object-level assignment onto rooms and score the null there."""
    return constant_predictor_share({
        scene: (ROOM_OF_ANCHOR[anchor], anchored)
        for scene, (anchor, anchored) in assignment.items()
    })


class TestTheUnitIsTheRoom(unittest.TestCase):
    """ADR-0018's amendment made the anchor a ROOM. Scoring the null over anchor OBJECTS
    under-states it, and that is what the first `anchor_yield` run on the box printed."""

    def test_two_anchor_objects_really_do_share_a_room(self):
        """Asserted against the live table, so the premise cannot drift out from under the
        rest of this class without a failure here."""
        self.assertEqual(ROOM_OF_ANCHOR["chair"], ROOM_OF_ANCHOR["tv_monitor"])
        self.assertLess(
            len(set(ROOM_OF_ANCHOR.values())), len(set(ROOM_OF_ANCHOR)),
            "the collapse this module corrects for no longer exists",
        )

    def test_a_scene_takes_the_better_object_in_a_room(self):
        """Two objects in one room are one answer, so the scene keeps the better of them."""
        per_room = rooms_by_scene(_TWO_ROOM_OBJECTS_CELLS)
        self.assertEqual(per_room["s1"][ROOM_OF_ANCHOR["chair"]], 10)

    def test_the_rooms_are_fewer_than_the_anchors(self):
        per_object = anchors_by_scene(_TWO_ROOM_OBJECTS_CELLS)
        per_room = rooms_by_scene(_TWO_ROOM_OBJECTS_CELLS)
        self.assertEqual(len(per_object["s1"]), 4)
        self.assertEqual(len(per_room["s1"]), 3)

    def test_the_object_level_null_under_states_the_real_one(self):
        """THE DEFECT, stated as a test. The object-balanced design prints one number and
        a store predicting rooms is scored against a strictly larger one."""
        balanced = balanced_assignment(
            anchors_by_scene(_TWO_ROOM_OBJECTS_CELLS), _SIX_SCENE_OBJECTS
        )
        _obj, obj_share, obj_total = constant_predictor_share(balanced)
        _room, room_share, room_total = _room_level_share(balanced)
        self.assertEqual(obj_total, room_total)
        self.assertGreater(room_share, obj_share)

    def test_balancing_over_rooms_caps_the_room_the_objects_could_not(self):
        """The mechanism: a room quota bounds the dominant room's scene count. Balancing
        over objects has no such bound, because two of them are the same room."""
        rooms = sorted({ROOM_OF_ANCHOR[a] for a in _SIX_SCENE_OBJECTS})
        by_room = balanced_assignment(rooms_by_scene(_TWO_ROOM_OBJECTS_CELLS), rooms)
        by_object = balanced_assignment(
            anchors_by_scene(_TWO_ROOM_OBJECTS_CELLS), _SIX_SCENE_OBJECTS
        )
        living = ROOM_OF_ANCHOR["chair"]
        scenes_by_room = collections.Counter(room for room, _n in by_room.values())
        scenes_by_object = collections.Counter(
            ROOM_OF_ANCHOR[anchor] for anchor, _n in by_object.values()
        )
        self.assertEqual(scenes_by_room[living], 2)     # ceil(6 / 3 rooms)
        self.assertEqual(scenes_by_object[living], 4)   # chair 2 + tv_monitor 2

    def test_balancing_over_rooms_leaves_the_null_lower(self):
        rooms = sorted({ROOM_OF_ANCHOR[a] for a in _SIX_SCENE_OBJECTS})
        by_room = balanced_assignment(rooms_by_scene(_TWO_ROOM_OBJECTS_CELLS), rooms)
        by_object = balanced_assignment(
            anchors_by_scene(_TWO_ROOM_OBJECTS_CELLS), _SIX_SCENE_OBJECTS
        )
        _r, room_share, room_total = constant_predictor_share(by_room)
        _o, object_share, object_total = _room_level_share(by_object)
        self.assertLess(room_share / room_total, object_share / object_total)

    def test_it_buys_that_with_fewer_anchored_episodes(self):
        """The trade is real and the tool must not hide it: balance costs episodes."""
        rooms = sorted({ROOM_OF_ANCHOR[a] for a in _SIX_SCENE_OBJECTS})
        by_room = balanced_assignment(rooms_by_scene(_TWO_ROOM_OBJECTS_CELLS), rooms)
        by_object = balanced_assignment(
            anchors_by_scene(_TWO_ROOM_OBJECTS_CELLS), _SIX_SCENE_OBJECTS
        )
        self.assertLess(
            sum(n for _room, n in by_room.values()),
            sum(n for _anchor, n in by_object.values()),
        )


class TestTheAnchorWithNoRoom(unittest.TestCase):
    def test_plant_is_still_the_live_case(self):
        """`plant` maps to no room on purpose. If that ever changes, the skip below is
        testing nothing and this fails first."""
        self.assertNotIn("plant", ROOM_OF_ANCHOR)

    def test_it_is_skipped_rather_than_guessed_at(self):
        cells = (
            CellYield("s1", "snoring", "bed", n_built=15, n_anchored=4, n_skipped=0),
            CellYield("s1", "a_plant_sound", "plant", n_built=15, n_anchored=9, n_skipped=0),
        )
        self.assertEqual(rooms_by_scene(cells)["s1"], {ROOM_OF_ANCHOR["bed"]: 4})

    def test_it_is_reported_rather_than_silent(self):
        cells = (
            CellYield("s1", "snoring", "bed", n_built=15, n_anchored=4, n_skipped=0),
            CellYield("s1", "a_plant_sound", "plant", n_built=15, n_anchored=9, n_skipped=0),
        )
        self.assertEqual(anchors_without_a_room(cells), ("plant",))
        text = format_report(cells, scenes=["s1"], n_episodes=15, split="val")
        self.assertIn("no `ROOM_OF_ANCHOR` row", text)
        self.assertIn("plant", text)

    def test_a_class_with_no_anchor_at_all_is_not_an_orphan_room(self):
        """`glass_break` has no anchor object, so it never reaches the room table."""
        cells = (CellYield("s1", "glass_break", None, n_built=15, n_anchored=0, n_skipped=0),)
        self.assertEqual(anchors_without_a_room(cells), ())


class TestTheAssignmentTable(unittest.TestCase):
    def test_it_names_a_class_that_anchors_in_the_assigned_room(self):
        rooms = sorted({ROOM_OF_ANCHOR[a] for a in _SIX_SCENE_OBJECTS})
        by_room = balanced_assignment(rooms_by_scene(_TWO_ROOM_OBJECTS_CELLS), rooms)
        detail = room_assignment_detail(_TWO_ROOM_OBJECTS_CELLS, by_room)
        for scene, (room, anchor, name, anchored) in detail.items():
            self.assertEqual(ROOM_OF_ANCHOR[anchor], room, scene)
            self.assertGreater(anchored, 0, scene)

    def test_a_living_room_scene_takes_the_better_of_its_two_objects(self):
        living = ROOM_OF_ANCHOR["chair"]
        detail = room_assignment_detail(
            _TWO_ROOM_OBJECTS_CELLS, {"s1": (living, 10)}
        )
        self.assertEqual(detail["s1"][1:], ("chair", "keyboard_typing", 10))

    def test_a_room_with_no_cell_is_omitted_not_invented(self):
        detail = room_assignment_detail(
            _TWO_ROOM_OBJECTS_CELLS, {"s99": (ROOM_OF_ANCHOR["bed"], 4)}
        )
        self.assertEqual(detail, {})

    def test_the_report_prints_the_table_and_both_units(self):
        text = format_report(
            _TWO_ROOM_OBJECTS_CELLS,
            scenes=["s1", "s2", "s3", "s4", "s5", "s6"],
            n_episodes=15,
            split="val",
        )
        self.assertIn("THE UNIT IS THE ROOM", text)
        self.assertIn("ALWAYS-{}".format(ROOM_OF_ANCHOR["chair"].upper()), text)
        self.assertIn("BALANCING OVER ANCHOR OBJECTS INSTEAD", text)
        self.assertIn("scene by scene", text)
        self.assertIn("keyboard_typing", text)

    def test_the_object_balanced_design_is_printed_under_both_units(self):
        """The reconciliation with the box output that exposed this: the same assignment,
        the number it printed and the number that was true."""
        text = format_report(
            _TWO_ROOM_OBJECTS_CELLS,
            scenes=["s1", "s2", "s3", "s4", "s5", "s6"],
            n_episodes=15,
            split="val",
        )
        block = text[text.index("BALANCING OVER ANCHOR OBJECTS"):]
        by_object = block[block.index("scored by object"):block.index("scored by room")]
        by_room = block[block.index("scored by room"):block.index("The second")]
        self.assertIn("ALWAYS-CHAIR", by_object)
        self.assertIn("ALWAYS-{}".format(ROOM_OF_ANCHOR["chair"].upper()), by_room)
        # Same denominator, larger numerator: one assignment, two readings of it.
        self.assertIn("of 44", by_object)
        self.assertIn("of 44", by_room)


class TestWriteAssignmentTsv(unittest.TestCase):
    """The handoff `matrix_sweep.sh` reads: `scene<TAB>class`, nothing else, one line
    per scene, so a bash `while read -r scene class` loop needs no parsing."""

    def test_it_writes_one_line_per_scene_tab_separated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "assignment.tsv")
            write_assignment_tsv(path, _TWO_ROOM_OBJECTS_CELLS)
            with open(path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        self.assertEqual(len(lines), 6)
        for line in lines:
            scene, name = line.split("\t")
            self.assertTrue(scene)
            self.assertTrue(name)

    def test_the_returned_mapping_matches_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "assignment.tsv")
            result = write_assignment_tsv(path, _TWO_ROOM_OBJECTS_CELLS)
            with open(path, encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        from_file = dict(line.split("\t") for line in lines)
        self.assertEqual(sorted(from_file), sorted(result))
        for scene, name in from_file.items():
            self.assertEqual(result[scene][2], name)

    def test_it_uses_the_room_balanced_design_not_the_greedy_one(self):
        """The whole point of the tool: the living room does not take every scene."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "assignment.tsv")
            result = write_assignment_tsv(path, _TWO_ROOM_OBJECTS_CELLS)
        rooms_assigned = collections.Counter(room for room, _a, _c in result.values())
        self.assertLess(max(rooms_assigned.values()), len(_TWO_ROOM_OBJECTS_CELLS) // 4)

    def test_no_scenes_at_all_raises_rather_than_writing_an_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "assignment.tsv")
            with self.assertRaises(ValueError):
                write_assignment_tsv(path, ())
            self.assertFalse(os.path.exists(path))


class TestTheExitCode(unittest.TestCase):
    def test_a_bad_episode_count_is_two(self):
        self.assertEqual(main(["--n-episodes", "0"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
