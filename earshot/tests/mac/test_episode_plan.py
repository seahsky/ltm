"""Crossing scenes with sounds: the supply is lopsided, so the plan must not be.

HM3D val publishes 40 bathroom anchors, 68 bedroom and 296 living-room, and one scene holds 37
against another's 8. Uniform sampling over that supply builds a dataset that is 73% living room
and four times more about `cvZr5TUy5C5` than about `mL8ThkuaVTM`. Neither imbalance shows up in
a success rate, and both move it.

The real HM3D val counts are used below rather than round numbers, so the tests fail against the
distribution the plan will actually meet.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.task.episode_plan import (
    MEMORY_CONDITIONS,
    anchor_slots,
    balance_report,
    plan_episodes,
    supply,
)
from earshot.tools.build_plan import load_bank

# provenance: `room_yield --split val`, 2026-08-21. bathroom 40 / bedroom 68 / living 296.
VAL_ROOMS = {
    "4ok3usBNeis": {"bathroom": 2, "bedroom": 2, "living_room": 11},
    "5cdEh9F2hJL": {"bathroom": 2, "bedroom": 2, "living_room": 15},
    "6s7QHgap2fW": {"bathroom": 1, "bedroom": 2, "living_room": 20},
    "DYehNKdT76V": {"bathroom": 3, "bedroom": 3, "living_room": 10},
    "Dd4bFSTQ8gi": {"bathroom": 2, "bedroom": 3, "living_room": 14},
    "Nfvxx8J5NCo": {"bathroom": 2, "bedroom": 2, "living_room": 12},
    "QaLdnwvtxbs": {"bedroom": 1, "living_room": 14},
    "TEEsavR23oF": {"bathroom": 1, "bedroom": 2, "living_room": 15},
    "XB4GS9ShBRE": {"bathroom": 2, "bedroom": 3, "living_room": 12},
    "bxsVRursffK": {"bathroom": 3, "bedroom": 3, "living_room": 5},
    "cvZr5TUy5C5": {"bathroom": 5, "bedroom": 4, "living_room": 28},
    "mL8ThkuaVTM": {"bathroom": 2, "bedroom": 3, "living_room": 3},
    "mv2HUxq3B53": {"bathroom": 1, "bedroom": 5, "living_room": 28},
    "p53SfW6mjZe": {"bathroom": 3, "bedroom": 9, "living_room": 23},
    "q3zU7Yy5E5s": {"bathroom": 2, "bedroom": 8, "living_room": 10},
    "qyAac8rV8Zk": {"bathroom": 1, "bedroom": 3, "living_room": 13},
    "svBbv1Pavdk": {"bathroom": 1, "bedroom": 2, "living_room": 13},
    "wcojb4TFT35": {"bathroom": 3, "bedroom": 4, "living_room": 12},
    "ziup5kvtCCR": {"bathroom": 2, "bedroom": 3, "living_room": 11},
    "zt1RVoi7PcG": {"bathroom": 2, "bedroom": 4, "living_room": 27},
}

# The bank of record: 11 classes, 3 / 5 / 3.
BANK = {
    "bathroom": ("brushing_teeth", "pouring_water", "toilet_flush"),
    "bedroom": ("breathing", "clock_alarm", "clock_tick", "crying_baby", "snoring"),
    "living_room": ("clapping", "keyboard_typing", "laughing"),
}

DEV_CLIPS = tuple(range(20))  # the development block from dataset_split


class TestTheSupplyIsLopsided(unittest.TestCase):
    def test_the_real_anchor_counts(self):
        totals = supply(VAL_ROOMS, BANK, len(DEV_CLIPS))["per_room"]
        self.assertEqual(totals["bathroom"]["anchors"], 40)
        self.assertEqual(totals["bedroom"]["anchors"], 68)
        self.assertEqual(totals["living_room"]["anchors"], 296)

    def test_one_scene_has_no_bathroom_and_it_is_counted(self):
        """`QaLdnwvtxbs` cannot host a bathroom episode. 19 scenes, not 20."""
        totals = supply(VAL_ROOMS, BANK, len(DEV_CLIPS))["per_room"]
        self.assertEqual(totals["bathroom"]["scenes"], 19)
        self.assertEqual(totals["bedroom"]["scenes"], 20)

    def test_the_living_room_holds_most_of_the_raw_supply(self):
        """73%. This is the number that makes naive expansion a bad dataset."""
        share = supply(VAL_ROOMS, BANK, len(DEV_CLIPS))["largest_room_share"]
        self.assertAlmostEqual(share, 296 / 404, places=6)
        self.assertGreater(share, 0.70)

    def test_the_combination_count_is_large_enough_not_to_be_the_constraint(self):
        """Supply is never the limit; balance is. 26,960 against the 200 the plan needs."""
        self.assertEqual(supply(VAL_ROOMS, BANK, len(DEV_CLIPS))["combinations"], 26_960)

    def test_zero_recordings_raises(self):
        with self.assertRaises(ValueError):
            supply(VAL_ROOMS, BANK, 0)


class TestAnchorSlotsAreInstanceMajor(unittest.TestCase):
    def test_every_scene_offers_one_anchor_before_any_offers_two(self):
        """Scene-major order would make the plan a study of the biggest house."""
        slots = anchor_slots(VAL_ROOMS, "living_room")
        first_pass = slots[:20]
        self.assertEqual(len({scene for scene, _i in first_pass}), 20)
        self.assertTrue(all(instance == 0 for _s, instance in first_pass))

    def test_every_anchor_appears_exactly_once(self):
        slots = anchor_slots(VAL_ROOMS, "bathroom")
        self.assertEqual(len(slots), 40)
        self.assertEqual(len(set(slots)), 40)

    def test_a_room_no_scene_has_returns_empty(self):
        self.assertEqual(anchor_slots(VAL_ROOMS, "kitchen"), [])


class TestThePlanIsBalanced(unittest.TestCase):
    def _plan(self, n=200):
        return plan_episodes(VAL_ROOMS, BANK, DEV_CLIPS, n)

    def test_it_produces_exactly_the_requested_episodes(self):
        self.assertEqual(len(self._plan(200)), 200)

    def test_rooms_get_near_equal_shares_despite_296_against_40(self):
        """The whole point. Raw supply is 73% living room; the plan is 34%."""
        report = balance_report(self._plan(200))
        self.assertEqual(report["by_room"], {"bathroom": 67, "bedroom": 67, "living_room": 66})

    def test_classes_inside_a_room_get_near_equal_shares(self):
        by_class = balance_report(self._plan(200))["by_class"]
        bedroom = [by_class[name] for name in BANK["bedroom"]]
        self.assertLessEqual(max(bedroom) - min(bedroom), 1)

    def test_no_scene_dominates_among_those_that_can_host_every_room(self):
        """cvZr5TUy5C5 holds 4.6x mL8ThkuaVTM's anchors. The plan must not inherit that.

        Measured on the scenes that CAN host all three rooms, because `QaLdnwvtxbs` has no
        toilet and sits a third below the rest whatever the planner does. The all-scene ratio
        is dominated by that structural gap and does not move when balancing improves.
        """
        report = balance_report(self._plan(200))
        self.assertLessEqual(report["scene_ratio_complete"], 1.2)
        self.assertEqual(report["scenes_incomplete"], ["QaLdnwvtxbs"])

    def test_the_all_scene_ratio_is_worse_and_that_is_the_structural_gap(self):
        """Pinned so the two numbers are not confused again: the wider one is not the planner."""
        report = balance_report(self._plan(200))
        self.assertGreater(report["scene_ratio"], report["scene_ratio_complete"])

    def test_the_scene_spread_is_at_most_one_episode_where_it_can_be(self):
        report = balance_report(self._plan(200))
        hostable = [
            count for scene, count in report["by_scene"].items()
            if scene not in report["scenes_incomplete"]
        ]
        self.assertLessEqual(max(hostable) - min(hostable), 1)

    def test_the_plan_is_identical_across_calls(self):
        self.assertEqual([s.as_dict() for s in self._plan()], [s.as_dict() for s in self._plan()])

    def test_indices_are_dense_and_ordered(self):
        specs = self._plan(200)
        self.assertEqual([s.index for s in specs], list(range(200)))

    def test_recordings_are_drawn_from_the_block_given(self):
        """A plan reaching outside the development block would contaminate the test set."""
        for spec in self._plan(200):
            self.assertIn(spec.recording, DEV_CLIPS)

    def test_a_room_with_no_anchor_raises_rather_than_shrinking_the_plan(self):
        """A cell quietly missing a room is exactly what the balancing exists to prevent."""
        with self.assertRaises(ValueError):
            plan_episodes(VAL_ROOMS, dict(BANK, kitchen=("frying",)), DEV_CLIPS, 200)

    def test_an_empty_class_list_raises(self):
        with self.assertRaises(ValueError):
            plan_episodes(VAL_ROOMS, {"bathroom": ()}, DEV_CLIPS, 200)

    def test_no_recordings_raises(self):
        with self.assertRaises(ValueError):
            plan_episodes(VAL_ROOMS, BANK, (), 200)


class TestTheFourConditionsCrossTheSamePlan(unittest.TestCase):
    """The structural finding: the cells are conditions on the PRIOR PHASE, not on the episode.

    So one 200-episode plan crossed with four conditions gives 800 episode-runs, 200 a cell,
    and the cells are paired on identical episodes rather than four independent samples.
    """

    def test_there_are_four_named_conditions(self):
        self.assertEqual(len(MEMORY_CONDITIONS), 4)
        self.assertEqual(len(set(MEMORY_CONDITIONS)), 4)

    def test_the_crossing_lands_on_the_pre_registered_budget(self):
        specs = plan_episodes(VAL_ROOMS, BANK, DEV_CLIPS, 200)
        self.assertEqual(len(specs) * len(MEMORY_CONDITIONS), 800)

    def test_every_condition_sees_the_identical_episode_set(self):
        """Pairing only works if nothing about the episode changes between conditions."""
        specs = plan_episodes(VAL_ROOMS, BANK, DEV_CLIPS, 200)
        per_condition = {name: [s.as_dict() for s in specs] for name in MEMORY_CONDITIONS}
        reference = per_condition[MEMORY_CONDITIONS[0]]
        for name in MEMORY_CONDITIONS[1:]:
            self.assertEqual(per_condition[name], reference, name)


class TestLoadBankRefusesTheCandidateSet(unittest.TestCase):
    """The plan must come from the BANK OF RECORD, never from the candidate vocabulary.

    The candidate set carries weak-affinity and unresolvable classes on purpose so the gate can
    prune rather than the author. Building a plan from it would put `mouse_click` (0.308 anchor
    recall) and `coughing` (a sound people make in every room) into a dataset that is supposed
    to measure room memory. So a malformed bank raises instead of falling back.
    """

    def _write(self, payload):
        import json, os, tempfile
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "bank.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_a_real_bank_loads_as_room_to_classes(self):
        path = self._write({"by_anchor": {room: list(names) for room, names in BANK.items()}})
        self.assertEqual(load_bank(path), {room: sorted(names) for room, names in BANK.items()})

    def test_a_file_without_by_anchor_raises(self):
        with self.assertRaises(ValueError):
            load_bank(self._write({"kept": ["toilet_flush"]}))

    def test_a_bank_listing_no_class_raises(self):
        with self.assertRaises(ValueError):
            load_bank(self._write({"by_anchor": {"bathroom": [], "bedroom": []}}))

    def test_a_room_with_no_classes_is_dropped_not_planned_empty(self):
        """An empty room would otherwise raise deep inside the planner with a worse message."""
        path = self._write({"by_anchor": {"bathroom": ["toilet_flush"], "bedroom": []}})
        self.assertEqual(load_bank(path), {"bathroom": ["toilet_flush"]})


class TestBalanceReport(unittest.TestCase):
    def test_an_empty_plan_raises(self):
        with self.assertRaises(ValueError):
            balance_report([])

    def test_it_counts_distinct_recordings(self):
        report = balance_report(plan_episodes(VAL_ROOMS, BANK, DEV_CLIPS, 200))
        self.assertEqual(report["n_distinct_recordings"], 20)


if __name__ == "__main__":
    unittest.main()
