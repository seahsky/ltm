"""The anchor-room ceiling: the fold that sizes a matrix cell.

ADR-0018's power question is being argued at roughly 90 episodes a cell against a measured
16.2% flip rate, and nobody has counted the denominator. A scene that publishes no `toilet`
goal cannot host a bathroom episode, so the cell ceiling is a scene count and not a division.

Tested here rather than only on the box because the fold is arithmetic: `task/episodes.py` is
stdlib gzip and json, so the only simulator-dependent part of `room_yield` is having the files.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.vocabulary import ROOMS
from earshot.tools.room_yield import summarise_yield


class TestSummariseYield(unittest.TestCase):
    def test_a_room_absent_from_a_scene_lowers_that_room_only(self):
        summary = summarise_yield(
            {
                "A": {"bathroom": 1, "bedroom": 2, "living_room": 3},
                "B": {"bedroom": 1, "living_room": 1},
            }
        )
        self.assertEqual(
            summary["scenes_per_room"],
            {"bathroom": 1, "bedroom": 2, "living_room": 2},
        )

    def test_only_scenes_offering_every_room_are_complete(self):
        summary = summarise_yield(
            {
                "A": {"bathroom": 1, "bedroom": 1, "living_room": 1},
                "B": {"bathroom": 1},
            }
        )
        self.assertEqual(summary["scenes_with_all_rooms"], ["A"])

    def test_every_room_appears_even_when_no_scene_hosts_it(self):
        """A room at zero must READ as zero. A missing key would look like an oversight."""
        summary = summarise_yield({"A": {"bedroom": 1}})
        self.assertEqual(set(summary["scenes_per_room"]), set(ROOMS))
        self.assertEqual(summary["scenes_per_room"]["bathroom"], 0)

    def test_instance_counts_do_not_inflate_the_scene_count(self):
        """Three sofas are not three living rooms."""
        summary = summarise_yield({"A": {"living_room": 7}})
        self.assertEqual(summary["scenes_per_room"]["living_room"], 1)

    def test_the_histogram_counts_scenes_by_how_many_rooms_they_offer(self):
        summary = summarise_yield(
            {
                "A": {"bathroom": 1, "bedroom": 1, "living_room": 1},
                "B": {"bedroom": 1, "living_room": 1},
                "C": {"bedroom": 1, "living_room": 1},
            }
        )
        self.assertEqual(summary["rooms_per_scene_histogram"], {"2": 2, "3": 1})
        self.assertEqual(summary["n_scenes"], 3)

    def test_an_empty_split_is_not_a_crash_but_is_visibly_empty(self):
        summary = summarise_yield({})
        self.assertEqual(summary["n_scenes"], 0)
        self.assertEqual(summary["scenes_with_all_rooms"], [])
        self.assertEqual(set(summary["scenes_per_room"].values()), {0})


if __name__ == "__main__":
    unittest.main()
