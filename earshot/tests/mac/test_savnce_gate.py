"""The SAVN-CE gate is judged on both arms, on a Mac, with no simulator.

ADR-0014's rule is that a detector ships the healthy path passing *and* the forced
failure firing. The box runs the forced-failure arms for real (`savnce_eval.sh
--forced-failure`), and those runs cost hours each. This is the same two arms per
criterion at zero box cost, which is the only reason the gate can be trusted before its
first real run rather than after it.

The specific thing being pinned is the repo's `NOT_RUN` rule. A gate that reads an
absent measurement as "fine" is exactly the failure this project has already paid for
twice: a probe that skipped and reported success, and a canary that was never armed
reading as a pass. So every criterion is tested with its input *missing*, not only with
its input wrong.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401  (raises on the wrong Python)

from earshot.tools.savnce_gate import (
    ACCEPTANCE_BAND_SR_POINTS,
    EXCLUDED_STAT_KEYS,
    GREEN,
    NOT_RUN,
    RED,
    acceptance,
    aggregate,
    judge,
)


def healthy_probe():
    """What the eval driver writes after a run that worked."""
    return {
        "habitat_sim_audio_capable": True,
        "multi_audio_sensor_patch": True,
        "missing_paths": [],
        "episodes_available": 1000,
        "ckpt_loaded": True,
        "ckpt_missing_submodules": [],
        "max_abs_audio": 0.163,
        "episodes_per_hour": 41.0,
        "fps": 11.4,
        "split": "test",
        "requested_episodes": 1000,
        "stats_file": "model/tb/test_stats_0.json",
    }


def stats(count, success=1.0, spl=0.9):
    """`count` episodes in their `<split>_stats_<seed>.json` shape."""
    return {
        "scene{},{}".format(index, index): {
            "success": success,
            "spl": spl,
            "distance_to_goal": 2.0,
            "reward": 1.0,
            "audio_duration": 5.0,
        }
        for index in range(count)
    }


def verdicts(criteria):
    return {criterion.name: criterion.verdict for criterion in criteria}


class TestTheHealthyArm(unittest.TestCase):
    def test_a_good_run_is_seven_green(self):
        criteria = judge(healthy_probe(), stats(1000), 1000)
        self.assertEqual(len(criteria), 7)
        self.assertTrue(
            all(criterion.is_green for criterion in criteria),
            "healthy run was not all green: {}".format(verdicts(criteria)),
        )

    def test_every_criterion_prints_a_measurement(self):
        for criterion in judge(healthy_probe(), stats(1000), 1000):
            self.assertTrue(
                criterion.measurement.strip(),
                "criterion {} is green with nothing to show for it".format(criterion.name),
            )


class TestTheForcedFailureArms(unittest.TestCase):
    """One per criterion. Each is a real way this run can go wrong."""

    def test_wrong_checkpoint_fires_criterion_three(self):
        probe = healthy_probe()
        probe["ckpt_missing_submodules"] = ["goal_descriptor", "scene_memory"]
        self.assertEqual(verdicts(judge(probe, stats(1000), 1000))["ckpt"], RED)

    def test_a_short_run_is_red_not_a_warning(self):
        self.assertEqual(verdicts(judge(healthy_probe(), stats(998), 1000))["episodes"], RED)

    def test_no_stats_file_at_all_is_distinct_from_an_empty_one(self):
        self.assertEqual(verdicts(judge(healthy_probe(), None, 1000))["episodes"], NOT_RUN)
        self.assertEqual(verdicts(judge(healthy_probe(), {}, 1000))["episodes"], RED)

    def test_silent_audio_is_a_hard_failure(self):
        probe = healthy_probe()
        probe["max_abs_audio"] = 0.0
        self.assertEqual(verdicts(judge(probe, stats(1000), 1000))["audio"], RED)

    def test_an_unpatched_habitat_sim_fires_criterion_one(self):
        probe = healthy_probe()
        probe["multi_audio_sensor_patch"] = False
        self.assertEqual(verdicts(judge(probe, stats(1000), 1000))["env"], RED)

    def test_a_missing_data_path_fires_criterion_two(self):
        probe = healthy_probe()
        probe["missing_paths"] = ["data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"]
        self.assertEqual(verdicts(judge(probe, stats(1000), 1000))["data"], RED)

    def test_zero_episodes_available_fires_criterion_two(self):
        probe = healthy_probe()
        probe["episodes_available"] = 0
        self.assertEqual(verdicts(judge(probe, stats(1000), 1000))["data"], RED)


class TestNotRunIsNeverGreen(unittest.TestCase):
    """The rule, applied to every criterion that reads a probe key."""

    CASES = {
        "habitat_sim_audio_capable": "env",
        "multi_audio_sensor_patch": "env",
        "missing_paths": "data",
        "episodes_available": "data",
        "ckpt_loaded": "ckpt",
        "ckpt_missing_submodules": "ckpt",
        "max_abs_audio": "audio",
        "episodes_per_hour": "throughput",
    }

    def test_a_missing_measurement_reads_not_run(self):
        for key, name in self.CASES.items():
            probe = healthy_probe()
            del probe[key]
            with self.subTest(missing=key):
                self.assertEqual(
                    verdicts(judge(probe, stats(1000), 1000))[name],
                    NOT_RUN,
                    "dropping {} left criterion {} evaluable".format(key, name),
                )

    def test_not_run_never_counts_as_green(self):
        probe = healthy_probe()
        del probe["episodes_per_hour"]
        criteria = judge(probe, stats(1000), 1000)
        self.assertFalse(any(c.is_green for c in criteria if c.verdict == NOT_RUN))
        self.assertFalse(all(c.is_green for c in criteria))

    def test_a_nan_measurement_is_not_a_measurement(self):
        probe = healthy_probe()
        probe["episodes_per_hour"] = float("nan")
        self.assertEqual(verdicts(judge(probe, stats(1000), 1000))["throughput"], NOT_RUN)

    def test_a_run_without_the_na_metric_still_reports_throughput(self):
        """fps needs their `na` key; episodes-per-hour never does. Missing fps is not red."""
        probe = healthy_probe()
        del probe["fps"]
        criterion = {c.name: c for c in judge(probe, stats(1000), 1000)}["throughput"]
        self.assertEqual(criterion.verdict, GREEN)
        self.assertIn("unavailable", criterion.measurement)

    def test_throughput_projects_the_cost_of_the_full_arm(self):
        criterion = {c.name: c for c in judge(healthy_probe(), stats(1000), 1000)}["throughput"]
        self.assertIn("h for 1000 episodes", criterion.measurement)


class TestAggregationMirrorsTheirs(unittest.TestCase):
    """Their table, recomputed. A different key set is a different table."""

    def test_their_four_skipped_keys_are_skipped(self):
        table = aggregate(stats(4))
        for key in EXCLUDED_STAT_KEYS:
            self.assertNotIn(key, table)
        self.assertIn("audio_duration", stats(4)["scene0,0"], "the fixture must contain it to prove the skip")

    def test_the_denominator_is_the_episode_count(self):
        mixed = stats(4)
        for index, key in enumerate(sorted(mixed)):
            mixed[key]["success"] = 1.0 if index < 1 else 0.0
        self.assertAlmostEqual(aggregate(mixed)["success"], 0.25)

    def test_an_empty_run_aggregates_to_nothing_rather_than_crashing(self):
        self.assertEqual(aggregate({}), {})


class TestThePreRegisteredBand(unittest.TestCase):
    """ADR-0015 fixed this before the first run, which is the whole point of it."""

    def test_the_paper_is_percent_and_the_stats_are_fractions(self):
        inside, line = acceptance(stats(1000, success=0.377), "test", 1000)
        self.assertTrue(inside, line)
        self.assertIn("37.7", line)

    def test_a_miss_outside_the_band_is_a_miss(self):
        inside, line = acceptance(stats(1000, success=0.300), "test", 1000)
        self.assertFalse(inside)
        self.assertIn("MISS", line)

    def test_the_edge_of_the_band_is_inside_it(self):
        edge = (37.7 - ACCEPTANCE_BAND_SR_POINTS) / 100.0
        inside, line = acceptance(stats(1000, success=edge), "test", 1000)
        self.assertTrue(inside, line)

    def test_a_smoke_is_never_judged_against_the_band(self):
        inside, line = acceptance(stats(20, success=1.0), "test", 20)
        self.assertFalse(inside)
        self.assertIn("not applicable", line)

    def test_the_val_split_is_never_judged_against_the_band(self):
        inside, line = acceptance(stats(1000, success=0.377), "val", 1000)
        self.assertFalse(inside)
        self.assertIn("not applicable", line)


if __name__ == "__main__":
    unittest.main()
