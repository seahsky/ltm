"""The per-episode pairing, over run directories written by the REAL writers.

An aggregate that is correct over injected dictionaries and never finds the files on disk
is a report that prints "no episode paired" and reads as a clean refusal, so the loading
seam is exercised here rather than mocked — the same reason `test_detour_report.py` builds
real directories.
"""

import pathlib
import shutil
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_episode
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.tools.episode_diff import (
    format_report,
    load_outcomes,
    main,
    mcnemar,
    pair_episodes,
)
from earshot.tools.funnel_diff import two_sided_exact_binomial
from earshot.types import Xyz

REACHED = FunnelStage.PRIMARY_RESUMED
ABANDONED = FunnelStage.INVESTIGATE_ENTERED


def write_scene(root, scene, outcomes, *, source=Xyz(1.0, 0.0, 2.0), sources=None):
    """One scene directory. `outcomes` is a list of FunnelStage, one per episode."""
    scene_dir = pathlib.Path(root) / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    for index, stage in enumerate(outcomes):
        write_episode(str(scene_dir), index, AgentReport(resumed=True), EpisodeAudit(
            episode_index=index,
            scene_id=scene,
            source_xyz=(sources[index] if sources else source),
            funnel_stage=stage,
        ))
    return scene_dir


class TestTheEpisodesArePairedAndVerified(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def tag(self, name):
        path = pathlib.Path(self.root) / name
        path.mkdir()
        return path

    def test_a_pair_needs_the_same_index_and_the_same_source(self):
        """The index alone is not identity. A builder change that reshuffles which task
        lands at index 2 would make the comparison meaningless without saying so."""
        before, after = self.tag("before"), self.tag("after")
        moved = [Xyz(1.0, 0.0, 2.0), Xyz(1.0, 0.0, 2.0), Xyz(9.0, 0.0, 9.0)]
        same = [Xyz(1.0, 0.0, 2.0), Xyz(1.0, 0.0, 2.0), Xyz(1.0, 0.0, 2.0)]
        write_scene(before, "sceneA", [REACHED, ABANDONED, ABANDONED], sources=same)
        write_scene(after, "sceneA", [REACHED, REACHED, REACHED], sources=moved)

        pairing = pair_episodes(load_outcomes(str(before)), load_outcomes(str(after)))

        self.assertEqual(pairing["n_pairs"], 2, "the two agreeing episodes pair")
        self.assertEqual(pairing["mismatched_source"], ["sceneA#2"])
        self.assertEqual(pairing["n_dropped"], 1)
        print("paired {} / dropped {} on a moved source".format(
            pairing["n_pairs"], pairing["n_dropped"]))

    def test_the_dropped_episode_is_named_in_the_report(self):
        """A silent drop turns a partial comparison into a confident one."""
        before, after = self.tag("before"), self.tag("after")
        write_scene(before, "sceneA", [REACHED, ABANDONED])
        write_scene(after, "sceneA", [REACHED, ABANDONED, REACHED])

        pairing = pair_episodes(load_outcomes(str(before)), load_outcomes(str(after)))
        text = format_report(pairing, mcnemar(pairing["pairs"]))

        self.assertEqual(pairing["unmatched_index"], ["sceneA#2"])
        self.assertIn("sceneA#2", text)
        self.assertIn("DROPPED", text)

    def test_an_episode_with_no_recorded_source_never_pairs(self):
        """A missing source cannot be verified against a present one, and pairing them
        anyway is the failure the verification exists to prevent."""
        before, after = self.tag("before"), self.tag("after")
        scene = pathlib.Path(before) / "sceneA"
        scene.mkdir(parents=True)
        write_episode(str(scene), 0, AgentReport(resumed=True), EpisodeAudit(
            episode_index=0, scene_id="sceneA", source_xyz=None, funnel_stage=REACHED))
        write_scene(after, "sceneA", [REACHED])

        pairing = pair_episodes(load_outcomes(str(before)), load_outcomes(str(after)))

        self.assertEqual(pairing["n_pairs"], 0)
        self.assertEqual(pairing["mismatched_source"], ["sceneA#0"])

    def test_a_scene_only_one_sweep_built_is_reported_not_dropped(self):
        before, after = self.tag("before"), self.tag("after")
        write_scene(before, "sceneA", [REACHED])
        write_scene(before, "gone", [REACHED])
        write_scene(after, "sceneA", [REACHED])

        pairing = pair_episodes(load_outcomes(str(before)), load_outcomes(str(after)))
        text = format_report(pairing, mcnemar(pairing["pairs"]))

        self.assertEqual(pairing["scenes_before_only"], ["gone"])
        self.assertIn("gone", text)

    def test_a_scene_that_built_nothing_survives_as_an_empty_mapping(self):
        """`mL8ThkuaVTM` yields zero in every run this repo has. A loader that dropped it
        would report a twenty-scene sweep as a nineteen-scene one."""
        before = self.tag("before")
        write_scene(before, "sceneA", [REACHED])
        (pathlib.Path(before) / "mL8ThkuaVTM").mkdir()

        outcomes = load_outcomes(str(before))

        self.assertIn("mL8ThkuaVTM", outcomes)
        self.assertEqual(outcomes["mL8ThkuaVTM"], {})

    def test_a_non_directory_at_the_tag_level_is_ignored(self):
        """`yield_sweep.sh` writes `provenance.txt` and `.hermeticity-before.json` beside
        the scene directories. A loader that took either for a scene would report a
        phantom one with no episodes, and a phantom scene is a silent hole in a
        denominator."""
        before = self.tag("before")
        write_scene(before, "sceneA", [REACHED])
        (pathlib.Path(before) / "provenance.txt").write_text("commit: abc123\n")
        (pathlib.Path(before) / ".hermeticity-before.json").write_text("{}")

        outcomes = load_outcomes(str(before))

        self.assertEqual(sorted(outcomes), ["sceneA"])

    def test_a_scene_directory_passed_by_mistake_raises(self):
        """Zero paired episodes is a finding; a mistyped path must not manufacture one."""
        before = self.tag("before")
        scene = write_scene(before, "sceneA", [REACHED])
        with self.assertRaises(ValueError):
            load_outcomes(str(scene))


class TestTheMcNemarArithmetic(unittest.TestCase):
    def test_concordant_pairs_are_excluded_and_discordant_ones_decide(self):
        """The whole point: 300 episodes both arms agreed on carry no information about a
        difference, and including them is where the scene-level reading loses its power."""
        pairs = (
            [{"scene": "a", "episode": i, "before": True, "after": True} for i in range(50)]
            + [{"scene": "a", "episode": 50 + i, "before": False, "after": False}
               for i in range(50)]
            + [{"scene": "a", "episode": 100 + i, "before": False, "after": True}
               for i in range(9)]
            + [{"scene": "b", "episode": 200, "before": True, "after": False}]
        )
        result = mcnemar(pairs)

        self.assertEqual((result["n_both"], result["n_neither"]), (50, 50))
        self.assertEqual((result["n_gained"], result["n_lost"]), (9, 1))
        self.assertEqual(result["n_discordant"], 10)
        self.assertEqual(result["net"], 8)
        self.assertAlmostEqual(result["p_value"], 2 * (10 + 1) / 1024)
        print("9 gained / 1 lost of 110 pairs -> p = {:.4f}".format(result["p_value"]))

    def test_two_arms_that_never_disagree_have_no_p_at_all(self):
        """A criterion that could not be evaluated is never green: zero discordant pairs
        is not p = 1.0, it is no test."""
        pairs = [{"scene": "a", "episode": i, "before": True, "after": True}
                 for i in range(20)]
        result = mcnemar(pairs)

        self.assertEqual(result["n_discordant"], 0)
        self.assertIsNone(result["p_value"])
        self.assertIn("NO DISCORDANT PAIRS", format_report(
            {"n_pairs": 20, "scenes_paired": ["a"], "scenes_before_only": [],
             "scenes_after_only": [], "unmatched_index": [], "mismatched_source": [],
             "n_dropped": 0}, result))

    def test_the_discordance_is_split_by_scene_so_clustering_is_visible(self):
        """Episodes in one room are not independent draws. If the whole imbalance comes
        from two scenes, the p is anti-conservative and the reader has to be able to see
        that from the report rather than take it on trust."""
        pairs = ([{"scene": "a", "episode": i, "before": False, "after": True}
                  for i in range(6)]
                 + [{"scene": "b", "episode": i, "before": True, "after": False}
                    for i in range(2)])
        result = mcnemar(pairs)

        self.assertEqual(result["per_scene"]["a"], {"gained": 6, "lost": 0})
        self.assertEqual(result["per_scene"]["b"], {"gained": 0, "lost": 2})
        self.assertEqual(result["scenes_with_discordance"], 2)

    def test_the_exact_tail_is_symmetric_and_shared_with_the_sign_test(self):
        """The sign test over scenes and the McNemar over episodes are the same
        arithmetic, so they use one function and it must not care which way the imbalance
        points."""
        self.assertEqual(two_sided_exact_binomial(8, 9), two_sided_exact_binomial(1, 9))
        self.assertAlmostEqual(two_sided_exact_binomial(9, 9), 2 / 512)
        self.assertAlmostEqual(two_sided_exact_binomial(5, 10), 1.0)
        self.assertIsNone(two_sided_exact_binomial(0, 0))


class TestTheCommandLine(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def tag(self, name):
        path = pathlib.Path(self.root) / name
        path.mkdir()
        return path

    def test_it_reads_two_tag_directories_and_reports(self):
        before, after = self.tag("cast-1"), self.tag("arrive-2")
        write_scene(before, "sceneA", [REACHED, ABANDONED, ABANDONED])
        write_scene(after, "sceneA", [REACHED, REACHED, ABANDONED])

        self.assertEqual(main([str(before), str(after)]), 0)

    def test_it_refuses_two_sweeps_with_no_episode_in_common(self):
        """A comparison that could not be made must not exit 0 — that is the same rule
        `arrival_audit.sh` follows when a tag holds no records."""
        before, after = self.tag("cast-1"), self.tag("arrive-2")
        write_scene(before, "sceneA", [REACHED])
        write_scene(after, "sceneB", [REACHED])

        self.assertEqual(main([str(before), str(after)]), 2)

    def test_an_unknown_stage_is_refused_rather_than_defaulted(self):
        before, after = self.tag("cast-1"), self.tag("arrive-2")
        write_scene(before, "sceneA", [REACHED])
        write_scene(after, "sceneA", [REACHED])

        self.assertEqual(main([str(before), str(after), "--stage", "NOT_A_STAGE"]), 2)


if __name__ == "__main__":
    unittest.main()
