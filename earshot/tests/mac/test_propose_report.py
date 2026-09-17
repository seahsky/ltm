"""`propose_report` — and the bug it exists because of.

`propose-1` ran 1128 episodes over about eleven hours and its readout said the mechanism
never ran: every counter 0, on both propose arms. The counters were on disk the whole
time. The check was an inline Python heredoc inside `propose_sweep.sh` and it globbed
`<scene>/episodes/<N>/audit.json`, while `report/artifacts.episode_paths` writes
`<scene>/episodes/ep0000.audit.json` -- a flat file, no directory per episode. The glob
matched nothing.

`dream-1` is the same failure (a central quantity no reader could print) and `pilot-1` is
the other one (a reader inside a bash string that reported three dead arms over 120
episodes on disk). The rule those two bought is that a reader is a tested module, and it
was broken by writing a new reader into a shell script where no test could see it.

So the first test here writes a REAL audit through `write_episode` and reads it back. A
fixture that hand-builds the path it expects would have passed against the broken glob
too, which is the whole reason that bug survived to a night of box time.
"""

import pathlib
import shutil
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.artifacts import write_episode
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.report.agent import AgentReport
from earshot.tools.propose_report import (
    COUNTERS,
    RAIL_KEY,
    arm_totals,
    format_report,
    main,
)
from earshot.types import Xyz

LIVE = {
    "memory_propose_eligible": 40.0,
    "memory_propose_ranked_first": 6.0,
    "memory_propose_emitted": 40.0,
    "memory_propose_railed": 2.0,
    "memory_propose_unrouted": 0.0,
    "memory_propose_no_acoustic": 1.0,
    RAIL_KEY: 6.0,
}


def write_arm(root, arm, scenes, metrics):
    """One arm directory, through the real writer. `scenes` is {name: n_episodes}."""
    for scene, n in scenes.items():
        scene_dir = pathlib.Path(root) / arm / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        for index in range(n):
            write_episode(str(scene_dir), index, AgentReport(resumed=True), EpisodeAudit(
                episode_index=index,
                scene_id=scene,
                source_xyz=Xyz(1.0, 0.0, 2.0),
                funnel_stage=FunnelStage.SOURCE_REACHED,
                metrics=dict(metrics),
            ))
    return pathlib.Path(root) / arm


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)


class TestItReadsWhatTheRunnerWrote(Fixture):
    """THE ARM THIS FILE EXISTS FOR. Through `write_episode`, never a hand-built path."""

    def test_the_counters_come_back_summed_over_every_episode(self):
        write_arm(self.root, "propose-a", {"sceneA": 3, "sceneB": 2}, LIVE)
        totals = arm_totals(str(pathlib.Path(self.root) / "propose-a"))
        self.assertEqual(totals["episodes"], 5.0)
        self.assertEqual(totals["scenes"], 2.0)
        self.assertEqual(totals["memory_propose_eligible"], 200.0)
        self.assertEqual(totals["memory_propose_ranked_first"], 30.0)

    def test_it_finds_the_flat_audit_filename_the_writer_uses(self):
        """The broken heredoc needed a directory per episode. Asserting the count is
        nonzero is what a path-shape regression would trip."""
        write_arm(self.root, "propose-a", {"sceneA": 1}, LIVE)
        totals = arm_totals(str(pathlib.Path(self.root) / "propose-a"))
        self.assertEqual(totals["episodes_with_counters"], 1.0)


class TestUnreadableIsNotInert(Fixture):
    """The distinction that cost the night. A counter of 0 and a read of 0 are different
    facts and the report must never print them the same way."""

    def test_an_arm_with_no_episodes_says_so(self):
        (pathlib.Path(self.root) / "propose-a" / "sceneA").mkdir(parents=True)
        text = format_report(
            {"propose-a": arm_totals(str(pathlib.Path(self.root) / "propose-a"))})
        self.assertIn("NO EPISODE ON DISK", text)

    def test_an_arm_that_never_recorded_the_counters_says_so(self):
        """An arm run without --memory-proposes, or one predating PR #133."""
        write_arm(self.root, "propose-a", {"sceneA": 2}, {"dream_tau_steps": 1.0})
        text = format_report(
            {"propose-a": arm_totals(str(pathlib.Path(self.root) / "propose-a"))})
        self.assertIn("COUNTERS NOT RECORDED", text)
        self.assertIn("Unreadable is not inert", text)

    def test_that_case_exits_nonzero(self):
        write_arm(self.root, "propose-a", {"sceneA": 2}, {"dream_tau_steps": 1.0})
        self.assertEqual(main([self.root]), 2)

    def test_a_sweep_with_no_proposing_arm_exits_nonzero(self):
        write_arm(self.root, "replace-a", {"sceneA": 2}, {})
        self.assertEqual(main([self.root]), 2)

    def test_a_live_arm_exits_zero(self):
        write_arm(self.root, "propose-a", {"sceneA": 2}, LIVE)
        self.assertEqual(main([self.root]), 0)


class TestTheFourthBranch(unittest.TestCase):
    """ADR-0026's own stopping rule, and the three ways it can fire. Pure."""

    def verdict(self, **counters):
        base = {key: 0.0 for key in COUNTERS}
        base.update(counters)
        base.update({"episodes": 10.0, "episodes_with_counters": 10.0, "scenes": 1.0})
        return format_report({"propose-a": base})

    def test_never_emitted_is_named_as_a_contrast_against_no_memory(self):
        """The most misleading case: the arm looks live and is measuring nothing but the
        acoustic estimate."""
        text = self.verdict(memory_propose_railed=80.0)
        self.assertIn("THE PROPOSAL NEVER ENTERED A POOL", text)
        self.assertIn("contrast against NO MEMORY", text)

    def test_emitted_but_filtered_by_the_navmesh_is_its_own_verdict(self):
        text = self.verdict(memory_propose_emitted=50.0, memory_propose_eligible=0.0)
        self.assertIn("EMITTED BUT NEVER RANKED", text)

    def test_under_five_percent_fires_the_fourth_branch(self):
        text = self.verdict(memory_propose_emitted=100.0,
                            memory_propose_eligible=100.0,
                            memory_propose_ranked_first=4.0)
        self.assertIn("FOURTH BRANCH", text)
        self.assertIn("NOT a result about memory", text)

    def test_at_or_above_five_percent_is_live(self):
        text = self.verdict(memory_propose_emitted=100.0,
                            memory_propose_eligible=100.0,
                            memory_propose_ranked_first=5.0)
        self.assertIn("THE MECHANISM IS LIVE", text)
        self.assertNotIn("FOURTH BRANCH", text)

    def test_two_rails_in_one_arm_is_two_experiments(self):
        base = {key: 0.0 for key in COUNTERS}
        base.update({"episodes": 4.0, "episodes_with_counters": 4.0, "scenes": 1.0,
                     "memory_propose_emitted": 10.0, "memory_propose_eligible": 10.0,
                     "memory_propose_ranked_first": 5.0, "rail_disagreement": 2.0})
        self.assertIn("RAIL DISAGREEMENT", format_report({"propose-a": base}))


class TestTheRailIsReportedNotAveraged(Fixture):
    def test_a_removed_rail_prints_as_removed(self):
        metrics = dict(LIVE)
        metrics[RAIL_KEY] = -1.0
        write_arm(self.root, "propose-a", {"sceneA": 2}, metrics)
        text = format_report(
            {"propose-a": arm_totals(str(pathlib.Path(self.root) / "propose-a"))})
        self.assertIn("rail: REMOVED", text)

    def test_two_different_rails_are_flagged_rather_than_merged(self):
        write_arm(self.root, "propose-a", {"sceneA": 1}, LIVE)
        other = dict(LIVE)
        other[RAIL_KEY] = 12.0
        write_arm(self.root, "propose-a", {"sceneB": 1}, other)
        totals = arm_totals(str(pathlib.Path(self.root) / "propose-a"))
        self.assertEqual(totals.get("rail_disagreement"), 2.0)
        self.assertNotIn(RAIL_KEY, totals)


if __name__ == "__main__":
    unittest.main(verbosity=2)
