"""`placement_report`, over run directories written by the REAL writer.

The question this reader answers is whether ADR-0022's placement change reached a finished
sweep. It has exactly one way to be dangerously wrong, and it is the way `pilot-1` was wrong:
find nothing on disk and say so calmly. So the loading seam is exercised here and not mocked,
and the healthy path and the forced failure both ship (ADR-0014).

The forced-failure arm is the record with NO `source_at_class_anchor` at all. Every audit
written before 2026-09-02 is that record, and a reader that scored it as `False` would report
a pre-ADR-0022 sweep as a fully geometric one — a confident, wrong answer to the only question
worth asking of `abl-2`.
"""

import pathlib
import shutil
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_episode
from earshot.report.audit import EpisodeAudit, FunnelStage
from earshot.tools.placement_report import (
    ABLATION_ARMS,
    ANCHOR_METRIC,
    UNRECORDED_SCENE,
    branch_of,
    format_arm,
    main,
    read_arm,
    read_sweep,
)

REACHED = FunnelStage.PRIMARY_RESUMED
ABANDONED = FunnelStage.INVESTIGATE_ENTERED

DRIVER = pathlib.Path(__file__).resolve().parents[2] / "tools" / "ablation_sweep.sh"


def write_scene(arm_dir, scene, episodes):
    """One scene directory. `episodes` is a list of (stage, metrics) pairs."""
    scene_dir = pathlib.Path(arm_dir) / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    for index, (stage, metrics) in enumerate(episodes):
        audit = EpisodeAudit(
            episode_index=index,
            scene_id=scene,
            funnel_stage=stage,
            metrics=metrics,
        )
        write_episode(str(scene_dir), index, AgentReport(resumed=True), audit)
    return scene_dir


def anchored(stage=REACHED):
    return (stage, {ANCHOR_METRIC: 1.0})


def geometric(stage=REACHED):
    return (stage, {ANCHOR_METRIC: 0.0})


def pre_adr(stage=REACHED):
    """The record every episode this repo ran before 2026-09-02 wrote: no such key."""
    return (stage, {})


class TestTheReaderFindsWhatTheWriterWrote(unittest.TestCase):
    """`pilot-1`'s defect, pinned. The writer names files `ep0000.audit.json`."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_written_arm_is_read_back_and_not_reported_empty(self):
        write_scene(self.tmp / "full", "sceneA", [anchored(), anchored(ABANDONED)])
        placement = read_arm(str(self.tmp / "full"), arm="full")
        self.assertEqual(placement.n, 2)
        self.assertEqual(placement.n_anchored, 2)
        self.assertEqual(placement.n_reached_anchored, 1)

    def test_an_arm_directory_that_does_not_exist_reads_as_empty_not_as_a_crash(self):
        placement = read_arm(str(self.tmp / "nope"), arm="nope")
        self.assertEqual(placement.n, 0)
        self.assertFalse(placement.evaluable)


class TestMissingIsNotFalse(unittest.TestCase):
    """The forced-failure arm. A pre-ADR-0022 record must never count as geometric."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_branch_of_is_tri_state(self):
        def audit(metrics):
            return EpisodeAudit(metrics=metrics)

        self.assertIs(branch_of(audit({ANCHOR_METRIC: 1.0})), True)
        self.assertIs(branch_of(audit({ANCHOR_METRIC: 0.0})), False)
        self.assertIsNone(branch_of(audit({})))

    def test_an_absent_field_lands_in_its_own_column(self):
        write_scene(self.tmp / "full", "sceneA", [pre_adr(), pre_adr(), geometric()])
        placement = read_arm(str(self.tmp / "full"), arm="full")
        self.assertEqual(placement.n_missing, 2)
        self.assertEqual(placement.n_geometric, 1)
        self.assertEqual(placement.n_anchored, 0)

    def test_an_arm_that_never_recorded_the_field_is_not_evaluable(self):
        write_scene(self.tmp / "full", "sceneA", [pre_adr(), pre_adr()])
        placement = read_arm(str(self.tmp / "full"), arm="full")
        self.assertFalse(placement.evaluable)
        text = "\n".join(format_arm(placement))
        self.assertIn("NOT RECORDED on all 2", text)
        self.assertIn("BEFORE the placement", text)

    def test_an_arm_that_recorded_it_as_false_everywhere_IS_evaluable(self):
        """The two failures are different findings and the reader must not merge them.

        All-geometric means the anchor never qualified, which is ADR-0022's own fallback and
        can be the design (a class with no `anchor_object` row). All-missing means the change
        never ran. Reporting the second as the first is how a stale sweep passes.
        """
        write_scene(self.tmp / "full", "sceneA", [geometric(), geometric()])
        placement = read_arm(str(self.tmp / "full"), arm="full")
        self.assertTrue(placement.evaluable)
        text = "\n".join(format_arm(placement))
        self.assertIn("RECORDED AND NEVER TRUE", text)
        self.assertIn("provenance.txt", text)

    def test_a_mixed_arm_says_so(self):
        write_scene(self.tmp / "full", "sceneA", [anchored(), pre_adr()])
        text = "\n".join(format_arm(read_arm(str(self.tmp / "full"), arm="full")))
        self.assertIn("MIXED:", text)
        self.assertIn("two different runners", text)


class TestTheBranchesAreNeverPooled(unittest.TestCase):
    """ADR-0022: an episode placed geometrically is one the memory prior could not win."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_the_two_reached_rates_are_counted_separately(self):
        write_scene(self.tmp / "full", "sceneA", [
            anchored(REACHED), anchored(REACHED), anchored(ABANDONED),
            geometric(ABANDONED), geometric(ABANDONED),
        ])
        placement = read_arm(str(self.tmp / "full"), arm="full")
        self.assertEqual((placement.n_anchored, placement.n_reached_anchored), (3, 2))
        self.assertEqual((placement.n_geometric, placement.n_reached_geometric), (2, 0))
        text = "\n".join(format_arm(placement))
        self.assertIn("BOTH BRANCHES PRESENT", text)

    def test_an_empty_branch_prints_n_a_and_not_zero_percent(self):
        write_scene(self.tmp / "full", "sceneA", [anchored(), anchored()])
        text = "\n".join(format_arm(read_arm(str(self.tmp / "full"), arm="full")))
        self.assertIn("n/a", text)
        self.assertNotIn("0/0", text)

    def test_reached_uses_the_funnel_stage_window_report_uses(self):
        """Two definitions of "reached" in one repo is how a reader quotes the wrong one."""
        write_scene(self.tmp / "full", "sceneA", [
            (FunnelStage.SOURCE_REACHED, {ANCHOR_METRIC: 1.0}),
            (FunnelStage.INVESTIGATE_ENTERED, {ANCHOR_METRIC: 1.0}),
        ])
        placement = read_arm(str(self.tmp / "full"), arm="full")
        self.assertEqual(placement.n_reached_anchored, 1)


class TestScenesAndBarrenDirectories(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_zero_yield_scene_is_carried_as_a_measurement(self):
        """`mL8ThkuaVTM` builds no episode in any sweep this repo has run, and calling that
        a failure is what made `abl-1` exit 1 over five complete arms."""
        arm = self.tmp / "full"
        write_scene(arm, "sceneA", [anchored()])
        (arm / "barren" / "episodes").mkdir(parents=True)
        placement = read_arm(str(arm), arm="full")
        self.assertEqual(placement.barren, ("barren",))
        self.assertEqual(placement.n, 1)
        self.assertIn("zero yield", "\n".join(format_arm(placement)))

    def test_an_unrecorded_scene_id_is_its_own_row(self):
        arm = self.tmp / "full"
        scene_dir = arm / "sceneA"
        scene_dir.mkdir(parents=True)
        write_episode(
            str(scene_dir), 0, AgentReport(resumed=True),
            EpisodeAudit(episode_index=0, metrics={ANCHOR_METRIC: 1.0}),
        )
        placement = read_arm(str(arm), arm="full")
        self.assertEqual([scene.scene for scene in placement.scenes], [UNRECORDED_SCENE])


class TestReadSweep(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_arms_are_read_in_the_order_asked_for_and_absent_ones_are_skipped(self):
        write_scene(self.tmp / "full", "sceneA", [anchored()])
        write_scene(self.tmp / "scan-only", "sceneA", [anchored()])
        placements = read_sweep(str(self.tmp))
        self.assertEqual([p.arm for p in placements], ["full", "scan-only"])

    def test_a_bare_arm_directory_is_read_as_one_arm(self):
        write_scene(self.tmp / "full", "sceneA", [anchored()])
        placements = read_sweep(str(self.tmp / "full"))
        self.assertEqual([p.arm for p in placements], ["full"])
        self.assertEqual(placements[0].n_anchored, 1)


class TestTheExitCode(unittest.TestCase):
    """NOT_RUN is red. A reader that exits 0 over a sweep it could not evaluate is how
    `pilot-1` let its driver print a summary over three arms it never read."""

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_a_missing_directory_is_two(self):
        self.assertEqual(main([str(self.tmp / "nope")]), 2)

    def test_an_evaluable_sweep_is_zero(self):
        write_scene(self.tmp / "full", "sceneA", [anchored()])
        self.assertEqual(main([str(self.tmp)]), 0)

    def test_a_sweep_with_no_recorded_field_anywhere_is_one(self):
        write_scene(self.tmp / "full", "sceneA", [pre_adr(), pre_adr()])
        self.assertEqual(main([str(self.tmp)]), 1)

    def test_an_empty_sweep_is_one(self):
        self.assertEqual(main([str(self.tmp)]), 1)


class TestTheArmNamesMatchTheDriver(unittest.TestCase):
    """Two lists of arm names that drift apart is a reader that silently skips an arm and
    prints a complete-looking sweep. The driver's list is read as shipped."""

    def test_ablation_arms_equals_the_drivers_arm_names(self):
        for line in DRIVER.read_text().splitlines():
            if line.startswith("ARM_NAMES=("):
                names = tuple(line[len("ARM_NAMES=("):].rstrip(")").split())
                self.assertEqual(names, ABLATION_ARMS)
                return
        self.fail("ARM_NAMES=( is not defined in {}".format(DRIVER))


if __name__ == "__main__":
    unittest.main(verbosity=2)
