"""The DREAM readout, over run directories written by the REAL writer.

``dream-1`` is the reason this file exists in the shape it does. Four hours and
forty-seven minutes of box time wrote ``dream_omega_e_spread`` onto 282 episodes and no
reader in the tree could print it, so the run's own central quantity reached nobody. The
loading seam is therefore exercised against ``report.artifacts.write_episode`` rather
than against injected dictionaries, for the reason ``test_window_report.py`` gives: an
aggregate that is correct over hand-built objects and never finds the files on disk
prints a clean, confident "nothing here", which is what an empty run also prints.

Both arms of every capability, per ADR-0014. A reader that detects a flat omega is only
worth having if it also detects a live one, and each verdict below is asserted with its
opposite beside it.
"""

import pathlib
import shutil
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_episode
from earshot.report.audit import EpisodeAudit, FunnelStage, StepRecord
from earshot.task.dream import DreamKnobs
from earshot.memory.consolidate import ImportanceWeights
from earshot.task.plan import PlanWeights
from earshot.tools.dream_report import (
    FLAT_OMEGA_SPREAD,
    KNOB_KEYS,
    ROW_BANDS,
    band_of,
    format_knobs,
    format_omega,
    format_precondition,
    format_reach,
    format_report,
    format_retention,
    main,
    present,
    read_rows,
)
from earshot.tools.window_report import read_arm

REACHED = FunnelStage.PRIMARY_RESUMED
ABANDONED = FunnelStage.INVESTIGATE_ENTERED


def steps(n):
    return tuple(
        StepRecord(step=i, measured_rms=0.1, audio_render_s=0.05) for i in range(n)
    )


def dream_metrics(
    *,
    informed=20.0,
    spread=0.002,
    rows_at_start=5.0,
    rows_added=2.0,
    cosine=(0.91, 0.95, 0.99),
    step_s=0.05,
    omega_e=0.5,
    omega_p=0.3,
    omega_k=0.2,
):
    """One episode's DREAM metrics, in the shape the runner writes them.

    ``rows_at_start`` is the number the reader has to DERIVE: the runner records the
    count after consolidation plus the delta, never the starting size, so a reader that
    reads ``dream_experience_rows`` as "the memory this episode used" is off by whatever
    the episode itself added.
    """
    metrics = {
        "dream_informed_steps": informed,
        "dream_experience_rows": rows_at_start + rows_added,
        "dream_rows_added": rows_added,
        "dream_pattern_rows": 1.0,
        "dream_tau_steps": 20.0,
        "dream_step_s_mean": step_s,
        "dream_step_s_worst": step_s * 2.0,
        "dream_importance_min": 1.0,
        "dream_importance_max": 1.29,
    }
    if informed:
        metrics.update(
            {
                "dream_omega_e_spread": spread,
                "dream_omega_e_mean": omega_e,
                "dream_omega_e_min": omega_e - spread / 2.0,
                "dream_omega_e_max": omega_e + spread / 2.0,
                "dream_omega_p_mean": omega_p,
                "dream_omega_k_mean": omega_k,
            }
        )
    if cosine is not None:
        low, mid, high = cosine
        metrics.update(
            {
                "dream_me_cosine_min": low,
                "dream_me_cosine_mean": mid,
                "dream_me_cosine_max": high,
            }
        )
    metrics.update(dict(KNOBS.as_metrics()))
    return metrics


KNOBS = DreamKnobs(
    stm_horizon=8,
    stm_decay=0.8,
    present_weight=0.7,
    coherence=0.99,
    min_segment=3,
    max_segment=12,
    importance=ImportanceWeights(alpha=1.0, beta=1.0, gamma=1.0),
    eta=0.5,
    max_retained=8,
    min_support=2,
    k_experience=3,
    k_pattern=2,
    k_knowledge=1,
    temperature=0.5,
    plan_weights=PlanWeights(plan=1.0, memory=0.5, feasibility=0.5),
)


def write_scene(arm_dir, scene, episodes):
    """One scene directory, through the real writer."""
    scene_dir = pathlib.Path(arm_dir) / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    for index, (stage, metrics) in enumerate(episodes):
        audit = EpisodeAudit(
            episode_index=index,
            scene_id=scene,
            funnel_stage=stage,
            steps=steps(20),
            metrics=metrics,
        )
        write_episode(str(scene_dir), index, AgentReport(resumed=True), audit)
    return scene_dir


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def arm(self, name="dream"):
        path = pathlib.Path(self.root) / name
        path.mkdir(parents=True, exist_ok=True)
        return path


class TestTheReaderFindsWhatTheWriterWrote(Fixture):
    """The `dream-1` gap, pinned: the numbers exist on disk and are read back."""

    def test_episodes_written_by_the_real_writer_are_read_back(self):
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, dream_metrics()),
            (ABANDONED, dream_metrics(spread=0.4)),
        ])
        write_scene(arm, "sceneB", [(REACHED, dream_metrics())])

        rows = read_rows(str(arm))

        self.assertEqual(len(rows), 3)
        self.assertEqual(tuple(row.scene for row in rows),
                         ("sceneA", "sceneA", "sceneB"))
        self.assertEqual([row.omega_e_spread for row in rows], [0.002, 0.4, 0.002])
        print("read {} episode(s) over {}".format(
            len(rows), sorted({row.scene for row in rows})))

    def test_reached_agrees_with_window_report_on_the_same_records(self):
        """One definition of 'reached' in this repo, not two.

        `window_report` counts `funnel_stage >= SOURCE_REACHED` and says in its own
        comment that a second definition is how a reader comes to quote the wrong one.
        This asserts the two agree on records the real writer produced.
        """
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, dream_metrics()),
            (ABANDONED, dream_metrics()),
            (FunnelStage.SOURCE_REACHED, dream_metrics()),
        ])

        rows = read_rows(str(arm))
        reading = read_arm(str(arm), arm="dream")

        self.assertEqual(sum(1 for row in rows if row.reached),
                         reading.n_source_reached)
        print("reached: dream_report {} == window_report {}".format(
            sum(1 for row in rows if row.reached), reading.n_source_reached))

    def test_rows_at_start_is_the_memory_the_episode_actually_used(self):
        """The derivation, because the runner records no such field.

        An episode that ended with 7 rows having added 2 retrieved against 5. A reader
        that quoted `dream_experience_rows` would band it as a memory it never saw.
        """
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, dream_metrics(rows_at_start=5.0, rows_added=2.0)),
        ])

        row = read_rows(str(arm))[0]

        self.assertEqual(row.experience_rows, 7.0)
        self.assertEqual(row.rows_added, 2.0)
        self.assertEqual(row.rows_at_start, 5.0)
        print("ended at {:.0f} rows, added {:.0f}, retrieved against {:.0f}".format(
            row.experience_rows, row.rows_added, row.rows_at_start))


class TestOmegaVerdictHasBothArms(Fixture):
    """A reader that can only say FLAT is a reader that says FLAT about anything."""

    def read(self, episodes, name=None):
        # A fresh arm directory per call: `write_episode` refuses to overwrite, which is
        # the artefact rule (`report/` never overwrites) and not a test inconvenience.
        arm = self.arm(name or "dream-{}".format(len(list(pathlib.Path(self.root)
                                                          .iterdir()))))
        write_scene(arm, "sceneA", episodes)
        return format_omega(read_rows(str(arm)))

    def test_a_flat_omega_is_called_flat(self):
        text = self.read([(REACHED, dream_metrics(spread=0.002))] * 5)
        self.assertIn("FLAT ON EVERY EPISODE", text)
        self.assertNotIn("VERDICT: LIVE", text)
        print(text)

    def test_a_live_omega_is_called_live(self):
        """THE FORCED-FAILURE ARM. Same reader, spreads well over the floor."""
        text = self.read([(REACHED, dream_metrics(spread=0.35))] * 5)
        self.assertIn("VERDICT: LIVE", text)
        self.assertNotIn("FLAT ON EVERY EPISODE", text)
        print(text)

    def test_a_mixed_arm_is_called_mixed(self):
        text = self.read(
            [(REACHED, dream_metrics(spread=0.002))] * 3
            + [(REACHED, dream_metrics(spread=0.35))] * 2
        )
        self.assertIn("VERDICT: MIXED", text)
        print(text)

    def test_the_floor_is_the_constant_and_not_a_number_in_prose(self):
        """A spread just under and just over `FLAT_OMEGA_SPREAD` decide differently."""
        under = self.read([(REACHED, dream_metrics(spread=FLAT_OMEGA_SPREAD - 1e-6))])
        over = self.read([(REACHED, dream_metrics(spread=FLAT_OMEGA_SPREAD + 1e-6))])
        self.assertIn("FLAT ON EVERY EPISODE", under)
        self.assertIn("VERDICT: LIVE", over)
        print("floor {:.2f}: {:.6f} -> flat, {:.6f} -> live".format(
            FLAT_OMEGA_SPREAD, FLAT_OMEGA_SPREAD - 1e-6, FLAT_OMEGA_SPREAD + 1e-6))


class TestAbsentIsNeverZero(Fixture):
    """The rule this tree has paid for twice, asserted rather than commented."""

    def test_an_episode_that_recorded_nothing_is_not_an_episode_with_omega_zero(self):
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, {}),
            (REACHED, {}),
            (REACHED, dream_metrics(spread=0.3)),
        ])

        rows = read_rows(str(arm))
        text = format_omega(rows)

        values, absent = present([row.informed_steps for row in rows])
        self.assertEqual((len(values), absent), (1, 2))
        self.assertIn("NOT RECORDED: 2", text)
        # And the one episode that DID record is the only one in the spread statistic.
        self.assertIn("min 0.3000", text)
        print(text)

    def test_omega_never_defined_is_counted_apart_from_omega_flat(self):
        """`informed_steps == 0` and a missing key are different findings.

        A retrieval that never resolved wrote 0 informed steps and no omega at all; an
        episode that predates the instrumentation wrote neither. Folding them loses the
        distinction PR #117 added the denominator for.
        """
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, dream_metrics(informed=0.0)),
            (REACHED, dream_metrics(informed=0.0)),
            (REACHED, dream_metrics(informed=12.0, spread=0.3)),
            (REACHED, {}),
        ])

        text = format_omega(read_rows(str(arm)))

        self.assertIn("omega NEVER DEFINED (0 informed steps): 2", text)
        self.assertIn("omega defined on >=1 step:             1", text)
        self.assertIn("NOT RECORDED: 1", text)
        print(text)

    def test_an_empty_memory_has_no_key_spread_and_is_not_a_cosine_of_one(self):
        """`key_spread` returns None below two rows, and the band prints dashes.

        A 1.0 in that cell would read as maximal degeneracy — the worst possible result —
        where the truth is that there was nothing to measure.
        """
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, dream_metrics(rows_at_start=0.0, rows_added=2.0, cosine=None)),
            (REACHED, dream_metrics(rows_at_start=6.0, rows_added=1.0,
                                    cosine=(0.90, 0.95, 0.98))),
        ])

        text = format_precondition(read_rows(str(arm)))

        self.assertIn("0 (empty)", text)
        self.assertIn("DASHES BY CONSTRUCTION", text)
        self.assertIn("0.9500", text)
        self.assertNotIn("1.0000", text)
        print(text)


class TestTheBandsSeparateDegeneracyFromSize(Fixture):
    def test_every_band_boundary_lands_where_it_says(self):
        for size, expected in ((0, "0 (empty)"), (1, "1"), (2, "2-4"), (4, "2-4"),
                               (5, "5-9"), (9, "5-9"), (10, "10-19"), (19, "10-19"),
                               (20, "20-49"), (49, "20-49"), (50, "50+"), (900, "50+")):
            self.assertEqual(band_of(float(size)), expected, size)
        print("{} band(s), boundaries checked at both ends of each".format(
            len(ROW_BANDS)))

    def test_a_falling_cosine_is_visible_across_the_bands(self):
        """The shape the run has to produce for omega to have a chance.

        If degeneracy falls as `M^E` fills, the bands show it. This asserts the reader
        can display that shape at all — a pooled median could not.
        """
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, dream_metrics(rows_at_start=2.0, cosine=(0.97, 0.98, 0.99))),
            (REACHED, dream_metrics(rows_at_start=7.0, cosine=(0.80, 0.85, 0.90))),
            (REACHED, dream_metrics(rows_at_start=25.0, cosine=(0.40, 0.55, 0.70))),
        ])

        text = format_precondition(read_rows(str(arm)))
        lines = [line for line in text.splitlines() if line.strip().startswith(
            ("2-4", "5-9", "20-49"))]

        self.assertEqual(len(lines), 3)
        self.assertIn("0.9800", lines[0])
        self.assertIn("0.8500", lines[1])
        self.assertIn("0.5500", lines[2])
        print("\n".join(lines))


class TestRetentionBothArms(Fixture):
    def read(self, episodes, name=None):
        arm = self.arm(name or "dream-{}".format(len(list(pathlib.Path(self.root)
                                                          .iterdir()))))
        write_scene(arm, "sceneA", episodes)
        return format_retention(read_rows(str(arm)))

    def test_an_eta_that_refuses_nothing_is_named(self):
        text = self.read([(REACHED, dream_metrics(rows_added=2.0))] * 4)
        self.assertIn("eta REFUSED NOTHING", text)
        print(text)

    def test_an_eta_that_retains_nothing_is_named(self):
        """THE OTHER ARM. Same reader, the opposite pathology."""
        text = self.read([(REACHED, dream_metrics(rows_added=0.0))] * 4)
        self.assertIn("eta RETAINED NOTHING", text)
        self.assertNotIn("eta REFUSED NOTHING", text)
        print(text)

    def test_a_threshold_that_is_thresholding_is_called_neither(self):
        text = self.read(
            [(REACHED, dream_metrics(rows_added=2.0))] * 2
            + [(REACHED, dream_metrics(rows_added=0.0))] * 2
        )
        self.assertNotIn("eta REFUSED NOTHING", text)
        self.assertNotIn("eta RETAINED NOTHING", text)
        self.assertIn("episodes that retained NOTHING: 2 of 4", text)
        print(text)

    def test_an_empty_pattern_store_is_named_as_an_absence_omega_weighted(self):
        arm = self.arm()
        write_scene(arm, "sceneA", [
            (REACHED, dict(dream_metrics(), dream_pattern_rows=0.0)) for _ in range(3)
        ])
        text = format_retention(read_rows(str(arm)))
        self.assertIn("M^P WAS EMPTY ON EVERY EPISODE", text)
        print(text)


class TestTheMemoryReachIsMeasuredNotAssumed(Fixture):
    def test_a_per_scene_reset_is_detected_and_named(self):
        """The `dream-1` shape: every scene starts from empty.

        The sweep invokes the runner once per scene, so `run()`'s carry-over never
        crosses a scene boundary. This is the finding that says the run could not have
        tested a memory built across scenes.
        """
        arm = self.arm()
        for scene in ("sceneA", "sceneB", "sceneC"):
            write_scene(arm, scene, [
                (REACHED, dream_metrics(rows_at_start=0.0, rows_added=2.0,
                                        cosine=None)),
                (REACHED, dream_metrics(rows_at_start=2.0, rows_added=2.0)),
                (REACHED, dream_metrics(rows_at_start=4.0, rows_added=2.0)),
            ])

        text = format_reach(read_rows(str(arm)))

        self.assertIn("EVERY SCENE STARTED FROM EMPTY", text)
        self.assertIn("episodes that ran against an EMPTY M^E: 3 of 9", text)
        print(text)

    def test_a_memory_carried_across_scenes_is_not_called_a_reset(self):
        """THE FORCED-FAILURE ARM: the shape a fixed driver would produce."""
        arm = self.arm()
        start = 0.0
        for scene in ("sceneA", "sceneB", "sceneC"):
            episodes = []
            for _ in range(3):
                episodes.append(
                    (REACHED, dream_metrics(rows_at_start=start, rows_added=2.0,
                                            cosine=None if start == 0 else (0.8, 0.9,
                                                                            0.95)))
                )
                start += 2.0
            write_scene(arm, scene, episodes)

        text = format_reach(read_rows(str(arm)))

        self.assertNotIn("EVERY SCENE STARTED FROM EMPTY", text)
        self.assertIn("episodes that ran against an EMPTY M^E: 1 of 9", text)
        self.assertIn("largest M^E any episode ever saw:      16 row(s)", text)
        print(text)


class TestTheKnobsComeFromTheRunAndNotTheDriver(Fixture):
    def test_the_literal_matches_what_dreamknobs_actually_writes(self):
        """THE FENCE. A knob added to `as_metrics` and not to `KNOB_KEYS` is invisible.

        This is why the literal is allowed to be a literal: the drift it could suffer is
        asserted away here against a REAL `DreamKnobs`, so the reader never needs to
        import torch to enumerate eighteen strings.
        """
        written = tuple(name for name, _value in KNOBS.as_metrics())
        self.assertEqual(KNOB_KEYS, written)
        print("{} knob(s), reader and writer agree exactly".format(len(KNOB_KEYS)))

    def test_the_knobs_printed_are_the_ones_on_the_episodes(self):
        arm = self.arm()
        write_scene(arm, "sceneA", [(REACHED, dream_metrics())] * 3)

        text = format_knobs(read_rows(str(arm)))

        self.assertIn("coherence", text)
        self.assertIn("0.99", text)
        self.assertIn("eta", text)
        self.assertNotIn("MIXED ARM", text)
        print(text)

    def test_two_configurations_in_one_directory_are_called_a_mixed_arm(self):
        """THE FORCED-FAILURE ARM. An aggregate over two runs is not one arm."""
        arm = self.arm()
        other = dict(dream_metrics(), dream_eta=0.9)
        write_scene(arm, "sceneA", [(REACHED, dream_metrics()), (REACHED, other)])

        text = format_knobs(read_rows(str(arm)))

        self.assertIn("MIXED ARM", text)
        self.assertIn("dream_eta", text)
        print(text)


class TestTheCommandLine(Fixture):
    def test_a_sweep_directory_and_an_arm_directory_both_read(self):
        arm = self.arm("dream")
        write_scene(arm, "sceneA", [(REACHED, dream_metrics())] * 2)

        self.assertEqual(main([self.root]), 0)
        self.assertEqual(main([str(arm)]), 0)
        print("both layouts read: runs/<tag> and runs/<tag>/dream")

    def test_an_arm_that_recorded_omega_nowhere_exits_nonzero(self):
        """Unreadable is not flat, and the exit code says so.

        Same rule as `placement_report`: a field absent everywhere is a run that cannot
        be read, which must not be reported as a run whose mechanism did nothing.
        """
        arm = self.arm("full")
        write_scene(arm, "sceneA", [(REACHED, {})] * 3)

        self.assertEqual(main([str(arm)]), 2)
        print("an arm with no DREAM metrics exits 2")

    def test_an_empty_directory_exits_nonzero_rather_than_printing_a_clean_zero(self):
        self.assertEqual(main([str(self.arm("dream"))]), 2)
        print("no episodes on disk exits 2")

    def test_the_whole_report_runs_end_to_end(self):
        arm = self.arm("dream")
        write_scene(arm, "sceneA", [
            (REACHED, dream_metrics(rows_at_start=0.0, cosine=None)),
            (ABANDONED, dream_metrics(rows_at_start=2.0)),
            (REACHED, dream_metrics(rows_at_start=4.0, spread=0.3)),
        ])

        text = format_report(read_rows(str(arm)), arm="dream", arm_dir=str(arm))

        for section in ("A. omega_t", "B. THE PRECONDITION", "C. RETENTION",
                        "D. HOW FAR", "E. WHAT IT COST", "F. THE KNOBS"):
            self.assertIn(section, text)
        print(text)


if __name__ == "__main__":
    unittest.main()


class TestASingletonSoftmaxIsNotAFlatWeighting(Fixture):
    """`dream-1`'s real shape: eq. 23 with one element in it.

    `_softmax` EXCLUDES a level that retrieved nothing rather than handing it a 0.0
    logit, so one live level returns 1.0 and its spread is exactly 0.0. Reading that as
    "the weighting did not move" points at the retrieval; the fix is the empty levels.
    """

    def read(self, episodes, name):
        arm = self.arm(name)
        write_scene(arm, "sceneA", episodes)
        return format_omega(read_rows(str(arm)))

    def test_one_live_level_is_named_as_one_live_level(self):
        text = self.read(
            [(REACHED, dream_metrics(spread=0.0, omega_e=1.0, omega_p=0.0,
                                     omega_k=0.0))] * 4,
            "solo",
        )
        self.assertIn("exactly 1 of M^E/M^P/M^K: 4 episode(s)", text)
        self.assertIn("HAD ONLY ONE LIVE LEVEL", text)
        self.assertIn("The empty levels are the fix.", text)
        print(text)

    def test_three_live_levels_draw_no_such_warning(self):
        """THE FORCED-FAILURE ARM. Same reader, a memory whose levels all answered."""
        text = self.read(
            [(REACHED, dream_metrics(spread=0.2, omega_e=0.5, omega_p=0.3,
                                     omega_k=0.2))] * 4,
            "full-stack",
        )
        self.assertIn("exactly 3 of M^E/M^P/M^K: 4 episode(s)", text)
        self.assertNotIn("HAD ONLY ONE LIVE LEVEL", text)
        self.assertNotIn("CARRIED NO WEIGHT ON ANY EPISODE", text)
        print(text)

    def test_a_level_that_never_carried_weight_is_named_with_its_cause(self):
        text = self.read(
            [(REACHED, dream_metrics(spread=0.1, omega_e=0.7, omega_p=0.3,
                                     omega_k=0.0))] * 4,
            "no-knowledge",
        )
        self.assertIn("M^K CARRIED NO WEIGHT ON ANY EPISODE", text)
        self.assertIn("the semantic store was never populated", text)
        self.assertNotIn("M^P CARRIED NO WEIGHT", text)
        print(text)

    def test_the_descriptive_split_names_its_own_confound(self):
        """omega can only move when a second level is live, so the split selects on that.

        Without the sentence the 41.0%-vs-24.7% line in `dream-1` reads as an effect.
        """
        text = self.read(
            [(REACHED, dream_metrics(spread=0.0, omega_e=1.0, omega_p=0.0,
                                     omega_k=0.0))] * 3
            + [(REACHED, dream_metrics(spread=0.3, omega_e=0.6, omega_p=0.4,
                                       omega_k=0.0))] * 3,
            "confound",
        )
        self.assertIn("AND THE SPLIT MAY BE THE SELECTION ITSELF", text)
        self.assertIn("already working", text)
        print(text)
