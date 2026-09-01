"""The pilot's readout, over run directories written by the REAL writer.

``pilot-1`` is the reason this file exists. Three arms, 120 episodes, 42 minutes of V100
time, every episode on disk — and the readout said ``NO EPISODES ON DISK -- this arm did
not run`` for all three, because it walked for a file called ``audit.json`` and the writer
names them ``ep0000.audit.json``. The reader was a heredoc inside a bash string, so no
test in the tree could have caught a one-word mismatch that cost the entire run.

So the loading seam is exercised here rather than mocked, for the reason
``test_episode_diff.py`` gives: an aggregate that is correct over injected dictionaries
and never finds the files on disk prints a clean, confident "nothing here".
"""

import pathlib
import shutil
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.agent import AgentReport
from earshot.report.artifacts import write_episode
from earshot.report.audit import (
    EpisodeAudit,
    FunnelStage,
    SoundingWindowRecord,
    StepRecord,
)
from earshot.tools.window_report import (
    PILOT_ARMS,
    format_arm,
    format_report,
    main,
    read_arm,
    read_sweep,
    scene_dirs,
)

REACHED = FunnelStage.PRIMARY_RESUMED
ABANDONED = FunnelStage.INVESTIGATE_ENTERED

# A record every clause of `runner.tail_is_active` accepts: the buffer was built (hop,
# window) AND it was handed a real IR (max_ir_samples > 0) AND that width outlived a step
# (tail_steps > 0). Anything less is the forced-failure arm below.
ACTIVE_TAIL = dict(
    hop_samples=16000,
    analysis_window_samples=80000,
    max_ir_samples=9600,
    tail_steps=2,
    cue_tail_steps=1,
    n_buffer_grows=0,
    step_seconds=1.0,
)


def window(offset_step, *, audible=0, active=True):
    fields = dict(ACTIVE_TAIL) if active else dict(
        hop_samples=16000, analysis_window_samples=80000, max_ir_samples=0,
        tail_steps=None, cue_tail_steps=None, n_buffer_grows=0, step_seconds=1.0,
    )
    return SoundingWindowRecord(
        opens_at=0,
        offset_step=offset_step,
        policy="fixed_steps",
        post_offset_audible_steps=audible,
        **fields
    )


def steps(n, *, render_s=0.05):
    return tuple(
        StepRecord(step=i, measured_rms=0.1, audio_render_s=render_s) for i in range(n)
    )


def write_scene(arm_dir, scene, episodes):
    """One scene directory. `episodes` is a list of EpisodeAudit-shaped kwargs."""
    scene_dir = pathlib.Path(arm_dir) / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(episodes):
        audit = EpisodeAudit(episode_index=index, scene_id=scene, **spec)
        write_episode(str(scene_dir), index, AgentReport(resumed=True), audit)
    return scene_dir


def episode(*, stage=REACHED, n_steps=20, reached_step=None, win=None, metrics=None,
            render_s=0.05, arms=None):
    spec = dict(
        funnel_stage=stage,
        steps=steps(n_steps, render_s=render_s),
        source_reached_step=reached_step,
        sounding_window=win,
        metrics=metrics or {},
    )
    # Omitted by default, so the pre-ADR-0018 record — every audit already on disk — is
    # what most of this file exercises, and the arm columns are read on an episode that
    # actually carries them.
    if arms is not None:
        spec.update(arms)
    return spec


LIVE_ARMS = dict(climb_rule="live", lateral_cue="live", cast_policy="cast",
                 ir_policy="full")
OFF_ARMS = dict(climb_rule="off", lateral_cue="off", cast_policy="scan_only",
                ir_policy="anechoic")


class TestTheReaderFindsWhatTheWriterWrote(unittest.TestCase):
    """The pilot-1 defect, pinned."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def arm(self, name):
        path = pathlib.Path(self.root) / name
        path.mkdir()
        return path

    def test_episodes_written_by_the_real_writer_are_read_back(self):
        """The one assertion `pilot-1` needed and did not have."""
        arm = self.arm("win-alarm")
        write_scene(arm, "sceneA", [episode(), episode(stage=ABANDONED)])
        write_scene(arm, "sceneB", [episode()])

        reading = read_arm(str(arm), arm="win-alarm")

        self.assertEqual(reading.n_episodes, 3)
        self.assertEqual(reading.scenes, ("sceneA", "sceneB"))
        self.assertEqual(reading.n_source_reached, 2)
        print("read {} episode(s) over {}".format(reading.n_episodes, reading.scenes))

    def test_the_file_on_disk_is_not_called_audit_json(self):
        """The mismatch itself, stated as an assertion rather than as a memory.

        `pilot-1`'s heredoc did `if name != "audit.json": continue`, which is true of
        every file this writer produces. A reader keyed on the wrong name is not
        detectable from its output — it reports an empty run, which is what an empty run
        also reports."""
        arm = self.arm("win-alarm")
        scene = write_scene(arm, "sceneA", [episode()])

        names = sorted(path.name for path in (scene / "episodes").iterdir())

        self.assertEqual(names, ["ep0000.agent.json", "ep0000.audit.json"])
        self.assertEqual(
            [name for name in names if name == "audit.json"], [],
            "the exact-name walk that cost pilot-1 its readout finds nothing")

    def test_a_log_beside_the_scene_dirs_is_not_counted_as_a_scene(self):
        """The driver writes `<arm>-<scene>.log` beside the arm directories, and a
        stray directory without `episodes/` is not a scene that produced nothing."""
        arm = self.arm("win-alarm")
        write_scene(arm, "sceneA", [episode()])
        (arm / "not-a-scene").mkdir()

        self.assertEqual([path.name for path in scene_dirs(str(arm))], ["sceneA"])


class TestNothingReadIsNotRunAndNeverZero(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_an_empty_arm_reads_as_not_run(self):
        (pathlib.Path(self.root) / "win-burst").mkdir()

        reading = read_arm(str(pathlib.Path(self.root) / "win-burst"), arm="win-burst")

        self.assertEqual(reading.n_episodes, 0)
        self.assertIsNone(reading.reached_rate, "0 of 0 is not a rate of 0.0")
        self.assertIn("NOT_RUN", format_arm(reading)[0])

    def test_a_missing_arm_is_reported_not_dropped(self):
        """Three arms asked for, one on disk: the report still has three rows."""
        write_scene(pathlib.Path(self.root) / "cont-alarm", "sceneA", [episode()])

        readings = read_sweep(self.root)

        self.assertEqual(tuple(r.arm for r in readings), PILOT_ARMS)
        self.assertEqual([r.n_episodes for r in readings], [1, 0, 0])

    def test_main_exits_nonzero_when_the_whole_sweep_is_empty(self):
        self.assertEqual(main([self.root]), 1)
        write_scene(pathlib.Path(self.root) / "cont-alarm", "sceneA", [episode()])
        self.assertEqual(main([self.root]), 0)


class TestSwsComesFromTheTallyAndIsNeverRederived(unittest.TestCase):
    """Both arms, per ADR-0014: the healthy path publishing, and the bar firing."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_the_healthy_arm_publishes_an_sws_with_sr_beside_it(self):
        arm = pathlib.Path(self.root) / "win-alarm"
        write_scene(arm, "sceneA", [
            # ran past its offset step and reached the source after it: the numerator
            episode(n_steps=20, reached_step=15, win=window(10, audible=3)),
            # ran past the offset step, reached the source BEFORE it: eligible, not counted
            episode(n_steps=20, reached_step=4, win=window(10, audible=2)),
            # ended before its own offset step: not eligible at all
            episode(n_steps=8, stage=ABANDONED, win=window(10)),
        ])

        reading = read_arm(str(arm), arm="win-alarm")

        self.assertEqual(reading.tally.n_window_closed, 2)
        self.assertAlmostEqual(reading.sws, 0.5)
        self.assertIsNone(reading.sws_refused)
        self.assertEqual(reading.tally.n_tail_audible, 2)
        self.assertEqual(reading.n_source_reached, reading.tally.n_source_reached,
                         "one definition of reached, not two")
        print("SWS {}/{}".format(reading.tally.n_reached_after_offset,
                                 reading.tally.n_window_closed))

    def test_an_eligible_episode_with_no_tail_refuses_the_sws_and_says_so(self):
        """ADR-0017's bar. The refusal is REPORTED, not raised through the readout: the
        other numbers on the arm — the episode's cost above all — are what the pilot ran
        for, and losing them to an exception loses the run twice."""
        arm = pathlib.Path(self.root) / "win-alarm"
        write_scene(arm, "sceneA", [
            episode(n_steps=20, reached_step=15, win=window(10, active=False)),
        ])

        reading = read_arm(str(arm), arm="win-alarm")

        self.assertIsNone(reading.sws)
        self.assertIsNotNone(reading.sws_refused)
        self.assertIn("no active reverb tail", reading.sws_refused)
        self.assertEqual(reading.n_episodes, 1, "the arm is still read")
        self.assertIn("SWS REFUSED", "\n".join(format_arm(reading)))

    def test_the_continuous_arm_has_no_offset_step_and_reports_not_run(self):
        """Not a zero. A continuous episode has no silent phase to succeed in.

        `heard_within_window` is written by the runner only under an offset step, so on
        this arm it is absent rather than false. `pilot-1`'s first readable output put
        "heard while still sounding: 0 of 40" for `cont-alarm` beside "40 of 40" for the
        windowed arms, which reads as a stark difference between the arms and is a
        metric that does not exist on one of them."""
        arm = pathlib.Path(self.root) / "cont-alarm"
        write_scene(arm, "sceneA", [
            episode(n_steps=20, reached_step=15, win=window(None)),
        ])

        reading = read_arm(str(arm), arm="cont-alarm")
        lines = "\n".join(format_arm(reading))

        self.assertIsNone(reading.sws)
        self.assertIsNone(reading.sws_refused)
        self.assertIn("SWS NOT_RUN", lines)
        self.assertEqual(reading.n_windowed, 0)
        self.assertIn("heard while still sounding: n/a", lines)
        self.assertNotIn("0 of 1", lines, "absent is not zero")


class TestTheThreeNumbersThePilotRunsFor(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def test_the_episode_cost_is_read_off_the_render_summary(self):
        """Number one: nobody has ever timed a windowed episode."""
        arm = pathlib.Path(self.root) / "win-alarm"
        write_scene(arm, "sceneA", [
            episode(n_steps=10, render_s=0.1),
            episode(n_steps=20, render_s=0.1),
        ])

        reading = read_arm(str(arm), arm="win-alarm")

        self.assertAlmostEqual(reading.mean_steps, 15.0)
        self.assertAlmostEqual(reading.mean_audio_s, 1.5)
        self.assertAlmostEqual(reading.max_step_audio_s, 0.1)
        print("cost {:.1f} steps/ep, {:.2f} audio s/ep".format(
            reading.mean_steps, reading.mean_audio_s))

    def test_an_arm_that_never_heard_the_source_says_so_rather_than_printing_nothing(self):
        """Number three's failure mode: if `climb_eps` sits above the level a bursty clip
        reaches, `is_rising` never fires. An empty delay list is that result, and it must
        not read the same as a missing field."""
        arm = pathlib.Path(self.root) / "win-burst"
        write_scene(arm, "sceneA", [
            episode(metrics={"onset_delay_censored": 1.0}),
            episode(metrics={"onset_delay_censored": 1.0}),
        ])

        lines = "\n".join(format_arm(read_arm(str(arm), arm="win-burst")))

        self.assertIn("ONSET NEVER FIRED", lines)
        self.assertIn("2 censored", lines)

    def test_the_onset_delay_and_the_cue_tail_are_read_off_the_metrics(self):
        arm = pathlib.Path(self.root) / "win-alarm"
        write_scene(arm, "sceneA", [
            episode(win=window(10),
                    metrics={"onset_delay_steps": 2.0, "sounding_cue_tail_steps": 1.0,
                             "sounding_phase_folds": 5.0, "heard_within_window": 1.0}),
            episode(win=window(10),
                    metrics={"onset_delay_steps": 6.0, "sounding_cue_tail_steps": 3.0,
                             "sounding_phase_folds": 5.0, "heard_within_window": 1.0}),
            episode(win=window(10), metrics={"onset_delay_censored": 1.0}),
        ])

        reading = read_arm(str(arm), arm="win-alarm")

        self.assertEqual(reading.onset_delays, (2.0, 6.0))
        self.assertEqual(reading.n_onset_censored, 1)
        self.assertEqual(reading.n_heard_within_window, 2)
        self.assertEqual(reading.n_windowed, 3, "the denominator is windowed episodes")
        self.assertEqual(sorted(reading.cue_tail_steps), [1, 3])
        lines = "\n".join(format_arm(reading))
        self.assertIn("cue_tail_steps 1x1 1x3", lines)
        self.assertIn("CENSORED (never heard it): 1", lines)

    def test_the_report_carries_the_flip_rate_rather_than_leaving_it_to_the_reader(self):
        """At n=120 a between-arm delta is a direction. The readout says so where the
        numbers are, not in a driver header the emailed tail may have scrolled past."""
        text = format_report(read_sweep(self.root))
        self.assertIn("16.2% flip rate", text)


class TestTheSourceSideColumns(unittest.TestCase):
    """The new columns, both arms each: present-and-printed, absent-and-said-so.

    `pilot-1` is the standing reason. A metric the runner computes and the readout does
    not print is a metric that does not exist, and the failure is silent — an arm reads
    as ordinary attrition either way. So each column is asserted on a record the REAL
    writer produced, in both the populated and the empty case.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def arm_with(self, episodes, name="win-alarm"):
        arm = pathlib.Path(self.root) / name
        write_scene(arm, "sceneA", episodes)
        return read_arm(str(arm), arm=name)

    def test_the_source_spl_and_its_success_flag_are_read_and_printed(self):
        reading = self.arm_with([
            episode(reached_step=9,
                    metrics={"source_spl": 0.8, "source_find_sr_1m": 1.0}),
            episode(reached_step=12,
                    metrics={"source_spl": 0.4, "source_find_sr_1m": 1.0}),
            # Had a route and did not reach it: a real, reportable zero.
            episode(stage=ABANDONED,
                    metrics={"source_spl": 0.0, "source_find_sr_1m": 0.0}),
        ])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(reading.source_spls, (0.8, 0.4, 0.0))
        self.assertEqual(reading.n_source_spl_absent, 0)
        self.assertEqual(reading.n_source_find_sr_1m, 2)
        self.assertIn("source SPL: n=3 of 3", lines)
        self.assertIn("mean 0.400", lines)
        self.assertIn("Find-SR@1m 2 of 3", lines)
        print("source SPL n=3 mean {:.3f}, Find-SR@1m {}/3".format(
            sum(reading.source_spls) / 3.0, reading.n_source_find_sr_1m))

    def test_an_unwinnable_episode_is_absent_with_its_reason_and_never_a_zero(self):
        """The forced-failure arm. 23 of `yield-2`'s 365 episodes have no navmesh route
        to their source at all, so the runner writes no `source_spl` for them. Counting
        those as SPL 0.0 and as a failed find is the confusion the key exists to
        prevent."""
        reading = self.arm_with([
            episode(reached_step=9,
                    metrics={"source_spl": 0.8, "source_find_sr_1m": 1.0}),
            # No route from the start pose: no key at all.
            episode(stage=ABANDONED, metrics={}),
        ])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(reading.source_spls, (0.8,))
        self.assertEqual(reading.n_source_spl_absent, 1)
        self.assertEqual(reading.n_source_find_sr_1m, 1,
                         "the denominator is the episodes that had one, not the arm")
        self.assertIn("source SPL: n=1 of 2", lines)
        self.assertIn("ABSENT on 1 of 2", lines)
        self.assertIn("unwinnable, not a zero", lines)
        self.assertIn("Find-SR@1m 1 of 1", lines)
        self.assertNotIn("Find-SR@1m 1 of 2", lines, "absent is not a miss")

    def test_an_arm_with_no_source_spl_at_all_says_absent_rather_than_printing_zero(self):
        reading = self.arm_with([episode(), episode()])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(reading.source_spls, ())
        self.assertEqual(reading.n_source_spl_absent, 2)
        self.assertIn("source SPL: n/a on all 2 episode(s)", lines)
        self.assertIn("ABSENT, not 0.0.", lines)
        self.assertNotIn("mean 0.000", lines)

    def test_the_final_pose_route_is_read_and_labelled_as_not_the_closest_approach(self):
        """`min_d2source_m` is a MINIMUM over the episode and this is a distance at its
        end. An episode that walked to the source and back has a small one and a large
        other, so the column has to name which it is where it is printed."""
        reading = self.arm_with([
            episode(metrics={"dtg_source_final_m": 2.0, "min_d2source_m": 0.4}),
            episode(metrics={"dtg_source_final_m": 6.0, "min_d2source_m": 0.5}),
        ])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(reading.dtg_source_final, (2.0, 6.0))
        self.assertEqual(reading.n_dtg_source_final_absent, 0)
        self.assertIn("route to source at the FINAL pose (not the closest approach)",
                      lines)
        self.assertIn("n=2 of 2", lines)
        self.assertIn("median 4.00 m", lines)
        print("final-pose route median 4.00 m over 2 episode(s)")

    def test_an_unrouted_final_pose_is_absent_with_its_reason(self):
        reading = self.arm_with([
            episode(metrics={"dtg_source_final_m": 2.0}),
            episode(metrics={}),
        ])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(reading.dtg_source_final, (2.0,))
        self.assertEqual(reading.n_dtg_source_final_absent, 1)
        self.assertIn("ABSENT on 1 of 2: the final pose has no navmesh route", lines)
        self.assertIn("not a distance of 0", lines)

    def test_an_arm_with_no_final_pose_route_at_all_says_absent(self):
        reading = self.arm_with([episode(), episode()])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(reading.dtg_source_final, ())
        self.assertIn("route to source at the FINAL pose: n/a on all 2 episode(s)",
                      lines)
        self.assertNotIn("median 0.00 m", lines)


class TestTheAblationArmsAreReadOffTheRecord(unittest.TestCase):
    """Which arm ran, per episode. Both arms: recorded-and-printed, and absent."""

    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.root, True)

    def arm_with(self, episodes, name="win-alarm"):
        arm = pathlib.Path(self.root) / name
        write_scene(arm, "sceneA", episodes)
        return read_arm(str(arm), arm=name)

    def test_a_uniform_arm_prints_its_one_setting(self):
        reading = self.arm_with([episode(arms=OFF_ARMS), episode(arms=OFF_ARMS)])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(
            reading.ablation_arms,
            ("climb=off lateral=off cast=scan_only ir=anechoic",))
        self.assertEqual(reading.n_arms_unrecorded, 0)
        self.assertIn("arms: climb=off lateral=off cast=scan_only ir=anechoic", lines)
        self.assertNotIn("MIXED", lines)
        print("arms read back: {}".format(reading.ablation_arms))

    def test_two_settings_under_one_arm_directory_are_reported_as_mixed(self):
        """The forced failure. `episode_diff` pairs by index across two directories, so a
        directory holding both settings is a comparison that is not comparing like with
        like — and nothing else in the tree would say so."""
        reading = self.arm_with([episode(arms=LIVE_ARMS), episode(arms=OFF_ARMS)])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(len(reading.ablation_arms), 2)
        self.assertIn("MIXED", lines)
        self.assertIn("not comparing like with like", lines)

    def test_a_record_written_before_the_arms_reads_as_unknown_not_as_the_default(self):
        """`None` on all four is "this run predates the arms". Printing it as the
        shipped defaults would let an unmeasured run read as a measured one."""
        reading = self.arm_with([episode(), episode(arms=LIVE_ARMS)])
        lines = "\n".join(format_arm(reading))

        self.assertEqual(reading.n_arms_unrecorded, 1)
        self.assertIn("arms NOT RECORDED on 1 of 2", lines)
        self.assertIn("UNKNOWN rather than the default", lines)
        self.assertEqual(reading.ablation_arms,
                         ("climb=live lateral=live cast=cast ir=full",))

    def test_an_empty_arm_carries_no_arm_label_and_no_absence_count(self):
        (pathlib.Path(self.root) / "win-burst").mkdir()

        reading = read_arm(str(pathlib.Path(self.root) / "win-burst"), arm="win-burst")

        self.assertEqual(reading.ablation_arms, ())
        self.assertEqual(reading.n_arms_unrecorded, 0)
        self.assertEqual(reading.source_spls, ())
        self.assertEqual(reading.n_source_spl_absent, 0,
                         "0 of 0 absent is not 0 absences to explain")
        self.assertIn("NOT_RUN", format_arm(reading)[0])


class TestTheDriverUsesThisReader(unittest.TestCase):
    """The wiring, asserted. A tested reader the driver does not call is worth nothing,
    and `pilot-1` is the proof: every part of that run worked except the join."""

    def setUp(self):
        root = pathlib.Path(__file__).resolve().parents[2]
        self.driver = (root / "tools" / "window_pilot.sh").read_text()

    def test_the_driver_calls_the_tested_reader_and_not_a_second_one(self):
        self.assertIn("earshot.tools.window_report", self.driver)
        self.assertNotIn('python - "$OUT_DIR" <<', self.driver,
                         "the untested heredoc reader is gone, not shadowed by a new one")

    def test_episode_diff_is_handed_arm_directories_and_not_scene_directories(self):
        """`load_outcomes` refuses a scene directory by design — it wants the tag
        directory holding one subdirectory per scene, which for this sweep is the ARM
        directory. `pilot-1` passed it scene directories four times and printed four
        refusals where the comparison should have been."""
        self.assertIn('episode_diff \\\n    "$OUT_DIR/cont-alarm" "$OUT_DIR/win-alarm"',
                      self.driver)

    def test_criterion_9_is_armed_the_way_yield_sweep_arms_it(self):
        """NOT_RUN is red. A sweep that leaves criterion 9 structurally NOT_RUN goes red
        on every scene, which is how a reader learns to skip a criterion — the exact
        habit that let a never-armed canary read as a pass."""
        self.assertIn("reset_manifest --verify-absent --when before", self.driver)
        self.assertIn("reset_manifest --write-record", self.driver)


if __name__ == "__main__":
    unittest.main()
