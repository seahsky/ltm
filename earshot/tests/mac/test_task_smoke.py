"""Task spec §8's gate, judged both ways.

    python -m unittest discover earshot/tests/mac

``judge()`` is pure, so this is ticket 19's **third row** — *given a failing measurement,
does the gate go red* — which needs no box and is the row ticket 13's version-blind skip
would have failed. Every criterion is tested in both directions, because a gate that has
only ever been seen green is a gate whose red path has never run: exactly the shape of
ticket 16's canary that was never armed and ticket 13's probe that skipped and reported
success.

The one thing this cannot check is whether a green verdict means the deletion is safe.
That is criterion 9, which is a property of two runs, and it is NOT_RUN here by
construction rather than absent.
"""

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.guard import AudioContextReport
from earshot.report.audit import (
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    StepRecord,
)
from earshot.task.smoke import (
    REPORT_KEYS,
    CriterionStatus,
    judge,
    judge_run_dir,
)

CEILING_S = 0.5
T_ANOM = 30


def report(**overrides):
    base = {key: None for key in REPORT_KEYS}
    base.update(
        primary_completed=False, investigate_aborted=False, resumed=True, n_benign_ignored=0
    )
    base.update(overrides)
    return base


def steps(n=8, render_s=0.03):
    return tuple(
        StepRecord(
            i,
            1e-3,
            action="move_forward",
            audio_render_s=render_s,
            collided=False,
            displacement_m=0.25,
        )
        for i in range(n)
    )


def audit(**overrides):
    base = dict(
        localization_arm="realizable",
        detector_arm="oracle",
        funnel_stage=FunnelStage.PRIMARY_RESUMED,
        onset=OnsetRecord(
            onset_step=31, pre_onset_rms=1e-3, n_pre_onset_readings=31,
            provenance_asserted=True,
        ),
        audio_context=AudioContextReport(
            n_vertices=392364, submitted_n_vertices=392356, ir_peak_abs=0.37,
            ir_shape=(2, 72300), log_canary_seen=True,
        ),
        steps=steps(),
        metrics={"n_loop_steps": 8.0, "n_renders_in_loop": 8.0},
    )
    base.update(overrides)
    return EpisodeAudit(**base)


def env(**overrides):
    base = {"probes": [{"name": "torch_cuda", "status": "pass"}], "missing": []}
    base.update(overrides)
    return base


def verdict(**overrides):
    kwargs = dict(
        report=report(),
        audit=audit(),
        env_report=env(),
        run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": T_ANOM},
    )
    kwargs.update(overrides)
    return judge(**kwargs)


def status_of(v, number):
    return next(c.status for c in v.criteria if c.number == number)


class TestTheGateGoesGreenOnAGoodRun(unittest.TestCase):
    def test_all_nine_are_reported(self):
        self.assertEqual([c.number for c in verdict().criteria], list(range(1, 10)))

    def test_eight_pass_and_hermeticity_does_not(self):
        v = verdict()
        self.assertEqual(v.failed, (9,))

    def test_a_run_is_not_green_while_criterion_nine_is_outstanding(self):
        """The gate's whole job: eight of nine is not the deletion gate."""
        self.assertFalse(verdict().green)

    def test_criterion_nine_is_not_run_rather_than_failed(self):
        """It was never evaluated here — saying FAIL would blame ticket 27's run."""
        self.assertIs(status_of(verdict(), 9), CriterionStatus.NOT_RUN)


class TestEachCriterionHasARedPath(unittest.TestCase):
    def test_1_renders_short_of_steps(self):
        v = verdict(audit=audit(metrics={"n_loop_steps": 8.0, "n_renders_in_loop": 7.0}))
        self.assertIs(status_of(v, 1), CriterionStatus.FAIL)

    def test_1_missing_counters_are_not_run_rather_than_green(self):
        v = verdict(audit=audit(metrics={}))
        self.assertIs(status_of(v, 1), CriterionStatus.NOT_RUN)

    def test_1_a_zero_step_loop_does_not_pass_on_zero_equals_zero(self):
        v = verdict(audit=audit(metrics={"n_loop_steps": 0.0, "n_renders_in_loop": 0.0}))
        self.assertIs(status_of(v, 1), CriterionStatus.FAIL)

    def test_2_a_mesh_under_the_floor(self):
        """`> 0` is not the bar: a degenerate mesh gives an empty one's IR (ticket 12)."""
        ctx = AudioContextReport(n_vertices=9_999, ir_peak_abs=0.3, ir_shape=(2, 100),
                                 log_canary_seen=True)
        self.assertIs(status_of(verdict(audit=audit(audio_context=ctx)), 2),
                      CriterionStatus.FAIL)

    def test_2_a_canary_that_never_armed(self):
        """Ticket 16: unverified is not satisfied, and a clean log looks identical."""
        ctx = AudioContextReport(n_vertices=392364, ir_peak_abs=0.3, ir_shape=(2, 100),
                                 log_canary_seen=False)
        self.assertIs(status_of(verdict(audit=audit(audio_context=ctx)), 2),
                      CriterionStatus.FAIL)

    def test_2_no_guard_report_at_all(self):
        self.assertIs(status_of(verdict(audit=audit(audio_context=None)), 2),
                      CriterionStatus.NOT_RUN)

    def test_3_a_silent_ir(self):
        ctx = AudioContextReport(n_vertices=392364, ir_peak_abs=0.0, ir_shape=(2, 100),
                                 log_canary_seen=True)
        self.assertIs(status_of(verdict(audit=audit(audio_context=ctx)), 3),
                      CriterionStatus.FAIL)

    def test_3_a_monaural_ir(self):
        ctx = AudioContextReport(n_vertices=392364, ir_peak_abs=0.3, ir_shape=(1, 100),
                                 log_canary_seen=True)
        self.assertIs(status_of(verdict(audit=audit(audio_context=ctx)), 3),
                      CriterionStatus.FAIL)

    def test_4_provenance_that_never_ran(self):
        """The assertion raises, so its silence is not evidence it passed."""
        onset = OnsetRecord(onset_step=31, n_pre_onset_readings=31, provenance_asserted=False)
        self.assertIs(status_of(verdict(audit=audit(onset=onset)), 4), CriterionStatus.FAIL)

    def test_4_no_pre_onset_readings_leaves_the_first_invariant_unverified(self):
        onset = OnsetRecord(onset_step=31, n_pre_onset_readings=0, provenance_asserted=True)
        self.assertIs(status_of(verdict(audit=audit(onset=onset)), 4), CriterionStatus.FAIL)

    def test_4_zero_t_anom_has_no_pre_onset_steps_to_want(self):
        onset = OnsetRecord(onset_step=0, n_pre_onset_readings=0, provenance_asserted=True)
        v = judge(report=report(), audit=audit(onset=onset, t_anom=0), env_report=env(),
                  run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": 0})
        self.assertIs(status_of(v, 4), CriterionStatus.PASS)

    def test_4_an_unpinned_run_takes_the_bound_off_the_episode(self):
        """``t_anom`` is derived per episode, so an unpinned run's configuration does not
        know it and the record is the only place it exists.

        Without this the criterion goes quiet on exactly the runs the smoke performs: a
        ``None`` in the config used to skip the check, so an episode with no pre-onset
        reading at all would have passed §3.1's first invariant by omission.
        """
        onset = OnsetRecord(onset_step=31, n_pre_onset_readings=0, provenance_asserted=True)
        v = judge(report=report(), audit=audit(onset=onset, t_anom=16), env_report=env(),
                  run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": None})
        self.assertIs(status_of(v, 4), CriterionStatus.FAIL)

    def test_4_a_pin_the_episode_disagrees_with_is_a_build_that_did_not_obey(self):
        """Both numbers exist on a pinned run, and a mismatch means the episode did not
        get the onset step the run asked for — which nothing else would notice."""
        v = judge(report=report(), audit=audit(t_anom=9), env_report=env(),
                  run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": T_ANOM})
        self.assertIs(status_of(v, 4), CriterionStatus.FAIL)

    def test_4_a_bound_recorded_nowhere_is_not_run_rather_than_passed(self):
        """Neither the config nor the record has one, so the invariant has nothing to be
        checked against. Reporting PASS there is the shape ticket 16 keeps finding."""
        v = judge(report=report(), audit=audit(t_anom=None), env_report=env(),
                  run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": None})
        self.assertIs(status_of(v, 4), CriterionStatus.NOT_RUN)

    def test_5_an_aborted_detour_never_reached_check(self):
        """The over-credit ticket 26 found: stage 4 is not the full loop."""
        v = verdict(audit=audit(funnel_stage=FunnelStage.INVESTIGATE_ENTERED))
        self.assertIs(status_of(v, 5), CriterionStatus.FAIL)

    def test_5_reaching_the_source_without_resuming_is_not_the_full_loop(self):
        v = verdict(audit=audit(funnel_stage=FunnelStage.SOURCE_REACHED))
        self.assertIs(status_of(v, 5), CriterionStatus.FAIL)

    def test_6_a_missing_schema_key(self):
        short = report()
        del short["room"]
        self.assertIs(status_of(verdict(report=short), 6), CriterionStatus.FAIL)

    def test_6_a_key_outside_the_nine(self):
        """§5.1 is exactly nine fields; a tenth is the privilege leak ADR-0013 closed."""
        wide = report()
        wide["source_xyz"] = [1.0, 0.0, -2.0]
        self.assertIs(status_of(verdict(report=wide), 6), CriterionStatus.FAIL)

    def test_6_nulls_the_spec_permits_are_data_rather_than_gaps(self):
        """`anomaly_class` is null by design without CLAP; `visual_confirm_object` may be
        absent per §5.1. Neither is an unpopulated field."""
        self.assertIs(status_of(verdict(), 6), CriterionStatus.PASS)

    def test_7_a_step_over_the_ceiling(self):
        slow = steps(n=8)[:-1] + (
            StepRecord(7, 1e-3, action="move_forward", audio_render_s=CEILING_S + 0.01),
        )
        self.assertIs(status_of(verdict(audit=audit(steps=slow)), 7), CriterionStatus.FAIL)

    def test_7_an_unstated_ceiling_is_not_a_bound(self):
        v = judge(report=report(), audit=audit(), env_report=env(),
                  run_config={"t_anom": T_ANOM})
        self.assertIs(status_of(v, 7), CriterionStatus.NOT_RUN)

    def test_7_no_timings_at_all_is_not_run_rather_than_a_zero_that_passes(self):
        untimed = tuple(StepRecord(i, 1e-3, action="move_forward") for i in range(8))
        self.assertIs(status_of(verdict(audit=audit(steps=untimed)), 7),
                      CriterionStatus.NOT_RUN)

    def test_8_a_failed_probe(self):
        bad = env(probes=[{"name": "torch_cuda", "status": "fail"}])
        self.assertIs(status_of(verdict(env_report=bad), 8), CriterionStatus.FAIL)

    def test_8_a_probe_that_was_never_run_is_not_a_pass(self):
        bad = env(probes=[{"name": "torch_cuda", "status": "not_run"}])
        self.assertIs(status_of(verdict(env_report=bad), 8), CriterionStatus.FAIL)

    def test_8_a_probe_missing_by_absence(self):
        """Ticket 13's exact bug: a probe that stopped being emitted is red, not silent."""
        bad = env(missing=["habitat_audio_enum"])
        self.assertIs(status_of(verdict(env_report=bad), 8), CriterionStatus.FAIL)

    def test_8_an_empty_probe_list_is_not_run_rather_than_vacuously_green(self):
        self.assertIs(status_of(verdict(env_report=env(probes=[])), 8),
                      CriterionStatus.NOT_RUN)

    def test_8_no_env_report_at_all(self):
        self.assertIs(status_of(verdict(env_report=None), 8), CriterionStatus.NOT_RUN)


class TestTheDisclosuresRideWithTheVerdict(unittest.TestCase):
    """§8's required disclosure is not a thing to remember — the gate says it."""

    def test_the_oracle_stop_is_disclosed(self):
        self.assertTrue(any("ORACLE STOP" in n for n in verdict().notes))

    def test_running_the_wrong_arm_is_called_out(self):
        v = verdict(audit=audit(localization_arm="oracle"))
        self.assertTrue(any("REALIZABLE" in n for n in v.notes))

    def test_the_collision_tally_is_reported(self):
        """Ticket 26's measurement, in front of whoever reads the verdict."""
        walls = tuple(
            StepRecord(i, 1e-3, action="move_forward", audio_render_s=0.03,
                       collided=i % 2 == 0, displacement_m=0.0 if i % 2 == 0 else 0.25)
            for i in range(8)
        )
        v = verdict(audit=audit(steps=walls))
        self.assertTrue(any("forwards collided" in n for n in v.notes))


class TestItReadsWhatARunActuallyWrites(unittest.TestCase):
    """The seam ``judge()`` cannot check, and it had a real bug.

    ``judge()`` takes mappings, so every test above passes whatever shape it likes.
    ``judge_run_dir`` has to find the files and read the shape ``report/artifacts`` writes
    — and the first version read ``run_paths()[0]`` as if it were the env report *file*
    when it is the run *directory*, which would have reported criterion 8 as NOT_RUN on
    the box for a reason that has nothing to do with the environment.

    So this writes the artefacts with the real writers and reads them back with the real
    reader. A file that writes but does not parse fails only here.
    """

    def _run_dir(self):
        import tempfile

        from earshot.report.agent import AgentReport
        from earshot.report.artifacts import write_env_report, write_episode

        root = tempfile.mkdtemp()
        write_env_report(
            root,
            dict(
                probes=[{"name": "torch_cuda", "status": "pass"}],
                missing=[],
                environment={"python": "3.9.19"},
                run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": T_ANOM},
                scene="wcojb4TFT35",
            ),
        )
        write_episode(root, 0, AgentReport(resumed=True), audit())
        return root

    def test_the_env_report_is_found_and_its_probes_are_read(self):
        v = judge_run_dir(self._run_dir())
        self.assertIs(status_of(v, 8), CriterionStatus.PASS)

    def test_the_ceiling_comes_out_of_the_nested_run_config(self):
        """Criterion 7 NOT_RUN here would mean the bound was never found."""
        v = judge_run_dir(self._run_dir())
        self.assertIs(status_of(v, 7), CriterionStatus.PASS)

    def test_the_written_report_satisfies_the_schema_it_is_judged_against(self):
        """`AgentReport.as_dict` and `REPORT_KEYS` are two spellings of §5.1's nine."""
        v = judge_run_dir(self._run_dir())
        self.assertIs(status_of(v, 6), CriterionStatus.PASS)

    def test_the_audit_round_trips_through_the_file_without_losing_a_criterion(self):
        v = judge_run_dir(self._run_dir())
        self.assertEqual(v.failed, (9,))


class TestTheSummaryIsReadable(unittest.TestCase):
    def test_it_names_the_criteria_that_failed(self):
        text = verdict(audit=audit(funnel_stage=FunnelStage.RUN)).summary()
        self.assertIn("SMOKE RED", text)
        self.assertIn("5", text)

    def test_a_green_run_says_so(self):
        """Constructed by hand: no real run can be green while criterion 9 is outstanding."""
        from earshot.task.smoke import Criterion, SmokeVerdict

        all_pass = SmokeVerdict(
            criteria=tuple(
                Criterion(n, "c{}".format(n), CriterionStatus.PASS) for n in range(1, 10)
            )
        )
        self.assertTrue(all_pass.green)
        self.assertIn("SMOKE GREEN", all_pass.summary())


class TestTheTallyOverN(unittest.TestCase):
    """`tally` is pure, so *given n episodes of which some fail, does the run go red*
    is the same third-row question `judge` answers, one denominator up. It is the
    assertion a gate judging episode 0 could never make: at yield-1's 8/20 loop rate,
    judging index 0 is a coin flip printed as a verdict."""

    @staticmethod
    def _verdict(*statuses):
        """One episode's nine, from nine statuses."""
        from earshot.task.smoke import Criterion, SmokeVerdict

        return SmokeVerdict(criteria=tuple(
            Criterion(n, "c{}".format(n), s, "detail {}".format(n))
            for n, s in enumerate(statuses, start=1)))

    @classmethod
    def _all(cls, status):
        return cls._verdict(*([status] * 9))

    def test_every_episode_passing_is_the_only_green_for_a_non_rate_criterion(self):
        from earshot.task.smoke import tally

        run = tally([self._all(CriterionStatus.PASS)] * 20)
        self.assertTrue(run.green)
        self.assertEqual(run.n_episodes, 20)
        self.assertEqual([t.n_pass for t in run.tallies], [20] * 9)

    def test_one_failing_episode_reddens_a_criterion_that_must_hold_everywhere(self):
        """19/20 renders is a run whose audio dropped a step, not a 95% pass."""
        from earshot.task.smoke import tally

        bad = self._verdict(*([CriterionStatus.FAIL] + [CriterionStatus.PASS] * 8))
        run = tally([self._all(CriterionStatus.PASS)] * 19 + [bad])
        self.assertIn(1, run.failed)
        self.assertFalse(run.green)

    def test_criterion_5_is_a_rate_and_a_partial_loop_rate_is_not_red(self):
        """yield-1's ziup5kvtCCR: 8 of 20 resumed. That is a finding, not a broken gate."""
        from earshot.task.smoke import tally

        resumed = self._all(CriterionStatus.PASS)
        aborted = self._verdict(*([CriterionStatus.PASS] * 4 + [CriterionStatus.FAIL]
                                  + [CriterionStatus.PASS] * 4))
        run = tally([resumed] * 8 + [aborted] * 12)
        five = next(t for t in run.tallies if t.number == 5)
        self.assertEqual((five.n_pass, five.n), (8, 20))
        self.assertTrue(five.ok)
        self.assertTrue(run.green)
        self.assertIn("40%", five.line())

    def test_a_loop_that_never_once_ran_is_the_vacuous_arm_and_goes_red(self):
        from earshot.task.smoke import tally

        aborted = self._verdict(*([CriterionStatus.PASS] * 4 + [CriterionStatus.FAIL]
                                  + [CriterionStatus.PASS] * 4))
        run = tally([aborted] * 20)
        self.assertIn(5, run.failed)

    def test_not_run_is_counted_apart_from_fail_and_is_still_not_green(self):
        from earshot.task.smoke import tally

        run = tally([self._all(CriterionStatus.NOT_RUN)])
        self.assertFalse(run.green)
        self.assertEqual([t.n_not_run for t in run.tallies], [1] * 9)
        self.assertIn("NOT RUN", run.summary())

    def test_no_episodes_is_red_rather_than_vacuously_green(self):
        from earshot.task.smoke import tally

        self.assertFalse(tally([]).green)

    def test_the_failing_measurements_survive_into_the_tally(self):
        """ADR-0014: a verdict with the numbers thrown away is not decidable."""
        from earshot.task.smoke import tally

        bad = self._verdict(*([CriterionStatus.FAIL] + [CriterionStatus.PASS] * 8))
        run = tally([bad] * 20)
        one = next(t for t in run.tallies if t.number == 1)
        self.assertEqual(one.details, ("detail 1",) * 3)  # bounded, not all twenty
        self.assertIn("detail 1", one.line())

    def test_a_detail_names_the_episode_it_came_from(self):
        """detour-1 printed criterion 5 three times, identical and anonymous:
        "funnel stage 4 (INVESTIGATE_ENTERED)" with nothing saying which episode. That is
        a measurement you cannot go and read."""
        from earshot.task.smoke import tally

        bad = self._verdict(*([CriterionStatus.FAIL] + [CriterionStatus.PASS] * 8))
        good = self._all(CriterionStatus.PASS)
        run = tally([good, bad, good, bad], labels=[0, 1, 2, 3])
        one = next(t for t in run.tallies if t.number == 1)
        self.assertEqual(one.details, ("ep 1: detail 1", "ep 3: detail 1"))

    def test_unlabelled_verdicts_still_tally(self):
        """`labels` is optional, so a caller with no episode identity is not broken."""
        from earshot.task.smoke import tally

        bad = self._verdict(*([CriterionStatus.FAIL] + [CriterionStatus.PASS] * 8))
        one = next(t for t in tally([bad]).tallies if t.number == 1)
        self.assertEqual(one.details, ("detail 1",))

    def test_a_repeated_disclosure_collapses_but_is_not_dropped(self):
        from earshot.task.smoke import SmokeVerdict, tally

        one = SmokeVerdict(criteria=self._all(CriterionStatus.PASS).criteria,
                           notes=("oracle STOP",))
        run = tally([one] * 20)
        self.assertEqual(run.notes.count("oracle STOP"), 1)

    def test_per_episode_notes_are_bounded_and_say_how_many_were_dropped(self):
        from earshot.task.smoke import MAX_TALLY_NOTES, SmokeVerdict, tally

        run = tally([
            SmokeVerdict(criteria=self._all(CriterionStatus.PASS).criteria,
                         notes=("{} forwards collided".format(i),))
            for i in range(20)
        ])
        self.assertEqual(len(run.notes), MAX_TALLY_NOTES + 1)
        self.assertIn("further per-episode note(s) not shown", run.notes[-1])


class TestItFindsEveryEpisodeOnDisk(unittest.TestCase):
    """`episode_indices` reads filenames, not `summary.json`, because the summary is
    written last and the run worth judging is often the one that did not finish."""

    def test_it_returns_the_indices_in_order(self):
        import json
        import tempfile

        from earshot.report.artifacts import run_paths
        from earshot.task.smoke import episode_indices

        with tempfile.TemporaryDirectory() as tmp:
            _, episodes = run_paths(tmp)
            episodes.mkdir(parents=True)
            for i in (2, 0, 11):
                (episodes / "ep{:04d}.audit.json".format(i)).write_text(json.dumps({}))
            self.assertEqual(episode_indices(tmp), (0, 2, 11))

    def test_no_episodes_directory_is_an_empty_tuple_rather_than_a_crash(self):
        import tempfile

        from earshot.task.smoke import episode_indices

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(episode_indices(tmp), ())


if __name__ == "__main__":
    unittest.main(verbosity=2)
