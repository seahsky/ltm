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
        v = judge(report=report(), audit=audit(onset=onset), env_report=env(),
                  run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": 0})
        self.assertIs(status_of(v, 4), CriterionStatus.PASS)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
