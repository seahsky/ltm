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
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    SoundingWindowRecord,
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


# ADR-0017's window on a hand-built record, so criterion 4's new half can be bent one
# property at a time. The offset step is four steps after `t_anom` and the tail four
# steps after that, which is short enough to read and long enough that "ended inside the
# tail" is a distinct case from "the tail never decayed".
BED_RMS = 1e-3
OFFSET_STEP = T_ANOM + 4
# THE TWO TAILS, AND THEY ARE DELIBERATELY DIFFERENT NUMBERS. Before ADR-0019 one field
# carried both meanings, so a criterion reading the wrong one was unfalsifiable here.
# `TAIL_STEPS` is the CLIP readout emptying -- what `tail_is_active` reads and what the
# `missing` guard names -- and `CUE_TAIL_STEPS` is the room, which is what criterion 4's
# fence post is measured from. Keeping the cue at the old 4 leaves every fence-post number
# in the tests below (37 / 38) exactly where it was, so the split is visible as a changed
# FIELD rather than as a wave of changed constants.
TAIL_STEPS = 7
CUE_TAIL_STEPS = 4


def window_config(policy="fixed_steps", tol=0.05):
    return {
        "audio_step_ceiling_s": CEILING_S,
        "t_anom": T_ANOM,
        "sounding_policy": policy,
        "audio": {"pre_onset_rms_tol": tol},
    }


# A sentinel, because ``calibration=None`` is a case this helper has to be able to
# EXPRESS: the guard that refuses a window with no calibration record was unreachable
# from every fixture, and unreachable is how it returned PASS on an artefact it could not
# read for as long as it did.
_DEFAULT_CALIBRATION = object()


def windowed(
    *,
    offset_step=OFFSET_STEP,
    sounding_from=T_ANOM,
    sounding_to=OFFSET_STEP,
    policy="fixed_steps",
    n_steps=44,
    final_rms=BED_RMS,
    tail_steps=TAIL_STEPS,
    cue_tail_steps=CUE_TAIL_STEPS,
    calibration=_DEFAULT_CALIBRATION,
):
    """An audit carrying a window, whose per-step trace is bent by the two bounds.

    ``sounding_to=None`` is a source that never stopped -- the forced failure. Every row
    but the last reads the bed, because the level half of the criterion only ever asks
    about the final row and asking about the tail's shape would pin the accumulator's
    arithmetic in a gate that is about the task.

    ``calibration=None`` and ``tail_steps=None`` and ``n_steps=0`` are the three halves of
    the record ``run_episode`` writes TOGETHER, each removable here on its own -- which is
    what makes the guard that refuses such an audit exercisable at all.

    ``cue_tail_steps=None`` is a FOURTH and different case: not a broken record but an
    audit written before ADR-0019 split the readout. It is not in the guard, deliberately
    -- every audit already on disk would go red -- and criterion 4 declines to assert the
    level on it instead.
    """
    rows = tuple(
        StepRecord(
            i,
            BED_RMS if i + 1 < n_steps else final_rms,
            source_playing=(
                i >= sounding_from and (sounding_to is None or i < sounding_to)
            ),
            action="move_forward",
            audio_render_s=0.03,
            collided=False,
            displacement_m=0.25,
        )
        for i in range(n_steps)
    )
    if calibration is _DEFAULT_CALIBRATION:
        calibration = CalibrationRecord(
            onset_rms=3e-3, bed_rms=BED_RMS, separation_db=18.0, n_poses=16,
            global_volume=1.0,
        )
    return audit(
        t_anom=T_ANOM,
        steps=rows,
        calibration=calibration,
        sounding_window=SoundingWindowRecord(
            opens_at=T_ANOM,
            offset_step=offset_step,
            policy=policy,
            step_seconds=1.0,
            hop_samples=44100,
            analysis_window_samples=220500,
            max_ir_samples=72300,
            n_buffer_grows=0,
            tail_steps=tail_steps,
            cue_tail_steps=cue_tail_steps,
        ),
        metrics={"n_loop_steps": float(n_steps), "n_renders_in_loop": float(n_steps)},
    )


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

    def test_4_a_record_with_no_window_is_judged_exactly_as_before(self):
        """THE BACKWARDS-COMPATIBILITY ARM. Every audit on disk predates ADR-0017.

        A criterion that changed its answer on an unchanged record is a criterion nobody
        can compare across runs — which is also why the window went inside criterion 4
        rather than becoming a tenth that would be NOT_RUN, and therefore red, on all of
        them.
        """
        self.assertIsNone(audit().sounding_window)
        self.assertIs(status_of(verdict(), 4), CriterionStatus.PASS)
        # and the run's policy does not change that: there is no record to contradict
        v = judge(report=report(), audit=audit(), env_report=env(),
                  run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": T_ANOM,
                              "sounding_policy": "fixed_steps"})
        self.assertIs(status_of(v, 4), CriterionStatus.PASS)

    def test_4_a_source_that_kept_sounding_past_the_offset_step_fails(self):
        """THE FORCED-FAILURE ARM, and this is the shape of the ``anommxv`` break.

        The loop ran to completion with the audio doing something other than what the
        task specified, and every artefact looked healthy. Criterion 4 was structurally
        blind to it: nothing in ``observe_step`` or ``assert_provenance`` says anything
        about the offset step, so a source that failed to stop produced a green gate.

        The check reads the PER-STEP TRACE rather than the config echo, because a config
        that says "stop at 34" and a trace that never stopped is precisely the
        disagreement being looked for.
        """
        v = judge(report=report(), audit=windowed(sounding_to=None), env_report=env(),
                  run_config=window_config())
        self.assertIs(status_of(v, 4), CriterionStatus.FAIL)
        detail = next(c.detail for c in v.criteria if c.number == 4)
        self.assertIn("34", detail, "the failing step has to be named")

    def test_4_a_source_silent_inside_its_own_window_fails(self):
        """The mirror: a window that closed before it opened, or never opened at all."""
        v = judge(report=report(), audit=windowed(sounding_from=33), env_report=env(),
                  run_config=window_config())
        self.assertIs(status_of(v, 4), CriterionStatus.FAIL)

    def test_4_a_window_the_run_did_not_ask_for_is_a_build_that_did_not_obey(self):
        """Same argument the ``t_anom`` pin won: the run asked for one thing and the
        episode recorded another, and nothing else in the tree would notice."""
        v = judge(report=report(), audit=windowed(), env_report=env(),
                  run_config=window_config(policy="drawn"))
        self.assertIs(status_of(v, 4), CriterionStatus.FAIL)
        detail = next(c.detail for c in v.criteria if c.number == 4)
        self.assertIn("drawn", detail)
        self.assertIn("fixed_steps", detail)

    def test_4_a_continuous_arm_run_is_not_failed_for_having_no_offset_step(self):
        """``WindowPolicy.CONTINUOUS`` is the control arm, not a broken build.

        There is nothing to check, and saying so in the detail is not the same as
        passing vacuously.
        """
        v = judge(report=report(),
                  audit=windowed(offset_step=None, sounding_to=None, policy="continuous"),
                  env_report=env(), run_config=window_config(policy="continuous"))
        self.assertIs(status_of(v, 4), CriterionStatus.PASS)
        detail = next(c.detail for c in v.criteria if c.number == 4)
        self.assertIn("continuous arm", detail)

    def test_4_the_signal_reaches_the_bed_once_the_tail_has_run_out(self):
        """The healthy arm of the half that has to WAIT.

        The naive symmetric mirror of §3.1 — "after the offset step the RMS is the bed" —
        is FALSE for exactly the steps the reverb tail exists for. Waiting
        ``cue_tail_steps`` is what makes it true, and it comes off the record rather than
        off a constant because no module in this tree caps the IR's width.
        """
        v = judge(report=report(), audit=windowed(), env_report=env(),
                  run_config=window_config())
        self.assertIs(status_of(v, 4), CriterionStatus.PASS)
        detail = next(c.detail for c in v.criteria if c.number == 4)
        self.assertIn("decayed to the bed", detail)
        self.assertIn(
            "{} steps after the last sounding step".format(CUE_TAIL_STEPS),
            detail,
            "the detail names the CLIP tail, so an operator reading it would price the "
            "silent phase against the analysis window rather than against the room",
        )

    def test_4_the_fence_post_is_the_CUE_tail_and_not_the_clip_one(self):
        """THE HONESTY PROPERTY THE SPLIT ADDS, and it needs two different numbers.

        The fixture's clip tail is 7 and its cue tail is 4, which is what makes the two
        expressions separable at all: before ADR-0019 they were one field and a criterion
        reading the wrong one could not be caught. Reading ``tail_steps`` would put the
        fence post at ``34 + 7 - 1 = 40``, so an episode ending at step 37 would be
        skipped as "still inside its tail" -- and every one of those four steps was
        assertable evidence about the room being quiet.
        """
        on_the_cue_bound = judge(
            report=report(), audit=windowed(n_steps=38, final_rms=9e-3),
            env_report=env(), run_config=window_config())
        self.assertIs(
            status_of(on_the_cue_bound, 4),
            CriterionStatus.FAIL,
            "step 37 is past the CUE tail's fence post, so the loud final reading is "
            "assertable; reading tail_steps would have skipped it",
        )
        # ...and the clip tail's own fence post, 40, is where it would have landed. The
        # episode reaching that step is red too, so the test above is about WHICH bound
        # fired rather than about the level being unassertable at all.
        past_the_clip_bound = judge(
            report=report(), audit=windowed(n_steps=42, final_rms=9e-3),
            env_report=env(), run_config=window_config())
        self.assertIs(status_of(past_the_clip_bound, 4), CriterionStatus.FAIL)

    def test_4_a_record_written_before_the_split_readout_is_not_asserted_at_all(self):
        """PASS with a reason, and the trace half still binding. Both arms in one test.

        An audit with a window and no ``cue_tail_steps`` predates ADR-0019, so its RMS
        trace is the 5 s CLIP readout. Judging it at the cue's fence post can only go
        wrongly red -- the clip readout is still full of source where the cue has emptied
        -- so the level is not asserted. What must NOT lapse with it is the trace check:
        a source that failed to stop is still red on such a record.
        """
        loud = judge(report=report(),
                     audit=windowed(cue_tail_steps=None, final_rms=9e-3),
                     env_report=env(), run_config=window_config())
        self.assertIs(status_of(loud, 4), CriterionStatus.PASS)
        detail = next(c.detail for c in loud.criteria if c.number == 4)
        self.assertIn("predates the split readout", detail)
        never_stopped = judge(
            report=report(),
            audit=windowed(cue_tail_steps=None, sounding_to=None),
            env_report=env(), run_config=window_config())
        self.assertIs(
            status_of(never_stopped, 4),
            CriterionStatus.FAIL,
            "the trace half lapsed with the level half, so a source that never stopped "
            "would pass on every audit written before the split",
        )

    def test_4_a_tail_that_never_decayed_fails(self):
        """The forced failure of the same half: still loud long after the tail ran out."""
        v = judge(report=report(), audit=windowed(final_rms=9e-3), env_report=env(),
                  run_config=window_config())
        self.assertIs(status_of(v, 4), CriterionStatus.FAIL)

    def test_4_an_episode_that_ended_inside_its_tail_says_so_rather_than_asserting(self):
        """Not a NOT_RUN: the criterion IS evaluated — the trace check always runs — and
        this is an extra assertion whose premise the episode simply does not meet."""
        v = judge(report=report(), audit=windowed(n_steps=36, final_rms=9e-3),
                  env_report=env(), run_config=window_config())
        self.assertIs(status_of(v, 4), CriterionStatus.PASS)
        detail = next(c.detail for c in v.criteria if c.number == 4)
        self.assertIn("before the", detail)

    def test_4_the_episode_that_ends_EXACTLY_on_the_bound_is_asserted(self):
        """THE FENCE POST, and both sides of it, because neither fixture above touches it.

        ``cue_tail_steps`` counts from the LAST SOUNDING step, which is
        ``offset_step - 1``, so the reading is first exactly the bed at
        ``offset_step + cue_tail_steps - 1`` -- here 34 + 4 - 1 = 37. Reading it from the
        offset step instead over-states the room's post-offset lifetime by one and throws
        away a step of assertable evidence on every episode.

        The bug is INVISIBLE at the lengths the other tests use: at 44 steps both
        expressions assert and at 36 both skip. Only an episode whose last step IS 37
        separates them, and its neighbour at 36 is what stops this from being satisfied by
        a gate that asserts unconditionally.
        """
        on_the_bound = judge(
            report=report(), audit=windowed(n_steps=38, final_rms=9e-3),
            env_report=env(), run_config=window_config())
        self.assertIs(
            status_of(on_the_bound, 4),
            CriterionStatus.FAIL,
            "the tail had run out by the last step, so the level is assertable and loud",
        )
        one_short = judge(
            report=report(), audit=windowed(n_steps=37, final_rms=9e-3),
            env_report=env(), run_config=window_config())
        self.assertIs(status_of(one_short, 4), CriterionStatus.PASS)
        self.assertIn(
            "before the", next(c.detail for c in one_short.criteria if c.number == 4)
        )

    def test_4_no_tolerance_configured_leaves_the_level_unasserted_and_says_so(self):
        v = judge(report=report(), audit=windowed(final_rms=9e-3), env_report=env(),
                  run_config={"audio_step_ceiling_s": CEILING_S, "t_anom": T_ANOM,
                              "sounding_policy": "fixed_steps"})
        self.assertIs(status_of(v, 4), CriterionStatus.PASS)
        detail = next(c.detail for c in v.criteria if c.number == 4)
        self.assertIn("pre_onset_rms_tol", detail)

    def test_4_a_window_with_no_calibration_record_is_RED_and_does_not_traceback(self):
        """THE GUARD NO FIXTURE COULD REACH, and it had two failure modes at once.

        ``run_episode`` writes ``tail_steps``, a ``CalibrationRecord`` and at least one
        ``StepRecord`` on every episode that carries a window at all -- one constructor,
        one call -- so a window WITHOUT them is a record no build of this tree can
        produce: hand-edited, truncated mid-write, or spliced from two runs. It returned
        PASS, which is the gate reporting green on an artefact it could not read; and
        being unreachable it was also the only thing standing between a missing
        ``bed_rms`` and a ``float(None)`` TypeError seven lines down, which turns a gate
        run into a traceback rather than a verdict.

        All three halves, one at a time, because the guard names each of them. It is
        still ``tail_steps`` the guard names, not ``cue_tail_steps``: the guard asks
        whether the record is WHOLE, and every record this tree ever wrote carries the
        clip tail while only records written after ADR-0019 carry the cue one.
        """
        cases = {
            "a calibration record": windowed(calibration=None),
            "tail_steps": windowed(tail_steps=None),
            "any step records": windowed(n_steps=0),
        }
        for name, bent in cases.items():
            v = judge(report=report(), audit=bent, env_report=env(),
                      run_config=window_config())
            self.assertIs(status_of(v, 4), CriterionStatus.FAIL, name)
            detail = next(c.detail for c in v.criteria if c.number == 4)
            self.assertIn(name, detail)
            self.assertIn("did not come from one run", detail)
        # ...and the whole record present is the healthy arm, so the refusal above is
        # not a branch that always refuses.
        self.assertIs(
            status_of(
                judge(report=report(), audit=windowed(), env_report=env(),
                      run_config=window_config()),
                4,
            ),
            CriterionStatus.PASS,
        )

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
