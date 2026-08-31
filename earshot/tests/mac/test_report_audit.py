"""``EpisodeAudit`` — the funnel's ordering, the two projections, the derived series.

The projections are the interesting half. ADR-0013's layer graph gives ``report`` only
``report``, ``audio.guard`` and ``types``, so ``CalibrationRecord`` and ``OnsetRecord``
cannot *be* the audio types — they mirror them. A mirror drifts, and the drift is
silent: a rename in ``audio/calibration.py`` leaves an audit record quietly carrying a
field that no longer means what its name says.

A **test** sits outside the layer graph, so it can import both sides and pin the
projection. Same mechanism ticket 23 used to compare ``agent/occupancy``'s frame with
``audio/lateral``'s across a boundary a module may not cross, and ticket 20 used to read
``sim/world.py``'s sensor defaults out of its ``ast``.
"""

from __future__ import annotations

import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.audio.calibration import CalibrationResult
from earshot.audio.config import WindowPolicy
from earshot.audio.guard import AudioContextReport
from earshot.audio.onset import OnsetState
from earshot.audio.window import SoundingWindow
from earshot.report.audit import (
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
    SoundingWindowRecord,
    StepRecord,
)
from earshot.types import Xyz


class TestTheProjectionsCannotDriftFromTheAudioTypes(unittest.TestCase):
    """Every projected name must still exist on the type it projects."""

    def test_calibration_record_is_a_projection_of_calibration_result(self):
        missing = set(CalibrationRecord.__dataclass_fields__) - set(
            CalibrationResult.__dataclass_fields__
        )
        self.assertEqual(
            missing,
            set(),
            "CalibrationRecord names {} which CalibrationResult does not have — either "
            "audio/calibration.py renamed a field or the audit invented one".format(missing),
        )

    def test_onset_record_projects_onset_state_plus_its_own_evidence_field(self):
        """``provenance_asserted`` is the audit's own; everything else must come from
        ``OnsetState``.

        The extra field is not an exception being carved out — it is ticket 16's
        discipline. §3.1's assertions raise, so a record that exists looks like proof
        they passed unless they were never called, and only the *audit* can say whether
        ``assert_provenance`` actually ran.
        """
        projected = set(OnsetRecord.__dataclass_fields__) - {"provenance_asserted"}
        missing = projected - set(OnsetState.__dataclass_fields__)
        self.assertEqual(missing, set(), "OnsetRecord names {} that OnsetState lacks".format(missing))

    def test_the_sounding_window_record_projects_the_window_it_names(self):
        """ADR-0013 forbids ``report`` naming ``audio.window``, so this test does it.

        ``SoundingWindowRecord`` is a primitive-typed mirror of
        ``audio.window.SoundingWindow`` plus the accumulator's own measurements, and the
        same drift the ``CalibrationRecord`` check above catches applies: a rename in
        ``audio/window.py`` leaves the audit carrying a field that no longer means what
        its name says. The extra fields are the tail's, and they are named so the
        exception is a list rather than a hole.
        """
        accumulator_fields = {
            "step_seconds",
            "hop_samples",
            "analysis_window_samples",
            "max_ir_samples",
            "n_buffer_grows",
            "tail_steps",
            # The CLIP tail's sibling, `ceil((hop + L - 1)/hop)` -- what the agent's own
            # one-step-wide reading takes to empty. Same reason it is not on
            # `SoundingWindow`: it needs the hop and the IR's width, and the window's
            # boundaries depend on neither.
            "cue_tail_steps",
            # `ceil(N / hop)`, so it needs the hop and the clip length -- neither of
            # which `plan_window` is given. Putting it on `SoundingWindow` instead was
            # rejected for that reason: it would change that function's signature to
            # carry two numbers the window's boundaries do not depend on.
            "ramp_steps",
            "post_offset_audible_steps",
        }
        projected = set(SoundingWindowRecord.__dataclass_fields__) - accumulator_fields
        missing = projected - set(SoundingWindow.__dataclass_fields__)
        self.assertEqual(
            missing,
            set(),
            "SoundingWindowRecord names {} which SoundingWindow does not have — either "
            "audio/window.py renamed a field or the audit invented one".format(missing),
        )

    def test_the_projections_carry_what_section_5_2_names(self):
        """§5.2: "the calibration separation margin and the threshold in force"."""
        self.assertIn("separation_db", CalibrationRecord.__dataclass_fields__)
        self.assertIn("onset_rms", CalibrationRecord.__dataclass_fields__)
        self.assertIn("pre_onset_rms", OnsetRecord.__dataclass_fields__)


class TestTheFunnelIsOrdinal(unittest.TestCase):
    def test_the_six_stages_are_section_6s_six(self):
        self.assertEqual(
            [stage.name for stage in FunnelStage],
            [
                "RUN",
                "T_ANOM_REACHED",
                "ONSET_FIRED",
                "INVESTIGATE_ENTERED",
                "SOURCE_REACHED",
                "PRIMARY_RESUMED",
            ],
        )

    def test_reaching_a_later_stage_implies_the_earlier_ones(self):
        """The whole reason it is an ``IntEnum``: per-stage counts are comparisons.

        §6's denominator is stage 2, so an aggregator asks ``stage >= T_ANOM_REACHED``.
        With a plain ``Enum`` that ordering would live in whichever module happened to
        count, in a metric the spec singles out because an aggregate has hidden the
        mechanism on this project before.
        """
        self.assertGreater(FunnelStage.PRIMARY_RESUMED, FunnelStage.SOURCE_REACHED)
        self.assertGreaterEqual(FunnelStage.ONSET_FIRED, FunnelStage.T_ANOM_REACHED)
        reached = FunnelStage.SOURCE_REACHED
        self.assertTrue(reached >= FunnelStage.T_ANOM_REACHED)
        self.assertFalse(reached >= FunnelStage.PRIMARY_RESUMED)


def _steps():
    return (
        StepRecord(0, 1e-3, lateral_sign=0, source_playing=False, source_is_visible=False,
                   action="move_forward", audio_render_s=0.030,
                   collided=False, displacement_m=0.25),
        StepRecord(1, 2e-3, lateral_sign=1, source_playing=True, source_is_visible=True,
                   action="turn_left", audio_render_s=0.026,
                   collided=False, displacement_m=0.0),
        StepRecord(2, 3e-3, lateral_sign=1, source_playing=True, source_is_visible=None,
                   action="stop", audio_render_s=0.028),
    )


class TestTheDerivedSeries(unittest.TestCase):
    def test_source_is_visible_history_comes_from_the_step_rows(self):
        """§5.2's history, derived so it cannot disagree with §3.2's per-step record."""
        audit = EpisodeAudit(steps=_steps())
        self.assertEqual(audit.source_is_visible_history, (False, True, None))

    def test_render_summary_is_empty_rather_than_zero_when_nothing_rendered(self):
        """A ceiling check against a fabricated 0.0 would pass criterion 7 on an
        episode whose audio never rendered at all."""
        audit = EpisodeAudit(steps=(StepRecord(0, 1e-3),))
        self.assertEqual(audit.audio_render_summary(), {})
        self.assertEqual(audit.n_render_steps, 0)

    def test_render_summary_measures_what_criterion_7_bounds(self):
        audit = EpisodeAudit(steps=_steps())
        summary = audit.audio_render_summary()
        self.assertEqual(summary["n"], 3.0)
        self.assertAlmostEqual(summary["max_s"], 0.030)
        self.assertAlmostEqual(summary["min_s"], 0.026)
        self.assertAlmostEqual(summary["total_s"], 0.084)
        self.assertEqual(audit.n_render_steps, 3)

    def test_distance_to_source_is_derived_from_the_position_and_the_source(self):
        """The pairing that made `StepRecord.position` privileged, and the number
        yield-1 could not produce: was the abandoned detour getting closer?"""
        audit = EpisodeAudit(
            source_xyz=Xyz(0.0, 0.0, 0.0),
            steps=(StepRecord(0, 1e-3, position=Xyz(6.0, 0.0, 0.0)),
                   StepRecord(1, 1e-3, position=Xyz(0.0, 0.0, 3.0)),
                   StepRecord(2, 1e-3, position=Xyz(3.0, 9.9, 4.0))),
        )
        # Horizontal: the third row is 9.9 m up and still 5 m away, because a source a
        # storey above is ADR-0003's fabricated audio, not a source 11 m distant.
        self.assertEqual(audit.distance_to_source_history, (6.0, 3.0, 5.0))

    def test_a_step_without_a_position_reads_none_rather_than_a_distance(self):
        """Every record written before the field existed. Substituting a number there is
        how an un-measured detour would come to look like a converging one."""
        audit = EpisodeAudit(source_xyz=Xyz(0.0, 0.0, 0.0), steps=_steps())
        self.assertEqual(audit.distance_to_source_history, (None, None, None))

    def test_an_episode_with_no_source_has_no_distances(self):
        audit = EpisodeAudit(steps=(StepRecord(0, 1e-3, position=Xyz(6.0, 0.0, 0.0)),))
        self.assertEqual(audit.distance_to_source_history, (None,))

    def test_a_step_with_no_render_is_not_counted_as_rendered(self):
        """Smoke criterion 1 is "render count equals step count exactly", so a step
        whose audio did not render has to be visibly missing rather than assumed."""
        audit = EpisodeAudit(steps=_steps() + (StepRecord(3, 1e-3, audio_render_s=None),))
        self.assertEqual(len(audit.steps), 4)
        self.assertEqual(audit.n_render_steps, 3)


class TestTheForwardSummary(unittest.TestCase):
    """Ticket 26's addition to §3.2: what separates a forward that moved from a wall.

    The first box episode reported 110 forwards for 6.57 m of path and no way to tell
    which of those forwards were walls, because ``World.step`` returns habitat's
    collision flag and the runner discarded it. That one number decides whether the
    climb needs obstacle awareness or a bigger budget, so it belongs in the record
    rather than in an inference from path length.
    """

    def _forwards(self, *pairs):
        return EpisodeAudit(
            steps=tuple(
                StepRecord(i, 1e-3, action="move_forward", collided=c, displacement_m=d)
                for i, (c, d) in enumerate(pairs)
            )
        )

    def test_it_counts_walls_against_forwards(self):
        audit = self._forwards((False, 0.25), (True, 0.0), (True, 0.02), (False, 0.25))
        summary = audit.forward_summary()
        self.assertEqual(summary["n_forward"], 4)
        self.assertEqual(summary["n_collided"], 2)
        self.assertAlmostEqual(summary["total_displacement_m"], 0.52)

    def test_turns_are_not_forwards(self):
        """A turn displaces nothing by design, so averaging it in would hide the walls."""
        audit = EpisodeAudit(
            steps=(
                StepRecord(0, 1e-3, action="move_forward", collided=False, displacement_m=0.25),
                StepRecord(1, 1e-3, action="turn_left", collided=False, displacement_m=0.0),
                StepRecord(2, 1e-3, action="stop"),
            )
        )
        self.assertEqual(audit.forward_summary()["n_forward"], 1)

    def test_an_episode_with_no_forwards_is_empty_rather_than_zero(self):
        """The same rule ``audio_render_summary`` follows: absent is not 0.0."""
        audit = EpisodeAudit(steps=(StepRecord(0, 1e-3, action="turn_left"),))
        self.assertEqual(audit.forward_summary(), {})

    def test_the_fields_are_optional_so_an_older_record_still_reads(self):
        audit = EpisodeAudit(steps=(StepRecord(0, 1e-3, action="move_forward"),))
        self.assertEqual(audit.forward_summary()["n_forward"], 1)
        self.assertEqual(audit.forward_summary()["n_collided"], 0)


class TestTheAuditRoundTrips(unittest.TestCase):
    def _audit(self):
        return EpisodeAudit(
            episode_index=7,
            scene_id="wcojb4TFT35",
            localization_arm="realizable",
            detector_arm="oracle",
            source_xyz=Xyz(1.5, 0.1, -2.25),
            t_anom=16,
            dist_at_stop=0.83,
            funnel_stage=FunnelStage.PRIMARY_RESUMED,
            onset=OnsetRecord(onset_step=130, pre_onset_rms=1e-3,
                              n_pre_onset_readings=130, provenance_asserted=True),
            calibration=CalibrationRecord(onset_rms=0.065, bed_rms=1e-3, separation_db=11.2,
                                          n_poses=16, global_volume=1.0),
            audio_context=AudioContextReport(n_vertices=392364, ir_peak_abs=0.37,
                                             ir_shape=(2, 72300), log_canary_seen=True),
            steps=_steps(),
            metrics={"soft_spl": 0.375, "benchmark_spl": 0.0},
        )

    def test_every_field_survives(self):
        original = self._audit()
        self.assertEqual(EpisodeAudit.from_dict(original.as_dict()), original)

    def test_a_position_survives_the_json_as_a_position(self):
        """`_steps()` deliberately carries none — it stands for a pre-yield-1 record —
        so the new field needs its own round trip or it round-trips only as absent."""
        original = EpisodeAudit(
            source_xyz=Xyz(1.0, 0.0, 2.0),
            steps=(StepRecord(0, 1e-3, position=Xyz(4.0, 0.25, -1.5)),
                   StepRecord(1, 1e-3, position=None)),
        )
        restored = EpisodeAudit.from_dict(original.as_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.steps[0].position, Xyz(4.0, 0.25, -1.5))
        self.assertIsNone(restored.steps[1].position,
                          "an absent position must not round-trip into the origin")
        self.assertEqual(restored.distance_to_source_history,
                         original.distance_to_source_history)

    def test_the_derived_keys_are_re_derived_rather_than_trusted(self):
        """A hand-edited history that disagrees with its own step rows must not survive.

        Serialising the derived series is for the analyst reading the JSON; reading it
        back would let the file carry two answers to one question.
        """
        data = self._audit().as_dict()
        data["source_is_visible_history"] = [True, True, True]
        data["audio_render_summary"] = {"max_s": 99.0}
        restored = EpisodeAudit.from_dict(data)
        self.assertEqual(restored.source_is_visible_history, (False, True, None))
        self.assertAlmostEqual(restored.audio_render_summary()["max_s"], 0.030)

    def test_the_nested_audio_context_survives(self):
        restored = EpisodeAudit.from_dict(self._audit().as_dict())
        self.assertIsNotNone(restored.audio_context)
        self.assertEqual(restored.audio_context.n_vertices, 392364)
        self.assertEqual(tuple(restored.audio_context.ir_shape), (2, 72300))
        self.assertTrue(restored.audio_context.log_canary_seen)


class TestTheSoundingWindowOnTheRecord(unittest.TestCase):
    """ADR-0017's two new fields, and why each is on the answer key at all.

    The offset step is nowhere else on disk — the per-step ``source_playing`` trace shows
    WHEN the source stopped, never what the task asked for, and a source that failed to
    stop leaves a trace that agrees with itself. The source-reach step was recoverable
    from NOTHING: the primary STOP is ``len(steps) - 1``, but
    ``InvestigationEvent.investigate_steps`` is a relative count and the ORACLE arm
    leaves no ``realizable_action`` trail, so SWS could not have been computed from any
    artefact this tree wrote before these two fields existed.
    """

    WINDOW = SoundingWindowRecord(
        opens_at=30,
        offset_step=90,
        policy=WindowPolicy.FIXED_STEPS.value,
        step_seconds=1.0,
        hop_samples=44100,
        analysis_window_samples=220500,
        max_ir_samples=72300,
        n_buffer_grows=1,
        tail_steps=7,
        # THE TWO TAILS, and they are different numbers at the box's own configuration:
        # `ceil((220500 + 72299)/44100)` is 7 and `ceil((44100 + 72299)/44100)` is 3. The
        # clip tail is the analysis window emptying; the cue tail is the room.
        cue_tail_steps=3,
    )

    def test_the_window_and_the_reach_step_round_trip_through_the_audit(self):
        original = EpisodeAudit(
            t_anom=30,
            sounding_window=self.WINDOW,
            source_reached_step=141,
            steps=_steps(),
        )
        restored = EpisodeAudit.from_dict(original.as_dict())
        self.assertEqual(restored, original)
        self.assertEqual(restored.sounding_window.offset_step, 90)
        self.assertEqual(restored.sounding_window.tail_steps, 7)
        self.assertEqual(restored.sounding_window.cue_tail_steps, 3)
        self.assertEqual(restored.source_reached_step, 141)

    def test_the_record_alone_round_trips(self):
        self.assertEqual(
            SoundingWindowRecord.from_dict(self.WINDOW.as_dict()), self.WINDOW
        )

    def test_a_record_written_before_the_split_readout_carries_no_cue_tail(self):
        """``None``, and it must never resolve to 1 -- which is a claim about the ROOM.

        ``cue_tail_steps`` of 1 says the IR fits inside one simulator step, i.e. the
        silent phase is an honest hard cut. A default that produced 1 for a record that
        never measured it would state exactly the thing this field exists to make
        checkable, on an artefact that says nothing about it. Smoke criterion 4 reads the
        None and declines to assert the level rather than judging a clip-domain trace at a
        cue-domain fence post.
        """
        payload = self.WINDOW.as_dict()
        del payload["cue_tail_steps"]
        restored = SoundingWindowRecord.from_dict(payload)
        self.assertIsNone(restored.cue_tail_steps)
        # ...and the clip tail beside it is untouched, which is what makes the two
        # separable on disk rather than one field wearing two meanings.
        self.assertEqual(restored.tail_steps, 7)

    def test_a_continuous_arm_window_keeps_its_null_offset_step(self):
        """``None`` is the CONTINUOUS arm and it must never become a step index."""
        window = SoundingWindowRecord(opens_at=30, offset_step=None, policy="continuous")
        restored = SoundingWindowRecord.from_dict(window.as_dict())
        self.assertIsNone(restored.offset_step)
        self.assertEqual(restored.opens_at, 30)

    def test_an_audit_written_before_the_window_existed_still_loads(self):
        """Absent means UNKNOWN, never step 0 and never "the source never stopped".

        Every audit on disk predates ADR-0017. A default that resolved to a number would
        make ``tail_is_active`` answer True for a run that had no accumulator, which is
        the one question that gates whether an SWS may be computed at all.
        """
        payload = EpisodeAudit(t_anom=30, steps=_steps()).as_dict()
        del payload["sounding_window"]
        del payload["source_reached_step"]
        restored = EpisodeAudit.from_dict(payload)
        self.assertIsNone(restored.sounding_window)
        self.assertIsNone(restored.source_reached_step)


class TestTheThreeScatterArmsOnTheCalibrationRecord(unittest.TestCase):
    """ADR-0019's rename, and the legacy key that means two different things.

    ``climb_eps``' input changed measurement domain twice under one field name. Before
    ADR-0017 ``render_scatter`` was independent whole-clip renders at a held pose; after
    it, successive CLIP readouts of the accumulator, measured 1.91x apart; after ADR-0019
    the climb reads the CUE readout, which is a third domain again. The field's WRITTEN
    definition -- "the spread of the reading the climb compares" -- stayed true through
    all three, which is exactly what would have let the domain move silently.

    So there are three named arms and the legacy key is not emitted at all. A record
    carrying both ``render_scatter`` and ``cue_render_scatter`` would let a reader pick
    the wrong one, and picking wrong is a mispriced epsilon rather than an error.
    """

    FULL = CalibrationRecord(
        onset_rms=3e-3,
        bed_rms=1e-3,
        separation_db=18.0,
        n_poses=16,
        global_volume=1.0,
        cue_render_scatter=2.7e-4,
        cue_scatter_repeats=12,
        clip_render_scatter=1.83e-4,
        clip_scatter_repeats=12,
        single_render_scatter=3.49e-4,
        single_render_repeats=12,
        cue_phase_folds=5,
        cue_phase_crest=2.2361,
        cue_phase_min_ratio=0.0,
        cue_phase_aggregation="quadratic_mean_over_loop_phases",
    )

    def test_the_three_arms_and_the_phase_block_round_trip(self):
        self.assertEqual(CalibrationRecord.from_dict(self.FULL.as_dict()), self.FULL)

    def test_the_legacy_key_is_never_emitted(self):
        """Both halves of the rename: the new keys are written and the old one is not."""
        payload = self.FULL.as_dict()
        self.assertNotIn("render_scatter", payload)
        self.assertNotIn("scatter_repeats", payload)
        self.assertEqual(payload["cue_render_scatter"], 2.7e-4)
        self.assertEqual(payload["clip_render_scatter"], 1.83e-4)
        self.assertEqual(payload["single_render_scatter"], 3.49e-4)

    def test_a_post_adr_0017_record_maps_its_legacy_key_onto_the_CLIP_arm(self):
        """The disambiguator is ``single_render_scatter``'s presence, and this is the era
        where it is there: ADR-0017 added that field in the same commit that made
        ``render_scatter`` the clip-loop estimate, so a record carrying both is a
        clip-loop number under the old name."""
        restored = CalibrationRecord.from_dict({
            "onset_rms": 3e-3, "bed_rms": 1e-3, "separation_db": 18.0,
            "n_poses": 16, "global_volume": 1.0,
            "render_scatter": 1.83e-4, "scatter_repeats": 12,
            "single_render_scatter": 3.49e-4, "single_render_repeats": 12,
        })
        self.assertEqual(restored.clip_render_scatter, 1.83e-4)
        self.assertEqual(restored.clip_scatter_repeats, 12)
        self.assertEqual(restored.single_render_scatter, 3.49e-4)
        self.assertIsNone(
            restored.cue_render_scatter,
            "an ADR-0017 record has no cue-domain number, and inventing one would replay "
            "the climb at a threshold no controller ever ran",
        )

    def test_a_pre_adr_0017_record_maps_its_legacy_key_onto_the_SINGLE_arm(self):
        """The other era, and the same key: ``detour-2`` and ``eps-1`` wrote whole-clip
        renders under ``render_scatter`` and had no ``single_render_scatter`` beside it."""
        restored = CalibrationRecord.from_dict({
            "onset_rms": 3e-3, "bed_rms": 1e-3, "separation_db": 18.0,
            "n_poses": 16, "global_volume": 1.0,
            "render_scatter": 3.49e-4, "scatter_repeats": 12,
        })
        self.assertEqual(restored.single_render_scatter, 3.49e-4)
        self.assertEqual(restored.single_render_repeats, 12)
        self.assertIsNone(restored.clip_render_scatter)
        self.assertIsNone(restored.cue_render_scatter)

    def test_a_record_written_before_any_scatter_existed_leaves_all_three_absent(self):
        restored = CalibrationRecord.from_dict({
            "onset_rms": 3e-3, "bed_rms": 1e-3, "separation_db": 18.0,
            "n_poses": 16, "global_volume": 1.0,
        })
        self.assertIsNone(restored.cue_render_scatter)
        self.assertIsNone(restored.clip_render_scatter)
        self.assertIsNone(restored.single_render_scatter)
        self.assertEqual(restored.cue_phase_folds, 0)
        self.assertIsNone(restored.cue_phase_aggregation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
