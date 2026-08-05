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
from earshot.audio.guard import AudioContextReport
from earshot.audio.onset import OnsetState
from earshot.report.audit import (
    CalibrationRecord,
    EpisodeAudit,
    FunnelStage,
    OnsetRecord,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
