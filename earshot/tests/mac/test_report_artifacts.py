"""The one writer: the layout, the round trip, and separability asserted on the bytes.

``test_report_boundary.py`` asserts the testimony cannot *hold* a privileged value.
This asserts the artefact does not *contain* one — the same property one layer down,
where a reviewer actually meets it. Both are needed: a type-level guarantee says nothing
about what a writer chose to serialise beside it.

Atomic-write and refuse-overwrite are here for one reason each, and both are answers to
the same incident. This project's audit found committed run directories holding a
different run's data, quoted against numbers they did not come from. A half-written JSON
and a silently re-used ``--tag`` are the two ways that happens without leaving a trace.
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from _interpreter import assert_interpreter  # noqa: F401

from earshot.report.agent import AgentReport, missing_schema_keys
from earshot.report.artifacts import (
    RUN_SUMMARY_NAME,
    ArtifactExistsError,
    episode_paths,
    read_agent_report,
    read_episode,
    write_env_report,
    write_episode,
    write_run_summary,
)
from earshot.report.audit import EpisodeAudit, FunnelStage, OnsetRecord, StepRecord
from earshot.types import Pose, Xyz

# The names §5.2 keeps and §5.1 must never see, as they appear in the serialised form.
PRIVILEGED_KEYS = ("source_xyz", "dist_at_stop", "source_is_visible_history", "audit")


def _agent_report():
    return AgentReport(
        primary_completed=True,
        heard_at_step=101,
        room="bedroom",
        anomaly_class="baby_cry",
        stopped_at_pose=Pose(Xyz(1.25, 0.07, -3.5), 0.7853981633974483),
        visual_confirm_object="bed",
        investigate_aborted=False,
        resumed=True,
        n_benign_ignored=2,
    )


def _audit(index=0):
    return EpisodeAudit(
        episode_index=index,
        scene_id="TEEsavR23oF",
        localization_arm="realizable",
        detector_arm="oracle",
        source_xyz=Xyz(2.0, 0.1, -4.0),
        dist_at_stop=0.94,
        funnel_stage=FunnelStage.PRIMARY_RESUMED,
        onset=OnsetRecord(onset_step=101, pre_onset_rms=1e-3, n_pre_onset_readings=101,
                          provenance_asserted=True),
        steps=(StepRecord(0, 1e-3, action="move_forward", audio_render_s=0.027),),
    )


class TestTheLayout(unittest.TestCase):
    def test_the_paths_are_adr_0013s_paths(self):
        agent_path, audit_path = episode_paths("runs/smoke", 12)
        self.assertEqual(agent_path.as_posix(), "runs/smoke/episodes/ep0012.agent.json")
        self.assertEqual(audit_path.as_posix(), "runs/smoke/episodes/ep0012.audit.json")

    def test_the_index_is_zero_padded_so_a_lexical_listing_is_chronological(self):
        first, _ = episode_paths("runs/smoke", 2)
        tenth, _ = episode_paths("runs/smoke", 10)
        self.assertLess(first.name, tenth.name)

    def test_writing_creates_the_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = pathlib.Path(tmp) / "runs" / "smoke"
            write_episode(run_dir, 0, _agent_report(), _audit())
            self.assertTrue((run_dir / "episodes" / "ep0000.agent.json").exists())
            self.assertTrue((run_dir / "episodes" / "ep0000.audit.json").exists())

    def test_the_env_report_is_per_run_and_the_episodes_are_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_env_report(tmp, {"green": True, "probes": []})
            write_episode(tmp, 0, _agent_report(), _audit())
            self.assertTrue((pathlib.Path(tmp) / "env_report.json").exists())
            self.assertTrue((pathlib.Path(tmp) / "episodes").is_dir())


class TestSeparability(unittest.TestCase):
    """The property the paper uses: hand over the testimony, keep the answer key."""

    def test_the_testimony_file_contains_no_privileged_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent_path, _ = write_episode(tmp, 0, _agent_report(), _audit())
            raw = agent_path.read_text(encoding="utf-8")
            payload = json.loads(raw)
            for key in PRIVILEGED_KEYS:
                self.assertNotIn(key, payload)
                self.assertNotIn(key, raw, "{} appears in the testimony's bytes".format(key))

    def test_the_testimony_reads_back_without_the_audit_file_existing(self):
        """A reviewer handed one file must be able to read it with nothing else.

        Asserted by deleting the audit rather than by argument — a reader that quietly
        reached for its sibling would pass every other test here.
        """
        with tempfile.TemporaryDirectory() as tmp:
            agent_path, audit_path = write_episode(tmp, 0, _agent_report(), _audit())
            audit_path.unlink()
            self.assertEqual(read_agent_report(agent_path), _agent_report())

    def test_the_audit_file_is_where_the_answer_key_lives(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, audit_path = write_episode(tmp, 0, _agent_report(), _audit())
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source_xyz"], [2.0, 0.1, -4.0])
            self.assertAlmostEqual(payload["dist_at_stop"], 0.94)

    def test_the_privileged_key_list_is_not_vacuous(self):
        """A misspelled name here would make the test above pass while checking nothing —
        the inert-pin class. So each name must genuinely be on the audit's side."""
        with tempfile.TemporaryDirectory() as tmp:
            _, audit_path = write_episode(tmp, 0, _agent_report(), _audit())
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            for key in ("source_xyz", "dist_at_stop", "source_is_visible_history"):
                self.assertIn(key, payload)


class TestTheRoundTrip(unittest.TestCase):
    def test_both_halves_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_episode(tmp, 3, _agent_report(), _audit(3))
            agent_report, audit = read_episode(tmp, 3)
            self.assertEqual(agent_report, _agent_report())
            self.assertEqual(audit, _audit(3))

    def test_the_pose_survives_to_full_precision(self):
        """``stopped_at_pose`` is the field §5.1 substitutes for the source coordinate,
        so a rounded yaw would quietly degrade the one thing the testimony asserts."""
        with tempfile.TemporaryDirectory() as tmp:
            write_episode(tmp, 0, _agent_report(), _audit())
            restored, _ = read_episode(tmp, 0)
            self.assertEqual(restored.stopped_at_pose, _agent_report().stopped_at_pose)

    def test_the_serialised_testimony_carries_all_nine_keys(self):
        """Smoke criterion 6, as a key check rather than a null check — ``room`` and
        ``visual_confirm_object`` are legitimately absent on a run that never saw them."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_path, _ = write_episode(tmp, 0, AgentReport(), _audit())
            payload = json.loads(agent_path.read_text(encoding="utf-8"))
            self.assertEqual(missing_schema_keys(payload), ())
            self.assertEqual(len(payload), 9)

    def test_an_unknown_key_is_refused_rather_than_dropped(self):
        """A tolerant reader is how a privileged field gets into a run directory and
        out of it again unnoticed: the artefact carries it, the type does not, and the
        disjointness test stays green."""
        payload = _agent_report().as_dict()
        payload["source_xyz"] = [2.0, 0.1, -4.0]
        with self.assertRaises(ValueError):
            AgentReport.from_dict(payload)


class TestTheWriteIsSafe(unittest.TestCase):
    def test_re_using_a_run_tag_raises_rather_than_overwriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_episode(tmp, 0, _agent_report(), _audit())
            with self.assertRaises(ArtifactExistsError):
                write_episode(tmp, 0, _agent_report(), _audit())

    def test_the_env_report_refuses_the_same_way(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_env_report(tmp, {"green": True})
            with self.assertRaises(ArtifactExistsError):
                write_env_report(tmp, {"green": False})

    def test_overwrite_is_available_when_it_is_the_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_env_report(tmp, {"green": True})
            path = write_env_report(tmp, {"green": False}, overwrite=True)
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["green"])

    def test_a_refused_write_leaves_the_original_intact(self):
        """The refusal must not be a partial clobber — the first run's data survives."""
        with tempfile.TemporaryDirectory() as tmp:
            agent_path, _ = write_episode(tmp, 0, _agent_report(), _audit())
            with self.assertRaises(ArtifactExistsError):
                write_episode(tmp, 0, AgentReport(), _audit())
            self.assertEqual(read_agent_report(agent_path), _agent_report())

    def test_no_temp_file_is_left_behind(self):
        """The atomic write goes through a temp file in the destination directory, so a
        leaked one would sit inside the run directory a reviewer is handed."""
        with tempfile.TemporaryDirectory() as tmp:
            write_episode(tmp, 0, _agent_report(), _audit())
            stray = [p.name for p in (pathlib.Path(tmp) / "episodes").iterdir()
                     if p.name.endswith(".tmp")]
            self.assertEqual(stray, [])

    def test_the_index_and_the_audit_must_agree(self):
        """The same fact stored twice, and a mismatch is the renumbering class of bug
        that silently dropped pairs from an earlier analyzer."""
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_episode(tmp, 4, _agent_report(), _audit(5))

    def test_a_failed_second_write_does_not_leave_a_lone_testimony(self):
        """Testimony is written first on purpose: a crash between the two leaves the
        file that cannot mislead. This pins the ordering rather than assuming it."""
        with tempfile.TemporaryDirectory() as tmp:
            _, audit_path = episode_paths(tmp, 0)
            audit_path.parent.mkdir(parents=True)
            audit_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(ArtifactExistsError):
                write_episode(tmp, 0, _agent_report(), _audit())
            agent_path, _ = episode_paths(tmp, 0)
            self.assertTrue(agent_path.exists())


class TestTheRunSummary(unittest.TestCase):
    """`summary.json` — the attrition number, in an artefact rather than a console line.

    ``RunSummary.skipped`` is how many of a scene's ObjectNav episodes could not express a
    decoupled anomaly response, and why. It bounds every ``n`` the experiment matrix can
    quote, and until this landed it existed only in stdout. The carried notifier had been
    digesting ``runs/*/summary.json`` into every emailed run report the whole time,
    finding nothing, and saying so in a line nobody read as a defect.
    """

    def _summary(self, **over):
        payload = {
            "run_dir": "runs/x",
            "scene": "4ok3usBNeis",
            "n_episodes": 1,
            "n_skipped": 1,
            "funnel": {"RUN": 1, "PRIMARY_RESUMED": 1},
            "skipped": [{
                "episode_id": "0",
                "reason": "no object in 4ok3usBNeis is >= 3.00 m (xz) from every 'bed' "
                          "goal (rejected: 11 too near, 4 on another floor, 0 with no "
                          "view point).\nThis episode spans floors.",
            }],
        }
        payload.update(over)
        return payload

    def test_it_round_trips_with_the_reasons_whole(self):
        """First-lining the reason would drop the per-rule counts, which are the point."""
        with tempfile.TemporaryDirectory() as td:
            path = write_run_summary(td, self._summary())
            back = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(path.name, RUN_SUMMARY_NAME)
        self.assertEqual(back["n_skipped"], 1)
        self.assertIn("11 too near", back["skipped"][0]["reason"])
        self.assertIn("spans floors", back["skipped"][0]["reason"],
                      "the reason was truncated to its first line")

    def test_it_refuses_to_overwrite_like_every_other_artefact(self):
        with tempfile.TemporaryDirectory() as td:
            write_run_summary(td, self._summary())
            with self.assertRaises(ArtifactExistsError):
                write_run_summary(td, self._summary())
            write_run_summary(td, self._summary(n_episodes=9), overwrite=True)

    def test_the_notifier_looks_for_exactly_this_name(self):
        """The seam that was silently broken: two files, one name, no shared constant.

        ``notify_email.py`` is stdlib-only and standalone by design, so it cannot import
        this module — it globs ``runs/*/summary.json`` as a literal. Pinned here rather
        than trusted, because the last time these two disagreed it went unnoticed across
        every run report since the rebuild.
        """
        notifier = (pathlib.Path(__file__).resolve().parents[3]
                    / "earshot/tools/notify/notify_email.py")
        text = notifier.read_text(encoding="utf-8").replace("'", '"')
        self.assertIn('"{}"'.format(RUN_SUMMARY_NAME), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
