"""The memory arm, driven through the whole episode loop against fakes.

ADR-0018's four cells differ only if a store changes what the agent does. Two identical
runs that differ only in which store they were handed, producing identical trajectories,
is the failure `window_pilot.sh` already warns about in its own header and the reason
511b52f's note says a half-wired store is worse than none. So this file does not check
that a context was stored: it runs the loop twice, with and without, and asserts the
agent WENT SOMEWHERE ELSE.

The mechanism under test is the tour-learned category. The semantic store has learned
"this sound is heard at a stove"; the scene has a stove; after ADR-0017's window closes
the live cue has nothing left to say and the prior names the stove. While the source is
still sounding the cue is real evidence and the prior must stay out of its way, which is
its own test below.

Everything here is on the Mac against `FakeWorld` / `FakeAudioSensorHandle`. What a green
licenses is the wiring and the gating; it licenses nothing about CLAP's real embeddings,
which is `tests/box/` territory.
"""

import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401
from _task_fakes import (
    FakeAudioSensorHandle,
    FakeWorld,
    make_anomaly_episode,
    make_episode,
    make_goal,
)

from earshot.audio.clap import NORMAL_PROMPTS
from earshot.audio.config import AudioConfig
from earshot.audio.calibration import CalibrationResult
from earshot.audio.clips import synthetic_burst
from earshot.config import Detector, Localization, RunConfig
from earshot.memory.store import MemoryCondition, SemanticEntry, SemanticStore
from earshot.task.memory_prior import MemoryContext, PriorMiss
from earshot.task.runner import make_detector, run_episode
from earshot.types import Xyz

CLIP = synthetic_burst(seconds=0.05, sample_rate=44100)
# The same fixture `test_task_runner` drives its own loop with, so a difference here is
# never the reason an arm behaved differently.
CALIBRATION = CalibrationResult(
    onset_rms=0.003,
    bed_rms=1e-3,
    anomaly_low=0.008,
    anomaly_median=0.01,
    anomaly_min=0.005,
    anomaly_max=0.05,
    separation_db=18.0,
    n_poses=16,
    global_volume=1.0,
)

# The one direction the fake encoder ever returns for audio. The store's rows sit on it,
# so the k-NN vote is unambiguous and the test is about the WIRING, not about CLAP.
HEARD = np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeClapEncoder:
    """`encode_audio` / `encode_text`, enough for `is_anomaly` and `audio_embedding`.

    Text lands on `[1,0,0]` for an anomaly prompt and `[0,1,0]` for a normal one, so the
    anomaly cosine is 1.0 against 0.0 and `is_anomaly` fires: the gate is not what this
    file is testing and it must not be what decides whether the detour happens.
    """

    def encode_audio(self, waveform, sample_rate):  # noqa: ARG002 - signature is the contract
        return HEARD

    def encode_text(self, prompt):
        normal = prompt in set(NORMAL_PROMPTS)
        return np.array([0.0, 1.0, 0.0] if normal else [1.0, 0.0, 0.0], dtype=np.float32)


def semantic_store(category="stove", embedding=HEARD):
    return SemanticStore(
        entries=(
            SemanticEntry(
                sound_class="alarm",
                room="kitchen",
                category=category,
                embedding=np.asarray(embedding, dtype=np.float32),
                donor_scene="donor_1",
            ),
        )
    )


class MemoryArmCase(unittest.TestCase):
    """One geometry, run several ways. The stove is deliberately BEHIND the agent.

    The source is 5 m ahead and goes silent after two steps. A cast/scan cycle steering on
    a dead cue keeps hunting forward; a prior that names the stove turns the agent around.
    Putting the recalled object where the blind probe would never send it is what makes the
    trajectory difference a signal rather than a coincidence.
    """

    SOURCE = Xyz(0.0, 0.0, -5.0)
    STOVE = Xyz(0.0, 0.0, 6.0)

    def config(self, **overrides):
        base = dict(
            run_dir="/nonexistent",
            max_steps=40,
            t_anom=2,
            localization=Localization.REALIZABLE,
            detector=Detector.ORACLE,
            audio=AudioConfig(step_seconds=0.01),
            sounding_steps=2,
        )
        base.update(overrides)
        return RunConfig(**base)

    def run_arm(self, memory, **kwargs):
        cfg = self.config(**kwargs.pop("config", {}))
        world = FakeWorld(start=Xyz(0.0, 0.0, 0.0), yaw=0.0)
        handle = FakeAudioSensorHandle(world, self.SOURCE)
        episode = make_episode(goals=[make_goal(Xyz(0.0, 0.0, -9.0))])
        anomaly = make_anomaly_episode(source=self.SOURCE, episode=episode, t_anom=2)
        return run_episode(
            world,
            handle,
            anomaly,
            cfg,
            clip=CLIP,
            detector=make_detector(cfg, world, anomaly),
            calibration=CALIBRATION,
            clap_encoder=kwargs.pop("clap_encoder", FakeClapEncoder()),
            memory=memory,
            **kwargs,
        )

    def context(self, **overrides):
        base = dict(
            condition=MemoryCondition.HEARD_UNSEEN,
            semantic=semantic_store(),
            points_by_category={"stove": (self.STOVE,)},
            k=1,
        )
        base.update(overrides)
        return MemoryContext(**base)


class TestTheMemoryChangesWhatTheAgentDoes(MemoryArmCase):
    """The assertion the whole matrix rests on: two arms, two trajectories."""

    @classmethod
    def setUpClass(cls):
        case = cls()
        cls.without = case.run_arm(None)
        cls.with_memory = case.run_arm(case.context())

    def test_the_arm_with_memory_recalled_the_planted_category(self):
        audit = self.with_memory.audit
        print(
            "\n  [memory arm] condition={} recalled={!r} miss={!r}".format(
                audit.memory_condition,
                audit.memory_prior_category,
                audit.memory_prior_miss,
            ),
            flush=True,
        )
        self.assertEqual(audit.memory_prior_category, "stove")
        self.assertIsNone(audit.memory_prior_miss)

    def test_the_arm_without_memory_recorded_no_cell_at_all(self):
        # `None` here means "no memory arm ran". `MemoryCondition.NONE` means "a memory arm
        # ran and its stores were empty". The two are different facts and the audit keeps
        # them apart by the presence of a value.
        audit = self.without.audit
        self.assertIsNone(audit.memory_condition)
        self.assertIsNone(audit.memory_prior_category)
        self.assertIsNone(audit.memory_prior_miss)

    def test_the_two_arms_walked_different_paths(self):
        """Not a stored flag: the agent physically went somewhere else."""
        a = [(row.position.x, row.position.z) for row in self.without.audit.steps]
        b = [(row.position.x, row.position.z) for row in self.with_memory.audit.steps]
        print(
            "\n  [memory arm] {} step(s) without memory, {} with; final z {:.2f} vs "
            "{:.2f} (the stove is at z={:.1f})".format(
                len(a), len(b), a[-1][1], b[-1][1], self.STOVE.z
            ),
            flush=True,
        )
        self.assertNotEqual(a, b)

    def test_the_memory_arm_moved_toward_the_recalled_object(self):
        # The stove is behind the agent; the source and the primary goal are both ahead.
        # Any movement to positive z is movement the un-cued arm has no reason to make.
        reached = max(row.position.z for row in self.with_memory.audit.steps)
        baseline = max(row.position.z for row in self.without.audit.steps)
        self.assertGreater(reached, baseline)

    def test_the_prior_is_recorded_as_numbers_a_reader_can_check(self):
        metrics = self.with_memory.audit.metrics
        self.assertIn("memory_prior_confidence", metrics)
        self.assertIn("memory_prior_distance_m", metrics)
        self.assertEqual(metrics["memory_prior_instances"], 1.0)
        self.assertAlmostEqual(metrics["memory_prior_confidence"], 1.0, places=5)
        self.assertTrue(all(isinstance(v, float) for v in metrics.values()))


class TestTheGating(MemoryArmCase):
    """The prior speaks only into silence, and only once."""

    def test_a_run_whose_window_never_closes_never_consults_the_prior(self):
        # Sounding for the whole episode: the live cue is real evidence throughout and the
        # memory must stay out of its way. Both fields stay None, which is the third fact
        # -- "never consulted", distinct from a recall and from a miss.
        result = self.run_arm(self.context(), config={"sounding_steps": 500})
        self.assertIsNone(result.audit.memory_prior_category)
        self.assertIsNone(result.audit.memory_prior_miss)
        self.assertEqual(result.audit.memory_condition, MemoryCondition.HEARD_UNSEEN.value)


class TestTheMissesReachTheAudit(MemoryArmCase):
    """Three named reasons, three different facts, none of them a blank."""

    def test_an_empty_store_records_no_prediction(self):
        # The `not_heard` cells. The arm RAN and the memory had nothing, which must not
        # read the same as the arm not running.
        result = self.run_arm(self.context(semantic=SemanticStore()))
        self.assertEqual(result.audit.memory_prior_miss, PriorMiss.NO_PREDICTION.value)
        self.assertIsNone(result.audit.memory_prior_category)
        self.assertEqual(
            result.audit.memory_condition, MemoryCondition.HEARD_UNSEEN.value
        )

    def test_a_scene_without_the_recalled_object_records_category_absent(self):
        result = self.run_arm(self.context(points_by_category={"toilet": (self.STOVE,)}))
        self.assertEqual(result.audit.memory_prior_miss, PriorMiss.CATEGORY_ABSENT.value)

    def test_a_miss_leaves_the_prior_metrics_off_the_record(self):
        # Absent, never 0.0: a confidence of zero would read as a recall that scored badly.
        result = self.run_arm(self.context(semantic=SemanticStore()))
        self.assertNotIn("memory_prior_confidence", result.audit.metrics)
        self.assertNotIn("memory_prior_distance_m", result.audit.metrics)


class TestTheWiringMistakeThatWouldLookLikeAResult(MemoryArmCase):
    def test_a_memory_arm_without_an_encoder_raises_rather_than_missing_silently(self):
        with self.assertRaises(ValueError) as caught:
            self.run_arm(self.context(), clap_encoder=None)
        message = str(caught.exception)
        self.assertIn("CLAP", message)
        self.assertIn("silently never fire", message)

    def test_k_below_one_is_refused_when_the_context_is_built(self):
        with self.assertRaises(ValueError):
            self.context(k=0)


class TestTheCellsAreCarvedNotBranched(MemoryArmCase):
    """`is_live` is the rule the runner would otherwise encode twice."""

    def test_an_empty_store_is_not_live(self):
        self.assertFalse(self.context(semantic=SemanticStore()).is_live)

    def test_a_scene_with_no_objects_is_not_live(self):
        self.assertFalse(self.context(points_by_category={}).is_live)

    def test_both_halves_present_is_live(self):
        self.assertTrue(self.context().is_live)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
