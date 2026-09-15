#!/usr/bin/env python3
"""`omega_t` on real observations, because the hypothesis can be true and carry nothing.

    conda activate ss2
    python -m earshot.task.models          # stages CLAP and CLIP
    bash earshot/tools/box_gate.sh

**THE QUESTION THE MAC SUITE CANNOT ASK.** `tests/mac/test_memory_retrieve.py` proves
`omega_t` shifts toward the abstraction level as a query leaves the cluster of stored
experience -- using fixtures built to be well separated. Its
`test_a_pattern_over_near_identical_experiences_carries_no_information` records the
opposite case, and that case is not hypothetical: if real fused observations inside one
house all sit at ~0.99 cosine of each other, then every `M^E` row, every `M^P` signature
and every query are the same vector, every level scores the same, and `omega_t` is uniform
no matter how familiar the condition is. Eq. 23 would be correct, implemented, and
measuring nothing.

**THE ASSERTION IS TEMPERATURE-FREE ON PURPOSE.** `omega_t` itself can be made to look
lively by shrinking `temperature`, so asserting on it would be asserting on a knob. What
`temperature` cannot manufacture is the SPREAD of the underlying per-level similarity gap
`s_E - s_P` along a walk. If that gap is flat, no temperature rescues it; if it moves, the
mechanism has something to work with. The gap is the headline measurement here and
`omega_t` is printed beside it.

**These tests print their measurements** (ADR-0014).

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from earshot.audio.clap import audio_embedding, heard_clip_for_clap
from earshot.audio.clips import render_through_ir, synthetic_burst
from earshot.audio.config import AudioConfig
from earshot.audio.sensor import AudioSensorHandle
from earshot.audio.spec import audio_sensor_spec
from earshot.memory.consolidate import TrajectoryStep, segment_trajectory
from earshot.memory.longterm import (
    ExperienceEntry,
    ExperienceStore,
    LongTermMemory,
    Outcome,
    abstract,
    trajectory_descriptor,
)
from earshot.memory.retrieve import query, retrieve
from earshot.memory.store import SemanticStore
from earshot.task.dataset import available_scenes, find_scenes_dir, find_split_dir, load_scene

SPLIT = "val"
PLACEMENT_SEED = 20260821
WALK_STEPS = 24
GOAL_RADIUS = 0.3
COHERENCE = 0.995
MIN_SEGMENT = 2
MAX_SEGMENT = 6
TEMPERATURE = 0.1
DECAY = 0.8
PRESENT_WEIGHT = 0.7

# The claim: along a real walk, the per-level similarity gap `s_E - s_P` moves. Below this
# it is a constant, `omega_t` is whatever `temperature` says it is, and the dynamic
# retrieval carries no information about the navigation context.
MIN_GAP_SPREAD = 0.01

_DATASET = None


def setUpModule():
    global _DATASET
    split_dir = find_split_dir(SPLIT)
    scenes_dir = find_scenes_dir()
    override = os.environ.get("SS2_SCENE_LABEL")
    for label in ([override] if override else list(available_scenes(split_dir))):
        dataset = load_scene(split_dir, label, scenes_dir=scenes_dir)
        if os.path.exists(dataset.scene_path):
            _DATASET = dataset
            break
    if _DATASET is None:
        raise unittest.SkipTest("no ObjectNav {} scene has its mesh on this box".format(SPLIT))
    print("\n  scene: {}".format(_DATASET.scene_label), flush=True)


class TestOmegaOnRealObservations(unittest.TestCase):
    """Two real walks: one becomes `M^L`, the other queries it step by step."""

    @classmethod
    def setUpClass(cls):
        from earshot.agent.stm import fuse
        from earshot.sim.world import World, audio_spec_parts, camera_sensor_specs
        from earshot.task.models import load_clap_encoder, load_clip_encoder
        from earshot.types import NoRouteError
        from earshot.vlm.encode import visual_embedding

        config = AudioConfig()
        spec, binaural = audio_spec_parts()
        audio_sensor_spec(spec, config, binaural)
        cls.world = World(
            _DATASET.scene_path, camera_sensor_specs(width=256, height=256) + [spec]
        )
        cls.world.seed_navmesh(PLACEMENT_SEED)
        clip_encoder = load_clip_encoder()
        clap_encoder = load_clap_encoder()
        source = cls.world.random_navigable_point()
        handle = AudioSensorHandle(
            cls.world.sensor_handle(str(spec.uuid)), cls.world.observe, source,
            uuid=str(spec.uuid),
        )
        clip = synthetic_burst(config.sample_rate)

        def walk():
            cls.world.set_pose(cls.world.random_navigable_point())
            target = cls.world.random_navigable_point()
            follow = cls.world.follower(goal_radius=GOAL_RADIUS)
            steps = []
            for _ in range(WALK_STEPS):
                observation, _guard = handle.observe()
                z_v = visual_embedding(np.asarray(observation["rgb"]), clip_encoder)
                mono, rate = heard_clip_for_clap(
                    render_through_ir(handle.audio_of(observation), clip),
                    config.sample_rate,
                )
                steps.append(TrajectoryStep(
                    observation=fuse(z_v, audio_embedding(mono, rate, clap_encoder)),
                    position=cls.world.pose().position,
                    belief=None,
                ))
                try:
                    action = follow(target)
                except NoRouteError:
                    break
                if action is None:
                    break
                cls.world.step(action)
            return steps

        cls.stored = walk()
        cls.probe = walk()
        print("  walks: {} steps stored, {} steps probing".format(
            len(cls.stored), len(cls.probe)), flush=True)

        segments = segment_trajectory(
            cls.stored, coherence=COHERENCE, min_length=MIN_SEGMENT, max_length=MAX_SEGMENT
        )
        # Two concept groups, so `G` has something to group. The concepts are labels here
        # rather than measurements -- the point is the VECTORS, which are real.
        half = max(1, len(segments) // 2)
        entries = []
        for index, segment in enumerate(segments):
            sound, obj = (
                ("toilet_flush", "toilet") if index < half else ("snoring", "bed")
            )
            entries.append(ExperienceEntry(
                context=segment.representation,
                trajectory=trajectory_descriptor([step.position for step in segment.steps]),
                target_concept=obj,
                outcome=Outcome(reached=True, final_gap_m=0.5),
                sound_concept=sound,
                room_concept=None,
            ))
        store = ExperienceStore().extend(entries)
        cls.memory = LongTermMemory(
            experience=store,
            pattern=abstract(store, min_support=2),
            knowledge=SemanticStore(),
        )
        print("  M^L from the stored walk: M^E={} rows, M^P={} pattern(s)".format(
            len(cls.memory.experience), len(cls.memory.pattern)), flush=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "world", None) is not None:
            cls.world.close()

    def test_the_two_walks_and_the_abstraction_all_exist(self):
        """The control. An `M^P` with no patterns makes every gap below undefined, and a
        skip here would read as green rather than as unmeasured."""
        self.assertGreaterEqual(len(self.probe), 6, "the probing walk was too short")
        self.assertGreaterEqual(len(self.memory.experience), 2)
        self.assertGreaterEqual(
            len(self.memory.pattern), 1,
            "G produced no pattern from the stored walk, so there is no M^P to weigh "
            "against M^E and eq. 23 has only one live level",
        )

    def test_real_experiences_are_not_all_the_same_vector(self):
        """The precondition for everything else. If every `M^E` key is the same key, the
        retrieval is a formality."""
        keys = np.stack(self.memory.experience.keys)
        cosines = [
            float(np.dot(keys[i], keys[j]))
            for i in range(len(keys)) for j in range(i + 1, len(keys))
        ]
        print("  M^E key-to-key cosine: min {:.4f} mean {:.4f} max {:.4f} over {} "
              "pair(s)".format(min(cosines), sum(cosines) / len(cosines), max(cosines),
                               len(cosines)))
        self.assertLess(
            min(cosines), 0.999,
            "every pair of real M^E keys is identical to three places; the store holds "
            "one vector under several names",
        )

    def test_the_per_level_gap_moves_along_a_real_walk(self):
        """**THE MEASUREMENT THIS FILE EXISTS FOR, AND IT OWES NOTHING TO `temperature`.**
        `s_E - s_P` is the quantity eq. 23's softmax acts on. If it is flat along a real
        walk then `omega_t` is a function of the knob alone."""
        from earshot.agent.stm import ShortTermMemory, StmEntry

        stm = ShortTermMemory(horizon=4)
        # `f_fuse` concatenates [visual, audio], and CLIP and CLAP are both 512 wide here
        # -- which is the whole reason `CLIP_MODEL_ID` was chosen 512-d. Splitting at the
        # midpoint is therefore exact, and asserted rather than assumed.
        total = int(self.probe[0].observation.size)
        self.assertEqual(total % 2, 0, "the fused width is odd; the halves are not equal")
        width = total // 2
        gaps, omegas = [], []
        for step in self.probe:
            stm = stm.push(StmEntry(
                visual=step.observation[:width], audio=step.observation[width:],
                pose=_flat_pose(step.position), prev_action=None,
            ))
            q = query(step.observation, stm.context(decay=DECAY),
                      present_weight=PRESENT_WEIGHT)
            context = retrieve(
                q, self.memory,
                k_experience=1, k_pattern=1, k_knowledge=1, temperature=TEMPERATURE,
            )
            if not context.experience or not context.pattern:
                continue
            gaps.append(context.experience[0].score - context.pattern[0].score)
            omegas.append(context.weights.experience)
        print("  s_E - s_P over {} step(s): min {:+.4f} median {:+.4f} max {:+.4f} "
              "spread {:.4f}".format(
                  len(gaps), min(gaps), sorted(gaps)[len(gaps) // 2], max(gaps),
                  max(gaps) - min(gaps)))
        print("  omega^E at temperature {}: min {:.4f} median {:.4f} max {:.4f}".format(
            TEMPERATURE, min(omegas), sorted(omegas)[len(omegas) // 2], max(omegas)))
        self.assertGreater(len(gaps), 4, "too few steps retrieved from both levels")
        self.assertGreater(
            max(gaps) - min(gaps), MIN_GAP_SPREAD,
            "the per-level similarity gap is constant to within {} along a real walk, so "
            "omega_t is a function of temperature and not of the navigation "
            "context".format(MIN_GAP_SPREAD),
        )

    def test_the_weights_are_a_distribution_on_every_real_step(self):
        """Eq. 23's only hard guarantee, checked on real queries rather than fixtures."""
        for step in self.probe:
            context = retrieve(
                query(step.observation, None, present_weight=1.0), self.memory,
                k_experience=2, k_pattern=1, k_knowledge=1, temperature=TEMPERATURE,
            )
            self.assertIsNotNone(context.weights)
            self.assertAlmostEqual(sum(context.weights.as_tuple()), 1.0, places=5)
            blended = context.fused_context()
            self.assertAlmostEqual(float(np.linalg.norm(blended)), 1.0, places=4)


def _flat_pose(position):
    from earshot.types import Pose

    return Pose(position=position, yaw_rad=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
