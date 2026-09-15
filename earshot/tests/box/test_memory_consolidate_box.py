#!/usr/bin/env python3
"""Eq. 9's segmentation against a REAL walk, because its knob could be inert and look fine.

    conda activate ss2
    python -m earshot.task.models          # stages CLAP and CLIP
    bash earshot/tools/box_gate.sh

**THE QUESTION THE MAC SUITE CANNOT ASK.** `tests/mac/test_memory_consolidate.py` proves
`segment_trajectory` cuts a slow drift, using a fixture built to drift. It cannot say
whether a REAL trajectory of real `z^av` vectors drifts at all. If consecutive fused
observations in a house sit at 0.999 cosine from the first step to the last, then
`coherence` never fires at any value a person would write down, only `max_length` ever
cuts, and eq. 9's "temporally coherent" segmentation is a fixed-length window wearing a
similarity rule's name.

That failure has a record in this repo: grid-A* found no path on ~92% of steps and fell
back to straight lines; the S1+ captioner never lifted; the Step-4 affordance head
proposed and was never chosen. Each was correct code that no run could distinguish from
its absence. This test is the arm that would catch the same shape here, before PR 6 wires
a threshold nobody has measured.

**These tests print their measurements** (ADR-0014). The number this file exists to
produce is the distribution of consecutive-step cosine on real fused observations. It is
what sets `coherence` in the sweep, and no Mac fixture can supply it.

**The source clip is a `synthetic_burst`, on purpose.** An episode plays an ESC-50
recording, and that recording is IDENTICAL at every step of the walk — so it contributes
none of the step-to-step variation this test measures, and which clip it is cannot change
the answer. Depending on `data/anomaly_audio` would instead make this test SKIP on a box
without the corpus staged, and a criterion that could not be evaluated is never green.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import math
import os
import unittest

import numpy as np

from earshot.audio.clap import audio_embedding, heard_clip_for_clap
from earshot.audio.clips import render_through_ir, synthetic_burst
from earshot.audio.config import AudioConfig
from earshot.audio.sensor import AudioSensorHandle
from earshot.audio.spec import audio_sensor_spec
from earshot.memory.consolidate import (
    ImportanceWeights,
    TrajectoryStep,
    contribution,
    importance,
    novelty,
    segment_trajectory,
    surprise,
)
from earshot.task.dataset import available_scenes, find_scenes_dir, find_split_dir, load_scene

SPLIT = "val"
PLACEMENT_SEED = 20260821
WALK_STEPS = 30
GOAL_RADIUS = 0.3

# The claim: a real indoor walk spreads far enough that `coherence` is a knob a person can
# set, not a fourth decimal place. Below this floor at least once over the walk means a
# threshold placed anywhere sensible has something to cut on. Above it everywhere means
# eq. 9 degenerates to fixed windows and PR 6 must be told so.
FUSED_COHERENCE_FLOOR = 0.95

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


def _cosines(vectors):
    return [float(np.dot(vectors[i], vectors[i + 1])) for i in range(len(vectors) - 1)]


def _spread(name, values):
    print("  {:<22} min {:.4f}  median {:.4f}  max {:.4f}  over {} pairs".format(
        name, min(values), sorted(values)[len(values) // 2], max(values), len(values)))


class TestTheSegmentationOnARealWalk(unittest.TestCase):
    """One walk, rendered and encoded at every step, then put through eq. 9."""

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
        start = cls.world.random_navigable_point()
        target = cls.world.random_navigable_point()
        handle = AudioSensorHandle(
            cls.world.sensor_handle(str(spec.uuid)), cls.world.observe, source,
            uuid=str(spec.uuid),
        )
        clip = synthetic_burst(config.sample_rate)

        cls.world.set_pose(start)
        follow = cls.world.follower(goal_radius=GOAL_RADIUS)
        cls.visual, cls.audio, cls.steps = [], [], []
        for _ in range(WALK_STEPS):
            observation, _guard = handle.observe()
            z_v = visual_embedding(np.asarray(observation["rgb"]), clip_encoder)
            heard = render_through_ir(handle.audio_of(observation), clip)
            mono, rate = heard_clip_for_clap(heard, config.sample_rate)
            z_u = audio_embedding(mono, rate, clap_encoder)
            cls.visual.append(z_v)
            cls.audio.append(z_u)
            cls.steps.append(TrajectoryStep(
                observation=fuse(z_v, z_u),
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

        cls.walked = cls.steps[0].position.horizontal_distance_to(cls.steps[-1].position)
        print("  walked {} steps, {:.2f} m between the two ends".format(
            len(cls.steps), cls.walked), flush=True)
        print("  z^v width {}  z^u width {}  z^av width {}".format(
            cls.visual[0].size, cls.audio[0].size, cls.steps[0].observation.size), flush=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "world", None) is not None:
            cls.world.close()

    def test_the_walk_actually_moved(self):
        """The control for everything below. A follower that never left the start would
        make every observation identical and every measurement here vacuous."""
        self.assertGreater(len(self.steps), 5, "the walk was too short to segment")
        self.assertGreater(
            self.walked, 1.0,
            "the agent covered {:.2f} m, so the cosines below describe standing "
            "still".format(self.walked),
        )

    def test_the_fused_observation_spreads_enough_for_coherence_to_be_a_knob(self):
        """**THE MEASUREMENT THIS FILE EXISTS FOR.** Prints the consecutive-step cosine
        for each half and for the fusion, and asserts the fused one reaches below the
        floor at least once. Red here means eq. 9's similarity rule has nothing to cut on
        in a real house and degenerates to a fixed window."""
        fused = [step.observation for step in self.steps]
        visual_cos, audio_cos, fused_cos = (
            _cosines(self.visual), _cosines(self.audio), _cosines(fused)
        )
        _spread("z^v consecutive", visual_cos)
        _spread("z^u consecutive", audio_cos)
        _spread("z^av consecutive", fused_cos)
        print("  (f_fuse halves the spread by construction: cos(z^av) = "
              "[cos(z^v) + cos(z^u)] / 2 on unit halves)")
        print("  ends of the walk: z^v {:.4f}  z^u {:.4f}  z^av {:.4f}".format(
            float(np.dot(self.visual[0], self.visual[-1])),
            float(np.dot(self.audio[0], self.audio[-1])),
            float(np.dot(fused[0], fused[-1]))))
        self.assertLess(
            min(fused_cos), FUSED_COHERENCE_FLOOR,
            "no consecutive pair of real fused observations fell below {}, so coherence "
            "is not a knob anyone can set and eq. 9 is a fixed-length window under "
            "another name".format(FUSED_COHERENCE_FLOOR),
        )

    def test_coherence_cuts_a_real_trajectory_with_the_cap_disabled(self):
        """Any cut here is a COHERENCE cut: `max_length` is set to the whole walk, so the
        cap cannot be what ended a segment. This is the arm that separates "the rule
        fires" from "the window expired"."""
        fused_cos = _cosines([step.observation for step in self.steps])
        threshold = (min(fused_cos) + max(fused_cos)) / 2.0
        segments = segment_trajectory(
            self.steps, coherence=threshold, min_length=2, max_length=len(self.steps)
        )
        print("  at coherence {:.4f} (midpoint of the observed range), cap disabled: "
              "{} segment(s), lengths {}".format(
                  threshold, len(segments), [len(s) for s in segments]))
        self.assertGreater(
            len(segments), 1,
            "with max_length disabled the whole walk stayed one segment, so no cut in "
            "this trajectory ever came from coherence",
        )

    def test_f_seg_summarises_rather_than_copies_a_step(self):
        """`h_j` (eq. 11) on real observations. A representation identical to its own
        first step would mean the mean is being taken over a run of duplicates."""
        segments = segment_trajectory(
            self.steps, coherence=0.99, min_length=3, max_length=10
        )
        for index, segment in enumerate(segments):
            h = segment.representation
            inside = [float(np.dot(h, step.observation)) for step in segment.steps]
            print("  delta_{}: {} steps, h_j to its own steps min {:.4f} max {:.4f}".format(
                index, len(segment), min(inside), max(inside)))
            self.assertAlmostEqual(float(np.linalg.norm(h)), 1.0, places=5)
        multi = [segment for segment in segments if len(segment) > 1]
        self.assertTrue(multi, "every segment was one step; nothing was summarised")
        self.assertLess(
            min(float(np.dot(segment.representation, segment.steps[0].observation))
                for segment in multi),
            1.0 - 1e-6,
            "h_j equals its segment's first step exactly, so f_seg is copying rather "
            "than averaging",
        )

    def test_novelty_separates_the_far_half_of_a_walk_from_the_near_half(self):
        """`N_j` (eq. 12) with a memory that holds something real, which the Mac suite
        can only fake. The second half of a walk should be measurably more novel against
        a memory of the first half than the first half is against itself."""
        half = len(self.steps) // 2
        early = segment_trajectory(
            self.steps[:half], coherence=0.99, min_length=2, max_length=8
        )
        late = segment_trajectory(
            self.steps[half:], coherence=0.99, min_length=2, max_length=8
        )
        memory = [segment.representation for segment in early]
        self_novelty = [novelty(segment.representation, memory) for segment in early]
        late_novelty = [novelty(segment.representation, memory) for segment in late]
        print("  memory of {} early segment(s); N_j of the early half {} ".format(
            len(memory), ["{:.4f}".format(value) for value in self_novelty]))
        print("  N_j of the late half {}".format(
            ["{:.4f}".format(value) for value in late_novelty]))
        print("  mean early {:.4f} vs mean late {:.4f}".format(
            sum(self_novelty) / len(self_novelty), sum(late_novelty) / len(late_novelty)))
        self.assertLess(
            max(self_novelty), 1e-5,
            "a segment scored novel against a memory that literally contains it",
        )
        self.assertGreater(
            max(late_novelty), 0.0,
            "no part of the second half of the walk was novel against the first, so "
            "N_j cannot distinguish anything in a real episode",
        )

    def test_the_three_terms_score_a_real_trajectory(self):
        """Eq. 10 end to end on real segments, with the numbers printed so PR 6 can see
        what range `eta` has to sit in.

        `U_j` is 0 throughout and says so: `belief` is the CONTROLLER's per-step source
        estimate and no controller ran here. That half is exercised on the Mac, where a
        belief can be constructed, and is wired in PR 6.
        """
        segments = segment_trajectory(
            self.steps, coherence=0.99, min_length=3, max_length=10
        )
        # The episode's own final position, NEVER the source: the memory is written by
        # the agent, and a consolidation scored against ground truth makes every later
        # retrieval GT-privileged.
        target = self.steps[-1].position
        weights = ImportanceWeights(alpha=1.0, beta=1.0, gamma=1.0)
        contributions = contribution(segments, target=target)
        surprises = surprise(segments)
        scores = importance(segments, target=target, memory=[], weights=weights)
        print("  alpha=beta=gamma=1, memory empty, target = the episode's final pose")
        for index, segment in enumerate(segments):
            print("    delta_{}: {:2d} steps  C {:.4f}  U {:.4f}  I {:.4f}".format(
                index, len(segment), contributions[index], surprises[index], scores[index]))
        print("  I range {:.4f} to {:.4f}".format(min(scores), max(scores)))
        self.assertAlmostEqual(sum(contributions), 1.0, places=4)
        self.assertEqual(set(surprises), {0.0}, "a belief appeared where none was set")
        self.assertTrue(all(math.isfinite(score) for score in scores))


if __name__ == "__main__":
    unittest.main(verbosity=2)
