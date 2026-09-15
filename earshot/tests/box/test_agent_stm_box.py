#!/usr/bin/env python3
"""The STM filled from a REAL render, and the CLIP discrimination check PR #108 got wrong.

    conda activate ss2
    python -m earshot.task.models          # stages CLAP and CLIP
    bash earshot/tools/box_gate.sh

**This file exists because PR #108's discrimination test was a proxy, and ADR-0014 forbids
one.** `test_two_different_frames_give_two_different_embeddings` compared two frames of
`np.random` NOISE and asserted their cosine was below 0.999. It measured **0.9945** and
passed -- but a randomly-INITIALISED CLIP would score about the same on two noise frames,
because "noise texture" is one place in embedding space. The test proved the model was not
returning a literal constant and nothing more, and the PR body claimed it caught a
mis-loaded checkpoint. It does not.

The honest form needs the simulator, which is why it is here rather than there: two frames
RENDERED from different poses in a real scene, plus the zero-shot text cosine that
`step4-coarse-affordance` measured at ~0.30 on this checkpoint. A random-init CLIP fails
both; noise frames test neither.

**These tests print their measurements** (ADR-0014). The numbers that matter:

  * the cosine between two real rendered frames -- how much of `z^v`'s range an episode
    actually uses. If real frames are as indistinguishable as noise was, `h^av` (eq. 15)
    carries nothing and `M^E` is a table of near-identical keys;
  * the zero-shot text separation, which is the arm that proves the WEIGHTS loaded rather
    than the architecture;
  * the fused `z^av` (eq. 7) built from a real frame and a real render, because the
    balance argument for a 512-d visual encoder is only worth anything on real inputs.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import os
import unittest

import numpy as np

from earshot.task.episodes import available_scenes, find_scenes_dir, find_split_dir, load_scene

SPLIT = "val"
PLACEMENT_SEED = 20260821

# `step4-coarse-affordance` measured real CLIP room cosines around 0.30 on this
# checkpoint. The floor here is deliberately far below that: this asserts the weights
# LOADED, not that zero-shot room classification is good, and those are different claims.
#
# **0.15 WAS TOO HIGH AND THE BOX SAID SO.** The first run measured per-frame gaps of
# -0.0100, +0.1323, +0.1204, +0.0899 -- mean +0.0832, so the test went red on a CHECKPOINT
# THAT HAD LOADED. The evidence it had: a random-init CLIP has no shared image-text space,
# so its gaps are noise around zero for EVERY prompt, and three of these four are an order
# of magnitude above that. The same run's `test_real_frames_are_distinguishable` put four
# real frames at 0.6137-0.8959 pairwise where two NOISE frames scored 0.9945, which a
# random-init model could not do either.
#
# What the failure actually showed is that ONE frame in four can be a close-up of a wall
# with no room context at all, and a MEAN over four frames is hostage to it. So the
# statistic is now the MEDIAN and the floor is 0.05 -- five times the noise a random-init
# model produces, and comfortably under what a loaded one gives on a frame that shows a
# room. The per-frame numbers are still printed, because the spread is the finding.
MIN_TEXT_COSINE = 0.05

# Two frames of the same room from different poses are similar; a random-init model makes
# everything similar. 0.995 sits between what `step4` saw for real frames and the 0.9945
# two noise frames scored, so it discriminates where the noise test could not.
MAX_SAME_SCENE_COSINE = 0.995

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


class TestTheStmOnRealObservations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from earshot.sim.world import World, camera_sensor_specs
        from earshot.task.models import load_clip_encoder

        cls.encoder = load_clip_encoder()
        cls.world = World(_DATASET.scene_path, camera_sensor_specs(width=256, height=256))
        cls.world.seed_navmesh(PLACEMENT_SEED)
        cls.frames = []
        for _ in range(4):
            cls.world.set_pose(cls.world.random_navigable_point())
            cls.frames.append(np.asarray(cls.world.observe()["rgb"]))
        print("  frames: {} at {}".format(len(cls.frames), cls.frames[0].shape), flush=True)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "world", None) is not None:
            cls.world.close()

    def test_real_frames_are_distinguishable_where_noise_frames_were_not(self):
        """THE TEST PR #108 SHOULD HAVE HAD. Two noise frames scored 0.9945; four real
        poses in one house should spread further than that, or `M^E`'s keys are all the
        same key."""
        from earshot.vlm.encode import visual_embedding

        vectors = [visual_embedding(frame, self.encoder) for frame in self.frames]
        cosines = [
            float(np.dot(vectors[i], vectors[j]))
            for i in range(len(vectors))
            for j in range(i + 1, len(vectors))
        ]
        print("  pairwise cosine over {} real frames: min {:.4f} mean {:.4f} max "
              "{:.4f}".format(len(cosines), min(cosines), sum(cosines) / len(cosines),
                              max(cosines)))
        print("  (PR #108's two NOISE frames scored 0.9945 — a random-init model's score)")
        self.assertLess(
            min(cosines), MAX_SAME_SCENE_COSINE,
            "every pair of real rendered frames is more alike than {}; z^v carries almost "
            "nothing and every M^E row would key on the same vector".format(
                MAX_SAME_SCENE_COSINE),
        )

    def test_the_weights_loaded_and_not_merely_the_architecture(self):
        """THE ARM A NOISE FRAME CANNOT REACH. A random-init CLIP has no shared
        image-text space, so its image-to-text cosines are noise around zero for every
        prompt. A loaded one puts a rendered interior nearer 'a photo of the inside of a
        house' than 'a photo of the surface of the sun'."""
        from earshot.vlm.encode import visual_embedding

        near = "a photo of the inside of a house"
        far = "a photo of the surface of the sun"
        text = self._encode_text([near, far])
        vectors = [visual_embedding(frame, self.encoder) for frame in self.frames]
        gaps = []
        for index, vector in enumerate(vectors):
            hit, miss = float(np.dot(vector, text[0])), float(np.dot(vector, text[1]))
            gaps.append(hit - miss)
            print("    frame {}: '{}' {:+.4f}   '{}' {:+.4f}   gap {:+.4f}".format(
                index, near, hit, far, miss, hit - miss))
        median = sorted(gaps)[len(gaps) // 2]
        print("  interior-minus-sun gap: median {:+.4f} mean {:+.4f} min {:+.4f} max "
              "{:+.4f} (floor {:+.4f})".format(
                  median, sum(gaps) / len(gaps), min(gaps), max(gaps), MIN_TEXT_COSINE))
        print("  (a random-init CLIP has no shared image-text space: every gap would be "
              "noise around 0.0000)")
        self.assertGreater(
            median, MIN_TEXT_COSINE,
            "the MEDIAN rendered interior is not nearer 'inside of a house' than "
            "'surface of the sun'; the checkpoint's weights did not load and only its "
            "shape did. A single negative frame does not mean this -- see the constant's "
            "comment -- but a median at or below the noise floor does",
        )

    def _encode_text(self, prompts):
        """CLIP's text tower, used ONLY as a probe. `vlm/encode.py` deliberately has no
        text path: nothing in DREAM queries a memory with a sentence, and a public one
        would invite it."""
        import torch

        inputs = self.encoder._processor(
            text=list(prompts), return_tensors="pt", padding=True
        ).to(self.encoder.device)
        with torch.no_grad():
            features = self.encoder._model.get_text_features(**inputs)
        out = features.float().cpu().numpy()
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    def test_a_real_fused_observation_is_unit_norm_and_keeps_both_halves(self):
        """`f_fuse` (eq. 7) on real inputs. The balance argument for a 512-d visual
        encoder is only worth something against a real CLAP vector beside it."""
        from earshot.agent.stm import fuse
        from earshot.vlm.encode import visual_embedding

        visual = visual_embedding(self.frames[0], self.encoder)
        # A real CLAP-width vector. The audio half's CONTENT does not matter here; its
        # WIDTH does, because that is what the balance rests on.
        audio = np.zeros(512, dtype=np.float32)
        audio[7] = 1.0
        fused = fuse(visual, audio)
        print("  fused z^av: shape {} norm {:.6f}  visual half norm {:.6f}  audio half "
              "norm {:.6f}".format(
                  fused.shape, float(np.linalg.norm(fused)),
                  float(np.linalg.norm(fused[:512])), float(np.linalg.norm(fused[512:]))))
        self.assertEqual(fused.shape, (1024,))
        self.assertAlmostEqual(float(np.linalg.norm(fused)), 1.0, places=5)
        self.assertAlmostEqual(
            float(np.linalg.norm(fused[:512])), float(np.linalg.norm(fused[512:])),
            places=5,
            msg="the two halves of a REAL fused observation do not weigh the same, so "
                "q_t leans on one modality before any retrieval has happened",
        )

    def test_an_episodes_worth_of_real_steps_fills_and_evicts(self):
        """`M^S_t` (eq. 4) driven by real observations rather than fixtures, and the
        context (eq. 19's STM half) computed over them."""
        from earshot.agent.stm import ShortTermMemory, StmEntry
        from earshot.vlm.encode import visual_embedding

        horizon = 3
        memory = ShortTermMemory(horizon=horizon)
        audio = np.zeros(512, dtype=np.float32)
        audio[7] = 1.0
        for index, frame in enumerate(self.frames):
            memory = memory.push(StmEntry(
                visual=visual_embedding(frame, self.encoder),
                audio=audio,
                pose=self.world.pose(),
                prev_action=None if index == 0 else "move_forward",
            ))
        context = memory.context(decay=0.8)
        print("  M^S after {} real steps: len {} (horizon {})   context norm {:.6f}".format(
            len(self.frames), len(memory), horizon, float(np.linalg.norm(context))))
        print("  cosine(context, latest e_t): {:.4f}".format(
            float(np.dot(context, memory.latest.fused))))
        self.assertEqual(len(memory), horizon)
        self.assertAlmostEqual(float(np.linalg.norm(context)), 1.0, places=5)

    def test_the_episode_boundary_empties_it(self):
        """The one structural guarantee `M^S` has, on real entries."""
        from earshot.agent.stm import ShortTermMemory, StmEntry
        from earshot.vlm.encode import visual_embedding

        audio = np.zeros(512, dtype=np.float32)
        audio[7] = 1.0
        memory = ShortTermMemory(horizon=4).push(StmEntry(
            visual=visual_embedding(self.frames[0], self.encoder),
            audio=audio, pose=self.world.pose(), prev_action=None,
        ))
        self.assertEqual(len(memory.reset()), 0)
        self.assertEqual(len(memory), 1, "reset emptied the memory in place")


if __name__ == "__main__":
    unittest.main(verbosity=2)
