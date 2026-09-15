#!/usr/bin/env python3
"""``f_v`` against the real CLIP. V100 + ``ss2``.

    conda activate ss2
    python -m earshot.task.models          # stages CLAP and CLIP
    bash earshot/tools/box_gate.sh

**These tests print their measurements** (ADR-0014), and one of them decides whether
DREAM is affordable at all.

Three things a Mac cannot settle:

1. **THE PER-FRAME COST, which is the gating number for everything after this PR.**
   CLAP fires ONCE per onset. DREAM's `f_v` fires on EVERY step (eq. 5), so it lands
   inside criterion 7's 0.5 s per-step ceiling on all ~180 steps of every episode.
   `matrix-2` measured a worst step of 0.3278-0.3356 s across its four arms, so the
   headroom is about 0.16 s and it is not generous. If a CLIP forward costs more than
   that, the STM cannot be filled every step and §III.C needs a stride -- which is a
   change to the method, not a tuning knob, and it has to be decided from a measurement
   rather than a guess. This test prints the number and fails only against the ceiling,
   because what to do at 0.05 s and at 0.4 s are different decisions and both are Sky's.

2. **The embedding width.** `CLIP_MODEL_ID`'s comment claims 512-d and says the balance
   with CLAP's 512-d is why that checkpoint was chosen -- an unbalanced `f_fuse` (eq. 7)
   makes `q_t` a visual query wearing an audio hat, and the `omega_t` shift the paper's
   central hypothesis is about would be measuring the fuse rather than the memory. That
   claim is asserted here against the real model rather than trusted from a docstring.

3. **That the staged checkpoint loads at all**, under the box's torch 2.2.2+cu118 pin.
   `load_clip_encoder`'s whole error message is about transformers substituting a
   `DummyObject` that imports cleanly and raises on construction, and the only way to
   know it does not is to construct one.

Python 3.9 (the SoundSpaces pin).
"""

from __future__ import annotations

import time
import unittest

import numpy as np

# The ceiling criterion 7 audits, and the headroom `matrix-2` actually left under it.
PER_STEP_CEILING_S = 0.5
MEASURED_WORST_STEP_S = 0.3356
HEADROOM_S = PER_STEP_CEILING_S - MEASURED_WORST_STEP_S

# habitat-lab's ObjectNav HM3D benchmark frame, which is what `camera_sensor_specs`
# builds and therefore what `f_v` will actually be handed.
FRAME_SHAPE = (480, 640, 3)

N_WARMUP = 3
N_TIMED = 20


def _frame(seed: int = 0):
    """A deterministic pseudo-render. Content does not matter for cost or width."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=FRAME_SHAPE, dtype=np.uint8)


class TestClipEncoderOnTheBox(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from earshot.task.models import CLIP_MODEL_ID, load_clip_encoder, resolve_clip_source

        cls.source = resolve_clip_source(CLIP_MODEL_ID)
        print("\n  CLIP source: {}".format(cls.source))
        if cls.source == CLIP_MODEL_ID:
            print("  NOTE: not staged locally — `python -m earshot.task.models` stages it.")
        cls.encoder = load_clip_encoder()
        print("  device: {}".format(cls.encoder.device))

    def test_the_embedding_is_512_wide_and_unit_norm(self):
        """`CLIP_MODEL_ID`'s comment rests on this and `f_fuse` breaks quietly without it."""
        from earshot.vlm.encode import visual_embedding

        out = visual_embedding(_frame(1), self.encoder)
        print("  embedding: shape {} dtype {} norm {:.6f}".format(
            out.shape, out.dtype, float(np.linalg.norm(out))))
        self.assertEqual(out.ndim, 1)
        self.assertEqual(
            out.shape[0], 512,
            "f_fuse (eq. 7) concatenates this with CLAP's 512-d audio embedding; a "
            "different width makes the fuse lopsided and CLIP_MODEL_ID's comment wrong",
        )
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)

    def test_two_different_frames_give_two_different_embeddings(self):
        """The arm that proves the model is doing something. A frozen or mis-loaded
        checkpoint returns the same vector for everything, which would make every `M^E`
        row identical and every retrieval a tie -- and nothing downstream would say so."""
        from earshot.vlm.encode import visual_embedding

        a = visual_embedding(_frame(1), self.encoder)
        b = visual_embedding(_frame(2), self.encoder)
        cosine = float(np.dot(a, b))
        print("  cosine(frame 1, frame 2): {:.4f}".format(cosine))
        self.assertLess(
            cosine, 0.999,
            "two different frames encoded to the same vector — the checkpoint did not "
            "load, or the weights are random",
        )

    def test_the_same_frame_encodes_to_the_same_vector(self):
        """`eval()` and `no_grad()` doing their job. Dropout left on would make the store
        answer a slightly different question every time it was read."""
        from earshot.vlm.encode import visual_embedding

        frame = _frame(3)
        a = visual_embedding(frame, self.encoder)
        b = visual_embedding(frame, self.encoder)
        drift = float(np.max(np.abs(a - b)))
        print("  max drift over two encodes of one frame: {:.3e}".format(drift))
        self.assertLess(drift, 1e-5)

    def test_a_four_channel_frame_is_accepted(self):
        """habitat-sim can hand back RGBA. `ClipEncoder.encode_image` drops alpha so a
        caller that forgot gets a shape error on the first frame rather than a resize
        failure several frames into an episode."""
        from earshot.vlm.encode import visual_embedding

        rgba = np.concatenate(
            [_frame(4), np.full(FRAME_SHAPE[:2] + (1,), 255, dtype=np.uint8)], axis=-1
        )
        self.assertEqual(visual_embedding(rgba, self.encoder).shape, (512,))

    def test_the_per_frame_cost_fits_inside_criterion_7s_headroom(self):
        """THE NUMBER THIS FILE EXISTS FOR."""
        from earshot.vlm.encode import visual_embedding

        for index in range(N_WARMUP):
            visual_embedding(_frame(100 + index), self.encoder)

        samples = []
        for index in range(N_TIMED):
            frame = _frame(200 + index)
            started = time.time()
            visual_embedding(frame, self.encoder)
            samples.append(time.time() - started)

        samples.sort()
        mean = sum(samples) / len(samples)
        median = samples[len(samples) // 2]
        worst = samples[-1]
        print("  f_v cost over {} frames at {}: mean {:.4f} s  median {:.4f} s  "
              "worst {:.4f} s".format(N_TIMED, FRAME_SHAPE, mean, median, worst))
        print("  criterion 7 ceiling {:.3f} s, matrix-2 worst step {:.4f} s, "
              "headroom {:.4f} s".format(
                  PER_STEP_CEILING_S, MEASURED_WORST_STEP_S, HEADROOM_S))
        print("  -> a step that also encodes a frame would cost about {:.4f} s "
              "({:.0f}% of the ceiling)".format(
                  MEASURED_WORST_STEP_S + worst,
                  100.0 * (MEASURED_WORST_STEP_S + worst) / PER_STEP_CEILING_S))
        self.assertLess(
            MEASURED_WORST_STEP_S + worst, PER_STEP_CEILING_S,
            "f_v on every step would breach criterion 7's {:.3f} s ceiling. DREAM's "
            "§III.C needs a stride, which is a change to the method rather than a "
            "tuning knob".format(PER_STEP_CEILING_S),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
