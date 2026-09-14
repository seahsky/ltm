"""`f_v` (DREAM eq. 6), and the staging that keeps it from silently becoming a wrong model.

The tree had `f_u` and no `f_v`. `earshot/vlm/` was a name in ADR-0013's layer table with
no package under it, `RoomLabeler` had only `NullRoomLabeler`, and RGB reached `runner.py`
to be consumed by nothing that produced a vector. Four of DREAM's equations are blocked on
that: the STM entry (5), the fused observation (7), the episodic row's `h^av` (15), and the
retrieval query (19).

Two things are tested here and they fail differently:

  * `visual_embedding` is the ONE path a visual vector is produced through, for the reason
    `audio_embedding` is public -- consolidation writes these into `M^E` and retrieval
    queries `M^E` with them, and a memory learned under one path and queried under another
    answers a different question with no symptom. Its guards are about a render that
    failed, which CLIP would happily embed.
  * `resolve_clip_source` decides WHICH checkpoint the run used. Its identity check is
    incident-driven: the first `resolve_clap_source` returned the staged directory for any
    `model_id`, so `env_check`'s forced-failure arm asked for a model that does not exist,
    got the real checkpoint, and reported PASS. CLIP now shares that code rather than
    carrying a second copy of the bug.

No torch here. These run on a Mac against a three-line double, which is the whole reason
`vlm/` holds no model and no device.
"""

import json
import os
import pathlib
import tempfile
import unittest

import numpy as np

from _interpreter import assert_interpreter  # noqa: F401

from earshot.task.models import (
    CLIP_LOCAL_DIR,
    CLIP_MODEL_ID,
    STAGED_MARKER,
    resolve_clip_source,
)
from earshot.vlm.encode import VisualEmbeddingError, visual_embedding


class _Encoder:
    """The whole protocol: one call, an array out. Records what it was handed."""

    def __init__(self, vector=(3.0, 4.0)):
        self.vector = vector
        self.seen = None

    def encode_image(self, frame):
        self.seen = np.asarray(frame)
        return np.asarray(self.vector, dtype=np.float32)


def _frame(value=7, shape=(4, 4, 3)):
    return np.full(shape, value, dtype=np.uint8)


class TestVisualEmbedding(unittest.TestCase):
    def test_the_embedding_is_unit_norm(self):
        """`f_fuse` (eq. 7) concatenates this with CLAP's, and `M^K` is queried by cosine.
        A vector that is not unit-norm makes the fuse lopsided by whatever the encoder's
        scale happens to be that step."""
        out = visual_embedding(_frame(), _Encoder((3.0, 4.0)))
        self.assertAlmostEqual(float(np.linalg.norm(out)), 1.0, places=5)
        np.testing.assert_allclose(out, np.array([0.6, 0.8], dtype=np.float32), atol=1e-6)

    def test_it_is_float32_and_one_dimensional(self):
        out = visual_embedding(_frame(), _Encoder(((1.0, 0.0), (0.0, 1.0))))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)

    def test_the_width_is_the_encoders_own_and_is_never_reshaped(self):
        """`audio_embedding` says the same and for the same reason: a differently sized
        encoder must raise where the vector is STORED rather than broadcast into a table
        it does not belong in."""
        self.assertEqual(visual_embedding(_frame(), _Encoder(tuple(range(7)))).shape, (7,))
        self.assertEqual(visual_embedding(_frame(), _Encoder(tuple(range(512)))).shape, (512,))

    def test_the_frame_reaches_the_encoder_unmodified(self):
        encoder = _Encoder()
        frame = _frame(value=13)
        visual_embedding(frame, encoder)
        np.testing.assert_array_equal(encoder.seen, frame)

    def test_an_empty_frame_raises(self):
        """An empty render is a failed render, and its embedding would be a row in `M^E`
        for an observation that never happened."""
        with self.assertRaises(VisualEmbeddingError) as caught:
            visual_embedding(np.zeros((0, 4, 3), dtype=np.uint8), _Encoder())
        self.assertIn("empty", str(caught.exception))

    def test_a_non_finite_frame_raises(self):
        """THE FORCED-FAILURE ARM THAT MATTERS. habitat-sim hands back a rendered array,
        and a NaN in it means the render failed -- CLIP returns a perfectly ordinary
        vector for it and nothing downstream can tell."""
        frame = np.ones((4, 4, 3), dtype=np.float32)
        frame[2, 2, 1] = np.nan
        with self.assertRaises(VisualEmbeddingError) as caught:
            visual_embedding(frame, _Encoder())
        self.assertIn("non-finite", str(caught.exception))
        self.assertIn("1 non-finite value(s) of 48", str(caught.exception))

    def test_an_infinite_frame_raises_too(self):
        frame = np.ones((2, 2, 3), dtype=np.float32)
        frame[0, 0, 0] = np.inf
        with self.assertRaises(VisualEmbeddingError):
            visual_embedding(frame, _Encoder())

    def test_a_uint8_frame_is_not_checked_for_finiteness(self):
        """The healthy path of the same guard: `np.isfinite` is undefined on some dtypes
        and every real habitat-sim frame is `uint8`, so the kind is checked before the
        values are. A guard that raised on the normal case would fail every episode."""
        self.assertEqual(visual_embedding(_frame(), _Encoder()).shape, (2,))

    def test_an_empty_encoder_output_raises(self):
        with self.assertRaises(VisualEmbeddingError) as caught:
            visual_embedding(_frame(), _Encoder(()))
        self.assertIn("empty vector", str(caught.exception))

    def test_a_non_finite_encoder_output_on_a_finite_frame_raises(self):
        """Separate message from the frame guard, because the fault is a different
        component: the render was fine and the MODEL produced nonsense."""
        with self.assertRaises(VisualEmbeddingError) as caught:
            visual_embedding(_frame(), _Encoder((1.0, np.nan)))
        self.assertIn("the model failing rather than the render", str(caught.exception))


class TestResolveClipSource(unittest.TestCase):
    """Which checkpoint the run actually used."""

    def _stage(self, tmp, *, model_id=CLIP_MODEL_ID, omit=(), marker=True):
        root = pathlib.Path(tmp) / "clip"
        root.mkdir()
        for name in (
            "model.safetensors", "config.json",
            "preprocessor_config.json", "tokenizer.json",
        ):
            if name not in omit:
                (root / name).write_text("{}", encoding="utf-8")
        if marker:
            (root / STAGED_MARKER).write_text(
                json.dumps({"model_id": model_id}), encoding="utf-8"
            )
        return str(root)

    def test_a_complete_staged_copy_that_names_the_model_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = self._stage(tmp)
            self.assertEqual(resolve_clip_source(CLIP_MODEL_ID, local), local)

    def test_a_directory_that_names_a_DIFFERENT_model_is_not_used(self):
        """THE INCIDENT ARM. `resolve_clap_source`'s first version returned the staged
        directory for any `model_id`, so a probe asking for a model that does not exist
        got the real checkpoint and reported PASS. CLIP shares that code now, so this
        asserts the shared fix rather than a second copy of it."""
        with tempfile.TemporaryDirectory() as tmp:
            local = self._stage(tmp, model_id="someone/else")
            self.assertEqual(
                resolve_clip_source(CLIP_MODEL_ID, local), CLIP_MODEL_ID,
                "a directory that says it holds another model was used anyway",
            )

    def test_a_directory_with_no_marker_cannot_say_what_it_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = self._stage(tmp, marker=False)
            self.assertEqual(resolve_clip_source(CLIP_MODEL_ID, local), CLIP_MODEL_ID)

    def test_a_half_written_conversion_is_not_preferred_over_the_hub(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = self._stage(tmp, omit=("model.safetensors",))
            self.assertEqual(resolve_clip_source(CLIP_MODEL_ID, local), CLIP_MODEL_ID)

    def test_an_unparseable_marker_is_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            local = self._stage(tmp)
            (pathlib.Path(local) / STAGED_MARKER).write_text("{not json", encoding="utf-8")
            self.assertEqual(resolve_clip_source(CLIP_MODEL_ID, local), CLIP_MODEL_ID)

    def test_nothing_staged_returns_the_hub_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                resolve_clip_source(CLIP_MODEL_ID, os.path.join(tmp, "nope")),
                CLIP_MODEL_ID,
            )

    def test_clip_and_clap_do_not_stage_to_the_same_directory(self):
        """They are two checkpoints and `STAGED_MARKER` is one filename. Sharing a
        directory would make each stager overwrite the other's identity claim, and
        whichever ran last would answer for both."""
        from earshot.task.models import CLAP_LOCAL_DIR

        self.assertNotEqual(CLIP_LOCAL_DIR, CLAP_LOCAL_DIR)


class TestStagingIsWiredIntoTheCli(unittest.TestCase):
    """`python -m earshot.task.models` has to stage BOTH now.

    A sweep that staged only CLAP gets through the env probe and dies inside its first
    episode, which is the shape of failure `matrix-2` spent three runs on.
    """

    def test_the_parser_offers_both_models_and_a_clap_only_escape(self):
        import inspect

        from earshot.task import models

        source = inspect.getsource(models._main)
        for flag in ("--clip-model-id", "--clip-out-dir", "--clap-only"):
            self.assertIn(flag, source)

    def test_the_default_path_stages_clip(self):
        """The forced-failure arm for the wiring: if `stage_clip_safetensors` is not
        called outside the `--clap-only` branch, the default invocation silently keeps the
        old behaviour and every DREAM run fails on the box."""
        import inspect

        from earshot.task import models

        source = inspect.getsource(models._main)
        self.assertIn("if not args.clap_only:", source)
        self.assertLess(
            source.index("if not args.clap_only:"),
            source.index("stage_clip_safetensors(args.clip_model_id"),
            "CLIP is staged outside the --clap-only guard",
        )


if __name__ == "__main__":
    unittest.main()
