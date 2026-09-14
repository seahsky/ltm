"""``f_v``: one RGB frame to one unit embedding, and the guards that keep it honest.

The deliberate mirror of `audio/clap.py`'s `audio_embedding`, down to the reason it is
public: two different processes have to agree on the vector. DREAM's consolidation writes
``h^av`` into long-term memory (eq. 15) and retrieval queries that memory at run time
(eq. 19-22); a memory learned under one encoder path and queried under another answers a
different question with no symptom. One path, and it is this function.

**Width is the encoder's own and is never reshaped.** `audio_embedding` says the same and
for the same reason: a differently sized encoder must raise where the vector is stored,
not broadcast silently into a table it does not belong in. `f_fuse` (eq. 7) concatenates
these with CLAP's, so a width change shows up as a shape error at the fuse rather than as
a quietly wrong cosine six equations later.

**A frame that is not finite RAISES.** habitat-sim hands back a rendered array, and a NaN
in it means the render failed; CLIP would happily return a vector for it, and that vector
would be a row in `M^E` recording an observation that never happened. This repo has found
that class of silent fabrication more than once, so it is checked here rather than
downstream.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "VISUAL_EMBEDDING_ERROR",
    "VisualEmbeddingError",
    "visual_embedding",
]

# Named so `task/` can quote it in a diagnosis without importing the exception type into a
# layer that does not need it.
VISUAL_EMBEDDING_ERROR = "the RGB frame could not be encoded"


class VisualEmbeddingError(ValueError):
    """A frame that cannot honestly become an embedding.

    A `ValueError` rather than a bespoke hierarchy: the caller's only two choices are to
    end the episode or to record the failure, and both read the message.
    """


def _unit(vector: Any) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    return values / (float(np.linalg.norm(values)) + 1e-8)


def visual_embedding(frame: Any, encoder: Any) -> np.ndarray:
    """``z^v_t = f_v(v_t)`` (eq. 6): the unit image embedding for one RGB frame.

    `encoder` satisfies one call, ``encode_image(frame) -> array``; `task/models.py`'s
    `ClipEncoder` is the connector that provides it, and a test double is three lines.
    Injected rather than constructed here so this module holds no model and no device
    (ADR-0013: ``"vlm": ()``).

    Returns a 1-D ``float32`` of the encoder's own width.
    """
    values = np.asarray(frame)
    if values.size == 0:
        raise VisualEmbeddingError(
            "{}: the frame is empty (shape {}). An empty render is a failed render, and "
            "an embedding of it would be a row in M^E for an observation that never "
            "happened".format(VISUAL_EMBEDDING_ERROR, values.shape)
        )
    # `np.isfinite` is undefined on the object dtype habitat-sim never returns but a test
    # double might, so the kind is checked before the values are.
    if values.dtype.kind in "fc" and not bool(np.isfinite(values).all()):
        raise VisualEmbeddingError(
            "{}: the frame carries {} non-finite value(s) of {}. CLIP would return a "
            "vector for it and nothing downstream could tell that the render "
            "failed".format(
                VISUAL_EMBEDDING_ERROR,
                int((~np.isfinite(values)).sum()),
                values.size,
            )
        )
    embedding = _unit(encoder.encode_image(values))
    if embedding.size == 0:
        raise VisualEmbeddingError(
            "{}: the encoder returned an empty vector. A 0-length embedding cannot be a "
            "row in any store and must not reach one".format(VISUAL_EMBEDDING_ERROR)
        )
    if not bool(np.isfinite(embedding).all()):
        raise VisualEmbeddingError(
            "{}: the encoder returned a non-finite vector for a finite frame, which is "
            "the model failing rather than the render".format(VISUAL_EMBEDDING_ERROR)
        )
    return embedding
