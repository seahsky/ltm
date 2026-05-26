"""
Text-encoder helpers for the embodied LTM.

The fine layer is a FAISS ``IndexFlatL2`` consumed as a cosine index via
``cos = 1 - L2^2 / 2`` (see ``memory_bridge.propose_memory_candidates``), which
holds ONLY for unit-normalized vectors. CLIP's ``encode_text`` normalizes;
SentenceTransformer's ``.encode()`` does not, so after the SBERT re-index the
fine-layer vectors were non-unit and that cosine clamped to -1 — every memory
candidate was rejected. ``l2_normalize_encoder`` wraps an encode_fn so its output
is always unit-norm, restoring the invariant.

Numpy-only and free of relative imports so it unit-tests without faiss/habitat.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def l2_normalize_encoder(
    encode_fn: Callable[[str], "np.ndarray"]
) -> Callable[[str], "np.ndarray"]:
    """Wrap ``encode_fn`` so its output is L2-normalized float32. A zero vector
    is returned unchanged (no divide-by-zero NaN)."""

    def _encode(text: str) -> np.ndarray:
        v = np.asarray(encode_fn(text), dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    return _encode
