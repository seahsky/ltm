"""
Sanity test for ``text_encode_util.l2_normalize_encoder`` — the wrapper that
makes the embodied LTM's text encoder emit UNIT-NORM vectors.

Why it matters: the fine layer is a FAISS ``IndexFlatL2`` used as a cosine index
via ``cos = 1 - L2^2 / 2``, which is only valid for unit-normalized vectors. The
CLIP encoder normalizes; SentenceTransformer's ``.encode()`` does NOT. After the
Run-7 SBERT re-index, fine-layer vectors were non-unit, so ``propose_memory_
candidates`` computed a garbage (clamped to -1) cosine and rejected EVERY memory
candidate (n_memory_candidates=0 in the revisit-b1/b2 smokes). Normalizing at the
encoder boundary restores the invariant.

Numpy-only — runs locally without faiss/habitat/sentence-transformers.

Invoke with::

    python embodied_memory/scripts/test_text_encode_util.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Import the module directly (top-level, not via the embodied_memory package)
# so we don't trigger the package __init__'s faiss import.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import text_encode_util  # noqa: E402


def case_normalizes_to_unit_norm():
    enc = text_encode_util.l2_normalize_encoder(lambda s: np.array([3.0, 4.0]))
    v = enc("anything")
    assert np.isclose(np.linalg.norm(v), 1.0), np.linalg.norm(v)
    assert np.allclose(v, [0.6, 0.8]), v
    print("  case normalizes_to_unit_norm: OK")


def case_preserves_direction():
    raw = np.array([1.0, 2.0, 2.0])           # norm 3
    enc = text_encode_util.l2_normalize_encoder(lambda s: raw)
    v = enc("x")
    assert np.allclose(v, raw / 3.0), v
    print("  case preserves_direction: OK")


def case_zero_vector_safe():
    enc = text_encode_util.l2_normalize_encoder(lambda s: np.zeros(4))
    v = enc("x")
    assert not np.any(np.isnan(v)), v          # no divide-by-zero NaN
    assert np.allclose(v, np.zeros(4)), v
    print("  case zero_vector_safe: OK")


def case_returns_float32():
    enc = text_encode_util.l2_normalize_encoder(lambda s: [3, 4])  # list, ints
    v = enc("x")
    assert v.dtype == np.float32, v.dtype
    assert np.isclose(np.linalg.norm(v), 1.0)
    print("  case returns_float32: OK")


def main() -> int:
    print("text_encode_util.l2_normalize_encoder sanity tests")
    case_normalizes_to_unit_norm()
    case_preserves_direction()
    case_zero_vector_safe()
    case_returns_float32()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
