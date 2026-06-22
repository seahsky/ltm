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


def cosine_sim(a, b) -> float:
    """Raw cosine similarity in [-1, 1], normalized at comparison time so it's
    invariant to either input's magnitude. Returns 0.0 if either vector is
    ~zero. Used by propose_memory_candidates instead of the `1 - L2^2/2` index
    shortcut, which silently under-reports when a side isn't unit-norm."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _unit(v: "np.ndarray") -> "np.ndarray":
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def expand_query(
    q_cat,
    hit_embeddings,
    mode: str = "prf",
    alpha: float = 0.6,
    beta: float = 0.4,
    top_m: int = 3,
) -> "np.ndarray":
    """Query-side instance fix (Stage-1): refine the bare-category query toward
    the first-pass recalled caption embeddings (the agent's OWN prior sightings),
    recovering the instance gap the category query discards.

    This is reference-free pseudo-relevance feedback (Rocchio): the LTM's own
    top-``top_m`` first-pass hits are the pseudo-relevant set, and the returned
    ``q'`` is re-used to re-query the fine layer. Modes:

    * ``prf`` — ``q' = unit(alpha * unit(q_cat) + beta * unit(mean(top_m hits)))``;
      keeps the category anchor (conservative — cannot drift fully to a
      distractor cluster), the safe default for a first live A/B.
    * ``caption`` — ``q' = unit(mean(top_m hits))``; queries purely with the
      recalled captions (the diagnostic's strongest variant, but more exposed to
      a wrong-instance top-``top_m``).

    Always returns a unit vector (the FAISS ``1 - L2^2/2`` cosine read requires
    it). ``mode`` falsy/``off`` or an empty/all-zero hit set returns ``unit(q_cat)``
    unchanged — so the default-OFF path is byte-identical.
    """
    qn = _unit(q_cat)
    if mode in (None, "", "off") or hit_embeddings is None or len(hit_embeddings) == 0:
        return qn
    embs = [_unit(np.asarray(e, dtype=np.float32)) for e in list(hit_embeddings)[:top_m]]
    embs = [e for e in embs if float(np.linalg.norm(e)) > 1e-8]
    if not embs:
        return qn
    centroid = _unit(np.mean(embs, axis=0))
    if mode == "caption":
        qp = centroid
    else:  # prf (default)
        qp = alpha * qn + beta * centroid
    return _unit(qp.astype(np.float32))
