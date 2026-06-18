"""TDD for the Lever-1 caption-to-caption rerank head.

Lever 1 exploits the SBERT instance signal the bare category query throws away
(diagnose_sbert_cosines: within-instance caption cosine 0.628 vs
between-instance-same-category 0.535 → +0.093, but the ``"there is a {cat}"``
query collapses the instance rank gap to +0.047; a caption-to-caption pre-screen
recovers it to +0.080). Reference-free mechanism: among the recalled
same-category sightings, the WELL-OBSERVED instance (seen repeatedly → a tight
caption cluster) is more *central* than a one-off distractor sighting, so a small
additive centrality bonus prefers it. Env-gated (``LTM_CAPTION_RERANK``),
DEFAULT-OFF, so objectnav/revisit/audiogoal stay byte-identical; A/B-ablated.

Exercises the pure ``_caption_centrality_bonus`` helper (sibling of
``_temporal_recency_bonus``) with ``memory_bridge.py`` loaded under the same
faiss-free spec-load as ``test_temporal_context.py``.

    python embodied_memory/scripts/test_caption_rerank.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

_EMB_DIR = Path(__file__).resolve().parent.parent       # …/embodied_memory
_REPO = _EMB_DIR.parent


def _stub_mod(name, attrs, is_pkg=False, path=None):
    m = types.ModuleType(name)
    if is_pkg:
        m.__path__ = path or []
    for a in attrs:
        setattr(m, a, type(a, (), {}))
    sys.modules[name] = m
    return m


def _bootstrap():
    if "embodied_memory" not in sys.modules:
        pkg = types.ModuleType("embodied_memory")
        pkg.__path__ = [str(_EMB_DIR)]
        sys.modules["embodied_memory"] = pkg
    _stub_mod("dialogue_memory", [], is_pkg=True, path=[str(_REPO / "dialogue_memory")])
    _stub_mod("dialogue_memory.consolidation", ["DialogueConsolidation", "DialogueSegment"])
    _stub_mod("dialogue_memory.ltm", ["HierarchicalLTM", "MemoryEntry"])
    _stub_mod("dialogue_memory.pattern_cluster", ["MidLayerMemory", "PatternClusterer"])
    _stub_mod("dialogue_memory.reranking",
              ["CoherenceScorer", "HistorySuccessScorer", "MemorySimilarityScorer",
               "RerankingResult", "ResponseReranker", "Scorer", "ScoredResponse"])
    _stub_mod("embodied_memory.frontier_planner", ["FrontierCandidate"])
    _stub_mod("embodied_memory.perception", ["Keyframe"])
    rr = types.ModuleType("embodied_memory.room_resolver")
    rr.preferred_room = lambda *a, **k: None
    rr.resolve_room = lambda *a, **k: None
    sys.modules["embodied_memory.room_resolver"] = rr
    te = types.ModuleType("embodied_memory.text_encode_util")
    te.cosine_sim = lambda a, b: 0.0
    sys.modules["embodied_memory.text_encode_util"] = te

    spec = importlib.util.spec_from_file_location(
        "embodied_memory.memory_bridge", str(_EMB_DIR / "memory_bridge.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["embodied_memory.memory_bridge"] = mod
    spec.loader.exec_module(mod)
    return mod


mb = _bootstrap()
cent = mb._caption_centrality_bonus


# ----------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------


def _v(*xs):
    return np.asarray(xs, dtype=np.float32)


def case_weight_zero_is_noop():
    embs = [_v(1, 0, 0), _v(0, 1, 0)]
    assert cent(embs, 0.0) == [0.0, 0.0]
    print("  case weight_zero_is_noop: OK")


def case_negative_weight_is_noop():
    assert cent([_v(1, 0), _v(0, 1)], -0.3) == [0.0, 0.0]
    print("  case negative_weight_is_noop: OK")


def case_too_few_candidates_is_noop():
    # <2 candidates → no caption-to-caption structure → no-op.
    assert cent([_v(1, 0, 0)], 0.1) == [0.0]
    assert cent([], 0.1) == []
    print("  case too_few_candidates_is_noop: OK")


def case_no_spread_is_noop():
    # All identical embeddings → every candidate equally central → no spread →
    # all-zero (the head never invents a signal where centrality is uniform).
    embs = [_v(1, 0, 0), _v(1, 0, 0), _v(1, 0, 0)]
    assert cent(embs, 0.1) == [0.0, 0.0, 0.0]
    print("  case no_spread_is_noop: OK")


def case_central_boosted_outlier_demoted():
    # v0,v1 are a tight cluster (cos 0.8 — the well-observed instance seen twice);
    # v2 is orthogonal (a one-off distractor). ZERO-SUM: the clustered pair is
    # BOOSTED (>0), the outlier DEMOTED (<0), and the bonuses sum to ~0 (no
    # memory-mass inflation). cent=[0.4,0.4,0]→centered=[.133,.133,-.267]→
    # /max=.267, ×0.1 → [+0.05,+0.05,-0.1].
    v0, v1, v2 = _v(1, 0, 0), _v(0.8, 0.6, 0), _v(0, 0, 1)
    b = cent([v0, v1, v2], 0.1)
    assert abs(b[0] - 0.05) < 1e-6, b
    assert abs(b[1] - 0.05) < 1e-6, b
    assert abs(b[2] + 0.10) < 1e-6, b      # outlier demoted (negative)
    assert b[0] > 0 and b[1] > 0 and b[2] < 0, b
    assert abs(sum(b)) < 1e-6, b           # zero-sum
    print("  case central_boosted_outlier_demoted: OK")


def case_zero_sum_invariant():
    # The load-bearing contract: bonuses sum to ~0 for any non-degenerate input,
    # so the head re-orders WITHIN memory without inflating memory-vs-frontier
    # mass (this is what kills the +84% over-fire).
    for embs in ([_v(1, 0, 0), _v(0.9, 0.4, 0), _v(0, 1, 0)],
                 [_v(1, 0, 0), _v(0.8, 0.6, 0), _v(0, 0, 1), _v(0.1, 0.1, 0.99)]):
        b = cent(embs, 0.07)
        assert abs(sum(b)) < 1e-6, (embs, b)
        assert all(abs(x) <= 0.07 + 1e-6 for x in b), b   # bounded by weight
    print("  case zero_sum_invariant: OK")


def case_normalizes_unnormalized_embeddings():
    # Scaling an embedding must not change its cosine-derived centrality.
    v0, v1, v2 = _v(2, 0, 0), _v(0.8, 0.6, 0), _v(0, 0, 5)
    b = cent([v0, v1, v2], 0.1)
    assert abs(b[0] - 0.05) < 1e-6, b
    assert abs(b[2] + 0.10) < 1e-6, b
    assert abs(sum(b)) < 1e-6, b
    print("  case normalizes_unnormalized_embeddings: OK")


def case_zero_vector_is_safe():
    # A zero-norm embedding must not crash or NaN; it is treated as maximally
    # un-central (cos 0 with everything) → most DEMOTED (most negative), finite.
    v0, v1, vz = _v(1, 0, 0), _v(0.8, 0.6, 0), _v(0, 0, 0)
    b = cent([v0, v1, vz], 0.1)
    assert all(np.isfinite(x) for x in b), b
    assert b[2] < 0, b                      # zero vector is least central → demoted
    assert abs(sum(b)) < 1e-6, b
    print("  case zero_vector_is_safe: OK")


def main() -> int:
    print("Lever-1 caption-to-caption rerank head tests")
    case_weight_zero_is_noop()
    case_negative_weight_is_noop()
    case_too_few_candidates_is_noop()
    case_no_spread_is_noop()
    case_central_boosted_outlier_demoted()
    case_zero_sum_invariant()
    case_normalizes_unnormalized_embeddings()
    case_zero_vector_is_safe()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
