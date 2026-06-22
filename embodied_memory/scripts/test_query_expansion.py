"""
Sanity tests for the query-side instance fix (Stage-1): ``expand_query`` in
``text_encode_util``. Pseudo-relevance feedback that refines the bare-category
query toward the first-pass recalled captions, recovering the instance gap the
category query discards (measured GREEN by diagnose_sbert_cosines.query_template_ab:
bare goal-vs-distractor rank gap -0.039 -> prior-sighting query +0.051).

The wiring in ``memory_bridge.propose_memory_candidates`` is gated on the
``LTM_QUERY_EXPANSION`` env var and SKIPS the expansion entirely when unset, so
the default path is byte-identical by construction; these tests pin the pure
helper's behaviour (numpy-only, no faiss/habitat).

Invoke with::

    python embodied_memory/scripts/test_query_expansion.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

# import the module under test without triggering the embodied_memory package
# __init__ (which pulls faiss): load text_encode_util.py directly by path.
import importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_TEU = os.path.join(_HERE, "..", "text_encode_util.py")
_spec = importlib.util.spec_from_file_location("text_encode_util", _TEU)
teu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(teu)

expand_query = teu.expand_query
cosine_sim = teu.cosine_sim


def _u(*xyz):
    v = np.asarray(xyz, dtype=np.float32)
    return v / np.linalg.norm(v)


def case_off_is_identity():
    q = _u(1.0, 0.0, 0.0)
    hits = [_u(0, 1, 0), _u(0, 1, 0)]
    for mode in (None, "", "off"):
        out = expand_query(q, hits, mode=mode)
        assert np.allclose(out, q, atol=1e-6), (mode, out)
    print("  case_off_is_identity: OK")


def case_empty_hits_is_identity():
    q = _u(1.0, 0.0, 0.0)
    assert np.allclose(expand_query(q, [], mode="prf"), q, atol=1e-6)
    assert np.allclose(expand_query(q, None, mode="prf"), q, atol=1e-6)
    # all-zero hits also degenerate to identity (no usable pseudo-relevant set)
    z = np.zeros(3, dtype=np.float32)
    assert np.allclose(expand_query(q, [z, z], mode="prf"), q, atol=1e-6)
    print("  case_empty_hits_is_identity: OK")


def case_output_is_unit_norm():
    q = _u(1.0, 0.0, 0.0)
    hits = [_u(0, 1, 0), _u(0, 0, 1)]
    for mode in ("prf", "caption"):
        out = expand_query(q, hits, mode=mode)
        assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5, (mode, np.linalg.norm(out))
    print("  case_output_is_unit_norm: OK")


def case_prf_shifts_toward_goal_cluster_increasing_gap():
    # Category query is orthogonal to both instances (no preference). The goal
    # instance's prior sightings cluster at (0,1,0); a distractor at (0,0,1).
    # PRF over the goal-cluster hits must move q' toward the goal -> open a
    # goal-vs-distractor cosine gap the category query did not have.
    q_cat = _u(1, 0, 0)
    goal, distractor = _u(0, 1, 0), _u(0, 0, 1)
    hits = [goal, goal]  # first-pass pseudo-relevant set is the goal cluster
    qp = expand_query(q_cat, hits, mode="prf", alpha=0.6, beta=0.4)
    gap_cat = cosine_sim(q_cat, goal) - cosine_sim(q_cat, distractor)
    gap_prf = cosine_sim(qp, goal) - cosine_sim(qp, distractor)
    assert cosine_sim(qp, goal) > cosine_sim(q_cat, goal), (qp,)
    assert gap_prf > gap_cat + 0.1, (gap_cat, gap_prf)
    print("  case_prf_shifts_toward_goal_cluster_increasing_gap: OK")


def case_prf_keeps_category_anchor():
    # Even when the pseudo-relevant set is a DISTRACTOR cluster, prf keeps the
    # category component (alpha) -> it cannot drift fully onto the distractor
    # (the conservative property that makes prf the safe first A/B mode).
    q_cat = _u(1, 0, 0)
    distractor = _u(0, 1, 0)
    qp = expand_query(q_cat, [distractor, distractor], mode="prf", alpha=0.6, beta=0.4)
    # retains meaningful mass on the category axis
    assert cosine_sim(qp, q_cat) > 0.5, cosine_sim(qp, q_cat)
    # caption mode (no anchor) drifts fully onto the distractor
    qc = expand_query(q_cat, [distractor, distractor], mode="caption")
    assert cosine_sim(qc, distractor) > 0.99, cosine_sim(qc, distractor)
    assert cosine_sim(qc, q_cat) < 0.01, cosine_sim(qc, q_cat)
    print("  case_prf_keeps_category_anchor: OK")


def case_caption_is_centroid_of_top_m():
    q_cat = _u(1, 0, 0)
    hits = [_u(0, 1, 0), _u(0, 1, 0), _u(0, 0, 1)]  # 3 hits
    # top_m=2 uses only the first two (both (0,1,0)) -> centroid (0,1,0)
    out = expand_query(q_cat, hits, mode="caption", top_m=2)
    assert cosine_sim(out, _u(0, 1, 0)) > 0.99, out
    # top_m=3 folds in the (0,0,1) hit -> centroid tilts off (0,1,0)
    out3 = expand_query(q_cat, hits, mode="caption", top_m=3)
    assert cosine_sim(out3, _u(0, 1, 0)) < cosine_sim(out, _u(0, 1, 0)), (out, out3)
    print("  case_caption_is_centroid_of_top_m: OK")


def main() -> int:
    print("query-expansion (Stage-1) sanity tests")
    case_off_is_identity()
    case_empty_hits_is_identity()
    case_output_is_unit_norm()
    case_prf_shifts_toward_goal_cluster_increasing_gap()
    case_prf_keeps_category_anchor()
    case_caption_is_centroid_of_top_m()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
