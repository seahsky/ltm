"""
Sanity tests for the instance-separability extension of
``diagnose_sbert_cosines`` (step 1 of the diagnose-first program).

The category-vs-noncategory separation the original script measured is the
*wrong axis* for the "instance discrimination is THE bottleneck" claim. The
mechanism in ``memory_bridge.propose_memory_candidates`` ranks stored captions
by cosine to a bare category query (``"there is a {cat}"``); if two *different
physical instances* of the same category produce near-identical query cosines,
retrieval cannot prefer the goal instance over a distractor instance. These
tests exercise the pure functions that quantify that, with a deterministic fake
encoder so the arithmetic is checkable (no SBERT / GPU needed).

Invoke with::

    python embodied_memory/scripts/test_diagnose_sbert_cosines.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import diagnose_sbert_cosines as ds  # noqa: E402


# ----------------------------------------------------------------------
# fake encoder — maps known strings to fixed unit vectors, default elsewhere
# ----------------------------------------------------------------------


def _fake_encode(mapping, default=(0.0, 0.0, 1.0)):
    dv = np.asarray(default, dtype=np.float32)
    table = {k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()}

    def enc(s: str) -> np.ndarray:
        return table.get(s, dv)

    return enc


# ----------------------------------------------------------------------
# pairwise_cosines — within-group vs between-group cosine lists
# ----------------------------------------------------------------------


def case_pairwise_within_and_between():
    enc = _fake_encode({
        "a1": (1, 0, 0), "a2": (1, 0, 0),   # instance A, identical -> within cos 1
        "b1": (0, 1, 0),                      # instance B, orthogonal -> between cos 0
    })
    within, between = ds.pairwise_cosines([["a1", "a2"], ["b1"]], enc)
    assert within == [1.0], within
    assert between == [0.0, 0.0], between  # a1-b1, a2-b1
    print("  case pairwise_within_and_between: OK")


def case_pairwise_skips_singletons():
    # a group with one caption contributes no within pairs
    enc = _fake_encode({"x": (1, 0, 0), "y": (0, 1, 0)})
    within, between = ds.pairwise_cosines([["x"], ["y"]], enc)
    assert within == [], within
    assert between == [0.0], between
    print("  case pairwise_skips_singletons: OK")


# ----------------------------------------------------------------------
# instance_separability — within-instance vs between-instance(same category)
# ----------------------------------------------------------------------


def case_separability_orthogonal_instances_separate():
    # two chair instances on orthogonal axes: within=1, between=0 -> sep=1
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0),
        "cB1": (0, 1, 0), "cB2": (0, 1, 0),
    })
    corpus = {"chair": [["cA1", "cA2"], ["cB1", "cB2"]]}
    stats = ds.instance_separability(corpus, enc)
    assert abs(stats["within_mean"] - 1.0) < 1e-6, stats
    assert abs(stats["between_mean"] - 0.0) < 1e-6, stats
    assert abs(stats["separation"] - 1.0) < 1e-6, stats
    print("  case separability_orthogonal_instances_separate: OK")


def case_separability_identical_instances_overlap():
    # two chair instances mapped to the SAME vector: within=between=1 -> sep=0
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0),
        "cB1": (1, 0, 0), "cB2": (1, 0, 0),
    })
    corpus = {"chair": [["cA1", "cA2"], ["cB1", "cB2"]]}
    stats = ds.instance_separability(corpus, enc)
    assert abs(stats["separation"] - 0.0) < 1e-6, stats
    print("  case separability_identical_instances_overlap: OK")


def case_separability_per_category_breakdown():
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0),
        "cB1": (0, 1, 0), "cB2": (0, 1, 0),
        "bA1": (1, 0, 0), "bA2": (1, 0, 0),   # bed instance A == chair A vector (irrelevant cross-cat)
        "bB1": (1, 0, 0), "bB2": (1, 0, 0),   # bed instance B == A -> bed sep 0
    })
    corpus = {
        "chair": [["cA1", "cA2"], ["cB1", "cB2"]],
        "bed": [["bA1", "bA2"], ["bB1", "bB2"]],
    }
    stats = ds.instance_separability(corpus, enc)
    assert abs(stats["per_category"]["chair"]["separation"] - 1.0) < 1e-6, stats
    assert abs(stats["per_category"]["bed"]["separation"] - 0.0) < 1e-6, stats
    print("  case separability_per_category_breakdown: OK")


# ----------------------------------------------------------------------
# goal_query_rank_gap — can the category query rank goal above distractor?
# ----------------------------------------------------------------------


def case_rank_gap_zero_when_instances_tie():
    # query "there is a chair" equidistant from both instances -> gap 0
    enc = _fake_encode({
        "there is a chair": (1, 0, 0),
        "cA1": (1, 0, 0), "cB1": (1, 0, 0),   # both perfectly match query -> tie
    })
    out = ds.goal_query_rank_gap({"chair": [["cA1"], ["cB1"]]}, enc, template="there is a {}")
    assert abs(out["per_category"]["chair"]["rank_gap"] - 0.0) < 1e-6, out
    print("  case rank_gap_zero_when_instances_tie: OK")


def case_rank_gap_positive_when_instances_differ():
    # instance A aligns with the query, instance B is orthogonal -> gap 1
    enc = _fake_encode({
        "there is a chair": (1, 0, 0),
        "cA1": (1, 0, 0), "cB1": (0, 1, 0),
    })
    out = ds.goal_query_rank_gap({"chair": [["cA1"], ["cB1"]]}, enc, template="there is a {}")
    assert abs(out["per_category"]["chair"]["rank_gap"] - 1.0) < 1e-6, out
    print("  case rank_gap_positive_when_instances_differ: OK")


# ----------------------------------------------------------------------
# caption_to_caption_rank_gap — does ranking by a prior-sighting caption
# (not the bare category query) recover the instance gap? (Lever 1 pre-screen)
# ----------------------------------------------------------------------


def case_c2c_rank_gap_orthogonal_instances_high():
    # Query with a held-out caption of the goal instance. Goal's other captions
    # are identical to the query (cos 1); the distractor instance is orthogonal
    # (cos 0) -> gap 1.0. The category query gave 0 here when both instances tie.
    enc = _fake_encode({
        "a1": (1, 0, 0), "a2": (1, 0, 0),   # instance A (the goal)
        "b1": (0, 1, 0), "b2": (0, 1, 0),   # instance B (distractor)
    })
    out = ds.caption_to_caption_rank_gap({"chair": [["a1", "a2"], ["b1", "b2"]]}, enc)
    assert abs(out["per_category"]["chair"]["rank_gap"] - 1.0) < 1e-6, out
    assert out["per_category"]["chair"]["n_samples"] > 0, out
    print("  case c2c_rank_gap_orthogonal_instances_high: OK")


def case_c2c_rank_gap_identical_instances_zero():
    # Goal and distractor captions identical -> query can't separate -> gap 0.
    enc = _fake_encode({
        "a1": (1, 0, 0), "a2": (1, 0, 0),
        "b1": (1, 0, 0), "b2": (1, 0, 0),
    })
    out = ds.caption_to_caption_rank_gap({"chair": [["a1", "a2"], ["b1", "b2"]]}, enc)
    assert abs(out["per_category"]["chair"]["rank_gap"] - 0.0) < 1e-6, out
    print("  case c2c_rank_gap_identical_instances_zero: OK")


def case_c2c_rank_gap_skips_singleton_goal():
    # A goal instance with a single caption has no held-out reference -> it
    # contributes no samples; only the 2-caption instance can be the goal.
    enc = _fake_encode({
        "a1": (1, 0, 0), "a2": (1, 0, 0),   # 2-caption goal -> contributes
        "b1": (0, 1, 0),                      # singleton -> cannot be goal
    })
    out = ds.caption_to_caption_rank_gap({"chair": [["a1", "a2"], ["b1"]]}, enc)
    # one goal instance usable, 2 query choices (a1, a2) -> 2 samples
    assert out["per_category"]["chair"]["n_samples"] == 2, out
    assert abs(out["per_category"]["chair"]["rank_gap"] - 1.0) < 1e-6, out
    print("  case c2c_rank_gap_skips_singleton_goal: OK")


# ----------------------------------------------------------------------
# instance_verdict — the plan's decision rule
# ----------------------------------------------------------------------


def case_verdict_overlap_blames_embedding():
    # captions DON'T separate at all -> embedding is the bottleneck (detector)
    v = ds.instance_verdict(separation=0.02, rank_gap=0.01, sep_threshold=0.05)
    assert v.startswith("OVERLAP"), v
    assert "bottleneck" in v.lower(), v
    assert "do not train" not in v.lower(), v
    print("  case verdict_overlap_blames_embedding: OK")


def case_verdict_mixed_signal_exists_query_collapses():
    # captions DO separate (sep > threshold) but the bare category query collapses
    # them (rank_gap <= threshold) -> the cheap, correct lever is the query /
    # retrieval, NOT a detector. This is the state the real measurement lands in.
    v = ds.instance_verdict(separation=0.093, rank_gap=0.047,
                            sep_threshold=0.05, gap_threshold=0.05)
    assert v.startswith("MIXED"), v
    assert "query" in v.lower(), v
    assert "do not train" in v.lower(), v
    print("  case verdict_mixed_signal_exists_query_collapses: OK")


def case_verdict_separated_no_bottleneck():
    # captions AND the query separate instances -> no instance bottleneck at all
    v = ds.instance_verdict(separation=0.30, rank_gap=0.28,
                            sep_threshold=0.05, gap_threshold=0.05)
    assert v.startswith("SEPARATED"), v
    assert "do not train" in v.lower(), v
    assert "no instance" in v.lower(), v
    print("  case verdict_separated_no_bottleneck: OK")


def case_verdict_inconclusive_on_nan():
    v = ds.instance_verdict(separation=float("nan"), rank_gap=float("nan"))
    assert v.startswith("INCONCLUSIVE"), v
    print("  case verdict_inconclusive_on_nan: OK")


def main() -> int:
    print("instance-separability diagnostic sanity tests")
    case_pairwise_within_and_between()
    case_pairwise_skips_singletons()
    case_separability_orthogonal_instances_separate()
    case_separability_identical_instances_overlap()
    case_separability_per_category_breakdown()
    case_rank_gap_zero_when_instances_tie()
    case_rank_gap_positive_when_instances_differ()
    case_c2c_rank_gap_orthogonal_instances_high()
    case_c2c_rank_gap_identical_instances_zero()
    case_c2c_rank_gap_skips_singleton_goal()
    case_verdict_overlap_blames_embedding()
    case_verdict_mixed_signal_exists_query_collapses()
    case_verdict_separated_no_bottleneck()
    case_verdict_inconclusive_on_nan()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
