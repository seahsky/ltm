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


# ----------------------------------------------------------------------
# query_template_ab + recommend_query_variant — Stage-0 query-fix selection
# ----------------------------------------------------------------------


def case_query_ab_returns_per_variant_gaps():
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0),
        "cB1": (0, 1, 0), "cB2": (0, 1, 0),
        "there is a chair": (1, 1, 0),
    })
    ab = ds.query_template_ab({"chair": [["cA1", "cA2"], ["cB1", "cB2"]]}, enc)
    # every builder is present and reports a pooled + per-category gap
    for name in ("bare_category", "caption", "prf_interp", "hyde", "attribute"):
        assert name in ab, ab.keys()
        assert "pooled_rank_gap" in ab[name], ab[name]
        assert "chair" in ab[name]["per_category"], ab[name]
    print("  case query_ab_returns_per_variant_gaps: OK")


def case_query_ab_caption_beats_bare():
    # Bare category query is equidistant from both instances (gap 0); querying
    # with the goal's prior-sighting caption separates them (gap 1).
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0),
        "cB1": (0, 1, 0), "cB2": (0, 1, 0),
        "bA1": (1, 0, 0), "bA2": (1, 0, 0),
        "bB1": (0, 1, 0), "bB2": (0, 1, 0),
        "there is a chair": (1, 1, 0),
        "there is a bed": (1, 1, 0),
    })
    corpus = {"chair": [["cA1", "cA2"], ["cB1", "cB2"]],
              "bed": [["bA1", "bA2"], ["bB1", "bB2"]]}
    ab = ds.query_template_ab(corpus, enc)
    assert abs(ab["bare_category"]["pooled_rank_gap"] - 0.0) < 1e-6, ab["bare_category"]
    assert ab["caption"]["pooled_rank_gap"] > 0.9, ab["caption"]
    print("  case query_ab_caption_beats_bare: OK")


def case_recommend_picks_caption_winner():
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0), "cB1": (0, 1, 0), "cB2": (0, 1, 0),
        "bA1": (1, 0, 0), "bA2": (1, 0, 0), "bB1": (0, 1, 0), "bB2": (0, 1, 0),
        "there is a chair": (1, 1, 0), "there is a bed": (1, 1, 0),
    })
    corpus = {"chair": [["cA1", "cA2"], ["cB1", "cB2"]],
              "bed": [["bA1", "bA2"], ["bB1", "bB2"]]}
    rec = ds.recommend_query_variant(ds.query_template_ab(corpus, enc))
    assert rec["winner"] == "caption", rec
    assert "RECOMMEND" in rec["verdict"], rec
    print("  case recommend_picks_caption_winner: OK")


def case_recommend_honest_negative_when_no_separation():
    # all instances identical -> no variant beats the baseline -> honest negative
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0), "cB1": (1, 0, 0), "cB2": (1, 0, 0),
        "bA1": (1, 0, 0), "bA2": (1, 0, 0), "bB1": (1, 0, 0), "bB2": (1, 0, 0),
        "there is a chair": (1, 0, 0), "there is a bed": (1, 0, 0),
    })
    corpus = {"chair": [["cA1", "cA2"], ["cB1", "cB2"]],
              "bed": [["bA1", "bA2"], ["bB1", "bB2"]]}
    rec = ds.recommend_query_variant(ds.query_template_ab(corpus, enc))
    assert rec["winner"] is None, rec
    assert "HONEST NEGATIVE" in rec["verdict"], rec
    print("  case recommend_honest_negative_when_no_separation: OK")


def case_recommend_requires_two_categories():
    # caption separates chair (gap 1) but bed instances are identical (gap 0) ->
    # only ONE category beats the baseline -> below the >=2 bar -> no winner.
    enc = _fake_encode({
        "cA1": (1, 0, 0), "cA2": (1, 0, 0), "cB1": (0, 1, 0), "cB2": (0, 1, 0),
        "bA1": (1, 0, 0), "bA2": (1, 0, 0), "bB1": (1, 0, 0), "bB2": (1, 0, 0),
        "there is a chair": (1, 1, 0), "there is a bed": (1, 0, 0),
    })
    corpus = {"chair": [["cA1", "cA2"], ["cB1", "cB2"]],
              "bed": [["bA1", "bA2"], ["bB1", "bB2"]]}
    rec = ds.recommend_query_variant(ds.query_template_ab(corpus, enc), min_cats=2)
    assert rec["winner"] is None, rec
    print("  case recommend_requires_two_categories: OK")


# ----------------------------------------------------------------------
# Phase-0 captioner-swap gate
# ----------------------------------------------------------------------
def _rec(captioner, obj, cap, scene="s", category="chair"):
    return {"captioner": captioner, "scene": scene, "category": category,
            "object_id": obj, "caption": cap}


def case_captions_to_corpus_groups_by_scene_cat_and_instance():
    records = [
        _rec("qwen", 1, "a"), _rec("qwen", 1, "b"), _rec("qwen", 2, "c"),
        _rec("caprl", 1, "d"),            # different captioner -> excluded
        _rec("qwen", 1, "   "),           # blank -> dropped
        _rec("qwen", 1, "e", scene="s2"),  # different scene -> different key
    ]
    corp = ds.captions_to_instance_corpus(records, "qwen")
    assert set(corp) == {"s/chair", "s2/chair"}, corp
    # s/chair has 2 instances: obj1=[a,b], obj2=[c]; obj order = insertion
    assert sorted(sorted(g) for g in corp["s/chair"]) == [["a", "b"], ["c"]], corp
    assert corp["s2/chair"] == [["e"]]
    print("  case captions_to_corpus_groups_by_scene_cat_and_instance: OK")


def case_gate_GO_when_candidate_separates_instances():
    # qwen: every caption -> same vector -> within==between -> sep 0.
    # caprl: instance1 -> A, instance2 -> B (orthogonal) -> within 1, between 0 -> sep 1.
    enc = _fake_encode({
        "qa1": (1, 0, 0), "qa2": (1, 0, 0), "qb1": (1, 0, 0), "qb2": (1, 0, 0),
        "ca1": (1, 0, 0), "ca2": (1, 0, 0), "cb1": (0, 1, 0), "cb2": (0, 1, 0),
    })
    records = [
        _rec("qwen", 1, "qa1"), _rec("qwen", 1, "qa2"),
        _rec("qwen", 2, "qb1"), _rec("qwen", 2, "qb2"),
        _rec("caprl", 1, "ca1"), _rec("caprl", 1, "ca2"),
        _rec("caprl", 2, "cb1"), _rec("caprl", 2, "cb2"),
    ]
    res = ds.compare_captioners(records, enc, baseline="qwen", candidate="caprl", margin=0.02)
    assert res["caprl"]["separation"] > 0.9, res
    assert abs(res["qwen"]["separation"]) < 0.05, res
    assert res["delta_separation"] > 0.9
    assert res["gate_pass"] is True and res["result"] == "GO", res
    assert res["verdict"].startswith("GO"), res["verdict"]
    print("  case gate_GO_when_candidate_separates_instances: OK")


def case_gate_INSUFFICIENT_when_candidate_corpus_empty():
    # candidate has NO records (e.g. a stub-loaded captioner emitted nothing) ->
    # its separation is NaN -> result INSUFFICIENT, NOT a misleading HOLD.
    enc = _fake_encode({"qa1": (1, 0, 0), "qa2": (0, 1, 0), "qb1": (1, 0, 0), "qb2": (0, 1, 0)})
    records = [  # baseline only
        _rec("qwen", 1, "qa1"), _rec("qwen", 1, "qa2"),
        _rec("qwen", 2, "qb1"), _rec("qwen", 2, "qb2"),
    ]
    res = ds.compare_captioners(records, enc, baseline="qwen", candidate="caprl", margin=0.02)
    assert res["result"] == "INSUFFICIENT", res
    assert res["gate_pass"] is False
    assert res["verdict"].startswith("INSUFFICIENT"), res["verdict"]
    print("  case gate_INSUFFICIENT_when_candidate_corpus_empty: OK")


def case_corpus_skips_malformed_records_no_keyerror():
    # records missing object_id must be SKIPPED, not raise KeyError.
    records = [
        _rec("qwen", 1, "a"), _rec("qwen", 1, "b"), _rec("qwen", 2, "c"),
        {"captioner": "qwen", "scene": "s", "category": "chair", "caption": "no objid"},  # malformed
        {"captioner": "qwen", "caption": "no scene/cat/obj"},                              # malformed
    ]
    corp = ds.captions_to_instance_corpus(records, "qwen")  # must not raise
    assert sorted(sorted(g) for g in corp["s/chair"]) == [["a", "b"], ["c"]], corp
    print("  case corpus_skips_malformed_records_no_keyerror: OK")


def case_compare_report_prints_machine_marker():
    import contextlib
    import io
    enc = _fake_encode({
        "qa1": (1, 0, 0), "qa2": (1, 0, 0), "qb1": (1, 0, 0), "qb2": (1, 0, 0),
        "ca1": (1, 0, 0), "ca2": (1, 0, 0), "cb1": (0, 1, 0), "cb2": (0, 1, 0),
    })
    records = [
        _rec("qwen", 1, "qa1"), _rec("qwen", 1, "qa2"), _rec("qwen", 2, "qb1"), _rec("qwen", 2, "qb2"),
        _rec("caprl", 1, "ca1"), _rec("caprl", 1, "ca2"), _rec("caprl", 2, "cb1"), _rec("caprl", 2, "cb2"),
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = ds.compare_captioners_report(records, enc, baseline="qwen", candidate="caprl", margin=0.02)
    out = buf.getvalue()
    assert "GATE_RESULT=GO" in out, out
    assert res["result"] == "GO"
    print("  case compare_report_prints_machine_marker: OK")


def case_gate_HOLD_when_candidate_no_better():
    # both captioners collapse all captions to one vector -> both sep 0 -> HOLD.
    enc = _fake_encode({k: (1, 0, 0) for k in
                        ["x1", "x2", "x3", "x4", "y1", "y2", "y3", "y4"]})
    records = [
        _rec("qwen", 1, "x1"), _rec("qwen", 1, "x2"),
        _rec("qwen", 2, "x3"), _rec("qwen", 2, "x4"),
        _rec("caprl", 1, "y1"), _rec("caprl", 1, "y2"),
        _rec("caprl", 2, "y3"), _rec("caprl", 2, "y4"),
    ]
    res = ds.compare_captioners(records, enc, baseline="qwen", candidate="caprl", margin=0.02)
    assert res["gate_pass"] is False, res
    assert res["verdict"].startswith("HOLD"), res["verdict"]
    print("  case gate_HOLD_when_candidate_no_better: OK")


# ----------------------------------------------------------------------
# Phase-3a encoder-swap gate
# ----------------------------------------------------------------------
def case_parse_encoders_default_and_pairs():
    d = ds.parse_encoders([])
    assert d["all-MiniLM-L6-v2"] == "sentence-transformers/all-MiniLM-L6-v2", d
    assert ds.parse_encoders(["a=org/A", "b=org/B"]) == {"a": "org/A", "b": "org/B"}
    try:
        ds.parse_encoders(["noequals"])
    except ValueError:
        print("  case parse_encoders_default_and_pairs: OK")
        return
    raise AssertionError("expected ValueError on malformed --encoders entry")


def case_encoder_gate_GO_when_candidate_separates():
    # captions FIXED; baseline encoder collapses all -> sep 0; candidate encoder maps
    # instance1->A, instance2->B (orthogonal) -> sep 1 -> GO, best=candidate.
    records = [
        _rec("qwen", 1, "c_a1"), _rec("qwen", 1, "c_a2"),
        _rec("qwen", 2, "c_b1"), _rec("qwen", 2, "c_b2"),
    ]
    base = _fake_encode({k: (1, 0, 0) for k in ["c_a1", "c_a2", "c_b1", "c_b2"]})
    cand = _fake_encode({"c_a1": (1, 0, 0), "c_a2": (1, 0, 0), "c_b1": (0, 1, 0), "c_b2": (0, 1, 0)})
    res = ds.compare_encoders(records, {"base": base, "cand": cand},
                              captioner="qwen", baseline="base", margin=0.02)
    assert res["encoders"]["cand"]["separation"] > 0.9, res
    assert abs(res["encoders"]["base"]["separation"]) < 0.05, res
    assert res["best_candidate"] == "cand"
    assert res["result"] == "GO" and res["gate_pass"] is True
    assert res["verdict"].startswith("GO"), res["verdict"]
    print("  case encoder_gate_GO_when_candidate_separates: OK")


def case_encoder_gate_HOLD_when_no_candidate_better():
    records = [_rec("qwen", 1, "a"), _rec("qwen", 1, "b"), _rec("qwen", 2, "c"), _rec("qwen", 2, "d")]
    same = _fake_encode({k: (1, 0, 0) for k in ["a", "b", "c", "d"]})
    res = ds.compare_encoders(records, {"base": same, "x": same},
                              captioner="qwen", baseline="base", margin=0.02)
    assert res["result"] == "HOLD", res
    assert res["verdict"].startswith("HOLD")
    print("  case encoder_gate_HOLD_when_no_candidate_better: OK")


def case_encoder_gate_INSUFFICIENT_when_baseline_failed():
    # baseline encoder failed to load (None) -> can't compare -> INSUFFICIENT.
    records = [_rec("qwen", 1, "a"), _rec("qwen", 1, "b"), _rec("qwen", 2, "c"), _rec("qwen", 2, "d")]
    cand = _fake_encode({"a": (1, 0, 0), "b": (1, 0, 0), "c": (0, 1, 0), "d": (0, 1, 0)})
    res = ds.compare_encoders(records, {"base": None, "cand": cand},
                              captioner="qwen", baseline="base", margin=0.02)
    assert res["result"] == "INSUFFICIENT", res
    assert res["encoders"]["base"] == {"loaded": False}
    assert res["verdict"].startswith("INSUFFICIENT")
    print("  case encoder_gate_INSUFFICIENT_when_baseline_failed: OK")


def case_encoder_gate_tracks_dropped_candidates():
    # a candidate that failed to load (None) must be RECORDED in `dropped`, not
    # silently ignored — else a HOLD could be missing the strongest encoder.
    records = [_rec("qwen", 1, "a"), _rec("qwen", 1, "b"), _rec("qwen", 2, "c"), _rec("qwen", 2, "d")]
    base = _fake_encode({k: (1, 0, 0) for k in ["a", "b", "c", "d"]})
    good = _fake_encode({"a": (1, 0, 0), "b": (1, 0, 0), "c": (0, 1, 0), "d": (0, 1, 0)})
    res = ds.compare_encoders(records, {"base": base, "good": good, "broken": None},
                              captioner="qwen", baseline="base", margin=0.02)
    assert res["dropped"] == ["broken"], res["dropped"]
    assert res["n_candidates_requested"] == 2 and res["n_candidates_loaded"] == 1
    assert res["result"] == "GO" and res["best_candidate"] == "good"  # still decides on the loaded ones
    print("  case encoder_gate_tracks_dropped_candidates: OK")


def case_load_caption_records_rejects_non_list():
    import json as _json
    import os as _os
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        _json.dump(42, f)
        p = f.name
    try:
        ds.load_caption_records(p)
    except ValueError:
        _os.unlink(p)
        print("  case load_caption_records_rejects_non_list: OK")
        return
    _os.unlink(p)
    raise AssertionError("expected ValueError on non-list JSON")


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
    case_query_ab_returns_per_variant_gaps()
    case_query_ab_caption_beats_bare()
    case_recommend_picks_caption_winner()
    case_recommend_honest_negative_when_no_separation()
    case_recommend_requires_two_categories()
    case_captions_to_corpus_groups_by_scene_cat_and_instance()
    case_gate_GO_when_candidate_separates_instances()
    case_gate_INSUFFICIENT_when_candidate_corpus_empty()
    case_corpus_skips_malformed_records_no_keyerror()
    case_compare_report_prints_machine_marker()
    case_gate_HOLD_when_candidate_no_better()
    case_parse_encoders_default_and_pairs()
    case_encoder_gate_GO_when_candidate_separates()
    case_encoder_gate_HOLD_when_no_candidate_better()
    case_encoder_gate_INSUFFICIENT_when_baseline_failed()
    case_encoder_gate_tracks_dropped_candidates()
    case_load_caption_records_rejects_non_list()
    print("All cases passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
