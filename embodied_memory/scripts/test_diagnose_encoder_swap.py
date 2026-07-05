"""
TDD for diagnose_encoder_swap.encoder_gate_verdict — the $0 decision that asks:
would swapping the RETRIEVAL EMBEDDER (or the query) fix the wrong-instance
over-fire that made the powered val matrix a null (warm S3-S1 +0.02, n=48)?

The powered result localised the memory null to instance discrimination: the
SBERT (all-MiniLM, 384-d) category query "there is a {cat}" collapses the
instance signal to a ~0.047 rank gap, so retrieval can't prefer the goal
instance -> over-fire on wrong instances. This gate re-measures instance
separation + the live query gap under CANDIDATE encoders (bge/gte/e5/...) on the
controlled instance corpus, and decides whether a Lightning re-run with a
different backbone is worth it BEFORE any setup cost.

Only the pure verdict logic is unit-tested (encoder loading is I/O).

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_diagnose_encoder_swap.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_encoder_swap as es  # noqa: E402


def _row(name, sep, goal_gap, cap_gap):
    return {"name": name, "separation": sep,
            "goal_query_gap": goal_gap, "caption_query_gap": cap_gap}


# ----------------------------------------------------------------------
# GO-ENCODER: a candidate materially raises the LIVE (category) query gap
# ----------------------------------------------------------------------
def case_go_encoder_when_candidate_lifts_query_gap():
    rows = [
        _row("all-MiniLM-L6-v2", 0.093, 0.047, 0.050),   # baseline SBERT
        _row("bge-large", 0.180, 0.110, 0.120),          # lifts goal_query_gap +0.063
    ]
    v = es.encoder_gate_verdict(rows, baseline_name="all-MiniLM-L6-v2")
    assert v["verdict"] == "GO-ENCODER", v
    assert v["best_encoder"] == "bge-large", v


# ----------------------------------------------------------------------
# GO-QUERY: the encoder doesn't help the category query, but querying with a
# prior-sighting CAPTION does -> a $0 query-construction fix, no re-embed/Lightning
# ----------------------------------------------------------------------
def case_go_query_when_caption_query_beats_category_query():
    rows = [
        _row("all-MiniLM-L6-v2", 0.093, 0.047, 0.140),   # caption query gap >> category gap
        _row("bge-large", 0.150, 0.050, 0.160),          # encoder barely moves category gap
    ]
    v = es.encoder_gate_verdict(rows, baseline_name="all-MiniLM-L6-v2")
    assert v["verdict"] == "GO-QUERY", v


def case_go_query_priority_when_cheaper_than_marginal_encoder():
    # both a query fix AND a (barely) better encoder exist -> prefer the CHEAPER
    # query fix (no Lightning, no re-embed).
    rows = [
        _row("all-MiniLM-L6-v2", 0.093, 0.047, 0.150),   # strong query-side signal
        _row("gte-large", 0.160, 0.079, 0.170),          # encoder gap +0.032 (just over margin)
    ]
    v = es.encoder_gate_verdict(rows, baseline_name="all-MiniLM-L6-v2")
    assert v["verdict"] == "GO-QUERY", v


# ----------------------------------------------------------------------
# HOLD: neither a stronger encoder nor a caption query separates instances ->
# the null is text-retrieval-fundamental; Lightning re-run won't help
# ----------------------------------------------------------------------
def case_hold_when_all_flat():
    rows = [
        _row("all-MiniLM-L6-v2", 0.093, 0.047, 0.055),
        _row("bge-large", 0.100, 0.050, 0.058),
        _row("gte-large", 0.088, 0.044, 0.052),
    ]
    v = es.encoder_gate_verdict(rows, baseline_name="all-MiniLM-L6-v2")
    assert v["verdict"] == "HOLD", v


def case_hold_higher_separation_but_query_still_collapses():
    # a candidate separates instances better in embedding space (higher separation)
    # but the LIVE category query still collapses it (gap flat) AND no caption-query
    # signal -> swapping the encoder alone won't fix the live pipeline -> HOLD,
    # but the reason must flag that the *embedding* carries signal the query wastes.
    rows = [
        _row("all-MiniLM-L6-v2", 0.093, 0.047, 0.050),
        _row("bge-large", 0.220, 0.050, 0.055),          # sep up a lot, query gap flat
    ]
    v = es.encoder_gate_verdict(rows, baseline_name="all-MiniLM-L6-v2")
    assert v["verdict"] == "HOLD", v
    assert "separation" in v["reason"].lower() or "query" in v["reason"].lower(), v


def case_single_baseline_row_is_hold():
    v = es.encoder_gate_verdict([_row("all-MiniLM-L6-v2", 0.093, 0.047, 0.050)],
                                baseline_name="all-MiniLM-L6-v2")
    assert v["verdict"] in ("HOLD", "GO-QUERY"), v  # no candidate encoder to compare


def main() -> int:
    cases = [
        case_go_encoder_when_candidate_lifts_query_gap,
        case_go_query_when_caption_query_beats_category_query,
        case_go_query_priority_when_cheaper_than_marginal_encoder,
        case_hold_when_all_flat,
        case_hold_higher_separation_but_query_still_collapses,
        case_single_baseline_row_is_hold,
    ]
    print(f"running {len(cases)} diagnose_encoder_swap cases…")
    for c in cases:
        c()
        print(f"  case {c.__name__}: OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
