"""
diagnose_encoder_swap — the $0 "different backbone?" gate.

The powered val matrix made the memory soft-SPL a NULL at scale (warm S3-S1
+0.0197, n=48, p=0.29), diagnosed to WRONG-INSTANCE OVER-FIRE: the SBERT
(all-MiniLM, 384-d) live query "there is a {cat}" collapses the instance signal
to a ~0.047 rank gap, so retrieval can't prefer the goal instance. Before
setting up Lightning + re-running the whole Habitat matrix with a different
backbone, this measures — LOCALLY, no GPU/sim, reusing the controlled
instance-labelled caption corpus — whether a stronger RETRIEVAL EMBEDDER (bge /
gte / e5 / Qwen3-Embedding) or a caption-based QUERY separates instances better
than SBERT. That decides whether a backbone swap is worth the compute.

Reuses diagnose_sbert_cosines' pure metrics (instance_separability /
goal_query_rank_gap / caption_to_caption_rank_gap) so the measurement is
identical to the prior instance-bottleneck diagnostic, just under other encoders.

    python embodied_memory/scripts/diagnose_encoder_swap.py \
        --encoders all-MiniLM-L6-v2 BAAI/bge-large-en-v1.5 thenlper/gte-large

Verdict (GATE_RESULT):
  GO-ENCODER — a candidate materially lifts the LIVE category-query gap -> swap
               the embedder and re-run (Lightning justified).
  GO-QUERY   — the encoder doesn't move the category query, but a prior-sighting
               CAPTION query does -> a $0 query-construction fix in
               propose_memory_candidates (NO re-embed, NO Lightning).
  HOLD       — neither separates instances -> the null is text-retrieval-
               fundamental; a text-embedder backbone swap won't help (consider a
               multimodal/image retrieval or accept the characterised negative).
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnose_sbert_cosines as dc  # noqa: E402

_BASELINE = "all-MiniLM-L6-v2"
# A candidate must lift the LIVE category-query rank gap by at least this (abs)
# over baseline to justify a full re-run; the same margin flags a query-side fix.
_GAP_MARGIN = 0.03


def _named_encoder(model_name: str) -> Callable[[str], np.ndarray]:
    """Return encode(str)->vec for a sentence-transformers model (runs on
    CPU/MPS locally). e5-family models want a 'query:'/'passage:' prefix; for
    this symmetric caption-vs-caption measurement we encode raw (the relative
    comparison across encoders stays valid) and note it in the header."""
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    return lambda s: np.asarray(m.encode(s), dtype=np.float32)


def encoder_gate_verdict(rows: List[Dict[str, Any]], *,
                         baseline_name: str = _BASELINE,
                         gap_margin: float = _GAP_MARGIN) -> Dict[str, Any]:
    """Decide GO-ENCODER / GO-QUERY / HOLD from per-encoder metric rows.

    Each row: {name, separation, goal_query_gap, caption_query_gap}.
      * GO-QUERY (cheapest, preferred): for ANY encoder, the caption-based query
        beats its own category query by >= gap_margin -> fix the query, no
        re-embed / no Lightning.
      * GO-ENCODER: some CANDIDATE encoder's category-query gap beats the
        baseline's by >= gap_margin -> swap the embedder + re-run.
      * HOLD: neither. Pure."""
    baseline = next((r for r in rows if r["name"] == baseline_name), rows[0] if rows else None)
    if baseline is None:
        return {"verdict": "HOLD", "best_encoder": None, "reason": "no rows"}

    # cheapest first: a caption query that beats the category query (any encoder)
    query_wins = [r for r in rows
                  if (r.get("caption_query_gap", 0.0) - r.get("goal_query_gap", 0.0)) >= gap_margin]

    candidates = [r for r in rows if r["name"] != baseline_name]
    best = max(candidates, key=lambda r: r.get("goal_query_gap", 0.0), default=None)
    encoder_wins = (best is not None
                    and (best["goal_query_gap"] - baseline["goal_query_gap"]) >= gap_margin)

    if query_wins:
        best_q = max(query_wins, key=lambda r: r["caption_query_gap"] - r["goal_query_gap"])
        return {"verdict": "GO-QUERY", "best_encoder": best_q["name"],
                "reason": (f"caption query gap {best_q['caption_query_gap']:.3f} beats the category "
                           f"query gap {best_q['goal_query_gap']:.3f} by >= {gap_margin} on "
                           f"{best_q['name']} -> a $0 query-construction fix, no re-embed / Lightning.")}
    if encoder_wins:
        return {"verdict": "GO-ENCODER", "best_encoder": best["name"],
                "reason": (f"{best['name']} lifts the live category-query gap to "
                           f"{best['goal_query_gap']:.3f} vs baseline {baseline['goal_query_gap']:.3f} "
                           f"(>= +{gap_margin}) -> swap the embedder and re-run (Lightning justified).")}

    # HOLD — surface whether the embedding carries signal the query wastes.
    best_sep = max(rows, key=lambda r: r.get("separation", 0.0))
    flags = ""
    if best_sep["separation"] - baseline["separation"] >= gap_margin:
        flags = (f" NB {best_sep['name']} separates instances better in embedding space "
                 f"(separation {best_sep['separation']:.3f}) but the category query still collapses it "
                 f"(gap flat) -> the query, not the encoder, wastes the signal.")
    return {"verdict": "HOLD", "best_encoder": None,
            "reason": ("no candidate encoder lifts the live query gap and no caption query beats the "
                       "category query -> the null is text-retrieval-fundamental; a text-embedder "
                       "backbone swap won't help (consider multimodal/image retrieval)." + flags)}


def measure_encoder(name: str) -> Dict[str, Any]:
    """Run the three instance metrics on the controlled corpus under one encoder."""
    enc = _named_encoder(name)
    sep = dc.instance_separability(dc.INSTANCE_CORPUS, enc)
    gq = dc.goal_query_rank_gap(dc.INSTANCE_CORPUS, enc)
    cc = dc.caption_to_caption_rank_gap(dc.INSTANCE_CORPUS, enc)
    return {
        "name": name,
        "separation": float(sep.get("separation", float("nan"))),
        "within": float(sep.get("within_mean", float("nan"))),
        "between": float(sep.get("between_mean", float("nan"))),
        "goal_query_gap": float(gq.get("mean_rank_gap", float("nan"))),
        "caption_query_gap": float(cc.get("mean_rank_gap", float("nan"))),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoders", nargs="+",
                    default=[_BASELINE, "BAAI/bge-large-en-v1.5", "thenlper/gte-large"],
                    help="sentence-transformers model names; first should be the SBERT baseline.")
    ap.add_argument("--gap-margin", type=float, default=_GAP_MARGIN)
    args = ap.parse_args(argv)

    print("[encoder-swap] controlled instance corpus (no GPU/sim); metrics identical to "
          "diagnose_sbert_cosines. e5-family encoded raw (symmetric caption-caption).")
    print(f"[encoder-swap] baseline={args.encoders[0]}  gap_margin={args.gap_margin}\n")
    print(f"  {'encoder':<32} {'separation':>10} {'goalQgap':>9} {'capQgap':>8}")
    print(f"  {'(within-between)':<32} {'':>10} {'(live)':>9} {'(fix?)':>8}")
    rows: List[Dict[str, Any]] = []
    for name in args.encoders:
        try:
            r = measure_encoder(name)
        except Exception as e:
            print(f"  {name:<32}  FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        rows.append(r)
        print(f"  {name:<32} {r['separation']:>+10.3f} {r['goal_query_gap']:>+9.3f} "
              f"{r['caption_query_gap']:>+8.3f}")

    if not rows:
        print("\nGATE_RESULT=ERROR (no encoder loaded)")
        return 2
    v = encoder_gate_verdict(rows, baseline_name=args.encoders[0], gap_margin=args.gap_margin)
    print(f"\nGATE_RESULT={v['verdict']}")
    print(f"  {v['reason']}")
    if v.get("best_encoder"):
        print(f"  best={v['best_encoder']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
