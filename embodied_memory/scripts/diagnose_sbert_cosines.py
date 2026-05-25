"""
Measure the SBERT goal-vs-caption cosine scale for LTM memory calibration.

The LTM fine layer is now indexed on SBERT caption-text (commit 3546779).
``propose_memory_candidates`` queries it with ``"a photo of a {goal}"`` and
``FrontierPhysicsScorer`` gates the result with ``_MEM_COS_NULL`` /
``_MEM_COS_FULL`` (+ a ``min_cosine`` pre-filter). Those thresholds were set by
estimate and the re-index mini collapsed memory to 3 candidates (too high),
so we need the *actual* SBERT scale to calibrate — not another guess.

This is model-only (no Habitat): it loads the same SentenceTransformer the
bridge uses (``all-MiniLM-L6-v2`` via ``dialogue_memory.encoder``) and computes
cosines between several query phrasings and a labelled corpus of real captions
from the minival runs. A caption is auto-labelled a *match* for a goal if any
of the goal's synonyms appears as a whole word.

Run (in the race-setup env)::

    source scripts/race-setup.sh
    python3 embodied_memory/scripts/diagnose_sbert_cosines.py

Read the per-template "match vs non-match" separation and the suggested
min_cosine / _MEM_COS_NULL / _MEM_COS_FULL at the bottom.
"""

from __future__ import annotations

import re
from typing import Callable, Dict, List

import numpy as np


# ----------------------------------------------------------------------
# encoder (identical to the live pipeline's text_encode_fn)
# ----------------------------------------------------------------------


def _build_encoder() -> Callable[[str], np.ndarray]:
    """Return an encode(str)->vec fn using the same SBERT model as the bridge."""
    try:
        from dialogue_memory.encoder import SentenceTransformerEncoder

        enc = SentenceTransformerEncoder(model_name="all-MiniLM-L6-v2")
        return lambda s: np.asarray(enc.encode(s), dtype=np.float32)
    except Exception as e:  # pragma: no cover - fallback for a bare env
        print(f"[warn] dialogue_memory encoder unavailable ({e}); using raw ST")
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda s: np.asarray(m.encode(s), dtype=np.float32)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ----------------------------------------------------------------------
# goals + synonyms (mirror remembr_backbone keyword-STOP synonyms) + corpus
# ----------------------------------------------------------------------


GOAL_SYNONYMS: Dict[str, List[str]] = {
    "chair": ["chair", "chairs"],
    "bed": ["bed", "bedroom", "bedspread"],
    "toilet": ["toilet", "bathroom"],
    "sofa": ["sofa", "couch"],
    "tv_monitor": ["television", "tv", "monitor", "screen"],
    "plant": ["plant", "flowers", "vase", "potted"],
}

# Real captions emitted by the Qwen-VL captioner across the minival runs.
CAPTIONS: List[str] = [
    "A room with freshly painted walls and a partially installed ceiling, featuring a wooden chair and a small table.",
    "A cozy living room with hardwood floors, two patterned chairs, and a large window.",
    "A cozy living room with a brown couch, a lamp, and a bookshelf.",
    "A cozy living room with a black leather couch, a lamp, and a Christmas tree.",
    "The scene depicts a cozy living room with a fireplace, a television, and a dining area.",
    "The scene depicts a hallway leading to a bedroom with a bed and a window.",
    "A spacious bedroom with a bed, a ceiling fan, and a window.",
    "A cozy bedroom with a purple bedspread and a window overlooking a yard.",
    "A bathroom with a white door, tiled walls, and a wooden floor.",
    "A bathroom scene with a white sink, mirror, and cabinets.",
    "A long, narrow hallway with wooden floors, white walls, and a blue door on the right.",
    "A small, empty room with a window, wooden floor, and a wall socket.",
    "The image shows a white wall with a door leading into a room.",
    "A kitchen scene with a stove and oven, featuring a person standing near the oven.",
    "A spacious living room with hardwood floors, a piano, and framed pictures on the walls.",
    "A cozy dining room with a wooden table, chairs, and a vase of flowers on the table.",
]

# SBERT query phrasings to compare (the bridge currently uses "a photo of a {}").
TEMPLATES = ["a photo of a {}", "a {}", "{}", "there is a {}", "a room with a {}"]


def _is_match(caption: str, goal: str) -> bool:
    low = caption.lower()
    return any(re.search(rf"\b{re.escape(s)}\b", low) for s in GOAL_SYNONYMS[goal])


def main() -> int:
    encode = _build_encoder()
    cap_vecs = [encode(c) for c in CAPTIONS]

    # Per template: collect match / non-match cosines across all goals.
    print("Goal-vs-caption SBERT cosine separation (all-MiniLM-L6-v2)\n")
    template_stats = {}
    for tmpl in TEMPLATES:
        match_cos: List[float] = []
        non_cos: List[float] = []
        print(f"=== query template: {tmpl!r} ===")
        for goal in GOAL_SYNONYMS:
            qword = "television" if goal == "tv_monitor" else goal
            qv = encode(tmpl.format(qword))
            ms, ns = [], []
            for cap, cv in zip(CAPTIONS, cap_vecs):
                c = _cos(qv, cv)
                (ms if _is_match(cap, goal) else ns).append(c)
            match_cos += ms
            non_cos += ns
            mtxt = f"match[{len(ms)}] max={max(ms):.3f} mean={np.mean(ms):.3f}" if ms else "match[0] —"
            ntxt = f"nonmatch[{len(ns)}] mean={np.mean(ns):.3f} p90={np.percentile(ns,90):.3f}" if ns else "nonmatch[0] —"
            print(f"  {goal:<11} {mtxt:<34} {ntxt}")
        mm = np.array(match_cos)
        nn = np.array(non_cos)
        sep = float(mm.mean() - nn.mean()) if len(mm) and len(nn) else float("nan")
        template_stats[tmpl] = (mm, nn, sep)
        print(f"  --> ALL: match mean={mm.mean():.3f} (min {mm.min():.3f}) | "
              f"nonmatch mean={nn.mean():.3f} (p90 {np.percentile(nn,90):.3f}) | "
              f"separation={sep:+.3f}\n")

    # Recommend a calibration from the best-separating template.
    best = max(template_stats, key=lambda t: template_stats[t][2])
    mm, nn, sep = template_stats[best]
    null = float(np.percentile(nn, 75))          # most non-matches contribute 0
    full = float(np.percentile(mm, 50))           # a median true match saturates
    floor = float(max(np.percentile(nn, 50), null - 0.05))  # pre-filter
    print("=== recommendation ===")
    print(f"  best-separating template: {best!r} (separation {sep:+.3f})")
    print(f"  suggested min_cosine ~= {floor:.2f}   (discard below ~nonmatch median)")
    print(f"  suggested _MEM_COS_NULL ~= {null:.2f}  (nonmatch p75 — baseline contributes 0)")
    print(f"  suggested _MEM_COS_FULL ~= {full:.2f}  (match median — true sighting saturates)")
    if full - null < 0.05:
        print("  WARNING: match/nonmatch overlap heavily — SBERT barely discriminates here;")
        print("           consider a different query phrasing or accept memory is near-neutral.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
