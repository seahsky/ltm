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

Step-1 extension (diagnose-first program): the original measurement above is
the *category* axis (does a caption mention the goal category). The "instance
discrimination is THE bottleneck" claim rests on the orthogonal *instance* axis
— can SBERT tell the goal chair from a distractor chair? ``instance_report``
(printed after the calibration block) measures that with ``INSTANCE_CORPUS``, a
controlled corpus of rich Qwen-VL-style captions labelled by physical instance.

Why a built-in corpus rather than the run logs: the locally-present minival
logs (``runs/abl-s*-qwen``) are Run-2 era, when the HM3D semantic sensor
returned all-zeros so every keyframe captioned the degenerate "room interior" —
useless for instance separation. The rich captions live only on the RACE
revisit/wide-matrix logs, and attributing a log caption to a *physical instance*
is unreliable without the goal geometry. A controlled instance-labelled corpus
is both reproducible and model-only (no Habitat / GPU / logs), so it answers the
decision rule directly: sep ≈ 0 ⇒ embedding is the bottleneck (train a
detector); sep > 0 but the category-query rank gap ≈ 0 ⇒ the signal exists but
the query discards it (fix the query first); both > 0 ⇒ no instance bottleneck.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

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


# ----------------------------------------------------------------------
# instance separability (step 1 of the diagnose-first program)
# ----------------------------------------------------------------------
#
# The category-vs-noncategory separation above is the WRONG axis for the claim
# "instance discrimination is THE bottleneck". The action-path retrieval
# (``memory_bridge.propose_memory_candidates``) ranks stored captions by cosine
# to a bare category query (``_GOAL_QUERY_TEMPLATE = "there is a {}"``). If two
# *different physical instances* of the same category (the goal chair vs a
# distractor chair) yield near-identical captions in SBERT space, retrieval
# cannot prefer the goal instance — every same-category sighting looks equally
# good. We quantify that here with two measurements:
#
#   * instance_separability: within-instance cosine (two captions of the SAME
#     object) minus between-instance-same-category cosine (captions of DIFFERENT
#     objects of the same category). sep ≈ 0 ⇒ the embedding cannot tell
#     instances apart ⇒ the embedding is the bottleneck (justifies a detector /
#     embedding swap). sep clearly > 0 ⇒ the embedding *does* separate instances
#     and the bottleneck is elsewhere (retrieval / query construction).
#   * goal_query_rank_gap: for the live query "there is a {cat}", the spread of
#     per-instance mean cosine. A small gap means the query cannot rank the goal
#     instance above a same-category distractor — the exact failure mode.
#
# A caption corpus labelled by physical instance (multiple captions per object,
# multiple objects per category) — rich Qwen-VL-style captions, the same kind
# the live captioner emits — so the measurement needs only the SBERT model,
# no Habitat / GPU / logs.

# category -> list of instances; each instance -> list of caption variants
# (different views / phrasings of the SAME physical object).
INSTANCE_CORPUS: Dict[str, List[List[str]]] = {
    "chair": [
        [  # instance A: a wooden dining chair
            "A wooden dining chair tucked under a rustic table in a sunlit dining room.",
            "A close-up of a wooden chair with a woven seat beside a dining table.",
        ],
        [  # instance B: a black leather office chair
            "A black leather office chair on casters in front of a cluttered desk.",
            "An ergonomic black swivel chair next to a computer monitor and keyboard.",
        ],
        [  # instance C: a patterned armchair
            "A floral patterned armchair in the corner of a cozy living room.",
            "An upholstered armchair with a striped cushion near a bright window.",
        ],
    ],
    "bed": [
        [  # instance A: a king bed with white linens
            "A king-size bed with crisp white linens and two pillows in a bright bedroom.",
            "A neatly made white bed beneath a window overlooking a yard.",
        ],
        [  # instance B: a child's bunk bed
            "A wooden bunk bed with a ladder against a blue bedroom wall.",
            "A small bunk bed with a cartoon blanket in a child's room.",
        ],
    ],
    "sofa": [
        [  # instance A: a brown leather couch
            "A large brown leather couch facing a fireplace in a living room.",
            "A worn brown leather sofa with a throw blanket draped over the armrest.",
        ],
        [  # instance B: a grey sectional
            "A grey fabric sectional sofa wrapping around a coffee table.",
            "A modern L-shaped grey couch beneath a wall of framed photos.",
        ],
    ],
    "toilet": [
        [  # instance A
            "A white toilet beside a pedestal sink in a small tiled bathroom.",
            "A close view of a white porcelain toilet next to a wooden cabinet.",
        ],
        [  # instance B
            "A low-profile toilet in a narrow bathroom with grey floor tiles.",
            "A toilet with a wall-mounted tank in a windowless restroom.",
        ],
    ],
}


def _encode_cached(encode: Callable[[str], np.ndarray]) -> Callable[[str], np.ndarray]:
    """Memoize an encoder so each unique string hits the model only once."""
    cache: Dict[str, np.ndarray] = {}

    def enc(s: str) -> np.ndarray:
        v = cache.get(s)
        if v is None:
            v = np.asarray(encode(s), dtype=np.float32)
            cache[s] = v
        return v

    return enc


def pairwise_cosines(
    groups: List[List[str]], encode: Callable[[str], np.ndarray]
) -> "tuple[List[float], List[float]]":
    """Return ``(within, between)`` cosine lists for a list of caption groups.

    ``within`` holds the cosine of every unordered pair of *distinct* captions
    drawn from the SAME group; ``between`` holds the cosine of every pair drawn
    from two DIFFERENT groups. Singleton groups contribute no within pairs.
    """
    enc = _encode_cached(encode)
    vecs = [[enc(c) for c in g] for g in groups]
    within: List[float] = []
    between: List[float] = []
    for gi, gvecs in enumerate(vecs):
        for a in range(len(gvecs)):
            for b in range(a + 1, len(gvecs)):
                within.append(_cos(gvecs[a], gvecs[b]))
        for gj in range(gi + 1, len(vecs)):
            for va in gvecs:
                for vb in vecs[gj]:
                    between.append(_cos(va, vb))
    return within, between


def instance_separability(
    instance_corpus: Dict[str, List[List[str]]],
    encode: Callable[[str], np.ndarray],
) -> Dict[str, Any]:
    """Within-instance vs between-instance(same-category) cosine separation.

    For each category the instances are the groups: within = two captions of
    the same object, between = captions of two different objects of the same
    category. Aggregates over all categories and reports per-category too.
    ``separation = within_mean - between_mean``; ≈ 0 means the embedding cannot
    distinguish instances of the same category.
    """
    enc = _encode_cached(encode)
    all_within: List[float] = []
    all_between: List[float] = []
    per_cat: Dict[str, Any] = {}
    for cat, instances in instance_corpus.items():
        w, b = pairwise_cosines(instances, enc)
        all_within += w
        all_between += b
        wm = float(np.mean(w)) if w else float("nan")
        bm = float(np.mean(b)) if b else float("nan")
        per_cat[cat] = {
            "within_mean": wm,
            "between_mean": bm,
            "separation": (wm - bm) if (w and b) else float("nan"),
            "n_instances": len(instances),
        }
    wmean = float(np.mean(all_within)) if all_within else float("nan")
    bmean = float(np.mean(all_between)) if all_between else float("nan")
    return {
        "within": all_within,
        "between": all_between,
        "within_mean": wmean,
        "between_mean": bmean,
        "separation": (wmean - bmean) if (all_within and all_between) else float("nan"),
        "per_category": per_cat,
    }


def goal_query_rank_gap(
    instance_corpus: Dict[str, List[List[str]]],
    encode: Callable[[str], np.ndarray],
    template: str = "there is a {}",
    query_words: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Per-category spread of the live retrieval signal across instances.

    For the query ``template.format(word)`` (the same phrasing
    ``propose_memory_candidates`` uses), compute each instance's mean cosine to
    the query, then ``rank_gap = max_instance_mean - min_instance_mean``. A
    small gap means the category query scores every instance about equally, so
    retrieval cannot rank the goal instance above a same-category distractor.
    Only categories with ≥2 instances contribute to ``mean_rank_gap``.
    """
    enc = _encode_cached(encode)
    query_words = query_words or {}
    per_cat: Dict[str, Any] = {}
    gaps: List[float] = []
    for cat, instances in instance_corpus.items():
        word = query_words.get(cat, cat)
        qv = enc(template.format(word))
        inst_means = [
            float(np.mean([_cos(qv, enc(c)) for c in inst])) for inst in instances if inst
        ]
        gap = (max(inst_means) - min(inst_means)) if len(inst_means) >= 2 else float("nan")
        per_cat[cat] = {
            "instance_means": inst_means,
            "rank_gap": gap,
            "mean": float(np.mean(inst_means)) if inst_means else float("nan"),
        }
        if len(inst_means) >= 2:
            gaps.append(gap)
    return {"per_category": per_cat, "mean_rank_gap": float(np.mean(gaps)) if gaps else float("nan")}


def caption_to_caption_rank_gap(
    instance_corpus: Dict[str, List[List[str]]],
    encode: Callable[[str], np.ndarray],
) -> Dict[str, Any]:
    """Leave-one-out caption-to-caption retrieval rank gap (Lever-1 pre-screen).

    Models the realistic warm-revisit retrieval: instead of the bare category
    query (``goal_query_rank_gap``), query with ONE prior-sighting caption of the
    goal instance and rank same-category instances by caption-to-caption
    similarity — the +0.093 within>between signal the category query discards.

    For each instance treated as the goal, each of its captions is held out as
    the query in turn; the goal's score is that query's mean cosine to its
    *other* captions (leave-one-out, so a sighting never matches itself), each
    distractor instance's score is the query's mean cosine to all its captions,
    and ``rank_gap = goal_mean − max_distractor_mean``. A goal instance needs ≥2
    captions (to have a held-out reference); a category needs ≥2 instances.
    ``mean_rank_gap`` clearly above ``goal_query_rank_gap``'s ⇒ caption-to-caption
    retrieval recovers instance discrimination the category query throws away.
    """
    enc = _encode_cached(encode)
    per_cat: Dict[str, Any] = {}
    gaps: List[float] = []
    for cat, instances in instance_corpus.items():
        vecs = [[enc(c) for c in inst] for inst in instances]
        cat_gaps: List[float] = []
        for gi, gvecs in enumerate(vecs):
            if len(gvecs) < 2:
                continue  # goal needs a held-out caption to reference itself
            for qi in range(len(gvecs)):
                q = gvecs[qi]
                others = [gvecs[k] for k in range(len(gvecs)) if k != qi]
                goal_mean = float(np.mean([_cos(q, v) for v in others]))
                distractor_means = [
                    float(np.mean([_cos(q, v) for v in vecs[dj]]))
                    for dj in range(len(vecs)) if dj != gi and vecs[dj]
                ]
                if not distractor_means:
                    continue
                cat_gaps.append(goal_mean - max(distractor_means))
        gap = float(np.mean(cat_gaps)) if cat_gaps else float("nan")
        per_cat[cat] = {"rank_gap": gap, "n_samples": len(cat_gaps)}
        if cat_gaps:
            gaps.append(gap)
    return {"per_category": per_cat, "mean_rank_gap": float(np.mean(gaps)) if gaps else float("nan")}


def instance_verdict(
    separation: float,
    rank_gap: float,
    sep_threshold: float = 0.05,
    gap_threshold: float = 0.05,
) -> str:
    """Apply the plan's decision rule (refined to three states) to the numbers.

    Two axes matter: caption-level instance separation (does the embedding carry
    instance signal at all) and the goal-query rank gap (does the live
    category-only query *exploit* that signal). Three outcomes:

    * OVERLAP — sep ≤ threshold: the embedding can't distinguish same-category
      instances even at the caption level → the EMBEDDING is the bottleneck → an
      instance-discriminative detector / embedding (step 3) is justified.
    * MIXED — sep > threshold but rank_gap ≤ gap_threshold: the captions *do*
      carry instance signal, but the bare category query collapses it → the
      cheap, correct first lever is the QUERY / retrieval construction, NOT a
      detector. (This is the state the real measurement lands in.)
    * SEPARATED — sep > threshold and rank_gap > gap_threshold: both the captions
      and the query separate instances → there is no instance-discrimination
      bottleneck; the gain ceiling is elsewhere.
    """
    if not np.isfinite(separation):
        return "INCONCLUSIVE: not enough instances/captions to measure separation."
    if separation <= sep_threshold:
        return (
            f"OVERLAP (sep={separation:+.3f} ≤ {sep_threshold}, rank_gap={rank_gap:.3f}): the "
            "EMBEDDING cannot distinguish same-category instances. Storing more goal-ish frames "
            "surfaces more wrong-instance candidates → the embedding is the bottleneck. An "
            "instance-discriminative detector/embedding (step 3) is justified; more importance-head "
            "training is not."
        )
    if np.isfinite(rank_gap) and rank_gap <= gap_threshold:
        return (
            f"MIXED (sep={separation:+.3f} > {sep_threshold} but rank_gap={rank_gap:.3f} ≤ "
            f"{gap_threshold}): the captions DO carry instance signal, but the bare category query "
            "('there is a {cat}') collapses it — every same-category instance scores about equally, "
            "so retrieval can't rank the goal instance above a distractor. The cheap, correct first "
            "lever is the QUERY / retrieval construction, NOT a detector — do NOT train a detector "
            "before exhausting query-side fixes."
        )
    return (
        f"SEPARATED (sep={separation:+.3f} > {sep_threshold}, rank_gap={rank_gap:.3f} > "
        f"{gap_threshold}): the SBERT embedding AND the category query both separate same-category "
        "instances. There is no instance-discrimination bottleneck — do NOT train a detector; the "
        "gain ceiling is elsewhere (exploration / termination), not retrieval/query."
    )


def instance_report(encode: Callable[[str], np.ndarray]) -> Dict[str, Any]:
    """Print the instance-separability table + verdict; return the raw stats."""
    enc = _encode_cached(encode)
    sep = instance_separability(INSTANCE_CORPUS, enc)
    gaps = goal_query_rank_gap(INSTANCE_CORPUS, enc)
    c2c = caption_to_caption_rank_gap(INSTANCE_CORPUS, enc)

    print("Within-category INSTANCE separability (all-MiniLM-L6-v2)")
    print("  within-instance = two captions of the SAME object;")
    print("  between-instance = captions of DIFFERENT objects of the same category.\n")
    print(f"  {'category':<10} {'#inst':>5}  {'within':>7} {'betwn':>7} {'sep':>7}  "
          f"{'qry_gap':>7}  instance query-cosines")
    for cat in INSTANCE_CORPUS:
        pc = sep["per_category"][cat]
        gc = gaps["per_category"][cat]
        means = " ".join(f"{m:.3f}" for m in gc["instance_means"])
        print(f"  {cat:<10} {pc['n_instances']:>5}  {pc['within_mean']:>7.3f} "
              f"{pc['between_mean']:>7.3f} {pc['separation']:>+7.3f}  {gc['rank_gap']:>7.3f}  [{means}]")
    print(f"\n  --> ALL: within-instance mean={sep['within_mean']:.3f} | "
          f"between-instance(same-cat) mean={sep['between_mean']:.3f} | "
          f"separation={sep['separation']:+.3f}")
    print(f"      mean goal-query rank gap across instances    = {gaps['mean_rank_gap']:.3f}  (bare category query 'there is a {{cat}}')")
    print(f"      mean caption-to-caption rank gap (Lever 1)    = {c2c['mean_rank_gap']:.3f}  (query with a prior-sighting caption)")
    recovered = c2c["mean_rank_gap"] - gaps["mean_rank_gap"]
    print(f"      --> Lever-1 recovery = {recovered:+.3f}  ("
          f"{'caption-to-caption retrieval recovers the instance gap the category query discards' if recovered > 0.02 else 'no meaningful recovery — query-side fix unlikely to help'})")
    verdict = instance_verdict(sep["separation"], gaps["mean_rank_gap"])
    print(f"\n  VERDICT: {verdict}\n")
    return {"separability": sep, "rank_gap": gaps, "c2c_rank_gap": c2c, "verdict": verdict}


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

    # ------------------------------------------------------------------
    # step-1 measurement: within-category INSTANCE separability. The block
    # above measures the *category* axis (does a caption mention the goal
    # category); this measures the orthogonal *instance* axis that the
    # "instance discrimination is the bottleneck" claim actually rests on.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70 + "\n")
    instance_report(encode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
