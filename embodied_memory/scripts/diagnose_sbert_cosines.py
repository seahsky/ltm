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

import json
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


# ----------------------------------------------------------------------
# query-construction A/B (Stage-0 query-fix selection)
# ----------------------------------------------------------------------
#
# The instance signal EXISTS in the embedding (within-instance 0.628 vs
# between-instance-same-category 0.535, sep +0.093) but the live query
# ``"there is a {cat}"`` collapses it (VERDICT MIXED) — so the cheap, correct
# lever is the QUERY, not a detector. This A/B scores several DEPLOYABLE
# query-construction variants on the SAME instance corpus with ONE consistent
# retrieval metric: a leave-one-out goal-vs-best-distractor rank gap. For each
# category, each instance is treated as the goal in turn; one of its captions is
# held out as the retrieval target; the query is built (per variant) from the
# category word and/or the goal's OTHER (prior-sighting) captions — never the
# held-out target — so a winning variant is realizable in
# ``propose_memory_candidates`` (the agent has its own prior sightings in the
# LTM), not an oracle. ``gap = cos(query, held-out goal caption) − max over
# distractor instances of cos(query, distractor captions)``: how much the query
# ranks the goal instance above the best same-category distractor. The bare
# category query carries no instance preference (gap ≈ 0); a caption/PRF query
# built from prior sightings should recover the embedding's instance gap.
# ``recommend_query_variant`` declares a winner only if it clears a margin on
# the POOLED gap AND on >=2 categories, so a single chair-driven win can't carry
# a flat result.

# Generic enrichment templates (NO instance-specific info) — test whether a
# richer *generic* phrasing alone helps (it should not, much).
_HYDE_TEMPLATES: Dict[str, str] = {
    "chair": "a photo of a chair with a seat, backrest and legs in a room",
    "bed": "a photo of a bed with a mattress, pillows and linens in a bedroom",
    "sofa": "a photo of a sofa, a long upholstered couch with cushions",
    "toilet": "a photo of a white porcelain toilet in a tiled bathroom",
    "tv_monitor": "a photo of a television screen on a wall or stand",
    "plant": "a photo of a potted plant with green leaves in a vase",
}


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n == 0.0 else (v / n)


def _query_builders() -> Dict[str, Callable]:
    """variant name -> ``build(cat, qword, others, enc) -> query vector``.

    ``others`` are the goal instance's prior-sighting captions available to
    construct the query; every builder is deployable from prior sightings + the
    category and none peeks at the held-out target caption.
    """

    def bare_category(cat, qword, others, enc):
        return enc("there is a {}".format(qword))

    def caption(cat, qword, others, enc):
        # query with the most recent prior-sighting caption of the goal instance
        return enc(others[0]) if others else enc("there is a {}".format(qword))

    def prf_interp(cat, qword, others, enc, alpha=0.5, beta=0.5):
        qv = _normalize(enc("there is a {}".format(qword)))
        if not others:
            return qv
        mean_other = _normalize(
            np.mean([_normalize(enc(c)) for c in others], axis=0)
        )
        return _normalize(alpha * qv + beta * mean_other)

    def hyde(cat, qword, others, enc):
        return enc(_HYDE_TEMPLATES.get(cat, "a photo of a {} in a room".format(qword)))

    def attribute(cat, qword, others, enc):
        return enc("a {} in a room with furniture and walls".format(qword))

    return {"bare_category": bare_category, "caption": caption,
            "prf_interp": prf_interp, "hyde": hyde, "attribute": attribute}


def query_template_ab(
    instance_corpus: Dict[str, List[List[str]]],
    encode: Callable[[str], np.ndarray],
    variants: Optional[Dict[str, Callable]] = None,
) -> Dict[str, Any]:
    """A/B query-construction variants on the instance corpus (see section doc).

    Returns ``{variant_name: {"per_category": {cat: {"rank_gap", "n_samples"}},
    "pooled_rank_gap": float}}``. A category needs >=2 instances (a distractor)
    and a goal instance needs >=2 captions (a held-out target).
    """
    enc = _encode_cached(encode)
    builders = variants or _query_builders()
    results: Dict[str, Any] = {}
    for name, build in builders.items():
        per_cat: Dict[str, Any] = {}
        pooled: List[float] = []
        for cat, instances in instance_corpus.items():
            qword = "television" if cat == "tv_monitor" else cat
            cat_gaps: List[float] = []
            for gi, goal_caps in enumerate(instances):
                if len(goal_caps) < 2:
                    continue
                distractors = [instances[dj] for dj in range(len(instances))
                               if dj != gi and instances[dj]]
                if not distractors:
                    continue
                for hi in range(len(goal_caps)):
                    held = goal_caps[hi]
                    others = [goal_caps[k] for k in range(len(goal_caps)) if k != hi]
                    q = build(cat, qword, others, enc)
                    goal_score = _cos(q, enc(held))
                    distractor_score = max(
                        max(_cos(q, enc(c)) for c in d) for d in distractors
                    )
                    cat_gaps.append(goal_score - distractor_score)
            gap = float(np.mean(cat_gaps)) if cat_gaps else float("nan")
            per_cat[cat] = {"rank_gap": gap, "n_samples": len(cat_gaps)}
            if cat_gaps:
                pooled.append(gap)
        results[name] = {
            "per_category": per_cat,
            "pooled_rank_gap": float(np.mean(pooled)) if pooled else float("nan"),
        }
    return results


def recommend_query_variant(
    ab_results: Dict[str, Any],
    baseline: str = "bare_category",
    pooled_margin: float = 0.02,
    cat_margin: float = 0.02,
    min_cats: int = 2,
) -> Dict[str, Any]:
    """Pick the best query variant that beats the baseline by ``pooled_margin``
    on the pooled gap AND by ``cat_margin`` on at least ``min_cats`` categories;
    else return an honest-negative verdict (winner ``None``).
    """
    base = ab_results.get(baseline, {})
    base_pooled = base.get("pooled_rank_gap", float("nan"))
    base_cat = {c: v["rank_gap"] for c, v in base.get("per_category", {}).items()}
    cands: List[Any] = []
    for name, res in ab_results.items():
        if name == baseline:
            continue
        pooled = res.get("pooled_rank_gap", float("nan"))
        if not (np.isfinite(pooled) and np.isfinite(base_pooled)):
            continue
        beats_pooled = pooled > base_pooled + pooled_margin
        n_beat = 0
        for c, v in res.get("per_category", {}).items():
            g = v.get("rank_gap", float("nan"))
            bg = base_cat.get(c, float("nan"))
            if np.isfinite(g) and np.isfinite(bg) and g > bg + cat_margin:
                n_beat += 1
        if beats_pooled and n_beat >= min_cats:
            cands.append((name, pooled, n_beat))
    cands.sort(key=lambda t: t[1], reverse=True)
    if not cands:
        return {
            "winner": None, "baseline_pooled": base_pooled,
            "verdict": (
                "HONEST NEGATIVE: no query variant beats the bare-category baseline "
                f"by >{pooled_margin:.2f} on the pooled rank gap AND on >={min_cats} "
                "categories -> do NOT spend a RACE run; the query-side lever does not "
                "clear the bar on this corpus."
            ),
        }
    name, pooled, n_beat = cands[0]
    return {
        "winner": name, "pooled": pooled, "baseline_pooled": base_pooled,
        "n_cat_beat": n_beat,
        "verdict": (
            f"RECOMMEND query variant '{name}' (pooled rank gap {pooled:+.3f} vs "
            f"baseline {base_pooled:+.3f}, beats baseline on {n_beat} categories) -> "
            "wire it default-OFF in propose_memory_candidates and run ONE S3-only RACE "
            "A/B against the cached baseline."
        ),
    }


def query_ab_report(encode: Callable[[str], np.ndarray]) -> Dict[str, Any]:
    """Print the query-construction A/B table + RECOMMEND line; return stats."""
    enc = _encode_cached(encode)
    ab = query_template_ab(INSTANCE_CORPUS, enc)
    rec = recommend_query_variant(ab)
    cats = list(INSTANCE_CORPUS.keys())
    print("Query-construction A/B (Stage-0 query-fix selection) — leave-one-out")
    print("goal-vs-best-distractor instance rank gap on INSTANCE_CORPUS "
          "(all-MiniLM-L6-v2)")
    print("  (bare_category = the live query; caption/prf_interp use the agent's")
    print("   OWN prior sightings; hyde/attribute = generic enrichment)\n")
    header = "  {:<14}".format("variant") + \
        "".join(" {:>8}".format(c[:8]) for c in cats) + "   {:>8}".format("POOLED")
    print(header)
    for name in ab:
        row = "  {:<14}".format(name)
        for c in cats:
            g = ab[name]["per_category"].get(c, {}).get("rank_gap", float("nan"))
            row += " {:>8}".format(f"{g:+.3f}" if np.isfinite(g) else "n/a")
        p = ab[name]["pooled_rank_gap"]
        row += "   {:>8}".format(f"{p:+.3f}" if np.isfinite(p) else "n/a")
        print(row)
    print(f"\n  RECOMMEND: {rec['verdict']}\n")
    return {"ab": ab, "recommend": rec}


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


# ----------------------------------------------------------------------
# Phase 0 — captioner-swap GATE. Does a CANDIDATE captioner (e.g. CapRL-3B)
# widen the within-vs-between instance separation vs the CURRENT one
# (Qwen2-VL-2B)? Consumes a captions-by-instance file built offline by
# build_instance_caption_corpus.py (real HM3D keyframes captioned by BOTH
# models) and reuses instance_separability / caption_to_caption_rank_gap per
# captioner. This is the $0 gate that decides whether the swap is worth a GPU
# matrix run — if CapRL does NOT widen the gap, the ceiling is the embedding/
# query, not the caption, and we pivot to a retriever fix instead of swapping.
# ----------------------------------------------------------------------
def load_caption_records(path: str) -> List[Dict[str, Any]]:
    """Load the captions-by-instance file: a list of records, or {"records": [...]}.

    Each record: {captioner, scene, category, object_id, caption[, viewpoint_idx]}.
    """
    with open(path) as f:
        data = json.load(f)
    recs = data["records"] if isinstance(data, dict) and "records" in data else data
    if not isinstance(recs, list):
        raise ValueError(f"caption records must be a list (or {{'records': [...]}}), "
                         f"got {type(recs).__name__}")
    return recs


def captions_to_instance_corpus(records, captioner) -> Dict[str, List[List[str]]]:
    """Reshape flat caption records into the {cat_key: [[inst captions], ...]}
    instance corpus for ONE captioner.

    ``cat_key = "<scene>/<category>"`` so between-instance pairs stay WITHIN one
    scene+category (different chairs in the same room, never across scenes), and
    instances are grouped by ``object_id``. Blank captions are dropped.
    """
    by_key: Dict[str, Dict[Any, List[str]]] = {}
    skipped = 0
    for r in records:
        if r.get("captioner") != captioner:
            continue
        if any(r.get(k) is None for k in ("scene", "category", "object_id")):
            skipped += 1  # malformed record — skip rather than KeyError
            continue
        cap = (r.get("caption") or "").strip()
        if not cap:
            continue
        key = f"{r['scene']}/{r['category']}"
        by_key.setdefault(key, {}).setdefault(r["object_id"], []).append(cap)
    if skipped:
        print(f"  [warn] {captioner}: skipped {skipped} record(s) missing scene/category/object_id")
    return {key: list(insts.values()) for key, insts in by_key.items()}


def _caption_gate_verdict(result, candidate, baseline, d_sep, d_gap, margin, stats) -> str:
    if result == "INSUFFICIENT":
        return (f"INSUFFICIENT DATA: {baseline} has {stats[baseline]['n_cells']} and {candidate} "
                f"{stats[candidate]['n_cells']} scene/category cell(s) with measurable within+between "
                f"pairs (Δsep={d_sep:+.3f}). The corpus is too thin OR a captioner emitted no captions "
                f"(model load/stub failure?) — the gate is MEANINGLESS; fix the corpus before deciding.")
    if result == "HOLD":
        return (f"HOLD: {candidate} does NOT widen instance separation over {baseline} "
                f"(Δsep={d_sep:+.3f} <= +{margin:.2f}). The ceiling is the EMBEDDING/QUERY, not the "
                f"caption — do NOT spend a GPU matrix on the swap; pivot to an instance-aware / "
                f"asymmetric retriever (the read-side query fix) instead.")
    if d_gap > margin:
        return (f"GO: {candidate} widens instance separation by {d_sep:+.3f} (> +{margin:.2f}) AND the "
                f"caption-to-caption rank gap by {d_gap:+.3f}. The richer captions carry more instance "
                f"signal that survives retrieval — worth the Phase-1 fit-smoke + held A/B.")
    return (f"GO (write-side only): {candidate} widens instance separation by {d_sep:+.3f} (> +{margin:.2f}) "
            f"but the caption-to-caption rank gap barely moves (Δ={d_gap:+.3f}). The captioner helps the "
            f"WRITE side; PAIR it with the read-side query fix — run the A/B but expect query construction "
            f"to be the second half of the gain.")


def compare_captioners(records, encode, *, baseline: str, candidate: str,
                       margin: float = 0.02) -> Dict[str, Any]:
    """Per-captioner instance separation + a GATE verdict on the candidate.

    For each captioner: ``separation = within-instance − between-instance(same
    scene+category)`` cosine, plus the caption-to-caption rank gap (the realistic
    warm-revisit retrieval signal). GATE = the candidate widens the separation by
    > ``margin`` (its captions carry MORE instance signal, so the swap is
    justified). The rank-gap delta is also reported: separation up but rank gap
    flat ⇒ the captioner helps the write side but a read-side query fix is still
    needed.
    """
    enc = _encode_cached(encode)
    out: Dict[str, Any] = {}
    for cap in (baseline, candidate):
        corpus = captions_to_instance_corpus(records, cap)
        sep = instance_separability(corpus, enc)
        c2c = caption_to_caption_rank_gap(corpus, enc)
        out[cap] = {
            "n_cells": len(corpus),
            "within_mean": sep["within_mean"],
            "between_mean": sep["between_mean"],
            "separation": sep["separation"],
            "c2c_rank_gap": c2c["mean_rank_gap"],
            "per_category": sep["per_category"],
        }
    bs, cs = out[baseline]["separation"], out[candidate]["separation"]
    d_sep = cs - bs
    d_gap = out[candidate]["c2c_rank_gap"] - out[baseline]["c2c_rank_gap"]
    # INSUFFICIENT when a captioner has no measurable within+between pairs (empty
    # corpus / all-singleton / a stub-loaded captioner that emitted nothing) -> the
    # separation is NaN and a bare `NaN > margin` would masquerade as a HOLD.
    if (bs != bs) or (cs != cs):  # NaN check
        result = "INSUFFICIENT"
    elif d_sep > margin:
        result = "GO"
    else:
        result = "HOLD"
    out["delta_separation"] = d_sep
    out["delta_c2c_rank_gap"] = d_gap
    out["result"] = result
    out["gate_pass"] = (result == "GO")
    out["verdict"] = _caption_gate_verdict(result, candidate, baseline, d_sep, d_gap, margin, out)
    return out


def compare_captioners_report(records, encode, *, baseline, candidate, margin) -> Dict[str, Any]:
    """Print the captioner-swap gate table + verdict; return the raw stats."""
    res = compare_captioners(records, encode, baseline=baseline, candidate=candidate, margin=margin)
    b, c = res[baseline], res[candidate]
    print(f"CAPTIONER-SWAP GATE (Phase 0): {candidate} vs {baseline}  [all-MiniLM-L6-v2]")
    print("  within-instance = two captions of the SAME object; "
          "between = DIFFERENT objects of the same scene+category.\n")
    keys = sorted(set(b["per_category"]) | set(c["per_category"]))
    print(f"  {'scene/category':<28} {(baseline+'_sep'):>14} {(candidate+'_sep'):>14} {'Δsep':>8}")
    for k in keys:
        bsep = b["per_category"].get(k, {}).get("separation", float("nan"))
        csep = c["per_category"].get(k, {}).get("separation", float("nan"))
        dd = (csep - bsep) if (bsep == bsep and csep == csep) else float("nan")
        print(f"  {k:<28} {bsep:>+14.3f} {csep:>+14.3f} {dd:>+8.3f}")
    print(f"\n  POOLED separation:           {baseline}={b['separation']:+.3f}  "
          f"{candidate}={c['separation']:+.3f}  Δ={res['delta_separation']:+.3f}")
    print(f"  caption-to-caption rank gap: {baseline}={b['c2c_rank_gap']:.3f}  "
          f"{candidate}={c['c2c_rank_gap']:.3f}  Δ={res['delta_c2c_rank_gap']:+.3f}")
    print(f"\n  GATE: {res['verdict']}\n")
    # machine-readable marker on its own line — the driver greps this, NOT the prose.
    print(f"GATE_RESULT={res['result']}")
    return res


# ----------------------------------------------------------------------
# Phase 3a — encoder-swap GATE. Phase 0 showed the CAPTIONER is not the
# instance-discrimination bottleneck (CapRL-3B was HOLD); this gate tests the
# READ side: with the captions FIXED (the production Qwen captions), does a
# stronger TEXT EMBEDDER widen the within-vs-between instance separation over the
# current SBERT all-MiniLM-L6-v2? GO -> swap the fine-index encoder; HOLD -> not
# the text encoder either (the next lever is a VISUAL instance embedder like
# DINOv3, or the instance-aware query). Reuses the SAME captions corpus — no
# re-render / re-caption.
# ----------------------------------------------------------------------
# A small spread of robust, no-remote-code, no-prefix text embedders (plain
# symmetric .encode() — see compare_encoders). bge-large is the strong top-end so a
# HOLD means "even the strongest text encoder didn't separate the instances". (gte-v1.5
# was dropped: it needs trust_remote_code and is broken on recent sentence-transformers.)
DEFAULT_ENCODERS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",   # baseline (production, 384-d)
    "bge-base": "BAAI/bge-base-en-v1.5",                            # 768-d
    "bge-large": "BAAI/bge-large-en-v1.5",                          # 1024-d, strong + robust
    "qwen3-emb-0.6b": "Qwen/Qwen3-Embedding-0.6B",                 # newest small (1024-d)
}


def parse_encoders(pairs) -> Dict[str, str]:
    """``["label=hf/model", ...]`` -> ``{label: model_id}`` (default shortlist if empty)."""
    if not pairs:
        return dict(DEFAULT_ENCODERS)
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"--encoders entry must be label=model_id, got {p!r}")
        label, model = p.split("=", 1)
        out[label.strip()] = model.strip()
    return out


def build_named_encoder(model_name: str):
    """Return an encode(str)->vec fn for an arbitrary sentence-transformers model,
    or None if it fails to load (so the gate skips it rather than crashing)."""
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name, trust_remote_code=True)
        return lambda s: np.asarray(m.encode(s), dtype=np.float32)
    except Exception as e:  # noqa: BLE001 — any load failure -> skip this encoder
        print(f"  [warn] could not load encoder {model_name!r}: {e}")
        return None


def _encoder_gate_verdict(result, best, baseline, d_sep, margin, stats) -> str:
    if result == "INSUFFICIENT":
        return (f"INSUFFICIENT: the baseline encoder {baseline} or all candidates failed to load / "
                f"produced no measurable separation — the gate is meaningless; check the encoder loads.")
    if result == "HOLD":
        return (f"HOLD: NO candidate encoder widens instance separation over {baseline} "
                f"(best={best}, Δsep={d_sep:+.3f} <= +{margin:.2f}). The ceiling is NOT the TEXT embedder "
                f"either — the captions don't carry separable instance signal for these categories. The "
                f"next lever is a VISUAL instance embedder (DINOv3) indexing the keyframe IMAGE, or the "
                f"instance-aware query — NOT another text encoder.")
    return (f"GO: encoder {best} widens instance separation over {baseline} by {d_sep:+.3f} "
            f"(> +{margin:.2f}). The TEXT EMBEDDING was a limiter — swap the SBERT fine-index encoder to "
            f"{best} (the dialogue_memory.encoder seam) and A/B the warm-revisit matrix vs +0.2505.")


def compare_encoders(records, encoders, *, captioner: str, baseline: str,
                     margin: float = 0.02) -> Dict[str, Any]:
    """Per-ENCODER instance separation on FIXED ``captioner`` captions + a GATE on
    the best candidate vs ``baseline``.

    ``encoders``: ``{label: encode_fn_or_None}``. GATE = the best loaded candidate
    encoder widens the within-vs-between separation by > ``margin`` over baseline.
    The measurement is deliberately SYMMETRIC — plain ``encode(caption)`` with no
    query/passage instruction (caption-vs-caption has no retrieval roles), so do not
    "fix" it by bolting on an asymmetric query prompt. Candidate encoders that failed
    to load (None) are tracked in ``dropped`` so a silently-missing encoder can't
    masquerade as a thinner-but-clean HOLD.
    """
    corpus = captions_to_instance_corpus(records, captioner)  # captions held FIXED
    stats: Dict[str, Any] = {}
    for label, enc in encoders.items():
        if enc is None:
            stats[label] = {"loaded": False}
            continue
        e = _encode_cached(enc)
        sep = instance_separability(corpus, e)
        c2c = caption_to_caption_rank_gap(corpus, e)
        stats[label] = {"loaded": True, "separation": sep["separation"],
                        "within_mean": sep["within_mean"], "between_mean": sep["between_mean"],
                        "c2c_rank_gap": c2c["mean_rank_gap"], "per_category": sep["per_category"]}
    requested = [l for l in encoders if l != baseline]
    dropped = [l for l in requested if not stats.get(l, {}).get("loaded")]
    bs = stats.get(baseline, {}).get("separation", float("nan"))
    base_ok = stats.get(baseline, {}).get("loaded") and (bs == bs)
    cands = {l: v for l, v in stats.items()
             if l != baseline and v.get("loaded") and v.get("separation") == v.get("separation")}
    if not base_ok or not cands:
        result, best, d_sep = "INSUFFICIENT", None, float("nan")
    else:
        best = max(cands, key=lambda l: cands[l]["separation"])
        d_sep = cands[best]["separation"] - bs
        result = "GO" if d_sep > margin else "HOLD"
    return {"encoders": stats, "baseline": baseline, "best_candidate": best,
            "delta_separation": d_sep, "result": result, "gate_pass": result == "GO",
            "n_candidates_requested": len(requested), "n_candidates_loaded": len(requested) - len(dropped),
            "dropped": dropped,
            "verdict": _encoder_gate_verdict(result, best, baseline, d_sep, margin, stats)}


def compare_encoders_report(records, encoders, *, captioner, baseline, margin) -> Dict[str, Any]:
    """Print the encoder-swap gate table + verdict; return the raw stats."""
    res = compare_encoders(records, encoders, captioner=captioner, baseline=baseline, margin=margin)
    bs = res["encoders"].get(baseline, {}).get("separation", float("nan"))
    print(f"ENCODER-SWAP GATE (Phase 3a): captioner={captioner} FIXED, varying the fine-index encoder\n")
    print(f"  {'encoder':<40} {'within':>7} {'betwn':>7} {'sep':>7} {'c2c_gap':>8} {'Δsep_vs_base':>13}")
    for label, v in res["encoders"].items():
        if not v.get("loaded"):
            print(f"  {label:<40} {'(failed to load — skipped)':>53}")
            continue
        d = (v["separation"] - bs) if (bs == bs) else float("nan")
        tag = "  <- baseline" if label == baseline else ""
        print(f"  {label:<40} {v['within_mean']:>7.3f} {v['between_mean']:>7.3f} "
              f"{v['separation']:>+7.3f} {v['c2c_rank_gap']:>8.3f} {d:>+13.3f}{tag}")
    if res["dropped"]:
        print(f"\n  [WARN] only {res['n_candidates_loaded']}/{res['n_candidates_requested']} candidate "
              f"encoders loaded — DROPPED: {', '.join(res['dropped'])}. A HOLD here may be missing the "
              f"strongest encoder; install/fix them and re-run before trusting a HOLD.")
    print(f"\n  GATE: {res['verdict']}\n")
    print(f"GATE_RESULT={res['result']}")
    return res


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="SBERT instance-separation diagnostics + the captioner/encoder swap gates")
    ap.add_argument("--compare-captions", metavar="JSON", default=None,
                    help="captions-by-instance file; runs ONLY the captioner-swap gate (Phase 0)")
    ap.add_argument("--compare-encoders", metavar="JSON", default=None,
                    help="captions-by-instance file; runs ONLY the encoder-swap gate (Phase 3a) — "
                         "fixes the captioner, varies the SBERT fine-index encoder")
    ap.add_argument("--baseline", default="qwen2-vl-2b", help="baseline CAPTIONER label (Phase 0)")
    ap.add_argument("--candidate", default="caprl-3b", help="candidate CAPTIONER label (Phase 0)")
    ap.add_argument("--captioner", default="qwen2-vl-2b",
                    help="which captioner's captions to FIX for the encoder gate (Phase 3a)")
    ap.add_argument("--encoders", nargs="+", default=[],
                    help="label=hf/model encoder pairs for Phase 3a (default: a small shortlist)")
    ap.add_argument("--baseline-encoder", default="all-MiniLM-L6-v2",
                    help="baseline ENCODER label = the production fine-index encoder (Phase 3a)")
    ap.add_argument("--margin", type=float, default=0.02, help="min Δseparation for GATE=GO")
    args = ap.parse_args(argv)

    if args.compare_captions:
        encode = _build_encoder()
        records = load_caption_records(args.compare_captions)
        compare_captioners_report(records, encode, baseline=args.baseline,
                                  candidate=args.candidate, margin=args.margin)
        return 0

    if args.compare_encoders:
        records = load_caption_records(args.compare_encoders)
        specs = parse_encoders(args.encoders)
        encoders = {label: build_named_encoder(mid) for label, mid in specs.items()}
        compare_encoders_report(records, encoders, captioner=args.captioner,
                                baseline=args.baseline_encoder, margin=args.margin)
        return 0

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

    # ------------------------------------------------------------------
    # Stage-0 query-fix selection: A/B query-construction variants on the
    # SAME instance corpus and RECOMMEND a winner (or honest negative). This
    # is the $0 gate that must clear the bar BEFORE wiring a query-side fix in
    # propose_memory_candidates and spending a RACE A/B run.
    # ------------------------------------------------------------------
    print("\n" + "=" * 70 + "\n")
    query_ab_report(encode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
