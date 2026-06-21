"""
check_seed_not_los — the DECISIVE ``$0`` Tier-3 gate for a non-LOS lifelong build.

The non-LOS seed picker (``make_audiogoal_smoke.pick_non_los_seed``) moves the seed
to an audible-but-occluded RIR cell, but a geodesic-detour ratio is only a PROXY for
occlusion. This gate is the adjudicator: it asks the question that actually matters
for the redundancy argument — *does vision map the source from the seed pose?* If the
seed's caption already names the goal object (or scores high against the goal query),
then write-OFF will visually map the source anyway and the oracle audio write stays
redundant → the build is RED and no GPU matrix should run.

The verdict is GREEN iff, at the seed pose, the goal-object token is ABSENT from the
caption AND the SBERT cosine of the caption against ``"there is a {goal}"`` is below
``cos_bar`` (default 0.23 — the same retrieval bar ``memory_bridge`` uses). This file
holds the pure, unit-tested decision (``seed_not_los_verdict`` /
``goal_query_cosine``); ``main`` is a thin operator front-end that takes the seed-pose
caption (produced by a one-step run of the real captioner / read off the video HUD)
and either uses a supplied ``--cos`` or computes it with the runner's SBERT encoder.

    # GREEN (exit 0) iff vision does NOT map the source from the seed:
    python embodied_memory/scripts/check_seed_not_los.py \
        --goal bed --caption "a hallway with a closed door and a rug"
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, List, Optional, Tuple

import numpy as np

_GOAL_QUERY_TEMPLATE = "there is a {}"   # matches memory_bridge._GOAL_QUERY_TEMPLATE
_DEFAULT_COS_BAR = 0.23                   # matches the live retrieval gate
_DEFAULT_MODEL = "all-MiniLM-L6-v2"       # matches run_hm3d_pol._build_text_encoder


def seed_not_los_verdict(
    caption: str,
    goal_category: str,
    goal_cos: Optional[float],
    *,
    cos_bar: float = _DEFAULT_COS_BAR,
) -> Tuple[bool, str]:
    """GREEN iff vision does NOT map the source from the seed: the goal-object token
    is ABSENT from the caption AND ``goal_cos < cos_bar``. Returns ``(ok, reason)``.
    A missing/``None`` cosine is treated as below the bar (token check still applies)."""
    cap = (caption or "").lower()
    token = (goal_category or "").lower().strip()
    token_present = bool(token) and token in cap
    cos_high = goal_cos is not None and float(goal_cos) >= cos_bar
    cos_str = f"{goal_cos:.3f}" if goal_cos is not None else "n/a"
    if token_present and cos_high:
        return False, (f"RED: caption names '{token}' AND cos {cos_str} >= {cos_bar} "
                       "→ vision maps the source from the seed (write redundant)")
    if token_present:
        return False, (f"RED: caption names goal token '{token}' "
                       "→ vision sees the source from the seed (write redundant)")
    if cos_high:
        return False, (f"RED: goal cos {cos_str} >= {cos_bar} → caption is goal-ish "
                       "→ vision likely maps the source (write redundant)")
    return True, (f"GREEN: goal token '{token}' absent + cos {cos_str} < {cos_bar} "
                  "→ vision does NOT map the source from the seed (write has a job)")


def goal_query_cosine(
    caption: str,
    goal_category: str,
    encode_fn: Callable[[List[str]], "np.ndarray"],
    *,
    template: str = _GOAL_QUERY_TEMPLATE,
) -> float:
    """Cosine of ``caption`` vs ``"there is a {goal}"`` under ``encode_fn`` (a
    ``list[str] -> (2, d)`` text encoder). Normalizes both vectors so the result is a
    true cosine even if the encoder does not L2-normalize. Returns 0.0 on a degenerate
    (zero-norm) embedding."""
    query = template.format(goal_category)
    embs = np.asarray(encode_fn([caption, query]), dtype=np.float64)
    if embs.ndim != 2 or embs.shape[0] != 2:
        raise ValueError(f"encode_fn must return (2, d); got {embs.shape}")
    a, b = embs[0], embs[1]
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _build_encoder(model_name: str = _DEFAULT_MODEL) -> Callable[[List[str]], "np.ndarray"]:
    """The runner's SBERT text encoder (all-MiniLM-L6-v2). Lazy — only imported when
    ``--cos`` is not supplied, so the pure helpers stay dependency-free."""
    from dialogue_memory.encoder import SentenceTransformerEncoder
    enc = SentenceTransformerEncoder(model_name=model_name)
    return lambda texts: np.asarray(enc.encode(texts), dtype=np.float64)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="$0 seed-not-LOS captioner gate")
    ap.add_argument("--goal", required=True, help="goal object category, e.g. bed")
    ap.add_argument("--caption", required=True,
                    help="the seed-pose caption (from a 1-step run of the real "
                         "captioner / the video HUD)")
    ap.add_argument("--cos", type=float, default=None,
                    help="supply the goal-query cosine directly (skip the encoder)")
    ap.add_argument("--cos-bar", type=float, default=_DEFAULT_COS_BAR)
    ap.add_argument("--model", default=_DEFAULT_MODEL)
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    cos = args.cos
    if cos is None:
        cos = goal_query_cosine(args.caption, args.goal, _build_encoder(args.model))
    ok, reason = seed_not_los_verdict(args.caption, args.goal, cos, cos_bar=args.cos_bar)
    print(f"[check-seed-not-los] goal={args.goal!r} cos={cos:.3f} bar={args.cos_bar}")
    print(f"  caption: {args.caption!r}")
    print(f"  {reason}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
