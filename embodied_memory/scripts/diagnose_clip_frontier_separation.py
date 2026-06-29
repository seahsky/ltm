"""
diagnose_clip_frontier_separation — a $0/cheap GATE for the CLIP semantic-frontier
value-map lever.

Before building a CLIP semantic-frontier value map on top of the ObjectNav
backbone, we need a cheap, model-only decision: on THIS renderer (HM3D sim
renders), does ``cos(CLIP_image(rgb), CLIP_text("a photo of a {goal}"))`` actually
DISCRIMINATE goal-facing views from non-goal views? The whole frontier-value idea
rests on that cosine being a usable signal — if it is flat, the cheap CLIP variant
is dead on arrival.

The project has TWICE measured this cross-modal CLIP cosine as nearly FLAT on these
renders (~0.25 for a sighting vs ~0.228 for a baseline; OWLv2 at the noise floor),
so a flat result is the *expected* outcome — and it is itself decision-relevant: a
flat result means the cheap CLIP frontier lever is dead, which PROTECTS the existing
+0.2505 warm-revisit headline (do not spend a GPU matrix chasing it).

Mirrors ``diagnose_sbert_cosines.py``: a pure-logic core
(``clip_value_separation`` / ``_separation_verdict``) that is unit-testable with no
torch / CLIP / habitat, plus a render ``main(argv=None)`` (RACE-only) that emits the
machine-readable marker line ``GATE_RESULT=<GO|HOLD|INSUFFICIENT>`` for a driver to
grep — exactly like the captioner/encoder gates in ``diagnose_sbert_cosines.py``.

Decision rule (``_separation_verdict``, default ``margin=0.05``):

* GO          — ``separation = mean_goal_facing − mean_away >= margin`` (finite):
  CLIP discriminates goal-facing views on this renderer → the semantic-frontier
  lever is worth building.
* HOLD        — ``separation`` finite but ``< margin``: CLIP is non-discriminative
  on this renderer (matches the project's twice-measured flatness) → do NOT build
  the cheap CLIP variant; the result PROTECTS the existing +0.2505 headline.
* INSUFFICIENT — ``separation`` is NaN (empty / thin / all-NaN inputs): the gate is
  meaningless; render more frames before deciding.

Run (in the race-setup / ltm-embodied env, needs habitat_sim + open_clip)::

    python embodied_memory/scripts/diagnose_clip_frontier_separation.py \
        --scenes wcojb4TFT35 TEEsavR23oF --categories chair bed sofa \
        --n-viewpoints 6 --out runs/clip-frontier-gate/separation.json

The pure-logic core is unit-tested in
test_diagnose_clip_frontier_separation.py; the render path is habitat/GPU and
RACE-verified. Importing this module does NOT load torch / habitat (the heavy
imports live inside the render functions).
"""

from __future__ import annotations

import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Pure logic (no torch / CLIP / habitat) — unit-tested.
# ----------------------------------------------------------------------


def _finite(xs: Sequence[float]) -> List[float]:
    """Drop non-finite (NaN / inf) values from a sequence of floats."""
    out: List[float] = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            out.append(v)
    return out


def clip_value_separation(
    goal_facing_cos: Sequence[float], away_cos: Sequence[float]
) -> Dict[str, Any]:
    """Summarize goal-facing vs away CLIP image-text cosines.

    Returns ``{"mean_goal", "mean_away", "separation" (=mean_goal-mean_away),
    "n_goal", "n_away"}``. ``n_*`` count the FINITE samples used. If either side
    has no finite samples the corresponding mean and the ``separation`` are NaN
    (never raises — empty / NaN inputs degrade gracefully).
    """
    g = _finite(goal_facing_cos)
    a = _finite(away_cos)
    mean_goal = float(np.mean(g)) if g else float("nan")
    mean_away = float(np.mean(a)) if a else float("nan")
    if g and a:
        separation = mean_goal - mean_away
    else:
        separation = float("nan")
    return {
        "mean_goal": mean_goal,
        "mean_away": mean_away,
        "separation": separation,
        "n_goal": len(g),
        "n_away": len(a),
    }


def _separation_verdict(separation: float, margin: float = 0.05) -> str:
    """GO / HOLD / INSUFFICIENT from the separation and the GO margin.

    * INSUFFICIENT — ``separation`` is NaN / non-finite (empty or all-NaN inputs).
    * GO           — finite and ``>= margin``.
    * HOLD         — finite and ``< margin``.
    """
    try:
        sep = float(separation)
    except (TypeError, ValueError):
        return "INSUFFICIENT"
    if not math.isfinite(sep):
        return "INSUFFICIENT"
    return "GO" if sep >= margin else "HOLD"


def _verdict_prose(result: str, separation: float, margin: float) -> str:
    """Human-readable justification line paired with the machine marker."""
    if result == "GO":
        return (
            f"GO (separation={separation:+.3f} >= margin {margin:.2f}): CLIP "
            "discriminates goal-facing views from non-goal views on this renderer "
            "-> the semantic-frontier value-map lever is worth building. Wire the "
            "CLIP frontier-value head default-OFF and run a held S3 A/B."
        )
    if result == "HOLD":
        return (
            f"HOLD (separation={separation:+.3f} < margin {margin:.2f}): CLIP is "
            "non-discriminative on this renderer (matches the project's "
            "twice-measured flatness ~0.25 sighting vs ~0.228 baseline; OWLv2 at "
            "the noise floor) -> do NOT build the cheap CLIP semantic-frontier "
            "variant. The flat result is decision-relevant: it PROTECTS the "
            "existing +0.2505 warm-revisit headline (no GPU matrix to chase a dead "
            "lever)."
        )
    return (
        f"INSUFFICIENT (separation={separation}): too few finite goal-facing / away "
        "cosines to decide (empty / thin / all-NaN). Render more viewpoints across "
        "more scenes+categories before trusting this gate."
    )


def separation_report(
    goal_facing_cos: Sequence[float],
    away_cos: Sequence[float],
    margin: float = 0.05,
) -> Dict[str, Any]:
    """Print the separation table + the machine marker ``GATE_RESULT=...``.

    Returns ``{**clip_value_separation(...), "margin", "result", "verdict"}`` so a
    caller can JSON-dump it. The ``GATE_RESULT=`` line is printed on its own line
    (the grep target), exactly like the gates in diagnose_sbert_cosines.
    """
    stats = clip_value_separation(goal_facing_cos, away_cos)
    sep = stats["separation"]
    result = _separation_verdict(sep, margin)
    verdict = _verdict_prose(result, sep, margin)

    def _fmt(v: float) -> str:
        return f"{v:+.4f}" if (isinstance(v, float) and math.isfinite(v)) else "n/a"

    print("CLIP semantic-frontier separation GATE  [ViT-B/32 image-text cosine]")
    print("  goal-facing = CLIP cos at the goal-instance view_point pose;")
    print("  away        = same position, yaw rotated ~180deg (non-goal view).\n")
    print(f"  {'metric':<22} {'value':>10}")
    print(f"  {'mean_goal_facing':<22} {_fmt(stats['mean_goal']):>10}  "
          f"(n={stats['n_goal']})")
    print(f"  {'mean_away':<22} {_fmt(stats['mean_away']):>10}  "
          f"(n={stats['n_away']})")
    print(f"  {'separation':<22} {_fmt(sep):>10}")
    print(f"  {'margin (GO bar)':<22} {margin:>+10.4f}")
    print(f"\n  GATE: {verdict}\n")
    # machine-readable marker on its own line — the driver greps THIS, not the prose.
    print(f"GATE_RESULT={result}")

    return {**stats, "margin": margin, "result": result, "verdict": verdict}


# ----------------------------------------------------------------------
# Away-pose geometry (pure) — rotate a [x,y,z,w] quaternion by yaw about +Y.
# ----------------------------------------------------------------------


def yaw_rotated_quat(rotation_xyzw: Sequence[float], yaw_rad: float) -> List[float]:
    """Compose a [x,y,z,w] (scalar-last) quaternion with a yaw of ``yaw_rad`` about
    the world +Y (up) axis, returning a new [x,y,z,w] list.

    Used to build the AWAY view (same position, look the other way): pass
    ``yaw_rad = pi`` for a ~180deg turn. Pure quaternion algebra so it is testable
    without habitat. ``q_out = q_yaw * q_in`` (rotate the existing orientation by
    the yaw in the world frame).
    """
    x, y, z, w = (float(c) for c in rotation_xyzw)
    half = yaw_rad / 2.0
    # yaw about +Y: [x,y,z,w] = [0, sin(half), 0, cos(half)]
    yx, yy, yz, yw = 0.0, math.sin(half), 0.0, math.cos(half)
    # Hamilton product q_yaw * q_in (both scalar-last).
    ox = yw * x + yx * w + yy * z - yz * y
    oy = yw * y - yx * z + yy * w + yz * x
    oz = yw * z + yx * y - yy * x + yz * w
    ow = yw * w - yx * x - yy * y - yz * z
    n = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
    if n == 0.0:
        return [x, y, z, w]
    return [ox / n, oy / n, oz / n, ow / n]


# ----------------------------------------------------------------------
# Render main (RACE-only). Heavy imports (torch / habitat) live inside the
# functions reused from build_instance_caption_corpus, so importing this module
# for the unit tests never loads them.
# ----------------------------------------------------------------------


def _build_clip_encoder():
    """Return a CLIPKeyframeEncoder (lazy-loads open_clip on first encode)."""
    from embodied_memory.perception import CLIPKeyframeEncoder

    return CLIPKeyframeEncoder()


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import os

    # Reuse the proven render machinery (content load / instance discovery /
    # viewpoint sampling / sim / render) from the Phase-0 corpus builder. These
    # imports are module-level in that file but only pull habitat inside the
    # render functions, so this import stays light until we actually render.
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import build_instance_caption_corpus as bic  # noqa: E402

    ap = argparse.ArgumentParser(
        description="CLIP semantic-frontier separation GATE (GO/HOLD/INSUFFICIENT)")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--categories", nargs="+", default=["chair", "bed", "sofa", "toilet"])
    ap.add_argument("--n-viewpoints", type=int, default=6, help="frames per instance")
    ap.add_argument("--goal-prompt", default="a photo of a {goal}",
                    help="CLIP text query; {goal} is filled with the category")
    ap.add_argument("--margin", type=float, default=0.05, help="min separation for GATE=GO")
    ap.add_argument("--split", default="val_mini")
    ap.add_argument("--content-dir", default=None,
                    help="default: data/hm3d/datasets/objectnav/hm3d/v1/<split>/content")
    ap.add_argument("--out", default=None, help="write per-frame cosines + summary JSON here")
    args = ap.parse_args(argv)

    content_dir = (args.content_dir
                   or f"data/hm3d/datasets/objectnav/hm3d/v1/{args.split}/content")

    enc = _build_clip_encoder()
    # Cache the text query embedding per category (joint CLIP space, L2-normalized).
    text_cache: Dict[str, np.ndarray] = {}

    def _text_vec(category: str) -> np.ndarray:
        v = text_cache.get(category)
        if v is None:
            v = np.asarray(enc.encode_text(args.goal_prompt.format(goal=category)),
                           dtype=np.float32)
            text_cache[category] = v
        return v

    goal_facing_cos: List[float] = []
    away_cos: List[float] = []
    per_frame: List[Dict[str, Any]] = []

    for scene in args.scenes:
        cpath = os.path.join(content_dir, f"{scene}.json.gz")
        if not os.path.isfile(cpath):
            print(f"WARN: no content for {scene} at {cpath} — skipping")
            continue
        content = bic.load_content(cpath)
        glb = bic._find_glb(scene)
        if glb is None:
            print(f"WARN: no .glb for {scene} — skipping")
            continue

        # Enumerate (category, instance, viewpoint) render jobs for this scene.
        jobs: List[Dict[str, Any]] = []
        for cat in args.categories:
            for inst in bic.find_goal_instances(content, cat):
                obj_id = inst.get("object_id")
                vps = bic.sample_viewpoints(inst.get("view_points", []), args.n_viewpoints)
                for vp_i, vp in enumerate(vps):
                    pos, rot = bic.viewpoint_pose(vp)
                    jobs.append({"scene": scene, "category": cat, "object_id": obj_id,
                                 "viewpoint_idx": vp_i, "position": pos, "rotation": rot})
        if not jobs:
            print(f"  {scene}: 0 goal instances for {args.categories} — skipping")
            continue

        sim = bic.make_sim(glb)
        try:
            for j in jobs:
                # Goal-facing view: the viewpoint pose itself.
                rgb_goal = bic.render_rgb_at(sim, j["position"], j["rotation"])
                # Away view: same position, yaw rotated ~180deg.
                away_rot = yaw_rotated_quat(j["rotation"], math.pi)
                rgb_away = bic.render_rgb_at(sim, j["position"], away_rot)
                qv = _text_vec(j["category"])
                cg = float(np.dot(np.asarray(enc.encode(rgb_goal), dtype=np.float32), qv))
                ca = float(np.dot(np.asarray(enc.encode(rgb_away), dtype=np.float32), qv))
                goal_facing_cos.append(cg)
                away_cos.append(ca)
                per_frame.append({"scene": j["scene"], "category": j["category"],
                                  "object_id": j["object_id"], "viewpoint_idx": j["viewpoint_idx"],
                                  "goal_facing_cos": cg, "away_cos": ca})
        finally:
            sim.close()
        print(f"  {scene}: rendered {len(jobs)} goal/away frame pairs")

    summary = separation_report(goal_facing_cos, away_cos, margin=args.margin)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        payload = {
            "meta": {"scenes": args.scenes, "categories": args.categories,
                     "n_viewpoints": args.n_viewpoints, "goal_prompt": args.goal_prompt,
                     "margin": args.margin, "split": args.split},
            "per_frame": per_frame,
            "goal_facing_cos": goal_facing_cos,
            "away_cos": away_cos,
            "summary": summary,
        }
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=1)
        print(f"\nDONE. {len(per_frame)} frame pairs -> {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
