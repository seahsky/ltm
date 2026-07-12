"""
Calibrate the coarse head's CLIP zero-shot ROOM classifier on REAL HM3D keyframes
(step-4, Stage-5 MF-1). Sim + CLIP only — NO 7B backbone, so it is fast.

WHY: ``classify_room_clip`` gates on ``top_cos >= min_cos`` AND ``(top - second) >=
margin``. Those defaults (0.25 / 0.02) are a PRIOR for the CLIP ViT-B/32 image-text
scale (~0.18-0.30) — they must be SET FROM DATA before the cross-env / revisit run,
or the abstain gate is either a no-op (fires on every frame -> over-fire) or shut
(never fires -> the coarse head stays inert, the coarse-1/2 failure). The adversarial
review made this the single blocker to running on RACE.

WHAT it does, per scene: walks the agent (turn+forward) to traverse rooms, encodes
each RGB keyframe with the SAME ``CLIPKeyframeEncoder`` the loop uses, scores it
against ``build_room_text_embeddings`` (the 6 room prompts), and dumps the
distributions of (a) the top room cosine, (b) the top-minus-second margin, and (c)
the argmax-room histogram. Read the percentiles to choose ``min_cos`` / ``margin``:

  - min_cos  ~ the 50-70th percentile of top_cos (admit clearly-room-like frames,
               abstain on the bottom half).
  - margin   ~ a value that the top-minus-second distribution clears only when the
               frame leans to ONE room (e.g. its 50-60th percentile), so noise-wins
               are abstained.
  - room hist: if it COLLAPSES to one room (e.g. all 'hallway'), the dense signal is
               uninformative even when it "fires" — a NO-GO for the affordance prior.
  - any top_cos > 1.0 => a normalization/dtype bug (unit-norm cosine is bounded by 1).

Run on RACE (after ``source scripts/race-setup.sh``):
    python embodied_memory/scripts/diagnose_room_clip_cosines.py --scene all \
        --episodes-path data/hm3d/datasets/objectnav/hm3d/v1/val_mini/val_mini.json.gz \
        --n-scenes 2 --steps 120
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, List, Optional

import numpy as np

# Run as a script -> put the repo root on sys.path so `import embodied_memory.*` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _pct(xs: List[float], ps=(5, 10, 25, 50, 60, 70, 75, 90, 95)) -> str:
    if not xs:
        return "(no samples)"
    a = np.asarray(xs, dtype=np.float64)
    return "  ".join(f"p{p}={np.percentile(a, p):.3f}" for p in ps)


# ----------------------------------------------------------------------
# G0.1 — scene-conditioning kill-switch (keyword #16)
# ----------------------------------------------------------------------
# A room-conditioned anomaly gate is only trustworthy if the CLIP room classifier
# reliably tells the sound's normal-room from its anomalous-room. These pure
# functions turn (true_room, pred_room) pairs — ground truth from the object
# category's CATEGORY_ROOM_PRIOR, prediction from classify_room_clip — into a
# pairwise accuracy + a GO/BORDERLINE/STOP gate verdict.

def room_pair_accuracy(pairs, rooms) -> dict:
    """Accuracy of the room classifier over frames whose TRUE room is in ``rooms``.

    ``pairs`` is a sequence of ``(true_room, pred_room)`` where ``pred_room`` may be
    ``None`` (abstain). A prediction is correct iff ``pred_room == true_room``; an
    abstain counts as wrong (an abstain means the gate cannot tell the rooms apart).
    Frames whose true room is not in ``rooms`` are ignored.
    """
    rooms = set(rooms)
    n = n_correct = n_abstain = 0
    confusion: dict = {}
    for true_room, pred_room in pairs:
        if true_room not in rooms:
            continue
        n += 1
        confusion[(true_room, pred_room)] = confusion.get((true_room, pred_room), 0) + 1
        if pred_room is None:
            n_abstain += 1
        elif pred_room == true_room:
            n_correct += 1
    return {
        "n": n,
        "n_correct": n_correct,
        "n_abstain": n_abstain,
        "accuracy": (n_correct / n) if n else 0.0,
        "abstain_rate": (n_abstain / n) if n else 0.0,
        "confusion": confusion,
    }


def room_gate_verdict(accuracy: float, *, go: float = 0.75, borderline: float = 0.60) -> str:
    """GO / BORDERLINE / STOP for the scene-conditioning gate.

    GO (>= ``go``) — the classifier separates the two rooms reliably enough to
    trust a room-conditioned anomaly verdict. STOP (< ``borderline``) — drop #16
    to future work; keep the context-free gate. BORDERLINE in between.
    """
    if accuracy >= go:
        return "GO"
    if accuracy >= borderline:
        return "BORDERLINE"
    return "STOP"


def _walk_actions(n: int) -> List[int]:
    """A deterministic turn-heavy walk so the agent sees several rooms: a few
    forwards then a turn, repeating. 1=move_forward, 2=turn_left, 3=turn_right."""
    out: List[int] = []
    for i in range(n):
        out.append(1 if (i % 4) != 3 else (2 if (i // 4) % 2 == 0 else 3))
    return out


def _probe_scene(scene: str, episodes_path: Optional[str], scene_dataset_path: Optional[str],
                 enc, room_text, steps: int, min_cos: float, margin: float) -> dict:
    from embodied_memory.habitat_env import HabitatObjectNavSource
    from embodied_memory.room_resolver import classify_room_clip, room_clip_top_cos

    src = HabitatObjectNavSource(
        scene_id=scene, scene_dataset_path=scene_dataset_path,
        episodes_path=episodes_path, n_episodes=1, target_category=None,
    )
    top_cos: List[float] = []
    margins: List[float] = []
    argmax_hist: Counter = Counter()
    fired_default: Counter = Counter()   # room when classify_room_clip(default) fires
    n_abstain_default = 0
    n_frames = 0
    try:
        step, _ = src.reset(0)
        frames = [step.rgb] if getattr(step, "rgb", None) is not None else []
        for a in _walk_actions(steps):
            try:
                step = src.step(a)
            except Exception:
                break
            if getattr(step, "rgb", None) is not None:
                frames.append(step.rgb)
        for rgb in frames:
            try:
                vemb = enc.encode(rgb)
            except Exception as e:
                print(f"  encode failed: {e}")
                continue
            n_frames += 1
            cos, room = room_clip_top_cos(vemb, room_text)
            if room is not None and cos == cos:
                top_cos.append(float(cos))
                argmax_hist[room] += 1
            # second-best for the margin distribution
            from embodied_memory.room_resolver import _room_cosines
            sims = _room_cosines(vemb, room_text)
            if len(sims) >= 2:
                margins.append(float(sims[0][0] - sims[1][0]))
            tag = classify_room_clip(vemb, room_text, min_cos=min_cos, margin=margin)
            if tag is None:
                n_abstain_default += 1
            else:
                fired_default[tag] += 1
    except Exception as e:
        print(f"  FAILED scene {scene}: {e!r}")
    finally:
        try:
            src.close()
        except Exception:
            pass

    print(f"  frames encoded: {n_frames}")
    print(f"  top_cos      : {_pct(top_cos)}")
    print(f"  top-second   : {_pct(margins)}")
    print(f"  max top_cos  : {max(top_cos) if top_cos else float('nan'):.3f}"
          + ("   <-- WARNING >1.0 = normalization bug" if (top_cos and max(top_cos) > 1.0) else ""))
    print(f"  argmax-room histogram (no gate): {dict(argmax_hist.most_common())}")
    print(f"  @ default gate (min_cos={min_cos}, margin={margin}): "
          f"fired={sum(fired_default.values())} abstained={n_abstain_default}  "
          f"fired-rooms={dict(fired_default.most_common())}")
    return {"top_cos": top_cos, "margins": margins, "argmax_hist": argmax_hist,
            "fired": fired_default, "abstain": n_abstain_default, "n_frames": n_frames}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Calibrate the CLIP room classifier on real HM3D keyframes")
    p.add_argument("--scene", default="all")
    p.add_argument("--episodes-path", default=None)
    p.add_argument("--scene-dataset-path", default=None)
    p.add_argument("--n-scenes", type=int, default=2)
    p.add_argument("--steps", type=int, default=120, help="walk length per scene")
    p.add_argument("--min-cos", type=float, default=0.25, help="default-gate min_cos to report fire-rate at")
    p.add_argument("--margin", type=float, default=0.02, help="default-gate margin to report fire-rate at")
    args = p.parse_args(argv)

    from embodied_memory.run_hm3d_pol import _resolve_scene_list
    from embodied_memory.perception import CLIPKeyframeEncoder
    from embodied_memory.room_resolver import build_room_text_embeddings, ROOM_TEXT_PROMPTS

    scenes = _resolve_scene_list(args.scene, args.episodes_path)
    print(f"discovered scenes: {scenes}")
    scenes = scenes[: max(1, args.n_scenes)]

    enc = CLIPKeyframeEncoder()
    room_text = build_room_text_embeddings(enc.encode_text)
    print(f"room prompts: {list(ROOM_TEXT_PROMPTS.values())}")
    print(f"CLIP model: {enc.model_name} device={enc.device}")

    agg_top: List[float] = []
    agg_margin: List[float] = []
    agg_hist: Counter = Counter()
    agg_fired: Counter = Counter()
    agg_abstain = 0
    for scene in scenes:
        print(f"\n========== scene {scene} ==========")
        r = _probe_scene(scene, args.episodes_path, args.scene_dataset_path, enc,
                         room_text, args.steps, args.min_cos, args.margin)
        agg_top += r["top_cos"]; agg_margin += r["margins"]
        agg_hist += r["argmax_hist"]; agg_fired += r["fired"]; agg_abstain += r["abstain"]

    print("\n==================== AGGREGATE ====================")
    print(f"  frames           : {len(agg_top)}")
    print(f"  top_cos          : {_pct(agg_top)}")
    print(f"  top-second margin: {_pct(agg_margin)}")
    print(f"  argmax-room hist : {dict(agg_hist.most_common())}")
    print(f"  @ default gate (min_cos={args.min_cos}, margin={args.margin}): "
          f"fired={sum(agg_fired.values())} abstained={agg_abstain}  "
          f"fired-rooms={dict(agg_fired.most_common())}")
    n = len(agg_top)
    if n:
        fire_rate = sum(agg_fired.values()) / max(1, n)
        print(f"  default-gate fire-rate = {fire_rate:.2%}")
        print("\n  GUIDANCE: pick min_cos near the 50-70th pct of top_cos and margin near the")
        print("  50-60th pct of top-second; aim for a fire-rate that is NEITHER ~0% (inert,")
        print("  coarse-1/2 failure) NOR ~100% (over-fire, no confidence floor). A COLLAPSED")
        print("  argmax histogram (one room dominates) means the dense signal is uninformative.")
        if agg_top and max(agg_top) > 1.0:
            print("  *** top_cos > 1.0 SEEN -> normalization/dtype BUG, fix before trusting any of this.")
        # Machine-parseable data-driven thresholds (a driver greps this line):
        # min_cos = p50(top_cos) admits the top half of room-like frames; margin =
        # p50(top-second) commits only when a frame leans to one room. The collapse
        # flag warns a downstream driver NOT to trust an A/B at these thresholds.
        rec_min = float(np.percentile(np.asarray(agg_top), 50)) if agg_top else 0.25
        rec_margin = float(np.percentile(np.asarray(agg_margin), 50)) if agg_margin else 0.02
        top_room, top_cnt = (agg_hist.most_common(1)[0] if agg_hist else ("none", 0))
        collapsed = bool(agg_hist) and top_cnt >= 0.80 * sum(agg_hist.values())
        print(f"\nRECOMMEND min_cos={rec_min:.3f} margin={rec_margin:.3f} "
              f"fire_rate={fire_rate:.3f} collapsed={int(collapsed)} dom_room={top_room}")
    else:
        print("  NO frames encoded — check scene loading / CLIP availability.")
        print("RECOMMEND min_cos=0.250 margin=0.020 fire_rate=0.000 collapsed=0 dom_room=none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
