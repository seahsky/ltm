"""
diagnose_goal_anchored_recall — a FREE offline re-score (#2) that fixes the two
anchor bugs in ``diagnose_audio_doa_calib`` and re-measures the recall gap against
the CORRECT geometry.

WHY. The earlier recall diagnostic measured a recalled memory candidate's stored
``world_xy`` against ``target_position = goals[0].position`` (the FIRST instance's
object CENTER) at a 1.5 m ring. Two bugs:
  (a) ANCHOR: a stored memory candidate IS a caption-time VIEWING POSE, ~1-1.5 m
      off the object center, so a *correct* recall is mislabeled "absent" against
      the center; the success metric ``distance_to_goal`` is geodesic-to-VIEW_POINT,
      a different anchor — so the two disagree by exactly that offset.
  (b) RE-KEY: ``goals[0]`` is an arbitrary instance, not the COLD-SIGHTED one the
      agent actually captioned (``pick_cold_instance``). For multi-instance
      categories these differ by metres, corrupting presence into a lower bound.

This fork re-scores presence against (a) the nearest COLD-SIGHTED-instance
VIEW_POINT and (b) re-keys "the correct instance" to ``pick_cold_instance``, and
(c) reports the view_point->object-center offset distribution = the binary-SPL@0.1m
STORAGE headroom (whether storing the view_point can ever reach the 0.1 m ring).

The cold instance is recomputed OFFLINE from the content ``.json.gz`` (deterministic
argmax of view_point IoU). The ONE thing not computable offline — the true
back-projected object center per individual recall (no depth is logged at the
recall step) — is substituted by the GT content geometry, the same upper-bound
caveat the audio-DOA gate already lives with.

Pure stdlib; reuses ``check_seed_pose`` content loaders. The BUILD + tests are
local; only the RUN points at ``runs/m3q-*`` (+ co-located ``content/*.json.gz``)
on RACE. Run::

    python embodied_memory/scripts/diagnose_goal_anchored_recall.py runs/m3q-* \
        --content-dir data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_seed_pose import _load_gz  # noqa: E402  (stdlib-only module)
from diagnose_audio_doa_calib import _load_run_dirs, _memory_cands  # noqa: E402

NEAR_RADII = (0.1, 0.5, 1.0, 1.5, 2.5)
MIN_PRESENCE = 0.50          # presence bar (at the RING radius) for the verdict
RING = 1.0                   # benchmark success radius the verdict keys on
DEFAULT_CONTENT_DIR = "data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"


# ----------------------------------------------------------------------
# goal geometry (stdlib; ports make_revisit_smoke helpers, import-light)
# ----------------------------------------------------------------------


def _goals_key(content: Dict[str, Any], category: str) -> Optional[str]:
    suffix = f"_{category}"
    for key in content.get("goals_by_category") or {}:
        if key.endswith(suffix):
            return key
    return None


def _category_instances(content: Dict[str, Any], category: str) -> List[Dict[str, Any]]:
    key = _goals_key(content, category)
    return list((content.get("goals_by_category") or {}).get(key) or []) if key else []


def cold_instance(content: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
    """The goal INSTANCE owning the highest-IoU view_point across all instances of
    ``category`` — the one the cold visit starts at and captions
    (``make_revisit_smoke.pick_cold_instance``). None if none has a view_point."""
    best, best_iou = None, -math.inf
    for inst in _category_instances(content, category):
        for vp in inst.get("view_points") or []:
            iou = float(vp.get("iou", 0.0))
            if iou > best_iou:
                best_iou, best = iou, inst
    return best


def instance_view_points(inst: Dict[str, Any]) -> List[List[float]]:
    out = []
    for vp in inst.get("view_points") or []:
        pos = (vp.get("agent_state") or {}).get("position")
        if pos:
            out.append([float(v) for v in pos])
    return out


def instance_center(inst: Dict[str, Any]) -> Optional[List[float]]:
    """The annotated object center (``inst['position']``), or the view_point
    centroid as a fallback."""
    pos = inst.get("position")
    if pos:
        return [float(v) for v in pos]
    vps = instance_view_points(inst)
    if not vps:
        return None
    n = len(vps)
    return [sum(v[i] for v in vps) / n for i in range(len(vps[0]))]


def _xz(p: List[float]) -> Tuple[float, float]:
    """(x, z) of a 3-vec instance pose, or (x, y) of a 2-vec candidate world_xy."""
    return (float(p[0]), float(p[2])) if len(p) >= 3 else (float(p[0]), float(p[1]))


def _dist_xz(a: List[float], b: List[float]) -> float:
    ax, az = _xz(a)
    bx, bz = _xz(b)
    return math.hypot(ax - bx, az - bz)


def nearest_instance(content: Dict[str, Any], category: str,
                     pose: List[float]) -> Optional[Dict[str, Any]]:
    """Instance whose nearest view_point is closest to ``pose`` (Path-B cross-check
    of the cold instance against a logged cold start_position)."""
    best, best_d = None, math.inf
    for inst in _category_instances(content, category):
        vps = instance_view_points(inst)
        if not vps:
            continue
        d = min(_dist_xz(pose, vp) for vp in vps)
        if d < best_d:
            best_d, best = d, inst
    return best


def offset_vp_to_center(inst: Dict[str, Any]) -> List[float]:
    """Per-view_point xz distance to the instance's object center (the storage
    offset that bounds binary-SPL@0.1m)."""
    center = instance_center(inst)
    if center is None:
        return []
    return [_dist_xz(vp, center) for vp in instance_view_points(inst)]


def offset_stats(instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    offs: List[float] = []
    for inst in instances:
        offs.extend(offset_vp_to_center(inst))
    if not offs:
        return {"n": 0, "min": float("nan"), "median": float("nan"),
                "max": float("nan"), "headroom_0p1": False}
    s = sorted(offs)
    median = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
    return {"n": len(offs), "min": s[0], "median": median, "max": s[-1],
            "headroom_0p1": s[0] <= 0.1}


# ----------------------------------------------------------------------
# per-episode + aggregate re-score
# ----------------------------------------------------------------------


def analyze_episode(ep: Dict[str, Any], content: Dict[str, Any],
                    near_radii: Tuple[float, ...] = NEAR_RADII) -> Dict[str, Any]:
    """Re-scored presence counters for one warm episode_*.json against the
    cold-sighted instance of ``(scene, target_category)``."""
    cat = ep.get("target_category")
    cold = cold_instance(content, cat) if cat else None
    center = instance_center(cold) if cold else None
    vps = instance_view_points(cold) if cold else []
    legacy = ep.get("target_position")  # goals[0] center (the LEGACY anchor)
    a = {"fire_decisions": 0,
         "present_vp": {r: 0 for r in near_radii},
         "present_center": {r: 0 for r in near_radii},
         "present_legacy": {r: 0 for r in near_radii}}
    for d in ep.get("decisions", []):
        cands = _memory_cands(d)
        if not cands:
            continue
        a["fire_decisions"] += 1
        xys = [c["world_xy"] for c in cands]
        if vps:
            mind = min(min(_dist_xz(xy, vp) for vp in vps) for xy in xys)
            for r in near_radii:
                if mind <= r:
                    a["present_vp"][r] += 1
        if center is not None:
            mind = min(_dist_xz(xy, center) for xy in xys)
            for r in near_radii:
                if mind <= r:
                    a["present_center"][r] += 1
        if legacy is not None:
            mind = min(_dist_xz(xy, legacy) for xy in xys)
            for r in near_radii:
                if mind <= r:
                    a["present_legacy"][r] += 1
    return a


def aggregate(episodes: List[Dict[str, Any]], content_by_scene: Dict[str, Dict[str, Any]],
              near_radii: Tuple[float, ...] = NEAR_RADII) -> Dict[str, Any]:
    agg = {"fire_decisions": 0, "n_episodes": 0, "unresolved": 0,
           "present_vp": {r: 0 for r in near_radii},
           "present_center": {r: 0 for r in near_radii},
           "present_legacy": {r: 0 for r in near_radii},
           "cold_by_cell": {}}
    for ep in episodes:
        scene, cat = ep.get("scene_id"), ep.get("target_category")
        content = content_by_scene.get(scene)
        cold = cold_instance(content, cat) if (content and cat) else None
        if cold is None:
            agg["unresolved"] += 1
            continue
        a = analyze_episode(ep, content, near_radii)
        agg["fire_decisions"] += a["fire_decisions"]
        for kind in ("present_vp", "present_center", "present_legacy"):
            for r in near_radii:
                agg[kind][r] += a[kind][r]
        agg["n_episodes"] += 1
        agg["cold_by_cell"][(scene, cat)] = cold
    return agg


def recommend(agg: Dict[str, Any], min_presence: float = MIN_PRESENCE,
              ring: float = RING) -> Tuple[str, str]:
    """Verdict: did re-anchoring DISSOLVE the recall gap (artifact) or CONFIRM it?"""
    fd = agg.get("fire_decisions", 0)
    if not fd:
        return ("INSUFFICIENT-DATA",
                "no memory-firing decisions with a resolvable cold instance — "
                "check the run has S0 fields and content/<scene>.json.gz is present")
    p_vp = agg["present_vp"].get(ring, 0) / fd
    p_legacy = agg["present_legacy"].get(ring, 0) / fd
    if p_vp >= min_presence:
        if p_legacy < min_presence:
            return ("ANCHOR-ARTIFACT",
                    f"view_point-anchored presence {p_vp:.0%} (>= {min_presence:.0%}) but the "
                    f"legacy goals[0]-center anchor read only {p_legacy:.0%} at {ring}m — the "
                    f"earlier recall 'gap' was largely a REFERENCE-FRAME ARTIFACT; recall is "
                    f"adequate at the correct (view_point) anchor.")
        return ("GOAL-ANCHORED-RECALL-OK",
                f"presence {p_vp:.0%} at {ring}m to the cold-instance view_point — recall is "
                f"adequate and the anchors agree; no recall gap.")
    return ("RECALL-GAP-CONFIRMED",
            f"presence only {p_vp:.0%} at {ring}m EVEN against the correct cold-instance "
            f"view_point anchor (legacy {p_legacy:.0%}) — a GENUINE recall gap, not an anchor "
            f"artifact; the missing recalls are far/wrong-instance, addressable only by better "
            f"WRITE-side seeding or instance perception, not a query/rerank tweak.")


# ----------------------------------------------------------------------
# wrong-instance recall (Part B analyzer readout)
# ----------------------------------------------------------------------


def _allegiance(world_xy, target_center, distractor_centers):
    """(tag, dist) — is ``world_xy`` xz-nearer the TARGET instance or a DISTRACTOR?
    Purely geometric (candidates carry no object_id) — 'xz nearer', never 'IS'."""
    dt = _dist_xz(world_xy, target_center)
    dd = min(_dist_xz(world_xy, dc) for dc in distractor_centers)
    return ("target", dt) if dt <= dd else ("distractor", dd)


def wrong_instance_recall_rate(episodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Fraction of memory-firing decisions whose MOST-CONFIDENT recalled candidate
    (the one xz-nearest ANY same-category instance) is nearer a DISTRACTOR than the
    keyed TARGET — a wrong-instance recall — using the offline ``instance_labels``
    the instance-keyed (Part B) build logs. ``None`` if no episode carries labels
    (single-goal runs), so the readout is silent unless the harness is multi-instance.
    """
    fires = wrong = 0
    for ep in episodes:
        labels = ep.get("instance_labels") or {}
        target = labels.get("target_center")
        distractors = labels.get("distractor_centers") or []
        if target is None or not distractors:
            continue
        for d in ep.get("decisions", []):
            cands = _memory_cands(d)
            if not cands:
                continue
            fires += 1
            tagged = [_allegiance(c["world_xy"], target, distractors) for c in cands]
            tag, _ = min(tagged, key=lambda t: t[1])   # nearest-to-any-instance candidate
            if tag == "distractor":
                wrong += 1
    if fires == 0:
        return None
    return {"fires": fires, "wrong": wrong, "rate": wrong / fires}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _resolve_content_by_scene(episodes: List[Dict[str, Any]],
                              content_dir: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        scene = ep.get("scene_id")
        if not scene or scene in out:
            continue
        path = os.path.join(content_dir, f"{scene}.json.gz")
        if os.path.isfile(path):
            try:
                out[scene] = _load_gz(path)
            except (OSError, ValueError):
                pass
    return out


def _fmt_sweep(d: Dict[float, int], fd: int) -> str:
    return "  ".join(f"<={r}m {d.get(r, 0)}/{fd}={d.get(r, 0) / fd:.0%}" for r in NEAR_RADII)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Goal-anchored recall re-score (#2)")
    ap.add_argument("run_globs", nargs="*", help="run dir glob(s) with episode_*.json")
    ap.add_argument("--content-dir", default=DEFAULT_CONTENT_DIR,
                    help="dir of <scene>.json.gz content files for the goal geometry")
    args = ap.parse_args(argv)
    if not args.run_globs:
        print("usage: diagnose_goal_anchored_recall.py <run_dir_glob> ... [--content-dir DIR]")
        return 2
    episodes = _load_run_dirs(args.run_globs)
    if not episodes:
        print(f"[goal-anchored] no episode_*.json under {args.run_globs}")
        return 2
    content_by_scene = _resolve_content_by_scene(episodes, args.content_dir)
    if not content_by_scene:
        print(f"[goal-anchored] no content files under {args.content_dir} for the run's scenes "
              f"-> cannot re-derive the cold instance. INSUFFICIENT-DATA.")
        return 2

    agg = aggregate(episodes, content_by_scene)
    verdict, reason = recommend(agg)
    fd = agg["fire_decisions"]
    print(f"[goal-anchored] {agg['n_episodes']} episodes, {fd} memory-firing decisions "
          f"({agg['unresolved']} unresolved)")
    if fd:
        print(f"  present @VIEW_POINT (cold instance) : {_fmt_sweep(agg['present_vp'], fd)}")
        print(f"  present @CENTER     (cold instance) : {_fmt_sweep(agg['present_center'], fd)}")
        print(f"  present @LEGACY     (goals[0] center): {_fmt_sweep(agg['present_legacy'], fd)}")
        offs = offset_stats(list(agg["cold_by_cell"].values()))
        if offs["n"]:
            print(f"  vp->center offset (storage headroom): min {offs['min']:.2f}m  "
                  f"median {offs['median']:.2f}m  max {offs['max']:.2f}m  "
                  f"-> @0.1m reachable: {offs['headroom_0p1']}")
        # Path-A vs Path-B cross-check (cold instance argmax-iou vs nearest-to-cold-start)
        for (scene, cat), cold in agg["cold_by_cell"].items():
            content = content_by_scene.get(scene)
            seed = _cold_seed_pose(content, cat)
            if seed is not None:
                nb = nearest_instance(content, cat, seed)
                if nb is not None and nb.get("object_id") != cold.get("object_id"):
                    print(f"  [warn] {scene}:{cat} cold-instance cross-check DISAGREES "
                          f"(argmax-iou {cold.get('object_id')} vs nearest-to-cold-start "
                          f"{nb.get('object_id')}) — treat this cell's presence with caution")
    # Part B readout: only fires when the run is instance-keyed (instance_labels
    # present). Silent for single-goal runs.
    wi = wrong_instance_recall_rate(episodes)
    if wi is not None:
        print(f"\n  [instance-keyed run] scoring is cold-instance-keyed (success counts only the "
              f"cold-sighted instance).")
        print(f"  wrong-instance recall rate: {wi['wrong']}/{wi['fires']} = {wi['rate']:.0%}  "
              f"(most-confident recalled candidate xz-nearer a DISTRACTOR than the target)")
    print(f"\nRECOMMEND: {verdict} — {reason}")
    return 0


def _cold_seed_pose(content: Dict[str, Any], category: str) -> Optional[List[float]]:
    """The cold episode's start_position from the built content (a ``*-cold-*``
    episode), for the Path-B cross-check. None if not present."""
    for ep in content.get("episodes") or []:
        if "-cold-" in str(ep.get("episode_id", "")):
            info = ep.get("info") or {}
            if str(info.get("target_category", category)) != category and \
                    not str(ep.get("episode_id", "")).startswith(f"{category}-"):
                continue
            sp = ep.get("start_position")
            if sp:
                return [float(v) for v in sp]
    return None


if __name__ == "__main__":
    sys.exit(main())
