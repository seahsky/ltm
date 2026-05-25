"""
Phase-B1 controlled-start revisit dataset builder (no Habitat / sim needed).

Phase-2 closed net-neutral because, on the interleaved single-goal `val_mini`
ablation, the LTM's one real value — recalling a *past sighting of the goal* —
almost never applies: the cold visit rarely captioned the goal closely, so the
warm-visit memory holds nothing matching (the Run-7 `inspect_memory_rerank`
diagnostic: every memory candidate is a non-match ~0.235 cosine).

This script removes that confound. For each target category it emits a tiny
HM3D-ObjectNav dataset where:

  * **episode 0 (cold)** starts *at a high-iou goal view_point* — a navigable
    pose from which the goal is, by the dataset's own definition, visible. The
    agent therefore captions the goal on the first observation and the LTM
    *provably* holds a goal-matching sighting.
  * **episodes 1..N (warm)** start *far from every goal view_point*, so reaching
    the goal benefits from recalling the cold sighting.

The cold episode is ordered first; run the whole dataset in one process (the
`EmbodiedMemoryBridge` persists across episodes) so the cold LTM entry is live
when the warm visits run. Compare S1 (memory off) vs S3 (full) on the warm
visits with `analyze_revisit.py`.

The builder is pure data: it reuses the source scene's `goals_by_category`
(so view_points / success still compute) and clones a real episode as the
template (valid `goals` / `info` / `scene_id`), overriding only the start pose
and `episode_id`. It writes the standard habitat layout
(`<name>.json.gz` with empty episodes + category maps, plus
`content/<scene>.json.gz` with the built episodes) so the existing
`--episodes-path` override loads it unchanged.

Usage:
    python embodied_memory/scripts/make_revisit_smoke.py \
        --src data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/wcojb4TFT35.json.gz \
        --scene wcojb4TFT35 --categories chair bed --n-warm 3 \
        --out-dir data/hm3d/datasets/objectnav/hm3d/v1/revisit_wcojb4TFT35

Then run (S1 then S3), one process each so the LTM persists across episodes:
    python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
        --episodes-path <out-dir>/revisit_wcojb4TFT35.json.gz \
        --scene wcojb4TFT35 --target any --setting {1,3} \
        --n-episodes 99 --out-dir runs/revisit-b1-s{1,3}
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------
# pose selection
# ----------------------------------------------------------------------


def _dist(a: List[float], b: List[float]) -> float:
    return math.dist(a, b)


def pick_cold_pose(goal_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return ``{position, rotation}`` of the highest-iou view_point across all
    goal instances — the pose from which the goal is most clearly visible.

    Raises ``ValueError`` if no goal instance carries any view_point.
    """
    best: Optional[Dict[str, Any]] = None
    best_iou = -math.inf
    for inst in goal_instances:
        for vp in inst.get("view_points") or []:
            iou = float(vp.get("iou", 0.0))
            if iou > best_iou:
                best_iou = iou
                state = vp.get("agent_state") or {}
                best = {
                    "position": list(state.get("position", [])),
                    "rotation": list(state.get("rotation", [])),
                }
    if best is None or not best["position"]:
        raise ValueError("no goal view_point available for cold start")
    return best


def _goal_view_point_positions(goal_instances: List[Dict[str, Any]]) -> List[List[float]]:
    out: List[List[float]] = []
    for inst in goal_instances:
        for vp in inst.get("view_points") or []:
            state = vp.get("agent_state") or {}
            pos = state.get("position")
            if pos:
                out.append(list(pos))
    return out


def pick_warm_poses(
    candidate_poses: List[Dict[str, Any]],
    goal_vp_positions: List[List[float]],
    n: int,
    min_dist: float = 2.0,
) -> List[Dict[str, Any]]:
    """Return up to ``n`` candidate poses, farthest-first by distance to the
    nearest goal view_point, dropping any closer than ``min_dist`` (so the warm
    agent does not start already on top of the goal).
    """
    scored: List[Any] = []
    for pose in candidate_poses:
        pos = pose.get("position")
        if not pos:
            continue
        if goal_vp_positions:
            d = min(_dist(pos, g) for g in goal_vp_positions)
        else:
            d = math.inf
        if d < min_dist:
            continue
        scored.append((d, pose))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [pose for _, pose in scored[:n]]


# ----------------------------------------------------------------------
# episode assembly
# ----------------------------------------------------------------------


def build_category_episodes(
    template: Dict[str, Any],
    cold_pose: Dict[str, Any],
    warm_poses: List[Dict[str, Any]],
    category: str,
) -> List[Dict[str, Any]]:
    """Clone ``template`` into [cold, warm_1, ..., warm_k], overriding start
    pose + episode_id. The template supplies valid ``goals`` / ``info`` /
    ``scene_id``; it is not mutated.
    """
    out: List[Dict[str, Any]] = []

    def _clone(pose: Dict[str, Any], eid: str) -> Dict[str, Any]:
        ep = copy.deepcopy(template)
        ep["episode_id"] = eid
        ep["object_category"] = category
        ep["start_position"] = list(pose["position"])
        ep["start_rotation"] = list(pose["rotation"])
        return ep

    out.append(_clone(cold_pose, f"{category}-cold-0"))
    for i, pose in enumerate(warm_poses):
        out.append(_clone(pose, f"{category}-warm-{i + 1}"))
    return out


def _goals_key(goals_by_category: Dict[str, Any], category: str) -> Optional[str]:
    suffix = f"_{category}"
    for key in goals_by_category:
        if key.endswith(suffix):
            return key
    return None


def build_dataset(
    src_content: Dict[str, Any],
    categories: List[str],
    n_warm: int,
    min_dist: float = 2.0,
) -> Dict[str, Any]:
    """Assemble a content dict (goals_by_category preserved) with, per category,
    one cold + ``n_warm`` warm episodes. Categories absent from the source are
    skipped. Warm-start candidates are drawn from the source episodes' own
    (navigable) start poses across the scene.
    """
    goals_by_category = src_content.get("goals_by_category") or {}
    src_eps = src_content.get("episodes") or []

    # navigable candidate poses for warm starts = every source episode start
    candidate_poses = [
        {"position": list(ep["start_position"]), "rotation": list(ep["start_rotation"])}
        for ep in src_eps
        if ep.get("start_position") and ep.get("start_rotation")
    ]

    out_eps: List[Dict[str, Any]] = []
    for cat in categories:
        gkey = _goals_key(goals_by_category, cat)
        if gkey is None:
            continue
        template = next((ep for ep in src_eps if ep.get("object_category") == cat), None)
        if template is None:
            continue
        goal_instances = goals_by_category[gkey]
        cold_pose = pick_cold_pose(goal_instances)
        goal_vps = _goal_view_point_positions(goal_instances)
        warm_poses = pick_warm_poses(candidate_poses, goal_vps, n=n_warm, min_dist=min_dist)
        out_eps.extend(build_category_episodes(template, cold_pose, warm_poses, cat))

    return {
        "category_to_task_category_id": src_content.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src_content.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": goals_by_category,
        "episodes": out_eps,
    }


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------


def _write_gz(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def write_dataset(
    out_dir: str,
    scene: str,
    content: Dict[str, Any],
    category_maps: Dict[str, Any],
    name: Optional[str] = None,
) -> str:
    """Write the habitat layout and return the top-level json.gz path.

    ``<out_dir>/<name>.json.gz``         — category maps + empty episodes
    ``<out_dir>/content/<scene>.json.gz`` — goals_by_category + built episodes
    """
    name = name or os.path.basename(os.path.normpath(out_dir))
    top_path = os.path.join(out_dir, f"{name}.json.gz")
    content_path = os.path.join(out_dir, "content", f"{scene}.json.gz")

    top = {
        "category_to_task_category_id":
            category_maps.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            category_maps.get("category_to_scene_annotation_category_id", {}),
        "episodes": [],
    }
    _write_gz(top_path, top)
    _write_gz(content_path, content)
    return top_path


def _load_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-B1 controlled-start revisit dataset")
    parser.add_argument("--src", required=True,
                        help="Source content json.gz "
                             "(…/val_mini/content/<scene>.json.gz).")
    parser.add_argument("--scene", required=True, help="Bare scene name, e.g. wcojb4TFT35.")
    parser.add_argument("--categories", nargs="+", default=["chair", "bed"])
    parser.add_argument("--n-warm", type=int, default=3)
    parser.add_argument("--min-dist", type=float, default=2.0,
                        help="Min metres a warm start must be from any goal view_point.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    src = _load_gz(args.src)
    content = build_dataset(src, args.categories, args.n_warm, args.min_dist)
    if not content["episodes"]:
        print(f"ERROR: no episodes built for categories={args.categories} "
              f"(none present in {args.src}).", file=sys.stderr)
        return 1

    top = write_dataset(args.out_dir, args.scene, content, src)

    # report
    by_cat: Dict[str, int] = {}
    for ep in content["episodes"]:
        by_cat[ep["object_category"]] = by_cat.get(ep["object_category"], 0) + 1
    print(f"wrote {top}")
    print(f"  content/{args.scene}.json.gz: {len(content['episodes'])} episodes")
    for cat, n in by_cat.items():
        print(f"    {cat}: 1 cold + {n - 1} warm")
    print(f"  goals_by_category: {len(content['goals_by_category'])} categories preserved")

    # re-load verify (cheap structural check the GPU run will rely on)
    re = _load_gz(top)
    assert re["episodes"] == [], "top-level must have empty episodes"
    cj = _load_gz(os.path.join(args.out_dir, "content", f"{args.scene}.json.gz"))
    assert cj["episodes"] and "goals_by_category" in cj, "content malformed"
    print("  verify: re-loaded OK (top empty, content has goals + episodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
