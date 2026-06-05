"""
MultiON (sequential semantic ObjectNav) dataset builder — flavour B.

Single-goal ObjectNav structurally under-tests the LTM (Run 7: one useful
recall per episode at most). MultiON fixes that: an episode specifies an
ordered chain of K goal categories ``[c1, …, cK]`` that co-occur in one scene;
the agent must reach each in order, so a ``c_{i+1}`` glimpsed while hunting
``c_i`` is recallable from the LTM when the goal advances — the value
compounds across sub-goals. See ``docs/MULTION_PORT_PLAN.md``.

Flavour B chains *existing HM3D categories* using the scene's own
``goals_by_category`` (no cylinder assets, no new detector): the episode
record stays a valid single-goal ObjectNav episode —

  * ``object_category`` stays ``c1`` so Habitat's native goals/metrics and
    the runner's category filter work unchanged, and
  * the full ordered chain rides in ``info["object_categories"]`` (habitat's
    NavigationEpisode ``info`` dict tolerates arbitrary keys; the loaders
    preserve it).

No flag + no info key -> existing single-goal behaviour, byte-identical.

The start pose reuses the revisit builder's warm-pose picker (a reachable
pose validated by the source dataset, away from c1's goal view_points so the
agent does not start on top of its first sub-goal).

Usage:
    python embodied_memory/scripts/make_multion_smoke.py \
        --src data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/wcojb4TFT35.json.gz \
        --scene wcojb4TFT35 --k 3 --n-episodes 4 --seed 7 \
        --out-dir data/hm3d/datasets/objectnav/hm3d/v1/multion_wcojb4TFT35

Then run (S1/S2/S3, one process each so the LTM persists across sub-goals):
    python -m embodied_memory.run_hm3d_pol --mode live --backbone remembr \
        --episodes-path <out-dir>/multion_wcojb4TFT35.json.gz \
        --scene wcojb4TFT35 --target any --setting {1,2,3} \
        --n-episodes 99 --out-dir runs/multion-s{1,2,3}
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the revisit builder's pose/goal/IO machinery — import, don't copy.
from make_revisit_smoke import (  # noqa: E402
    _goal_view_point_positions,
    _goals_key,
    _load_gz,
    write_dataset,
)


# ----------------------------------------------------------------------
# pure builders
# ----------------------------------------------------------------------


def co_occurring_categories(content: Dict[str, Any]) -> List[str]:
    """Categories usable in a multion chain for this scene: present in
    ``goals_by_category`` with at least one view_point AND with at least one
    source episode (the template / start-pose donor). Sorted for determinism.
    """
    goals_by_category = content.get("goals_by_category") or {}
    src_eps = content.get("episodes") or []
    ep_cats = {ep.get("object_category") for ep in src_eps} - {None}
    out: List[str] = []
    for cat in sorted(ep_cats):
        gkey = _goals_key(goals_by_category, cat)
        if gkey is None:
            continue
        if not _goal_view_point_positions(goals_by_category[gkey]):
            continue
        out.append(cat)
    return out


def sample_orderings(
    categories: List[str], k: int, n: int, rng: random.Random
) -> List[List[str]]:
    """Sample up to ``n`` pairwise-distinct ordered k-chains (no category
    repeats within a chain). Deterministic given ``rng``."""
    if k > len(categories):
        raise ValueError(
            f"k={k} exceeds the {len(categories)} usable categories: {categories}"
        )
    seen: set = set()
    out: List[List[str]] = []
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        perm = tuple(rng.sample(categories, k))
        if perm in seen:
            continue
        seen.add(perm)
        out.append(list(perm))
    return out


def pick_start_poses(
    candidate_poses: List[Dict[str, Any]],
    goal_vp_positions: List[List[float]],
    n: int,
    min_dist: float = 2.0,
) -> List[Dict[str, Any]]:
    """Return up to ``n`` candidate poses, NEAREST-first by distance to the
    closest c1 view_point, dropping any closer than ``min_dist``.

    Deliberate contrast with the revisit builder's ``pick_warm_poses``
    (farthest-first): revisit warm episodes start far so memory can guide
    the agent back; a multion episode's first hop is COLD, so the start
    must make c1 findable (multion-micro2: farthest-first starts left the
    best approach at 4.52 m after 750 steps — zero sub-goals found).
    """
    scored: List[Any] = []
    for pose in candidate_poses:
        pos = pose.get("position")
        if not pos:
            continue
        if goal_vp_positions:
            d = min(math.dist(pos, g) for g in goal_vp_positions)
        else:
            d = math.inf
        if d < min_dist:
            continue
        scored.append((d, pose))
    scored.sort(key=lambda t: t[0])
    return [pose for _, pose in scored[:n]]


def _cat_vp_positions(
    goals_by_category: Dict[str, Any], category: str
) -> List[List[float]]:
    gkey = _goals_key(goals_by_category, category)
    if gkey is None:
        return []
    return _goal_view_point_positions(goals_by_category[gkey])


def _min_vp_dist(a_vps: List[List[float]], b_vps: List[List[float]]) -> float:
    if not a_vps or not b_vps:
        return math.inf
    return min(math.dist(a, b) for a in a_vps for b in b_vps)


def short_hop_order(
    chain: List[str], goals_by_category: Dict[str, Any]
) -> List[str]:
    """Keep c1, greedy-order the rest by nearest viewpoint-to-viewpoint hop
    (c_{i+1} = remaining category closest to c_i).

    Every hop after the first is a fresh cold ObjectNav leg for the backbone;
    long hops re-enter the absorbing-loop regime that produced 0 advances in
    multion-micro/micro2. Short hops keep each leg inside the range the
    navmesh follower demonstrably handles (revisit: success@1m 0.67).
    Categories without resolvable view_points sort last (inf distance).
    """
    if len(chain) <= 2:
        return list(chain)
    out = [chain[0]]
    rest = list(chain[1:])
    while rest:
        cur_vps = _cat_vp_positions(goals_by_category, out[-1])
        rest.sort(key=lambda c: _min_vp_dist(
            cur_vps, _cat_vp_positions(goals_by_category, c)))
        out.append(rest.pop(0))
    return out


def build_multion_episodes(
    content: Dict[str, Any],
    orderings: List[List[str]],
    min_dist: float = 2.0,
) -> List[Dict[str, Any]]:
    """One episode per ordering. Clones a real c1 episode as the template
    (valid ``goals`` / ``scene_id``), sets ``object_category = c1``, writes the
    full chain to ``info["object_categories"]``, and overrides the start pose
    with a warm pose (reachable, >= ``min_dist`` from any c1 view_point).
    Orderings whose c1 lacks goals/episodes are skipped. Source not mutated.
    """
    goals_by_category = content.get("goals_by_category") or {}
    src_eps = content.get("episodes") or []

    # Pre-pick warm poses per c1 so multiple orderings sharing a c1 get
    # different starts (cycled when poses run out).
    c1_counts: Dict[str, int] = {}
    for ordering in orderings:
        if ordering:
            c1_counts[ordering[0]] = c1_counts.get(ordering[0], 0) + 1
    poses_by_c1: Dict[str, List[Dict[str, Any]]] = {}
    for c1, need in c1_counts.items():
        gkey = _goals_key(goals_by_category, c1)
        if gkey is None:
            continue
        cat_candidate_poses = [
            {"position": list(ep["start_position"]),
             "rotation": list(ep["start_rotation"])}
            for ep in src_eps
            if ep.get("object_category") == c1
            and ep.get("start_position") and ep.get("start_rotation")
        ]
        goal_vps = _goal_view_point_positions(goals_by_category[gkey])
        poses_by_c1[c1] = pick_start_poses(
            cat_candidate_poses, goal_vps, n=need, min_dist=min_dist
        )

    out: List[Dict[str, Any]] = []
    used_by_c1: Dict[str, int] = {}
    for i, ordering in enumerate(orderings):
        if not ordering:
            continue
        c1 = ordering[0]
        template = next(
            (ep for ep in src_eps if ep.get("object_category") == c1), None
        )
        poses = poses_by_c1.get(c1) or []
        if template is None or not poses:
            continue
        pose = poses[used_by_c1.get(c1, 0) % len(poses)]
        used_by_c1[c1] = used_by_c1.get(c1, 0) + 1

        ep = copy.deepcopy(template)
        ep["episode_id"] = f"multion-{i}-{'-'.join(ordering)}"
        ep["object_category"] = c1
        ep["start_position"] = list(pose["position"])
        ep["start_rotation"] = list(pose["rotation"])
        info = dict(ep.get("info") or {})
        info["object_categories"] = list(ordering)
        ep["info"] = info
        out.append(ep)
    return out


def build_dataset(
    src_content: Dict[str, Any],
    k: int = 3,
    n_episodes: int = 4,
    seed: int = 7,
    categories: Optional[List[str]] = None,
    min_dist: float = 2.0,
) -> Dict[str, Any]:
    """Assemble a content dict (``goals_by_category`` preserved verbatim so
    every chained category's view_points stay resolvable) with ``n_episodes``
    K-chain multion episodes. ``categories`` optionally restricts the pool."""
    usable = co_occurring_categories(src_content)
    if categories:
        usable = [c for c in usable if c in set(categories)]
    rng = random.Random(seed)
    orderings = sample_orderings(usable, k=k, n=n_episodes, rng=rng)
    # Short-hop reorder (c1 kept, rest greedy-nearest). Reordering can
    # collapse same-c1 permutations of one set -> dedup, first kept.
    gbc = src_content.get("goals_by_category") or {}
    seen: set = set()
    hop_ordered: List[List[str]] = []
    for o in orderings:
        ho = short_hop_order(o, gbc)
        key = tuple(ho)
        if key in seen:
            continue
        seen.add(key)
        hop_ordered.append(ho)
    episodes = build_multion_episodes(src_content, hop_ordered, min_dist=min_dist)
    return {
        "category_to_task_category_id":
            src_content.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src_content.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": src_content.get("goals_by_category") or {},
        "episodes": episodes,
    }


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MultiON (sequential semantic ObjectNav) dataset builder"
    )
    parser.add_argument("--src", required=True,
                        help="Source content json.gz "
                             "(…/val_mini/content/<scene>.json.gz).")
    parser.add_argument("--scene", required=True,
                        help="Bare scene name, e.g. wcojb4TFT35.")
    parser.add_argument("--k", type=int, default=3,
                        help="Sub-goals per episode (default 3).")
    parser.add_argument("--n-episodes", type=int, default=4,
                        help="Orderings (episodes) to sample per scene.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Restrict the category pool (default: all usable).")
    parser.add_argument("--min-dist", type=float, default=2.0,
                        help="Min metres the start must be from any c1 view_point.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)

    src = _load_gz(args.src)
    usable = co_occurring_categories(src)
    print(f"usable categories in {args.scene}: {usable}")
    content = build_dataset(
        src, k=args.k, n_episodes=args.n_episodes, seed=args.seed,
        categories=args.categories, min_dist=args.min_dist,
    )
    if not content["episodes"]:
        print(f"ERROR: no multion episodes built (k={args.k}, "
              f"categories={args.categories or usable}).", file=sys.stderr)
        return 1

    top = write_dataset(args.out_dir, args.scene, content, src)

    print(f"wrote {top}")
    print(f"  content/{args.scene}.json.gz: {len(content['episodes'])} episodes (K={args.k})")
    for ep in content["episodes"]:
        print(f"    {ep['episode_id']}: {' -> '.join(ep['info']['object_categories'])}")
    print(f"  goals_by_category: {len(content['goals_by_category'])} categories preserved")

    # re-load verify (cheap structural check the GPU run will rely on)
    re = _load_gz(top)
    assert re["episodes"] == [], "top-level must have empty episodes"
    cj = _load_gz(os.path.join(args.out_dir, "content", f"{args.scene}.json.gz"))
    assert cj["episodes"] and "goals_by_category" in cj, "content malformed"
    for ep in cj["episodes"]:
        assert len(ep["info"]["object_categories"]) == args.k, "chain lost in round-trip"
    print("  verify: re-loaded OK (top empty, content has goals + K-chains)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
