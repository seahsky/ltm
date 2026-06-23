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
import random
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


def pick_cold_instance(goal_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the goal INSTANCE (one ``ObjectGoal`` dict) that owns the
    highest-iou view_point across all instances — i.e. the instance the cold
    visit starts at and provably captions.

    Used by the instance-keyed build: the warm goal set is restricted to *this*
    instance so reaching a different same-category instance no longer counts,
    giving instance discrimination a metric target. Raises ``ValueError`` if no
    instance carries any view_point.
    """
    best: Optional[Dict[str, Any]] = None
    best_iou = -math.inf
    for inst in goal_instances:
        for vp in inst.get("view_points") or []:
            iou = float(vp.get("iou", 0.0))
            if iou > best_iou:
                best_iou = iou
                best = inst
    if best is None:
        raise ValueError("no goal view_point available to key the cold instance")
    return best


def _instance_centroid(inst: Dict[str, Any]) -> Optional[List[float]]:
    """Mean of an instance's view_point positions (its spatial centre), or None."""
    vps = _goal_view_point_positions([inst])
    if not vps:
        return None
    n = len(vps)
    return [sum(v[i] for v in vps) / n for i in range(len(vps[0]))]


def pick_distractor_instances(
    target_inst: Dict[str, Any],
    all_instances: List[Dict[str, Any]],
    n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return the same-category DISTRACTOR instances (every instance != target
    that owns a view_point), ordered NEAREST-to-target first.

    Used by the ``seed_distractors`` build to emit one seed-only cold episode per
    distractor: the agent starts at the distractor's view_point, captions it, and
    consolidation seeds it into the LTM — so the warm visit's retrieval faces
    MULTIPLE same-category sightings and must rank the RIGHT one (a retrieval-level
    disambiguation test, not just navigation-level steering past a nearer
    distractor). NEAREST-first because the closest same-category instances are the
    genuine confusers (and the cap ``n`` keeps the seed-episode count — and run
    cost — modest on categories like chair with 9-12 instances). ``n=None`` seeds
    all distractors. Returns ``[]`` for a single-instance category.
    """
    ct = _instance_centroid(target_inst)
    scored: List["tuple[float, Dict[str, Any]]"] = []
    for inst in all_instances:
        if inst is target_inst:
            continue
        c = _instance_centroid(inst)
        if c is None:
            continue
        d = _dist(ct, c) if ct is not None else math.inf
        scored.append((d, inst))
    scored.sort(key=lambda t: t[0])
    insts = [inst for _, inst in scored]
    return insts if n is None else insts[: max(0, n)]


def _instance_labels(target_inst: Dict[str, Any],
                     all_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Offline disambiguation labels for the instance-keyed (Part B) build: the
    keyed TARGET instance's object_id + centroid, and the centroids of every OTHER
    same-category instance (the DISTRACTORS the warm agent must not mistake for the
    goal). The retrieval query stays category-level ("there is a {cat}"); these
    labels are read only by the analyzer's wrong-instance-recall readout."""
    distractors: List[List[float]] = []
    for inst in all_instances:
        if inst is target_inst:
            continue
        c = _instance_centroid(inst)
        if c is not None:
            distractors.append(c)
    return {"target_object_id": target_inst.get("object_id"),
            "target_center": _instance_centroid(target_inst),
            "distractor_centers": distractors}


def pick_warm_instance(
    goal_instances: List[Dict[str, Any]],
    cold_instance: Dict[str, Any],
    min_move: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """Return a DIFFERENT goal INSTANCE to serve as the moved-to goal B for the
    changed-world build, so the cold sighting of ``cold_instance`` (A) is stale.

    Prefer the NEAREST other instance whose view_point centroid is at least
    ``min_move`` m from A: it is still a genuine move (B != A) yet most likely on
    the SAME navmesh component, so the warm starts can reach it. (The earlier
    FARTHEST choice frequently landed on a disconnected island / different floor
    → Infinity start→B geodesic → NaN soft_SPL, which voided whole cells.) If no
    other instance clears ``min_move``, fall back to the single nearest other
    instance so a >=2-instance category is never silently dropped (the analyzer
    NaN-guard is the backstop). Returns ``None`` only when there is no other
    instance with view_points.
    """
    ca = _instance_centroid(cold_instance)
    if ca is None:
        return None
    others: List["tuple[float, Dict[str, Any]]"] = []
    for inst in goal_instances:
        if inst is cold_instance:
            continue
        c = _instance_centroid(inst)
        if c is None:
            continue
        others.append((_dist(ca, c), inst))
    if not others:
        return None
    moved = [t for t in others if t[0] >= min_move]
    pool = moved if moved else others
    return min(pool, key=lambda t: t[0])[1]


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
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return up to ``n`` candidate poses, farthest-first by distance to the
    nearest goal view_point, dropping any closer than ``min_dist`` (so the warm
    agent does not start already on top of the goal).

    ``seed`` (default ``None``) controls a RESAMPLE for an independent second
    sample of the warm-revisit headline. The eligible pool is computed
    IDENTICALLY (same ``min_dist`` / category filters); only the FINAL pick
    among the survivors changes:

      * ``seed is None`` → the historical deterministic farthest-first top-``n``
        (BYTE-IDENTICAL to before this parameter existed).
      * ``seed is not None`` → ``random.Random(seed).sample(eligible,
        min(n, len(eligible)))`` — a different valid ``n``-subset of the SAME
        eligible pool (every member already satisfies the filters). ``sample``
        takes ALL when the pool ``<= n``, so the set is seed-independent there.
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
    if seed is not None:
        # Resample among the SAME eligible survivors (filters already applied).
        eligible = [pose for _, pose in scored]
        return random.Random(seed).sample(eligible, min(n, len(eligible)))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [pose for _, pose in scored[:n]]


def pick_warm_poses_changed_world(
    candidate_poses: List[Dict[str, Any]],
    reachable_poses: List[Dict[str, Any]],
    goal_b_vp_positions: List[List[float]],
    n: int,
    min_dist: float = 2.0,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Reachability-biased warm-start selection for the CHANGED-WORLD build.

    The regular ``pick_warm_poses`` is FARTHEST-first, which on a real navmesh
    maximises the chance of grabbing a start on a navmesh component DISCONNECTED
    from goal B (a different floor / balcony / island) → Infinity start→B
    geodesic → NaN soft_SPL (proven on RACE run cw-2: every wcojb chair warm
    episode was Infinity/NaN, yet the cold episode — which starts at an instance
    A view_point — had a FINITE geodesic to B). Same lesson already learned for
    ``pick_warm_instance`` (nearest beats farthest for reachability).

    So here we (1) draw from a PROVEN-REACHABLE pool — instance A's view_point
    poses FIRST (the cold episode starts at A's best view_point and reaches B, so
    A's region is navmesh-connected to B), then the category source starts as a
    backfill — and (2) rank NEAREST-to-B first (proximity correlates with being
    on the same navmesh component), while still dropping anything closer than
    ``min_dist`` so a real path to B remains (the warm agent does not start on
    top of B). The analyzer NaN-guard stays the backstop.

    ``reachable_poses`` (instance A's view_point poses) are preferred over
    ``candidate_poses`` (the category source starts) at equal distance, so an
    A view_point always out-ranks a source start the same distance from B.
    """
    def _filter(poses: List[Dict[str, Any]]) -> List[Any]:
        out: List[Any] = []
        for pose in poses:
            pos = pose.get("position")
            if not pos:
                continue
            if goal_b_vp_positions:
                d = min(_dist(pos, g) for g in goal_b_vp_positions)
            else:
                d = math.inf
            if d < min_dist:
                continue
            out.append((d, pose))
        return out

    # Tag the proven-reachable A view_points with priority 0 (preferred) and the
    # category source starts with priority 1, then sort NEAREST-to-B first with
    # the proven-reachable poses winning ties.
    scored: List[Any] = [(d, 0, pose) for d, pose in _filter(reachable_poses)]
    scored += [(d, 1, pose) for d, pose in _filter(candidate_poses)]
    if seed is not None:
        # RESAMPLE among the SAME eligible survivors (the reachability/min_dist
        # filters in ``_filter`` are already applied) — a different valid
        # n-subset of the same pool. ``seed is None`` keeps the deterministic
        # nearest-first pick. See ``pick_warm_poses`` for the full rationale.
        eligible = [pose for _, _, pose in scored]
        return random.Random(seed).sample(eligible, min(n, len(eligible)))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [pose for _, _, pose in scored[:n]]


# ----------------------------------------------------------------------
# episode assembly
# ----------------------------------------------------------------------


def build_category_episodes(
    template: Dict[str, Any],
    cold_pose: Dict[str, Any],
    warm_poses: List[Dict[str, Any]],
    category: str,
    seed_poses: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Clone ``template`` into [cold, seed_0..seed_m, warm_1, ..., warm_k],
    overriding start pose + episode_id. The template supplies valid ``goals`` /
    ``info`` / ``scene_id``; it is not mutated.

    ``seed_poses`` (distractor view_point poses) are emitted as ``{cat}-seed-{k}``
    episodes ORDERED BETWEEN the cold visit and the warm visits, each flagged
    ``info['seed_only']=True``. They run a full episode from the distractor
    view_point so the agent captions it and episode-end consolidation seeds it
    into the LTM — populating the LTM with same-category DISTRACTOR sightings
    BEFORE the warm visit (so warm retrieval must rank the right instance). The
    ``seed_only`` flag is the analyzer's exclusion key (it is NOT a scored/paired
    visit). Ordering relies on ``pin_episode_order`` (shuffle=False), so dataset
    order == run order == ``episode_idx`` order: cold(0) seeds(1..m) warm(m+1..).
    """
    out: List[Dict[str, Any]] = []

    def _clone(pose: Dict[str, Any], eid: str,
               seed_only: bool = False) -> Dict[str, Any]:
        ep = copy.deepcopy(template)
        ep["episode_id"] = eid
        ep["object_category"] = category
        ep["start_position"] = list(pose["position"])
        ep["start_rotation"] = list(pose["rotation"])
        if seed_only:
            # Habitat overwrites episode_id with the load index, so the analyzer
            # CANNOT detect a seed via the id substring — the flag must ride
            # episode.info -> metadata -> ep_log (same path as instance_labels).
            ep.setdefault("info", {})["seed_only"] = True
        return ep

    out.append(_clone(cold_pose, f"{category}-cold-0"))
    for k, pose in enumerate(seed_poses or []):
        out.append(_clone(pose, f"{category}-seed-{k}", seed_only=True))
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
    instance_keyed: bool = False,
    seed_distractors: bool = False,
    n_distractors: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble a content dict with, per category, one cold + ``n_warm`` warm
    episodes. Categories absent from the source are skipped. Warm-start
    candidates are drawn from the SAME category's own source episode starts
    (see below).

    ``instance_keyed`` (default ``False`` → category-level, the historical
    behaviour) restricts each built category's ``goals_by_category`` entry to
    the SINGLE cold-sighted (highest-iou) instance. Habitat keys success/
    distance on ``goals_by_category['{scene}_{cat}']`` (the full multi-instance
    list), so shrinking it to one instance means reaching a *different*
    same-category instance no longer counts — giving instance discrimination a
    metric target. ``object_category`` stays "{cat}" so the "there is a {cat}"
    retrieval query is unchanged; only the goal *set* shrinks. Warm starts are
    then filtered for distance to *that instance's* view_points only.

    ``seed`` (default ``None``) RESAMPLES the warm starts for a genuinely
    independent second sample of the warm-revisit headline (the pipeline is
    otherwise fully deterministic). It threads to the warm-pose selectors, which
    draw a different valid ``n``-subset of the SAME eligible pool. The cold pose
    and the instance choice (argmax-iou) stay deterministic, so success-keying is
    unchanged across seeds. When not ``None`` it is stamped as ``revisit_seed`` in
    the returned content for provenance; ``None`` → field absent (byte-identical).
    """
    goals_by_category = src_content.get("goals_by_category") or {}
    src_eps = src_content.get("episodes") or []

    out_eps: List[Dict[str, Any]] = []
    out_goals: Dict[str, Any] = dict(goals_by_category)
    for cat in categories:
        gkey = _goals_key(goals_by_category, cat)
        if gkey is None:
            continue
        template = next((ep for ep in src_eps if ep.get("object_category") == cat), None)
        if template is None:
            continue

        # Warm-start candidates = THIS category's own source-episode starts.
        # The original ObjectNav dataset validated each as reachable to a goal of
        # this category, so the synthesized warm episode has a finite start->goal
        # geodesic (valid soft_SPL). Drawing from *any* category's starts (the
        # global Euclidean-farthest) can land on a disconnected navmesh island /
        # different floor -> Infinity geodesic -> NaN soft_SPL (observed: every
        # bed warm episode was NaN in the revisit-b1 smoke).
        cat_candidate_poses = [
            {"position": list(ep["start_position"]),
             "rotation": list(ep["start_rotation"])}
            for ep in src_eps
            if ep.get("object_category") == cat
            and ep.get("start_position") and ep.get("start_rotation")
        ]

        goal_instances = goals_by_category[gkey]
        if instance_keyed:
            # Multi-instance disambiguation (Part B): restrict success to the
            # single cold-sighted instance — reaching any OTHER same-category
            # instance no longer succeeds / reduces distance-to-goal. Warm starts
            # are REACHABILITY-biased (the caveat-A NaN-collapse fix the
            # changed-world build already uses): draw from the TARGET instance's
            # own view_point poses (proven navmesh-connected — the cold episode
            # starts there) + the category source starts, ranked NEAREST-to-target
            # (the farthest-first pick_warm_poses grabbed disconnected-island
            # starts → Infinity geodesic → NaN soft_SPL). Plus offline distractor
            # labels for the analyzer's wrong-instance-recall readout.
            target_inst = pick_cold_instance(goal_instances)
            cold_pose = pick_cold_pose([target_inst])
            goal_vps = _goal_view_point_positions([target_inst])
            out_goals[gkey] = [target_inst]
            a_reachable_poses = [
                {"position": list(vp), "rotation": list(cold_pose["rotation"])}
                for vp in goal_vps
            ]
            warm_poses = pick_warm_poses_changed_world(
                cat_candidate_poses, a_reachable_poses, goal_vps,
                n=n_warm, min_dist=min_dist, seed=seed)
            # seed_distractors: ALSO start a seed-only cold episode at each
            # same-category DISTRACTOR's view_point (between cold and warm), so
            # consolidation seeds the distractor into the LTM and the warm visit's
            # retrieval faces MULTIPLE same-category sightings (a retrieval-level
            # disambiguation test). Faithful: the agent captions the distractor the
            # same way it captions the target — no privileged direct LTM write.
            seed_poses: List[Dict[str, Any]] = []
            if seed_distractors:
                for dist_inst in pick_distractor_instances(
                        target_inst, goal_instances, n=n_distractors):
                    seed_poses.append(pick_cold_pose([dist_inst]))
            eps = build_category_episodes(template, cold_pose, warm_poses, cat,
                                          seed_poses=seed_poses)
            labels = _instance_labels(target_inst, goal_instances)
            for ep in eps:
                ep.setdefault("info", {})["instance_labels"] = labels
            out_eps.extend(eps)
        else:
            cold_pose = pick_cold_pose(goal_instances)
            goal_vps = _goal_view_point_positions(goal_instances)
            warm_poses = pick_warm_poses(cat_candidate_poses, goal_vps,
                                         n=n_warm, min_dist=min_dist, seed=seed)
            out_eps.extend(build_category_episodes(template, cold_pose, warm_poses, cat))

    out: Dict[str, Any] = {
        "category_to_task_category_id": src_content.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src_content.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": out_goals,
        "episodes": out_eps,
    }
    if seed is not None:
        out["revisit_seed"] = seed  # provenance; None → field absent (byte-identical)
    return out


def build_changed_world_dataset(
    src_content: Dict[str, Any],
    categories: List[str],
    n_warm: int,
    min_dist: float = 2.0,
    min_move: float = 1.5,
) -> Dict[str, Any]:
    """Changed-world revisit (the regime the M4 temporal-context head was built
    for). Per category: the cold episode STARTS at instance A (the highest-iou
    instance → provably captioned & seeded) but success for the WHOLE category is
    keyed to a DIFFERENT instance B (``pick_warm_instance`` — the NEAREST other
    instance at least ``min_move`` m away, for navmesh reachability), so the cold
    sighting of A is now STALE — recalling it leads the agent to the wrong place.
    Warm starts are drawn reachability-biased toward B (``pick_warm_poses_changed_world``:
    A's view_point poses, proven navmesh-connected to B since cold-from-A reaches
    it, plus same-category source starts ranked NEAREST-to-B, kept >= ``min_dist``
    away) — the prior farthest-first selection grabbed disconnected-island starts
    (Infinity geodesic → NaN soft_SPL on RACE cw-2). Every episode is marked
    ``info['goal_changed']=True`` (+ diagnostic stale/goal positions). Categories
    with <2 instances (no genuine move) are SKIPPED.

    Mechanically this is the instance-keyed restriction (success keyed to ``[B]``)
    with the cold START moved onto A, so it reuses the existing goal-restriction
    path and needs NO runtime override. ``object_category`` stays "{cat}" → the
    "there is a {cat}" retrieval query is unchanged. Source is not mutated.
    """
    goals_by_category = src_content.get("goals_by_category") or {}
    src_eps = src_content.get("episodes") or []

    out_eps: List[Dict[str, Any]] = []
    out_goals: Dict[str, Any] = dict(goals_by_category)
    for cat in categories:
        gkey = _goals_key(goals_by_category, cat)
        if gkey is None:
            continue
        template = next((ep for ep in src_eps if ep.get("object_category") == cat), None)
        if template is None:
            continue
        goal_instances = goals_by_category[gkey]
        cold_inst = pick_cold_instance(goal_instances)
        warm_inst = pick_warm_instance(goal_instances, cold_inst, min_move=min_move)
        if warm_inst is None:
            continue  # single-instance category: no moved goal to express → skip

        cold_pose = pick_cold_pose([cold_inst])              # start AT A → seed it
        goal_vps_b = _goal_view_point_positions([warm_inst])  # warm distance ref = B
        cat_candidate_poses = [
            {"position": list(ep["start_position"]),
             "rotation": list(ep["start_rotation"])}
            for ep in src_eps
            if ep.get("object_category") == cat
            and ep.get("start_position") and ep.get("start_rotation")
        ]
        # Reachability-biased warm starts (see ``pick_warm_poses_changed_world``):
        # the farthest-first ``pick_warm_poses`` grabbed a start on a navmesh
        # component DISCONNECTED from B (→ Infinity geodesic / NaN soft_SPL on
        # RACE cw-2). Draw from the PROVEN-reachable region — instance A's
        # view_point poses (the cold episode starts there and reaches B) — plus
        # the category source starts, ranked NEAREST-to-B (same-component bias),
        # keeping a >= min_dist gap so a real path to B remains.
        a_reachable_poses = [
            {"position": list(vp), "rotation": list(cold_pose["rotation"])}
            for vp in _goal_view_point_positions([cold_inst])
        ]
        warm_poses = pick_warm_poses_changed_world(
            cat_candidate_poses, a_reachable_poses, goal_vps_b,
            n=n_warm, min_dist=min_dist)
        out_goals[gkey] = [warm_inst]                        # success keyed to B
        eps = build_category_episodes(template, cold_pose, warm_poses, cat)
        a_pos = _instance_centroid(cold_inst)
        b_pos = _instance_centroid(warm_inst)
        for ep in eps:
            info = ep.setdefault("info", {})
            info["goal_changed"] = True
            if a_pos is not None:
                info["stale_instance_position"] = a_pos
            if b_pos is not None:
                info["goal_instance_position"] = b_pos
        out_eps.extend(eps)

    return {
        "category_to_task_category_id": src_content.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src_content.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": out_goals,
        "episodes": out_eps,
    }


def _content_with_episodes(src: Dict[str, Any], episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "category_to_task_category_id": src.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": src.get("goals_by_category") or {},
        "episodes": episodes,
    }


def build_cross_env_dataset(
    home: "tuple[str, Dict[str, Any]]",
    away: "tuple[str, Dict[str, Any]]",
    categories: List[str],
    n_warm: int,
    min_dist: float = 2.0,
) -> Dict[str, Dict[str, Any]]:
    """Build a CROSS-ENVIRONMENT revisit dataset: the cold sighting accumulates
    in the ``home`` scene and the warm visit is queried in a DIFFERENT ``away``
    scene. This is the eval vehicle for the proposal's actual thesis
    (cross-environment reuse), which the same-scene revisit eval cannot test.

    ``home`` / ``away`` are ``(scene_name, src_content)`` tuples. For each
    category present in BOTH scenes:

      * one **cold** episode in ``home`` starting at the home goal's view_point
        (so the agent captions that category and the LTM holds a home sighting);
      * ``n_warm`` **warm** episodes in ``away`` starting far from the away goal
        (drawn from the away scene's OWN category starts so the start→goal
        geodesic is finite), where reaching the goal could benefit from the
        cross-env recall — IF the system transfers across scenes.

    Returns ``{scene_name: content}`` for each scene that has ≥1 episode; each
    content keeps its own ``goals_by_category`` so success computes per scene.
    Run home-before-away in ONE process (memory persisting) and gate the warm
    away visits on ``LTM_CROSS_SCENE`` to measure cross-env transfer.
    """
    home_scene, home_src = home
    away_scene, away_src = away
    home_goals = home_src.get("goals_by_category") or {}
    away_goals = away_src.get("goals_by_category") or {}
    home_eps = home_src.get("episodes") or []
    away_eps = away_src.get("episodes") or []

    def _clone(template: Dict[str, Any], pose: Dict[str, Any], category: str,
               eid: str) -> Dict[str, Any]:
        ep = copy.deepcopy(template)
        ep["episode_id"] = eid
        ep["object_category"] = category
        ep["start_position"] = list(pose["position"])
        ep["start_rotation"] = list(pose["rotation"])
        return ep

    home_out: List[Dict[str, Any]] = []
    away_out: List[Dict[str, Any]] = []
    for cat in categories:
        hkey = _goals_key(home_goals, cat)
        akey = _goals_key(away_goals, cat)
        if hkey is None or akey is None:
            continue  # cross-env needs the category in BOTH scenes
        home_tmpl = next((ep for ep in home_eps if ep.get("object_category") == cat), None)
        away_tmpl = next((ep for ep in away_eps if ep.get("object_category") == cat), None)
        if home_tmpl is None or away_tmpl is None:
            continue

        # cold sighting in the home scene: start at the home goal view_point
        cold_pose = pick_cold_pose(home_goals[hkey])
        home_out.append(_clone(home_tmpl, cold_pose, cat, f"{cat}-cold-home-0"))

        # warm visits in the away scene, from the away scene's own reachable starts
        away_candidates = [
            {"position": list(ep["start_position"]), "rotation": list(ep["start_rotation"])}
            for ep in away_eps
            if ep.get("object_category") == cat
            and ep.get("start_position") and ep.get("start_rotation")
        ]
        away_vps = _goal_view_point_positions(away_goals[akey])
        warm_poses = pick_warm_poses(away_candidates, away_vps, n=n_warm, min_dist=min_dist)
        for i, pose in enumerate(warm_poses):
            away_out.append(_clone(away_tmpl, pose, cat, f"{cat}-warm-away-{i + 1}"))

    out: Dict[str, Dict[str, Any]] = {}
    if home_out:
        out[home_scene] = _content_with_episodes(home_src, home_out)
    if away_out:
        out[away_scene] = _content_with_episodes(away_src, away_out)
    return out


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


def _main_cross_env(args, parser) -> int:
    """Cross-environment mode: cold sighting in --home-scene, warm visit in
    --away-scene. Writes both scenes' content into one shared --out-dir (the
    runner loads them with --scene all), like the Phase-C multi-scene build."""
    missing = [f"--{a.replace('_', '-')}" for a in
               ("home_src", "home_scene", "away_src", "away_scene")
               if not getattr(args, a)]
    if missing:
        parser.error(f"--cross-env requires {', '.join(missing)}")

    home = (args.home_scene, _load_gz(args.home_src))
    away = (args.away_scene, _load_gz(args.away_src))
    scenes = build_cross_env_dataset(home, away, args.categories, args.n_warm, args.min_dist)
    if not scenes:
        print(f"ERROR: no cross-env episodes built for categories={args.categories} "
              f"(need each category in BOTH {args.home_scene} and {args.away_scene}).",
              file=sys.stderr)
        return 1

    top = None
    for scene_name, content in scenes.items():
        top = write_dataset(args.out_dir, scene_name, content, content)
        by_cat: Dict[str, int] = {}
        for ep in content["episodes"]:
            by_cat[ep["object_category"]] = by_cat.get(ep["object_category"], 0) + 1
        role = "cold" if scene_name == args.home_scene else "warm"
        print(f"  content/{scene_name}.json.gz ({role}): {len(content['episodes'])} episodes "
              f"({', '.join(f'{c}:{n}' for c, n in by_cat.items())})")
    print(f"wrote {top}")

    # re-load verify
    re = _load_gz(top)
    assert re["episodes"] == [], "top-level must have empty episodes"
    for scene_name in scenes:
        cj = _load_gz(os.path.join(args.out_dir, "content", f"{scene_name}.json.gz"))
        assert cj["episodes"] and "goals_by_category" in cj, f"content malformed: {scene_name}"
    print("  verify: re-loaded OK (top empty; home cold + away warm content present)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase-B1 controlled-start revisit dataset")
    parser.add_argument("--src", help="Source content json.gz "
                                      "(…/val_mini/content/<scene>.json.gz). Single-scene mode.")
    parser.add_argument("--scene", help="Bare scene name, e.g. wcojb4TFT35. Single-scene mode.")
    parser.add_argument("--categories", nargs="+", default=["chair", "bed"])
    parser.add_argument("--n-warm", type=int, default=3)
    parser.add_argument("--min-dist", type=float, default=2.0,
                        help="Min metres a warm start must be from any goal view_point.")
    parser.add_argument("--instance-keyed", action="store_true",
                        help="Restrict each category's goal set to the single "
                             "cold-sighted (highest-iou) instance, so reaching a "
                             "different same-category instance no longer counts "
                             "(gives instance discrimination a metric target).")
    parser.add_argument("--seed-distractors", action="store_true",
                        help="With --instance-keyed: ALSO emit one seed-only cold "
                             "episode per same-category DISTRACTOR (starting at its "
                             "view_point so the agent captions+seeds it into the LTM, "
                             "between cold and warm). The warm visit's retrieval then "
                             "faces MULTIPLE same-category sightings and must rank the "
                             "RIGHT one — a retrieval-level disambiguation test. Seed "
                             "episodes carry info['seed_only']=True; the analyzer "
                             "EXCLUDES them from cold/warm pairing (they only seed the "
                             "LTM). Default OFF -> byte-identical.")
    parser.add_argument("--n-distractors", type=int, default=2,
                        help="--seed-distractors: cap the number of (nearest-first) "
                             "distractor instances seeded per category, so chair "
                             "(9-12 instances) does not balloon into a dozen extra "
                             "episode runs. 0 disables seeding; <0 seeds ALL.")
    parser.add_argument("--changed-world", action="store_true",
                        help="Changed-world mode: cold starts AT instance A (seeds it) "
                             "but success is keyed to a DIFFERENT instance B, so the "
                             "cold sighting is STALE — the regime the M4 temporal head "
                             "was built for. Needs >=2 instances per category.")
    parser.add_argument("--min-move", type=float, default=1.5,
                        help="changed-world: min metres B must be from A (a genuine "
                             "move) while preferring the NEAREST such instance for "
                             "navmesh reachability.")
    parser.add_argument("--seed", type=int, default=None,
                        help="RESAMPLE the warm starts for a genuinely independent "
                             "SECOND sample of the warm-revisit headline (the pipeline "
                             "is otherwise fully deterministic). Picks a different valid "
                             "n-subset of the SAME eligible warm-start pool; the cold "
                             "pose + instance choice stay deterministic (success-keying "
                             "unchanged). Stamped as revisit_seed for provenance. Default "
                             "unset → byte-identical to the deterministic build.")
    parser.add_argument("--out-dir", required=True)
    # cross-environment mode (step 2): a sighting in --home-scene, queried in --away-scene.
    parser.add_argument("--cross-env", action="store_true",
                        help="Cross-environment mode: cold sighting in --home-scene, "
                             "warm visit in --away-scene. Requires the --home-*/--away-* args.")
    parser.add_argument("--home-src", help="Cross-env: home scene content json.gz (cold sighting).")
    parser.add_argument("--home-scene", help="Cross-env: home scene name.")
    parser.add_argument("--away-src", help="Cross-env: away scene content json.gz (warm visit).")
    parser.add_argument("--away-scene", help="Cross-env: away scene name.")
    args = parser.parse_args(argv)

    if args.cross_env:
        return _main_cross_env(args, parser)

    if not args.src or not args.scene:
        parser.error("--src and --scene are required (single-scene mode), or use --cross-env")

    if args.instance_keyed and args.changed_world:
        parser.error("--instance-keyed and --changed-world are mutually exclusive")
    if args.seed_distractors and not args.instance_keyed:
        parser.error("--seed-distractors requires --instance-keyed (success must be "
                     "keyed to the single target so a recalled DISTRACTOR mis-routes)")

    src = _load_gz(args.src)
    if args.changed_world:
        content = build_changed_world_dataset(src, args.categories, args.n_warm,
                                              args.min_dist, min_move=args.min_move)
    else:
        _n_distract = None if args.n_distractors < 0 else args.n_distractors
        content = build_dataset(src, args.categories, args.n_warm, args.min_dist,
                                instance_keyed=args.instance_keyed,
                                seed_distractors=args.seed_distractors,
                                n_distractors=_n_distract,
                                seed=args.seed)
    if not content["episodes"]:
        print(f"ERROR: no episodes built for categories={args.categories} "
              f"({'need >=2 instances per category for --changed-world; ' if args.changed_world else ''}"
              f"check they are present in {args.src}).", file=sys.stderr)
        return 1

    top = write_dataset(args.out_dir, args.scene, content, src)

    # report
    by_cat: Dict[str, int] = {}
    for ep in content["episodes"]:
        by_cat[ep["object_category"]] = by_cat.get(ep["object_category"], 0) + 1
    _mode = "CHANGED-WORLD" if args.changed_world else (
        "INSTANCE-KEYED" if args.instance_keyed else "category-level")
    print(f"wrote {top}")
    print(f"  mode: {_mode}")
    print(f"  content/{args.scene}.json.gz: {len(content['episodes'])} episodes")
    for cat, n in by_cat.items():
        print(f"    {cat}: 1 cold + {n - 1} warm")
    if args.instance_keyed:
        src_goals = src.get("goals_by_category") or {}
        for cat in args.categories:
            gkey = _goals_key(src_goals, cat)
            if gkey is None:
                continue
            n_src = len(src_goals.get(gkey) or [])
            out_inst = content["goals_by_category"].get(gkey) or []
            if out_inst:
                tgt = out_inst[0]
                n_vp = len(tgt.get("view_points") or [])
                oid = tgt.get("object_id", "?")
                print(f"    {cat}: goal set {n_src} -> {len(out_inst)} instance "
                      f"(object_id={oid}, {n_vp} view_points)")
    else:
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
