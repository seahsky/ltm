"""
make_audiogoal_smoke — warm-episode dataset builder for the AudioGoal task (M2).

A fork of ``make_revisit_smoke`` that reuses its controlled-start machinery
VERBATIM (cold = start at a high-iou goal view_point → seeds the LTM sighting;
warm = start far → must recall) and ADDS the anomaly config to each episode's
``info`` dict so the live loop renders + reacts to audio:

  * ``anomaly_class``   — baby_cry | alarm | glass_break (FSD50K / CLAP).
  * ``anomaly_object``  — the object the source sits near (defaults to the goal
    category, e.g. ``bed``); the runtime prefers it over the static
    CLASS_TO_OBJECT affordance, so warm recall queries the object the agent
    actually mapped.
  * ``source_position`` — the anomaly source world xyz (identical for the cold
    and warm episodes of one (scene, category, class) so they share ONE RIR
    grid). MVP placement = a small offset from the cold goal view_point; the
    M3 driver renders the grid at exactly this point (single source of truth =
    the source manifest).
  * ``t_anom``          — per-episode onset step: a high value on the cold pass
    (silent mapping) and ``t_anom_warm`` on the warm passes (the anomaly fires).

Cold-first ordering (episode_order pins shuffle=False) means the cold silent
pass seeds the LTM before the warm episodes recall from it, in one process.

This is pure data (no Habitat / sim / captioner). The source placement here is
a geometric MVP; the navmesh-reachability + "reliably captioned object" gates
run on RACE (render_rir_grid + the real captioner). Reuse, don't copy.

    python embodied_memory/scripts/make_audiogoal_smoke.py \
        --src .../val_mini/content/TEEsavR23oF.json.gz --scene TEEsavR23oF \
        --categories bed --n-warm 3 --anomaly-class alarm \
        --out-dir runs/audiogoal-ds --source-manifest runs/audiogoal-ds/source_manifest.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_revisit_smoke as mk  # noqa: E402

# Reuse the pure selection/IO functions verbatim (single source of truth).
pick_cold_pose = mk.pick_cold_pose
pick_cold_instance = mk.pick_cold_instance
pick_warm_poses = mk.pick_warm_poses
_goal_view_point_positions = mk._goal_view_point_positions
_goals_key = mk._goals_key
write_dataset = mk.write_dataset
_load_gz = mk._load_gz

_T_ANOM_COLD_DEFAULT = 10000   # >> max_steps → the cold pass never fires (silent)
_T_ANOM_WARM_DEFAULT = 30
# Lifelong cross-visit (oracle-source upper bound): INVERT the M3 polarity. The
# SEED (cold slot) FIRES the anomaly from step 1 so the audio→LTM write seeds the
# source in visit-1; the RECALL episodes (warm slots) are SILENT (high t_anom) +
# start FAR, so visit-2 navigation is driven by the LTM write, not re-heard audio.
_T_ANOM_SEED_DEFAULT = 1
_T_ANOM_RECALL_DEFAULT = 10000
_LIFELONG_MIN_DIST_DEFAULT = 4.0   # recall starts >= this from the source (> dedup 1.5)


def pick_source_position(cold_goal_vp_pos: List[float], *, offset_m: float = 0.5) -> List[float]:
    """MVP heuristic anomaly-source placement: a small +x offset from the cold
    goal view_point (so the source sits near the object the cold pass captions).
    The caller / M3 render gate validates navmesh-reachability (no pathfinder
    here)."""
    p = list(cold_goal_vp_pos)
    return [float(p[0]) + float(offset_m), float(p[1]), float(p[2])]


def build_category_episodes(
    template: Dict[str, Any],
    cold_pose: Dict[str, Any],
    warm_poses: List[Dict[str, Any]],
    category: str,
    *,
    anomaly_class: Optional[str] = None,
    anomaly_object: Optional[str] = None,
    source_position: Optional[List[float]] = None,
    t_anom_cold: int = _T_ANOM_COLD_DEFAULT,
    t_anom_warm: int = _T_ANOM_WARM_DEFAULT,
) -> List[Dict[str, Any]]:
    """Clone ``template`` into [cold, warm_1, ..., warm_k]. When ``anomaly_class``
    is None this is exactly ``make_revisit_smoke.build_category_episodes`` (same
    episode_ids, no audio info → objectnav/revisit byte-identical). Otherwise the
    episode_ids are class-qualified and each episode's ``info`` gains the anomaly
    config (a high t_anom for the cold pass, ``t_anom_warm`` for warm)."""
    if anomaly_class is None:
        return mk.build_category_episodes(template, cold_pose, warm_poses, category)

    obj = anomaly_object or category
    src = list(source_position) if source_position is not None else None
    out: List[Dict[str, Any]] = []

    def _clone(pose: Dict[str, Any], eid: str, is_cold: bool) -> Dict[str, Any]:
        ep = copy.deepcopy(template)
        ep["episode_id"] = eid
        ep["object_category"] = category
        ep["start_position"] = list(pose["position"])
        ep["start_rotation"] = list(pose["rotation"])
        info = ep.setdefault("info", {})          # preserve geodesic_distance etc.
        info["anomaly_class"] = anomaly_class
        info["anomaly_object"] = obj
        info["source_position"] = src
        info["t_anom"] = t_anom_cold if is_cold else t_anom_warm
        return ep

    out.append(_clone(cold_pose, f"{category}-{anomaly_class}-cold-0", True))
    for i, pose in enumerate(warm_poses):
        out.append(_clone(pose, f"{category}-{anomaly_class}-warm-{i + 1}", False))
    return out


def build_dataset(
    src_content: Dict[str, Any],
    categories: List[str],
    n_warm: int,
    min_dist: float = 2.0,
    instance_keyed: bool = False,
    *,
    anomaly_class: Optional[str] = None,
    source_position: Optional[List[float]] = None,
    offset_m: float = 0.5,
    t_anom_cold: int = _T_ANOM_COLD_DEFAULT,
    t_anom_warm: int = _T_ANOM_WARM_DEFAULT,
) -> Dict[str, Any]:
    """Assemble the content dict. With ``anomaly_class=None`` this is exactly
    ``make_revisit_smoke.build_dataset``. With a class set, each category also
    gets an anomaly source (the explicit ``source_position`` for all, else a
    per-category offset from that category's cold goal view_point) written into
    every episode's ``info``."""
    if anomaly_class is None:
        return mk.build_dataset(src_content, categories, n_warm, min_dist=min_dist,
                                instance_keyed=instance_keyed)

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

        cat_candidate_poses = [
            {"position": list(ep["start_position"]),
             "rotation": list(ep["start_rotation"])}
            for ep in src_eps
            if ep.get("object_category") == cat
            and ep.get("start_position") and ep.get("start_rotation")
        ]

        goal_instances = goals_by_category[gkey]
        if instance_keyed:
            target_inst = pick_cold_instance(goal_instances)
            cold_pose = pick_cold_pose([target_inst])
            goal_vps = _goal_view_point_positions([target_inst])
            out_goals[gkey] = [target_inst]
        else:
            cold_pose = pick_cold_pose(goal_instances)
            goal_vps = _goal_view_point_positions(goal_instances)
        warm_poses = pick_warm_poses(cat_candidate_poses, goal_vps, n=n_warm, min_dist=min_dist)

        src_xyz = (list(source_position) if source_position is not None
                   else pick_source_position(cold_pose["position"], offset_m=offset_m))
        out_eps.extend(build_category_episodes(
            template, cold_pose, warm_poses, cat,
            anomaly_class=anomaly_class, anomaly_object=cat, source_position=src_xyz,
            t_anom_cold=t_anom_cold, t_anom_warm=t_anom_warm))

    return {
        "category_to_task_category_id": src_content.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src_content.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": out_goals,
        "episodes": out_eps,
    }


def build_lifelong_dataset(
    src_content: Dict[str, Any],
    categories: List[str],
    n_warm: int,
    *,
    anomaly_class: str,
    source_position: Optional[List[float]] = None,
    offset_m: float = 0.5,
    min_dist: float = _LIFELONG_MIN_DIST_DEFAULT,
    instance_keyed: bool = False,
    t_anom_seed: int = _T_ANOM_SEED_DEFAULT,
    t_anom_recall: int = _T_ANOM_RECALL_DEFAULT,
) -> Dict[str, Any]:
    """Lifelong cross-visit AudioGoal (oracle-source upper bound). Mechanically this
    is :func:`build_dataset` with the t_anom polarity INVERTED — the SEED (cold
    slot) fires (``t_anom_seed``) and writes the source to the LTM, the RECALL
    episodes (warm slots) are silent (``t_anom_recall``) and start ``>= min_dist``
    away so visit-2 is driven by the LTM write, not re-heard audio.

    KNOWN CAVEAT (the ``$0`` checker flags it): the seed start is the goal
    view_point (``pick_cold_pose``), which is line-of-sight to the source, so the
    seed VISUALLY maps the source — the oracle audio write is then redundant with
    that visual sighting (write-ON == write-OFF). ``lifelong_construction_issues``
    surfaces this as a REDUNDANCY-RISK; a cheap write-OFF recall probe MEASURES it
    before any paid matrix. A non-redundant build needs an audible-but-not-LOS seed
    (a follow-up sub-build)."""
    return build_dataset(
        src_content, categories, n_warm, min_dist=min_dist,
        instance_keyed=instance_keyed, anomaly_class=anomaly_class,
        source_position=source_position, offset_m=offset_m,
        t_anom_cold=t_anom_seed, t_anom_warm=t_anom_recall)


def _xz_dist(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    """Floor-plane (x,z) distance — matches the propose-time dedup metric."""
    if not a or not b or len(a) < 3 or len(b) < 3:
        return None
    return float(((float(a[0]) - float(b[0])) ** 2 + (float(a[2]) - float(b[2])) ** 2) ** 0.5)


def lifelong_construction_issues(
    content: Dict[str, Any],
    *,
    audible_radius_m: float = 4.0,
    min_recall_dist_m: float = _LIFELONG_MIN_DIST_DEFAULT,
    dedup_radius_m: float = 1.5,
    seed_los_warn_m: float = 2.0,
) -> List[str]:
    """``$0`` construction gate for a lifelong AudioGoal dataset. Returns a list of
    issue strings (empty ⇒ OK to run); ``FAIL:`` = the write can never fire / never
    recall, ``REDUNDANCY-RISK:`` = the seed likely visually maps the source so the
    oracle write is redundant. Pure — operates on the built ``content`` dict, no sim.
    """
    issues: List[str] = []
    eps = content.get("episodes") or []
    seeds = [e for e in eps if "-cold-" in str(e.get("episode_id", ""))]
    recalls = [e for e in eps if "-warm-" in str(e.get("episode_id", ""))]
    if not seeds:
        issues.append("FAIL: no seed (cold) episode — nothing fires/writes")
    if not recalls:
        issues.append("FAIL: no recall (warm) episodes — nothing to measure")
    for e in eps:
        eid = e.get("episode_id")
        info = e.get("info") or {}
        src = info.get("source_position")
        if not src or len(src) < 3:
            issues.append(f"FAIL: {eid} source_position not 3D ({src})")
            continue
        if info.get("anomaly_object") != e.get("object_category"):
            issues.append(f"FAIL: {eid} anomaly_object {info.get('anomaly_object')} != "
                          f"object_category {e.get('object_category')} (recall query won't "
                          "match the written caption)")
        d = _xz_dist(e.get("start_position"), src)
        t = info.get("t_anom")
        is_seed = "-cold-" in str(eid)
        if is_seed:
            if t is None or t > 100:
                issues.append(f"FAIL: seed {eid} t_anom={t} must be small so it FIRES")
            if d is not None and d > audible_radius_m:
                issues.append(f"FAIL: seed {eid} start {d:.2f}m from source > audible "
                              f"radius {audible_radius_m}m → onset won't fire")
            if d is not None and d < seed_los_warn_m:
                issues.append(f"REDUNDANCY-RISK: seed {eid} start {d:.2f}m from source "
                              f"< {seed_los_warn_m}m → likely line-of-sight → seed visually "
                              "maps the source → oracle write redundant (write-ON==write-OFF)")
        else:
            if t is not None and t <= 100:
                issues.append(f"FAIL: recall {eid} t_anom={t} must be high so it's SILENT")
            far = max(min_recall_dist_m, dedup_radius_m)
            if d is not None and d < far:
                issues.append(f"FAIL: recall {eid} start {d:.2f}m from source < {far}m → "
                              "recalled waypoint deduped / source re-mapped by vision")
    return issues


def collect_source_manifest(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read the anomaly sources back out of the built episodes (single source of
    truth = episode.info), deduped by (scene_id, object_category, anomaly_class),
    so the M3 driver can render exactly one RIR grid per source."""
    seen = {}
    out: List[Dict[str, Any]] = []
    for ep in content.get("episodes") or []:
        info = ep.get("info") or {}
        cls = info.get("anomaly_class")
        if not cls:
            continue
        scene_id = ep.get("scene_id")
        scene_label = os.path.basename(str(scene_id)).split(".", 1)[0] if scene_id else None
        cat = ep.get("object_category")
        key = (scene_label, cat, cls)
        if key in seen:
            continue
        seen[key] = True
        out.append({
            "scene_id": scene_label,
            "object_category": cat,
            "anomaly_class": cls,
            "anomaly_object": info.get("anomaly_object"),
            "source_position": info.get("source_position"),
        })
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AudioGoal warm-episode dataset builder")
    parser.add_argument("--src", required=True,
                        help="Source content json.gz (…/val_mini/content/<scene>.json.gz)")
    parser.add_argument("--scene", required=True, help="Bare scene name, e.g. TEEsavR23oF")
    parser.add_argument("--categories", nargs="+", default=["bed"])
    parser.add_argument("--n-warm", type=int, default=3)
    parser.add_argument("--min-dist", type=float, default=2.0)
    parser.add_argument("--instance-keyed", action="store_true")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="audiogoal")
    parser.add_argument("--anomaly-class", required=True,
                        choices=["baby_cry", "alarm", "glass_break"])
    parser.add_argument("--source-position", default=None,
                        help="Override anomaly source 'x,y,z' for ALL categories "
                             "(else a +offset from each category's cold view_point)")
    parser.add_argument("--offset-m", type=float, default=0.5,
                        help="MVP source offset (m) from the cold goal view_point")
    parser.add_argument("--t-anom-cold", type=int, default=_T_ANOM_COLD_DEFAULT,
                        help="cold-pass onset step (high → silent mapping)")
    parser.add_argument("--t-anom-warm", type=int, default=_T_ANOM_WARM_DEFAULT,
                        help="warm-pass onset step (anomaly fires)")
    parser.add_argument("--lifelong", action="store_true",
                        help="Lifelong cross-visit (oracle-source upper bound): INVERT "
                             "the t_anom polarity — seed (cold) FIRES + writes, recall "
                             "(warm) is SILENT + starts far. Sets t_anom_cold=1, "
                             "t_anom_warm=10000, min_dist=4.0 unless overridden; runs the "
                             "$0 construction check and refuses to build on a FAIL.")
    parser.add_argument("--source-manifest", default=None,
                        help="Where to write the source manifest JSON "
                             "(default <out-dir>/source_manifest.json)")
    args = parser.parse_args(argv)

    src_pos = None
    if args.source_position:
        src_pos = [float(v) for v in args.source_position.replace(",", " ").split()]
        if len(src_pos) != 3:
            parser.error("--source-position must be 'x,y,z'")

    src = _load_gz(args.src)
    if args.lifelong:
        # Invert the t_anom polarity + far recall unless explicitly overridden.
        t_seed = (args.t_anom_cold if args.t_anom_cold != _T_ANOM_COLD_DEFAULT
                  else _T_ANOM_SEED_DEFAULT)
        t_recall = (args.t_anom_warm if args.t_anom_warm != _T_ANOM_WARM_DEFAULT
                    else _T_ANOM_RECALL_DEFAULT)
        min_dist = args.min_dist if args.min_dist != 2.0 else _LIFELONG_MIN_DIST_DEFAULT
        content = build_lifelong_dataset(
            src, args.categories, args.n_warm, anomaly_class=args.anomaly_class,
            source_position=src_pos, offset_m=args.offset_m, min_dist=min_dist,
            instance_keyed=args.instance_keyed, t_anom_seed=t_seed, t_anom_recall=t_recall)
        # $0 construction gate: refuse to build on a FAIL; surface redundancy-risk.
        issues = lifelong_construction_issues(content, min_recall_dist_m=min_dist)
        for i in issues:
            print(f"  [lifelong-check] {i}")
        fails = [i for i in issues if i.startswith("FAIL")]
        if fails:
            print(f"RED: lifelong construction has {len(fails)} FAIL(s) — not writing dataset")
            return 1
        print("  [lifelong-check] "
              + ("WARN: redundancy-risk present (above) — a write-OFF recall probe must "
                 "measure whether the oracle write is redundant before any paid matrix"
                 if issues else "GREEN: construction OK"))
    else:
        content = build_dataset(
            src, args.categories, args.n_warm, min_dist=args.min_dist,
            instance_keyed=args.instance_keyed, anomaly_class=args.anomaly_class,
            source_position=src_pos, offset_m=args.offset_m,
            t_anom_cold=args.t_anom_cold, t_anom_warm=args.t_anom_warm)
    if not content["episodes"]:
        print(f"RED: no episodes built for categories {args.categories} in {args.scene}")
        return 1

    top = write_dataset(args.out_dir, args.scene, content, src, name=args.name)

    manifest = collect_source_manifest(content)
    manifest_path = args.source_manifest or os.path.join(args.out_dir, "source_manifest.json")
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[make_audiogoal_smoke] scene={args.scene} class={args.anomaly_class} "
          f"episodes={len(content['episodes'])} -> {top}")
    print(f"  source manifest: {manifest_path}")
    for m in manifest:
        sp = m["source_position"]
        sp_str = ",".join(f"{v:.4f}" for v in sp) if sp else "None"
        print(f"SOURCE scene={m['scene_id']} class={m['anomaly_class']} "
              f"object={m['anomaly_object']} xyz={sp_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
