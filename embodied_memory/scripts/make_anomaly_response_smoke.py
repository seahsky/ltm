"""
make_anomaly_response_smoke — N3 dataset builder for the anomaly-RESPONSE task.

The anomaly-response task is a PRIMARY find-goal (ObjectNav) with an anomaly-sound
INTERRUPT: the agent searches for ``object_category``; an abnormal sound fires
from ``source_position`` during the search; the agent must INVESTIGATE the source,
CHECK, then RESUME and complete the find-task, then REPORT. The headline is the
interrupt→investigate→resume→report CONTROLLER as a working system.

Why a NEW module (not make_audiogoal_smoke): the AudioGoal builder pins the
source ~0.5 m from the goal view_point (``pick_source_position(offset_m=0.5)``,
``anomaly_object=cat``) → source==goal, line-of-sight. That makes the investigate
detour DEGENERATE (investigating == reaching the goal) and, because the agent's
cold start is the goal view_point (the LOUDEST RIR cell), the loud diotic
background bed FALSE-FIRES the CLAP anomaly gate at step 0. N3 fixes this
STRUCTURALLY by DECOUPLING the source from the goal:

  * ``object_category`` = the genuine find-target (drives SPL/success/report).
  * ``source_position`` = a DIFFERENT real object's navmesh-validated view_point,
    ``>= min_source_sep_m`` (xz) from the primary goal → the detour is real and the
    agent starts away from the loudest cell (the audible-not-loud regime where the
    recalibrated convolved text-gate works, EER 0.094 / delta -0.2557).
  * ``anomaly_object`` = the object AT the source (may DIFFER from the find-target).
  * ``t_anom`` = M3 polarity: cold pass silent (high, maps the scene), warm/response
    pass fires (low, interrupts the search).

The RUNTIME already reads goal (``ep.target_category``) and source
(``metadata.audio_config.source_position``) as SEPARATE fields, so decoupling is a
pure DATA problem — zero runtime change. This module reuses make_audiogoal_smoke /
make_revisit_smoke VERBATIM and leaves the AudioGoal path byte-identical.

A/B/C are RUN/analysis splits, not builder variants: A(warm)=seed+response under
S3; C(cold)=response under --disable-ltm/S1 (same dataset); B(new audio)=A with a
different clip. Controller-only arm = --task anomaly_response --disable-ltm + the
oracle source. The source is GT-privileged (oracle) — an UPPER BOUND until a
realizable DOA-derived source (a separate arm). The audible-not-loud, reachable-
to-goal feasibility of the start is a cheap RENDER gate on RACE (the builder can't
compute point-to-point geodesics — the two-env split), not a pure build-time check.

This is pure data (no Habitat / sim / captioner). Reuse, don't copy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_revisit_smoke as mk  # noqa: E402
import make_audiogoal_smoke as ag  # noqa: E402

# Reuse verbatim (single source of truth).
pick_cold_pose = mk.pick_cold_pose
pick_warm_poses = mk.pick_warm_poses
_goal_view_point_positions = mk._goal_view_point_positions
_goals_key = mk._goals_key
_xz_dist = ag._xz_dist
write_dataset = mk.write_dataset

# M3 polarity: the cold pass silent-maps the scene, the warm/response pass FIRES
# the anomaly during the search so it INTERRUPTS the find-task.
_T_ANOM_COLD_DEFAULT = ag._T_ANOM_COLD_DEFAULT   # 10000 (silent)
_T_ANOM_WARM_DEFAULT = ag._T_ANOM_WARM_DEFAULT   # 30    (fires)
_MIN_SOURCE_SEP_DEFAULT = 3.0                    # source >= this (xz) from the goal vp
# A warm t_anom above this is treated as "never fires during search" (FAIL); a
# cold t_anom at/below it is "not silent" (FAIL). Matches the audiogoal gate.
_FIRE_T_BOUND = 100


def pick_anomaly_source(
    goals_by_category: Dict[str, Any],
    all_categories: List[str],
    primary_category: str,
    primary_goal_pos: List[float],
    *,
    min_sep_m: float = _MIN_SOURCE_SEP_DEFAULT,
) -> Dict[str, Any]:
    """Choose the DECOUPLED anomaly source: a real, navmesh-validated goal
    view_point of a DIFFERENT object, ``>= min_sep_m`` (xz) from the primary goal
    view_point ``primary_goal_pos``.

    Preference order: a DIFFERENT category first (so ``anomaly_object`` differs
    from the find-target — the genuinely-decoupled regime), else a DIFFERENT
    INSTANCE of the primary category. Among qualifiers pick the NEAREST that still
    clears ``min_sep_m`` — proximity correlates with the same navmesh component
    (the farthest-first pick has repeatedly landed on disconnected islands →
    Infinity geodesic → NaN soft_SPL). Using a real goal view_point gives navmesh
    validity AND a real captioned object at the source for free.

    Returns ``{position, anomaly_object, object_id}``. Raises ``ValueError`` if the
    scene has no object ``>= min_sep_m`` from the goal (cannot decouple → skip the
    cell). Pure (no sim); the source's actual navmesh snap happens in
    ``render_rir_grid``."""
    primary_gkey = _goals_key(goals_by_category, primary_category)
    # candidate = (xz_dist, is_same_category(bool), anomaly_object, position, object_id)
    candidates: List[Any] = []
    for cat in all_categories:
        gkey = _goals_key(goals_by_category, cat)
        if gkey is None:
            continue
        for inst in goals_by_category.get(gkey) or []:
            vps = _goal_view_point_positions([inst])
            if not vps:
                continue
            # highest-iou view_point = navmesh-valid pose near the object.
            pos = pick_cold_pose([inst])["position"]
            d = _xz_dist(pos, primary_goal_pos)
            if d is None or d < min_sep_m:
                continue
            candidates.append((d, gkey == primary_gkey, cat, list(pos), inst.get("object_id")))
    if not candidates:
        raise ValueError(
            f"no object >= {min_sep_m}m (xz) from the primary '{primary_category}' "
            f"goal to decouple the anomaly source (single-object scene?)")
    # different category (False) before same category (True); then NEAREST first.
    candidates.sort(key=lambda t: (t[1], t[0]))
    _, _, obj, pos, oid = candidates[0]
    return {"position": pos, "anomaly_object": obj, "object_id": oid}


def build_dataset(
    src_content: Dict[str, Any],
    categories: List[str],
    n_warm: int,
    min_dist: float = 2.0,
    *,
    anomaly_class: Optional[str] = None,
    min_source_sep_m: float = _MIN_SOURCE_SEP_DEFAULT,
    t_anom_cold: int = _T_ANOM_COLD_DEFAULT,
    t_anom_warm: int = _T_ANOM_WARM_DEFAULT,
    background_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the anomaly-response content. ``anomaly_class=None`` delegates to
    ``make_revisit_smoke.build_dataset`` VERBATIM (objectnav/revisit byte-identical).

    Otherwise, per category: pick the cold (goal view_point) seed + far warm
    starts (reusing the revisit machinery), pick a DECOUPLED source
    (``pick_anomaly_source``), and stamp via ``make_audiogoal_smoke.build_category_episodes``
    with ``anomaly_object`` = the DECOUPLED object and the M3 (fires-during-warm)
    t_anom polarity. A category with no decoupled source available is skipped."""
    if anomaly_class is None:
        return mk.build_dataset(src_content, categories, n_warm, min_dist=min_dist)

    goals_by_category = src_content.get("goals_by_category") or {}
    src_eps = src_content.get("episodes") or []
    all_cats = list((src_content.get("category_to_task_category_id") or {}).keys()) or list(categories)

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
            {"position": list(ep["start_position"]), "rotation": list(ep["start_rotation"])}
            for ep in src_eps
            if ep.get("object_category") == cat
            and ep.get("start_position") and ep.get("start_rotation")
        ]

        goal_instances = goals_by_category[gkey]
        cold_pose = pick_cold_pose(goal_instances)
        goal_vps = _goal_view_point_positions(goal_instances)
        warm_poses = pick_warm_poses(cat_candidate_poses, goal_vps, n=n_warm, min_dist=min_dist)

        try:
            source = pick_anomaly_source(
                goals_by_category, all_cats, cat, cold_pose["position"],
                min_sep_m=min_source_sep_m)
        except ValueError:
            continue   # cannot decouple in this scene/category → skip (not a crash)

        out_eps.extend(ag.build_category_episodes(
            template, cold_pose, warm_poses, cat,
            anomaly_class=anomaly_class, anomaly_object=source["anomaly_object"],
            source_position=source["position"],
            t_anom_cold=t_anom_cold, t_anom_warm=t_anom_warm,
            background_class=background_class))

    return {
        "category_to_task_category_id": src_content.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src_content.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": out_goals,
        "episodes": out_eps,
    }


def anomaly_response_construction_issues(
    content: Dict[str, Any],
    *,
    min_source_sep_m: float = _MIN_SOURCE_SEP_DEFAULT,
) -> List[str]:
    """``$0`` construction gate for an anomaly-RESPONSE dataset. Returns a list of
    issue strings (empty ⇒ OK). Pure — operates on the built ``content`` dict.

    Unlike ``make_audiogoal_smoke.lifelong_construction_issues`` this PERMITS
    ``anomaly_object != object_category`` (that is the whole point — the source is
    a decoupled, different object). It FAILs when:
      * a ``source_position`` is not 3D;
      * the source is co-located with the primary goal (xz(seed.start, source) <
        ``min_source_sep_m`` — the cold/seed start_position IS the goal view_point,
        so this measures source-vs-goal separation → the degenerate/loud-bed case);
      * the cold/seed t_anom is not silent (must be high so the mapping pass does
        not fire);
      * the warm/response t_anom does not fire during the search (must be low so
        the anomaly actually INTERRUPTS the find-task)."""
    issues: List[str] = []
    eps = content.get("episodes") or []
    warms = [e for e in eps if "-warm-" in str(e.get("episode_id", ""))]
    if not warms:
        issues.append("FAIL: no warm/response episode — the anomaly never fires")
    for e in eps:
        eid = e.get("episode_id")
        info = e.get("info") or {}
        src = info.get("source_position")
        if not src or len(src) < 3:
            issues.append(f"FAIL: {eid} source_position not 3D ({src})")
            continue
        t = info.get("t_anom")
        is_seed = "-cold-" in str(eid)
        if is_seed:
            if t is None or t <= _FIRE_T_BOUND:
                issues.append(f"FAIL: seed {eid} t_anom={t} must be HIGH (silent mapping)")
            # the cold/seed start_position IS the primary goal view_point
            # (pick_cold_pose), so this is the source-vs-goal decoupling check.
            d = _xz_dist(e.get("start_position"), src)
            if d is not None and d < min_source_sep_m:
                issues.append(
                    f"FAIL: {eid} source {d:.2f}m from the goal view_point < "
                    f"{min_source_sep_m}m → source co-located with the goal "
                    "(degenerate detour / loud-bed step-0 false-fire)")
        else:
            if t is None or t > _FIRE_T_BOUND:
                issues.append(
                    f"FAIL: warm {eid} t_anom={t} must be LOW (<= {_FIRE_T_BOUND}) so "
                    "the anomaly FIRES during the search")
    return issues


def collect_source_manifest(content: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read the decoupled anomaly sources back out of the built episodes (single
    source of truth = episode.info), deduped by (scene, object_category, class), so
    the driver renders exactly one RIR grid per source. Reuses the audiogoal
    manifest reader verbatim."""
    return ag.collect_source_manifest(content)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build an N3 anomaly-response dataset")
    ap.add_argument("--src", required=True, help="source ObjectNav content .json.gz")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--categories", nargs="+", required=True)
    ap.add_argument("--n-warm", type=int, default=2)
    ap.add_argument("--anomaly-class", required=True,
                    choices=["baby_cry", "alarm", "glass_break"])
    ap.add_argument("--background-class", default=None)
    ap.add_argument("--min-source-sep", type=float, default=_MIN_SOURCE_SEP_DEFAULT)
    ap.add_argument("--t-anom-warm", type=int, default=_T_ANOM_WARM_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source-manifest", default=None)
    args = ap.parse_args(argv)

    src_content = mk._load_gz(args.src)
    content = build_dataset(
        src_content, args.categories, args.n_warm,
        anomaly_class=args.anomaly_class, min_source_sep_m=args.min_source_sep,
        t_anom_warm=args.t_anom_warm, background_class=args.background_class)

    issues = anomaly_response_construction_issues(content, min_source_sep_m=args.min_source_sep)
    for s in issues:
        print(f"[construction] {s}")
    fails = [s for s in issues if s.startswith("FAIL")]
    if fails:
        print(f"[construction] {len(fails)} FAIL(s) — refusing to write.")
        return 1

    write_dataset(content, args.scene, args.out_dir)
    manifest = collect_source_manifest(content)
    if args.source_manifest:
        os.makedirs(os.path.dirname(os.path.abspath(args.source_manifest)), exist_ok=True)
        with open(args.source_manifest, "w") as f:
            json.dump(manifest, f, indent=2)
    for m in manifest:
        print(f"[SOURCE] {m['scene_id']} {m['object_category']} {m['anomaly_class']} "
              f"obj={m['anomaly_object']} @ {m['source_position']}")
    print(f"[make_anomaly_response_smoke] wrote {len(content['episodes'])} episodes "
          f"to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
