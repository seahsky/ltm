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
# embodied_memory/ (the parent dir) on path so the light, faiss-free modules
# audio.py / room_resolver.py import directly — keeps this builder "pure data".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_revisit_smoke as mk  # noqa: E402
import make_audiogoal_smoke as ag  # noqa: E402
import audio as _audio  # noqa: E402  (embodied_memory/audio.py — numpy only)
import room_resolver as _rr  # noqa: E402  (dependency-free str ops)

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
# ADR-0003: source within this |dy| of the goal = the SAME FLOOR. Matches the
# render path's own band (render_rir_grid._nearest_same_floor, y_tol=1.0) and the
# runtime guard's default (run_hm3d_pol._MAX_DY_DEFAULT), so builder, render and
# runtime all agree on what a floor is.
_MAX_SOURCE_DY_DEFAULT = 1.0
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
    max_source_dy_m: float = _MAX_SOURCE_DY_DEFAULT,
) -> Dict[str, Any]:
    """Choose the DECOUPLED anomaly source: a real, navmesh-validated goal
    view_point of a DIFFERENT object, ``>= min_sep_m`` (xz) AND within
    ``max_source_dy_m`` in y (the SAME FLOOR) of the primary goal view_point
    ``primary_goal_pos``.

    The floor constraint is not cosmetic (ADR-0003). The RIR grid is rendered on
    the SOURCE's floor only, and the live lookup resolves cells by xz, so an
    off-floor source hands every on-goal-floor start a fabricated impulse
    response — the agent "hears" the sound through a storey of concrete. Worse,
    the xz-only bar SYSTEMATICALLY PREFERS cross-floor sources: they are xz-near
    (so they win the nearest-first tie-break) while still clearing ``min_sep_m``.
    That is precisely what happened in TEEsavR23oF (bed upstairs at y≈3.16, chair
    picked downstairs at y≈0.16, 3.56 m away in xz), where it also made the
    detour a stair-climb no investigate budget could fund. The band matches the
    render path's own same-floor tolerance (``render_rir_grid._nearest_same_floor``,
    ``y_tol=1.0``) so the builder and the render agree on what a floor is.

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
            # A navmesh-valid pose near the object = the first view_point that
            # carries a position. (NOT pick_cold_pose, which selects the highest-iou
            # view_point even when THAT vp lacks a position — a malformed-instance
            # edge that would raise and skip the whole category. _goal_view_point_positions
            # already filters to vps with a real position, so vps[0] is safe.)
            pos = list(vps[0])
            d = _xz_dist(pos, primary_goal_pos)
            if d is None or d < min_sep_m:
                continue
            # ADR-0003: same floor as the goal, else the grid rendered at this
            # source cannot describe the goal's floor and every on-goal-floor
            # start hears fabricated audio. Checked BEFORE the nearest-first
            # tie-break, which would otherwise actively prefer the cross-floor
            # candidate for being xz-near.
            if len(pos) < 3 or len(primary_goal_pos) < 3:
                continue
            if abs(float(pos[1]) - float(primary_goal_pos[1])) > max_source_dy_m:
                continue
            candidates.append((d, gkey == primary_gkey, cat, list(pos), inst.get("object_id")))
    if not candidates:
        raise ValueError(
            f"no object >= {min_sep_m}m (xz) and within {max_source_dy_m}m in y "
            f"(same floor, ADR-0003) of the primary '{primary_category}' "
            f"goal to decouple the anomaly source (single-object scene?)")
    # different category (False) before same category (True); then NEAREST first.
    candidates.sort(key=lambda t: (t[1], t[0]))
    _, _, obj, pos, oid = candidates[0]
    return {"position": pos, "anomaly_object": obj, "object_id": oid}


# ----------------------------------------------------------------------
# ADR-0002 same-sound / two-rooms scene-conditioning variant (P3.2)
# ----------------------------------------------------------------------
#
# One AMBIGUOUS clip (running water / appliance hum) is placed at TWO decoupled
# sources: one whose room makes the sound room-NORMAL (the agent must NOT
# interrupt) and one whose room makes it room-ANOMALOUS (must interrupt). The room
# flips the ground-truth verdict on the SAME clip — so the interrupt decision
# genuinely depends on the room-conditioned gate, not on the audio alone. Each
# episode still has a SINGLE source (single RIR grid → the O(1) live-convolution
# invariant is preserved; we deliberately reject a 2-source distractor).


def expected_interrupt(ambiguous_class: str, source_object_category: str) -> Optional[bool]:
    """Ground-truth discrimination label (the metric's target): should the agent
    INTERRUPT when ``ambiguous_class`` fires from a source at an object of
    ``source_object_category``? ``True`` iff the sound is UNEXPECTED (anomalous) in
    that object's room, ``False`` if room-normal, ``None`` if the room is
    unknown/uncovered (cannot scene-condition). Pure: room from the object's
    category prior (``room_resolver``), normality from ``audio.ROOM_PRIOR``."""
    room = _rr.preferred_room(source_object_category)
    return _audio.room_conditioned_anomaly(ambiguous_class, room, _audio.ROOM_PRIOR)


def pick_two_rooms_sources(
    goals_by_category: Dict[str, Any],
    all_categories: List[str],
    primary_category: str,
    primary_goal_pos: List[float],
    ambiguous_class: str,
    *,
    min_sep_m: float = _MIN_SOURCE_SEP_DEFAULT,
    max_source_dy_m: float = _MAX_SOURCE_DY_DEFAULT,
) -> Dict[str, Dict[str, Any]]:
    """Pick a room-NORMAL and a room-ANOMALOUS decoupled source for the SAME
    ``ambiguous_class``. Each is a real navmesh-valid goal view_point of an object
    whose room (from ``expected_interrupt``) has the required polarity, ``>=
    min_sep_m`` (xz) from the primary goal, on the goal's FLOOR (within
    ``max_source_dy_m`` in y), NEAREST-first (reachability discipline, like
    ``pick_anomaly_source``). Returns ``{"normal": src, "anomalous": src}``.
    Raises ``ValueError`` if EITHER polarity has no qualifying object (the scene
    cannot exercise the two-rooms test → skip the cell).

    The floor constraint applies per-polarity (ADR-0003): each family renders its
    OWN single-source grid at its OWN source, so an off-floor source fabricates
    the audio for that whole family — and this arm carries the paper's entire
    discrimination claim (ADR-0004), which would then be measuring a sound heard
    through a ceiling."""
    normal: List[Dict[str, Any]] = []
    anomalous: List[Dict[str, Any]] = []
    for cat in all_categories:
        gkey = _goals_key(goals_by_category, cat)
        if gkey is None:
            continue
        pol = expected_interrupt(ambiguous_class, cat)
        if pol is None:            # object's room carries no normality knowledge
            continue
        for inst in goals_by_category.get(gkey) or []:
            vps = _goal_view_point_positions([inst])
            if not vps:
                continue
            pos = list(vps[0])
            d = _xz_dist(pos, primary_goal_pos)
            if d is None or d < min_sep_m:
                continue
            if len(pos) < 3 or len(primary_goal_pos) < 3:
                continue
            if abs(float(pos[1]) - float(primary_goal_pos[1])) > max_source_dy_m:
                continue      # different floor → this family's grid would fabricate
            rec = {"position": list(pos), "anomaly_object": cat,
                   "object_id": inst.get("object_id"), "_d": float(d)}
            (anomalous if pol else normal).append(rec)
    if not normal or not anomalous:
        raise ValueError(
            f"two-rooms needs BOTH a room-normal and a room-anomalous source for "
            f"'{ambiguous_class}' on the goal's floor, >= {min_sep_m}m (xz) and "
            f"within {max_source_dy_m}m in y (normal={len(normal)}, "
            f"anomalous={len(anomalous)}); the scene lacks the required room "
            "diversity on one floor")
    normal.sort(key=lambda r: r["_d"])
    anomalous.sort(key=lambda r: r["_d"])
    return {"normal": normal[0], "anomalous": anomalous[0]}


def _tag_family(eps: List[Dict[str, Any]], tag: str, expected: bool) -> List[Dict[str, Any]]:
    """Return a NEW list of episodes with the ``tag`` prefixed onto each
    ``episode_id`` (keeps the ``-cold-``/``-warm-`` markers the construction gate
    greps) and the ground-truth ``expected_interrupt`` stamped. Pure — the input
    episodes are deep-copied, never mutated (global functional-programming rule)."""
    import copy
    out: List[Dict[str, Any]] = []
    for e in eps:
        ne = copy.deepcopy(e)
        ne["episode_id"] = f"{tag}-{e['episode_id']}"
        ne.setdefault("info", {})["expected_interrupt"] = bool(expected)
        out.append(ne)
    return out


def build_two_rooms_dataset(
    src_content: Dict[str, Any],
    categories: List[str],
    n_warm: int,
    min_dist: float = 2.0,
    *,
    ambiguous_class: str,
    min_source_sep_m: float = _MIN_SOURCE_SEP_DEFAULT,
    max_source_dy_m: float = _MAX_SOURCE_DY_DEFAULT,
    t_anom_cold: int = _T_ANOM_COLD_DEFAULT,
    t_anom_warm: int = _T_ANOM_WARM_DEFAULT,
    background_class: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the same-sound / two-rooms scene-conditioning dataset (P3.2).

    Per category: reuse the primary cold/warm poses (as ``build_dataset``), pick a
    room-normal AND a room-anomalous source (:func:`pick_two_rooms_sources`), and
    stamp the SAME ``ambiguous_class`` at each — one family tagged ``normal-``
    (``expected_interrupt=False``) and one ``anom-`` (``True``). A category with no
    valid two-rooms pair is skipped (not a crash)."""
    goals_by_category = src_content.get("goals_by_category") or {}
    src_eps = src_content.get("episodes") or []
    all_cats = list((src_content.get("category_to_task_category_id") or {}).keys()) or list(categories)

    out_eps: List[Dict[str, Any]] = []
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
            pair = pick_two_rooms_sources(
                goals_by_category, all_cats, cat, cold_pose["position"],
                ambiguous_class, min_sep_m=min_source_sep_m,
                max_source_dy_m=max_source_dy_m)
        except ValueError:
            continue

        for tag, expected, key in (("normal", False, "normal"), ("anom", True, "anomalous")):
            src = pair[key]
            fam = ag.build_category_episodes(
                template, cold_pose, warm_poses, cat,
                anomaly_class=ambiguous_class, anomaly_object=src["anomaly_object"],
                source_position=src["position"],
                t_anom_cold=t_anom_cold, t_anom_warm=t_anom_warm,
                background_class=background_class)
            out_eps.extend(_tag_family(fam, tag, expected))

    return {
        "category_to_task_category_id": src_content.get("category_to_task_category_id", {}),
        "category_to_scene_annotation_category_id":
            src_content.get("category_to_scene_annotation_category_id", {}),
        "goals_by_category": dict(goals_by_category),
        "episodes": out_eps,
    }


def two_rooms_construction_issues(content: Dict[str, Any]) -> List[str]:
    """``$0`` gate for a two-rooms dataset: BOTH polarities present, the two
    families share ONE clip (same anomaly_class), and every ``expected_interrupt``
    label agrees with its object's room-conditioned verdict (no mislabeled cell).
    Returns a list of issue strings (empty ⇒ OK)."""
    issues: List[str] = []
    eps = content.get("episodes") or []
    normals = [e for e in eps if (e.get("info") or {}).get("expected_interrupt") is False]
    anoms = [e for e in eps if (e.get("info") or {}).get("expected_interrupt") is True]
    if not normals:
        issues.append("FAIL: no room-NORMAL episode (expected_interrupt=False)")
    if not anoms:
        issues.append("FAIL: no room-ANOMALOUS episode (expected_interrupt=True)")
    classes = {(e.get("info") or {}).get("anomaly_class") for e in eps}
    if len(classes) > 1:
        issues.append(f"FAIL: two-rooms must use ONE ambiguous clip; saw classes {classes}")
    for e in eps:
        info = e.get("info") or {}
        cls, obj = info.get("anomaly_class"), info.get("anomaly_object")
        label = info.get("expected_interrupt")
        if label is None:
            continue
        truth = expected_interrupt(cls, obj)
        if truth is not None and bool(truth) != bool(label):
            issues.append(
                f"FAIL: {e.get('episode_id')} expected_interrupt={label} disagrees with "
                f"room verdict for {cls}@{obj} (={truth})")
    return issues


def build_dataset(
    src_content: Dict[str, Any],
    categories: List[str],
    n_warm: int,
    min_dist: float = 2.0,
    *,
    anomaly_class: Optional[str] = None,
    min_source_sep_m: float = _MIN_SOURCE_SEP_DEFAULT,
    max_source_dy_m: float = _MAX_SOURCE_DY_DEFAULT,
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
                min_sep_m=min_source_sep_m, max_source_dy_m=max_source_dy_m)
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
    ap.add_argument("--anomaly-class", default=None,
                    choices=["baby_cry", "alarm", "glass_break"],
                    help="unambiguous anomaly class (required unless --two-rooms)")
    ap.add_argument("--two-rooms", action="store_true",
                    help="ADR-0002 same-sound / two-rooms scene-conditioning variant: place "
                         "one AMBIGUOUS clip room-normal (no interrupt) vs room-anomalous "
                         "(interrupt). Requires --ambiguous-class.")
    ap.add_argument("--ambiguous-class", default=None,
                    choices=list(_audio.AMBIGUOUS_CLASSES),
                    help="the context-dependent sound for --two-rooms (running_water / appliance_hum)")
    ap.add_argument("--background-class", default=None)
    ap.add_argument("--min-source-sep", type=float, default=_MIN_SOURCE_SEP_DEFAULT)
    ap.add_argument("--max-source-dy", type=float, default=_MAX_SOURCE_DY_DEFAULT,
                    help="ADR-0003: source must be within this |dy| of the goal (same floor). "
                         "An off-floor source makes the single-floor RIR grid fabricate audio "
                         "for every on-goal-floor start. Matches render_rir_grid y_tol=1.0.")
    ap.add_argument("--min-dist", type=float, default=2.0)
    ap.add_argument("--t-anom-warm", type=int, default=_T_ANOM_WARM_DEFAULT)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="anomaly_response")
    ap.add_argument("--source-manifest", default=None)
    args = ap.parse_args(argv)

    # Validate the mode BEFORE any I/O so a bad invocation fails fast (argparse exit).
    if args.two_rooms and not args.ambiguous_class:
        ap.error("--two-rooms requires --ambiguous-class (running_water / appliance_hum)")
    if not args.two_rooms and not args.anomaly_class:
        ap.error("--anomaly-class is required unless --two-rooms is set")

    src_content = mk._load_gz(args.src)
    if args.two_rooms:
        content = build_two_rooms_dataset(
            src_content, args.categories, args.n_warm, min_dist=args.min_dist,
            ambiguous_class=args.ambiguous_class, min_source_sep_m=args.min_source_sep,
            max_source_dy_m=args.max_source_dy,
            t_anom_warm=args.t_anom_warm, background_class=args.background_class)
        issues = (two_rooms_construction_issues(content)
                  + anomaly_response_construction_issues(content, min_source_sep_m=args.min_source_sep))
    else:
        content = build_dataset(
            src_content, args.categories, args.n_warm, min_dist=args.min_dist,
            anomaly_class=args.anomaly_class, min_source_sep_m=args.min_source_sep,
            max_source_dy_m=args.max_source_dy,
            t_anom_warm=args.t_anom_warm, background_class=args.background_class)
        issues = anomaly_response_construction_issues(content, min_source_sep_m=args.min_source_sep)

    for s in issues:
        print(f"[construction] {s}")
    fails = [s for s in issues if s.startswith("FAIL")]
    if fails:
        print(f"[construction] {len(fails)} FAIL(s) — refusing to write.")
        return 1
    if not content.get("episodes"):
        print("[construction] FAIL: 0 episodes built (no category could decouple a "
              "source? single-object scene) — refusing to write.")
        return 1

    write_dataset(args.out_dir, args.scene, content, src_content, name=args.name)
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
