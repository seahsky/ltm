"""
reaudit_partb_seeded — the re-audit tool for the seed-distractor instance-keyed
revisit run (``runs/partb-seeded-s{1,2,3}``, commit 8747a7d).

WHY. The partb-seeded run reported warm soft-SPL S3-S1 = +0.2085 (n=17, 4 non-
finite pairs dropped from 21), wrong-instance recall ~35% aggregate, and recall
present@view_point 91%@1m. But the summary did NOT report the per-cell
VALID / DEGENERATE / UNREACHABLE verdict. Without it, "+0.2085 survives 35%
wrong-instance recall" cannot be distinguished from "the dataset never forced
the instance choice" — a DEGENERATE cell (target always geodesically nearest)
rewards go-to-nearest, reproducing the single-goal null, and a +0.2085 there is
NOT a de-confound. This tool closes that hole and two adjacent ones:

  (a) PER-CELL (scene, category) VALID / DEGENERATE / UNREACHABLE verdict —
      reuses ``check_instance_keyed_validity.scan_cells`` on the built
      instance-keyed ``content/<scene>.json.gz``. Geodesic with
      ``--use-pathfinder`` (RACE, habitat-sim navmesh); else Euclidean-xz proxy,
      LOUDLY flagged APPROX.
  (b) PER-CELL wrong-instance recall rate (+ the pooled aggregate, which should
      reproduce the ~35%) — reuses ``diagnose_goal_anchored_recall``'s
      ``_allegiance`` / ``_memory_cands`` logic, grouped by ``(scene, category)``.
  (c) DROP SENSITIVITY — which warm pairs were dropped for a non-finite
      soft_spl / Inf distance (the n=21->17), tabulated (scene, category,
      visit_order, why), and the paired warm S3-S1 mean WITH vs WITHOUT them.

Nothing here is reimplemented: the validity classifier, the wrong-instance
allegiance, the visit-order pairing, and the NaN-drop filter are all imported
from the existing scripts and merely RE-GROUPED / RE-SURFACED per cell.

Run (local Euclidean pre-check)::

    python embodied_memory/scripts/reaudit_partb_seeded.py \
        --run-dirs runs/partb-seeded-s1 runs/partb-seeded-s2 runs/partb-seeded-s3 \
        --episodes data/hm3d/datasets/objectnav/hm3d/v1/revisit_partb-seeded/content/*.json.gz

Run (RACE, true geodesic validity gate)::

    python embodied_memory/scripts/reaudit_partb_seeded.py \
        --run-dirs runs/partb-seeded-s1 runs/partb-seeded-s2 runs/partb-seeded-s3 \
        --episodes data/hm3d/datasets/objectnav/hm3d/v1/revisit_partb-seeded/content/*.json.gz \
        --use-pathfinder
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# (a) validity — reuse the classifier + record builder verbatim.
import check_instance_keyed_validity as ck  # noqa: E402
from check_seed_pose import _load_gz  # noqa: E402  (stdlib gz loader)

# (b) wrong-instance — reuse the allegiance + memory-candidate extractor.
import diagnose_goal_anchored_recall as dg  # noqa: E402
from diagnose_audio_doa_calib import _load_run_dirs, _memory_cands  # noqa: E402

# (c) drop sensitivity — reuse the loader + visit-order + non-finite filter.
import analyze_revisit as ar  # noqa: E402


# ----------------------------------------------------------------------
# scene-id canonicalization (the join-key / navmesh-lookup fix)
# ----------------------------------------------------------------------


def _canonical_scene_id(s: Optional[str]) -> Optional[str]:
    """Collapse BOTH scene_id representations to the same short scene token.

    The dataset CONTENT keys cells on the full glb path
    (``"hm3d/val/00802-wcojb4TFT35/wcojb4TFT35.basis.glb"``) while the RUN-DIR
    episodes key on the short id (``"wcojb4TFT35"``). The per-cell views are
    joined on scene_id, so they MUST agree — and ``_find_navmesh`` globs on the
    short token, so it must be handed the short token too. This maps either to the
    token before the first ``.`` of the basename (so ``…/wcojb4TFT35.basis.glb`` ->
    ``"wcojb4TFT35"``) and is idempotent on an already-short id.
    """
    if s is None:
        return None
    base = os.path.basename(str(s))
    # token before the first dot strips ``.basis.glb`` / ``.json.gz`` / ``.glb``.
    return base.split(".", 1)[0]


# ----------------------------------------------------------------------
# (a) PER-CELL VALIDITY
# ----------------------------------------------------------------------


def _read_content(path: str) -> Dict[str, Any]:
    """Load a built content/<scene>.json.gz (thin wrapper over the stdlib loader)."""
    return _load_gz(path)


def per_cell_validity_from_contents(contents: List[Dict[str, Any]],
                                    use_pathfinder: bool,
                                    dist_fns_by_scene: Optional[Dict[str, Any]] = None
                                    ) -> Dict[str, Any]:
    """Per-(scene, category) verdict over a list of already-loaded content dicts.

    ``dist_fns_by_scene`` (only with ``use_pathfinder``) maps a CANONICAL scene
    token (see ``_canonical_scene_id``) to a geodesic ``dist_fn(a, b)``; a scene
    without an entry falls back to the Euclidean proxy (so a single missing navmesh
    degrades that scene, not the run). The content episodes carry the FULL glb path
    as ``scene_id`` while ``_build_geodesic_dist_fns`` keys on the short token, so
    the lookup is canonicalized on BOTH sides — without this, geodesic SILENTLY
    fell back to Euclidean even when the navmesh loaded. Without ``use_pathfinder``
    everything is Euclidean.

    Returns the ``scan_cells`` shape (``{"cells": {(scene,cat): verdict_dict},
    "green": bool}``) with an added ``"approx"`` flag (True if ANY cell used the
    Euclidean proxy) and a ``"per_scene_geodesic"`` map
    (``{canonical_scene: "geodesic" | "euclidean"}``) so the caller can print an
    HONEST per-scene status.
    """
    dist_fns_by_scene = dist_fns_by_scene or {}
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    approx = False
    per_scene_geodesic: Dict[str, str] = {}
    for content in contents:
        # group this content's warm episodes by their scene's distance fn.
        # records_from_content keys on (scene_id, object_category); but the
        # dist_fn must be chosen per scene, so partition by scene first.
        by_scene: Dict[str, Dict[str, Any]] = {}
        for ep in content.get("episodes") or []:
            sid = ep.get("scene_id")
            by_scene.setdefault(sid, {"episodes": []})["episodes"].append(ep)
        for sid, sub in by_scene.items():
            # canonicalize the lookup key: content sid is the full glb path,
            # the dist_fn map is keyed on the short token.
            canon = _canonical_scene_id(sid)
            geo = dist_fns_by_scene.get(canon) if use_pathfinder else None
            if geo is None:
                approx = approx or use_pathfinder  # asked for geodesic, got proxy
                dist_fn = ck.euclidean_xz
                if use_pathfinder:
                    per_scene_geodesic[canon] = "euclidean"
            else:
                dist_fn = geo
                per_scene_geodesic[canon] = "geodesic"
            for key, recs in ck.records_from_content(sub, dist_fn).items():
                cells.setdefault(key, []).extend(recs)
    rep = ck.scan_cells(cells)
    rep["approx"] = (not use_pathfinder) or approx
    rep["per_scene_geodesic"] = per_scene_geodesic
    return rep


def _build_geodesic_dist_fns(content_paths: List[str], roots) -> Dict[str, Any]:
    """RACE-only: build a {canonical_scene: geodesic dist_fn} from per-scene
    navmeshes.

    Mirrors ``check_instance_keyed_validity``'s scene-from-filename + navmesh
    discovery + lazy habitat-sim load. The map is keyed on the CANONICAL short
    scene token (``_canonical_scene_id``) — and ``_find_navmesh`` is handed that
    same short token (NOT the glb path), so the ``*<scene>*.navmesh`` glob actually
    hits. A scene whose navmesh is missing or fails to load is simply absent from
    the returned map (=> that scene degrades to the Euclidean proxy in
    ``per_cell_validity_from_contents``). MOCKED in tests.
    """
    out: Dict[str, Any] = {}
    for path in content_paths:
        scene = _canonical_scene_id(os.path.basename(path))
        navmesh = ck._find_navmesh(scene, roots)
        if navmesh is None:
            print(f"  [warn] no navmesh for scene {scene!r} under {roots} — that "
                  f"scene degrades to Euclidean (APPROX).", file=sys.stderr)
            continue
        try:
            pf, shortest_path_cls = ck._load_pathfinder(navmesh)
        except Exception as ex:  # noqa: BLE001 (RACE env issues -> skip, don't crash)
            print(f"  [warn] pathfinder load failed for {scene}: {ex} — Euclidean "
                  f"(APPROX).", file=sys.stderr)
            continue
        out[scene] = ck.make_geodesic_dist_fn(pf, shortest_path_cls)
    return out


def run_validity(content_globs: List[str], use_pathfinder: bool,
                 navmesh_root: Optional[str]) -> Dict[str, Any]:
    """Top-level (a): expand globs, load contents, optionally build geodesic fns,
    classify per cell. Robust to a missing pathfinder (degrades to Euclidean +
    sets ``approx=True``)."""
    paths: List[str] = []
    for pat in content_globs:
        paths.extend(sorted(glob.glob(pat)))
    contents: List[Dict[str, Any]] = []
    loaded_paths: List[str] = []
    for path in paths:
        try:
            contents.append(_read_content(path))
            loaded_paths.append(path)
        except (OSError, ValueError):
            print(f"  [warn] could not read {path}", file=sys.stderr)
    dist_fns_by_scene: Dict[str, Any] = {}
    roots: List[str] = []
    if use_pathfinder:
        roots = [navmesh_root] if navmesh_root else list(ck.DEFAULT_NAVMESH_ROOTS)
        dist_fns_by_scene = _build_geodesic_dist_fns(loaded_paths, roots)
    rep = per_cell_validity_from_contents(
        contents, use_pathfinder=use_pathfinder, dist_fns_by_scene=dist_fns_by_scene)
    # promote the flat {canon: "geodesic"|"euclidean"} map into the richer
    # {canon: {"mode", "reason"}} shape format_geodesic_status consumes.
    status: Dict[str, Dict[str, Any]] = {}
    for scene, mode in (rep.get("per_scene_geodesic") or {}).items():
        reason = None
        if mode != "geodesic":
            reason = (f"navmesh not found under {roots} / habitat_sim import failed"
                      if roots else "no navmesh roots searched")
        status[scene] = {"mode": mode, "reason": reason}
    rep["per_scene_geodesic_status"] = status
    return rep


# ----------------------------------------------------------------------
# (b) PER-CELL WRONG-INSTANCE RECALL RATE
# ----------------------------------------------------------------------


def wrong_instance_by_cell(episodes: List[Dict[str, Any]]
                           ) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]],
                                      Optional[Dict[str, Any]]]:
    """Group the wrong-instance recall computation by ``(scene_id, target_category)``.

    Reuses ``diagnose_goal_anchored_recall._allegiance`` / ``_memory_cands``
    verbatim (the per-decision "most-confident recalled candidate is xz-nearer a
    DISTRACTOR than the keyed TARGET" rule). Episodes without ``instance_labels``
    (single-goal) contribute nothing (silent). Returns
    ``(per_cell, aggregate_or_None)``; the aggregate reproduces
    ``diagnose_goal_anchored_recall.wrong_instance_recall_rate``.
    """
    per_cell: Dict[Tuple[str, str], Dict[str, int]] = {}
    tot_fires = tot_wrong = 0
    for ep in episodes:
        labels = ep.get("instance_labels") or {}
        target = labels.get("target_center")
        distractors = labels.get("distractor_centers") or []
        if target is None or not distractors:
            continue
        key = (ep.get("scene_id"), ep.get("target_category"))
        cell = per_cell.setdefault(key, {"fires": 0, "wrong": 0})
        for d in ep.get("decisions", []):
            cands = _memory_cands(d)
            if not cands:
                continue
            cell["fires"] += 1
            tot_fires += 1
            tagged = [dg._allegiance(c["world_xy"], target, distractors) for c in cands]
            tag, _ = min(tagged, key=lambda t: t[1])  # nearest-to-any-instance
            if tag == "distractor":
                cell["wrong"] += 1
                tot_wrong += 1
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, c in per_cell.items():
        f = c["fires"]
        out[key] = {"fires": f, "wrong": c["wrong"],
                    "rate": (c["wrong"] / f) if f else float("nan")}
    agg = ({"fires": tot_fires, "wrong": tot_wrong, "rate": tot_wrong / tot_fires}
           if tot_fires else None)
    return out, agg


# ----------------------------------------------------------------------
# (c) DROP SENSITIVITY (the n=21 -> 17)
# ----------------------------------------------------------------------


def drop_sensitivity(s1_dir: str, s3_dir: str, metric: str = "soft_spl"
                     ) -> Dict[str, Any]:
    """Identify which warm pairs were dropped for a non-finite ``metric`` (the
    n=21 -> 17), and report the paired warm S3-S1 mean WITH vs WITHOUT them.

    Reuses ``analyze_revisit``'s loader + visit-order + ``_visit_key`` + the exact
    non-finite filter (``analyze_revisit._paired_delta`` lines 311-314). The
    per-pair soft_spl IS in the run dirs (per-episode JSON / summary.json), so this
    is fully computable locally.

    Returns:
      n_kept, n_dropped_nonfinite, dropped[(scene,cat,visit_order,s1_val,s3_val,why)],
      mean_without_drops (the headline path), mean_with_drops (non-finite if a
      dropped pair is forced in), n_with_drops.
    """
    s1_run = ar.load_revisit_run(s1_dir)
    s3_run = ar.load_revisit_run(s3_dir)
    ar.assign_visit_order(s1_run.episodes)
    ar.assign_visit_order(s3_run.episodes)

    # warm episodes, keyed renumbering-invariantly (same as _paired_delta).
    s1_by = {ar._visit_key(e): e for e in s1_run.episodes if e.is_warm}
    s3_by = {ar._visit_key(e): e for e in s3_run.episodes if e.is_warm}
    paired = sorted(set(s1_by) & set(s3_by))

    kept_deltas: List[float] = []
    all_deltas: List[float] = []
    dropped: List[Dict[str, Any]] = []
    for k in paired:
        v1 = getattr(s1_by[k], metric)
        v3 = getattr(s3_by[k], metric)
        delta = v3 - v1
        all_deltas.append(delta)
        f1, f3 = math.isfinite(v1), math.isfinite(v3)
        if f1 and f3:
            kept_deltas.append(delta)
        else:
            scene, cat, vo = k
            sides = []
            if not f1:
                sides.append("S1")
            if not f3:
                sides.append("S3")
            why = (f"{metric} non-finite in {'+'.join(sides)} "
                   f"(S1={v1}, S3={v3}; navmesh-unreachable goal => Inf geodesic "
                   f"distance_to_goal => NaN/Inf soft_spl)")
            dropped.append({"scene_id": scene, "target_category": cat,
                            "visit_order": vo, "s1_val": v1, "s3_val": v3,
                            "why": why})

    mean_without = (sum(kept_deltas) / len(kept_deltas)
                    if kept_deltas else float("nan"))
    mean_with = (sum(all_deltas) / len(all_deltas)
                 if all_deltas else float("nan"))
    return {
        "metric": metric,
        "n_paired": len(paired),
        "n_kept": len(kept_deltas),
        "n_dropped_nonfinite": len(dropped),
        "dropped": dropped,
        "mean_without_drops": mean_without,
        "mean_with_drops": mean_with,
        "n_with_drops": len(all_deltas),
    }


# ----------------------------------------------------------------------
# (d) PER-CELL WARM S3-S1 DELTA (where does the +0.2085 live?)
# ----------------------------------------------------------------------


def per_cell_delta(run_dirs: List[str], metric: str = "soft_spl"
                   ) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Per-(scene_id, target_category) cell, compute the paired WARM S3-S1 mean.

    Reuses the SAME machinery ``drop_sensitivity`` uses (``analyze_revisit``'s
    ``load_revisit_run`` / ``assign_visit_order`` / ``_visit_key`` + the
    ``math.isfinite`` finite-pair filter matching ``_paired_delta`` lines 311-314)
    — here merely RE-GROUPED per cell. The binary-SPL delta (the ``spl`` field) is
    computed alongside on the SAME kept finite pairs.

    Never crashes on a cell with 0 finite pairs: it reports
    ``n_warm_pairs_kept=0`` and ``delta_<metric>=None``.

    Returns ``{(scene, cat): {n_warm_pairs_kept, n_dropped_nonfinite,
    delta_softspl, delta_binary_spl}}``.
    """
    s1_dir, _s2_dir, s3_dir = _split_settings(run_dirs)
    if not s1_dir or not s3_dir:
        return {}

    s1_run = ar.load_revisit_run(s1_dir)
    s3_run = ar.load_revisit_run(s3_dir)
    ar.assign_visit_order(s1_run.episodes)
    ar.assign_visit_order(s3_run.episodes)

    # warm episodes, keyed renumbering-invariantly (same as _paired_delta / drop).
    s1_by = {ar._visit_key(e): e for e in s1_run.episodes if e.is_warm}
    s3_by = {ar._visit_key(e): e for e in s3_run.episodes if e.is_warm}
    paired = sorted(set(s1_by) & set(s3_by))

    # accumulate per cell: kept finite soft-spl deltas + the matching binary deltas.
    acc: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for k in paired:
        scene, cat, _vo = k
        cell = acc.setdefault((scene, cat),
                              {"soft": [], "binary": [], "dropped": 0})
        e1, e3 = s1_by[k], s3_by[k]
        m1, m3 = getattr(e1, metric), getattr(e3, metric)
        # finite filter on the PRIMARY metric — identical to _paired_delta:311-314.
        if math.isfinite(m1) and math.isfinite(m3):
            cell["soft"].append(m3 - m1)
            b1, b3 = getattr(e1, "spl"), getattr(e3, "spl")
            if math.isfinite(b1) and math.isfinite(b3):
                cell["binary"].append(b3 - b1)
        else:
            cell["dropped"] += 1

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, cell in acc.items():
        soft = cell["soft"]
        binary = cell["binary"]
        out[key] = {
            "n_warm_pairs_kept": len(soft),
            "n_dropped_nonfinite": cell["dropped"],
            "delta_softspl": (sum(soft) / len(soft)) if soft else None,
            "delta_binary_spl": (sum(binary) / len(binary)) if binary else None,
        }
    return out


# ----------------------------------------------------------------------
# (e) DECISIVE JOIN — left-join validity + wrong-instance + per-cell delta
# ----------------------------------------------------------------------


def build_join_rows(validity: Dict[Tuple[str, str], Dict[str, Any]],
                    wrong: Dict[Tuple[str, str], Dict[str, Any]],
                    deltas: Dict[Tuple[str, str], Dict[str, Any]]
                    ) -> List[Dict[str, Any]]:
    """LEFT-JOIN the three per-cell views into one row per cell.

    The row set is the UNION of cell keys across all three views (a cell present in
    only one view still gets a row; the missing pieces are None / 0, never a
    KeyError). Sorted by ``delta_softspl`` descending with None LAST.

    The three views key scene_id INCONSISTENTLY — the VALIDITY view keys on the
    dataset content's full glb path while the WRONG-INSTANCE / DELTA views key on
    the run-dir short id — so every (scene_id, category) tuple is collapsed via
    ``_canonical_scene_id`` BEFORE joining. Without this the left-join matched
    nothing (every row printed verdict "-" / delta "None" and the roll-up summed to
    n=0). Only the JOIN KEY is canonicalized; no value is altered.

    Each row: ``{scene_id, category, verdict, fires, wrong, rate, n_warm_pairs,
    n_dropped_nonfinite, delta_softspl, delta_binary_spl}``.
    """
    def _canon(view: Dict[Tuple[str, str], Dict[str, Any]]
               ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        return {(_canonical_scene_id(scene), cat): val
                for (scene, cat), val in view.items()}

    validity = _canon(validity)
    wrong = _canon(wrong)
    deltas = _canon(deltas)

    keys = set(validity) | set(wrong) | set(deltas)
    rows: List[Dict[str, Any]] = []
    for scene, cat in keys:
        v = validity.get((scene, cat)) or {}
        w = wrong.get((scene, cat)) or {}
        d = deltas.get((scene, cat)) or {}
        rows.append({
            "scene_id": scene,
            "category": cat,
            "verdict": v.get("verdict"),
            "fires": w.get("fires"),
            "wrong": w.get("wrong"),
            "rate": w.get("rate"),
            "n_warm_pairs": d.get("n_warm_pairs_kept"),
            "n_dropped_nonfinite": d.get("n_dropped_nonfinite"),
            "delta_softspl": d.get("delta_softspl"),
            "delta_binary_spl": d.get("delta_binary_spl"),
        })
    # sort by delta_softspl descending, None last (None -> sort key pushes to end).
    rows.sort(key=lambda r: (r["delta_softspl"] is None,
                             -(r["delta_softspl"] if r["delta_softspl"] is not None
                               else 0.0)))
    return rows


def bucket_rollup(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Partition the per-cell warm delta into two buckets so a reader can see
    whether the +0.2085 lives in cells that REQUIRED disambiguation or not.

      - ``degenerate_0wrong``: cells that are DEGENERATE *and* have 0% wrong-instance
        recall — "recall worked but the eval didn't force disambiguation".
      - ``valid``: cells classified VALID — disambiguation WAS forced.

    Each bucket reports n_pairs (sum of per-cell kept-pair n), weighted_delta_sum
    (sum of delta*n over cells with a finite delta), and mean_delta
    (weighted_delta_sum / n_pairs). Cells with a None delta or 0 kept pairs
    contribute nothing.
    """
    buckets = {"degenerate_0wrong": {"n_pairs": 0, "weighted_delta_sum": 0.0,
                                     "cells": 0},
               "valid": {"n_pairs": 0, "weighted_delta_sum": 0.0, "cells": 0}}
    for r in rows:
        delta = r.get("delta_softspl")
        n = r.get("n_warm_pairs") or 0
        if delta is None or n <= 0:
            continue
        verdict = r.get("verdict")
        rate = r.get("rate")
        if verdict == "DEGENERATE" and rate == 0.0:
            b = buckets["degenerate_0wrong"]
        elif verdict == "VALID":
            b = buckets["valid"]
        else:
            continue
        b["n_pairs"] += n
        b["weighted_delta_sum"] += delta * n
        b["cells"] += 1
    for b in buckets.values():
        b["mean_delta"] = (b["weighted_delta_sum"] / b["n_pairs"]
                           if b["n_pairs"] else None)
    return buckets


# ----------------------------------------------------------------------
# geodesic status (explicit + honest per-scene)
# ----------------------------------------------------------------------

_EUCLID_BANNER = (
    "############################################################\n"
    "##  VALIDITY IS PROVISIONAL (EUCLIDEAN) — re-run with conda env\n"
    "##  so habitat_sim imports (source scripts/race-setup.sh) and\n"
    "##  pass --use-pathfinder for the TRUE geodesic gate. A wall can\n"
    "##  flip a Euclidean-VALID cell to DEGENERATE/UNREACHABLE.\n"
    "############################################################")


def format_geodesic_status(per_scene: Dict[str, Dict[str, Any]]) -> str:
    """Render an HONEST per-scene geodesic status block.

    ``per_scene`` maps a canonical scene token to ``{"mode": "geodesic" |
    "euclidean", "reason": str | None}``. Prints one line per scene:
      * ``geodesic: <scene> -> navmesh OK (geodesic)`` when it loaded, or
      * ``geodesic: <scene> -> EUCLIDEAN fallback (reason: ...)`` when it didn't.
    Then EXACTLY ONE of:
      * the big EUCLIDEAN-PROVISIONAL banner — ONLY if >=1 scene fell back;
      * a ``VALIDITY IS GEODESIC (--use-pathfinder)`` confirmation — if all loaded.
    An empty map (no --use-pathfinder requested) renders nothing (silent).
    """
    if not per_scene:
        return ""
    lines: List[str] = []
    any_euclid = False
    for scene in sorted(per_scene):
        st = per_scene[scene] or {}
        mode = st.get("mode")
        if mode == "geodesic":
            lines.append(f"  geodesic: {scene} -> navmesh OK (geodesic)")
        else:
            any_euclid = True
            reason = st.get("reason") or "navmesh not found / habitat_sim import failed"
            lines.append(f"  geodesic: {scene} -> EUCLIDEAN fallback (reason: {reason})")
    if any_euclid:
        lines.append(_EUCLID_BANNER)
    else:
        lines.append("  VALIDITY IS GEODESIC (--use-pathfinder) — the true gate "
                     "loaded for every scene.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# READOUT / verdict
# ----------------------------------------------------------------------


def overall_verdict_line(cells: Dict[Tuple[str, str], Dict[str, Any]]) -> str:
    """One-line de-confound verdict: k/m seeded cells VALID => GENUINE, else the
    eval did not force disambiguation."""
    m = len(cells)
    k = sum(1 for v in cells.values() if v.get("verdict") == "VALID")
    if k >= 1:
        return (f"OVERALL: {k}/{m} seeded cells VALID -> the de-confound is GENUINE "
                f"(>=1 cell geodesically forces instance disambiguation, so the warm "
                f"S3-S1 gain is NOT explainable by go-to-nearest).")
    return (f"OVERALL: {k}/{m} seeded cells VALID -> DEGENERATE: the eval did NOT "
            f"force disambiguation (every cell has the target nearest / unreachable), "
            f"so the warm gain cannot be distinguished from go-to-nearest. Re-place "
            f"warm starts or add a scene BEFORE quoting +0.2085 as a de-confound.")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _split_settings(run_dirs: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Map run dirs to (S1, S2, S3) by the ``-s<N>`` token in the dir name."""
    s = {1: None, 2: None, 3: None}
    for d in run_dirs:
        base = os.path.basename(os.path.normpath(d))
        for n in (1, 2, 3):
            if base.endswith(f"-s{n}") or f"-s{n}-" in base or f"-s{n}_" in base:
                s[n] = d
    return s[1], s[2], s[3]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Re-audit the partb-seeded instance-keyed revisit run "
                    "(per-cell validity + wrong-instance + drop sensitivity).")
    ap.add_argument("--run-dirs", nargs="+", required=True,
                    help="runs/partb-seeded-s1 [s2] s3 (settings inferred from -sN)")
    ap.add_argument("--episodes", nargs="+", required=True,
                    help="built instance-keyed content .json.gz (glob ok) carrying "
                         "info.instance_labels and -warm- episode ids")
    ap.add_argument("--use-pathfinder", action="store_true",
                    help="RACE-only: true geodesic validity (habitat-sim navmesh); "
                         "else Euclidean-xz proxy (APPROX)")
    ap.add_argument("--navmesh-root", default=None,
                    help="(with --use-pathfinder) dir searched recursively for "
                         "<scene>*.navmesh; defaults to the HM3D scene_datasets dirs")
    ap.add_argument("--metric", default="soft_spl",
                    help="drop-sensitivity metric (soft_spl primary; spl binary)")
    args = ap.parse_args(argv)

    s1_dir, _s2_dir, s3_dir = _split_settings(args.run_dirs)

    # ---- (a) per-cell validity ----
    print("========== PER-CELL VALIDITY (VALID / DEGENERATE / UNREACHABLE) ==========")
    val = run_validity(args.episodes, args.use_pathfinder, args.navmesh_root)
    per_scene_status = val.get("per_scene_geodesic_status") or {}
    any_euclid_fallback = any(
        (st or {}).get("mode") != "geodesic" for st in per_scene_status.values())
    if args.use_pathfinder:
        # EXPLICIT, HONEST per-scene status: one line per scene + EITHER the big
        # banner (>=1 fell back) OR the all-geodesic confirmation.
        status_block = format_geodesic_status(per_scene_status)
        if status_block:
            print(status_block)
    if not args.use_pathfinder:
        # no geodesic requested at all -> Euclidean-xz proxy everywhere.
        print(_EUCLID_BANNER)
        print("  !!! APPROX: Euclidean-xz proxy (ignores walls). Re-run with "
              "--use-pathfinder on RACE for the TRUE geodesic gate. A wall can flip "
              "a Euclidean-VALID cell to DEGENERATE/UNREACHABLE. !!!")
    if not val["cells"]:
        print("  no instance-keyed warm episodes with labels found in --episodes "
              "(need built content with info.instance_labels and -warm- ids).")
    else:
        print(f"  {'scene':<16} {'category':<11} {'verdict':<22} reach/forces/warm")
        for (scene, cat), v in sorted(val["cells"].items(), key=lambda kv: str(kv[0])):
            print(f"  {str(scene):<16} {str(cat):<11} {v['verdict']:<22} "
                  f"{v['n_reachable']}/{v['n_forces']}/{v['n_warm']}")

    # ---- (b) per-cell wrong-instance recall ----
    print("\n========== PER-CELL WRONG-INSTANCE RECALL RATE ==========")
    episodes = _load_run_dirs(args.run_dirs)
    wrong_per_cell: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not episodes:
        print(f"  no episode_*.json under {args.run_dirs}")
    else:
        per_cell, agg = wrong_instance_by_cell(episodes)
        wrong_per_cell = per_cell
        if agg is None:
            print("  no instance_labels on any episode (single-goal run) -> "
                  "wrong-instance readout is silent.")
        else:
            print(f"  {'scene':<16} {'category':<11} wrong/fires = rate")
            for (scene, cat), c in sorted(per_cell.items(), key=lambda kv: str(kv[0])):
                print(f"  {str(scene):<16} {str(cat):<11} "
                      f"{c['wrong']}/{c['fires']} = {c['rate']:.0%}")
            print(f"  ---- AGGREGATE: {agg['wrong']}/{agg['fires']} = "
                  f"{agg['rate']:.0%}  (should reproduce the reported ~35%)")

    # ---- (c) drop sensitivity ----
    print("\n========== DROP SENSITIVITY (the n=21 -> 17 non-finite pairs) ==========")
    if not s1_dir or not s3_dir:
        print("  need both an -s1 and an -s3 run dir to pair the warm delta; "
              f"got run-dirs {args.run_dirs}. Skipping.")
        drop = None
    else:
        drop = drop_sensitivity(s1_dir, s3_dir, metric=args.metric)
        print(f"  metric={drop['metric']}  paired warm visits={drop['n_paired']}  "
              f"kept={drop['n_kept']}  dropped(non-finite)={drop['n_dropped_nonfinite']}")
        if drop["dropped"]:
            print(f"  {'scene':<16} {'category':<11} {'visit':<6} why")
            for r in drop["dropped"]:
                print(f"  {str(r['scene_id']):<16} {str(r['target_category']):<11} "
                      f"{r['visit_order']:<6} {r['why']}")
        mwo = drop["mean_without_drops"]
        mw = drop["mean_with_drops"]
        print(f"  paired warm S3-S1 WITHOUT drops (headline) : {mwo:+.4f}  (n={drop['n_kept']})")
        if math.isfinite(mw):
            print(f"  paired warm S3-S1 WITH    drops           : {mw:+.4f}  (n={drop['n_with_drops']})")
        else:
            print(f"  paired warm S3-S1 WITH    drops           : non-finite "
                  f"(n={drop['n_with_drops']}) — a dropped pair has Inf/NaN "
                  f"{drop['metric']}, so it CANNOT be averaged in; this is exactly "
                  f"why the analyzer drops it. The 4 dropped pairs are navmesh-"
                  f"adverse (unreachable goal), not cherry-picked.")

    # ---- (d) per-cell warm S3-S1 delta + (e) decisive join ----
    print("\n========== PER-CELL DECISIVE JOIN ==========")
    if not s1_dir or not s3_dir:
        print("  need both an -s1 and an -s3 run dir to compute the per-cell warm "
              f"delta; got run-dirs {args.run_dirs}. Skipping the join.")
        rows: List[Dict[str, Any]] = []
        roll: Dict[str, Any] = {}
    else:
        deltas = per_cell_delta(args.run_dirs, metric=args.metric)
        rows = build_join_rows(val["cells"], wrong_per_cell, deltas)
        roll = bucket_rollup(rows)
        print("  LEFT JOIN of the three per-cell views (one row per cell), "
              "sorted by S3-S1_softspl descending (None last):")
        print(f"  {'scene':<16} {'category':<10} {'verdict':<12} "
              f"{'wrong/fires=rate':<18} {'n_warm':<7} {'S3-S1_softspl':<14} "
              f"{'S3-S1_binary':<13}")
        for r in rows:
            verdict = r["verdict"] if r["verdict"] is not None else "-"
            if r["fires"]:
                wif = f"{r['wrong']}/{r['fires']}={r['rate']:.0%}"
            else:
                wif = "-/0"
            nwp = r["n_warm_pairs"] if r["n_warm_pairs"] is not None else "-"
            ds = (f"{r['delta_softspl']:+.4f}" if r["delta_softspl"] is not None
                  else "None")
            db = (f"{r['delta_binary_spl']:+.4f}" if r["delta_binary_spl"] is not None
                  else "None")
            print(f"  {str(r['scene_id']):<16} {str(r['category']):<10} "
                  f"{str(verdict):<12} {wif:<18} {str(nwp):<7} {ds:<14} {db:<13}")

        # ---- bucket roll-up: where does the +delta live? ----
        deg = roll.get("degenerate_0wrong", {})
        valb = roll.get("valid", {})
        print("\n  --- WHERE THE DELTA LIVES (warm soft-SPL, pair-weighted) ---")
        dmean = deg.get("mean_delta")
        vmean = valb.get("mean_delta")
        print(f"  delta carried by DEGENERATE+0%-wrong cells "
              f"(disambiguation NOT required): "
              f"sum={deg.get('weighted_delta_sum', 0.0):+.4f}, "
              f"n_pairs={deg.get('n_pairs', 0)}, cells={deg.get('cells', 0)}, "
              f"mean={'None' if dmean is None else f'{dmean:+.4f}'}")
        print(f"  delta in VALID cells "
              f"(disambiguation REQUIRED):              "
              f"sum={valb.get('weighted_delta_sum', 0.0):+.4f}, "
              f"n_pairs={valb.get('n_pairs', 0)}, cells={valb.get('cells', 0)}, "
              f"mean={'None' if vmean is None else f'{vmean:+.4f}'}")
        # PRE-REGISTERED VERDICT STUB — finalized by the verify agent, NOT hard-coded.
        print("\n  PRE-REGISTERED VERDICT [TODO — verify agent finalizes after the "
              "geodesic re-run]: read the two buckets above. IF the warm soft-SPL "
              "delta is concentrated in DEGENERATE+0%-wrong cells while the VALID "
              "(disambiguation-required) cells show a small/zero/negative delta with "
              "high wrong-instance recall, THEN the +0.2085 is REAL RECALL but NOT "
              "instance disambiguation. Do NOT hard-code a conclusion here; the "
              "Euclidean validity verdict above is PROVISIONAL until --use-pathfinder.")

    # ---- READOUT ----
    print("\n========== READOUT ==========")
    if val["cells"]:
        print("  " + overall_verdict_line(val["cells"]))
        if val["approx"]:
            print("  (verdict is on the Euclidean PROXY — confirm with --use-pathfinder "
                  "before quoting it.)")
    else:
        print("  OVERALL: no labelled warm cells found — cannot issue a validity verdict.")

    # ---- END banner: make the Euclidean fallback impossible to miss ----
    if val["approx"]:
        print()
        print(_EUCLID_BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
