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

    ``dist_fns_by_scene`` (only with ``use_pathfinder``) maps a scene_id to a
    geodesic ``dist_fn(a, b)``; a scene without an entry falls back to the
    Euclidean proxy (so a single missing navmesh degrades that scene, not the
    run). Without ``use_pathfinder`` everything is Euclidean.

    Returns the ``scan_cells`` shape (``{"cells": {(scene,cat): verdict_dict},
    "green": bool}``) with an added ``"approx"`` flag (True if ANY cell used the
    Euclidean proxy).
    """
    dist_fns_by_scene = dist_fns_by_scene or {}
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    approx = False
    for content in contents:
        # group this content's warm episodes by their scene's distance fn.
        # records_from_content keys on (scene_id, object_category); but the
        # dist_fn must be chosen per scene, so partition by scene first.
        by_scene: Dict[str, Dict[str, Any]] = {}
        for ep in content.get("episodes") or []:
            sid = ep.get("scene_id")
            by_scene.setdefault(sid, {"episodes": []})["episodes"].append(ep)
        for sid, sub in by_scene.items():
            geo = dist_fns_by_scene.get(sid) if use_pathfinder else None
            if geo is None:
                approx = approx or use_pathfinder  # asked for geodesic, got proxy
                dist_fn = ck.euclidean_xz
            else:
                dist_fn = geo
            for key, recs in ck.records_from_content(sub, dist_fn).items():
                cells.setdefault(key, []).extend(recs)
    rep = ck.scan_cells(cells)
    rep["approx"] = (not use_pathfinder) or approx
    return rep


def _build_geodesic_dist_fns(content_paths: List[str], roots) -> Dict[str, Any]:
    """RACE-only: build a {scene_id: geodesic dist_fn} from per-scene navmeshes.

    Mirrors ``check_instance_keyed_validity``'s scene-from-filename + navmesh
    discovery + lazy habitat-sim load. A scene whose navmesh is missing or fails
    to load is simply absent from the returned map (=> that scene degrades to the
    Euclidean proxy in ``per_cell_validity_from_contents``). MOCKED in tests.
    """
    out: Dict[str, Any] = {}
    for path in content_paths:
        scene = os.path.basename(path).replace(".json.gz", "")
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
    if use_pathfinder:
        roots = [navmesh_root] if navmesh_root else list(ck.DEFAULT_NAVMESH_ROOTS)
        dist_fns_by_scene = _build_geodesic_dist_fns(loaded_paths, roots)
    return per_cell_validity_from_contents(
        contents, use_pathfinder=use_pathfinder, dist_fns_by_scene=dist_fns_by_scene)


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
    if val["approx"]:
        print("  !!! APPROX: Euclidean-xz proxy (ignores walls). Re-run with "
              "--use-pathfinder on RACE for the TRUE geodesic gate. A wall can flip "
              "a Euclidean-VALID cell to DEGENERATE/UNREACHABLE. !!!")
    else:
        print("  (geodesic via habitat-sim navmesh — the true gate)")
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
    if not episodes:
        print(f"  no episode_*.json under {args.run_dirs}")
    else:
        per_cell, agg = wrong_instance_by_cell(episodes)
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

    # ---- READOUT ----
    print("\n========== READOUT ==========")
    if val["cells"]:
        print("  " + overall_verdict_line(val["cells"]))
        if val["approx"]:
            print("  (verdict is on the Euclidean PROXY — confirm with --use-pathfinder "
                  "before quoting it.)")
    else:
        print("  OVERALL: no labelled warm cells found — cannot issue a validity verdict.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
