"""
check_instance_keyed_validity — the metric-validity gate for the instance-keyed
multi-instance harness (#1 Part B, rank 5). Decides, BEFORE any paid matrix,
whether the built dataset actually FORCES instance disambiguation.

WHY (caveat-B). An instance-keyed episode keys success to the cold-sighted TARGET
instance, so reaching a DISTRACTOR no longer counts. But if, from every warm
start, the TARGET is already the geodesically NEAREST same-category instance, then
a memoryless "go to the nearest" agent succeeds without any recall -- the harness
rewards go-to-nearest, not disambiguation, and reproduces the single-goal null.
Disambiguation is only genuinely tested when, from at least one warm start, a
DISTRACTOR is geodesically nearer than the target (so the agent MUST use its
recalled sighting of the target to override go-to-nearest).

Per-cell verdict:
  * VALID       -- >=1 reachable warm start where a distractor is geodesically
                   nearer than the target (disambiguation is forced).
  * DEGENERATE  -- every reachable warm start has the target nearest (go-to-nearest
                   wins; the harness will not test disambiguation; place warm starts
                   adversarially before spending GPU).
  * UNREACHABLE -- no warm start has a finite start->target geodesic (caveat-A NaN
                   collapse; rebuild with reachability-biased warm starts).

The CLASSIFICATION is pure stdlib and unit-tested locally on injected distances.
The DISTANCES are Euclidean-xz by default (a quick local proxy, clearly caveated)
or true GEODESICS with ``--use-pathfinder`` (RACE-only: needs habitat-sim + the
scene). Run after the instance-keyed build, on the built content::

    # local proxy (Euclidean) -- a fast pre-check on the built dataset
    python embodied_memory/scripts/check_instance_keyed_validity.py runs/<tag>/content/*.json.gz
    # the real gate (RACE, habitat-sim):
    python embodied_memory/scripts/check_instance_keyed_validity.py \
        runs/<tag>/content/*.json.gz --use-pathfinder --scene-dataset <hm3d cfg>

Exit 0 = GREEN (>=1 VALID cell), 2 = RED (none VALID).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_seed_pose import _load_gz  # noqa: E402  (stdlib-only)


# ----------------------------------------------------------------------
# classification (pure stdlib; unit-tested on injected distances)
# ----------------------------------------------------------------------


def classify_warm_start(geo_to_target: Optional[float],
                        geo_to_distractors: List[Optional[float]]) -> str:
    """Per warm start: FORCES-DISAMBIGUATION (a distractor is nearer than the
    target), TARGET-NEAREST (go-to-nearest succeeds), or UNREACHABLE (no finite
    start->target geodesic)."""
    if geo_to_target is None or not math.isfinite(geo_to_target):
        return "UNREACHABLE"
    nearest_d = min((g for g in geo_to_distractors
                     if g is not None and math.isfinite(g)), default=math.inf)
    return "FORCES-DISAMBIGUATION" if nearest_d < geo_to_target else "TARGET-NEAREST"


def classify_cell(warm_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cell verdict from its warm-start records (each
    ``{geo_to_target, geo_to_distractors}``)."""
    tags = [classify_warm_start(r.get("geo_to_target"), r.get("geo_to_distractors") or [])
            for r in warm_records]
    n_reach = sum(1 for t in tags if t != "UNREACHABLE")
    n_force = sum(1 for t in tags if t == "FORCES-DISAMBIGUATION")
    verdict = "UNREACHABLE" if n_reach == 0 else ("VALID" if n_force >= 1 else "DEGENERATE")
    return {"verdict": verdict, "n_warm": len(warm_records), "n_reachable": n_reach,
            "n_forces": n_force, "tags": tags}


def scan_cells(cells: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[str, Any]:
    out = {key: classify_cell(recs) for key, recs in cells.items()}
    return {"cells": out, "green": any(v["verdict"] == "VALID" for v in out.values())}


# ----------------------------------------------------------------------
# distances
# ----------------------------------------------------------------------


def euclidean_xz(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[float]:
    """(x, z) Euclidean distance -- the local proxy (NOT geodesic: ignores walls)."""
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def records_from_content(content: Dict[str, Any],
                         dist_fn: Callable[..., Optional[float]]
                         ) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Build per-(scene, category) warm-start records from a built instance-keyed
    content dict, using ``dist_fn`` for start->instance distances. Reads the
    offline ``info['instance_labels']`` (target_center + distractor_centers)."""
    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for ep in content.get("episodes") or []:
        if "-warm-" not in str(ep.get("episode_id", "")):
            continue
        labels = (ep.get("info") or {}).get("instance_labels")
        if not labels:
            continue
        start = ep.get("start_position")
        target = labels.get("target_center")
        distractors = labels.get("distractor_centers") or []
        rec = {
            "geo_to_target": dist_fn(start, target),
            "geo_to_distractors": [dist_fn(start, d) for d in distractors],
        }
        cells.setdefault((ep.get("scene_id"), ep.get("object_category")), []).append(rec)
    return cells


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


# Geodesic distances (RACE-only: habitat-sim navmesh). The classification +
# record-building above are scene-independent and unit-tested locally; only the
# navmesh load + pathfinder calls need habitat-sim, lazily imported below.

DEFAULT_NAVMESH_ROOTS = ("data/scene_datasets/hm3d", "data/hm3d/scene_datasets/hm3d",
                         "data/scene_datasets")


def _find_navmesh(scene: str, roots) -> Optional[str]:
    """First ``*<scene>*.navmesh`` under any of ``roots`` (recursive)."""
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        hits = sorted(glob.glob(os.path.join(root, "**", f"*{scene}*.navmesh"),
                                recursive=True))
        if hits:
            return hits[0]
    return None


def make_geodesic_dist_fn(pathfinder, shortest_path_cls) -> Callable[..., Optional[float]]:
    """``dist_fn(a, b) -> geodesic metres | None`` backed by a habitat-sim
    PathFinder. Both endpoints are navmesh-snapped; an unreachable pair (no path)
    or a non-finite geodesic returns None (-> classified UNREACHABLE). The
    ``shortest_path_cls`` is injected so this wiring unit-tests without
    habitat-sim."""
    def dist_fn(a, b):
        if a is None or b is None:
            return None
        sp = shortest_path_cls()
        sp.requested_start = pathfinder.snap_point(a)
        sp.requested_end = pathfinder.snap_point(b)
        if not pathfinder.find_path(sp):
            return None
        try:
            d = float(sp.geodesic_distance)
        except (TypeError, ValueError):
            return None
        return d if math.isfinite(d) else None
    return dist_fn


def _load_pathfinder(navmesh_path: str):
    """RACE-only: load a habitat-sim PathFinder from a ``.navmesh`` and return
    ``(pathfinder, ShortestPath class)``. Lazy import keeps the default path
    stdlib + local."""
    import habitat_sim  # noqa: F401 (RACE-only)
    pf = habitat_sim.nav.PathFinder()
    pf.load_nav_mesh(navmesh_path)
    if not pf.is_loaded:
        raise RuntimeError(f"navmesh failed to load: {navmesh_path}")
    return pf, habitat_sim.ShortestPath


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Instance-keyed harness metric-validity gate")
    ap.add_argument("content_globs", nargs="+", help="built instance-keyed content .json.gz")
    ap.add_argument("--use-pathfinder", action="store_true",
                    help="RACE-only: use true geodesics (habitat-sim navmesh) instead of "
                         "the Euclidean proxy")
    ap.add_argument("--navmesh-root", default=None,
                    help="(with --use-pathfinder) dir to search recursively for "
                         "<scene>*.navmesh; defaults to the HM3D scene_datasets dirs")
    args = ap.parse_args(argv)

    roots = [args.navmesh_root] if args.navmesh_root else list(DEFAULT_NAVMESH_ROOTS)
    if not args.use_pathfinder:
        print("NOTE: Euclidean-xz proxy (ignores walls). Re-run with --use-pathfinder on "
              "RACE for the true geodesic gate before any paid matrix.\n")

    cells: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for pat in args.content_globs:
        for path in sorted(glob.glob(pat)):
            try:
                content = _load_gz(path)
            except (OSError, ValueError):
                print(f"  [warn] could not read {path}", file=sys.stderr)
                continue
            if args.use_pathfinder:
                scene = os.path.basename(path).replace(".json.gz", "")
                navmesh = _find_navmesh(scene, roots)
                if navmesh is None:
                    print(f"  [warn] no navmesh for scene {scene!r} under {roots} — skipping "
                          f"(the geodesic gate needs it)", file=sys.stderr)
                    continue
                try:
                    pf, shortest_path_cls = _load_pathfinder(navmesh)
                except Exception as ex:  # noqa: BLE001 (RACE env issues -> skip, don't crash)
                    print(f"  [warn] pathfinder load failed for {scene}: {ex}", file=sys.stderr)
                    continue
                dist_fn: Callable[..., Optional[float]] = make_geodesic_dist_fn(pf, shortest_path_cls)
            else:
                dist_fn = euclidean_xz
            for key, recs in records_from_content(content, dist_fn).items():
                cells.setdefault(key, []).extend(recs)
    if not cells:
        print("no instance-keyed warm episodes found (need built content with instance_labels)")
        return 2

    rep = scan_cells(cells)
    print(f"  {'scene':<16} {'category':<11} {'verdict':<22} reachable/forces/warm")
    for (scene, cat), v in rep["cells"].items():
        print(f"  {str(scene):<16} {str(cat):<11} {v['verdict']:<22} "
              f"{v['n_reachable']}/{v['n_forces']}/{v['n_warm']}")
    print()
    if rep["green"]:
        valid = [f"{s}:{c}" for (s, c), v in rep["cells"].items() if v["verdict"] == "VALID"]
        print(f"  GREEN — disambiguation-forcing cells: {', '.join(valid)}")
        print("  (proceed to the paid instance-keyed matrix on these cells; confirm with "
              "--use-pathfinder geodesics first.)")
    else:
        print("  RED — no cell forces disambiguation (all DEGENERATE/UNREACHABLE). Place warm "
              "starts so a distractor is sometimes nearer/en-route, or the harness reproduces "
              "the single-goal null. Do NOT spend the paid matrix yet.")
    return 0 if rep["green"] else 2


if __name__ == "__main__":
    sys.exit(main())
