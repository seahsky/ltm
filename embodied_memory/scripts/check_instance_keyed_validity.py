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


def _pathfinder_dist_fn(scene_dataset: Optional[str], scene_glob: Optional[str]):
    """RACE-only: a geodesic dist_fn backed by habitat-sim's pathfinder. Imported
    lazily so the default (Euclidean) path stays stdlib + local."""
    import habitat_sim  # noqa: F401 (RACE-only)
    raise NotImplementedError(
        "geodesic --use-pathfinder is RACE-only: load the scene's navmesh "
        "(habitat_sim.PathFinder) and return snap_point+geodesic_distance; wire it "
        "to the scene the content targets. The classification logic above is the "
        "tested, scene-independent part.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Instance-keyed harness metric-validity gate")
    ap.add_argument("content_globs", nargs="+", help="built instance-keyed content .json.gz")
    ap.add_argument("--use-pathfinder", action="store_true",
                    help="RACE-only: use true geodesics (habitat-sim) instead of the "
                         "Euclidean proxy")
    ap.add_argument("--scene-dataset", default=None, help="(with --use-pathfinder) scene cfg")
    args = ap.parse_args(argv)

    if args.use_pathfinder:
        dist_fn = _pathfinder_dist_fn(args.scene_dataset, None)  # raises until wired on RACE
    else:
        dist_fn = euclidean_xz
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
