"""
check_multi_instance_feasible — the $0 precondition gate for the multi-instance
revisit harness (#1 Part A).

WHY. The single-goal eval (source == goal == target) gives instance
disambiguation nothing to do, so every disambiguation lever (query-expansion,
trained R/U, coarse, M4, audio-DOA) is inert *by construction*. The fix is a
harness with >=2 REACHABLE same-category instances per episode, instance-keyed
against the cold-sighted one — but that is only buildable if val_mini actually
HAS such instances. This gate scans the HM3D content files and reports, per
(scene, category): how many instances carry view_points, the pairwise centroid
separation, and a verdict — so we learn for $0 whether the harness is feasible
BEFORE any GPU, and which (scene, category) cells to build.

A pair of same-category instances is only useful if it is far enough apart to be
a *distinct* goal: a 0.05 m "co-located" pair (e.g. HM3D plant clusters) cannot
create disambiguation pressure even though the instance count is >= 2. So the
verdict keys on the pairwise CENTROID separation, not the bare count.

Verdicts per (scene, category):
  * FEASIBLE   — >=2 instances with view_points AND >=1 pair separated by
                 >= --min-sep (default 1.5 m). Buildable.
  * CO-LOCATED — >=2 instances with view_points but ALL pairs < --min-sep
                 (e.g. plant clusters); cannot force disambiguation.
  * SINGLE     — <2 instances with view_points (e.g. tv_monitor); no distractor.

CAVEAT (the Euclidean/geodesic boundary). This gate measures EUCLIDEAN centroid
separation with stdlib only, so it runs + unit-tests locally. It does NOT prove
GEODESIC reachability of the 2nd instance from a warm start, nor whether the
distractor is ever geodesically en-route — those need habitat-sim and are the
genuine Part-A->Part-B boundary (run the harness's cheap RACE dry-run for them).
A FEASIBLE verdict here is necessary, not sufficient.

Pure stdlib (gzip + json + math + argparse); reuses check_seed_pose's content
loader. Run::

    python embodied_memory/scripts/check_multi_instance_feasible.py
    python embodied_memory/scripts/check_multi_instance_feasible.py \
        --content-dir data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content \
        --categories "chair bed sofa toilet tv_monitor plant" --min-sep 1.5

Exit 0 = GREEN (>=1 feasible cell), 2 = RED (no feasible cell anywhere).
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_seed_pose import _load_gz, resolve_content_path  # noqa: E402  (stdlib-only module)

# The six HM3D ObjectNav goal categories.
DEFAULT_CATEGORIES = ["chair", "bed", "sofa", "toilet", "tv_monitor", "plant"]
DEFAULT_CONTENT_DIR = "data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content"
DEFAULT_MIN_SEP = 1.5


def load_content(path: str) -> Dict[str, Any]:
    """Read a content .json.gz (stdlib gzip+json via check_seed_pose._load_gz)."""
    return _load_gz(path)


# ----------------------------------------------------------------------
# geometry (stdlib; mirrors make_revisit_smoke / habitat_env, copied to stay
# import-light — habitat_env imports numpy at module top, unusable faiss-free)
# ----------------------------------------------------------------------


def category_instances_vps(content: Dict[str, Any], category: str
                           ) -> List[Tuple[Any, List[List[float]]]]:
    """``[(object_id, [vp_xyz, ...]), ...]`` for instances of ``category`` that
    carry at least one view_point. Suffix-matches ``..._{category}`` keys
    (so multi-token ``tv_monitor`` resolves), mirroring
    ``habitat_env._category_viewpoints_from_content``."""
    suffix = f"_{category}"
    out: List[Tuple[Any, List[List[float]]]] = []
    for key, instances in (content.get("goals_by_category") or {}).items():
        if not key.endswith(suffix):
            continue
        for inst in instances or []:
            vps = []
            for vp in inst.get("view_points") or []:
                pos = (vp.get("agent_state") or {}).get("position")
                if pos:
                    vps.append([float(v) for v in pos])
            if vps:
                out.append((inst.get("object_id"), vps))
    return out


def _centroid(vps: List[List[float]]) -> List[float]:
    n = len(vps)
    dim = len(vps[0])
    return [sum(v[i] for v in vps) / n for i in range(dim)]


def instance_centroids(content: Dict[str, Any], category: str
                       ) -> List[Tuple[Any, List[float]]]:
    """``[(object_id, centroid_xyz), ...]`` (centroid = mean view_point pose)."""
    return [(oid, _centroid(vps)) for oid, vps in category_instances_vps(content, category)]


def _sep_xz(a: List[float], b: List[float]) -> float:
    """Euclidean separation in the navigation (x, z) plane."""
    return math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2]))


def pairwise_seps(centroids: List[Tuple[Any, List[float]]]) -> List[Dict[str, Any]]:
    """All unordered instance pairs with their (x, z) centroid separation."""
    out: List[Dict[str, Any]] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            out.append({"a": centroids[i][0], "b": centroids[j][0],
                        "sep": _sep_xz(centroids[i][1], centroids[j][1])})
    return out


# ----------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------


def cell_verdict(content: Dict[str, Any], category: str,
                 min_sep: float = DEFAULT_MIN_SEP) -> Dict[str, Any]:
    """Per (scene-implied-by-content, category) feasibility verdict."""
    cents = instance_centroids(content, category)
    n = len(cents)
    seps = pairwise_seps(cents)
    sep_vals = [s["sep"] for s in seps]
    max_sep = max(sep_vals) if sep_vals else float("nan")
    min_sep_val = min(sep_vals) if sep_vals else float("nan")
    if n < 2:
        verdict = "SINGLE"
    elif max_sep >= min_sep:
        verdict = "FEASIBLE"
    else:
        verdict = "CO-LOCATED"
    return {"category": category, "n_vp_instances": n, "verdict": verdict,
            "min_sep": min_sep_val, "max_sep": max_sep, "n_pairs_ok": sum(
                1 for s in sep_vals if s >= min_sep), "pairs": seps}


def scan_contents(by_scene: Dict[str, Dict[str, Any]], categories: List[str],
                  min_sep: float = DEFAULT_MIN_SEP) -> Dict[str, Any]:
    """Scan {scene: content} over ``categories`` -> per-cell verdicts +
    a GREEN/RED scope decision + the recommended harness matrix."""
    cells: List[Dict[str, Any]] = []
    for scene, content in by_scene.items():
        for cat in categories:
            v = cell_verdict(content, cat, min_sep)
            v["scene"] = scene
            cells.append(v)
    feasible = [c for c in cells if c["verdict"] == "FEASIBLE"]
    matrix_scenes = sorted({c["scene"] for c in feasible})
    matrix_categories = sorted({c["category"] for c in feasible})
    return {"cells": cells, "feasible": feasible, "green": bool(feasible),
            "matrix_scenes": matrix_scenes, "matrix_categories": matrix_categories,
            "min_sep": min_sep}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _discover_contents(content_dir: Optional[str], scenes: Optional[List[str]],
                       paths: List[str]) -> Dict[str, Dict[str, Any]]:
    """Resolve {scene: content} from explicit paths, or a content dir (+ optional
    scene filter)."""
    files: List[str] = list(paths)
    if content_dir:
        for p in sorted(glob.glob(os.path.join(content_dir, "*.json.gz"))):
            files.append(p)
    by_scene: Dict[str, Dict[str, Any]] = {}
    for p in files:
        scene = os.path.basename(p).replace(".json.gz", "")
        if scenes and scene not in scenes:
            continue
        try:
            by_scene[scene] = load_content(p)
        except (OSError, ValueError):
            print(f"  [warn] could not read {p}", file=sys.stderr)
    return by_scene


def _print_report(report: Dict[str, Any]) -> None:
    print(f"multi-instance feasibility (min-sep {report['min_sep']} m, xz)\n")
    print(f"  {'scene':<16} {'category':<11} {'#inst':>5} {'min_sep':>8} "
          f"{'max_sep':>8}  verdict")
    for c in report["cells"]:
        ms = f"{c['min_sep']:.2f}" if math.isfinite(c['min_sep']) else "  -"
        xs = f"{c['max_sep']:.2f}" if math.isfinite(c['max_sep']) else "  -"
        print(f"  {c['scene']:<16} {c['category']:<11} {c['n_vp_instances']:>5} "
              f"{ms:>8} {xs:>8}  {c['verdict']}")
    print()
    if report["green"]:
        cells = ", ".join(f"{c['scene']}:{c['category']}" for c in report["feasible"])
        print(f"  GREEN — feasible cells: {cells}")
        print(f"  recommended matrix: --scenes \"{' '.join(report['matrix_scenes'])}\" "
              f"--categories \"{' '.join(report['matrix_categories'])}\"")
        print("  NOTE: Euclidean gate only — confirm GEODESIC reachability of the 2nd")
        print("  instance from the warm starts on RACE (habitat-sim) before a paid matrix.")
    else:
        print("  RED — no (scene, category) cell has >=2 separated instances; the")
        print("  multi-instance harness is INFEASIBLE on these content files.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-instance harness feasibility gate")
    ap.add_argument("paths", nargs="*", help="explicit content .json.gz paths (optional)")
    ap.add_argument("--content-dir", default=None,
                    help=f"dir of <scene>.json.gz content files (default {DEFAULT_CONTENT_DIR} "
                         "if no explicit paths given)")
    ap.add_argument("--scenes", default=None, help="space-separated scene filter")
    ap.add_argument("--categories", default=" ".join(DEFAULT_CATEGORIES),
                    help="space-separated categories to scan")
    ap.add_argument("--min-sep", type=float, default=DEFAULT_MIN_SEP,
                    help="min centroid xz separation (m) for a usable instance pair")
    args = ap.parse_args(argv)

    content_dir = args.content_dir
    if not args.paths and not content_dir:
        content_dir = DEFAULT_CONTENT_DIR
    scenes = args.scenes.split() if args.scenes else None
    by_scene = _discover_contents(content_dir, scenes, args.paths)
    if not by_scene:
        print("no content files found (pass paths or --content-dir)", file=sys.stderr)
        return 2
    report = scan_contents(by_scene, args.categories.split(), args.min_sep)
    _print_report(report)
    return 0 if report["green"] else 2


if __name__ == "__main__":
    sys.exit(main())
