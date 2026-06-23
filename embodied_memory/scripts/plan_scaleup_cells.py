"""
plan_scaleup_cells — expand the AudioGoal revisit ablation from the 2-scene
val_mini smoke to the full ~100-cell (20 HM3D val scenes × the categories each
scene actually contains) matrix.

A *cell* is one (scene, goal_category) pair; the anomaly source is co-located
with that category's goal, so the RIR grid and the retrieval target are both
category-keyed. The anomaly CLASS is decorative for retrieval (onset-trigger
framing — it only decides which sound fires), so we round-robin the available
classes across each scene's categories for trigger diversity. Because only 3
anomaly classes exist but a scene can hold 5–6 categories, classes are REUSED
across categories — which is why the downstream driver must key grids/out-dirs
by category (``--cell-tag``), not by class (two categories sharing a class
would otherwise collide).

Pure planning logic (read each scene's ObjectNav content .json.gz for category
availability, check the mesh is on disk, assign classes deterministically). No
Habitat / sim / model imports — unit-tested in ``test_plan_scaleup_cells.py``.
The ``race-scaleup-matrix.sh`` driver consumes the ``--format lines`` output.

    # full 20-scene plan (categories auto-discovered per scene):
    python embodied_memory/scripts/plan_scaleup_cells.py \
        --content-dir data/hm3d/datasets/objectnav/hm3d/v1/val/content \
        --mesh-root data/hm3d --format lines

    # stage a 5-cell smoke first:
    python embodied_memory/scripts/plan_scaleup_cells.py \
        --content-dir .../val/content --mesh-root data/hm3d --max-cells 5
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# The six HM3D ObjectNav goal categories, in a fixed priority order. The order
# is the assignment order for class round-robin and for --max-cells truncation,
# so it is deterministic and reproducible across runs.
DEFAULT_CATEGORIES = ["chair", "bed", "sofa", "toilet", "tv_monitor", "plant"]
DEFAULT_CLASSES = ["baby_cry", "alarm", "glass_break"]


# ----------------------------------------------------------------------
# Pure logic (no filesystem) — unit-tested.
# ----------------------------------------------------------------------
def present_categories(episode_categories: Sequence[str], targets: Sequence[str]) -> List[str]:
    """Targets that occur in ``episode_categories``, in TARGET order (stable)."""
    have = set(episode_categories)
    return [c for c in targets if c in have]


def assign_classes(categories: Sequence[str], classes: Sequence[str]) -> List[Tuple[str, str]]:
    """Round-robin an anomaly class onto each category (trigger diversity).

    Deterministic: category i gets ``classes[i % len(classes)]``. With 3 classes
    and 5 categories the classes reuse as [c0,c1,c2,c0,c1] — which is fine because
    retrieval keys on the category, not the class. Returns (category, class) pairs.
    """
    if not classes:
        raise ValueError("classes must be a non-empty list")
    return [(cat, classes[i % len(classes)]) for i, cat in enumerate(categories)]


# ----------------------------------------------------------------------
# Filesystem-backed helpers.
# ----------------------------------------------------------------------
def discover_scenes(content_dir: str) -> List[str]:
    """Sorted short scene ids discovered from ``<content_dir>/<scene>.json.gz``."""
    return sorted(
        f[: -len(".json.gz")]
        for f in os.listdir(content_dir)
        if f.endswith(".json.gz")
    )


def scene_categories_from_content(content_path: str, targets: Sequence[str]) -> List[str]:
    """Present target categories for one scene's ObjectNav content file."""
    with gzip.open(content_path, "rt") as f:
        data = json.load(f)
    cats = {e["object_category"] for e in data.get("episodes", [])}
    return present_categories(cats, targets)


def mesh_present(scene: str, mesh_root: str) -> bool:
    """True iff a non-semantic ``.basis.glb`` mesh for ``scene`` exists on disk.

    Mirrors the driver's own discovery (``find data/hm3d -name <scene>.basis.glb``)
    so the plan never schedules a cell whose scene can't load.
    """
    hits = glob.glob(os.path.join(mesh_root, "**", f"{scene}.basis.glb"), recursive=True)
    if hits:
        return True
    alt = [
        p
        for p in glob.glob(os.path.join(mesh_root, "**", f"*{scene}*.glb"), recursive=True)
        if "semantic" not in os.path.basename(p)
    ]
    return bool(alt)


def plan_cells(
    content_dir: str,
    *,
    targets: Sequence[str] = DEFAULT_CATEGORIES,
    classes: Sequence[str] = DEFAULT_CLASSES,
    scenes: Optional[Sequence[str]] = None,
    mesh_root: Optional[str] = None,
    max_cells: Optional[int] = None,
) -> Dict:
    """Build the (scene, category, anomaly_class) cell plan.

    ``scenes`` restricts to a subset (else auto-discovered from ``content_dir``).
    ``mesh_root`` (if given) drops scenes whose mesh is absent, recording them in
    ``skipped_no_mesh``. ``max_cells`` truncates the flat cell list (for staging).
    """
    if scenes is None:
        scenes = discover_scenes(content_dir)
    cells: List[Dict[str, str]] = []
    skipped_no_mesh: List[str] = []
    skipped_no_content: List[str] = []
    skipped_no_category: List[str] = []
    skipped_unreadable: List[str] = []
    for scene in scenes:
        cpath = os.path.join(content_dir, f"{scene}.json.gz")
        if not os.path.isfile(cpath):
            skipped_no_content.append(scene)
            continue
        if mesh_root is not None and not mesh_present(scene, mesh_root):
            skipped_no_mesh.append(scene)
            continue
        try:
            cats = scene_categories_from_content(cpath, targets)
        except (OSError, ValueError, EOFError):
            # A single corrupted/truncated content .json.gz must not abort the whole
            # 20-scene plan — drop that scene and record it.
            skipped_unreadable.append(scene)
            continue
        if not cats:
            skipped_no_category.append(scene)
            continue
        for cat, cls in assign_classes(cats, classes):
            cells.append({"scene": scene, "category": cat, "anomaly_class": cls})
    if max_cells is not None:
        cells = cells[:max_cells]
    return {
        "cells": cells,
        "n_cells": len(cells),
        "n_scenes": len({c["scene"] for c in cells}),
        "skipped_no_mesh": skipped_no_mesh,
        "skipped_no_content": skipped_no_content,
        "skipped_no_category": skipped_no_category,
        "skipped_unreadable": skipped_unreadable,
    }


def format_lines(plan: Dict) -> str:
    """One ``scene<TAB>category<TAB>anomaly_class`` row per cell (for bash read)."""
    return "\n".join(
        f"{c['scene']}\t{c['category']}\t{c['anomaly_class']}" for c in plan["cells"]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--content-dir", required=True, help="dir of <scene>.json.gz ObjectNav content files")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    ap.add_argument("--scenes", nargs="+", default=None, help="subset of scenes (default: all in content-dir)")
    ap.add_argument("--mesh-root", default=None, help="if set, drop scenes whose mesh is absent under this root")
    ap.add_argument("--max-cells", type=int, default=None, help="truncate the flat cell list (staging)")
    ap.add_argument("--format", choices=["json", "lines"], default="json")
    args = ap.parse_args(argv)

    plan = plan_cells(
        args.content_dir,
        targets=args.categories,
        classes=args.classes,
        scenes=args.scenes,
        mesh_root=args.mesh_root,
        max_cells=args.max_cells,
    )
    if args.format == "lines":
        print(format_lines(plan))
    else:
        json.dump(plan, sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
