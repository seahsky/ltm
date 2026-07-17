"""
TDD for the HM3D mesh inventory guard (inventory_hm3d_meshes).

r1v1 crashed 100/100 episodes because `--split val` references 20 scenes but the
val->minival symlink only exposes the ~2 overlapping meshes; every missing-mesh
episode dies at sim init. The inventory reports which split scenes have a
`.basis.glb` so the R1 driver fails fast with the fix instead of burning GPU on
per-episode crashes.

Key subtlety: meshes live under `scene_datasets/hm3d/minival` (and a `val`
symlink to it), NOT under a per-split dir — so presence must be tested across the
whole mesh tree, following the symlink once.

Run: PYTHONPATH=. python embodied_memory/scripts/test_inventory_hm3d_meshes.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embodied_memory.scripts.inventory_hm3d_meshes import inventory  # noqa: E402


def _tree(root: str, content_scenes, mesh_scenes, split="val", link_val=True):
    """Build a minimal HM3D layout: content .json.gz per scene; meshes under
    `minival/<NN-bare>/<bare>.basis.glb`; optional `val -> minival` symlink."""
    content_dir = os.path.join(
        root, "hm3d", "datasets", "objectnav", "hm3d", "v1", split, "content")
    os.makedirs(content_dir)
    for s in content_scenes:
        open(os.path.join(content_dir, f"{s}.json.gz"), "w").close()
    mesh_root = os.path.join(root, "hm3d", "scene_datasets", "hm3d")
    minival = os.path.join(mesh_root, "minival")
    for i, s in enumerate(mesh_scenes):
        d = os.path.join(minival, f"{i:05d}-{s}")
        os.makedirs(d)
        open(os.path.join(d, f"{s}.basis.glb"), "w").close()
    if link_val:
        os.symlink("minival", os.path.join(mesh_root, "val"))


def case_missing_meshes_reported():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, content_scenes=["A", "B", "C"], mesh_scenes=["A", "B"])
        usable, missing = inventory(root, "val")
        assert usable == ["A", "B"], usable
        assert missing == ["C"], missing
    print("  case_missing_meshes_reported: OK")


def case_all_present():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, content_scenes=["A", "B"], mesh_scenes=["A", "B", "Z"])
        usable, missing = inventory(root, "val")
        assert usable == ["A", "B"] and missing == [], (usable, missing)
    print("  case_all_present: OK")


def case_val_mini_resolves_via_minival_not_a_val_mini_dir():
    # The bug that first showed 0 usable: val_mini episodes but meshes only under
    # minival/. Presence-across-tree must still find them.
    with tempfile.TemporaryDirectory() as root:
        _tree(root, content_scenes=["A", "B"], mesh_scenes=["A", "B"], split="val_mini")
        usable, missing = inventory(root, "val_mini")
        assert usable == ["A", "B"] and missing == [], (usable, missing)
    print("  case_val_mini_resolves_via_minival_not_a_val_mini_dir: OK")


def case_symlink_not_double_counted():
    # val -> minival must not inflate or crash; A appears under both paths.
    with tempfile.TemporaryDirectory() as root:
        _tree(root, content_scenes=["A"], mesh_scenes=["A"])
        usable, missing = inventory(root, "val")
        assert usable == ["A"] and missing == [], (usable, missing)
    print("  case_symlink_not_double_counted: OK")


def case_no_content_is_empty():
    with tempfile.TemporaryDirectory() as root:
        usable, missing = inventory(root, "val")
        assert usable == [] and missing == [], (usable, missing)
    print("  case_no_content_is_empty: OK")


def main() -> int:
    print("running inventory_hm3d_meshes tests…")
    case_missing_meshes_reported()
    case_all_present()
    case_val_mini_resolves_via_minival_not_a_val_mini_dir()
    case_symlink_not_double_counted()
    case_no_content_is_empty()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
