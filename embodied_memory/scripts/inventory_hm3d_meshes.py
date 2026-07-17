"""Inventory which scenes of an HM3D ObjectNav split have a usable mesh on disk.

The full ObjectNav `val` split references 20 scenes, but the download script
defaults to `hm3d_minival_full` (10 scenes) and symlinks `val -> minival`, so a
`--split val` run finds meshes for only the ~2 scenes that overlap the minival
mesh set — every other scene crashes at sim init with `ESP_CHECK ... No Stage
Attributes exists`. That killed r1v1 (100/100 episodes on the first missing-mesh
scene → 0 completed → driver FATAL after 3m53s).

This is the $0 guard: for a split, report which scenes have a `<bare>.basis.glb`
and which don't, and emit a comma-joined USABLE_SCENES list the R1 driver can run
instead of `--scene all`. Pure stdlib (glob/os) — no habitat/torch — so it runs
before any GPU spend and is unit-testable.

    python embodied_memory/scripts/inventory_hm3d_meshes.py --split val
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple


def _content_scenes(root: str, split: str) -> List[str]:
    """Bare scene ids from the ObjectNav content dir for ``split``."""
    content_dir = os.path.join(
        root, "hm3d", "datasets", "objectnav", "hm3d", "v1", split, "content")
    if not os.path.isdir(content_dir):
        return []
    return sorted(
        os.path.basename(f)[: -len(".json.gz")]
        for f in glob.glob(os.path.join(content_dir, "*.json.gz"))
    )


def _present_mesh_ids(root: str) -> set:
    """Bare scene ids that have a physical ``<bare>.basis.glb`` anywhere in the
    mesh tree.

    Presence is what matters — not the split subdir the episode JSON names. The
    ObjectNav `val` episodes reference ``val/<scene>/...`` and `val` is a symlink
    to `minival`, so a scene's mesh may physically live under `minival/` yet
    resolve for both `val` and `val_mini`. Scanning every mesh dir (each realpath
    once, so the symlink is not double-counted) gets this right for any split.
    """
    mesh_root = os.path.join(root, "hm3d", "scene_datasets", "hm3d")
    present: set = set()
    seen_dirs: set = set()
    for split_dir in glob.glob(os.path.join(mesh_root, "*")):
        real = os.path.realpath(split_dir)
        if not os.path.isdir(real) or real in seen_dirs:
            continue
        seen_dirs.add(real)
        for glb in glob.glob(os.path.join(real, "*", "*.basis.glb")):
            present.add(os.path.basename(glb)[: -len(".basis.glb")])
    return present


def inventory(root: str, split: str) -> Tuple[List[str], List[str]]:
    """Return ``(usable, missing)`` bare scene ids for ``split``.

    A scene is usable iff its ObjectNav content file AND a `.basis.glb` mesh both
    exist. Missing = has episodes but no mesh (the crash set).
    """
    scenes = _content_scenes(root, split)
    present = _present_mesh_ids(root)
    usable = [s for s in scenes if s in present]
    missing = [s for s in scenes if s not in present]
    return usable, missing


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="val")
    ap.add_argument("--root", default="data", help="data root (default: data)")
    args = ap.parse_args(argv)

    usable, missing = inventory(args.root, args.split)
    total = len(usable) + len(missing)
    print(f"split={args.split} scenes={total} usable={len(usable)} missing={len(missing)}")
    if missing:
        print(f"  MISSING (episodes but no mesh): {','.join(missing)}")
    print(f"USABLE_SCENES={','.join(usable)}")
    if total == 0:
        print("VERDICT: no content scenes found — check --root/--split.")
    elif not missing:
        print(f"VERDICT: all {total} scenes usable.")
    else:
        print(
            f"VERDICT: {len(missing)}/{total} scenes have NO mesh. Full-split runs "
            f"will crash on them. Download the full mesh split "
            f"(HM3D_SCENE_GROUP=hm3d_val_full bash embodied_memory/scripts/download_hm3d.sh, "
            f"after removing the val->minival symlink) or run only USABLE_SCENES."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
