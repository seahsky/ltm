"""
Inspect + repair HM3D semantic-annotation file placement so the sim finds them
(sim.semantic_scene.regions populates) — step-4 GT-region grounding prereq.

The minival semantic download succeeded but `diagnose_hm3d_regions` still reports
0 regions and the sim still looks for `val/<scene>/<scene>.basis.scn` (not found).
The geometry (`*.basis.glb`) loads, so `val/<scene>/` exists; the semantic files
(`*.semantic.glb`/`*.semantic.txt`/`*.basis.scn`/`info_semantic.json`) most likely
landed under a DIFFERENT split dir (e.g. `minival/<scene>/`). This is a placement
mismatch, not a missing download.

This tool is filesystem-only (NO sim): it inventories where the geometry vs the
semantic files live per scene, then SYMLINKS the semantic files next to the
geometry the sim reads (for every scene dir that has the `.basis.glb` but is
missing the semantic files). It is idempotent and prints everything it finds/does.
If the semantic files use a name the sim's config doesn't expect (e.g. only
`.semantic.txt` and no `.basis.scn`), it says so — that's a config-path issue, not
a placement one.

Run via the wrapper (also re-runs the region diagnostic):
    bash scripts/fix_hm3d_semantics.sh
or directly:
    python3 embodied_memory/scripts/fix_hm3d_semantics.py [--root data/hm3d] [--apply]
(default is --apply; pass --dry-run to only inspect.)
"""

from __future__ import annotations

import argparse
import glob
import os
from collections import defaultdict
from typing import Dict, List, Set

# Files that constitute a scene's SEMANTIC annotation (everything the geometry
# .basis.glb is NOT). `.basis.scn` is the descriptor the sim's error names.
_SEM_PATTERNS = ["*.semantic.glb", "*.semantic.txt", "*.basis.scn", "*.scn",
                 "info_semantic.json", "*.semantic.json"]


def _scene_name(d: str) -> str:
    """The scene dir basename, e.g. '00800-TEEsavR23oF'."""
    return os.path.basename(os.path.normpath(d))


def _semantic_files(scene_dir: str) -> List[str]:
    out: List[str] = []
    for pat in _SEM_PATTERNS:
        out += glob.glob(os.path.join(scene_dir, pat))
    # de-dup (a file can match >1 pattern) and drop the geometry glb
    return sorted({f for f in out if not f.endswith(".basis.glb")})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Repair HM3D semantic file placement")
    p.add_argument("--root", default="data/hm3d")
    p.add_argument("--dry-run", action="store_true", help="inspect only, make no links")
    args = p.parse_args(argv)
    apply = not args.dry_run

    root = args.root
    if not os.path.isdir(root):
        print(f"ERROR: root not found: {root}")
        return 1

    # ---- inventory ----------------------------------------------------------
    geo_dirs: Dict[str, Set[str]] = defaultdict(set)   # scene -> dirs with *.basis.glb
    sem_dirs: Dict[str, Set[str]] = defaultdict(set)   # scene -> dirs with semantic files
    for glb in glob.glob(os.path.join(root, "**", "*.basis.glb"), recursive=True):
        d = os.path.dirname(glb)
        geo_dirs[_scene_name(d)].add(os.path.realpath(d))
    for pat in _SEM_PATTERNS:
        for f in glob.glob(os.path.join(root, "**", pat), recursive=True):
            if f.endswith(".basis.glb"):
                continue
            d = os.path.dirname(f)
            sem_dirs[_scene_name(d)].add(os.path.realpath(d))

    print(f"=== inventory under {root} ===")
    print(f"  scenes with geometry (.basis.glb): {len(geo_dirs)}")
    print(f"  scenes with semantic files:        {len(sem_dirs)}")
    sem_names: Set[str] = set()
    for scene, dirs in sorted(sem_dirs.items()):
        for d in dirs:
            for f in _semantic_files(d):
                sem_names.add(os.path.basename(f).split(".", 1)[-1])  # ext-ish
    print(f"  distinct semantic file kinds seen: {sorted(sem_names) or '(none)'}")
    cfgs = glob.glob(os.path.join(root, "**", "*scene_dataset_config.json"), recursive=True)
    print(f"  scene_dataset_config(s): {cfgs}")

    if not sem_dirs:
        print("\n  NO semantic files found anywhere under root — the semantic download did not")
        print("  land any *.semantic.*/*.basis.scn. Re-check the download uid/output.")
        return 2

    # ---- repair: co-locate semantics with geometry --------------------------
    linked = 0
    already = 0
    missing_scn = 0
    print("\n=== repair (symlink semantic files next to geometry the sim reads) ===")
    for scene, gdirs in sorted(geo_dirs.items()):
        sdirs = sem_dirs.get(scene)
        if not sdirs:
            continue
        # pick a semantic SOURCE dir for this scene (prefer one with a .basis.scn)
        src = None
        for d in sorted(sdirs):
            if glob.glob(os.path.join(d, "*.basis.scn")):
                src = d
                break
        src = src or sorted(sdirs)[0]
        src_files = _semantic_files(src)
        has_scn = any(f.endswith(".basis.scn") for f in src_files)
        if not has_scn:
            missing_scn += 1
        for gdir in sorted(gdirs):
            if os.path.realpath(gdir) == os.path.realpath(src):
                continue  # geometry and semantics already co-located
            for sf in src_files:
                target = os.path.join(gdir, os.path.basename(sf))
                if os.path.exists(target) or os.path.islink(target):
                    already += 1
                    continue
                if apply:
                    try:
                        os.symlink(os.path.realpath(sf), target)
                    except OSError as e:
                        print(f"  WARN: could not link {target}: {e}")
                        continue
                print(f"  {'linked' if apply else 'would link'}: {target} -> {os.path.realpath(sf)}")
                linked += 1

    print(f"\n=== summary ===")
    print(f"  {'linked' if apply else 'would link'}: {linked}   already present: {already}")
    if missing_scn:
        print(f"  WARNING: {missing_scn} scene(s) have semantic files but NO .basis.scn — the sim's")
        print(f"           config expects '<scene>.basis.scn'. If regions are still 0 after this,")
        print(f"           the fix is a config-path/naming issue (point the scene_dataset_config at")
        print(f"           the .semantic.txt), NOT placement. Paste this output.")
    if linked == 0 and already == 0:
        print("  Nothing to link (geometry and semantics already co-located, or no scene overlap).")
        print("  If regions are still 0, this is a config/naming issue — paste the inventory above.")
    print("\n  Re-run the region diagnostic to confirm (the wrapper does this automatically).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
