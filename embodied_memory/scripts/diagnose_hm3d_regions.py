"""
Cheap, Habitat-only diagnostic of HM3D scene REGION annotations — the prerequisite
for GT-region grounding of the coarse-affordance head (step 4).

coarse-1/coarse-2 showed the coarse head can't fire because caption-based room
perception is too sparse. The chosen fix is to room-tag the agent's position /
frontiers from HM3D's GROUND-TRUTH region annotations (sim.semantic_scene.regions).
But the API + whether these scenes are even region-annotated + the category names
are uncertain and untestable without the sim. This script loads a scene, dumps its
regions DEFENSIVELY (count, category name, AABB), tests region-at-position for the
episode start, and prints the distinct region category names — so the room-type
MAPPING can be written from real data, not guesses. NO 7B backbone (sim only) -> fast.

Run on RACE (after `source scripts/race-setup.sh`):
    python embodied_memory/scripts/diagnose_hm3d_regions.py \
        --scene all --episodes-path data/hm3d/datasets/objectnav/hm3d/v1/val_mini/content/../val_mini.json.gz
or simply, using the default minival discovery:
    python embodied_memory/scripts/diagnose_hm3d_regions.py --scene all
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Optional

# Run as a script (`python embodied_memory/scripts/diagnose_hm3d_regions.py`) ->
# put the repo root on sys.path so `import embodied_memory.*` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _safe(fn, default=None):
    try:
        return fn()
    except Exception as e:  # pragma: no cover - probing an uncertain API
        return f"<err: {e}>" if default is None else default


def _region_category_name(region: Any) -> str:
    """HM3D region category name, trying the known API shapes."""
    cat = getattr(region, "category", None)
    if cat is None:
        return "<no category>"
    # SemanticCategory.name() is the habitat-lab shape; fall back to str/attr.
    name = _safe(lambda: cat.name())
    if isinstance(name, str) and not name.startswith("<err"):
        return name
    return _safe(lambda: str(cat), default="<unprintable>")


def _aabb(region: Any):
    aabb = getattr(region, "aabb", None)
    if aabb is None:
        return None
    center = _safe(lambda: list(aabb.center))
    sizes = _safe(lambda: list(aabb.sizes))
    return {"center": center, "sizes": sizes}


def _dump_regions(sim) -> int:
    ss = getattr(sim, "semantic_scene", None)
    if ss is None:
        print("  NO sim.semantic_scene — this scene has no semantic annotations loaded.")
        return 0
    regions = _safe(lambda: list(ss.regions), default=[])
    if not regions:
        print("  semantic_scene present but EMPTY regions list — no region annotations.")
        return 0
    print(f"  regions: {len(regions)}")
    names = Counter()
    for i, r in enumerate(regions):
        name = _region_category_name(r)
        names[name] += 1
        if i < 25:  # cap the per-region dump
            rid = _safe(lambda: r.id, default="?")
            box = _aabb(r)
            nobj = _safe(lambda: len(list(r.objects)), default="?")
            print(f"    region[{i}] id={rid} category={name!r} n_objects={nobj} aabb={box}")
    print("\n  DISTINCT region category names (-> map these to the 6-class taxonomy):")
    for name, n in names.most_common():
        print(f"    {n:>3}x  {name!r}")
    return len(regions)


def _region_at(sim, position) -> Optional[str]:
    """Which region's AABB contains `position` (best-effort point-in-box)."""
    ss = getattr(sim, "semantic_scene", None)
    if ss is None:
        return None
    for r in _safe(lambda: list(ss.regions), default=[]):
        box = _aabb(r)
        if not box or not box.get("center") or not box.get("sizes"):
            continue
        c, s = box["center"], box["sizes"]
        if all(abs(float(position[k]) - float(c[k])) <= abs(float(s[k])) / 2.0 + 1e-3
               for k in range(3)):
            return _region_category_name(r)
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Dump HM3D region annotations (sim-only, no backbone)")
    p.add_argument("--scene", default="all")
    p.add_argument("--episodes-path", default=None)
    p.add_argument("--scene-dataset-path", default=None)
    p.add_argument("--n-scenes", type=int, default=2, help="how many scenes to probe")
    args = p.parse_args(argv)

    from embodied_memory.run_hm3d_pol import _resolve_scene_list  # reuse scene discovery
    from embodied_memory.habitat_env import HabitatObjectNavSource

    scenes = _resolve_scene_list(args.scene, args.episodes_path)
    print(f"discovered scenes: {scenes}")
    scenes = scenes[: max(1, args.n_scenes)]

    total_regions = 0
    for scene in scenes:
        print(f"\n========== scene {scene} ==========")
        src = HabitatObjectNavSource(
            scene_id=scene,
            scene_dataset_path=args.scene_dataset_path,
            episodes_path=args.episodes_path,
            n_episodes=1,
            target_category=None,
        )
        try:
            step, ep = src.reset(0)   # loads the sim for this scene
            sim = src.get_sim()
            if sim is None:
                print("  get_sim() returned None — cannot read regions.")
                continue
            total_regions += _dump_regions(sim)
            start = _safe(lambda: list(step.agent_state.position))
            if start:
                print(f"\n  episode start position {['%.2f' % x for x in start]} -> "
                      f"region: {_region_at(sim, start)!r}")
        except Exception as e:
            print(f"  FAILED to load/probe scene {scene}: {e!r}")
        finally:
            _safe(lambda: src.close())

    print(f"\n==== SUMMARY: {total_regions} total regions across {len(scenes)} scene(s) ====")
    if total_regions == 0:
        print("  -> NO region annotations available; GT-region grounding is NOT viable on these")
        print("     scenes. Fall back to a room classifier (option B) or accept the finding.")
    else:
        print("  -> region annotations PRESENT; map the distinct category names above to the")
        print("     6-class taxonomy and wire region_at() as the coarse head's room signal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
