"""
diagnose_anomaly_feasibility — the $0 render-time gate for the N3 anomaly-response
eval. Run AFTER building the dataset (decoupled source) and rendering the RIR grid
at that source, BEFORE any paid LLM run.

The builder decouples the anomaly source from the primary goal purely by data (it
CANNOT compute point-to-point geodesics — the two-env split), so whether the
resulting geometry is actually usable — does an AUDIBLE-NOT-LOUD warm/search start
exist near the decoupled source — can only be checked once the grid is rendered.
This gate answers exactly that, per (scene, category) cell, with a GO/SKIP verdict:

  * LOUD start (grid-relative cell energy near the max) → the warm start sits on
    top of the source → the loud diotic bed FALSE-FIRES onset at step 0 (the exact
    defect N3 removes). SKIP.
  * QUIET / OUT-OF-COVERAGE start → the source is inaudible from the search region
    → onset never fires → the interrupt→investigate→resume loop is never exercised.
    SKIP.
  * AUDIBLE (mid-band) start exists → the source is heard from a valid, reachable-
    to-goal search start without deafening it. GO.

Audibility is measured grid-relative (nearest-cell IR energy / the grid's max),
which captures occlusion AND distance from the rendered source (better than a raw
xz distance). Pure numpy on the built dataset + the rendered grid — no GPU/sim.

    python embodied_memory/scripts/diagnose_anomaly_feasibility.py \
        --dataset data/hm3d/.../anomaly_response_<tag>/content/<scene>.json.gz \
        --grid runs/audiogoal/<scene>_<cell>_rir_grid.npz
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# Defaults: the audible band is grid-relative (fraction of the grid's MAX cell
# energy). loud_frac 0.5 = "within ~half the loudest cell's energy = near the
# source = false-fire risk"; audible_frac 0.02 = "at least 2% of max = the source
# is heard". coverage_m 2.0 = the start's nearest grid cell must be within one
# grid-spacing (else the grid doesn't cover the start region).
_AUDIBLE_FRAC_DEFAULT = 0.02
_LOUD_FRAC_DEFAULT = 0.5
_COVERAGE_M_DEFAULT = 2.0


def default_coverage_m(cell_positions, *, factor: float = 1.5, floor_m: float = 2.0) -> float:
    """Coverage radius that TRACKS the grid's actual cell spacing, so a start's
    nearest cell is 'in coverage' iff it is within ~1.5x the typical inter-cell
    gap. A fixed radius false-rejects a SPARSE grid (no cell within 2 m of a
    genuinely-audible start) and is over-lenient on a DENSE one. Returns
    ``factor * median(nearest-neighbour xz spacing)``, floored at ``floor_m``.

    The ``floor_m=2.0`` (room-scale) matters: the LIVE runtime snaps a start to
    its nearest cell REGARDLESS of distance (``RIRGrid.nearest`` has no max), so a
    start a metre-plus from its nearest cell still gets that cell's convolved audio
    and WILL fire onset at runtime — classifying it OUT_OF_COVERAGE is a false
    negative that SKIPs a usable cell (the wcojb bed->toilet case: a warm start at
    d2cell 1.39 m with rel_energy 0.14 was wrongly rejected under a 1.0 m floor).
    The floor only guards genuinely off-grid starts (different floor / disconnected
    region), where the snapped IR is meaningless."""
    import numpy as np
    p = np.asarray(cell_positions, dtype=np.float64).reshape(-1, 3)
    n = p.shape[0]
    if n < 2:
        return float(floor_m)
    xz = p[:, [0, 2]]
    nn = np.empty(n, dtype=np.float64)
    for i in range(n):
        d = np.linalg.norm(xz - xz[i], axis=1)
        d[i] = np.inf
        nn[i] = d.min()
    return float(max(floor_m, factor * float(np.median(nn))))


def classify_start(rel_energy: float, dist_to_cell_m: float, *,
                   audible_frac: float = _AUDIBLE_FRAC_DEFAULT,
                   loud_frac: float = _LOUD_FRAC_DEFAULT,
                   coverage_m: float = _COVERAGE_M_DEFAULT) -> str:
    """Classify one warm/search start by its audibility to the rendered source.

    ``rel_energy`` = the start's nearest-cell IR energy / the grid's MAX cell
    energy (∈ [0, 1]). Returns ``OUT_OF_COVERAGE`` | ``QUIET`` | ``AUDIBLE`` |
    ``LOUD``."""
    if dist_to_cell_m > coverage_m:
        return "OUT_OF_COVERAGE"
    if rel_energy < audible_frac:
        return "QUIET"
    if rel_energy > loud_frac:
        return "LOUD"
    return "AUDIBLE"


def cell_verdict(start_classes: List[str]) -> Tuple[str, str]:
    """GO iff at least one AUDIBLE (audible-not-loud) start exists; else SKIP with
    the dominant reason. Pure."""
    if not start_classes:
        return ("SKIP", "no warm/search start in this cell")
    if any(c == "AUDIBLE" for c in start_classes):
        return ("GO", "an audible-not-loud search start exists")
    if all(c == "LOUD" for c in start_classes):
        return ("SKIP", "all starts LOUD (source co-located with the search start → "
                        "loud-bed step-0 false-fire) — decouple the source farther / "
                        "move the warm starts away")
    if all(c in ("QUIET", "OUT_OF_COVERAGE") for c in start_classes):
        return ("SKIP", "all starts inaudible (source not heard from the search region "
                        "→ onset never fires) — move the source nearer the search path / "
                        "raise the audible radius")
    return ("SKIP", "no audible-not-loud start (starts are a mix of LOUD and inaudible) "
                    "— the source is either on top of or out of range of every start")


def _load_gz(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def _scene_label(scene_id: Optional[str]) -> Optional[str]:
    if not scene_id:
        return None
    return os.path.basename(str(scene_id)).split(".", 1)[0]


def feasibility_from_grid(content: Dict[str, Any], grid, *,
                          audible_frac: float = _AUDIBLE_FRAC_DEFAULT,
                          loud_frac: float = _LOUD_FRAC_DEFAULT,
                          coverage_m: float = _COVERAGE_M_DEFAULT) -> Dict[Tuple, Dict[str, Any]]:
    """Group the WARM/response episodes by (scene, object_category) and classify
    each start against ``grid`` (an ``audio.RIRGrid``). Returns
    ``{(scene, category): {verdict, reason, per_start:[{episode_id, rel_energy,
    dist_to_cell_m, klass}]}}``. Only ``-warm-`` episodes are classified (the
    cold/seed episode starts at the goal view_point and is silent)."""
    energies = grid.cell_energies
    max_e = float(energies.max()) if len(energies) else 0.0
    groups: Dict[Tuple, List[Dict[str, Any]]] = {}
    for ep in content.get("episodes") or []:
        eid = str(ep.get("episode_id", ""))
        if "-warm-" not in eid:
            continue
        key = (_scene_label(ep.get("scene_id")), ep.get("object_category"))
        _, idx, dist = grid.nearest(ep.get("start_position"))
        rel = (float(energies[idx]) / max_e) if max_e > 0 else 0.0
        klass = classify_start(rel, dist, audible_frac=audible_frac,
                               loud_frac=loud_frac, coverage_m=coverage_m)
        groups.setdefault(key, []).append(
            {"episode_id": eid, "rel_energy": round(rel, 4),
             "dist_to_cell_m": round(float(dist), 3), "klass": klass})
    out: Dict[Tuple, Dict[str, Any]] = {}
    for key, per in groups.items():
        verdict, reason = cell_verdict([p["klass"] for p in per])
        out[key] = {"verdict": verdict, "reason": reason, "per_start": per}
    return out


def _load_rir_grid(path: str):
    """Load an ``audio.RIRGrid`` WITHOUT importing embodied_memory/__init__ (faiss).
    Mirrors make_audiogoal_smoke._load_rir_grid."""
    import importlib.util
    audio_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "audio.py"))
    spec = importlib.util.spec_from_file_location("_feas_audio", audio_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod   # register before exec so audio.AugmentSpec (@dataclass) resolves its fields
    spec.loader.exec_module(mod)
    return mod.RIRGrid.load(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="N3 render-time feasibility gate")
    ap.add_argument("--dataset", required=True,
                    help="built content .json.gz (or a glob of content/*.json.gz)")
    ap.add_argument("--grid", required=True, help="rendered RIR grid .npz")
    ap.add_argument("--audible-frac", type=float, default=_AUDIBLE_FRAC_DEFAULT)
    ap.add_argument("--loud-frac", type=float, default=_LOUD_FRAC_DEFAULT)
    ap.add_argument("--coverage-m", type=float, default=None,
                    help="in-coverage radius (m); default = derived from the grid's "
                         "actual cell spacing (default_coverage_m).")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(args.dataset)) if any(c in args.dataset for c in "*?[") \
        else [args.dataset]
    if not paths:
        print(f"FATAL: no dataset at {args.dataset}")
        return 2
    grid = _load_rir_grid(args.grid)
    coverage_m = args.coverage_m if args.coverage_m is not None \
        else default_coverage_m(grid.cell_positions)
    grid_src = [round(float(v), 3) for v in grid.source_position]
    print(f"[feasibility] grid={os.path.basename(args.grid)} cells={len(grid)} "
          f"source={grid_src} max_energy={float(grid.cell_energies.max()):.4g}")
    print(f"[feasibility] band: audible_frac={args.audible_frac} loud_frac={args.loud_frac} "
          f"coverage_m={coverage_m:.2f}{'' if args.coverage_m is not None else ' (derived from grid spacing)'}")

    any_go = False
    all_go = True
    for p in paths:
        content = _load_gz(p)
        # One grid per source: warn if this content spans multiple scenes (each has
        # its OWN decoupled source/grid — a single --grid can't adjudicate them all).
        scenes = {_scene_label(e.get("scene_id")) for e in (content.get("episodes") or [])}
        if len(scenes) > 1:
            print(f"  WARN: {os.path.basename(p)} spans {len(scenes)} scenes {sorted(scenes)} "
                  "but only one --grid was given — verdicts for non-rendered sources are unreliable.")
        res = feasibility_from_grid(content, grid, audible_frac=args.audible_frac,
                                    loud_frac=args.loud_frac, coverage_m=coverage_m)
        for (scene, cat), r in sorted(res.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            print(f"\n  [{scene}/{cat}] FEASIBILITY={r['verdict']}  {r['reason']}")
            for ps in r["per_start"]:
                print(f"      {ps['episode_id']:28} rel_energy={ps['rel_energy']:.4f} "
                      f"d2cell={ps['dist_to_cell_m']:.2f}m -> {ps['klass']}")
            if r["verdict"] == "GO":
                any_go = True
            else:
                all_go = False

    verdict = "GO" if all_go and any_go else ("PARTIAL" if any_go else "SKIP")
    print(f"\nFEASIBILITY_RESULT={verdict}  "
          f"({'all cells GO' if verdict == 'GO' else 'some/all cells SKIP — see above'})")
    # exit 0 on GO/PARTIAL (the driver decides per-cell); nonzero only on a total SKIP
    # so a single-cell driver can gate on the exit code.
    return 0 if any_go else 1


if __name__ == "__main__":
    sys.exit(main())
