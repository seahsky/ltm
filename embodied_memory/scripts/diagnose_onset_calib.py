"""
diagnose_onset_calib — recommend the AudioGoal ``onset_rms`` for a target audible
distance. ($0; reads a rendered RIR grid, no Habitat / model / RACE run.)

The first audio-DOA RACE run showed onset DETECTION firing at step 130 (point-blank)
even though the anomaly STARTS at t_anom=30 — because with ``onset_rms=0.05`` the
rendered energy at far cells (grid energy[min≈0.046]) is below threshold, so the
agent only "hears" the alarm once nearly on top of the source. That breaks the
"respond to a DISTANT anomaly" premise and starves the S0/S2 audio-DOA head of
firing decisions.

This pass loads the grid, reproduces the run's anomaly clip via
``audio_task.build_anomaly_clip`` (the deterministic synthetic burst by default,
or ``--anomaly-clip`` — the SAME builder ``habitat_env`` uses, so energy matches),
computes per-cell ``rms(render_at_pose)`` vs distance-to-source, and recommends an
``onset_rms`` that makes the anomaly first audible at ~``--target-dist`` meters.
Emits a parseable ``RECOMMEND_ONSET_RMS=<v>`` line the driver passes to
``--audio-onset-rms``.

Run::
    python embodied_memory/scripts/diagnose_onset_calib.py \
        --grid runs/audiogoal/<scene>_<class>_rir_grid.npz --target-dist 4.0
"""
from __future__ import annotations

import argparse
import math
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def cell_energy_vs_distance(grid, clip_norm) -> List[Tuple[float, float]]:
    """Per-cell ``(distance_to_source_xz, onset_energy)`` where ``onset_energy =
    rms(render_at_pose(cell))`` — EXACTLY what the live onset compares to
    ``onset_rms`` (energy falls off with distance from the source)."""
    from embodied_memory.audio import render_at_pose, rms
    src = np.asarray(grid.source_position, dtype=np.float64).reshape(3)
    out: List[Tuple[float, float]] = []
    for pos in grid.cell_positions:
        d = math.hypot(float(pos[0]) - float(src[0]), float(pos[2]) - float(src[2]))
        e = float(rms(render_at_pose(grid, pos, clip_norm)))
        out.append((d, e))
    return out


def fire_distance(samples: List[Tuple[float, float]], thresh: float) -> float:
    """Largest distance whose energy still clears ``thresh`` — the audible radius
    for that onset threshold (energy decreases with distance)."""
    audible = [d for d, e in samples if e >= thresh]
    return max(audible) if audible else 0.0


def recommend_onset_rms(samples: List[Tuple[float, float]], target_dist: float, *,
                        band: float = 0.75, current_rms: float = 0.05) -> Dict[str, Any]:
    """``onset_rms`` that makes the anomaly first audible at ~``target_dist``: the
    median energy of cells in a band around ``target_dist`` (so closer/louder cells
    fire, farther/quieter cells do not). Falls back to the single nearest cell when
    no cell lands in the band (sparse grids)."""
    if not samples:
        return {"recommended_onset_rms": current_rms, "n_cells": 0, "basis": "no cells",
                "target_dist": target_dist, "current_onset_rms": current_rms,
                "current_fire_dist": 0.0, "fire_dist_at_recommended": 0.0,
                "energy_min": 0.0, "energy_max": 0.0, "dist_min": 0.0, "dist_max": 0.0}
    in_band = sorted(e for d, e in samples if abs(d - target_dist) <= band)
    if in_band:
        rec = float(in_band[len(in_band) // 2])
        basis = f"median of {len(in_band)} cell(s) within {band:.2f}m of {target_dist:.1f}m"
    else:
        nearest = min(samples, key=lambda t: abs(t[0] - target_dist))
        rec = float(nearest[1])
        basis = f"nearest cell at {nearest[0]:.2f}m (no cell within {band:.2f}m of target)"
    return {
        "recommended_onset_rms": rec,
        "target_dist": target_dist,
        "basis": basis,
        "fire_dist_at_recommended": fire_distance(samples, rec),
        "current_onset_rms": current_rms,
        "current_fire_dist": fire_distance(samples, current_rms),
        "energy_min": min(e for _, e in samples),
        "energy_max": max(e for _, e in samples),
        "dist_min": min(d for d, _ in samples),
        "dist_max": max(d for d, _ in samples),
        "n_cells": len(samples),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Recommend AudioGoal onset_rms for a target audible distance.")
    ap.add_argument("--grid", required=True, help="rendered RIR grid .npz")
    ap.add_argument("--anomaly-clip", default=None, help="FSD50K .wav (default: deterministic synthetic burst)")
    ap.add_argument("--target-dist", type=float, default=4.0, help="desired audible radius in metres")
    ap.add_argument("--current-onset-rms", type=float, default=0.05)
    ap.add_argument("--target-norm-rms-db", type=float, default=-20.0)
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    from embodied_memory.audio import RIRGrid
    from embodied_memory.audio_task import build_anomaly_clip

    grid = RIRGrid.from_npz(args.grid)
    clip = build_anomaly_clip(args.anomaly_clip, int(grid.sample_rate), args.target_norm_rms_db)
    samples = cell_energy_vs_distance(grid, clip)
    rec = recommend_onset_rms(samples, args.target_dist, current_rms=args.current_onset_rms)

    clip_desc = f"wav:{args.anomaly_clip}" if args.anomaly_clip else "synthetic-burst"
    print(f"[onset-calib] grid={args.grid} cells={rec['n_cells']} sr={int(grid.sample_rate)} clip={clip_desc}")
    print(f"  energy {rec['energy_min']:.4g} .. {rec['energy_max']:.4g} over distances "
          f"{rec['dist_min']:.2f} .. {rec['dist_max']:.2f} m")
    print(f"  current onset_rms={rec['current_onset_rms']:.4g} -> audible radius "
          f"{rec['current_fire_dist']:.2f} m (this is why onset fired point-blank)")
    print(f"  target audible radius {rec['target_dist']:.1f} m -> onset_rms "
          f"{rec['recommended_onset_rms']:.4g} ({rec['basis']}; "
          f"gives ~{rec['fire_dist_at_recommended']:.2f} m)")
    print(f"RECOMMEND_ONSET_RMS={rec['recommended_onset_rms']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
