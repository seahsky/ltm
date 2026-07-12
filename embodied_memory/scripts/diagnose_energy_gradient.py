"""
G0.4 — realizable-localization kill-switch (ADR-0001).

Energy-gradient climbing (the realizable anomaly-source localization arm) reaches
the source only if a greedy ascent of live binaural loudness climbs toward it
instead of stalling on a spurious loud spot. This diagnostic measures, from a
rendered RIR grid, whether that field is climbable:

  - spearman(cell energy, distance-to-source) — strongly NEGATIVE means closer
    cells are reliably louder (the M0 grids measured ~-0.45).
  - the count of local ENERGY MAXIMA excluding the global (source) peak — the
    false summits a greedy ascent falls into.

``gradient_climbability`` and ``energy_gradient_verdict`` are pure (TDD); ``main``
reads a rendered grid, derives per-cell energy from the impulse responses, and
prints ``GATE_RESULT=`` — that render/read path is RACE integration.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, Optional, Sequence

import numpy as np

# Run as a script -> put the repo root on sys.path so `import embodied_memory.*` works.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _rankdata(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (numpy-only, no scipy)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0  # mean of the tied 1-based ranks
        i = j + 1
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra = _rankdata(a); rb = _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    if denom <= 0.0:
        return 0.0
    return float((ra * rb).sum() / denom)


def gradient_climbability(energies, positions, source, k: int = 4) -> Dict[str, Any]:
    """Climbability stats for an energy field over navmesh cells.

    ``energies`` per-cell loudness, ``positions`` (N,3) cell xyz, ``source`` the
    anomaly source xyz. Distances and neighbours use the (x,z) floor plane (every
    consumer of a waypoint uses only x,z). Returns the spearman of energy vs
    distance-to-source and the count of spurious local energy maxima (a greedy
    ascent's traps), excluding the global source peak.
    """
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    n = energies.size
    positions = np.asarray(positions, dtype=np.float64).reshape(n, -1)
    source = np.asarray(source, dtype=np.float64).reshape(-1)
    pxz = positions[:, [0, 2]]
    sxz = source[[0, 2]]
    dist = np.linalg.norm(pxz - sxz, axis=1)
    sp = _spearman(energies, dist)

    kk = min(k, n - 1) if n > 1 else 0
    g_peak = int(np.argmax(energies)) if n else -1
    n_local_maxima = 0
    for i in range(n):
        if i == g_peak:
            continue  # the global (source) peak is the destination, not a trap
        d = np.linalg.norm(pxz - pxz[i], axis=1)
        d[i] = np.inf
        nn = np.argsort(d)[:kk]
        if kk > 0 and bool(np.all(energies[i] > energies[nn])):
            n_local_maxima += 1

    return {
        "n_cells": int(n),
        "spearman": float(sp),
        "n_local_maxima": int(n_local_maxima),
        "frac_local_maxima": float(n_local_maxima / n) if n else 0.0,
    }


def energy_gradient_verdict(
    stats: Dict[str, Any],
    *,
    go_spearman: float = -0.4,
    stop_spearman: float = -0.2,
    go_trap_frac: float = 0.05,
    stop_trap_frac: float = 0.20,
) -> str:
    """GO / BORDERLINE / STOP for realizable localization.

    GO — the gradient is strongly negative (closer is reliably louder) AND
    nearly trap-free, so a greedy energy climb reaches the source. STOP — the
    gradient is too weak to localize OR there are too many false summits; fall
    back to the oracle source (ADR-0001). BORDERLINE in between.
    """
    sp = float(stats["spearman"])
    tf = float(stats["frac_local_maxima"])
    if sp <= go_spearman and tf <= go_trap_frac:
        return "GO"
    if sp > stop_spearman or tf > stop_trap_frac:
        return "STOP"
    return "BORDERLINE"


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="G0.4 — is a rendered RIR grid's energy field climbable to the source?")
    p.add_argument("--grid", required=True, help="path to a rendered RIR grid .npz (RIRGrid.load)")
    p.add_argument("-k", type=int, default=4, help="neighbours for local-maxima detection")
    args = p.parse_args(argv)

    from embodied_memory.audio import RIRGrid

    grid = RIRGrid.load(args.grid)
    stats = gradient_climbability(
        grid.cell_energies, grid.cell_positions, grid.source_position, k=args.k)
    verdict = energy_gradient_verdict(stats)
    print(f"grid: {args.grid}  cells={stats['n_cells']}")
    print(f"  spearman(energy, dist_to_source) = {stats['spearman']:.3f}  (want <= -0.4)")
    print(f"  spurious local maxima (traps)    = {stats['n_local_maxima']} "
          f"({stats['frac_local_maxima']:.1%} of cells)")
    print(f"GATE_RESULT={verdict} spearman={stats['spearman']:.4f} "
          f"n_local_maxima={stats['n_local_maxima']} "
          f"frac_local_maxima={stats['frac_local_maxima']:.4f} n_cells={stats['n_cells']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
