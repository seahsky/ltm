"""
TDD test for the pure cell-selection logic in ``render_rir_grid.py`` (M0a).

The renderer's habitat-sim / SoundSpaces parts run only in the
``soundspaces-spike`` env on RACE; the listener-cell SELECTION is pure
geometry and is unit-tested here. ``select_cells`` takes candidate navigable
points + their geodesic distance to the source and returns the indices of a
well-spread, in-range subset (so the rendered grid covers the scene without
clustering, and every cell is connected to the source).

render_rir_grid.py is loaded standalone (habitat_sim / audio are imported
lazily inside the render path, not at module top) so this test needs neither.

    python embodied_memory/scripts/test_render_rir_grid.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

_EMB_DIR = Path(__file__).resolve().parent.parent


def _load_file_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


rrg = _load_file_as("embodied_memory._render_rir_grid_under_test",
                    _EMB_DIR / "scripts" / "render_rir_grid.py")


def case_filters_out_of_range_and_unreachable():
    pts = np.array([
        [0.0, 1.5, 0.0],   # geo 0.3 < min_dist → drop (too close / on source)
        [2.0, 1.5, 0.0],   # geo 2.0 → keep
        [5.0, 1.5, 0.0],   # geo 20.0 > max_dist → drop
        [3.0, 1.5, 0.0],   # geo inf (disconnected island) → drop
    ], dtype=np.float32)
    geo = np.array([0.3, 2.0, 20.0, math.inf], dtype=np.float32)
    idx = rrg.select_cells(pts, geo, max_cells=10, min_dist_m=1.0,
                           max_dist_m=15.0, min_spacing_m=0.5)
    assert idx == [1], f"expected only the in-range reachable cell, got {idx}"
    print("  case filters_out_of_range_and_unreachable: OK")


def case_dedups_clustered_points():
    # five points within 0.1 m of each other → min_spacing 0.5 keeps exactly one
    base = np.array([4.0, 1.5, 4.0], dtype=np.float32)
    pts = np.stack([base + np.array([0.02 * i, 0.0, 0.0], np.float32)
                    for i in range(5)])
    geo = np.full(5, 3.0, dtype=np.float32)
    idx = rrg.select_cells(pts, geo, max_cells=10, min_dist_m=0.5,
                           max_dist_m=20.0, min_spacing_m=0.5)
    assert len(idx) == 1, f"clustered points should collapse to 1, got {idx}"
    print("  case dedups_clustered_points: OK")


def case_keeps_spread_points():
    pts = np.array([[float(i), 1.5, 0.0] for i in range(6)], dtype=np.float32)  # 1 m apart
    geo = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    idx = rrg.select_cells(pts, geo, max_cells=10, min_dist_m=0.5,
                           max_dist_m=20.0, min_spacing_m=0.5)
    assert len(idx) == 6, f"all 1m-spaced points should survive 0.5m spacing, got {idx}"
    # selected positions must actually respect the spacing
    sel = pts[idx][:, [0, 2]]
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            assert np.linalg.norm(sel[i] - sel[j]) >= 0.5 - 1e-6
    print("  case keeps_spread_points: OK")


def case_caps_at_max_cells():
    pts = np.array([[float(i), 1.5, 0.0] for i in range(20)], dtype=np.float32)
    geo = np.arange(1, 21, dtype=np.float32)
    idx = rrg.select_cells(pts, geo, max_cells=5, min_dist_m=0.5,
                           max_dist_m=50.0, min_spacing_m=0.5)
    assert len(idx) == 5, f"expected cap of 5, got {len(idx)}"
    assert all(0 <= i < 20 for i in idx)
    print("  case caps_at_max_cells: OK")


def case_returns_empty_when_none_in_range():
    pts = np.array([[0.0, 1.5, 0.0]], dtype=np.float32)
    geo = np.array([math.inf], dtype=np.float32)
    idx = rrg.select_cells(pts, geo, max_cells=5, min_dist_m=1.0,
                           max_dist_m=15.0, min_spacing_m=0.5)
    assert idx == [], f"no reachable cell → empty, got {idx}"
    print("  case returns_empty_when_none_in_range: OK")


def main() -> int:
    cases = [
        case_filters_out_of_range_and_unreachable,
        case_dedups_clustered_points,
        case_keeps_spread_points,
        case_caps_at_max_cells,
        case_returns_empty_when_none_in_range,
    ]
    print(f"running {len(cases)} select_cells cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
