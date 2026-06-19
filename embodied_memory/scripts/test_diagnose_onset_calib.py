"""
TDD for diagnose_onset_calib — recommend AudioGoal onset_rms for a target audible
distance. Pure numpy/scipy (a synthetic delta-IR RIRGrid), no Habitat/model/grid file.

    KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
        python embodied_memory/scripts/test_diagnose_onset_calib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_onset_calib as oc  # noqa: E402
from embodied_memory.audio import RIRGrid  # noqa: E402


def _delta_grid():
    """3 cells at distances 1/3/6 m from the source on +x; each IR is a delta
    scaled by 1/dist, so rms(render_at_pose) decreases monotonically with distance."""
    src = [0.0, 0.0, 0.0]
    cells = [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    irs = []
    for c in cells:
        amp = 1.0 / abs(c[0])
        ir = np.zeros((2, 4), dtype=np.float32)
        ir[:, 0] = amp
        irs.append(ir)
    return RIRGrid(np.array(cells, np.float32), np.array(src, np.float32),
                   np.stack(irs), 48000, "syn")


def case_fire_distance():
    s = [(1.0, 0.5), (3.0, 0.1), (6.0, 0.02)]
    assert oc.fire_distance(s, 0.05) == 3.0          # cells at 1,3 clear 0.05
    assert oc.fire_distance(s, 0.6) == 0.0           # none clear 0.6
    assert oc.fire_distance(s, 0.01) == 6.0          # all clear 0.01
    print("  case fire_distance: OK")


def case_recommend_band_median():
    s = [(1.0, 0.5), (3.8, 0.10), (4.1, 0.14), (8.0, 0.02)]
    r = oc.recommend_onset_rms(s, 4.0, band=0.75)
    assert abs(r["recommended_onset_rms"] - 0.14) < 1e-9, r       # median of {0.10,0.14}
    assert abs(r["fire_dist_at_recommended"] - 4.1) < 1e-9, r     # audible to 4.1m
    print("  case recommend_band_median: OK")


def case_recommend_fallback_nearest():
    s = [(1.0, 0.5), (8.0, 0.02)]                    # nothing within 0.75m of 4.0
    r = oc.recommend_onset_rms(s, 4.0, band=0.75)
    assert abs(r["recommended_onset_rms"] - 0.5) < 1e-9, r        # nearest cell (1.0m)
    assert "nearest" in r["basis"], r
    print("  case recommend_fallback_nearest: OK")


def case_recommend_empty_is_safe():
    r = oc.recommend_onset_rms([], 4.0, current_rms=0.05)
    assert r["recommended_onset_rms"] == 0.05 and r["n_cells"] == 0
    print("  case recommend_empty_is_safe: OK")


def case_cell_energy_decreases_with_distance():
    grid = _delta_grid()
    clip = (np.ones(100, dtype=np.float32) * 0.1)    # rms 0.1 mono clip
    samples = oc.cell_energy_vs_distance(grid, clip)
    dists = [d for d, _ in samples]
    energies = [e for _, e in samples]
    assert dists == [1.0, 3.0, 6.0], dists
    assert energies[0] > energies[1] > energies[2] > 0.0, energies   # 1/dist falloff
    print("  case cell_energy_decreases_with_distance: OK")


def case_end_to_end_recommend_on_synthetic_grid():
    grid = _delta_grid()
    clip = (np.ones(100, dtype=np.float32) * 0.1)
    samples = oc.cell_energy_vs_distance(grid, clip)
    r = oc.recommend_onset_rms(samples, 3.0, band=0.5)            # target the middle cell
    mid_e = [e for d, e in samples if d == 3.0][0]
    assert abs(r["recommended_onset_rms"] - mid_e) < 1e-9, r      # picks the 3m cell's energy
    print("  case end_to_end_recommend_on_synthetic_grid: OK")


def main() -> int:
    cases = [
        case_fire_distance,
        case_recommend_band_median,
        case_recommend_fallback_nearest,
        case_recommend_empty_is_safe,
        case_cell_energy_decreases_with_distance,
        case_end_to_end_recommend_on_synthetic_grid,
    ]
    print(f"running {len(cases)} diagnose_onset_calib cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
