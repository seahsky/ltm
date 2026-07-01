"""
TDD for diagnose_anomaly_feasibility.py — the $0 gate the N3 driver runs AFTER
rendering the RIR grid at the DECOUPLED source, BEFORE any paid LLM run. It asks:
does an AUDIBLE-NOT-LOUD, reachable-to-goal warm/search start actually exist near
the decoupled source? (The builder can't compute point-to-point geodesics — the
two-env split — so this render-time check is the decisive feasibility adjudicator.)

Two failure modes it catches:
  * LOUD start — the warm start is on top of the source (grid-relative cell energy
    near the max) → the loud diotic bed FALSE-FIRES onset at step 0 (the exact
    defect N3 removes).
  * QUIET / OUT-OF-COVERAGE start — the source is inaudible from the search region
    → onset never fires → the interrupt path is never exercised.

Pure decision logic is unit-tested here; the grid-load + dataset glue is exercised
with a hand-built RIRGrid (numpy only, no GPU/sim).

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_diagnose_anomaly_feasibility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_anomaly_feasibility as fz  # noqa: E402


# ----------------------------------------------------------------------
# classify_start — grid-relative audibility of one search start
# ----------------------------------------------------------------------
def case_classify_out_of_coverage():
    # nearest cell is > coverage_m away → the source grid doesn't cover the start
    assert fz.classify_start(0.5, dist_to_cell_m=5.0, coverage_m=2.0) == "OUT_OF_COVERAGE"


def case_classify_quiet():
    assert fz.classify_start(0.005, dist_to_cell_m=0.3, audible_frac=0.02) == "QUIET"


def case_classify_loud():
    # near the source (energy ~ the grid max) → step-0 false-fire risk
    assert fz.classify_start(0.8, dist_to_cell_m=0.3, loud_frac=0.5) == "LOUD"


def case_classify_audible_not_loud():
    assert fz.classify_start(0.2, dist_to_cell_m=0.3,
                             audible_frac=0.02, loud_frac=0.5) == "AUDIBLE"


# ----------------------------------------------------------------------
# cell_verdict — per (scene, category) GO/SKIP over its warm starts
# ----------------------------------------------------------------------
def case_verdict_go_when_one_audible():
    v, _ = fz.cell_verdict(["LOUD", "AUDIBLE", "QUIET"])
    assert v == "GO"


def case_verdict_skip_all_loud():
    v, reason = fz.cell_verdict(["LOUD", "LOUD"])
    assert v == "SKIP" and "LOUD" in reason.upper()


def case_verdict_skip_all_inaudible():
    v, reason = fz.cell_verdict(["QUIET", "OUT_OF_COVERAGE"])
    assert v == "SKIP" and "audi" in reason.lower()


def case_verdict_skip_empty():
    v, reason = fz.cell_verdict([])
    assert v == "SKIP" and "no warm" in reason.lower()


# ----------------------------------------------------------------------
# feasibility_from_grid — the dataset+grid glue (hand-built RIRGrid)
# ----------------------------------------------------------------------
def _grid(cell_positions, energies_scale):
    """A tiny RIRGrid whose per-cell IR energy = energies_scale[i] (a single-tap
    impulse per cell so cell_energies == energies_scale)."""
    import numpy as np
    from embodied_memory.audio import RIRGrid
    N = len(cell_positions)
    irs = np.zeros((N, 2, 4), dtype=np.float32)
    for i, s in enumerate(energies_scale):
        irs[i, 0, 0] = float(s) ** 0.5   # sum(ir**2) over [2,T] == s
    return RIRGrid(np.asarray(cell_positions, np.float32), [0.0, 0.0, 0.0], irs,
                   sample_rate=16000, scene_id="S")


def _warm(eid, cat, start):
    return {"episode_id": eid, "object_category": cat, "scene_id": "S.basis.glb",
            "start_position": list(start), "info": {"source_position": [0.0, 0.0, 0.0]}}


def case_from_grid_go_when_a_midband_start_exists():
    # cell 0 loud (energy 1.0, at origin=source), cell 1 mid (0.1), cell 2 quiet (0.001).
    grid = _grid([[0, 0, 0], [3, 0, 0], [8, 0, 0]], [1.0, 0.1, 0.001])
    content = {"episodes": [
        _warm("bed-alarm-warm-1", "bed", [3.0, 0.0, 0.0]),   # nearest cell 1 -> AUDIBLE
        _warm("bed-alarm-warm-2", "bed", [0.1, 0.0, 0.0]),   # nearest cell 0 -> LOUD
    ]}
    res = fz.feasibility_from_grid(content, grid, audible_frac=0.02, loud_frac=0.5, coverage_m=2.0)
    key = next(iter(res))
    assert res[key]["verdict"] == "GO", res
    assert any(p["klass"] == "AUDIBLE" for p in res[key]["per_start"]), res


def case_from_grid_skip_when_all_loud():
    grid = _grid([[0, 0, 0], [3, 0, 0]], [1.0, 0.9])
    content = {"episodes": [
        _warm("bed-alarm-warm-1", "bed", [0.05, 0.0, 0.0]),
        _warm("bed-alarm-warm-2", "bed", [3.0, 0.0, 0.0]),
    ]}
    res = fz.feasibility_from_grid(content, grid, audible_frac=0.02, loud_frac=0.5, coverage_m=2.0)
    key = next(iter(res))
    assert res[key]["verdict"] == "SKIP", res


def case_from_grid_only_warm_episodes_counted():
    # a cold/seed episode (loud, at the source) must NOT be classified — only warm.
    grid = _grid([[0, 0, 0], [3, 0, 0]], [1.0, 0.1])
    content = {"episodes": [
        {"episode_id": "bed-alarm-cold-0", "object_category": "bed", "scene_id": "S.basis.glb",
         "start_position": [0.0, 0.0, 0.0], "info": {"source_position": [0.0, 0.0, 0.0]}},
        _warm("bed-alarm-warm-1", "bed", [3.0, 0.0, 0.0]),
    ]}
    res = fz.feasibility_from_grid(content, grid, audible_frac=0.02, loud_frac=0.5, coverage_m=2.0)
    key = next(iter(res))
    assert len(res[key]["per_start"]) == 1, res
    assert res[key]["verdict"] == "GO", res


# ----------------------------------------------------------------------
# default_coverage_m — coverage must track the grid's actual cell spacing
# (a fixed 2.0 m false-rejects a sparse grid / is over-lenient on a dense one)
# ----------------------------------------------------------------------
def case_default_coverage_tracks_grid_spacing():
    import numpy as np
    dense = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)   # ~1 m spacing
    sparse = np.array([[0, 0, 0], [4, 0, 0], [8, 0, 0], [12, 0, 0]], float)  # ~4 m spacing
    cov_d = fz.default_coverage_m(dense)
    cov_s = fz.default_coverage_m(sparse)
    assert cov_s > cov_d, (cov_d, cov_s)          # sparse grid => larger coverage
    assert 1.0 <= cov_d <= 2.5, cov_d             # ~1.5x median 1 m
    assert cov_s >= 4.0, cov_s                    # ~1.5x median 4 m


def case_default_coverage_floor_for_tiny_grid():
    import numpy as np
    assert fz.default_coverage_m(np.array([[0, 0, 0]], float)) >= 1.0


def main() -> int:
    cases = [
        case_classify_out_of_coverage,
        case_classify_quiet,
        case_classify_loud,
        case_classify_audible_not_loud,
        case_verdict_go_when_one_audible,
        case_verdict_skip_all_loud,
        case_verdict_skip_all_inaudible,
        case_verdict_skip_empty,
        case_from_grid_go_when_a_midband_start_exists,
        case_from_grid_skip_when_all_loud,
        case_from_grid_only_warm_episodes_counted,
        case_default_coverage_tracks_grid_spacing,
        case_default_coverage_floor_for_tiny_grid,
    ]
    print(f"running {len(cases)} diagnose_anomaly_feasibility cases…")
    for c in cases:
        c()
        print(f"  case {c.__name__}: OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
