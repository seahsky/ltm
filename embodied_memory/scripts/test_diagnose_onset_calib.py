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


def case_recommend_bg_gain_puts_the_bed_under_onset():
    # ADR-0004: the bed must be a NOISE FLOOR, so its loudest cell has to sit
    # BELOW onset_rms. It cannot be hand-picked: the bed renders through the same
    # grid as the anomaly, so bed(x) ~= alarm(x) and the gain that separates them
    # is a property of the grid. Real numbers from runs/anomresp-bed-s3: the alarm
    # peaks ~0.284 near the source and onset_rms calibrated to 0.111.
    samples = [(1.0, 0.284), (2.0, 0.147), (4.0, 0.111), (6.0, 0.035)]
    rec = oc.recommend_bg_gain(samples, onset_rms=0.111, safety=0.7)
    assert rec["ok"] is True, rec
    g = rec["recommended_bg_gain"]
    assert 0.0 < g < 1.0, rec
    assert 0.284 * g <= 0.7 * 0.111 + 1e-9, \
        f"the loudest bed cell must land under safety*onset_rms; got {0.284 * g}"
    # bg_gain=1.0 (the pre-ADR-0004 setting) is exactly what false-fired at step 0
    assert 0.284 * 1.0 > 0.111, "sanity: the old gain DID clear onset (the bug)"
    print("  case recommend_bg_gain_puts_the_bed_under_onset: OK")


def case_recommend_bg_gain_caps_at_unit():
    # A bed already quiet everywhere needs no attenuation, but the gain must not
    # exceed 1.0 — above that the "background" is louder than the anomaly itself.
    samples = [(1.0, 0.010), (4.0, 0.004)]
    rec = oc.recommend_bg_gain(samples, onset_rms=0.111, safety=0.7)
    assert rec["ok"] is True and rec["recommended_bg_gain"] == 1.0, rec
    print("  case recommend_bg_gain_caps_at_unit: OK")


def case_recommend_bg_gain_not_ok_on_a_silent_bed():
    # A silent/absent bed cannot be scaled to anything meaningful — say so rather
    # than emit a gain the driver would trust.
    for samples in ([], [(1.0, 0.0), (4.0, 0.0)]):
        rec = oc.recommend_bg_gain(samples, onset_rms=0.111, safety=0.7)
        assert rec["ok"] is False, rec
    print("  case recommend_bg_gain_not_ok_on_a_silent_bed: OK")


def case_bed_energy_measured_through_diotic_collapse():
    # R4 domain-match (the lesson Gate-0b learned the hard way): the LIVE bed is
    # bg_gain * diotic_collapse(render_at_pose(...)), so calibrating on the raw
    # binaural render measures a different signal than the one onset_rms sees.
    # With L and R unequal, the collapse changes the RMS — so the flag must bite.
    src = [0.0, 0.0, 0.0]
    ir = np.zeros((1, 2, 4), dtype=np.float32)
    ir[0, 0, 0] = 1.0        # left loud
    ir[0, 1, 0] = 0.0        # right silent → collapse halves the energy
    grid = RIRGrid(np.array([[1.0, 0.0, 0.0]], np.float32), np.array(src, np.float32),
                   ir, 16000, "s")
    clip = np.ones(8, dtype=np.float32)
    raw = oc.cell_energy_vs_distance(grid, clip)
    dio = oc.cell_energy_vs_distance(grid, clip, diotic=True)
    assert dio[0][1] < raw[0][1], \
        f"diotic collapse must change the measured energy; raw={raw} diotic={dio}"
    print("  case bed_energy_measured_through_diotic_collapse: OK")


def case_bg_gain_end_to_end_on_the_same_grid():
    # The whole point of ADR-0004 in one assertion: bed and anomaly render through
    # the SAME grid at the same normalized level, so at unit gain the bed's peak
    # MUST clear an onset calibrated on the anomaly alone — and the recommended
    # gain must pull it back under.
    grid = _delta_grid()
    clip = (np.ones(200, dtype=np.float32) * 0.1)
    samples = oc.cell_energy_vs_distance(grid, clip)
    onset = oc.recommend_onset_rms(samples, 3.0, band=0.5)["recommended_onset_rms"]
    bed_samples = oc.cell_energy_vs_distance(grid, clip, diotic=True)   # same clip, same grid
    bg = oc.recommend_bg_gain(bed_samples, onset, safety=0.7)
    assert bg["ok"], bg
    assert bg["bed_max_at_unit_gain"] >= onset, \
        "the premise: an identically-rendered bed clears an anomaly-only onset at gain 1.0"
    assert bg["bed_max_at_recommended"] < onset, \
        f"the fix: at the recommended gain the bed must sit UNDER onset; got {bg}"
    print("  case bg_gain_end_to_end_on_the_same_grid: OK")


def main() -> int:
    cases = [
        case_fire_distance,
        case_recommend_band_median,
        case_recommend_fallback_nearest,
        case_recommend_empty_is_safe,
        case_cell_energy_decreases_with_distance,
        case_end_to_end_recommend_on_synthetic_grid,
        case_recommend_bg_gain_puts_the_bed_under_onset,
        case_recommend_bg_gain_caps_at_unit,
        case_recommend_bg_gain_not_ok_on_a_silent_bed,
        case_bed_energy_measured_through_diotic_collapse,
        case_bg_gain_end_to_end_on_the_same_grid,
    ]
    print(f"running {len(cases)} diagnose_onset_calib cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
