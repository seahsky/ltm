"""
TDD for diagnose_convolved_anomaly_calib.py — the $0 Gate-0b that asks: can the
open-set CLAP gate separate anomaly-vs-background on RIR-CONVOLVED (+ mixed)
audio? 0c proved the gate (calibrated on CLEAN clips) rejects the convolved
alarm. This recalibrates on the live signal (render_at_pose -> diotic bed mix ->
is_anomaly) and emits GATE_RESULT + RECOMMEND_DELTA/TAU/BG_GAIN.

Only the pure decision logic is unit-tested here (the render+CLAP scoring loop is
RACE/GPU-bound):
  * sweep_delta_tau — 2-D (delta, tau) threshold sweep (M3: tau axis, since
    convolution can depress s_anom absolutely — the 0c alarm->glass_break flip).
  * decide_gate — the R1-aware rule: STOP unless some bg_gain BOTH clears
    onset_rms on the post-convolution bed AND separates (EER <= 0.15).

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_diagnose_convolved_anomaly_calib.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_convolved_anomaly_calib as cc  # noqa: E402


# ----------------------------------------------------------------------
# sweep_delta_tau — (margin, s_anom) 2-D separation
# ----------------------------------------------------------------------
def case_sweep_clean_separation():
    accept = [(0.30, 0.40), (0.35, 0.45), (0.28, 0.42)]
    reject = [(0.00, 0.20), (0.05, 0.25), (-0.02, 0.18)]
    r = cc.sweep_delta_tau(accept, reject)
    assert r["ok"] is True
    assert r["eer"] == 0.0, r
    # a separating delta sits between the reject max margin and accept min margin
    assert 0.05 < r["delta"] <= 0.30


def case_sweep_tau_axis_helps_when_margin_overlaps():
    # margins overlap, but s_anom cleanly separates -> the tau axis must find it
    accept = [(0.10, 0.50), (0.12, 0.52)]
    reject = [(0.11, 0.20), (0.09, 0.22)]
    r = cc.sweep_delta_tau(accept, reject)
    assert r["ok"] is True
    assert r["eer"] == 0.0, r          # solvable only via tau
    assert r["tau"] > 0.22 and r["tau"] <= 0.50


def case_sweep_overlap_has_error():
    accept = [(0.10, 0.30), (0.30, 0.40)]
    reject = [(0.12, 0.31), (0.28, 0.39)]   # genuinely intermixed on both axes
    r = cc.sweep_delta_tau(accept, reject)
    assert r["ok"] is True
    assert r["eer"] > 0.0


def case_sweep_empty_not_ok():
    assert cc.sweep_delta_tau([], [(0.0, 0.1)])["ok"] is False


# ----------------------------------------------------------------------
# decide_gate — R1-aware GO/BORDERLINE/STOP over the bg_gain sweep
# ----------------------------------------------------------------------
def _g(bg_gain, eer, bed_rms_med, delta=0.1, tau=0.2):
    return {"bg_gain": bg_gain, "eer": eer, "bed_rms_med": bed_rms_med,
            "delta": delta, "tau": tau}


def case_decide_go_when_bed_clears_onset_and_separates():
    per_gain = [_g(0.0, 0.02, 0.0), _g(0.5, 0.10, 0.08)]
    res, rec = cc.decide_gate(per_gain, onset_rms=0.05)
    assert res == "GO", (res, rec)
    assert rec["bg_gain"] == 0.5 and rec["delta"] == 0.1


def case_decide_stop_when_no_gain_clears_onset_R1():
    # separation is great, but the bed never reaches the gate (bed_rms < onset)
    # -> the REJECT half is vacuous -> STOP (not GO). This is the R1/C2 blocker.
    per_gain = [_g(0.0, 0.0, 0.0), _g(0.5, 0.03, 0.02), _g(0.7, 0.05, 0.03)]
    res, rec = cc.decide_gate(per_gain, onset_rms=0.05)
    assert res == "STOP", (res, rec)
    assert "onset_rms" in rec["reason"]


def case_decide_borderline():
    per_gain = [_g(0.5, 0.22, 0.09)]
    res, _ = cc.decide_gate(per_gain, onset_rms=0.05)
    assert res == "BORDERLINE", res


def case_decide_stop_when_inseparable():
    per_gain = [_g(0.5, 0.40, 0.09)]
    res, _ = cc.decide_gate(per_gain, onset_rms=0.05)
    assert res == "STOP", res


def case_decide_picks_lowest_eer_among_qualifying():
    per_gain = [_g(0.3, 0.14, 0.06), _g(0.5, 0.08, 0.07), _g(0.7, 0.12, 0.10)]
    res, rec = cc.decide_gate(per_gain, onset_rms=0.05)
    assert res == "GO"
    assert rec["bg_gain"] == 0.5   # lowest EER among the gains that clear onset


def case_select_audible_cells_reads_property_and_selects_band():
    # exercises the render helper (numpy/scipy, no CLAP): cell_energies is a
    # @property (must NOT be called with ()), render_at_pose + rms compute the
    # per-cell audible rms, and the band filter returns a bounded int list.
    import numpy as np
    from embodied_memory.audio import RIRGrid
    N, T = 6, 16
    cell_pos = np.stack([np.linspace(0, 5, N), np.zeros(N), np.zeros(N)], axis=1)
    irs = np.zeros((N, 2, T), dtype=np.float32)
    for i in range(N):
        irs[i, :, 0] = 0.5 / (i + 1)   # decaying per-cell direct-path gain
    grid = RIRGrid(cell_pos, [0.0, 0.0, 0.0], irs, sample_rate=16000, scene_id="t")
    clip = (0.1 * np.sin(np.linspace(0, 20, 64))).astype(np.float32)
    cells = cc._select_audible_cells(grid, clip, onset_rms=1e-4, band_hi=1e6, max_cells=4)
    assert isinstance(cells, list) and 1 <= len(cells) <= 4
    assert all(isinstance(c, (int, np.integer)) for c in cells)


def case_select_loud_cells_returns_top_energy():
    import numpy as np
    from embodied_memory.audio import RIRGrid
    N, T = 6, 16
    cell_pos = np.stack([np.linspace(0, 5, N), np.zeros(N), np.zeros(N)], axis=1)
    irs = np.zeros((N, 2, T), dtype=np.float32)
    for i in range(N):
        irs[i, :, 0] = 0.5 / (i + 1)    # cell 0 loudest, monotone decreasing
    grid = RIRGrid(cell_pos, [0.0, 0.0, 0.0], irs, sample_rate=16000, scene_id="t")
    cells = cc._select_loud_cells(grid, max_cells=3)
    assert cells[0] == 0, "loudest cell (nearest source) first"
    assert len(cells) == 3 and all(isinstance(c, int) for c in cells)


def main() -> int:
    cases = [
        case_sweep_clean_separation,
        case_sweep_tau_axis_helps_when_margin_overlaps,
        case_sweep_overlap_has_error,
        case_sweep_empty_not_ok,
        case_decide_go_when_bed_clears_onset_and_separates,
        case_decide_stop_when_no_gain_clears_onset_R1,
        case_decide_borderline,
        case_decide_stop_when_inseparable,
        case_decide_picks_lowest_eer_among_qualifying,
        case_select_audible_cells_reads_property_and_selects_band,
        case_select_loud_cells_returns_top_energy,
    ]
    print(f"running {len(cases)} diagnose_convolved_anomaly_calib cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
