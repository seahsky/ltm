"""
diagnose_raw_itd — confirm the SoundSpaces binaural rendering is ITD-WEAK but
ILD-REAL, directly on the rendered RIR grid (M0b root-cause confirmation).

The AudioGoal M0 diagnosis (multi-agent, 2026-06-17) concluded the small DOA
azimuths are an INHERENT property of SoundSpaces 2.0 / RLRAudioPropagation
binaural rendering (Ambisonic→time-aligned-HRTF, Zaunschirm 2018, strips the
broadband ITD), NOT a bug in our DOA pipeline (convolution provably preserves
ITD) or convention. This script confirms that on the REAL grids by measuring
the interaural delay + level DIRECTLY off the raw binaural IRs (``grid.irs[i]``),
BEFORE any clip convolution.

For each near cell it reports the raw onset-ITD, peak-ITD and GCC-PHAT-ITD (all
in samples), the ITD-implied azimuth, and the interaural level difference (dB).

EXPECTED (H1 confirmed): on cells where the source is meaningfully lateral, the
raw ITDs are small (~1-8 samples) while the ILD has the correct sign (positive
when the source is to the right) and a several-dB magnitude — i.e. the engine
bakes weak ITD but a real ILD, so the M0b lateral-SIGN gate rests on a real
signal. Run in the ``ltm-embodied`` env:

    python embodied_memory/scripts/diagnose_raw_itd.py \
        runs/audiogoal/wcojb4TFT35_rir_grid.npz --near 8
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from embodied_memory.audio import RIRGrid  # noqa: E402


def _onset_idx(ir1d: np.ndarray, thresh_frac: float = 0.1) -> int:
    a = np.abs(np.asarray(ir1d, dtype=np.float64))
    pk = float(a.max()) if a.size else 0.0
    if pk <= 0.0:
        return 0
    return int(np.argmax(a >= thresh_frac * pk))


def _gccphat_itd(left: np.ndarray, right: np.ndarray, max_lag: int) -> int:
    n = max(len(left), len(right))
    nfft = 1
    while nfft < 2 * n:
        nfft <<= 1
    L = np.fft.rfft(left, nfft)
    R = np.fft.rfft(right, nfft)
    cross = L * np.conj(R)
    mag = np.abs(cross)
    mag[mag < 1e-12] = 1e-12
    cc = np.fft.irfft(cross / mag, nfft)
    cc = np.concatenate((cc[-max_lag:], cc[: max_lag + 1]))
    lags = np.arange(-max_lag, max_lag + 1)
    return int(lags[int(np.argmax(cc))])


def _true_bearing(cell, src) -> float:
    return math.atan2(float(src[0] - cell[0]), float(src[2] - cell[2]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("grid", help="rir_grid.npz from render_rir_grid.py")
    ap.add_argument("--near", type=int, default=8, help="how many nearest cells to inspect")
    ap.add_argument("--ear", type=float, default=0.18)
    ap.add_argument("--c", type=float, default=343.0)
    ap.add_argument("--lateral-min-deg", type=float, default=20.0)
    args = ap.parse_args()

    if not os.path.isfile(args.grid):
        print(f"RED: grid not found: {args.grid}")
        return 2

    g = RIRGrid.load(args.grid)
    src = g.source_position
    cxz = g.cell_positions[:, [0, 2]]
    d = np.linalg.norm(cxz - src[[0, 2]][None, :], axis=1)
    order = np.argsort(d)[: args.near]
    max_lag = int(np.ceil(args.ear / args.c * g.sample_rate)) + 1
    lat_thresh = math.sin(math.radians(args.lateral_min_deg))

    print(f"grid={os.path.basename(args.grid)} cells={len(g)} sr={g.sample_rate} "
          f"max_lag=±{max_lag} samples (=±90°)")
    print(f"{'d_m':>5} {'true°':>7} {'itd_on':>7} {'itd_pk':>7} {'itd_gcc':>8} "
          f"{'raw°':>7} {'ild_dB':>7} {'lat':>4} {'ildOK':>6}")

    rows = []
    for i in order:
        ir = np.asarray(g.irs[i], dtype=np.float64)
        L, R = ir[0], ir[1]
        itd_on = _onset_idx(L) - _onset_idx(R)
        itd_pk = int(np.argmax(np.abs(L))) - int(np.argmax(np.abs(R)))
        itd_gcc = _gccphat_itd(L, R, max_lag)
        s = float(np.clip(itd_gcc / g.sample_rate * args.c / args.ear, -1.0, 1.0))
        raw_deg = math.degrees(math.asin(s))
        rms_l = float(np.sqrt(np.mean(L ** 2)))
        rms_r = float(np.sqrt(np.mean(R ** 2)))
        ild_db = 20.0 * math.log10((rms_r + 1e-12) / (rms_l + 1e-12))
        tb = _true_bearing(g.cell_positions[i], src)
        s_true = math.sin(tb)
        lateral = abs(s_true) >= lat_thresh
        ild_ok = (np.sign(ild_db) == np.sign(s_true)) if lateral else None
        rows.append((bool(lateral), ild_ok))
        print(f"{float(d[i]):5.2f} {math.degrees(tb):7.1f} {itd_on:7d} {itd_pk:7d} "
              f"{itd_gcc:8d} {raw_deg:7.1f} {ild_db:7.2f} "
              f"{'Y' if lateral else '.':>4} "
              f"{('OK' if ild_ok else 'MISS') if lateral else '-':>6}")

    lat = [r for r in rows if r[0]]
    if lat:
        ok = sum(1 for _, k in lat if k)
        print(f"\nVERDICT: {len(lat)} lateral cells; ILD-sign correct on {ok}/{len(lat)}.")
        print("  H1 confirmed if raw itd_* are small (~1-8) on lateral cells AND ILD-sign is")
        print("  mostly correct → engine bakes weak ITD but a REAL ILD → the M0b lateral-sign")
        print("  DOA gate rests on a real signal (energy/ILD), not the engine-weak ITD.")
    else:
        print("\nNo lateral cells in the near set (source mostly ahead/behind these cells); "
              "pass a larger --near.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
