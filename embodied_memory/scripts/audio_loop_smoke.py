"""
audio_loop_smoke — M0b real-path GO/NO-GO: a precomputed RIR grid + a real
clip produce a HEARD and LOCALIZABLE signal in the live ``ltm-embodied`` env
(numpy + scipy only — NO habitat-sim / SoundSpaces import).

Loads ``rir_grid.npz`` (from ``render_rir_grid.py`` on RACE), walks the cells in
order of decreasing distance to the source, renders the clip at each via
``audio.render_at_pose`` (nearest-cell IR + fftconvolve), and checks:

  1. RMS energy RISES as the agent nears the source (the AudioGoal gradient);
  2. at the nearest cell, the ILD/ITD azimuth DOA points at the true source
     bearing within ~30° (front-back fold allowed — binaural is front-back
     ambiguous).

``--clip`` is optional: any mono/stereo .wav, OR (default) a synthetic
broadband burst, so this M0b gate does not block on FSD50K being staged (the
FSD50K-specific CLAP classification is exercised in M3). ``--self-test`` builds a
synthetic grid in a temp dir and runs the same checks — used to validate this
smoke's own logic off-RACE.

GREEN = monotone-toward-source energy + DOA within tolerance.

    python embodied_memory/scripts/audio_loop_smoke.py --rir-grid runs/audiogoal/wcojb4TFT35_rir_grid.npz
    python embodied_memory/scripts/audio_loop_smoke.py --self-test
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embodied_memory import audio  # noqa: E402


def _load_clip(path: str, target_sr: int) -> np.ndarray:
    from scipy.io import wavfile
    sr, data = wavfile.read(path)
    data = np.asarray(data, dtype=np.float32)
    if np.issubdtype(data.dtype, np.integer):
        data = data / 32768.0
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.reshape(-1).astype(np.float32)
    m = float(np.max(np.abs(data))) or 1.0
    data = data / m
    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    return data


def _synthetic_clip(target_sr: int, seconds: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(target_sr * seconds)
    # broadband noise with an onset envelope (alarm-ish, energy across bands)
    env = np.minimum(1.0, np.linspace(0, 4, n))
    return (rng.standard_normal(n).astype(np.float32) * env).astype(np.float32)


def _true_bearing(cell_xyz: np.ndarray, source_xyz: np.ndarray) -> float:
    """Horizontal bearing of the source in the listener's frame, agent facing
    the navmesh default (yaw=0 — render_rir_grid leaves rotation at identity).
    Same convention as memory_bridge: atan2(dx, dz)."""
    dx = float(source_xyz[0] - cell_xyz[0])
    dz = float(source_xyz[2] - cell_xyz[2])
    return math.atan2(dx, dz)


def _angle_err_with_fold(est: float, true: float) -> float:
    """Smallest azimuth error allowing the binaural front-back fold."""
    def _wrap(a):
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return abs(a)
    direct = _wrap(est - true)
    fold = _wrap(est - (math.pi - true))   # front-back mirror
    return min(direct, fold)


def _run(grid: audio.RIRGrid, clip: np.ndarray, doa_tol_deg: float = 30.0) -> int:
    src = grid.source_position
    cxz = grid.cell_positions[:, [0, 2]]
    dists = np.linalg.norm(cxz - src[[0, 2]][None, :], axis=1)
    order = np.argsort(-dists)  # far → near

    rmss, ds = [], []
    for i in order:
        out = audio.render_at_pose(grid, grid.cell_positions[i], clip)
        rmss.append(audio.rms(out))
        ds.append(float(dists[i]))
    rmss = np.array(rmss)
    ds = np.array(ds)

    # (1) energy rises toward the source: split far/near halves by distance.
    half = len(rmss) // 2
    far_mean = float(np.mean(rmss[:half])) if half else 0.0
    near_mean = float(np.mean(rmss[half:]))
    # rank correlation of (distance, rms) should be strongly negative.
    rank_d = np.argsort(np.argsort(ds))
    rank_r = np.argsort(np.argsort(rmss))
    n = len(ds)
    spearman = 1.0 - 6.0 * float(np.sum((rank_d - rank_r) ** 2)) / (n * (n * n - 1)) \
        if n > 2 else (-1.0 if rmss[-1] > rmss[0] else 1.0)
    print(f"  cells={n}  RMS far-half={far_mean:.5f}  near-half={near_mean:.5f}  "
          f"spearman(dist,rms)={spearman:.3f}")

    # (2) DOA at the nearest cell
    near_i = int(order[-1])
    near_cell = grid.cell_positions[near_i]
    near_out = audio.render_at_pose(grid, near_cell, clip)
    est = audio.estimate_doa(near_out, grid.sample_rate)
    true = _true_bearing(near_cell, src)
    err = math.degrees(_angle_err_with_fold(est, true))
    print(f"  nearest cell d={dists[near_i]:.2f}m  DOA est={math.degrees(est):.1f}°  "
          f"true={math.degrees(true):.1f}°  err(fold)={err:.1f}°")

    fails = []
    if not (near_mean > far_mean):
        fails.append(f"energy does not rise toward source "
                     f"(near {near_mean:.5f} <= far {far_mean:.5f})")
    if not (spearman < -0.3):
        fails.append(f"distance↔RMS not strongly negative (spearman {spearman:.3f})")
    if not (err <= doa_tol_deg):
        fails.append(f"DOA error {err:.1f}° > {doa_tol_deg}° tolerance")
    if fails:
        for f in fails:
            print(f"RED: {f}")
        return 1
    print(f"GREEN: signal is heard (energy rises toward source) and localizable "
          f"(DOA within {doa_tol_deg:.0f}°).")
    return 0


def _build_synthetic_grid(sr: int):
    """A line of cells receding from the source along +z, IR gain ∝ 1/d with an
    ITD that encodes each cell's true bearing — exercises the smoke end to end."""
    src = np.array([0.0, 1.5, 0.0], dtype=np.float32)
    cells, irs = [], []
    T = 256
    for k in range(1, 9):
        cell = np.array([0.6, 1.5, float(k)], dtype=np.float32)  # offset in x → nonzero az
        cells.append(cell)
        az = _true_bearing(cell, src)
        lag = int(round(math.sin(az) * 0.18 / 343.0 * sr))
        d = float(np.linalg.norm((cell - src)[[0, 2]]))
        g = 1.0 / d
        base = 32
        ir = np.zeros((2, T), dtype=np.float32)
        ir[0, base + lag] = g     # left later when az>0 (source to the right)
        ir[1, base] = g
        irs.append(ir)
    return audio.RIRGrid(np.stack(cells), src, np.stack(irs), sr, "synthetic")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rir-grid", default=None, help="rir_grid.npz from render_rir_grid.py")
    ap.add_argument("--clip", default=None, help="optional .wav (default: synthetic burst)")
    ap.add_argument("--doa-tol-deg", type=float, default=30.0)
    ap.add_argument("--self-test", action="store_true",
                    help="build a synthetic grid and validate this smoke's logic")
    args = ap.parse_args()

    if args.self_test:
        sr = 16000
        grid = _build_synthetic_grid(sr)
        clip = _synthetic_clip(sr)
        print("  [self-test] synthetic grid")
        return _run(grid, clip, args.doa_tol_deg)

    if not args.rir_grid:
        print("RED: --rir-grid is required (or pass --self-test)")
        return 2
    if not os.path.isfile(args.rir_grid):
        print(f"RED: rir grid not found: {args.rir_grid}")
        return 2
    grid = audio.RIRGrid.load(args.rir_grid)
    clip = _load_clip(args.clip, grid.sample_rate) if args.clip \
        else _synthetic_clip(grid.sample_rate)
    print(f"  grid: {len(grid)} cells  sr={grid.sample_rate}  scene={grid.scene_id}  "
          f"clip={'wav:' + os.path.basename(args.clip) if args.clip else 'synthetic'}")
    return _run(grid, clip, args.doa_tol_deg)


if __name__ == "__main__":
    sys.exit(main())
