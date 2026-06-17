"""
TDD sanity test for ``embodied_memory.audio`` — the runner-side audio layer
(M0b of the AudioGoal/FSD50K plan).

The audio module must, with NO habitat-sim / SoundSpaces import:
  * load a precomputed binaural RIR grid (``.npz``) and find the nearest cell
    to an agent pose (O(1)-ish 2-D xz lookup);
  * render-at-pose: ``scipy.signal.fftconvolve`` a mono FSD50K clip with the
    nearest cell's binaural IR → a (2, L) ear signal;
  * estimate azimuth DOA from that binaural signal via ILD/ITD;
  * 3-way zero-shot anomaly classification (cry/alarm/glass) given an injected
    CLAP-style encoder (mocked here — the real CLAP is heavy and lives in
    ``perception``).

These cases use SYNTHETIC IRs/signals so they are fully deterministic and run
in the live ``ltm-embodied`` env (numpy + scipy). The REAL-RIR green-bar (real
SoundSpaces grid + real FSD50K clip) is ``audio_loop_smoke.py`` on RACE.

The module is loaded standalone via ``spec_from_file_location`` (same pattern as
``test_propose_candidates.py``) so we never trigger the faiss-importing
``embodied_memory`` package ``__init__``.

    python embodied_memory/scripts/test_audio.py
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_EMB_DIR = Path(__file__).resolve().parent.parent  # …/embodied_memory


def _load_file_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


audio = _load_file_as("embodied_memory._audio_under_test", _EMB_DIR / "audio.py")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


_SR = 16000
_EARS_M = 0.18
_C = 343.0


def _impulse(n: int, at: int, amp: float = 1.0) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    if 0 <= at < n:
        x[at] = amp
    return x


def _binaural_for_azimuth(azimuth_rad: float, n: int = 256, base: int = 64,
                          amp: float = 1.0) -> np.ndarray:
    """Synthesize a (2, n) [left, right] impulse pair whose ITD encodes
    ``azimuth_rad`` (positive = source to the RIGHT) under the same simple
    free-field model the estimator inverts, so DOA must round-trip it."""
    # path-length difference d_ears*sin(theta); right ear closer for theta>0,
    # so LEFT arrives `lag` samples later than RIGHT.
    lag = int(round(math.sin(azimuth_rad) * _EARS_M / _C * _SR))
    left = _impulse(n, base + lag, amp)
    right = _impulse(n, base, amp)
    return np.stack([left, right], axis=0)


# ----------------------------------------------------------------------
# RIR grid: construct / save+load / nearest lookup
# ----------------------------------------------------------------------


def case_grid_nearest_lookup():
    # 3 cells on a line; source at x=0. Agent near cell 1 → cell 1 returned.
    cells = np.array([[0.0, 1.5, 0.0], [2.0, 1.5, 0.0], [4.0, 1.5, 0.0]],
                     dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.0) for _ in range(3)], axis=0)
    grid = audio.RIRGrid(
        cell_positions=cells, source_position=np.array([0.0, 1.5, 0.0], np.float32),
        irs=irs, sample_rate=_SR, scene_id="wcojb4TFT35")
    ir, idx, dist = grid.nearest(np.array([1.9, 1.5, 0.3], dtype=np.float32))
    assert idx == 1, f"expected nearest cell 1, got {idx}"
    assert ir.shape == (2, 256), ir.shape
    assert abs(dist - math.hypot(2.0 - 1.9, 0.0 - 0.3)) < 1e-4, dist
    # 2-D xz distance only — a big y offset must not change the pick.
    _, idx2, _ = grid.nearest(np.array([3.9, 99.0, 0.0], dtype=np.float32))
    assert idx2 == 2, f"y must be ignored in lookup, got {idx2}"
    print("  case grid_nearest_lookup: OK")


def case_grid_save_load_roundtrip():
    cells = np.array([[0.0, 1.5, 0.0], [2.0, 1.5, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.2), _binaural_for_azimuth(-0.2)], axis=0)
    src = np.array([1.0, 1.5, 1.0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "rir_grid.npz")
        audio.save_rir_grid(path, cell_positions=cells, source_position=src,
                            irs=irs, sample_rate=_SR, scene_id="TEEsavR23oF")
        # Plain .npz — loadable WITHOUT allow_pickle (no object arrays).
        raw = np.load(path)
        assert raw["irs"].shape == (2, 2, 256), raw["irs"].shape
        g2 = audio.RIRGrid.load(path)
    assert g2.sample_rate == _SR and g2.scene_id == "TEEsavR23oF"
    assert np.allclose(g2.cell_positions, cells)
    assert np.allclose(g2.source_position, src)
    assert np.allclose(g2.irs, irs)
    print("  case grid_save_load_roundtrip: OK")


def case_grid_save_variable_length_pads():
    # Renderer may produce slightly different IR lengths per cell; save must
    # zero-pad to a common T so the stored array is plain (no object dtype).
    cells = np.array([[0.0, 1.5, 0.0], [2.0, 1.5, 0.0]], dtype=np.float32)
    irs = [np.ones((2, 100), np.float32), np.ones((2, 160), np.float32)]
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "g.npz")
        audio.save_rir_grid(path, cell_positions=cells,
                            source_position=np.zeros(3, np.float32),
                            irs=irs, sample_rate=_SR, scene_id="s")
        raw = np.load(path)
        assert raw["irs"].dtype != object, "stored irs must not be an object array"
        assert raw["irs"].shape == (2, 2, 160), raw["irs"].shape
        # the short IR's tail is zero-padded
        assert np.allclose(raw["irs"][0, :, 100:], 0.0)
        assert np.allclose(raw["irs"][0, :, :100], 1.0)
    print("  case grid_save_variable_length_pads: OK")


# ----------------------------------------------------------------------
# render-at-pose: fftconvolve + RMS falls off with distance
# ----------------------------------------------------------------------


def case_render_shape_and_convolution():
    cells = np.array([[0.0, 1.5, 0.0]], dtype=np.float32)
    ir = np.zeros((1, 2, 8), dtype=np.float32)
    ir[0, 0, 0] = 1.0   # left = identity (passes clip through)
    ir[0, 1, 0] = 0.5   # right = half-gain identity
    grid = audio.RIRGrid(cells, np.zeros(3, np.float32), ir, _SR, "s")
    clip = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = audio.render_at_pose(grid, np.zeros(3, np.float32), clip)
    assert out.shape[0] == 2, out.shape
    # identity IR → left channel reproduces the clip (modulo conv tail zeros)
    assert np.allclose(out[0, :3], clip), out[0, :3]
    assert np.allclose(out[1, :3], 0.5 * clip), out[1, :3]
    print("  case render_shape_and_convolution: OK")


def case_render_rms_monotone_with_distance():
    # cells at increasing distance from the source; IR energy ∝ 1/d² (free
    # field). render → RMS must DECREASE as the agent stands at farther cells.
    src = np.array([0.0, 1.5, 0.0], dtype=np.float32)
    dists = [1.0, 2.0, 4.0, 8.0]
    cells = np.array([[d, 1.5, 0.0] for d in dists], dtype=np.float32)
    irs = np.zeros((len(dists), 2, 16), dtype=np.float32)
    for i, d in enumerate(dists):
        g = 1.0 / d
        delay = min(int(round(d / _C * _SR)), 15)
        irs[i, 0, delay] = g
        irs[i, 1, delay] = g
    grid = audio.RIRGrid(cells, src, irs, _SR, "s")
    rng = np.random.default_rng(0)
    clip = rng.standard_normal(2000).astype(np.float32)
    rmss = [audio.rms(audio.render_at_pose(grid, c, clip)) for c in cells]
    assert all(rmss[i] > rmss[i + 1] for i in range(len(rmss) - 1)), \
        f"RMS not monotone decreasing with distance: {rmss}"
    print(f"  case render_rms_monotone_with_distance (rms={[round(r,4) for r in rmss]}): OK")


# ----------------------------------------------------------------------
# DOA: ILD/ITD azimuth recovers the encoded bearing
# ----------------------------------------------------------------------


def case_doa_recovers_right_and_left():
    for method in ("xcorr", "gcc_phat"):
        for deg in (10.0, 30.0, 45.0, -20.0, -40.0):
            az = math.radians(deg)
            b = _binaural_for_azimuth(az, amp=1.0)
            est = audio.estimate_doa(b, _SR, ear_distance_m=_EARS_M,
                                     speed_of_sound=_C, method=method)
            assert abs(math.degrees(est) - deg) <= 5.0, \
                f"[{method}] azimuth {deg}° recovered as {math.degrees(est):.1f}°"
    print("  case doa_recovers_right_and_left (xcorr+gcc_phat): OK")


def case_doa_front_centered_is_zero():
    b = _binaural_for_azimuth(0.0)
    est = audio.estimate_doa(b, _SR, ear_distance_m=_EARS_M, speed_of_sound=_C)
    assert abs(math.degrees(est)) < 5.0, f"front source DOA {math.degrees(est):.1f}° not ~0"
    print("  case doa_front_centered_is_zero: OK")


def case_doa_sign_from_ild_when_itd_zero():
    # zero ITD but louder right ear → must still call it 'to the right' (>0).
    n, base = 256, 64
    left = _impulse(n, base, amp=0.3)
    right = _impulse(n, base, amp=1.0)
    est = audio.estimate_doa(np.stack([left, right]), _SR,
                             ear_distance_m=_EARS_M, speed_of_sound=_C)
    assert est > 0.0, f"louder-right with zero ITD should be +az, got {math.degrees(est):.1f}°"
    print("  case doa_sign_from_ild_when_itd_zero: OK")


# ----------------------------------------------------------------------
# CLAP zero-shot 3-way classification (encoder injected / mocked)
# ----------------------------------------------------------------------


class _FakeCLAP:
    """Stand-in for perception.CLAPAudioEncoder: maps known strings/markers to
    orthonormal 512-d vectors so cosine argmax is deterministic."""

    def __init__(self, audio_class: str):
        self._audio_class = audio_class
        self._basis = {c: i for i, c in enumerate(audio.ANOMALY_CLASSES)}

    def _vec(self, idx: int) -> np.ndarray:
        v = np.zeros(512, dtype=np.float32)
        v[idx] = 1.0
        return v

    def encode_audio(self, waveform, sample_rate):
        return self._vec(self._basis[self._audio_class])

    def encode_text(self, text):
        for c, idx in self._basis.items():
            if audio.CLASS_TO_CLAP_PROMPT[c] == text:
                return self._vec(idx)
        return np.zeros(512, dtype=np.float32)


def case_classify_anomaly_argmax():
    for true_c in audio.ANOMALY_CLASSES:
        enc = _FakeCLAP(true_c)
        wav = np.zeros(8000, dtype=np.float32)
        cls, scores = audio.classify_anomaly(wav, _SR, enc)
        assert cls == true_c, f"expected {true_c}, got {cls} (scores={scores})"
        assert set(scores.keys()) == set(audio.ANOMALY_CLASSES)
        assert abs(scores[true_c] - max(scores.values())) < 1e-6
    print("  case classify_anomaly_argmax: OK")


def case_class_maps_present_and_consistent():
    assert audio.ANOMALY_CLASSES == ("baby_cry", "alarm", "glass_break")
    for c in audio.ANOMALY_CLASSES:
        assert c in audio.CLASS_TO_OBJECT and audio.CLASS_TO_OBJECT[c]
        assert c in audio.CLASS_TO_CLAP_PROMPT and audio.CLASS_TO_CLAP_PROMPT[c]
    # the M0c demo mapping the plan pins
    assert audio.CLASS_TO_OBJECT["baby_cry"] == "crib"
    print("  case class_maps_present_and_consistent: OK")


# ----------------------------------------------------------------------


def main() -> int:
    cases = [
        case_grid_nearest_lookup,
        case_grid_save_load_roundtrip,
        case_grid_save_variable_length_pads,
        case_render_shape_and_convolution,
        case_render_rms_monotone_with_distance,
        case_doa_recovers_right_and_left,
        case_doa_front_centered_is_zero,
        case_doa_sign_from_ild_when_itd_zero,
        case_classify_anomaly_argmax,
        case_class_maps_present_and_consistent,
    ]
    print(f"running {len(cases)} audio.py cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
