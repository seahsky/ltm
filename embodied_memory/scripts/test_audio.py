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


def case_doa_ild_overrides_subresolution_itd():
    # SoundSpaces bakes a TINY (sub-resolution) interaural delay even for side
    # sources, but a real ILD. When |ITD| is below the confidence floor, the
    # ILD sign must drive the azimuth — not a sub-sample lag of the wrong sign.
    rng = np.random.default_rng(3)
    noise = rng.standard_normal(2000).astype(np.float32)
    left = noise.copy()
    right = np.empty_like(noise)
    right[1:] = 2.0 * noise[:-1]      # right LOUDER (×2) and lagged by 1 sample
    right[0] = 0.0
    b = np.stack([left, right])
    # |itd| here is 1 sample (< floor 3) and its sign says left-leads (would give
    # a NEGATIVE az on the old code); ILD says right is louder → must be +az.
    est = audio.estimate_doa(b, _SR, ear_distance_m=_EARS_M, speed_of_sound=_C)
    assert est > math.radians(10.0), \
        f"sub-resolution ITD + louder-right ILD must give a clear +az, got {math.degrees(est):.1f}°"
    print("  case doa_ild_overrides_subresolution_itd: OK")


def case_lateral_sign_helper():
    n, base = 256, 64
    louder_right = np.stack([_impulse(n, base, 0.3), _impulse(n, base, 1.0)])
    louder_left = np.stack([_impulse(n, base, 1.0), _impulse(n, base, 0.3)])
    equal = np.stack([_impulse(n, base, 0.7), _impulse(n, base, 0.7)])
    assert audio.lateral_sign(louder_right) == 1
    assert audio.lateral_sign(louder_left) == -1
    assert audio.lateral_sign(equal) == 0
    print("  case lateral_sign_helper: OK")


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


class _FakeCLAPScores:
    """CLAP stand-in whose audio↔text cosine is fully controllable: the audio
    embeds to ``e0`` and each text embeds to a UNIT vector whose first component
    is the desired cosine, so ``cos(audio, text) == text_cos[text]`` exactly."""

    def __init__(self, text_cos):
        self._text_cos = dict(text_cos)

    def encode_audio(self, waveform, sample_rate):
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        return v

    def encode_text(self, text):
        c = float(max(-1.0, min(1.0, self._text_cos.get(text, 0.0))))
        v = np.zeros(8, dtype=np.float32)
        v[0] = c
        v[1] = math.sqrt(max(0.0, 1.0 - c * c))
        return v


def _cos_map(anom_alarm=0.1, anom_baby=0.1, anom_glass=0.1, normal=0.1):
    cos = {
        audio.CLASS_TO_CLAP_PROMPT["alarm"]: anom_alarm,
        audio.CLASS_TO_CLAP_PROMPT["baby_cry"]: anom_baby,
        audio.CLASS_TO_CLAP_PROMPT["glass_break"]: anom_glass,
    }
    for p in audio.NORMAL_PROMPTS:
        cos[p] = normal
    return cos


def case_is_anomaly_fires_on_anomaly():
    enc = _FakeCLAPScores(_cos_map(anom_alarm=0.40, normal=0.10))
    fired, cls, scores = audio.is_anomaly(np.zeros(8000, np.float32), _SR, enc)
    assert fired and cls == "alarm", (fired, cls)
    assert abs(scores["s_anom"] - 0.40) < 1e-5 and abs(scores["s_norm"] - 0.10) < 1e-5
    assert abs(scores["margin"] - 0.30) < 1e-5
    print("  case is_anomaly_fires_on_anomaly: OK")


def case_is_anomaly_rejects_normal():
    # a benign sound: every anomaly prompt scores low, "people talking" high
    cos = _cos_map(anom_alarm=0.15, anom_baby=0.12, anom_glass=0.12, normal=0.10)
    cos["people talking"] = 0.50
    enc = _FakeCLAPScores(cos)
    fired, cls, scores = audio.is_anomaly(np.zeros(8000, np.float32), _SR, enc)
    assert not fired, (fired, scores)
    assert scores["margin"] < 0.0  # normal side wins
    print("  case is_anomaly_rejects_normal: OK")


def case_is_anomaly_delta_threshold():
    enc = _FakeCLAPScores(_cos_map(anom_alarm=0.30, normal=0.25))  # margin = 0.05
    assert audio.is_anomaly(np.zeros(8, np.float32), _SR, enc, delta=0.0)[0] is True
    assert audio.is_anomaly(np.zeros(8, np.float32), _SR, enc, delta=0.10)[0] is False
    print("  case is_anomaly_delta_threshold: OK")


def case_is_anomaly_tau_floor():
    # anomaly side wins (margin +0.05) but the absolute cosine is tiny
    enc = _FakeCLAPScores(_cos_map(anom_alarm=0.05, normal=0.00))
    assert audio.is_anomaly(np.zeros(8, np.float32), _SR, enc, tau_abs=0.0)[0] is True
    assert audio.is_anomaly(np.zeros(8, np.float32), _SR, enc, tau_abs=0.20)[0] is False
    print("  case is_anomaly_tau_floor: OK")


def case_is_anomaly_scores_keys():
    enc = _FakeCLAPScores(_cos_map())
    _, _, scores = audio.is_anomaly(np.zeros(8, np.float32), _SR, enc)
    for k in ("s_anom", "s_norm", "margin", *audio.ANOMALY_CLASSES):
        assert k in scores, k
    print("  case is_anomaly_scores_keys: OK")


def case_class_maps_present_and_consistent():
    assert audio.ANOMALY_CLASSES == ("baby_cry", "alarm", "glass_break")
    for c in audio.ANOMALY_CLASSES:
        assert c in audio.CLASS_TO_OBJECT and audio.CLASS_TO_OBJECT[c]
        assert c in audio.CLASS_TO_CLAP_PROMPT and audio.CLASS_TO_CLAP_PROMPT[c]
    # the M0c demo mapping the plan pins
    assert audio.CLASS_TO_OBJECT["baby_cry"] == "crib"
    print("  case class_maps_present_and_consistent: OK")


# ----------------------------------------------------------------------


def case_rirgrid_cell_geodesics_optional():
    # a grid built without geodesics exposes None (back-compat for legacy grids)
    cells = np.array([[0.0, 1.5, 0.0], [2.0, 1.5, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.0) for _ in range(2)], axis=0)
    g = audio.RIRGrid(cells, np.zeros(3, np.float32), irs, _SR, "s")
    assert g.cell_geodesics is None
    # wrong-length geodesics must raise
    try:
        audio.RIRGrid(cells, np.zeros(3, np.float32), irs, _SR, "s",
                      cell_geodesics=[1.0])
        assert False, "expected length mismatch to raise"
    except ValueError:
        pass
    print("  case rirgrid_cell_geodesics_optional: OK")


def case_rirgrid_cell_energies_from_irs():
    cells = np.array([[0.0, 1.5, 0.0], [2.0, 1.5, 0.0]], dtype=np.float32)
    irs = np.zeros((2, 2, 4), dtype=np.float32)
    irs[0, 0, 0] = 1.0          # energy 1.0
    irs[1, 0, 0] = 2.0          # energy 4.0
    g = audio.RIRGrid(cells, np.zeros(3, np.float32), irs, _SR, "s")
    e = g.cell_energies
    assert e.shape == (2,), e.shape
    assert abs(e[0] - 1.0) < 1e-9 and abs(e[1] - 4.0) < 1e-9, e
    print("  case rirgrid_cell_energies_from_irs: OK")


def case_grid_save_load_geodesics_roundtrip():
    cells = np.array([[0.0, 1.5, 0.0], [2.0, 1.5, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.2), _binaural_for_azimuth(-0.2)], axis=0)
    geo = np.array([1.2, 3.4], dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "g.npz")
        audio.save_rir_grid(path, cell_positions=cells,
                            source_position=np.zeros(3, np.float32), irs=irs,
                            sample_rate=_SR, scene_id="s", cell_geodesics=geo)
        raw = np.load(path)
        assert "cell_geodesics" in raw.files
        g2 = audio.RIRGrid.load(path)
    assert g2.cell_geodesics is not None and np.allclose(g2.cell_geodesics, geo)
    # length-mismatched geodesics at save time must raise
    try:
        with tempfile.TemporaryDirectory() as td:
            audio.save_rir_grid(os.path.join(td, "g.npz"), cell_positions=cells,
                                source_position=np.zeros(3, np.float32), irs=irs,
                                sample_rate=_SR, scene_id="s", cell_geodesics=[1.0])
        assert False, "expected save length mismatch to raise"
    except ValueError:
        pass
    print("  case grid_save_load_geodesics_roundtrip: OK")


def case_grid_load_legacy_without_geodesics():
    # a grid saved WITHOUT geodesics still loads → cell_geodesics is None
    cells = np.array([[0.0, 1.5, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.0)], axis=0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "g.npz")
        audio.save_rir_grid(path, cell_positions=cells,
                            source_position=np.zeros(3, np.float32), irs=irs,
                            sample_rate=_SR, scene_id="s")
        g2 = audio.RIRGrid.load(path)
    assert g2.cell_geodesics is None
    print("  case grid_load_legacy_without_geodesics: OK")


def _navmesh_floor_grid():
    """A grid as ``render_rir_grid`` actually emits one: cells carry the listener
    pose's NAVMESH y, not navmesh_y + ear_height.

    The renderer sets ``st.position = cell`` — an agent state, which must lie on
    the navmesh — and puts the ear at ``spec.position = ear``, a sensor-LOCAL
    offset that never enters ``cell_positions``. It offsets the SOURCE by ``ear``
    explicitly (``setAudioSourceTransform(source_pt + ear)``) because that one is
    a world transform; the listener cells get no such treatment.

    TEEsavR23oF geometry: ground navmesh y=0.163, upstairs y=3.163.
    """
    cells = np.array([[0.0, 0.163, 0.0], [2.0, 0.163, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.0) for _ in range(2)], axis=0)
    return audio.RIRGrid(cells, np.array([0.0, 0.163, 0.0], np.float32), irs,
                         _SR, "TEEsavR23oF", ear_height_m=1.5)


def case_nearest_guard_resolves_pose_on_rendered_floor():
    # The ADR-0003 guard must ADMIT an agent standing on the very floor the grid
    # was rendered on, and EXCLUDE one a storey up. Against a real grid it did
    # neither: it compared cell_y to agent_y + ear_height_m, so a same-floor pose
    # read as 1.5 m off-floor, every cell on every floor fell outside max_dy=1.0,
    # and the render returned silence — audio_energy_max=0.0 in 8/8 episodes of
    # runs/anomresp-bed-s{1,3}, with the controller never firing.
    #
    # Both halves are one spec on purpose: the accept half alone would also pass
    # if someone simply widened max_dy past 1.5, which would readmit the upstairs
    # poses ADR-0003 exists to exclude. Only the comparison itself satisfies both.
    g = _navmesh_floor_grid()
    _, idx, _ = g.nearest(np.array([1.9, 0.163, 0.0], np.float32), max_dy=1.0)
    assert idx == 1, f"a pose on the rendered floor must resolve; got {idx}"
    try:
        g.nearest(np.array([1.9, 3.163, 0.0], np.float32), max_dy=1.0)
        assert False, "a pose one floor up must NOT resolve to a ground-floor cell"
    except audio.OutOfCoverageError:
        pass
    print("  case nearest_guard_resolves_pose_on_rendered_floor: OK")


def case_nearest_guard_works_without_ear_height():
    # The floor guard asks "is the agent on the floor this grid covers?" — a
    # navmesh-vs-navmesh question that needs no ear height, so a legacy grid
    # carrying none is guarded just as well. ear_height_m is informational
    # metadata; it is deliberately NOT an input to the comparison.
    #
    # This replaces a test that asserted the opposite (guard on an ear-height-less
    # grid must raise). That spec followed from the ear-offset comparison, which
    # silenced every floor of every real grid. Guarding an unguarded lookup is
    # still the failure to prevent — under this convention it simply cannot arise.
    cells = np.array([[0.0, 0.163, 0.0], [2.0, 0.163, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.0) for _ in range(2)], axis=0)
    legacy = audio.RIRGrid(cells, np.zeros(3, np.float32), irs, _SR, "s")
    assert legacy.ear_height_m is None
    _, idx, _ = legacy.nearest(np.array([1.9, 0.163, 0.0], np.float32), max_dy=1.0)
    assert idx == 1, f"same-floor pose must resolve on a legacy grid; got {idx}"
    try:
        legacy.nearest(np.array([1.9, 3.163, 0.0], np.float32), max_dy=1.0)
        assert False, "a pose one floor up must NOT resolve, ear height or not"
    except audio.OutOfCoverageError:
        pass
    print("  case nearest_guard_works_without_ear_height: OK")


def case_grid_ear_height_roundtrips():
    # The listener grid sits at EAR height (navmesh_y + ear_height); the runtime
    # agent pose is at navmesh y. Without the offset a same-floor pose and a
    # one-floor-up pose are BOTH ~1.5 m from a cell in y and are indistinguishable
    # (ADR-0003). So the grid must carry the ear height it was rendered at.
    cells = np.array([[0.0, 1.663, 0.0], [2.0, 1.663, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.0) for _ in range(2)], axis=0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "g.npz")
        audio.save_rir_grid(path, cell_positions=cells,
                            source_position=np.zeros(3, np.float32), irs=irs,
                            sample_rate=_SR, scene_id="s", ear_height_m=1.5)
        g = audio.RIRGrid.load(path)
    assert g.ear_height_m is not None, "ear_height_m must survive the round-trip"
    assert abs(g.ear_height_m - 1.5) < 1e-6, g.ear_height_m
    print("  case grid_ear_height_roundtrips: OK")


def case_grid_load_legacy_without_ear_height():
    # a grid saved WITHOUT ear height still loads → ear_height_m is None, and the
    # on-disk format is unchanged (mirrors cell_geodesics). A None ear height means
    # the floor guard CANNOT engage, which is what keeps legacy grids byte-identical.
    cells = np.array([[0.0, 1.5, 0.0]], dtype=np.float32)
    irs = np.stack([_binaural_for_azimuth(0.0)], axis=0)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "g.npz")
        audio.save_rir_grid(path, cell_positions=cells,
                            source_position=np.zeros(3, np.float32), irs=irs,
                            sample_rate=_SR, scene_id="s")
        raw = np.load(path)
        assert "ear_height_m" not in raw.files, \
            f"legacy save must not add keys; got {raw.files}"
        g = audio.RIRGrid.load(path)
    assert g.ear_height_m is None
    print("  case grid_load_legacy_without_ear_height: OK")


def case_render_rir_grid_persists_ear_height():
    # $0 static source guard (the test_active_goal_noop pattern): render_rir_grid
    # is a soundspaces-env script we cannot execute here, but if it does not pass
    # ear_height_m to save_rir_grid then EVERY grid loads with ear_height_m=None
    # and the ADR-0003 floor guard can never engage — it would raise ValueError
    # on a grid that looks fine. The guard is only as real as the render that
    # feeds it.
    src = (_EMB_DIR / "scripts" / "render_rir_grid.py").read_text()
    assert "--ear-height" in src, "render_rir_grid must still take --ear-height"
    assert "ear_height_m=" in src, (
        "render_rir_grid must persist ear_height_m via save_rir_grid, or no "
        "rendered grid can ever support the ADR-0003 floor guard")
    print("  case render_rir_grid_persists_ear_height: OK")


def main() -> int:
    cases = [
        case_grid_nearest_lookup,
        case_grid_save_load_roundtrip,
        case_grid_save_variable_length_pads,
        case_rirgrid_cell_geodesics_optional,
        case_rirgrid_cell_energies_from_irs,
        case_grid_save_load_geodesics_roundtrip,
        case_grid_load_legacy_without_geodesics,
        case_grid_ear_height_roundtrips,
        case_grid_load_legacy_without_ear_height,
        case_nearest_guard_resolves_pose_on_rendered_floor,
        case_nearest_guard_works_without_ear_height,
        case_render_rir_grid_persists_ear_height,
        case_render_shape_and_convolution,
        case_render_rms_monotone_with_distance,
        case_doa_recovers_right_and_left,
        case_doa_front_centered_is_zero,
        case_doa_sign_from_ild_when_itd_zero,
        case_doa_ild_overrides_subresolution_itd,
        case_lateral_sign_helper,
        case_classify_anomaly_argmax,
        case_is_anomaly_fires_on_anomaly,
        case_is_anomaly_rejects_normal,
        case_is_anomaly_delta_threshold,
        case_is_anomaly_tau_floor,
        case_is_anomaly_scores_keys,
        case_class_maps_present_and_consistent,
    ]
    print(f"running {len(cases)} audio.py cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
