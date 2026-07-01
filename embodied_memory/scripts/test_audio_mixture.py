"""
TDD for the Phase-1 mixture render — a continuous non-directional (diotic)
background bed added to render_step_audio so EVERY scene is a blend of background
noise + the abnormal sound (the discrimination premise). Option B (no second RIR
grid — a second spatial source would corrupt the anomaly's lateral_sign DOA cue).

Byte-identity FIRST: bg_gain==0.0 OR bg_clip_norm is None => output IDENTICAL to
the current single-source render (objectnav/revisit/multion + existing
single-source audiogoal unchanged).

Run: KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. \
       /opt/anaconda3/envs/ltm-embodied/bin/python \
       embodied_memory/scripts/test_audio_mixture.py
"""
from __future__ import annotations

import sys

import numpy as np

from embodied_memory import audio
from embodied_memory import audio_task as at
from embodied_memory.audio import RIRGrid, render_at_pose, rms


def _grid(N=3, T=8):
    cell_pos = np.stack([np.linspace(0, 4, N), np.zeros(N), np.zeros(N)], axis=1)
    irs = np.zeros((N, 2, T), dtype=np.float32)
    for i in range(N):
        irs[i, 0, 0] = 0.6 / (i + 1)          # left  direct path
        irs[i, 1, 0] = 0.6 / (i + 1) * 0.5    # right (asymmetric -> spatial anomaly)
    return RIRGrid(cell_pos, [0.0, 0.0, 0.0], irs, sample_rate=16000, scene_id="t")


_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
_AN = (0.2 * np.sin(np.linspace(0, 30, 40))).astype(np.float32)
_BG = (0.2 * np.sin(np.linspace(0, 12, 28) + 1.0)).astype(np.float32)


# ----------------------------------------------------------------------
# diotic_collapse
# ----------------------------------------------------------------------
def case_diotic_collapse_equalizes_channels():
    d = audio.diotic_collapse(np.array([[1.0, 2, 3], [5, 6, 7]], dtype=np.float32))
    assert d.shape == (2, 3)
    assert np.allclose(d[0], d[1]) and np.allclose(d[0], [3.0, 4.0, 5.0])


# ----------------------------------------------------------------------
# byte-identity — the load-bearing regression
# ----------------------------------------------------------------------
def case_bytewise_identical_no_bed_default():
    g = _grid(); cfg = at.AudioTaskConfig(enabled=True, t_anom=5)
    # pre-t_anom -> None (unchanged)
    assert at.render_step_audio(g, _POS, _AN, 2, cfg) is None
    # post-t_anom -> exactly render_at_pose (unchanged)
    out = at.render_step_audio(g, _POS, _AN, 9, cfg)
    assert np.array_equal(out, render_at_pose(g, _POS, _AN))


def case_bytewise_identical_bed_present_but_gain_zero():
    g = _grid(); cfg = at.AudioTaskConfig(enabled=True, t_anom=5, bg_gain=0.0)
    assert at.render_step_audio(g, _POS, _AN, 2, cfg, bg_clip_norm=_BG) is None
    out = at.render_step_audio(g, _POS, _AN, 9, cfg, bg_clip_norm=_BG)
    assert np.array_equal(out, render_at_pose(g, _POS, _AN))


def case_bytewise_identical_gain_set_but_no_bed_clip():
    g = _grid(); cfg = at.AudioTaskConfig(enabled=True, t_anom=5, bg_gain=1.0)
    assert at.render_step_audio(g, _POS, _AN, 2, cfg, bg_clip_norm=None) is None
    out = at.render_step_audio(g, _POS, _AN, 9, cfg, bg_clip_norm=None)
    assert np.array_equal(out, render_at_pose(g, _POS, _AN))


# ----------------------------------------------------------------------
# mixture behaviour
# ----------------------------------------------------------------------
def case_bed_plays_before_t_anom_and_is_diotic():
    g = _grid(); cfg = at.AudioTaskConfig(enabled=True, t_anom=5, bg_gain=1.0)
    out = at.render_step_audio(g, _POS, _AN, 2, cfg, bg_clip_norm=_BG)   # pre-t_anom
    assert out is not None
    assert np.allclose(out[0], out[1]), "bed must be diotic (no lateral cue)"
    assert float(rms(out)) > 0.0


def case_anomaly_adds_energy_after_t_anom():
    g = _grid(); cfg = at.AudioTaskConfig(enabled=True, t_anom=5, bg_gain=1.0)
    bed_only = at.render_step_audio(g, _POS, _AN, 2, cfg, bg_clip_norm=_BG)
    mixed = at.render_step_audio(g, _POS, _AN, 9, cfg, bg_clip_norm=_BG)   # post-t_anom
    assert mixed is not None
    assert float(rms(mixed)) > float(rms(bed_only)), "anomaly must add energy over the bed"
    # the spatial anomaly breaks the bed's L==R symmetry
    assert not np.allclose(mixed[0], mixed[1])


def case_length_align_bed_shorter_and_longer():
    g = _grid(); cfg = at.AudioTaskConfig(enabled=True, t_anom=1, bg_gain=0.5)
    short_bed = (0.1 * np.sin(np.linspace(0, 3, 5))).astype(np.float32)
    long_bed = (0.1 * np.sin(np.linspace(0, 50, 400))).astype(np.float32)
    L = render_at_pose(g, _POS, _AN).shape[-1]
    for bed in (short_bed, long_bed):
        out = at.render_step_audio(g, _POS, _AN, 9, cfg, bg_clip_norm=bed)
        assert out.shape[-1] == L, (out.shape, L)


def case_load_background_clip_none_path():
    # the byte-identity witness for Phase 1b: no background_clip_path -> the bed
    # loader returns None (absent, NOT a synthetic burst) -> render seam falls back
    # to the single-source path. Uses a bare instance (habitat imports lazily).
    from embodied_memory.habitat_env import HabitatObjectNavSource
    s = HabitatObjectNavSource.__new__(HabitatObjectNavSource)
    s._background_clip_path = None
    assert s._load_background_clip() is None


def case_resolve_bg_clip_explicit_path_wins():
    # the driver passes --background-clip data/benign_audio/<bed>.wav; the explicit
    # path must win over the (anomaly-defaulted) dir so the bed loads from the right
    # place (guards the --background-dir defaults-to-anomaly_audio trap).
    assert at.resolve_anomaly_clip("vacuum", "data/benign_audio/vacuum.wav",
                                   "data/anomaly_audio") == "data/benign_audio/vacuum.wav"
    assert at.resolve_anomaly_clip(None, None, "data/anomaly_audio") is None


def main() -> int:
    cases = [
        case_diotic_collapse_equalizes_channels,
        case_load_background_clip_none_path,
        case_resolve_bg_clip_explicit_path_wins,
        case_bytewise_identical_no_bed_default,
        case_bytewise_identical_bed_present_but_gain_zero,
        case_bytewise_identical_gain_set_but_no_bed_clip,
        case_bed_plays_before_t_anom_and_is_diotic,
        case_anomaly_adds_energy_after_t_anom,
        case_length_align_bed_shorter_and_longer,
    ]
    print(f"running {len(cases)} audio_mixture cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
