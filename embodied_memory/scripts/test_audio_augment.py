"""
TDD for embodied_memory/audio.py :: augment_clip (P2.1).

augment_clip is the ONE public seam for deterministic clip augmentation used by
the Gate-0.3 calibration diagnostic so the CLAP anomaly gate is calibrated on the
distribution it actually hears at runtime (background-mixed, reverbed, pitch/time-
jittered, RIR-convolved) — removing the clean->convolved calibration cliff.

The transforms are PURE (no RNG that isn't seeded in the spec, no CLAP, no
simulator): tests assert input->output behavior at this seam, so they survive a
refactor of the internal sub-transforms.

Loaded via importlib (like test_audio.py) so we never trigger the faiss-importing
embodied_memory/__init__.py.

Run: PYTHONPATH=. /opt/anaconda3/envs/ltm-embodied/bin/python \
        embodied_memory/scripts/test_audio_augment.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_EMB_DIR = Path(__file__).resolve().parent.parent  # …/embodied_memory


def _load_file_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


audio = _load_file_as("embodied_memory._audio_under_test_aug", _EMB_DIR / "audio.py")

_SR = 16000


def _tone(n: int = 8000, freq: float = 440.0, sr: int = _SR) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _noise(n: int = 8000, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1


# ----------------------------------------------------------------------
# shape / sample-rate preservation (user story 25)
# ----------------------------------------------------------------------
def case_empty_spec_is_identity():
    clip = _tone()
    out = audio.augment_clip(clip, _SR, audio.AugmentSpec())
    assert out.shape == clip.shape, (out.shape, clip.shape)
    assert out.dtype == np.float32, out.dtype
    assert np.allclose(out, clip, atol=1e-6), float(np.max(np.abs(out - clip)))


def case_shape_and_dtype_preserved_under_all_transforms():
    clip = _tone()
    spec = audio.AugmentSpec(
        seed=3, background=_noise(), snr_db=10.0, pitch_semitones=2.0,
        time_shift_frac=0.1, reverb_decay=0.15, rir=np.array([1.0, 0.3, 0.1], dtype=np.float32))
    out = audio.augment_clip(clip, _SR, spec)
    assert out.ndim == 1, out.shape
    assert out.shape == clip.shape, (out.shape, clip.shape)
    assert out.dtype == np.float32, out.dtype
    assert np.all(np.isfinite(out))


# ----------------------------------------------------------------------
# background mix at a target SNR (user stories 21/24)
# ----------------------------------------------------------------------
def case_background_mix_raises_energy():
    clip = _tone()
    quiet = audio.rms(clip)
    out = audio.augment_clip(clip, _SR, audio.AugmentSpec(background=_noise(), snr_db=0.0))
    assert audio.rms(out) > quiet, (audio.rms(out), quiet)


def case_background_mix_hits_target_snr():
    # After mixing at target SNR, rms(signal)/rms(added-background) ~= 10^(snr/20).
    clip = _tone()
    bg = _noise(seed=5)
    snr_db = 6.0
    out = audio.augment_clip(clip, _SR, audio.AugmentSpec(background=bg, snr_db=snr_db))
    added = out - clip                       # only the scaled background was added
    ratio_db = 20.0 * np.log10(audio.rms(clip) / (audio.rms(added) + 1e-12))
    assert abs(ratio_db - snr_db) < 1.0, ratio_db


def case_lower_snr_adds_more_background():
    clip = _tone()
    bg = _noise(seed=7)
    loud_bg = audio.augment_clip(clip, _SR, audio.AugmentSpec(background=bg, snr_db=-6.0))
    soft_bg = audio.augment_clip(clip, _SR, audio.AugmentSpec(background=bg, snr_db=12.0))
    assert audio.rms(loud_bg) > audio.rms(soft_bg)


# ----------------------------------------------------------------------
# time-shift (circular: preserves energy + length)
# ----------------------------------------------------------------------
def case_time_shift_transforms_but_preserves_energy():
    clip = _tone()
    out = audio.augment_clip(clip, _SR, audio.AugmentSpec(time_shift_frac=0.25))
    assert out.shape == clip.shape
    assert not np.allclose(out, clip)                 # the signal moved
    assert abs(audio.rms(out) - audio.rms(clip)) < 1e-5   # circular shift keeps energy


# ----------------------------------------------------------------------
# pitch-shift
# ----------------------------------------------------------------------
def case_pitch_shift_transforms_signal():
    clip = _tone(freq=300.0)
    out = audio.augment_clip(clip, _SR, audio.AugmentSpec(pitch_semitones=4.0))
    assert out.shape == clip.shape
    assert not np.allclose(out, clip)
    # a pitch shift up moves spectral energy to a higher dominant bin
    def _dom(x):
        mag = np.abs(np.fft.rfft(x))
        return int(np.argmax(mag[1:]) + 1)
    assert _dom(out) > _dom(clip), (_dom(out), _dom(clip))


# ----------------------------------------------------------------------
# reverb / room-size jitter
# ----------------------------------------------------------------------
def case_reverb_transforms_and_preserves_length():
    clip = _tone()
    out = audio.augment_clip(clip, _SR, audio.AugmentSpec(seed=2, reverb_decay=0.2))
    assert out.shape == clip.shape
    assert not np.allclose(out, clip)


def case_reverb_seed_changes_output():
    clip = _tone()
    a = audio.augment_clip(clip, _SR, audio.AugmentSpec(seed=1, reverb_decay=0.2))
    b = audio.augment_clip(clip, _SR, audio.AugmentSpec(seed=2, reverb_decay=0.2))
    assert not np.allclose(a, b)


# ----------------------------------------------------------------------
# RIR convolution keeps the clip MONO (flows through render path unchanged)
# ----------------------------------------------------------------------
def case_rir_convolution_preserves_mono_shape():
    clip = _tone()
    ir_binaural = np.stack([np.array([1.0, 0.4, 0.2]), np.array([0.9, 0.3, 0.1])]).astype(np.float32)
    out = audio.augment_clip(clip, _SR, audio.AugmentSpec(rir=ir_binaural))
    assert out.ndim == 1 and out.shape == clip.shape, out.shape
    assert not np.allclose(out, clip)


# ----------------------------------------------------------------------
# determinism (user story 22) + order-stable composition
# ----------------------------------------------------------------------
def case_deterministic_given_spec():
    clip = _tone()
    spec = audio.AugmentSpec(seed=9, background=_noise(seed=9), snr_db=3.0,
                             pitch_semitones=1.5, time_shift_frac=0.05, reverb_decay=0.1)
    a = audio.augment_clip(clip, _SR, spec)
    b = audio.augment_clip(clip, _SR, spec)
    assert np.array_equal(a, b)


def case_composition_changes_signal_beyond_any_single_transform():
    clip = _tone()
    combined = audio.augment_clip(
        clip, _SR, audio.AugmentSpec(pitch_semitones=3.0, time_shift_frac=0.2))
    only_pitch = audio.augment_clip(clip, _SR, audio.AugmentSpec(pitch_semitones=3.0))
    only_shift = audio.augment_clip(clip, _SR, audio.AugmentSpec(time_shift_frac=0.2))
    assert not np.allclose(combined, only_pitch)
    assert not np.allclose(combined, only_shift)


def main() -> int:
    cases = [
        case_empty_spec_is_identity,
        case_shape_and_dtype_preserved_under_all_transforms,
        case_background_mix_raises_energy,
        case_background_mix_hits_target_snr,
        case_lower_snr_adds_more_background,
        case_time_shift_transforms_but_preserves_energy,
        case_pitch_shift_transforms_signal,
        case_reverb_transforms_and_preserves_length,
        case_reverb_seed_changes_output,
        case_rir_convolution_preserves_mono_shape,
        case_deterministic_given_spec,
        case_composition_changes_signal_beyond_any_single_transform,
    ]
    print(f"running {len(cases)} augment_clip cases…")
    for c in cases:
        c()
        print(f"  {c.__name__}: OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
