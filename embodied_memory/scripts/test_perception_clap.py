"""
TDD test for ``perception.CLAPAudioEncoder`` — the CLAP audio/text wrapper
(M0b). CLAP is the anomaly CLASSIFIER only (cry/alarm/glass); retrieval stays
on the proven SBERT path.

The heavy ``laion/clap-htsat-fused`` model is lazy-loaded; these cases stay
torch-free by testing the two pieces that have real logic:
  * ``_to_mono_48k`` — stereo→mono + resample to CLAP's 48 kHz;
  * ``encode_audio`` / ``encode_text`` normalize a 512-d feature and route the
    waveform through ``_to_mono_48k`` (the model forward ``_audio_features`` /
    ``_text_features`` is monkeypatched — that is the heavy seam, exercised for
    real only in the on-RACE / downloaded-model smoke).

perception.py is loaded standalone (it lazy-imports torch/open_clip/transformers)
so the faiss-importing package ``__init__`` is never triggered.

    python embodied_memory/scripts/test_perception_clap.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_EMB_DIR = Path(__file__).resolve().parent.parent


def _load_file_as(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


perception = _load_file_as("embodied_memory._perception_under_test",
                           _EMB_DIR / "perception.py")
CLAPAudioEncoder = perception.CLAPAudioEncoder


def case_to_mono_48k_resamples():
    enc = CLAPAudioEncoder()
    # 16 kHz mono, 1000 samples → 48 kHz → exactly 3000 samples (up=3, down=1).
    mono16 = np.sin(np.linspace(0, 10, 1000)).astype(np.float32)
    out = enc._to_mono_48k(mono16, 16000)
    assert out.ndim == 1 and out.shape[0] == 3000, out.shape
    assert out.dtype == np.float32, out.dtype
    print("  case to_mono_48k_resamples (1000@16k → 3000@48k): OK")


def case_to_mono_48k_stereo_to_mono():
    enc = CLAPAudioEncoder()
    # already 48 kHz stereo (2, L) → mono length L, no resample.
    stereo = np.stack([np.ones(500, np.float32), 3.0 * np.ones(500, np.float32)])
    out = enc._to_mono_48k(stereo, 48000)
    assert out.shape == (500,), out.shape
    assert np.allclose(out, 2.0), "mono should be the per-sample channel mean"
    print("  case to_mono_48k_stereo_to_mono: OK")


def case_encode_audio_normalizes_and_routes():
    enc = CLAPAudioEncoder()
    seen = {}

    def fake_audio_features(mono):
        seen["len"] = mono.shape[0]
        v = np.zeros(512, dtype=np.float32)
        v[0] = 3.0
        v[1] = 4.0  # ‖v‖ = 5 → normalized to (0.6, 0.8, 0, …)
        return v

    enc._audio_features = fake_audio_features
    out = enc.encode_audio(np.ones(1000, np.float32), 16000)
    assert out.shape == (512,), out.shape
    assert abs(np.linalg.norm(out) - 1.0) < 1e-5, np.linalg.norm(out)
    assert abs(out[0] - 0.6) < 1e-5 and abs(out[1] - 0.8) < 1e-5, out[:2]
    assert seen["len"] == 3000, f"waveform not resampled before model: {seen}"
    print("  case encode_audio_normalizes_and_routes: OK")


def case_encode_text_normalizes():
    enc = CLAPAudioEncoder()
    enc._text_features = lambda text: np.array([0.0, 0.0, 5.0] + [0.0] * 509, np.float32)
    out = enc.encode_text("a baby crying")
    assert out.shape == (512,), out.shape
    assert abs(np.linalg.norm(out) - 1.0) < 1e-5
    assert abs(out[2] - 1.0) < 1e-5
    print("  case encode_text_normalizes: OK")


def case_embed_dim_attr():
    assert CLAPAudioEncoder.EMBED_DIM == 512
    assert CLAPAudioEncoder.TARGET_SR == 48000
    print("  case embed_dim_attr: OK")


def main() -> int:
    cases = [
        case_to_mono_48k_resamples,
        case_to_mono_48k_stereo_to_mono,
        case_encode_audio_normalizes_and_routes,
        case_encode_text_normalizes,
        case_embed_dim_attr,
    ]
    print(f"running {len(cases)} CLAPAudioEncoder cases…")
    for c in cases:
        c()
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
