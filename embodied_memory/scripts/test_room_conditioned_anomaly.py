"""
TDD for the scene-conditioned anomaly gate (P3.1 / ADR-0002).

Whether a heard sound is an anomaly depends on the ROOM it is heard in (running
water is normal in a bathroom, anomalous in a bedroom). The decision is a PURE
function of ``(sound_class, detected_room, ROOM_PRIOR)`` — testable without CLAP
or the simulator (user story 32). ``is_anomaly`` gains an OPTIONAL room-conditioned
mode; with both room args absent the context-free path is byte-identical (user
story 19 / 30).

Loaded via importlib (like test_audio.py) so we never trigger the faiss-importing
embodied_memory/__init__.py.

Run: PYTHONPATH=. /opt/anaconda3/envs/ltm-embodied/bin/python \
        embodied_memory/scripts/test_room_conditioned_anomaly.py
"""
from __future__ import annotations

import importlib.util
import math
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

audio = _load_file_as("embodied_memory._audio_under_test_room", _EMB_DIR / "audio.py")
# room_resolver is dependency-free; load it directly so this test never needs faiss.
room_resolver = _load_file_as("embodied_memory._room_resolver_under_test", _EMB_DIR / "room_resolver.py")

_SR = 16000


class _FakeCLAPScores:
    """CLAP stand-in whose audio↔text cosine is fully controllable (see test_audio)."""
    def __init__(self, text_cos):
        self._text_cos = dict(text_cos)

    def encode_audio(self, waveform, sample_rate):
        v = np.zeros(8, dtype=np.float32); v[0] = 1.0; return v

    def encode_text(self, text):
        c = float(max(-1.0, min(1.0, self._text_cos.get(text, 0.0))))
        v = np.zeros(8, dtype=np.float32)
        v[0] = c; v[1] = math.sqrt(max(0.0, 1.0 - c * c)); return v


def _ambiguous_cos(water=0.1, hum=0.1, normal=0.1):
    cos = {audio.CLASS_TO_CLAP_PROMPT["running_water"]: water,
           audio.CLASS_TO_CLAP_PROMPT["appliance_hum"]: hum}
    for p in audio.NORMAL_PROMPTS:
        cos[p] = normal
    return cos


# ----------------------------------------------------------------------
# ROOM_PRIOR / vocabulary sanity
# ----------------------------------------------------------------------
def case_room_prior_covers_taxonomy_and_flips_a_sound():
    prior = audio.ROOM_PRIOR
    # water is EXPECTED somewhere (bathroom) and UNEXPECTED somewhere (bedroom)
    assert "running_water" in prior["bathroom"]
    assert "running_water" not in prior["bedroom"]
    # keys are room-types from the shared taxonomy (interchangeable with the CLIP rooms)
    for room in prior:
        assert room in room_resolver.ROOM_TEXT_PROMPTS, room


# ----------------------------------------------------------------------
# pure decision (user story 32)
# ----------------------------------------------------------------------
def case_unexpected_for_room_is_anomalous():
    assert audio.room_conditioned_anomaly("running_water", "bedroom", audio.ROOM_PRIOR) is True


def case_expected_for_room_is_normal():
    assert audio.room_conditioned_anomaly("running_water", "bathroom", audio.ROOM_PRIOR) is False


def case_unknown_room_abstains():
    assert audio.room_conditioned_anomaly("running_water", None, audio.ROOM_PRIOR) is None


def case_uncovered_room_abstains():
    # a room-type with no normality knowledge in the prior -> cannot scene-condition
    assert audio.room_conditioned_anomaly("running_water", "garage", audio.ROOM_PRIOR) is None


def case_empty_class_abstains():
    assert audio.room_conditioned_anomaly(None, "bedroom", audio.ROOM_PRIOR) is None
    assert audio.room_conditioned_anomaly("", "bedroom", audio.ROOM_PRIOR) is None


# ----------------------------------------------------------------------
# is_anomaly room-conditioned mode: the same clip flips on the room
# ----------------------------------------------------------------------
def case_same_clip_flips_verdict_on_room():
    enc = _FakeCLAPScores(_ambiguous_cos(water=0.40, normal=0.10))
    wav = np.zeros(8000, np.float32)
    # bathroom: running water is EXPECTED -> not an anomaly, no interrupt
    fired_bath, cls_b, _ = audio.is_anomaly(
        wav, _SR, enc, classes=audio.AMBIGUOUS_CLASSES,
        detected_room="bathroom", room_prior=audio.ROOM_PRIOR)
    # bedroom: running water is UNEXPECTED -> anomaly, interrupt
    fired_bed, cls_r, _ = audio.is_anomaly(
        wav, _SR, enc, classes=audio.AMBIGUOUS_CLASSES,
        detected_room="bedroom", room_prior=audio.ROOM_PRIOR)
    assert cls_b == "running_water" and cls_r == "running_water"
    assert fired_bath is False, "room-normal sound must NOT interrupt"
    assert fired_bed is True, "room-anomalous sound MUST interrupt"


def case_room_conditioning_falls_back_when_room_unknown():
    # detected_room None -> the pure decision abstains -> keep the context-free verdict
    enc = _FakeCLAPScores(_ambiguous_cos(water=0.40, normal=0.10))
    wav = np.zeros(8000, np.float32)
    fired, _, _ = audio.is_anomaly(
        wav, _SR, enc, classes=audio.AMBIGUOUS_CLASSES,
        detected_room=None, room_prior=audio.ROOM_PRIOR)
    ctx_free, _, _ = audio.is_anomaly(wav, _SR, enc, classes=audio.AMBIGUOUS_CLASSES)
    assert fired == ctx_free


# ----------------------------------------------------------------------
# default path byte-identical (user stories 19 / 30)
# ----------------------------------------------------------------------
def case_default_path_byte_identical():
    enc = _FakeCLAPScores(_ambiguous_cos(water=0.40, normal=0.10))
    wav = np.zeros(8000, np.float32)
    a = audio.is_anomaly(wav, _SR, enc, classes=audio.AMBIGUOUS_CLASSES)
    b = audio.is_anomaly(wav, _SR, enc, classes=audio.AMBIGUOUS_CLASSES,
                         detected_room=None, room_prior=None)
    assert a[0] == b[0] and a[1] == b[1]
    assert a[2].keys() == b[2].keys(), (a[2].keys(), b[2].keys())
    # no room_verdict key leaks into the context-free scores
    assert "room_verdict" not in a[2]
    for k in a[2]:
        assert abs(float(a[2][k]) - float(b[2][k])) < 1e-9, k


def main() -> int:
    cases = [
        case_room_prior_covers_taxonomy_and_flips_a_sound,
        case_unexpected_for_room_is_anomalous,
        case_expected_for_room_is_normal,
        case_unknown_room_abstains,
        case_uncovered_room_abstains,
        case_empty_class_abstains,
        case_same_clip_flips_verdict_on_room,
        case_room_conditioning_falls_back_when_room_unknown,
        case_default_path_byte_identical,
    ]
    print(f"running {len(cases)} room-conditioned-anomaly cases…")
    for c in cases:
        c()
        print(f"  {c.__name__}: OK")
    print("ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
