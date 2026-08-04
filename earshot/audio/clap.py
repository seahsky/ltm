"""CLAP: the open-set normal-versus-anomaly gate and its prompt banks.

Carried from ``audio.py`` (``classify_anomaly``, ``is_anomaly``, the prompt banks and
the calibrated thresholds). Task spec §7 lists it as carrying, and its calibration ran
**GO at perfect separation, EER 0.00**, so the gate is a measured component rather than
a hopeful one.

**The encoder is injected**, exactly as it was: an object exposing
``encode_audio(waveform, sample_rate)`` and ``encode_text(text)``, both returning a
512-d vector. So this module is numpy-only, imports no model, and unit-tests on the Mac
against a stub — ADR-0013's injection rule, which is also why ``perception.py``'s 506
LOC do not carry (ADR-0012).

**The room conditioning is not here.** ``is_anomaly`` used to take a ``detected_room``
and a ``room_prior`` and let the room verdict replace its own, which put the room
taxonomy, the hand-authored prior and the CLAP cosines in one function with two
different kinds of evidence in it. The split is ``normality.is_anomalous_here``:
this module answers "what does the audio sound like", that one answers "is that
normal *here*". ADR-0012 fixed the provider seam, not the composition, and a caller
that wants the context-free verdict now gets it without passing two ``None`` s.

**The smoke does not exercise any of this.** There is one sound and it is the anomaly
by construction (§4.3); the gate earns its place in R2's distractor arm, which is out
of scope for this map. It ships calibrated and tested rather than stubbed, because the
alternative is a seam that ships with one side.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from earshot.audio.clips import ANOMALY_CLASSES

__all__ = [
    "ANOMALY_CLASSES",
    "AMBIGUOUS_CLASSES",
    "CLASS_TO_CLAP_PROMPT",
    "NORMAL_PROMPTS",
    "ANOMALY_GATE_DELTA",
    "ANOMALY_GATE_TAU",
    "classify_anomaly",
    "is_anomaly",
    "heard_clip_for_clap",
]

# Ambiguous, CONTEXT-DEPENDENT sounds (ADR-0002, kept alive by ADR-0012): normal in some
# rooms, anomalous in others. They are what makes the room-conditioned arm testable at
# all — the three locked anomaly classes are anomalous everywhere, so they cannot
# exercise a gate that depends on the room.
AMBIGUOUS_CLASSES: Tuple[str, ...] = ("running_water", "appliance_hum")

# Zero-shot text prompts. Classification only, never retrieval: the old
# ``CLASS_TO_OBJECT`` map fed a memory query, and memory is out of this build.
CLASS_TO_CLAP_PROMPT: Dict[str, str] = {
    "baby_cry": "a baby crying",
    "alarm": "a loud alarm beeping",
    "glass_break": "the sound of breaking glass",
    "running_water": "running water or a faucet",
    "appliance_hum": "an appliance humming",
}

# The "routine, ignore it" reference set. Not anomaly classes: the gate fires only when
# the best anomaly prompt beats the best of these, so a merely loud benign sound is
# heard and does not consume the once-per-episode onset.
NORMAL_PROMPTS: Tuple[str, ...] = (
    "people talking",
    "a quiet room",
    "footsteps",
    "background noise",
    "an appliance humming",
)

# provenance: runtime — measured by the Gate-0b recalibration on RIR-CONVOLVED audio,
# which is the regime the live renderer also produces. The clean-clip calibration
# (delta 0.137) REJECTED the convolved alarm; on convolved audio CLAP's anomaly text
# margin is NEGATIVE and the separation is carried by the small absolute floor. That is
# why delta is below zero and tau is small — neither is a typo, and neither may be
# "tidied" without re-running the calibration.
#
# CARRIED WITH A CAVEAT THE BOX MUST CLOSE: they were measured against a grid render
# convolved offline. This renderer is the same convolution against a live IR, so the
# domain should match, but "should" is an inference. The first live run to exercise the
# gate re-measures them; until then nothing in the smoke depends on the numbers.
ANOMALY_GATE_DELTA = -0.2557
ANOMALY_GATE_TAU = 0.0341


def _unit(vector: Any) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    return values / (float(np.linalg.norm(values)) + 1e-8)


def classify_anomaly(
    waveform: Any,
    sample_rate: int,
    encoder: Any,
    classes: Sequence[str] = ANOMALY_CLASSES,
) -> Tuple[str, Dict[str, float]]:
    """Forced ``argmax`` over the anomaly prompts. Returns ``(class, {class: cosine})``.

    Can never say "normal" — it picks the closest of the classes it is given. That is
    the right shape for "what was it, given that it was one of these", and the wrong
    shape for "was it anything at all", which is ``is_anomaly``.
    """
    audio = _unit(encoder.encode_audio(waveform, sample_rate))
    scores = {
        name: float(np.dot(audio, _unit(encoder.encode_text(CLASS_TO_CLAP_PROMPT[name]))))
        for name in classes
    }
    return max(scores, key=lambda name: scores[name]), scores


def is_anomaly(
    waveform: Any,
    sample_rate: int,
    encoder: Any,
    *,
    classes: Sequence[str] = ANOMALY_CLASSES,
    normal_prompts: Sequence[str] = NORMAL_PROMPTS,
    delta: float = ANOMALY_GATE_DELTA,
    tau_abs: float = ANOMALY_GATE_TAU,
) -> Tuple[bool, str, Dict[str, float]]:
    """The open-set gate: does this sound like an anomaly at all?

    Scores the audio against both banks and fires when the best anomaly cosine beats the
    best normal cosine by ``delta`` **and** clears the absolute floor ``tau_abs``.

    Returns ``(fired, best_class, scores)``. ``best_class`` is the argmax anomaly class
    regardless of the verdict, so a caller can log what it *would* have classified —
    which is what made the convolved-audio recalibration diagnosable.

    ``scores`` carries every per-class cosine plus ``s_anom`` / ``s_norm`` / ``margin``.
    The defaults are the calibrated pair, not ``(0.0, 0.0)``: the old signature defaulted
    to "the anomaly side wins outright" and left the calibrated values to a caller that
    had to remember them, which is how the plain path needed two extra flags to get a
    working gate.
    """
    audio = _unit(encoder.encode_audio(waveform, sample_rate))

    def cosine(text: str) -> float:
        return float(np.dot(audio, _unit(encoder.encode_text(text))))

    anomaly_scores = {name: cosine(CLASS_TO_CLAP_PROMPT[name]) for name in classes}
    if not anomaly_scores:
        raise ValueError("no anomaly classes to score against")
    best_class = max(anomaly_scores, key=lambda name: anomaly_scores[name])
    s_anom = anomaly_scores[best_class]
    s_norm = max((cosine(prompt) for prompt in normal_prompts), default=0.0)
    margin = s_anom - s_norm
    scores: Dict[str, float] = dict(anomaly_scores)
    scores.update({"s_anom": s_anom, "s_norm": s_norm, "margin": margin})
    fired = bool(margin >= float(delta) and s_anom >= float(tau_abs))
    return fired, best_class, scores


def heard_clip_for_clap(heard: Any, sample_rate: int) -> Tuple[np.ndarray, int]:
    """The binaural signal the onset measured, as the mono waveform CLAP takes.

    One function rather than a ``.mean(axis=0)`` at each call site, because §4.3 says
    CLAP classifies **the heard clip** — the same signal ``onset.observe_step`` fired
    on, not a re-render. If the two ever diverge, the gate and the onset are answering
    questions about different sounds.

    Mono by averaging the ears: CLAP is not spatial, and the lateral cue is
    ``lateral.py``'s to read. ``sample_rate`` is passed through unchanged so the caller
    cannot forget that the encoder needs it.
    """
    signal = np.asarray(heard, dtype=np.float32)
    if signal.ndim == 2:
        signal = signal.mean(axis=0)
    return signal.reshape(-1).astype(np.float32), int(sample_rate)
