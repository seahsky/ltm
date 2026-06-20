"""
Runner-side audio layer for the AudioGoal / FSD50K anomaly task.

This module is the *live* side of the two-env split (see the AudioGoal plan):
RIRs are rendered OFFLINE in the ``soundspaces-spike`` env by
``scripts/render_rir_grid.py``; at run time the ``ltm-embodied`` runner does a
cheap nearest-cell lookup + ``scipy.signal.fftconvolve`` of a real FSD50K clip
with that cell's binaural impulse response. **Nothing here imports habitat_sim
or the audio simulator** — only numpy + scipy + (lazily, for CLAP) perception.

Pieces:
  * ``RIRGrid`` / ``save_rir_grid`` — the precomputed binaural RIR grid: plain
    ``.npz`` (no pickled objects), O(1)-ish nearest-cell lookup by 2-D xz.
  * ``render_at_pose`` / ``rms`` — convolve a mono clip with the nearest IR.
  * ``estimate_doa`` — azimuth from a binaural signal via ITD (primary) + ILD
    (sign tie-break). Azimuth-only / front-hemisphere (binaural can't resolve
    front-back — the plan reports time-to-source + detection latency to cover
    that ambiguity).
  * ``classify_anomaly`` — 3-way zero-shot (cry / alarm / glass) given an
    injected CLAP-style encoder (``encode_audio`` / ``encode_text`` → 512-d).
    The heavy CLAP model lives in ``perception.CLAPAudioEncoder``; here it is a
    plain dependency so this module stays import-light and unit-testable.

Class → object map note: ``CLASS_TO_OBJECT`` is the *static default* the
plan's pillar-1 retrieval path uses (heard class → object name → the EXISTING
SBERT query ``"there is a {object}"`` in ``memory_bridge``). A dataset builder
(M2) may record the actual Qwen-captioned source object per episode and
override this default per-episode; this map is the fallback.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np


# ----------------------------------------------------------------------
# Anomaly classes + the two maps the task hangs off of
# ----------------------------------------------------------------------

# Exactly three FSD50K-backed emergency classes (locked decision): confirmed
# present in FSD50K and well-separated for CLAP zero-shot.
ANOMALY_CLASSES: Tuple[str, ...] = ("baby_cry", "alarm", "glass_break")

# Heard class → object name fed to the EXISTING SBERT retrieval query
# (memory_bridge `_GOAL_QUERY_TEMPLATE` = "there is a {}"). Static default;
# overridable per-episode by the dataset builder (M2) which knows the actual
# Qwen-captioned source object. baby_cry→crib is the M0c demo mapping.
CLASS_TO_OBJECT: Dict[str, str] = {
    "baby_cry": "crib",
    "alarm": "oven",
    "glass_break": "window",
}

# Heard class → CLAP zero-shot text prompt (classification only, NOT retrieval).
CLASS_TO_CLAP_PROMPT: Dict[str, str] = {
    "baby_cry": "a baby crying",
    "alarm": "a loud alarm beeping",
    "glass_break": "the sound of breaking glass",
}

# Normal/background prompt bank for the open-set normal-vs-anomaly gate (Step 1).
# These are NOT anomaly classes — they are the "routine, ignore it" reference set.
# ``is_anomaly`` fires only when the best ANOMALY-prompt cosine beats the best
# NORMAL-prompt cosine by a margin, so a sound that is merely loud (people
# talking, footsteps) does not trigger an anomaly response. Calibrate the margin
# with ``scripts/diagnose_normal_anomaly_calib.py``.
NORMAL_PROMPTS: Tuple[str, ...] = (
    "people talking",
    "a quiet room",
    "footsteps",
    "background noise",
    "an appliance humming",
)


# ----------------------------------------------------------------------
# RIR grid
# ----------------------------------------------------------------------


class RIRGrid:
    """A precomputed binaural RIR grid for one (scene, source).

    Attributes
    ----------
    cell_positions : (N, 3) float32   navigable listener positions (world xyz)
    source_position : (3,) float32    the single anomaly source (world xyz)
    irs : (N, 2, T) float32           per-cell binaural [left, right] IR
    sample_rate : int
    scene_id : str
    """

    def __init__(self, cell_positions, source_position, irs, sample_rate, scene_id):
        self.cell_positions = np.asarray(cell_positions, dtype=np.float32).reshape(-1, 3)
        self.source_position = np.asarray(source_position, dtype=np.float32).reshape(3)
        self.irs = np.asarray(irs, dtype=np.float32)
        if self.irs.ndim != 3 or self.irs.shape[1] != 2:
            raise ValueError(f"irs must be (N, 2, T); got {self.irs.shape}")
        if self.irs.shape[0] != self.cell_positions.shape[0]:
            raise ValueError(
                f"cell/ir count mismatch: {self.cell_positions.shape[0]} cells "
                f"vs {self.irs.shape[0]} irs")
        self.sample_rate = int(sample_rate)
        self.scene_id = str(scene_id)

    def __len__(self) -> int:
        return int(self.cell_positions.shape[0])

    def nearest(self, agent_pos) -> Tuple[np.ndarray, int, float]:
        """Nearest cell to ``agent_pos`` by 2-D (x, z) distance — y is ignored
        (HM3D floors differ in y; the listener grid is at one ear height).

        Returns ``(ir (2, T), cell_idx, distance_m)``.
        """
        p = np.asarray(agent_pos, dtype=np.float32).reshape(-1)
        axz = np.array([p[0], p[2]], dtype=np.float32) if p.shape[0] >= 3 \
            else np.array([p[0], p[1]], dtype=np.float32)
        cxz = self.cell_positions[:, [0, 2]]
        d = np.linalg.norm(cxz - axz[None, :], axis=1)
        idx = int(np.argmin(d))
        return self.irs[idx], idx, float(d[idx])

    @classmethod
    def load(cls, path: str) -> "RIRGrid":
        raw = np.load(path)  # plain arrays only → no allow_pickle
        return cls(
            cell_positions=raw["cell_positions"],
            source_position=raw["source_position"],
            irs=raw["irs"],
            sample_rate=int(raw["sample_rate"]),
            scene_id=str(raw["scene_id"]),
        )


def save_rir_grid(
    path: str,
    *,
    cell_positions,
    source_position,
    irs: Union[np.ndarray, Sequence[np.ndarray]],
    sample_rate: int,
    scene_id: str,
) -> None:
    """Serialize an RIR grid to a plain ``.npz`` (no object arrays).

    ``irs`` may be a uniform ``(N, 2, T)`` array OR a list of ``(2, T_i)`` IRs of
    varying length (the renderer can emit slightly different tail lengths per
    cell); variable-length IRs are zero-padded to the common max T so the stored
    array is a plain float32 tensor.
    """
    cell_positions = np.asarray(cell_positions, dtype=np.float32).reshape(-1, 3)
    source_position = np.asarray(source_position, dtype=np.float32).reshape(3)

    if isinstance(irs, np.ndarray) and irs.dtype != object and irs.ndim == 3:
        irs_arr = irs.astype(np.float32)
    else:
        seq: List[np.ndarray] = [np.asarray(x, dtype=np.float32) for x in irs]
        if any(x.ndim != 2 or x.shape[0] != 2 for x in seq):
            raise ValueError("each IR must be (2, T_i)")
        T = max(int(x.shape[-1]) for x in seq)
        irs_arr = np.zeros((len(seq), 2, T), dtype=np.float32)
        for i, x in enumerate(seq):
            irs_arr[i, :, : x.shape[-1]] = x

    if irs_arr.shape[0] != cell_positions.shape[0]:
        raise ValueError(
            f"cell/ir count mismatch: {cell_positions.shape[0]} vs {irs_arr.shape[0]}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        cell_positions=cell_positions,
        source_position=source_position,
        irs=irs_arr,
        sample_rate=np.int64(int(sample_rate)),
        scene_id=np.str_(str(scene_id)),
    )


# ----------------------------------------------------------------------
# render-at-pose
# ----------------------------------------------------------------------


def render_at_pose(grid: RIRGrid, agent_pos, clip, max_len: int = None) -> np.ndarray:
    """Convolve a mono ``clip`` with the binaural IR of the cell nearest to
    ``agent_pos`` → a ``(2, L)`` float32 ear signal. ~0.01 s/step."""
    from scipy.signal import fftconvolve

    ir, _idx, _dist = grid.nearest(agent_pos)
    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    left = fftconvolve(clip, np.asarray(ir[0], dtype=np.float32))
    right = fftconvolve(clip, np.asarray(ir[1], dtype=np.float32))
    out = np.stack([left, right], axis=0).astype(np.float32)
    if max_len is not None:
        out = out[:, :int(max_len)]
    return out


def rms(signal) -> float:
    """Root-mean-square energy over all samples/channels."""
    s = np.asarray(signal, dtype=np.float64)
    if s.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(s))))


# ----------------------------------------------------------------------
# DOA: azimuth from a binaural signal (ITD primary, ILD sign tie-break)
# ----------------------------------------------------------------------


def _itd_xcorr(left, right, max_lag: int) -> int:
    """TDOA (samples, left-minus-right) by plain cross-correlation, bounded to
    the physical interaural lag range ``±max_lag`` (rejects spurious far peaks
    from reverberant multipath)."""
    from scipy.signal import correlate, correlation_lags

    corr = correlate(left, right, mode="full", method="fft")
    lags = correlation_lags(len(left), len(right), mode="full")
    mask = np.abs(lags) <= max_lag
    corr_m, lags_m = corr[mask], lags[mask]
    return int(lags_m[int(np.argmax(corr_m))])


def _itd_gcc_phat(left, right, max_lag: int) -> int:
    """TDOA (samples, left-minus-right) by GCC-PHAT: the cross-spectrum is
    whitened (magnitude normalized, phase kept). Sharpens the direct-path peak
    in moderate reverb, but amplifies empty/low-energy bins, so it can be noisier
    than plain x-corr at low direct-to-reverberant ratios — compared on real data
    by audio_loop_smoke before being made the default."""
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
    cc_lagged = np.concatenate((cc[-max_lag:], cc[: max_lag + 1]))
    lags = np.arange(-max_lag, max_lag + 1)
    return int(lags[int(np.argmax(cc_lagged))])


def estimate_doa(
    binaural,
    sample_rate: int,
    ear_distance_m: float = 0.18,
    speed_of_sound: float = 343.0,
    method: str = "xcorr",
) -> float:
    """Azimuth (radians) of the source from a ``(2, L)`` [left, right] signal.

    Positive = source to the agent's RIGHT, negative = LEFT, 0 = ahead. Range
    ``[-pi/2, pi/2]`` (front hemisphere — binaural is front-back ambiguous).

    The interaural TDOA is estimated by ``method`` — ``"xcorr"`` (default, plain
    bounded cross-correlation) or ``"gcc_phat"`` (phase-transform). A positive
    lag means the left ear arrives later → right ear is closer → source on the
    right → positive azimuth.

    SoundSpaces 2.0 renders binaural via an Ambisonic→time-aligned-HRTF path
    (Zaunschirm 2018) that strips the broadband ITD and leaves only a tiny
    residual lag (~1-8 samples) even for side sources, while keeping a real ILD.
    So when ``|ITD|`` is below a confidence floor (``AUDIO_ITD_MIN_SAMPLES``,
    default 3) AND there is a clear interaural LEVEL difference, the ILD sign —
    not a sub-resolution lag — drives the azimuth. Only when no ILD information
    is present do we trust a tiny ITD (so clean free-field signals still resolve
    small angles). The ITD path stays exact for any stronger-ITD config.
    """
    import os

    b = np.asarray(binaural, dtype=np.float64)
    left, right = b[0], b[1]
    max_lag = int(np.ceil(ear_distance_m / speed_of_sound * sample_rate)) + 1

    if method == "gcc_phat":
        itd = _itd_gcc_phat(left, right, max_lag)
    elif method == "xcorr":
        itd = _itd_xcorr(left, right, max_lag)
    else:
        raise ValueError(f"unknown DOA method {method!r} (xcorr | gcc_phat)")

    rms_l = float(np.sqrt(np.mean(np.square(left))))
    rms_r = float(np.sqrt(np.mean(np.square(right))))
    denom = rms_l + rms_r
    ild = (rms_r - rms_l) / denom if denom > 1e-12 else 0.0  # +ve = right louder

    itd_floor = float(os.environ.get("AUDIO_ITD_MIN_SAMPLES", "3"))
    ild_min = float(os.environ.get("AUDIO_ILD_MIN", "0.02"))

    if abs(itd) >= itd_floor:
        s = np.clip(itd / sample_rate * speed_of_sound / ear_distance_m, -1.0, 1.0)
        return float(np.arcsin(s))
    if abs(ild) >= ild_min:
        # sub-resolution ITD but a real level difference → ILD is the lateral cue
        return float(np.arcsin(np.clip(ild, -1.0, 1.0)))
    # no ILD info: trust the (tiny) ITD so clean free-field signals still resolve
    s = np.clip(itd / sample_rate * speed_of_sound / ear_distance_m, -1.0, 1.0)
    return float(np.arcsin(s))


def lateral_sign(binaural) -> int:
    """Fold-invariant left/right cue from the interaural level difference:
    ``+1`` source to the RIGHT, ``-1`` LEFT, ``0`` if ambiguous. This is the
    cue SoundSpaces spatializes reliably (ILD/energy), unlike the engine-weak
    ITD. Shared by the M0b gate and any audio-nav code so there is one
    definition of the lateral sign."""
    b = np.asarray(binaural, dtype=np.float64)
    rms_l = float(np.sqrt(np.mean(np.square(b[0]))))
    rms_r = float(np.sqrt(np.mean(np.square(b[1]))))
    denom = rms_l + rms_r
    if denom <= 1e-12:
        return 0
    ild = (rms_r - rms_l) / denom
    if abs(ild) < 1e-6:
        return 0
    return 1 if ild > 0.0 else -1


# ----------------------------------------------------------------------
# CLAP zero-shot 3-way classification
# ----------------------------------------------------------------------


def classify_anomaly(
    waveform,
    sample_rate: int,
    encoder,
    classes: Sequence[str] = ANOMALY_CLASSES,
) -> Tuple[str, Dict[str, float]]:
    """3-way zero-shot anomaly class via CLAP cosine argmax.

    ``encoder`` must expose ``encode_audio(waveform, sample_rate) -> (512,)`` and
    ``encode_text(text) -> (512,)`` (e.g. ``perception.CLAPAudioEncoder``).
    Returns ``(class, {class: cosine})``.
    """
    a = np.asarray(encoder.encode_audio(waveform, sample_rate), dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-8)
    scores: Dict[str, float] = {}
    for c in classes:
        t = np.asarray(encoder.encode_text(CLASS_TO_CLAP_PROMPT[c]), dtype=np.float32)
        t = t / (np.linalg.norm(t) + 1e-8)
        scores[c] = float(np.dot(a, t))
    best = max(scores, key=scores.get)
    return best, scores


def is_anomaly(
    waveform,
    sample_rate: int,
    encoder,
    *,
    classes: Sequence[str] = ANOMALY_CLASSES,
    normal_prompts: Sequence[str] = NORMAL_PROMPTS,
    delta: float = 0.0,
    tau_abs: float = 0.0,
) -> Tuple[bool, str, Dict[str, float]]:
    """Open-set normal-vs-anomaly gate on top of CLAP zero-shot.

    Unlike :func:`classify_anomaly` (a forced 3-way argmax that can NEVER say
    "normal"), this scores the audio against BOTH the anomaly-class prompts and a
    bank of normal/background prompts (:data:`NORMAL_PROMPTS`). It fires
    (``True``) iff the best anomaly cosine beats the best normal cosine by at
    least ``delta`` AND clears the absolute floor ``tau_abs``. The defaults
    ``(0.0, 0.0)`` reduce to "the anomaly side wins outright" — calibrate ``delta``
    / ``tau_abs`` with ``scripts/diagnose_normal_anomaly_calib.py``.

    ``encoder`` is the same object :func:`classify_anomaly` uses (``encode_audio``
    + ``encode_text``). Returns ``(fired, best_class, scores)`` where ``scores``
    carries every per-class anomaly cosine PLUS the summary keys ``s_anom`` /
    ``s_norm`` / ``margin``. ``best_class`` is the argmax anomaly class regardless
    of the gate decision, so the caller can log what it WOULD have classified.
    """
    a = np.asarray(encoder.encode_audio(waveform, sample_rate), dtype=np.float32)
    a = a / (np.linalg.norm(a) + 1e-8)

    def _cos(text: str) -> float:
        t = np.asarray(encoder.encode_text(text), dtype=np.float32)
        t = t / (np.linalg.norm(t) + 1e-8)
        return float(np.dot(a, t))

    anom = {c: _cos(CLASS_TO_CLAP_PROMPT[c]) for c in classes}
    s_norm = max((_cos(p) for p in normal_prompts), default=0.0)
    best_class = max(anom, key=anom.get)
    s_anom = anom[best_class]
    margin = s_anom - s_norm
    fired = bool(margin >= float(delta) and s_anom >= float(tau_abs))
    scores: Dict[str, float] = dict(anom)
    scores.update({"s_anom": s_anom, "s_norm": s_norm, "margin": margin})
    return fired, best_class, scores
