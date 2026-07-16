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
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

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

# Ambiguous, CONTEXT-DEPENDENT sounds (ADR-0002): normal in some rooms, anomalous
# in others. These drive the same-sound / two-rooms scene-conditioning test — the
# unambiguous ANOMALY_CLASSES (alarm/glass/cry) are anomalous everywhere, so they
# can't exercise a room-conditioned gate.
AMBIGUOUS_CLASSES: Tuple[str, ...] = ("running_water", "appliance_hum")

# Heard class → CLAP zero-shot text prompt (classification only, NOT retrieval).
CLASS_TO_CLAP_PROMPT: Dict[str, str] = {
    "baby_cry": "a baby crying",
    "alarm": "a loud alarm beeping",
    "glass_break": "the sound of breaking glass",
    "running_water": "running water or a faucet",
    "appliance_hum": "an appliance humming",
}

# ROOM_PRIOR (ADR-0002): room-type → the set of sound classes that are NORMAL
# (expected) there. The room-conditioned gate fires iff the heard class is NOT in
# this set for the detected room. This is a NEW, hand-authored table — distinct
# from room_resolver.CATEGORY_ROOM_PRIOR (which maps an OBJECT to its room). HM3D
# has no room-type ground truth, so this hand-authored prior IS the ground truth
# for normality. Keys are the shared room taxonomy (room_resolver.ROOM_TEXT_PROMPTS
# / ROOM_KEYWORDS) so a CLIP-classified room is interchangeable here. A room absent
# from this table carries no normality knowledge → the gate abstains (falls back to
# the context-free decision) rather than guessing.
ROOM_PRIOR: Dict[str, frozenset] = {
    "bathroom": frozenset({"running_water"}),
    "kitchen": frozenset({"running_water", "appliance_hum"}),
    "bedroom": frozenset(),
    "living_room": frozenset(),
    "dining_room": frozenset(),
    "hallway": frozenset(),
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


class OutOfCoverageError(LookupError):
    """The RIR grid has no cell describing this pose, so there is no audio to
    render. Raised only by the ``nearest`` floor guard (ADR-0003). Callers on the
    live path map it to silence; it is an expected condition, not a bug — unlike
    the ``ValueError`` raised when the guard itself is misconfigured."""


class RIRGrid:
    """A precomputed binaural RIR grid for one (scene, source).

    Attributes
    ----------
    cell_positions : (N, 3) float32   navigable listener positions (world xyz)
    source_position : (3,) float32    the single anomaly source (world xyz)
    irs : (N, 2, T) float32           per-cell binaural [left, right] IR
    sample_rate : int
    scene_id : str
    cell_geodesics : (N,) float32 | None   per-cell GEODESIC distance to the
        source (``None`` on legacy grids saved before this field existed). Lets
        the non-LOS seed picker detect "around-a-corner" cells (geodesic ≫ the
        straight-line distance) without re-opening the sim.
    ear_height_m : float | None       the listener height the cells were rendered
        at, i.e. ``cell_y == navmesh_y + ear_height_m`` (``None`` on legacy grids).
        Required by the ``nearest`` floor guard: a runtime agent pose carries the
        NAVMESH y, so only ``agent_y + ear_height_m`` is comparable to ``cell_y``.
        Without it a same-floor pose and a one-floor-up pose are both ~1.5 m from
        a cell in y and cannot be told apart (ADR-0003).
    """

    def __init__(self, cell_positions, source_position, irs, sample_rate, scene_id,
                 cell_geodesics=None, ear_height_m=None):
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
        if cell_geodesics is None:
            self.cell_geodesics = None
        else:
            cg = np.asarray(cell_geodesics, dtype=np.float32).reshape(-1)
            if cg.shape[0] != self.cell_positions.shape[0]:
                raise ValueError(
                    f"cell_geodesics length {cg.shape[0]} != "
                    f"{self.cell_positions.shape[0]} cells")
            self.cell_geodesics = cg
        self.ear_height_m = None if ear_height_m is None else float(ear_height_m)

    def __len__(self) -> int:
        return int(self.cell_positions.shape[0])

    @property
    def cell_energies(self) -> np.ndarray:
        """(N,) float64 per-cell total IR energy ``sum(ir**2)`` over ``[2, T]`` —
        a monotone proxy for "how audible the source is" at each cell."""
        return np.sum(np.square(self.irs, dtype=np.float64), axis=(1, 2))

    def nearest(self, agent_pos, *, max_dy: Optional[float] = None
                ) -> Tuple[np.ndarray, int, float]:
        """Nearest cell to ``agent_pos`` by 2-D (x, z) distance.

        ``max_dy=None`` (default): y is ignored entirely — the legacy behaviour,
        byte-identical for objectnav / audiogoal / revisit, where the source is
        co-located with the goal so an off-floor lookup cannot arise.

        ``max_dy`` set: only cells on the agent's FLOOR are eligible, i.e. those
        with ``|cell_y - agent_y| <= max_dy``. Both sides are NAVMESH y: the
        renderer stores each cell's listener pose (``st.position = cell``, which
        must lie on the navmesh) and carries the ear as a sensor-local offset
        that never enters ``cell_positions``. This is not cosmetic. The grid is
        rendered on ONE floor, so an off-floor pose has no cell that describes
        it, and resolving one anyway FABRICATES audio — the agent "hears" a
        source through a storey of concrete (ADR-0003; the render path already
        guards this via ``_nearest_same_floor``).

        Do NOT reintroduce an ear offset here. It was tried, and because the
        cells are navmesh-y it put a same-floor pose 1.5 m off its own floor and
        silenced the entire grid on every floor (``audio_energy_max=0.0`` in 8/8
        of runs/anomresp-bed-s{1,3}). A 1.5 m ear against a 3.0 m storey needs no
        offset to separate floors: navmesh-to-navmesh gives 0.0 vs 3.0.

        Raises ``OutOfCoverageError`` when guarded and no cell is on the agent's
        floor (the caller should render silence).

        Returns ``(ir (2, T), cell_idx, distance_m)``.
        """
        p = np.asarray(agent_pos, dtype=np.float32).reshape(-1)
        axz = np.array([p[0], p[2]], dtype=np.float32) if p.shape[0] >= 3 \
            else np.array([p[0], p[1]], dtype=np.float32)
        cxz = self.cell_positions[:, [0, 2]]
        d = np.linalg.norm(cxz - axz[None, :], axis=1)
        if max_dy is not None:
            if p.shape[0] < 3:
                raise ValueError("nearest(max_dy=...) needs a 3-D agent position")
            agent_y = float(p[1])
            on_floor = np.abs(self.cell_positions[:, 1] - agent_y) <= float(max_dy)
            if not bool(on_floor.any()):
                raise OutOfCoverageError(
                    f"no RIR cell within {max_dy} m of the agent's floor "
                    f"(agent navmesh y={agent_y:.3f}, grid cell y="
                    f"{float(self.cell_positions[0, 1]):.3f}); the grid does not "
                    f"cover this floor, so there is no audio to render.")
            d = np.where(on_floor, d, np.inf)
        idx = int(np.argmin(d))
        return self.irs[idx], idx, float(d[idx])

    @classmethod
    def load(cls, path: str) -> "RIRGrid":
        raw = np.load(path)  # plain arrays only → no allow_pickle
        cg = raw["cell_geodesics"] if "cell_geodesics" in raw.files else None
        eh = float(raw["ear_height_m"]) if "ear_height_m" in raw.files else None
        return cls(
            cell_positions=raw["cell_positions"],
            source_position=raw["source_position"],
            irs=raw["irs"],
            sample_rate=int(raw["sample_rate"]),
            scene_id=str(raw["scene_id"]),
            cell_geodesics=cg,
            ear_height_m=eh,
        )


def save_rir_grid(
    path: str,
    *,
    cell_positions,
    source_position,
    irs: Union[np.ndarray, Sequence[np.ndarray]],
    sample_rate: int,
    scene_id: str,
    cell_geodesics=None,
    ear_height_m=None,
) -> None:
    """Serialize an RIR grid to a plain ``.npz`` (no object arrays).

    ``irs`` may be a uniform ``(N, 2, T)`` array OR a list of ``(2, T_i)`` IRs of
    varying length (the renderer can emit slightly different tail lengths per
    cell); variable-length IRs are zero-padded to the common max T so the stored
    array is a plain float32 tensor.

    ``cell_geodesics`` (optional ``(N,)``) persists each cell's geodesic-to-source
    distance for the non-LOS seed picker; omitting it keeps the legacy on-disk
    format (``RIRGrid.load`` reads ``cell_geodesics`` as ``None`` when absent).

    ``ear_height_m`` (optional scalar) persists the listener height the cells were
    rendered at, which the ``nearest`` floor guard needs to compare a navmesh-y
    agent pose against an ear-height cell (ADR-0003). Omitting it likewise keeps
    the legacy on-disk format and leaves the guard unusable on that grid.
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

    extra = {}
    if cell_geodesics is not None:
        cg = np.asarray(cell_geodesics, dtype=np.float32).reshape(-1)
        if cg.shape[0] != cell_positions.shape[0]:
            raise ValueError(
                f"cell_geodesics length {cg.shape[0]} != "
                f"{cell_positions.shape[0]} cells")
        extra["cell_geodesics"] = cg
    if ear_height_m is not None:
        extra["ear_height_m"] = np.float32(float(ear_height_m))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        cell_positions=cell_positions,
        source_position=source_position,
        irs=irs_arr,
        sample_rate=np.int64(int(sample_rate)),
        scene_id=np.str_(str(scene_id)),
        **extra,
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


def diotic_collapse(sig) -> np.ndarray:
    """Collapse a ``(2, L)`` binaural signal to a non-directional (diotic) bed:
    the mono mean broadcast to both ears, so the bed carries NO lateral/ILD cue
    (it protects the anomaly's ``lateral_sign``/``estimate_doa`` when summed).
    Shared by the live mixture render (render_step_audio) and the Gate-0b
    diagnostic so the calibration domain matches the live domain exactly."""
    m = np.asarray(sig, dtype=np.float32)
    if m.ndim == 1:
        return np.stack([m, m])
    mono = m.mean(axis=0)
    return np.stack([mono, mono]).astype(np.float32)


def rms(signal) -> float:
    """Root-mean-square energy over all samples/channels."""
    s = np.asarray(signal, dtype=np.float64)
    if s.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(s))))


# ----------------------------------------------------------------------
# Clip augmentation (P2.1) — the one pure seam the Gate-0.3 calibration uses
# ----------------------------------------------------------------------
#
# The CLAP anomaly gate is calibrated on CLEAN clips but runs on RIR-convolved,
# room-varied, background-mixed audio (the documented clean->convolved cliff and
# the loud-bed false-fire). ``augment_clip`` deterministically generates the
# runtime distribution — background at a target SNR, reverb/room-size jitter,
# pitch- and time-shift, and (optionally) the same RIR convolution the live path
# applies — so the gate can be recalibrated on what it actually hears.
#
# Contract (user stories 22/25/26): ONE public function; sub-transforms are
# internal and exercised only through it. The output is a MONO clip of the SAME
# length and sample rate as the input (so it flows through render_at_pose
# unchanged). Deterministic given the spec — no unseeded RNG; the reverb IR is
# drawn from ``spec.seed`` (vary by index, never a wall-clock seed). Shares
# ``diotic_collapse`` so the (optional) RIR path matches the live signal domain.


@dataclass
class AugmentSpec:
    """A deterministic augmentation recipe for :func:`augment_clip`.

    Every field defaults to a no-op, so ``AugmentSpec()`` is the identity. All
    magnitudes are explicit (no internal randomness beyond the seeded reverb IR),
    so the same spec always yields the same waveform.

    Fields
    ------
    seed : int
        Seeds ONLY the synthetic reverb IR (the sole stochastic transform). Two
        specs that differ only in ``seed`` differ only in the reverb tail.
    background : (M,) array | None
        A mono background clip mixed in at ``snr_db`` (looped/cropped to length).
        ``None`` (or ``snr_db is None``) => no background.
    snr_db : float | None
        Target signal-to-background ratio in dB: the background is scaled so
        ``20*log10(rms(clip)/rms(scaled_bg)) == snr_db`` before it is added. Lower
        dB => louder background.
    pitch_semitones : float
        Resample-based pitch shift (positive = up), then crop/pad to length.
    time_shift_frac : float
        Circular shift by ``round(time_shift_frac * len)`` samples (energy- and
        length-preserving).
    reverb_decay : float
        Room-size / reverberation TIME in seconds of a synthetic
        exponential-decay noise IR convolved with the clip (0 => no reverb).
    rir : (T,) or (2, T) array | None
        An optional impulse response to convolve (the live-path RIR); a binaural
        ``(2, T)`` IR is diotic-collapsed to mono first so the output stays mono.
    """
    seed: int = 0
    background: Optional[np.ndarray] = None
    snr_db: Optional[float] = None
    pitch_semitones: float = 0.0
    time_shift_frac: float = 0.0
    reverb_decay: float = 0.0
    rir: Optional[np.ndarray] = None


def _fit_length(sig: np.ndarray, length: int) -> np.ndarray:
    """Crop or loop ``sig`` to exactly ``length`` samples (mono)."""
    s = np.asarray(sig, dtype=np.float32).reshape(-1)
    if s.size == length:
        return s
    if s.size == 0:
        return np.zeros(length, dtype=np.float32)
    if s.size > length:
        return s[:length]
    reps = int(np.ceil(length / s.size))
    return np.tile(s, reps)[:length].astype(np.float32)


def _pitch_shift(clip: np.ndarray, semitones: float) -> np.ndarray:
    """Resample-based pitch shift: stretch the time base by 2**(-semitones/12)
    then crop/pad back to the original length (mono, length-preserving). A pure
    linear resample — no librosa dependency."""
    if abs(float(semitones)) < 1e-9:
        return clip
    n = clip.size
    factor = 2.0 ** (float(semitones) / 12.0)          # >1 => higher pitch
    m = max(1, int(round(n / factor)))
    src = np.linspace(0.0, n - 1, num=m, dtype=np.float64)
    resampled = np.interp(src, np.arange(n, dtype=np.float64), clip).astype(np.float32)
    return _fit_length(resampled, n)


def _time_shift(clip: np.ndarray, frac: float) -> np.ndarray:
    """Circular shift by ``round(frac*len)`` samples — energy- and length-stable."""
    if abs(float(frac)) < 1e-12:
        return clip
    k = int(round(float(frac) * clip.size))
    if k % clip.size == 0:
        return clip
    return np.roll(clip, k).astype(np.float32)


def _reverb(clip: np.ndarray, decay_s: float, sample_rate: int, seed: int) -> np.ndarray:
    """Convolve with a synthetic exponential-decay noise IR of ``decay_s`` seconds
    (a cheap room-size proxy), cropped back to the clip length. Deterministic in
    ``seed``."""
    if float(decay_s) <= 0.0:
        return clip
    n_ir = max(1, int(round(float(decay_s) * int(sample_rate))))
    rng = np.random.default_rng(int(seed))
    t = np.arange(n_ir, dtype=np.float32)
    env = np.exp(-3.0 * t / max(1.0, n_ir))            # ~-9 dB per decay window
    ir = (rng.standard_normal(n_ir).astype(np.float32) * env)
    ir[0] += 1.0                                       # keep the direct path
    from scipy.signal import fftconvolve
    wet = fftconvolve(clip, ir)[: clip.size].astype(np.float32)
    return wet


def _rir_mono(clip: np.ndarray, rir) -> np.ndarray:
    """Convolve with a (possibly binaural) IR, diotic-collapsed to mono so the
    output stays a mono clip; cropped to the input length."""
    ir = np.asarray(rir, dtype=np.float32)
    if ir.ndim == 2:
        ir = diotic_collapse(ir)[0]                    # both rows equal after collapse
    ir = ir.reshape(-1)
    from scipy.signal import fftconvolve
    return fftconvolve(clip, ir)[: clip.size].astype(np.float32)


def augment_clip(clip, sample_rate: int, spec: "AugmentSpec") -> np.ndarray:
    """Deterministically augment a MONO ``clip`` per ``spec`` (see :class:`AugmentSpec`).

    Returns a mono float32 waveform of the SAME length and sample rate as the
    input, so it flows through ``render_at_pose`` / the convolution path unchanged.
    Transforms are applied in a FIXED order (pitch -> time-shift -> reverb -> RIR
    -> background mix) so the composition is order-stable and the same spec always
    yields the same array. Pure: no CLAP, no simulator, no unseeded RNG.
    """
    x = np.asarray(clip, dtype=np.float32).reshape(-1)
    n = x.size
    out = _pitch_shift(x, spec.pitch_semitones)
    out = _time_shift(out, spec.time_shift_frac)
    out = _reverb(out, spec.reverb_decay, sample_rate, spec.seed)
    if spec.rir is not None:
        out = _rir_mono(out, spec.rir)
    if spec.background is not None and spec.snr_db is not None:
        bg = _fit_length(spec.background, n)
        sig_rms = rms(out)
        bg_rms = rms(bg)
        if bg_rms > 1e-12 and sig_rms > 1e-12:
            target = sig_rms / (10.0 ** (float(spec.snr_db) / 20.0))
            bg = bg * np.float32(target / bg_rms)
        out = out + bg
    return _fit_length(out, n).astype(np.float32)


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


def room_conditioned_anomaly(
    sound_class: Optional[str],
    detected_room: Optional[str],
    room_prior: Dict[str, "frozenset"],
) -> Optional[bool]:
    """Scene-conditioned normality verdict (ADR-0002) — a PURE function of
    ``(sound_class, detected_room, ROOM_PRIOR)``.

    Returns ``True`` (the heard class is UNEXPECTED in this room → anomalous),
    ``False`` (EXPECTED here → normal, do not interrupt), or ``None`` (cannot
    scene-condition: no class, no detected room, or the room carries no normality
    knowledge in the prior). ``None`` means "abstain" — the caller falls back to
    the context-free decision. No CLAP, no simulator (user story 32).
    """
    if not sound_class or detected_room is None:
        return None
    if detected_room not in room_prior:
        return None
    return bool(sound_class not in room_prior[detected_room])


def is_anomaly(
    waveform,
    sample_rate: int,
    encoder,
    *,
    classes: Sequence[str] = ANOMALY_CLASSES,
    normal_prompts: Sequence[str] = NORMAL_PROMPTS,
    delta: float = 0.0,
    tau_abs: float = 0.0,
    detected_room: Optional[str] = None,
    room_prior: Optional[Dict[str, "frozenset"]] = None,
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

    Scene-conditioning (ADR-0002): pass a ``detected_room`` (from the CLIP room
    classifier) AND a ``room_prior`` (room → expected-sound set) to make the
    verdict depend on the room — the same clip is normal in one room and anomalous
    in another. When the room can be scene-conditioned (:func:`room_conditioned_anomaly`
    returns non-``None``) it REPLACES the margin decision and adds a ``room_verdict``
    key to ``scores``; when it abstains, the context-free margin decision stands.
    With both room args absent (the default) the path is byte-identical.
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
    if detected_room is not None and room_prior is not None:
        rc = room_conditioned_anomaly(best_class, detected_room, room_prior)
        if rc is not None:                     # room known + covered → room verdict wins
            fired = bool(rc)
            scores["room_verdict"] = 1.0 if rc else 0.0
    return fired, best_class, scores
