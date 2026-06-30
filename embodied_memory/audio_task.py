"""
audio_task — the pure per-step audio DECISION brain of the AudioGoal task (M1).

Responsibility split (sacred two-env boundary):
  * ``habitat_env`` RENDERS only: it writes ``Step.audio`` (a cached-RIR-grid
    nearest-cell lookup + fftconvolve of the FSD50K clip) and never decides.
  * ``audio_task`` (this module) DECIDES: onset-detection, once-per-episode CLAP
    classification → object override for retrieval, the lateral-sign DOA cue, and
    the audio-energy STOP gate. Pure numpy/scipy + ``embodied_memory.audio`` (and,
    only inside the runner, an injected ``perception.CLAPAudioEncoder``).
    **It never imports habitat_sim**, so all of it is unit-testable without a sim.
  * ``episode_runner`` ORCHESTRATES: it calls render (via the source) and these
    functions, threading the per-episode :class:`AudioEpisodeState`.

Every entry point tolerates ``audio=None`` / ``clap_encoder=None`` and, when the
task is disabled, returns the objectnav-identical result (e.g.
``audio_target_for_retrieval`` returns the fallback category verbatim), so the
non-audio paths are byte-unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from .audio import (
    CLASS_TO_OBJECT,
    RIRGrid,
    classify_anomaly,
    is_anomaly,
    lateral_sign,
    render_at_pose,
    rms,
)


@dataclass
class AudioTaskConfig:
    """Static config for one AudioGoal run (env-tunable from the CLI)."""
    enabled: bool = False               # True iff --task audiogoal AND a grid is loaded
    t_anom: int = 30                    # step index the anomaly begins; before = silence
    onset_rms: float = 0.05             # RMS threshold for first-onset detection
    energy_stop_db_rms: float = 0.20    # RMS threshold for the audio-energy STOP gate
    stop_distance_m: float = 1.5        # only allow the energy STOP within this d2g
    target_norm_rms_db: float = -20.0   # clip pre-normalization target (RMS dBFS)
    sample_rate: int = 48000            # of the rendered audio (threaded to classify)
    # Step 1 open-set normal-vs-anomaly gate (env LTM_AUDIO_ANOMALY_GATE). When
    # ON, the energy onset only FIRES if CLAP separates the sound from the
    # normal/background prompt bank (best anomaly cosine beats best normal cosine
    # by >= anomaly_delta AND clears anomaly_tau). A merely-loud benign sound
    # (people talking, footsteps) is heard but does NOT consume the once-per-
    # episode onset. OFF (default) => onset is energy-only, byte-identical.
    anomaly_gate: bool = False
    anomaly_delta: float = 0.0
    anomaly_tau: float = 0.0


@dataclass
class AudioEpisodeState:
    """Per-episode mutable audio state. ``reset()`` at every episode start so the
    once-per-episode onset/classification does not leak across episodes."""
    detected: bool = False
    anomaly_class: Optional[str] = None
    target_override: Optional[str] = None   # CLASS_TO_OBJECT[class] (CLAP affordance)
    # M2: the actual captioned object the anomaly source sits near, from the
    # dataset (episode.info["anomaly_object"]). Preferred over target_override so
    # warm recall queries the object the agent really mapped (e.g. 'bed') rather
    # than the static affordance ('crib') — set per-episode by the runner.
    anomaly_object_override: Optional[str] = None
    onset_step: Optional[int] = None
    last_energy: float = 0.0
    last_lateral: int = 0

    def reset(self) -> None:
        self.detected = False
        self.anomaly_class = None
        self.target_override = None
        self.anomaly_object_override = None
        self.onset_step = None
        self.last_energy = 0.0
        self.last_lateral = 0


def resolve_t_anom(ep_info, default: int) -> int:
    """Per-episode anomaly onset step: ``episode.info["t_anom"]`` when present
    (M2 writes a high value for cold-silent mapping passes and a low value for
    warm-fires episodes), else the run-level ``default``. Absent/None → default,
    so objectnav/revisit episodes are unchanged."""
    if isinstance(ep_info, dict) and ep_info.get("t_anom") is not None:
        return int(ep_info["t_anom"])
    return int(default)


_ANOMALY_CLIP_DIR_DEFAULT = "data/anomaly_audio"


def resolve_anomaly_clip(anomaly_class: Optional[str], explicit_path: Optional[str] = None,
                         clip_dir: str = _ANOMALY_CLIP_DIR_DEFAULT) -> Optional[str]:
    """Resolve which anomaly .wav to render: an explicit ``--anomaly-clip`` wins;
    else the staged per-class clip ``<clip_dir>/<class>.wav`` if present (real
    ESC-50 audio from ``fetch_anomaly_clips.py``); else ``None`` so the loader
    falls back to the deterministic synthetic burst. Pure path logic — no I/O
    beyond an ``isfile`` check, so it's safe to call before audio is wired."""
    import os
    if explicit_path:
        return explicit_path
    if anomaly_class:
        cand = os.path.join(clip_dir, f"{anomaly_class}.wav")
        if os.path.isfile(cand):
            return cand
    return None


def build_anomaly_clip(path: Optional[str], grid_sr: int,
                       target_norm_rms_db: float = -20.0) -> np.ndarray:
    """Mono, RMS-normalized anomaly clip at ``grid_sr``: a real FSD50K .wav when
    ``path`` exists, else a DETERMINISTIC synthetic broadband burst (seed 0). The
    SINGLE source of truth for both the live render (``habitat_env``) and the
    onset-calibration diagnostic, so their energy scales cannot drift."""
    import os
    if path and os.path.isfile(path):
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        data = np.asarray(data, dtype=np.float32)
        if np.issubdtype(np.asarray(data).dtype, np.integer):
            data = data / 32768.0
        if data.ndim == 2:
            data = data.mean(axis=1)
        data = data.reshape(-1).astype(np.float32)
        if int(sr) != int(grid_sr):
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), int(grid_sr))
            data = resample_poly(data, int(grid_sr) // g, int(sr) // g).astype(np.float32)
        return normalize_clip(data, target_norm_rms_db)

    rng = np.random.default_rng(0)
    n = int(int(grid_sr) * 0.5)
    envlp = np.minimum(1.0, np.linspace(0.0, 4.0, n))
    burst = (rng.standard_normal(n).astype(np.float32) * envlp).astype(np.float32)
    return normalize_clip(burst, target_norm_rms_db)


def normalize_clip(clip, target_db: float = -20.0) -> np.ndarray:
    """Mono float32, RMS-normalized to ``target_db`` (dBFS). Done ONCE per run so
    per-cell convolution does not saturate or vanish across the grid. Silence
    stays silence (no divide-by-zero)."""
    x = np.asarray(clip, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=0)
    x = x.reshape(-1).astype(np.float32)
    cur = rms(x)
    if cur <= 1e-8:
        return x
    target = 10.0 ** (target_db / 20.0)
    return (x * (target / cur)).astype(np.float32)


def render_step_audio(
    grid: Optional[RIRGrid],
    agent_pos,
    clip_norm: Optional[np.ndarray],
    step_idx: int,
    cfg: AudioTaskConfig,
) -> Optional[np.ndarray]:
    """The (2, L) binaural observation for this step, or ``None`` for silence.

    None when the grid or clip is missing or before ``cfg.t_anom``; otherwise the
    cached-RIR nearest-cell render at the agent's pose (``audio.render_at_pose``).
    habitat_env calls this; it never touches the audio simulator.
    """
    if grid is None or clip_norm is None:
        return None
    if int(step_idx) < int(cfg.t_anom):
        return None
    return render_at_pose(grid, agent_pos, clip_norm)


def process_audio_step(
    audio_obs: Optional[np.ndarray],
    step_idx: int,
    sample_rate: int,
    cfg: AudioTaskConfig,
    state: AudioEpisodeState,
    clap_encoder: Any = None,
) -> Dict[str, Any]:
    """The whole per-step audio brain. Mutates ``state`` in place: on the FIRST
    step whose RMS clears ``cfg.onset_rms`` it marks ``detected``, records
    ``onset_step``, and (if a CLAP encoder is supplied) classifies the anomaly
    EXACTLY ONCE → ``anomaly_class`` / ``target_override`` (= CLASS_TO_OBJECT).
    Returns a diagnostics dict the caller stuffs into ``step.info``. Tolerates
    ``audio_obs=None`` (no-op) and ``clap_encoder=None`` (onset still records,
    class stays None — graceful degrade).
    """
    diag: Dict[str, Any] = {
        "audio_energy": 0.0,
        "audio_lateral_sign": 0,
        "audio_class": state.anomaly_class,
        "audio_target_override": state.target_override,
        "onset_fired": False,
        # E5: anomaly-vs-benign verdict for the anomaly_response controller.
        # Only set (True/False) when the open-set gate runs (anomaly_gate ON);
        # stays None on the default/gate-off path so no existing consumer sees it.
        "is_anomaly": None,
    }
    if audio_obs is None:
        return diag

    energy = rms(audio_obs)
    lat = int(lateral_sign(audio_obs))
    state.last_energy = energy
    state.last_lateral = lat
    diag["audio_energy"] = energy
    diag["audio_lateral_sign"] = lat

    if not state.detected and energy >= cfg.onset_rms:
        # Step 1 open-set gate: a loud sound is only an ANOMALY onset if CLAP
        # separates it from the normal/background prompt bank. When the gate is
        # OFF (default) this is byte-identical to energy-only onset + a forced
        # 3-way classify. When ON and the sound reads as benign, we do NOT mark
        # detected (no onset consumed) so the agent keeps listening.
        fire_onset = True
        gate_class: Optional[str] = None
        if cfg.anomaly_gate and clap_encoder is not None:
            ok, gate_class, ascores = is_anomaly(
                audio_obs, sample_rate, clap_encoder,
                delta=cfg.anomaly_delta, tau_abs=cfg.anomaly_tau)
            diag["audio_anomaly_margin"] = float(ascores.get("margin", 0.0))
            diag["audio_anomaly_fired"] = bool(ok)
            diag["is_anomaly"] = bool(ok)  # E5: the controller's interrupt verdict
            fire_onset = bool(ok)
        if fire_onset:
            state.detected = True
            state.onset_step = int(step_idx)
            diag["onset_fired"] = True
            if gate_class is not None:
                # reuse the gate's argmax class — no second CLAP pass.
                state.anomaly_class = gate_class
                state.target_override = CLASS_TO_OBJECT.get(gate_class)
            elif clap_encoder is not None:
                cls, _scores = classify_anomaly(audio_obs, sample_rate, clap_encoder)
                state.anomaly_class = cls
                state.target_override = CLASS_TO_OBJECT.get(cls)

    diag["audio_class"] = state.anomaly_class
    diag["audio_target_override"] = state.target_override
    return diag


def audio_target_for_retrieval(state: AudioEpisodeState, fallback_category: str) -> str:
    """The retrieval target for ``propose_memory_candidates``: once the anomaly is
    detected, prefer the per-episode captioned object (``anomaly_object_override``,
    the object the agent actually mapped) over the static CLASS_TO_OBJECT
    affordance (``target_override``); else the fallback category VERBATIM (so
    objectnav/revisit/multion retrieval is byte-identical when audio is off)."""
    if state.detected:
        return state.anomaly_object_override or state.target_override or fallback_category
    return fallback_category


def gate_retrieval_target(
    resolved_target: Optional[str], *, onset_gate: bool, detected: bool
) -> Optional[str]:
    """S1 onset-gate (env-gated ``LTM_AUDIO_DOA``): when ``onset_gate`` is on,
    SUPPRESS memory retrieval (return ``None``) until the anomaly is ``detected``,
    so the audio onset is causally NECESSARY for warm recall — turn the audio off
    and there is no onset, no target, hence no injected memory candidate. When
    ``onset_gate`` is off, returns ``resolved_target`` VERBATIM, so the default
    objectnav/revisit/multion path is byte-identical.

    ``propose_memory_candidates`` returns ``[]`` on a falsy target
    (memory_bridge.py: ``not target_category``), so a ``None`` here yields zero
    memory injection pre-onset with no caller guard."""
    if onset_gate and not detected:
        return None
    return resolved_target


def should_audio_stop(
    state: AudioEpisodeState,
    energy: float,
    distance_to_goal: Optional[float],
    cfg: AudioTaskConfig,
) -> bool:
    """True iff the anomaly is detected AND the audio is loud (near the source)
    AND (distance is unknown or within ``cfg.stop_distance_m``). Conservative by
    construction; the runner OR-folds it into the existing STOP selection."""
    if not state.detected:
        return False
    if float(energy) < cfg.energy_stop_db_rms:
        return False
    if distance_to_goal is not None and float(distance_to_goal) > cfg.stop_distance_m:
        return False
    return True
