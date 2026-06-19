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
        state.detected = True
        state.onset_step = int(step_idx)
        diag["onset_fired"] = True
        if clap_encoder is not None:
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
